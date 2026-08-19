"""Reward builder: delayed, cost-aware, anti-visible-overfit rewards."""

from __future__ import annotations

import pytest

from famou.reliability.archives import CertifiedArchive, SearchArchive
from famou.reliability.reward import RewardBuilder, RewardConfig
from famou.reliability.trajectory import TrajectoryStore, build_transition
from famou.reliability.types import (
    DecisionRecord,
    EvaluationCost,
    EvidenceVector,
    ExpertKind,
    Fidelity,
    GateReasonCode,
    GateVerdict,
    GateVerdictKind,
    MarginBand,
    MetricDistribution,
    StructuredAction,
)


def _decision() -> DecisionRecord:
    return DecisionRecord(
        decision_id="dec_1",
        observation_digest="obs",
        structured_action=StructuredAction(expert=ExpertKind.MUTATE),
        state_version=1,
        timestamp=0.0,
    )


def _evidence(cid: str, samples, valid=True) -> EvidenceVector:
    return EvidenceVector(
        candidate_id=cid,
        episode_id="E1",
        eval_id=f"ev_{cid}",
        fidelity=Fidelity.F2_FULL,
        split_scope="visible_dev",
        data_contract_hash="h",
        rank_ic=MetricDistribution.from_samples(list(samples)),
        validity=1.0 if valid else 0.0,
        failure_stage=None if valid else "train",
    )


class TestRewardBuilder:
    def test_promote_beats_reject(self, state_store):
        search = SearchArchive(state_store)
        search.add_candidate("c1", episode_id="E1", model_family="gbdt", code_hash="h1")
        search.add_evidence(_evidence("c1", [0.07, 0.07, 0.07]))
        builder = RewardBuilder(search)
        store = TrajectoryStore()

        t_promote = build_transition(
            decision=_decision(), candidate_ids=["c1"], evidence_ids=["ev_c1"],
            costs=EvaluationCost(),
            gate_verdict=GateVerdict(
                verdict=GateVerdictKind.PROMOTE,
                reason_code=GateReasonCode.ROBUST_IMPROVEMENT,
                margin_band=MarginBand.CLEAR_PASS,
            ),
        )
        t_reject = build_transition(
            decision=_decision(), candidate_ids=["c1"], evidence_ids=["ev_c1"],
            costs=EvaluationCost(),
            gate_verdict=GateVerdict(
                verdict=GateVerdictKind.REJECT,
                reason_code=GateReasonCode.NO_IMPROVEMENT,
            ),
        )
        r_promote = builder.mature(store, t_promote, incumbent_rank_ic=0.05)
        r_reject = builder.mature(store, t_reject, incumbent_rank_ic=0.05)
        assert r_promote > 0
        assert r_reject < 0
        assert r_promote > r_reject

    def test_visible_overfit_candidate_scores_low(self, state_store):
        """High visible IC + high variance + gate reject => low/negative reward."""
        search = SearchArchive(state_store)
        search.add_candidate("overfit", episode_id="E1", model_family="mlp", code_hash="h")
        search.add_evidence(_evidence("overfit", [0.20, -0.05, 0.12]))  # huge variance
        builder = RewardBuilder(search)
        store = TrajectoryStore()
        t = build_transition(
            decision=_decision(), candidate_ids=["overfit"], evidence_ids=["ev_overfit"],
            costs=EvaluationCost(gpu_seconds=500.0),
            gate_verdict=GateVerdict(
                verdict=GateVerdictKind.REJECT,
                reason_code=GateReasonCode.UNSTABLE_ACROSS_SEEDS,
            ),
        )
        reward = builder.mature(store, t, incumbent_rank_ic=0.05)
        assert reward < 0

    def test_stable_cheap_improvement_scores_well(self, state_store):
        search = SearchArchive(state_store)
        search.add_candidate("solid", episode_id="E1", model_family="gbdt", code_hash="h")
        search.add_evidence(_evidence("solid", [0.065, 0.066, 0.064]))
        builder = RewardBuilder(search)
        store = TrajectoryStore()
        t = build_transition(
            decision=_decision(), candidate_ids=["solid"], evidence_ids=["ev_solid"],
            costs=EvaluationCost(gpu_seconds=10.0),
            gate_verdict=GateVerdict(
                verdict=GateVerdictKind.PROMOTE,
                reason_code=GateReasonCode.ROBUST_IMPROVEMENT,
                margin_band=MarginBand.CLEAR_PASS,
            ),
        )
        reward = builder.mature(store, t, incumbent_rank_ic=0.05)
        assert reward > 0.5

    def test_reward_written_back_to_store(self, state_store):
        search = SearchArchive(state_store)
        builder = RewardBuilder(search)
        store = TrajectoryStore()
        t = build_transition(
            decision=_decision(), candidate_ids=[], evidence_ids=[],
            costs=EvaluationCost(),
        )
        store.record_transition(t)
        builder.mature(store, t, incumbent_rank_ic=None)
        assert store.transitions()[0].reward is not None


def _promote_verdict() -> GateVerdict:
    return GateVerdict(
        verdict=GateVerdictKind.PROMOTE,
        reason_code=GateReasonCode.ROBUST_IMPROVEMENT,
        margin_band=MarginBand.CLEAR_PASS,
    )


class TestMatureAll:
    """Whole-run maturation: scope, idempotence, and a causal incumbent."""

    def _archives(self, state_store):
        search = SearchArchive(state_store)
        certified = CertifiedArchive(state_store)
        search.add_candidate("init_0", episode_id="E1", model_family="gbdt",
                             code_hash="h0")
        search.add_evidence(_evidence("init_0", [0.05, 0.05, 0.05]))
        certified.add_baseline("init_0", episode_id="E1", model_family="gbdt",
                               code_hash="h0")
        return search, certified

    def test_fills_every_session_transition(self, state_store):
        search, certified = self._archives(state_store)
        search.add_candidate("c1", episode_id="E1", model_family="gbdt",
                             code_hash="h1")
        search.add_evidence(_evidence("c1", [0.07, 0.07, 0.07]))
        store = TrajectoryStore()
        for _ in range(3):
            store.record_transition(build_transition(
                decision=_decision(), candidate_ids=["c1"],
                evidence_ids=["ev_c1"], costs=EvaluationCost(),
            ))

        filled = RewardBuilder(search).mature_all(
            store, certified_archive=certified
        )

        assert len(filled) == 3
        assert all(t.reward is not None for t in store.transitions())

    def test_is_idempotent(self, state_store):
        search, certified = self._archives(state_store)
        search.add_candidate("c1", episode_id="E1", model_family="gbdt",
                             code_hash="h1")
        search.add_evidence(_evidence("c1", [0.07, 0.07, 0.07]))
        store = TrajectoryStore()
        store.record_transition(build_transition(
            decision=_decision(), candidate_ids=["c1"], evidence_ids=["ev_c1"],
            costs=EvaluationCost(),
        ))
        builder = RewardBuilder(search)

        first = builder.mature_all(store, certified_archive=certified)
        second = builder.mature_all(store, certified_archive=certified)

        assert len(first) == 1
        assert second == []            # nothing left to mature

    def test_skips_transitions_replayed_from_disk(self, state_store, tmp_path):
        """A previous run's transitions must not be scored against this run's
        archives — they hold no evidence for those candidates."""
        path = tmp_path / "traj.jsonl"
        old = TrajectoryStore(str(path))
        old.record_transition(build_transition(
            decision=_decision(), candidate_ids=["ghost"], evidence_ids=[],
            costs=EvaluationCost(),
        ))

        search, certified = self._archives(state_store)
        reloaded = TrajectoryStore(str(path))       # replays the old transition
        assert len(reloaded.transitions()) == 1
        assert reloaded.session_transitions() == []

        filled = RewardBuilder(search).mature_all(
            reloaded, certified_archive=certified
        )

        assert filled == []
        assert reloaded.transitions()[0].reward is None

    def test_incumbent_is_replayed_not_read_from_the_end_state(self, state_store):
        """The second candidate is measured against the first one's promotion,
        and the first against the baseline only."""
        search, certified = self._archives(state_store)
        for cid, samples in (("c1", [0.07] * 3), ("c2", [0.075] * 3)):
            search.add_candidate(cid, episode_id="E1", model_family="gbdt",
                                 code_hash=cid)
            search.add_evidence(_evidence(cid, samples))
        store = TrajectoryStore()
        t1 = build_transition(
            decision=_decision(), candidate_ids=["c1"], evidence_ids=["ev_c1"],
            costs=EvaluationCost(), gate_verdict=_promote_verdict(),
        )
        t2 = build_transition(
            decision=_decision(), candidate_ids=["c2"], evidence_ids=["ev_c2"],
            costs=EvaluationCost(),
        )
        store.record_transition(t1)
        store.record_transition(t2)
        certified.admit("c1", episode_id="E1", model_family="gbdt",
                        code_hash="c1", verdict=_promote_verdict(),
                        protocol_version="protocol_b_v2")

        builder = RewardBuilder(search)
        builder.mature_all(store, certified_archive=certified)

        cfg = builder.config
        # c1 was judged against the 0.05 baseline: gain 0.02, minus the sealed
        # query the promotion spent.
        assert t1.reward == pytest.approx(
            cfg.w_promote
            + cfg.w_visible_gain * (0.07 - 0.05)
            - cfg.w_sealed_query * _promote_verdict().query_cost,
            abs=1e-6,
        )
        # c2 came after c1 was certified, so its bar is 0.07, not 0.05.
        assert t2.reward == pytest.approx(
            cfg.w_visible_gain * (0.075 - 0.07), abs=1e-6
        )

    def test_unadmitted_promote_does_not_raise_the_bar(self, state_store):
        """A PROMOTE verdict that CertifiedAdmission refused changes nothing."""
        search, certified = self._archives(state_store)
        for cid in ("c1", "c2"):
            search.add_candidate(cid, episode_id="E1", model_family="gbdt",
                                 code_hash=cid)
            search.add_evidence(_evidence(cid, [0.07, 0.07, 0.07]))
        store = TrajectoryStore()
        store.record_transition(build_transition(
            decision=_decision(), candidate_ids=["c1"], evidence_ids=["ev_c1"],
            costs=EvaluationCost(), gate_verdict=_promote_verdict(),
        ))
        t2 = build_transition(
            decision=_decision(), candidate_ids=["c2"], evidence_ids=["ev_c2"],
            costs=EvaluationCost(),
        )
        store.record_transition(t2)
        # c1 never reached the Certified Archive (admission refused).

        builder = RewardBuilder(search)
        builder.mature_all(store, certified_archive=certified)

        assert t2.reward == pytest.approx(
            builder.config.w_visible_gain * (0.07 - 0.05), abs=1e-6
        )
