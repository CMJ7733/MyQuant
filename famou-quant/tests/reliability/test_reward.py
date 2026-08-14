"""Reward builder: delayed, cost-aware, anti-visible-overfit rewards."""

from __future__ import annotations

import pytest

from famou.reliability.archives import SearchArchive
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
