"""Judge taxonomy and the Agentic RL loop (encode -> train -> serve).

The RL tests deliberately assert BEHAVIOUR (does cloning reproduce the
demonstrator? does AWR prefer high-reward actions? does a stale checkpoint get
refused?) rather than loss values, which are not stable enough to pin.
"""

from __future__ import annotations

import pytest

from famou.core.state import StateStore
from famou.reliability.judge import (
    FailureAnalyzer,
    FailureKind,
    summarise_failures,
)
from famou.reliability.types import (
    EvaluationCost,
    EvidenceVector,
    ExpertKind,
    Fidelity,
    MetricDistribution,
    StaticCheckResult,
    StructuredAction,
)

torch = pytest.importorskip("torch", reason="RL trainers need torch")

from famou.reliability.rl.encoding import (  # noqa: E402
    ENCODING_VERSION,
    ActionCodec,
    ObservationEncoder,
)


def evidence(
    *,
    validity: float = 1.0,
    stage=None,
    error: str = "",
    static: StaticCheckResult = None,
    rank_ic=None,
    fidelity: Fidelity = Fidelity.F2_FULL,
) -> EvidenceVector:
    return EvidenceVector(
        candidate_id="c1",
        episode_id="E1",
        eval_id="ev1",
        fidelity=fidelity,
        split_scope="visible_dev",
        data_contract_hash="h",
        rank_ic=MetricDistribution.from_samples(rank_ic) if rank_ic else None,
        validity=validity,
        failure_stage=stage,
        error_info=error or None,
        static_check=static,
        cost=EvaluationCost(wall_seconds=1.0),
    )


class TestFailureAnalyzer:
    def setup_method(self):
        self.analyzer = FailureAnalyzer()

    def test_clean_run_is_not_a_failure(self):
        ev = evidence(rank_ic=[0.05, 0.052, 0.048])
        assert self.analyzer.classify(ev) == FailureKind.NONE

    @pytest.mark.parametrize(
        "static,expected",
        [
            (StaticCheckResult(compiles=False, schema_errors=["SyntaxError: x"]),
             FailureKind.SYNTAX),
            (StaticCheckResult(forbidden_edits=["assignment to protected name 'topk'"]),
             FailureKind.FORBIDDEN_EDIT),
            (StaticCheckResult(leakage_flags=["future-reference idiom"]),
             FailureKind.LEAKAGE),
            (StaticCheckResult(schema_errors=["missing HYPERPARAMS assignment"]),
             FailureKind.CONTRACT),
            (StaticCheckResult(schema_errors=["package 'requests' is not in the "
                                              "TaskSpec allow-list"]),
             FailureKind.DEPENDENCY),
        ],
    )
    def test_static_findings_are_authoritative(self, static, expected):
        """Structured static results beat regex over free text."""
        ev = evidence(validity=0.0, stage="static_check", static=static)
        assert self.analyzer.classify(ev) == expected

    @pytest.mark.parametrize(
        "stage,error,expected",
        [
            ("timeout", "evaluation timeout", FailureKind.TIMEOUT),
            ("oom", "out of memory", FailureKind.OOM),
            ("train", "ValueError: shape mismatch in encoder", FailureKind.SHAPE),
            ("train", "RuntimeError: loss became nan", FailureKind.NAN_OUTPUT),
            ("train", "no valid IC days", FailureKind.DATA),
            ("train", "ModuleNotFoundError: no module named 'foo'",
             FailureKind.DEPENDENCY),
            ("train", "something nobody anticipated", FailureKind.CRASH),
        ],
    )
    def test_runtime_errors_classified(self, stage, error, expected):
        ev = evidence(validity=0.0, stage=stage, error=error)
        assert self.analyzer.classify(ev) == expected

    def test_degenerate_output_detected(self):
        """A model whose predictions are near-constant 'succeeds' with a
        meaningless RankIC — it must not enter the archive as a valid
        low-scoring candidate."""
        ev = evidence(rank_ic=[0.0001, 0.0002, 0.0001])
        assert self.analyzer.classify(ev) == FailureKind.DEGENERATE

    def test_no_improvement_is_a_distinct_outcome(self):
        ev = evidence(rank_ic=[0.03, 0.031, 0.029])
        verdict = self.analyzer.judge(ev, incumbent_rank_ic=0.05)
        assert verdict.kind == FailureKind.NO_IMPROVEMENT
        assert verdict.policy_level          # steer the policy, not the code
        assert not verdict.repairable

    def test_repairable_vs_policy_level(self):
        shape = self.analyzer.judge(
            evidence(validity=0.0, stage="train", error="shape mismatch"))
        assert shape.repairable and not shape.policy_level
        assert shape.repair_hint

        leak = self.analyzer.judge(evidence(
            validity=0.0, stage="static_check",
            static=StaticCheckResult(leakage_flags=["future-reference idiom"])))
        assert leak.policy_level and not leak.repairable

    def test_judge_never_touches_admission(self):
        """The judge is advisory. It exposes no promote/admit surface."""
        verdict = self.analyzer.judge(evidence(rank_ic=[0.09, 0.091, 0.089]))
        assert not any(
            attr in verdict.model_fields
            for attr in ("promote", "admit", "certified", "verdict")
        )

    def test_llm_review_failure_is_contained(self):
        def boom(code, verdict):
            raise RuntimeError("endpoint down")

        analyzer = FailureAnalyzer(llm_review_fn=boom)
        v = analyzer.judge(
            evidence(validity=0.0, stage="train", error="shape mismatch"),
            candidate_code="print(1)",
        )
        assert v.kind == FailureKind.SHAPE       # classification still works
        assert "unavailable" in v.llm_feedback

    def test_summarise_skips_successes(self):
        verdicts = [
            self.analyzer.judge(evidence(rank_ic=[0.05, 0.051, 0.049])),
            self.analyzer.judge(evidence(validity=0.0, stage="oom", error="oom")),
            self.analyzer.judge(evidence(validity=0.0, stage="oom", error="oom")),
        ]
        assert summarise_failures(verdicts) == {"oom": 2}


class TestEncoding:
    def test_action_round_trip(self):
        codec = ActionCodec()
        action = StructuredAction(
            expert=ExpertKind.CROSSOVER, model_family="mlp",
            fidelity=Fidelity.F2_FULL, batch_size=4, promotion_requested=True,
        )
        decoded = codec.decode(codec.encode(action))
        assert decoded.expert == action.expert
        assert decoded.model_family == action.model_family
        assert decoded.fidelity == action.fidelity
        assert decoded.batch_size == action.batch_size
        assert decoded.promotion_requested

    def test_batch_size_rounds_down(self):
        """A request for 3 must be recorded as the 2 that can be served, not
        the 4 that cannot."""
        codec = ActionCodec()
        action = StructuredAction(expert=ExpertKind.EXPLORE, batch_size=3)
        assert codec.decode(codec.encode(action)).batch_size == 2

    def test_features_are_bounded(self, ledger, state_store):
        """Every feature must be a ratio or a squashed quantity: raw magnitudes
        would not transfer across episodes with different budgets."""
        from famou.reliability.archives import CertifiedArchive, SearchArchive
        from famou.reliability.observation import ObservationBuilder

        builder = ObservationBuilder(
            SearchArchive(state_store), CertifiedArchive(state_store), ledger)
        obs = builder.build(episode_id="E1", state_version=3)
        vec = ObservationEncoder().encode(obs)
        assert len(vec) == ObservationEncoder().dim
        assert all(-1.0 <= v <= 1.0 for v in vec), "features must be bounded"

    def test_feature_names_match_dim(self):
        enc = ObservationEncoder()
        assert len(enc.FEATURE_NAMES) == enc.dim


class TestRLTrainers:
    """Train on a synthetic corpus whose optimal action is known."""

    def _corpus(self, n=64):
        """Two clearly separated states with different best actions.

        State A (budget nearly gone) -> cheap explore, no promotion.
        State B (budget healthy, strong F2 candidate) -> promote.
        Only the promote action is rewarded in state B.
        """
        from famou.reliability.rl.trainer import Sample

        enc_dim = ObservationEncoder().dim
        codec = ActionCodec()
        samples = []
        for i in range(n):
            in_state_b = i % 2 == 0
            feats = [0.0] * enc_dim
            feats[0] = 0.9 if in_state_b else 0.05     # budget_gpu_frac
            feats[2] = 0.8 if in_state_b else 0.0      # budget_sealed_frac
            action = StructuredAction(
                expert=ExpertKind.LOCAL_HPO if in_state_b else ExpertKind.EXPLORE,
                model_family="gbdt",
                fidelity=Fidelity.F2_FULL if in_state_b else Fidelity.F1_CHEAP,
                batch_size=1,
                promotion_requested=in_state_b,
            )
            s = Sample(features=feats, action=codec.encode(action),
                       reward=1.0 if in_state_b else 0.0)
            s.ret = s.reward
            samples.append(s)
        return samples

    def test_behaviour_cloning_learns_the_demonstrator(self):
        from famou.reliability.rl.trainer import BehaviorCloning

        report = BehaviorCloning(seed=0).fit(self._corpus(), epochs=120)
        assert report.stage == "behaviour_cloning"
        # the mapping is deterministic, so cloning should nail it
        assert report.head_accuracy["expert"] > 0.95
        assert report.head_accuracy["promote"] > 0.95
        assert report.losses[-1] < report.losses[0]

    def test_awr_runs_and_warm_starts(self):
        from famou.reliability.rl.trainer import (
            AdvantageWeightedRegression, BehaviorCloning,
        )

        bc = BehaviorCloning(seed=0)
        bc.fit(self._corpus(), epochs=60)
        awr = AdvantageWeightedRegression(seed=0)
        report = awr.fit(self._corpus(), epochs=60, init_from=bc)
        assert report.stage == "awr"
        assert awr.net is bc.net           # warm started, not re-initialised
        assert report.mean_reward == pytest.approx(0.5, abs=0.01)

    def test_training_refuses_tiny_corpus(self):
        from famou.reliability.rl.trainer import BehaviorCloning, TrainingDataError

        with pytest.raises(TrainingDataError):
            BehaviorCloning().fit(self._corpus(n=3), epochs=5)

    def test_awr_requires_rewards(self):
        from famou.reliability.rl.trainer import (
            AdvantageWeightedRegression, TrainingDataError,
        )

        samples = self._corpus()
        for s in samples:
            s.ret = None                    # RewardBuilder never ran
        with pytest.raises(TrainingDataError, match="mature"):
            AdvantageWeightedRegression().fit(samples, epochs=5)

    def test_dataset_skips_stale_encoding(self, tmp_path):
        """Records written by an older encoder must be dropped, not mixed in:
        the same feature index means something different."""
        from famou.reliability.rl.trainer import build_dataset
        from famou.reliability.trajectory import TrajectoryStore, build_transition
        from famou.reliability.types import DecisionRecord

        store = TrajectoryStore()
        for i, version in enumerate([ENCODING_VERSION, "enc_v0"]):
            d = DecisionRecord(
                decision_id=f"d{i}", observation_digest="x",
                observation_features=[0.0] * ObservationEncoder().dim,
                encoding_version=version,
                structured_action=StructuredAction(expert=ExpertKind.EXPLORE),
                state_version=i, timestamp=0.0,
            )
            store.record_decision(d)
            store.record_transition(build_transition(
                decision=d, candidate_ids=[f"c{i}"], evidence_ids=[],
                costs=EvaluationCost(), next_state_version=i + 1))
        assert len(build_dataset(store)) == 1

    def test_credit_does_not_bleed_across_runs(self):
        """Iterated offline training concatenates trajectories from several
        search runs. Each run restarts its state_version, and credit must not
        propagate backwards past that boundary — the last decision of run A
        cannot have caused the first reward of run B."""
        from famou.reliability.rl.trainer import build_dataset
        from famou.reliability.trajectory import TrajectoryStore, build_transition
        from famou.reliability.types import DecisionRecord

        store = TrajectoryStore()
        # run A: versions 0,1 with zero reward; run B: versions 0,1 with reward 1
        plan = [(0, 0.0), (1, 0.0), (0, 1.0), (1, 1.0)]
        for i, (version, reward) in enumerate(plan):
            d = DecisionRecord(
                decision_id=f"d{i}", observation_digest="x",
                observation_features=[0.0] * ObservationEncoder().dim,
                encoding_version=ENCODING_VERSION,
                structured_action=StructuredAction(expert=ExpertKind.EXPLORE),
                state_version=version, timestamp=float(i),
            )
            store.record_decision(d)
            t = build_transition(
                decision=d, candidate_ids=[f"c{i}"], evidence_ids=[],
                costs=EvaluationCost(), next_state_version=version + 1)
            store.record_transition(t)
            store.set_reward(t.transition_id, reward)

        samples = build_dataset(store, gamma=0.9)
        assert len(samples) == 4
        # run A earned nothing and must stay at zero
        assert samples[0].ret == pytest.approx(0.0)
        assert samples[1].ret == pytest.approx(0.0)
        # run B discounts within itself: 1 + 0.9*1 = 1.9, then 1
        assert samples[2].ret == pytest.approx(1.9)
        assert samples[3].ret == pytest.approx(1.0)


class TestLearnedPolicy:
    def _checkpoint(self, tmp_path):
        from famou.reliability.rl.trainer import BehaviorCloning, Sample

        codec = ActionCodec()
        action = StructuredAction(
            expert=ExpertKind.EXPLORE, model_family="gbdt",
            fidelity=Fidelity.F1_CHEAP, batch_size=1,
        )
        samples = [
            Sample(features=[0.5] * ObservationEncoder().dim,
                   action=codec.encode(action), reward=0.0, ret=0.0)
            for _ in range(16)
        ]
        trainer = BehaviorCloning(seed=0)
        trainer.fit(samples, epochs=40)
        path = tmp_path / "policy.pt"
        trainer.save(str(path), policy_version="test_v1", trained_on=len(samples))
        return path

    def _observation(self, ledger, state_store, sealed_left=True):
        from famou.reliability.archives import CertifiedArchive, SearchArchive
        from famou.reliability.observation import ObservationBuilder

        if not sealed_left:
            for _ in range(3):
                ledger.issue_gate_token("E1")
        builder = ObservationBuilder(
            SearchArchive(state_store), CertifiedArchive(state_store), ledger)
        return builder.build(episode_id="E1", state_version=1)

    def test_policy_fills_log_prob_and_value(self, tmp_path, ledger, state_store):
        """These DecisionRecord slots existed but were always None; a learned
        policy is what makes the trajectory usable beyond cloning."""
        from famou.reliability.rl.policy import LearnedMetaPolicy

        policy = LearnedMetaPolicy(str(self._checkpoint(tmp_path)))
        action = policy.act(self._observation(ledger, state_store))
        assert isinstance(action, StructuredAction)
        assert policy.last_log_prob is not None and policy.last_log_prob <= 0
        assert policy.last_value is not None
        assert policy.policy_version == "test_v1"

    def test_promotion_masked_without_sealed_budget(self, tmp_path, ledger, state_store):
        """A learned head has no notion of a hard constraint; masking is
        cheaper than hoping it learns one."""
        from famou.reliability.rl.policy import LearnedMetaPolicy

        policy = LearnedMetaPolicy(str(self._checkpoint(tmp_path)))
        obs = self._observation(ledger, state_store, sealed_left=False)
        for _ in range(12):
            assert policy.act(obs).promotion_requested is False

    def test_never_selects_f0(self, tmp_path, ledger, state_store):
        """F0 is a static check, not a search action: choosing it would burn an
        iteration producing no evidence."""
        from famou.reliability.rl.policy import LearnedMetaPolicy

        policy = LearnedMetaPolicy(str(self._checkpoint(tmp_path)))
        obs = self._observation(ledger, state_store)
        for _ in range(12):
            assert policy.act(obs).fidelity != Fidelity.F0_STATIC

    def test_stale_checkpoint_refused(self, tmp_path):
        from famou.reliability.rl.policy import CheckpointMismatch, LearnedMetaPolicy

        path = self._checkpoint(tmp_path)
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        blob["encoding_version"] = "enc_v0"
        stale = tmp_path / "stale.pt"
        torch.save(blob, str(stale))
        with pytest.raises(CheckpointMismatch, match="encoding"):
            LearnedMetaPolicy(str(stale))

    def test_greedy_is_deterministic(self, tmp_path, ledger, state_store):
        from famou.reliability.rl.policy import LearnedMetaPolicy

        policy = LearnedMetaPolicy(str(self._checkpoint(tmp_path)), greedy=True)
        obs = self._observation(ledger, state_store)
        first = policy.act(obs)
        for _ in range(5):
            again = policy.act(obs)
            assert again.expert == first.expert
            assert again.model_family == first.model_family

    def test_explain_exposes_head_probabilities(self, tmp_path, ledger, state_store):
        """A policy that spends GPU hours should be inspectable."""
        from famou.reliability.rl.policy import LearnedMetaPolicy

        policy = LearnedMetaPolicy(str(self._checkpoint(tmp_path)))
        report = policy.explain(self._observation(ledger, state_store))
        assert "value" in report and "features" in report
        assert sum(report["expert"].values()) == pytest.approx(1.0, abs=0.01)
        assert set(report["promote"]) == {"no", "yes"}
