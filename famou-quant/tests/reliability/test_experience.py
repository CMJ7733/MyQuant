"""Experience layer, stages 0-1.

The property under test that everything else depends on is E1: retrieval at
state version n never sees experience produced by the batch that committed
version n+1. Without it, two replays of the same trajectory feed the policy
different context while producing an identical observation_digest, and nothing
raises.
"""

from __future__ import annotations

import pytest

from famou.core.state import StateStore
from famou.reliability.archives import (
    ArchiveWriteOutsideBarrier,
    CertifiedAdmission,
    CertifiedArchive,
    CommitGuard,
    SearchArchive,
)
from famou.reliability.barrier import BarrierCommit, CandidateOutcome
from famou.reliability.experience import (
    EvidenceLevel,
    ExperienceConsolidator,
    ExperienceIndex,
    ExperienceRecord,
    ExperienceType,
    FailureMemory,
    MemoryRetriever,
    ObservedOutcome,
    QueryType,
    failure_experience_id,
    reliability_weight,
)
from famou.reliability.judge import FailureKind
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import (
    CandidateLineage,
    DecisionRecord,
    EvidenceVector,
    ExpertKind,
    Fidelity,
    MetricDistribution,
    StructuredAction,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def ok_evidence(candidate_id: str, *, eval_id=None, seeds=(0.06, 0.061, 0.059)):
    return EvidenceVector(
        candidate_id=candidate_id,
        episode_id="E1",
        eval_id=eval_id or f"ev_{candidate_id}",
        fidelity=Fidelity.F2_FULL,
        split_scope="visible_dev",
        data_contract_hash="h",
        rank_ic=MetricDistribution.from_samples(list(seeds)),
        validity=1.0,
    )


def failed_evidence(candidate_id: str, *, error: str, stage="evaluate", eval_id=None):
    return EvidenceVector(
        candidate_id=candidate_id,
        episode_id="E1",
        eval_id=eval_id or f"ev_{candidate_id}",
        fidelity=Fidelity.F1_CHEAP,
        split_scope="visible_dev",
        data_contract_hash="h",
        validity=0.0,
        error_info=error,
        failure_stage=stage,
    )


def observed(candidate_id, evidence, *, parents=(), expert="mutate", family="gbdt"):
    return ObservedOutcome(
        candidate_id=candidate_id,
        episode_id="E1",
        model_family=family,
        evidence=[evidence],
        parent_ids=list(parents),
        expert=expert,
    )


@pytest.fixture
def index():
    """Index with a permissive guard — the guard itself is tested separately."""
    return ExperienceIndex(StateStore())


def record(experience_id="e1", *, version=0, level=EvidenceLevel.PROVISIONAL, n=1):
    rec = ExperienceRecord(
        experience_id=experience_id,
        experience_type=ExperienceType.FAILURE,
        applicability={"failure_kind": "shape", "model_family": "gbdt"},
        evidence_level=level,
        sample_count=n,
        valid_from_state_version=version,
    )
    rec.refresh_weight()
    return rec


# ---------------------------------------------------------------------------


class TestVersionIsolation:
    """Invariant E1 — the reason this layer is not just a vector store."""

    def test_record_is_invisible_before_its_version(self, index):
        index.upsert(record("e_v5", version=5))

        assert index.visible_at(4) == []
        assert [r.experience_id for r in index.visible_at(5)] == ["e_v5"]
        assert [r.experience_id for r in index.visible_at(9)] == ["e_v5"]

    def test_retriever_never_returns_future_experience(self, index):
        index.upsert(record("old", version=1))
        index.upsert(record("new", version=7))
        retriever = MemoryRetriever(index)

        bundle = retriever.retrieve(at_version=3)

        assert bundle.retrieved_experience_ids == ["old"]
        assert bundle.n_candidates_considered == 1
        assert bundle.state_version == 3

    def test_upsert_does_not_advance_valid_from(self, index):
        """An aggregate accumulating evidence must not retroactively hide
        itself from decisions that legitimately saw it."""
        index.upsert(record("agg", version=2, n=1))
        index.upsert(record("agg", version=99, n=5))

        assert index.get("agg").valid_from_state_version == 2
        assert index.get("agg").sample_count == 5
        assert len(index.visible_at(2)) == 1

    def test_deprecated_records_are_never_visible(self, index):
        rec = record("gone", version=0)
        rec.deprecated = True
        index.upsert(rec)

        assert index.visible_at(1000) == []


class TestGuard:
    def test_write_outside_barrier_window_refused(self):
        store = StateStore()
        guarded = ExperienceIndex(store, guard=CommitGuard(enforce=True))

        with pytest.raises(ArchiveWriteOutsideBarrier):
            guarded.upsert(record("e1"))
        with pytest.raises(ArchiveWriteOutsideBarrier):
            guarded.record_candidate_failure(
                "c1", failure_kind="shape", model_family="gbdt"
            )

    def test_reads_are_not_guarded(self):
        store = StateStore()
        guard = CommitGuard(enforce=True)
        guarded = ExperienceIndex(store, guard=guard)
        with guard.window("test"):
            guarded.upsert(record("e1", version=0))

        assert len(guarded.visible_at(0)) == 1     # no window needed


class TestFailureMemory:
    def test_aggregates_by_kind_and_family(self, index):
        memory = FailureMemory()
        for i in range(3):
            memory.observe(
                index,
                observed(f"c{i}", failed_evidence(f"c{i}", error="shape mismatch")),
                valid_from_state_version=1,
                transition_id=f"tr_{i}",
                decision_id=f"dec_{i}",
            )

        expected_id = failure_experience_id(FailureKind.SHAPE, "gbdt")
        rec = index.get(expected_id)
        assert rec is not None
        assert rec.outcome_summary["occurrences"] == 3
        assert rec.outcome_summary["repairable"] is True
        assert index.size() == 1          # one aggregate, not three records

    def test_different_families_are_separate_patterns(self, index):
        memory = FailureMemory()
        memory.observe(
            index,
            observed("a", failed_evidence("a", error="shape mismatch"), family="gbdt"),
            valid_from_state_version=1, transition_id="tr", decision_id="dec",
        )
        memory.observe(
            index,
            observed("b", failed_evidence("b", error="shape mismatch"), family="mlp"),
            valid_from_state_version=1, transition_id="tr", decision_id="dec",
        )

        assert index.size() == 2

    def test_successful_candidate_creates_no_failure_record(self, index):
        FailureMemory().observe(
            index,
            observed("good", ok_evidence("good")),
            valid_from_state_version=1, transition_id="tr", decision_id="dec",
        )

        assert index.size() == 0

    def test_recovery_is_credited_to_the_parents_pattern(self, index):
        """A child that fixes its parent's failure counts as a repair of the
        PARENT's (kind, family), even when the child is another family."""
        memory = FailureMemory()
        memory.observe(
            index,
            observed("parent", failed_evidence("parent", error="nan in predictions"),
                     family="mlp"),
            valid_from_state_version=1, transition_id="tr_1", decision_id="dec_1",
        )
        memory.observe(
            index,
            observed("child", ok_evidence("child"), parents=["parent"],
                     expert="local_hpo", family="gbdt"),
            valid_from_state_version=2, transition_id="tr_2", decision_id="dec_2",
        )

        rec = index.get(failure_experience_id(FailureKind.NAN_OUTPUT, "mlp"))
        summary = rec.outcome_summary
        assert summary["recovery_attempts"] == 1
        assert summary["recoveries"] == 1
        assert summary["recovered_via"] == {"local_hpo": 1}
        assert "1/1 succeeded (100%)" in rec.statement

    def test_failed_repair_counts_as_an_attempt(self, index):
        """Counting only successes would make every pattern look 100% fixable."""
        memory = FailureMemory()
        memory.observe(
            index,
            observed("parent", failed_evidence("parent", error="shape mismatch")),
            valid_from_state_version=1, transition_id="tr_1", decision_id="dec_1",
        )
        memory.observe(
            index,
            observed("child", failed_evidence("child", error="shape mismatch"),
                     parents=["parent"], expert="debug"),
            valid_from_state_version=2, transition_id="tr_2", decision_id="dec_2",
        )

        summary = index.get(failure_experience_id(FailureKind.SHAPE, "gbdt")).outcome_summary
        assert summary["recovery_attempts"] == 1
        assert summary["recoveries"] == 0

    def test_statement_is_deterministic_and_mentions_the_hint(self, index):
        """Stage 1 uses no LLM: the same counts must render the same text."""
        memory = FailureMemory()
        for i in range(2):
            memory.observe(
                index,
                observed(f"c{i}", failed_evidence(f"c{i}", error="out of memory")),
                valid_from_state_version=1, transition_id="tr", decision_id="dec",
            )
        first = index.get(failure_experience_id(FailureKind.OOM, "gbdt")).statement

        again = ExperienceIndex(StateStore())
        memory2 = FailureMemory()
        for i in range(2):
            memory2.observe(
                again,
                observed(f"c{i}", failed_evidence(f"c{i}", error="out of memory")),
                valid_from_state_version=1, transition_id="tr", decision_id="dec",
            )

        assert first == again.get(failure_experience_id(FailureKind.OOM, "gbdt")).statement
        assert "batch size" in first          # the judge's repair hint

    def test_policy_level_failures_are_marked_not_repairable(self, index):
        FailureMemory().observe(
            index,
            observed("leaky", failed_evidence("leaky", error="lookahead detected")),
            valid_from_state_version=1, transition_id="tr", decision_id="dec",
        )

        rec = index.get(failure_experience_id(FailureKind.LEAKAGE, "gbdt"))
        assert rec.outcome_summary["policy_level"] is True
        assert rec.outcome_summary["repairable"] is False
        assert "steer away" in rec.statement

    def test_provenance_is_references_not_copies(self, index):
        FailureMemory().observe(
            index,
            observed("c1", failed_evidence("c1", error="shape mismatch")),
            valid_from_state_version=1, transition_id="tr_1", decision_id="dec_1",
        )

        rec = index.get(failure_experience_id(FailureKind.SHAPE, "gbdt"))
        assert rec.candidate_ids == ["c1"]
        assert rec.evidence_ids == ["ev_c1"]
        assert rec.transition_ids == ["tr_1"]
        # no code, no error text, no metric values copied into the record
        dumped = rec.model_dump(mode="json")
        assert "shape mismatch" not in str(dumped.get("outcome_summary"))


class TestReliabilityWeight:
    def test_saturates_in_sample_count(self):
        w1 = reliability_weight(EvidenceLevel.PROVISIONAL, 1)
        w5 = reliability_weight(EvidenceLevel.PROVISIONAL, 5)
        w50 = reliability_weight(EvidenceLevel.PROVISIONAL, 50)
        assert w1 < w5 < w50
        assert w50 - w5 < w5 - w1          # diminishing returns

    def test_ordered_by_evidence_level(self):
        assert (
            reliability_weight(EvidenceLevel.PROVISIONAL, 10)
            < reliability_weight(EvidenceLevel.VISIBLE_MULTISEED, 10)
            < reliability_weight(EvidenceLevel.SEALED_CERTIFIED, 10)
        )


class TestRetrieval:
    def test_hard_filters_before_ranking(self, index):
        memory = FailureMemory()
        for family in ("gbdt", "mlp"):
            memory.observe(
                index,
                observed(f"c_{family}", failed_evidence(f"c_{family}",
                                                        error="shape mismatch"),
                         family=family),
                valid_from_state_version=1, transition_id="tr", decision_id="dec",
            )
        retriever = MemoryRetriever(index)

        bundle = retriever.retrieve(at_version=1, filters={"model_family": "mlp"})

        assert bundle.retrieved_experience_ids == [
            failure_experience_id(FailureKind.SHAPE, "mlp")
        ]

    def test_top_k_is_respected_and_scores_descend(self, index):
        memory = FailureMemory()
        errors = ["shape mismatch", "out of memory", "invalid syntax",
                  "no module named x", "nan in predictions"]
        for i, err in enumerate(errors):
            for _ in range(i + 1):                 # different occurrence counts
                memory.observe(
                    index,
                    observed(f"c{i}", failed_evidence(f"c{i}", error=err)),
                    valid_from_state_version=1, transition_id="tr", decision_id="dec",
                )

        bundle = MemoryRetriever(index).retrieve(at_version=1, top_k=3)

        assert len(bundle.retrieved_experience_ids) == 3
        assert bundle.retrieval_scores == sorted(
            bundle.retrieval_scores, reverse=True
        )
        assert bundle.n_candidates_considered == len(errors)

    def test_bundle_reports_zero_consumption_in_stage_0(self, index):
        bundle = MemoryRetriever(index).retrieve(at_version=0)
        assert bundle.consumed_by_policy is False

    def test_explain_pairs_ids_with_statements(self, index):
        FailureMemory().observe(
            index,
            observed("c1", failed_evidence("c1", error="out of memory")),
            valid_from_state_version=1, transition_id="tr", decision_id="dec",
        )
        retriever = MemoryRetriever(index)
        bundle = retriever.retrieve(at_version=1)

        explained = retriever.explain(bundle)
        assert len(explained) == 1
        eid, score, statement = explained[0]
        assert eid == failure_experience_id(FailureKind.OOM, "gbdt")
        assert score > 0 and "oom" in statement


class TestBarrierIntegration:
    """Consolidation must happen inside the commit window, at next_version."""

    @pytest.fixture
    def wiring(self, manifest, ledger):
        store = StateStore()
        guard = CommitGuard(enforce=True)
        search = SearchArchive(store, guard=guard)
        certified = CertifiedArchive(store, guard=guard)
        index = ExperienceIndex(store, guard=guard)
        barrier = BarrierCommit(
            search_archive=search,
            certified_archive=certified,
            admission=CertifiedAdmission(search, certified, manifest, ledger),
            trajectory_store=TrajectoryStore(),
            state_store=store,
            guard=guard,
            consolidator=ExperienceConsolidator(index),
        )
        return barrier, index

    def _decision(self, version):
        return DecisionRecord(
            decision_id=f"dec_v{version}",
            observation_digest="d",
            structured_action=StructuredAction(expert=ExpertKind.MUTATE),
            state_version=version,
            timestamp=0.0,
        )

    def test_experience_becomes_visible_only_at_the_committed_version(self, wiring):
        barrier, index = wiring
        decision = self._decision(0)
        batch = barrier.open(decision, expected=1)
        batch.stage(CandidateOutcome(
            candidate_id="c1",
            episode_id="E1",
            model_family="gbdt",
            code_hash="h",
            lineage=CandidateLineage(candidate_id="c1", parent_ids=[], expert="mutate"),
            evidence=[failed_evidence("c1", error="shape mismatch")],
        ))

        transition = barrier.commit(batch)

        assert transition.next_state_version == 1
        # The decision was taken at v0 and must not see its own results.
        assert index.visible_at(0) == []
        assert len(index.visible_at(1)) == 1

    def test_consolidation_failure_does_not_abort_the_commit(self, wiring):
        barrier, index = wiring

        class Exploding:
            def consolidate(self, *a, **k):
                raise RuntimeError("boom")

        barrier._consolidator = Exploding()
        batch = barrier.open(self._decision(0), expected=1)
        batch.stage(CandidateOutcome(
            candidate_id="c1", episode_id="E1", model_family="gbdt", code_hash="h",
            evidence=[ok_evidence("c1")],
        ))

        transition = barrier.commit(batch)     # must not raise

        assert transition.next_state_version == 1
        assert index.size() == 0


class TestTrajectoryRecording:
    def test_retrieval_bundles_round_trip_through_jsonl(self, tmp_path, index):
        path = tmp_path / "traj.jsonl"
        store = TrajectoryStore(str(path))
        bundle = MemoryRetriever(index).retrieve(
            at_version=3, query_type=QueryType.POLICY_DECISION, decision_id="dec_1"
        )
        store.record_retrieval(bundle)

        reloaded = TrajectoryStore(str(path))

        assert len(reloaded.retrievals()) == 1
        assert reloaded.retrievals()[0].query_id == bundle.query_id
        assert reloaded.retrievals()[0].state_version == 3

    def test_decision_record_links_to_its_bundle(self):
        decision = DecisionRecord(
            decision_id="dec_1",
            observation_digest="d",
            structured_action=StructuredAction(expert=ExpertKind.MUTATE),
            state_version=1,
            retrieval_bundle_ids=["ret_abc"],
            timestamp=0.0,
        )
        assert decision.retrieval_bundle_ids == ["ret_abc"]

    def test_field_is_optional_so_old_records_still_load(self):
        """Additive field: trajectories written before stage 0 must still parse,
        and the encoder never saw it, so checkpoints stay valid."""
        old = {
            "decision_id": "dec_old",
            "observation_digest": "d",
            "structured_action": {"expert": "mutate"},
            "state_version": 1,
            "timestamp": 0.0,
        }
        parsed = DecisionRecord.model_validate(old)
        assert parsed.retrieval_bundle_ids == []
