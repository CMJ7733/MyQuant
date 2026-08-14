"""BarrierCommit: staging, validation, atomic apply, commit versioning.

Invariant C1 — the archives have exactly one writer, and nothing a batch
produces is observable until that batch commits.
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
from famou.reliability.barrier import (
    BarrierCommit,
    CandidateOutcome,
    DuplicateSubmission,
    IncompleteBatch,
)
from famou.reliability.observation import ObservationBuilder
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


def make_evidence(candidate_id: str, samples, *, eval_id="ev_1") -> EvidenceVector:
    return EvidenceVector(
        candidate_id=candidate_id,
        episode_id="E1",
        eval_id=eval_id,
        fidelity=Fidelity.F2_FULL,
        split_scope="visible_dev",
        data_contract_hash="h",
        rank_ic=MetricDistribution.from_samples(list(samples)),
        validity=1.0,
    )


def make_decision(state_version: int, decision_id="dec_1") -> DecisionRecord:
    return DecisionRecord(
        decision_id=decision_id,
        observation_digest="digest",
        structured_action=StructuredAction(expert=ExpertKind.EXPLORE),
        state_version=state_version,
        timestamp=0.0,
    )


def make_outcome(candidate_id: str, samples=(0.05, 0.051, 0.049)) -> CandidateOutcome:
    return CandidateOutcome(
        candidate_id=candidate_id,
        episode_id="E1",
        model_family="gbdt",
        code_hash=f"hash_{candidate_id}",
        lineage=CandidateLineage(candidate_id=candidate_id, parent_ids=["init_0"]),
        evidence=[make_evidence(candidate_id, samples, eval_id=f"ev_{candidate_id}")],
    )


@pytest.fixture
def wiring(manifest, ledger):
    store = StateStore()
    guard = CommitGuard(enforce=True)
    search = SearchArchive(store, guard=guard)
    certified = CertifiedArchive(store, guard=guard)
    admission = CertifiedAdmission(search, certified, manifest, ledger)
    barrier = BarrierCommit(
        search_archive=search,
        certified_archive=certified,
        admission=admission,
        trajectory_store=TrajectoryStore(),
        state_store=store,
        guard=guard,
    )
    return store, guard, search, certified, barrier


class TestCommitGuard:
    def test_write_outside_window_refused(self, wiring):
        _, _, search, certified, _ = wiring
        with pytest.raises(ArchiveWriteOutsideBarrier):
            search.add_candidate(
                "c1", episode_id="E1", model_family="gbdt", code_hash="h"
            )
        with pytest.raises(ArchiveWriteOutsideBarrier):
            search.add_evidence(make_evidence("c1", [0.05]))
        with pytest.raises(ArchiveWriteOutsideBarrier):
            certified.add_baseline(
                "c1", episode_id="E1", model_family="gbdt", code_hash="h"
            )

    def test_reads_never_blocked(self, wiring):
        _, _, search, certified, _ = wiring
        assert search.all_candidates() == {}
        assert certified.members() == {}
        assert search.get_evidence("nobody") == []

    def test_bootstrap_window_allows_seeding(self, wiring):
        _, _, search, certified, barrier = wiring
        with barrier.bootstrap():
            certified.add_baseline(
                "init_0", episode_id="E1", model_family="gbdt", code_hash="h0"
            )
        assert certified.is_certified("init_0")

    def test_permissive_guard_for_standalone_use(self):
        """Archives built without a guard stay writable, so unit tests and
        bootstrap scripts do not need a barrier."""
        search = SearchArchive(StateStore())
        search.add_candidate("c1", episode_id="E1", model_family="gbdt", code_hash="h")
        assert "c1" in search.all_candidates()


class TestBarrierIsolation:
    def test_staged_results_invisible_before_commit(self, wiring):
        _, _, search, _, barrier = wiring
        batch = barrier.open(make_decision(barrier.state_version), expected=2)
        batch.stage(make_outcome("c1"))

        # c1 exists only in the accumulator
        assert search.all_candidates() == {}
        assert barrier.state_version == 0

        batch.stage(make_outcome("c2"))
        barrier.commit(batch)

        assert set(search.all_candidates()) == {"c1", "c2"}
        assert barrier.state_version == 1

    def test_observation_only_sees_committed_state(self, wiring, ledger):
        _, _, search, certified, barrier = wiring
        builder = ObservationBuilder(search, certified, ledger)

        batch = barrier.open(make_decision(barrier.state_version), expected=1)
        batch.stage(make_outcome("c1"))
        mid = builder.build(episode_id="E1", state_version=barrier.state_version)
        assert mid.search_archive_summary["n_candidates"] == 0

        barrier.commit(batch)
        after = builder.build(episode_id="E1", state_version=barrier.state_version)
        assert after.search_archive_summary["n_candidates"] == 1
        assert after.state_version == 1

    def test_lineage_and_evidence_applied_together(self, wiring):
        _, _, search, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=1)
        batch.stage(make_outcome("c1"))
        barrier.commit(batch)

        assert search.get_lineage("c1").parent_ids == ["init_0"]
        assert len(search.get_evidence("c1")) == 1

    def test_unregistered_outcome_counts_but_writes_nothing(self, wiring):
        """Budget-exhausted / promotion-only rollouts must still balance the
        batch without creating an archive entry."""
        _, _, search, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=1)
        batch.stage(
            CandidateOutcome(candidate_id="c1", episode_id="E1", register=False)
        )
        transition = barrier.commit(batch)

        assert search.all_candidates() == {}
        assert transition.candidate_ids == ["c1"]


class TestBarrierValidation:
    def test_duplicate_submission_rejected(self, wiring):
        _, _, _, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=2)
        batch.stage(make_outcome("c1"))
        with pytest.raises(DuplicateSubmission):
            batch.stage(make_outcome("c1"))

    def test_incomplete_batch_rejected(self, wiring):
        _, _, _, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=3)
        batch.stage(make_outcome("c1"))
        with pytest.raises(IncompleteBatch):
            barrier.commit(batch)
        # ... unless the caller explicitly accepts a short batch
        transition = barrier.commit(batch, allow_incomplete=True)
        assert transition.candidate_ids == ["c1"]

    def test_double_commit_rejected(self, wiring):
        _, _, _, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=1)
        batch.stage(make_outcome("c1"))
        barrier.commit(batch)
        with pytest.raises(Exception):
            barrier.commit(batch)

    def test_staging_after_commit_rejected(self, wiring):
        _, _, _, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=1)
        batch.stage(make_outcome("c1"))
        barrier.commit(batch)
        with pytest.raises(Exception):
            batch.stage(make_outcome("c2"))

    def test_stale_batch_flagged_but_kept(self, wiring):
        """A batch decided at v0 but committed after another batch landed is
        stale. The compute was already spent, so it commits — flagged."""
        _, _, _, _, barrier = wiring
        stale_batch = barrier.open(make_decision(0, "dec_stale"), expected=1)
        stale_batch.stage(make_outcome("c_stale"))

        other = barrier.open(make_decision(0, "dec_other"), expected=1)
        other.stage(make_outcome("c_other"))
        fresh = barrier.commit(other)
        assert fresh.stale is False

        transition = barrier.commit(stale_batch)
        assert transition.stale is True
        assert transition.state_version == 0
        assert transition.next_state_version == 2


class TestCommitVersioning:
    def test_version_monotonic_and_persisted(self, wiring):
        store, _, _, _, barrier = wiring
        for expected_version in (1, 2, 3):
            batch = barrier.open(make_decision(barrier.state_version), expected=1)
            batch.stage(make_outcome(f"c{expected_version}"))
            barrier.commit(batch)
            assert barrier.state_version == expected_version
        # version lives in the StateStore, so it checkpoints with the archives
        assert store.get("reliability", "state_version") == 3

    def test_transition_carries_causal_boundary(self, wiring):
        _, _, _, _, barrier = wiring
        batch = barrier.open(make_decision(0), expected=1)
        batch.stage(make_outcome("c1"))
        transition = barrier.commit(batch)

        assert transition.state_version == 0        # s
        assert transition.next_state_version == 1   # s'
        assert transition.decision_ref == "dec_1"
        assert transition.evidence_refs == ["ev_c1"]

    def test_transition_recorded_in_trajectory(self, wiring, manifest, ledger):
        store, guard, search, certified, _ = wiring
        trajectory = TrajectoryStore()
        barrier = BarrierCommit(
            search_archive=search,
            certified_archive=certified,
            admission=CertifiedAdmission(search, certified, manifest, ledger),
            trajectory_store=trajectory,
            state_store=store,
            guard=guard,
        )
        batch = barrier.open(make_decision(0), expected=1)
        batch.stage(make_outcome("c1"))
        barrier.commit(batch)
        assert len(trajectory.transitions()) == 1
