"""BarrierCommit — the single transactional writer of the reliability state.

Invariant C1 (design review, frozen)
------------------------------------
A rollout's results do NOT become visible to the agent when that rollout
finishes. They are staged in a ``BatchAccumulator``; once the whole
WorkBatch is accounted for, the barrier validates it and applies everything
as ONE atomic state update:

    rollout results -> stage -> (await batch + validate)
        -> write Search Archive + Certified Archive
        -> state_version n -> n+1
        -> emit Transition

Why this matters: ``AgentObservation`` is built entirely from the two
archives, so archive contents *are* what the policy sees. If rollout #2 of a
batch wrote its evidence the moment it finished, a decision taken while
rollouts #1 and #3 were still running would observe a state that (a) has no
version number, (b) depends on worker scheduling order, and therefore (c)
cannot be reconstructed when the RL trainer replays the trajectory. With a
single transactional writer, "live archive" and "last committed state" are
the same object, so ``ObservationBuilder`` can keep reading the archives
directly and no shadow snapshot is needed.

The alternative (write eagerly, snapshot on read) was rejected: it leaves
two sources of truth, and any future caller that reaches for
``SearchArchive.all_candidates()`` silently bypasses the isolation.

Scope note: this module implements the *reliability-side* half. The Evolver
still calls ``Strategy.forward()`` once per rollout and gets a single
Rollout back, so today every batch has size 1 and the barrier commits
immediately. Wiring ``Decision{record, work_batch}`` through the Evolver so
batches are genuinely > 1 is the second half; the accumulator API is already
shaped for it (``expected``, ``complete``, per-outcome staging).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from famou.core.state import StateStore
from famou.reliability.archives import (
    CertifiedAdmission,
    CertifiedArchive,
    CommitGuard,
    SearchArchive,
)
from famou.reliability.trajectory import TrajectoryStore, build_transition
from famou.reliability.types import (
    CandidateLineage,
    DecisionRecord,
    EvaluationCost,
    EvidenceVector,
    GateRequest,
    GateVerdict,
    GateVerdictKind,
    Transition,
)


_STATE_VERSION_PATH = ("reliability", "state_version")


class BatchIntegrityError(RuntimeError):
    """A WorkBatch could not be committed as a coherent unit."""


class DuplicateSubmission(BatchIntegrityError):
    """The same candidate was staged twice in one batch."""


class IncompleteBatch(BatchIntegrityError):
    """Fewer results came back than the batch declared."""


@dataclass
class CandidateOutcome:
    """Everything one rollout produced, held out of the archives until commit.

    ``register=False`` covers rollouts that must count toward batch
    completeness without creating an archive entry: a budget-exhausted
    evaluation, or a promotion-only decision whose candidate is already
    committed from an earlier batch.
    """

    candidate_id: str
    episode_id: str
    model_family: str = "unknown"
    code_hash: str = ""
    register: bool = True
    lineage: Optional[CandidateLineage] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    evidence: List[EvidenceVector] = field(default_factory=list)
    gate_request: Optional[GateRequest] = None
    gate_verdict: Optional[GateVerdict] = None
    #: eval_ids of already-committed evidence this outcome is reasoning about
    prior_evidence_refs: List[str] = field(default_factory=list)
    #: evidence count when the promotion policy judged this candidate; None if
    #: promotion was never considered for it in this batch
    promotion_checked_at: Optional[int] = None
    cost: EvaluationCost = field(default_factory=EvaluationCost)

    def evidence_refs(self) -> List[str]:
        return [e.eval_id for e in self.evidence] + list(self.prior_evidence_refs)


class BatchAccumulator:
    """Staging area for one decision's WorkBatch. Not yet visible to anyone."""

    def __init__(self, decision: DecisionRecord, expected: int):
        self.decision = decision
        self.expected = expected
        self.outcomes: List[CandidateOutcome] = []
        self.committed = False
        self._lock = threading.Lock()

    def stage(self, outcome: CandidateOutcome) -> None:
        with self._lock:
            if self.committed:
                raise BatchIntegrityError(
                    f"batch for {self.decision.decision_id} already committed"
                )
            if any(o.candidate_id == outcome.candidate_id for o in self.outcomes):
                raise DuplicateSubmission(
                    f"candidate {outcome.candidate_id} staged twice in "
                    f"decision {self.decision.decision_id}"
                )
            self.outcomes.append(outcome)

    @property
    def complete(self) -> bool:
        return len(self.outcomes) >= self.expected

    def total_cost(self) -> EvaluationCost:
        total = EvaluationCost()
        for o in self.outcomes:
            total.wall_seconds += o.cost.wall_seconds
            total.gpu_seconds += o.cost.gpu_seconds
            total.peak_memory_mb = max(total.peak_memory_mb, o.cost.peak_memory_mb)
            total.llm_tokens += o.cost.llm_tokens
            total.visible_query_count += o.cost.visible_query_count
            total.sealed_query_count += o.cost.sealed_query_count
            total.retrieval_tokens += o.cost.retrieval_tokens
        return total


class BarrierCommit:
    """Waits for a batch, validates it, and applies it atomically.

    Validation follows the four checks the design calls for: which decision
    the results belong to, which state version they were produced against
    (stale detection), whether any result is missing, and whether any was
    submitted twice.
    """

    def __init__(
        self,
        *,
        search_archive: SearchArchive,
        certified_archive: CertifiedArchive,
        admission: CertifiedAdmission,
        trajectory_store: TrajectoryStore,
        state_store: StateStore,
        guard: CommitGuard,
        consolidator: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        self._search = search_archive
        self._certified = certified_archive
        self._admission = admission
        self._trajectory = trajectory_store
        self._store = state_store
        self._guard = guard
        self._consolidator = consolidator
        self.logger = logger
        self._commit_lock = threading.Lock()
        if self._store.get(*_STATE_VERSION_PATH, default=None) is None:
            self._store.set(*_STATE_VERSION_PATH, value=0)

    # ------------------------------------------------------------------
    # state version
    # ------------------------------------------------------------------

    @property
    def state_version(self) -> int:
        """The version every committed write is visible at."""
        return int(self._store.get(*_STATE_VERSION_PATH, default=0))

    def set_state_version(self, version: int) -> None:
        """Restore a checkpointed version (used by Strategy.load_state)."""
        self._store.set(*_STATE_VERSION_PATH, value=int(version))

    # ------------------------------------------------------------------
    # batch lifecycle
    # ------------------------------------------------------------------

    def open(self, decision: DecisionRecord, *, expected: int = 1) -> BatchAccumulator:
        return BatchAccumulator(decision, expected)

    def bootstrap(self):
        """Write window for pre-experiment seeding (certified baselines).

        Bootstrap writes are legitimate — they happen before any decision
        exists — but they still go through the guard so the "no unguarded
        writer" property holds for the whole process lifetime.
        """
        return self._guard.window("bootstrap")

    def commit(
        self,
        batch: BatchAccumulator,
        *,
        allow_incomplete: bool = False,
        done: bool = False,
    ) -> Transition:
        """Validate and apply one batch. Returns the emitted Transition."""
        with self._commit_lock:
            if batch.committed:
                raise BatchIntegrityError(
                    f"batch for {batch.decision.decision_id} committed twice"
                )
            if not batch.complete and not allow_incomplete:
                raise IncompleteBatch(
                    f"decision {batch.decision.decision_id}: expected "
                    f"{batch.expected} results, got {len(batch.outcomes)}"
                )

            version_at_decision = batch.decision.state_version
            current = self.state_version
            stale = version_at_decision != current
            if stale and self.logger:
                self.logger.warning(
                    f"[Barrier] stale batch {batch.decision.decision_id}: decided at "
                    f"v{version_at_decision}, committing against v{current}. "
                    "Results are kept (the compute was already spent) but the "
                    "transition is flagged."
                )

            evidence_refs: List[str] = []
            candidate_ids: List[str] = []
            verdict: Optional[GateVerdict] = None

            # Single write window: everything below lands in the same
            # state version, or the guard refuses it.
            with self._guard.window(f"commit:{batch.decision.decision_id}"):
                for outcome in batch.outcomes:
                    candidate_ids.append(outcome.candidate_id)
                    self._apply_outcome(outcome)
                    evidence_refs.extend(outcome.evidence_refs())
                    if outcome.gate_verdict is not None:
                        verdict = outcome.gate_verdict

                next_version = current + 1
                self._store.set(*_STATE_VERSION_PATH, value=next_version)

                transition = build_transition(
                    decision=batch.decision,
                    candidate_ids=candidate_ids,
                    evidence_ids=evidence_refs,
                    costs=batch.total_cost(),
                    gate_verdict=verdict,
                    done=done,
                    next_state_version=next_version,
                    stale=stale,
                )

                # Experience is indexed inside the same window, so a record
                # can never become retrievable at a version the archives have
                # not reached (experience invariant E2). It is written with
                # valid_from = next_version: the batch's lessons appear only
                # once its results do.
                self._consolidate(batch, transition, next_version)

            batch.committed = True
            self._trajectory.record_transition(transition)
            return transition

    # ------------------------------------------------------------------

    def _consolidate(self, batch: BatchAccumulator, transition, next_version: int) -> None:
        """Fold the committed batch into the experience index, if one is wired."""
        if self._consolidator is None:
            return
        from famou.reliability.experience import ObservedOutcome

        observed = [
            ObservedOutcome(
                candidate_id=o.candidate_id,
                episode_id=o.episode_id,
                model_family=o.model_family,
                evidence=list(o.evidence),
                parent_ids=list(o.lineage.parent_ids) if o.lineage else [],
                expert=(o.lineage.expert if o.lineage and o.lineage.expert
                        else str(o.meta.get("expert", "unknown"))),
            )
            for o in batch.outcomes
        ]
        try:
            self._consolidator.consolidate(
                observed,
                valid_from_state_version=next_version,
                transition_id=transition.transition_id,
                decision_id=batch.decision.decision_id,
                policy_version=batch.decision.policy_version,
            )
        except Exception as e:  # pragma: no cover - defensive
            # Never let a memory-layer fault destroy a committed batch.
            if self.logger:
                self.logger.warning(f"[Experience] consolidation skipped: {e}")

    # ------------------------------------------------------------------

    def _apply_outcome(self, outcome: CandidateOutcome) -> None:
        """Write one rollout's results. Runs inside the commit window only."""
        if outcome.register:
            self._search.add_candidate(
                outcome.candidate_id,
                episode_id=outcome.episode_id,
                model_family=outcome.model_family,
                code_hash=outcome.code_hash,
                meta=outcome.meta,
                lineage=outcome.lineage,
            )
        for evidence in outcome.evidence:
            self._search.add_evidence(evidence)

        if outcome.promotion_checked_at is not None:
            self._search.record_promotion_check(
                outcome.candidate_id, outcome.promotion_checked_at
            )

        if outcome.gate_verdict is None:
            return

        # The sealed query was already spent during execution; recording the
        # attempt is what stops the candidate being re-gated next round.
        self._search.record_gate_attempt(outcome.candidate_id, outcome.gate_verdict)
        if (
            outcome.gate_verdict.verdict == GateVerdictKind.PROMOTE
            and outcome.gate_request is not None
        ):
            admitted = self._admission.verify_and_admit(
                outcome.gate_request,
                outcome.gate_verdict,
                model_family=outcome.model_family,
            )
            if self.logger:
                if admitted:
                    self.logger.info(
                        f"[Certified] {outcome.candidate_id} admitted"
                    )
                else:
                    self.logger.warning(
                        f"[Certified] {outcome.candidate_id} refused: "
                        f"{self._admission.last_rejection}"
                    )
