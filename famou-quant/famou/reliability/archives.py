"""Search Archive and Certified Archive.

Search Archive  = "what we have tried"   (all candidates, all fidelities,
                  including failures — failure is a learning signal)
Certified Archive = "what we currently believe" (initial baselines + only
                  candidates that passed the sealed gate through
                  CertifiedAdmission)

Both are backed by the StateStore so they checkpoint with the experiment.
Programs themselves stay in the famou Experiment.archive (single source of
truth for code); these archives store *evidence and trust metadata* keyed by
candidate_id.

Invariant C1: in the running system the archives have exactly ONE writer —
``BarrierCommit``. ``CommitGuard`` enforces it: every mutating method
refuses to run outside a commit window. That is what makes it safe for
``ObservationBuilder`` to read the archives directly — with a single
transactional writer, "live" and "last committed" are the same thing, so no
separate snapshot copy is needed.
"""

from __future__ import annotations

import contextlib
import threading
from typing import Any, Dict, Iterator, List, Optional

from famou.core.state import StateStore
from famou.reliability.budget import BudgetLedger
from famou.reliability.types import (
    CandidateLineage,
    EvidenceVector,
    Fidelity,
    FrozenSplitManifest,
    GateRequest,
    GateVerdict,
    GateVerdictKind,
)


class ArchiveWriteOutsideBarrier(RuntimeError):
    """Raised when archive state is mutated outside a commit window.

    This is a design-invariant violation, not a runtime hiccup: a write that
    lands outside the barrier becomes visible to the next AgentObservation
    without a state_version bump, which breaks the s -> a -> s' boundary the
    RL trainer relies on.
    """


class CommitGuard:
    """Marks the window in which archive writes are legal.

    Depth is thread-local on purpose: the barrier commits on the controller
    thread, so a rollout worker that tries to write directly still sees depth
    0 and is refused even while a commit is in flight elsewhere.

    ``enforce=False`` builds a permissive guard for experiment bootstrap
    (seeding certified baselines before any barrier exists) and for unit
    tests that drive the archives directly.
    """

    def __init__(self, enforce: bool = True):
        self.enforce = enforce
        self._local = threading.local()

    @property
    def depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @property
    def is_open(self) -> bool:
        return self.depth > 0

    @contextlib.contextmanager
    def window(self, label: str = "commit") -> Iterator[None]:
        """Open a write window. Re-entrant so a commit may call helpers."""
        self._local.depth = self.depth + 1
        self._local.label = label
        try:
            yield
        finally:
            self._local.depth = self.depth - 1

    def check(self, operation: str) -> None:
        if self.enforce and not self.is_open:
            raise ArchiveWriteOutsideBarrier(
                f"{operation} called outside a BarrierCommit window. "
                "Archive writes must be staged and applied by the barrier "
                "(invariant C1); see famou.reliability.barrier."
            )


def permissive_guard() -> CommitGuard:
    """Guard that allows direct writes (bootstrap / standalone tests)."""
    return CommitGuard(enforce=False)



class SearchArchive:
    """Append-only store of all candidates and their evidence."""

    def __init__(self, state_store: StateStore, guard: Optional[CommitGuard] = None):
        self._store = state_store
        self._lock = threading.Lock()
        self._guard = guard or permissive_guard()
        if self._store.get("reliability", "search_archive", default=None) is None:
            self._store.set(
                "reliability", "search_archive",
                value={"candidates": {}, "evidence": {}, "lineage": {}},
            )

    # ------------------------------------------------------------------
    def add_candidate(
        self,
        candidate_id: str,
        *,
        episode_id: str,
        model_family: str,
        code_hash: str,
        lineage: Optional[CandidateLineage] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._guard.check("SearchArchive.add_candidate")
        with self._lock:
            candidates = self._store.get("reliability", "search_archive", "candidates")
            if candidate_id not in candidates:
                candidates[candidate_id] = {
                    "candidate_id": candidate_id,
                    "episode_id": episode_id,
                    "model_family": model_family,
                    "code_hash": code_hash,
                    "meta": meta or {},
                    "gate_attempts": 0,
                    "last_gate_verdict": None,
                }
                self._store.set(
                    "reliability", "search_archive", "candidates", value=candidates
                )
            # Lineage is written independently of the candidate record. It used
            # to sit behind an early `return` on duplicate ids, which meant a
            # caller that registered the candidate first and its lineage second
            # silently produced an empty lineage DAG.
            if lineage is not None:
                existing = self._store.get(
                    "reliability", "search_archive", "lineage", candidate_id,
                    default=None,
                )
                if existing is None:
                    self._store.set(
                        "reliability", "search_archive", "lineage", candidate_id,
                        value=lineage.model_dump(mode="json"),
                    )

    def add_evidence(self, evidence: EvidenceVector) -> None:
        """Append one EvidenceVector. Multiple per candidate accumulate."""
        self._guard.check("SearchArchive.add_evidence")
        with self._lock:
            all_evidence = self._store.get("reliability", "search_archive", "evidence")
            slot = all_evidence.setdefault(evidence.candidate_id, [])
            slot.append(evidence.model_dump(mode="json"))
            self._store.set(
                "reliability", "search_archive", "evidence", value=all_evidence
            )

    def record_gate_attempt(
        self, candidate_id: str, verdict: Optional[GateVerdict] = None
    ) -> None:
        """Record that one sealed query was spent on this candidate.

        Without this counter a candidate that keeps satisfying the promotion
        heuristics (stable F2 evidence, above incumbent, still uncertified
        because the gate REJECTed it) gets re-submitted every iteration and
        drains the whole per-episode sealed budget.
        """
        self._guard.check("SearchArchive.record_gate_attempt")
        with self._lock:
            candidates = self._store.get("reliability", "search_archive", "candidates")
            record = candidates.get(candidate_id)
            if record is None:
                return
            record["gate_attempts"] = int(record.get("gate_attempts", 0)) + 1
            if verdict is not None:
                record["last_gate_verdict"] = verdict.model_dump(mode="json")
            self._store.set(
                "reliability", "search_archive", "candidates", value=candidates
            )

    def gate_attempts(self, candidate_id: str) -> int:
        record = self.get_candidate(candidate_id)
        return int(record.get("gate_attempts", 0)) if record else 0

    def record_promotion_check(self, candidate_id: str, n_evidence: int) -> None:
        """Note that the promotion policy has judged this candidate.

        Distinct from ``gate_attempts``, which counts sealed queries actually
        spent. A candidate the policy DECLINED to gate spends nothing, so
        gate_attempts stays 0 — and a heuristic guarded only on that will
        re-propose the same candidate every iteration forever. Recording the
        evidence count at check time makes the guard precise: reconsider a
        candidate exactly when new evidence has arrived, not sooner.
        """
        self._guard.check("SearchArchive.record_promotion_check")
        with self._lock:
            candidates = self._store.get("reliability", "search_archive", "candidates")
            record = candidates.get(candidate_id)
            if record is None:
                return
            record["promotion_checked_at"] = int(n_evidence)
            self._store.set(
                "reliability", "search_archive", "candidates", value=candidates
            )

    def promotion_checked_at(self, candidate_id: str) -> int:
        record = self.get_candidate(candidate_id)
        return int(record.get("promotion_checked_at", 0)) if record else 0

    def evidence_count(self, candidate_id: str) -> int:
        return len(self._store.get(
            "reliability", "search_archive", "evidence", candidate_id, default=[]
        ))

    # ------------------------------------------------------------------
    def get_evidence(
        self,
        candidate_id: str,
        *,
        fidelity: Optional[Fidelity] = None,
    ) -> List[EvidenceVector]:
        slot = self._store.get(
            "reliability", "search_archive", "evidence", candidate_id, default=[]
        )
        out = [EvidenceVector.model_validate(e) for e in slot]
        if fidelity is not None:
            out = [e for e in out if e.fidelity == fidelity]
        return out

    def best_evidence(self, candidate_id: str) -> Optional[EvidenceVector]:
        """Highest-fidelity valid evidence for a candidate (F2 preferred)."""
        evs = [e for e in self.get_evidence(candidate_id) if e.is_valid]
        if not evs:
            return None
        evs.sort(key=lambda e: (int(e.fidelity.value), e.rank_ic.mean if e.rank_ic else float("-inf")))
        return evs[-1]

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(
            "reliability", "search_archive", "candidates", candidate_id, default=None
        )

    def all_candidates(self) -> Dict[str, Dict[str, Any]]:
        return self._store.get("reliability", "search_archive", "candidates", default={})

    def get_lineage(self, candidate_id: str) -> Optional[CandidateLineage]:
        data = self._store.get(
            "reliability", "search_archive", "lineage", candidate_id, default=None
        )
        return CandidateLineage.model_validate(data) if data else None

    def mark_lineage_certified(self, candidate_id: str) -> None:
        """Flag a lineage node as certified (called by CertifiedAdmission)."""
        self._guard.check("SearchArchive.mark_lineage_certified")
        with self._lock:
            data = self._store.get(
                "reliability", "search_archive", "lineage", candidate_id, default=None
            )
            if data is None:
                return
            lineage = CandidateLineage.model_validate(data)
            lineage.certified = True
            self._store.set(
                "reliability", "search_archive", "lineage", candidate_id,
                value=lineage.model_dump(mode="json"),
            )

    def code_hash_exists(self, code_hash: str) -> bool:
        """Dedup check: has this exact code been tried before?"""
        for cand in self.all_candidates().values():
            if cand.get("code_hash") == code_hash:
                return True
        return False

    def family_stats(self) -> Dict[str, Dict[str, float]]:
        """Success/failure rates per model family (for the observation)."""
        stats: Dict[str, Dict[str, float]] = {}
        for cid, cand in self.all_candidates().items():
            family = cand.get("model_family", "unknown")
            entry = stats.setdefault(family, {"n": 0, "n_valid": 0, "n_f2": 0})
            entry["n"] += 1
            evs = self.get_evidence(cid)
            if any(e.is_valid for e in evs):
                entry["n_valid"] += 1
            if any(e.fidelity == Fidelity.F2_FULL and e.is_valid for e in evs):
                entry["n_f2"] += 1
        for entry in stats.values():
            entry["valid_rate"] = entry["n_valid"] / max(1, entry["n"])
        return stats


class CertifiedArchive:
    """Trusted population: baselines + gate-promoted candidates only."""

    def __init__(self, state_store: StateStore, guard: Optional[CommitGuard] = None):
        self._store = state_store
        self._lock = threading.Lock()
        self._guard = guard or permissive_guard()
        if self._store.get("reliability", "certified_archive", default=None) is None:
            self._store.set("reliability", "certified_archive", value={"members": {}})

    def add_baseline(
        self,
        candidate_id: str,
        *,
        episode_id: str,
        model_family: str,
        code_hash: str,
        note: str = "initial baseline",
    ) -> None:
        """Seed a hand-crafted baseline (no gate required, marked as such)."""
        self._guard.check("CertifiedArchive.add_baseline")
        with self._lock:
            members = self._store.get("reliability", "certified_archive", "members")
            members[candidate_id] = {
                "candidate_id": candidate_id,
                "episode_id": episode_id,
                "model_family": model_family,
                "code_hash": code_hash,
                "admission": "baseline",
                "note": note,
                "gate_verdict": None,
                "protocol_version": None,
            }
            self._store.set("reliability", "certified_archive", "members", value=members)

    def admit(
        self,
        candidate_id: str,
        *,
        episode_id: str,
        model_family: str,
        code_hash: str,
        verdict: GateVerdict,
        protocol_version: str,
    ) -> None:
        """Admit a PROMOTED candidate. Caller must be CertifiedAdmission."""
        self._guard.check("CertifiedArchive.admit")
        if verdict.verdict != GateVerdictKind.PROMOTE:
            raise ValueError(
                f"CertifiedArchive.admit called with non-PROMOTE verdict: {verdict.verdict}"
            )
        with self._lock:
            members = self._store.get("reliability", "certified_archive", "members")
            members[candidate_id] = {
                "candidate_id": candidate_id,
                "episode_id": episode_id,
                "model_family": model_family,
                "code_hash": code_hash,
                "admission": "gate",
                "gate_verdict": verdict.model_dump(mode="json"),
                "protocol_version": protocol_version,
            }
            self._store.set("reliability", "certified_archive", "members", value=members)

    def members(self) -> Dict[str, Dict[str, Any]]:
        return self._store.get("reliability", "certified_archive", "members", default={})

    def is_certified(self, candidate_id: str) -> bool:
        return candidate_id in self.members()

    def parent_pool(self, episode_id: Optional[str] = None) -> List[str]:
        """Default parent pool for the next search round.

        Certified members are global (cross-episode transfer is the point of
        the archive), but each entry records its origin episode.
        """
        return list(self.members().keys())


class CertifiedAdmission:
    """Verifies a GateRequest/Verdict pair before Certified Archive entry.

    This is the last gate on the trust boundary, so every check is mandatory
    and none of the collaborators are optional:

    - verdict is PROMOTE
    - the candidate exists in the Search Archive
    - the frozen code hash still matches the archived record
    - the request's protocol_version matches the live manifest
    - the request's data_contract_hash matches the live manifest hash
    - the one-time query token was actually CONSUMED by the gate

    The token check is what makes a hand-constructed ``GateVerdict(PROMOTE)``
    useless: without a token that the real gate burned, admission fails.
    """

    def __init__(
        self,
        search_archive: SearchArchive,
        certified_archive: CertifiedArchive,
        manifest: FrozenSplitManifest,
        ledger: BudgetLedger,
    ):
        self._search = search_archive
        self._certified = certified_archive
        self._manifest = manifest
        self._ledger = ledger
        #: reason the last verify_and_admit() call refused (None if it admitted)
        self.last_rejection: Optional[str] = None

    def _refusal_reason(
        self, request: GateRequest, verdict: GateVerdict
    ) -> Optional[str]:
        if verdict.verdict != GateVerdictKind.PROMOTE:
            return f"verdict is {verdict.verdict.value}, not PROMOTE"

        record = self._search.get_candidate(request.candidate_id)
        if record is None:
            return f"candidate {request.candidate_id} not in search archive"
        if record.get("code_hash") != request.candidate_code_hash:
            return "code hash drifted after freezing"

        if request.protocol_version != self._manifest.protocol_version:
            return (
                f"protocol version mismatch: request={request.protocol_version} "
                f"manifest={self._manifest.protocol_version}"
            )
        if request.data_contract_hash != self._manifest.compute_hash():
            return "data contract hash mismatch"

        status = self._ledger.token_status(request.query_token)
        if status != "consumed":
            return f"query token not consumed by the gate (status={status})"
        return None

    def verify_and_admit(
        self,
        request: GateRequest,
        verdict: GateVerdict,
        *,
        model_family: str,
    ) -> bool:
        reason = self._refusal_reason(request, verdict)
        self.last_rejection = reason
        if reason is not None:
            return False

        self._certified.admit(
            request.candidate_id,
            episode_id=request.episode_id,
            model_family=model_family,
            code_hash=request.candidate_code_hash,
            verdict=verdict,
            protocol_version=request.protocol_version,
        )
        lineage = self._search.get_lineage(request.candidate_id)
        if lineage is not None:
            self._search.mark_lineage_certified(request.candidate_id)
        return True
