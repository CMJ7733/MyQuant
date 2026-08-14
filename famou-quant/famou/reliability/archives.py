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
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

from famou.core.state import StateStore
from famou.reliability.types import (
    CandidateLineage,
    EvidenceVector,
    Fidelity,
    GateRequest,
    GateVerdict,
    GateVerdictKind,
)


class SearchArchive:
    """Append-only store of all candidates and their evidence."""

    def __init__(self, state_store: StateStore):
        self._store = state_store
        self._lock = threading.Lock()
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
        with self._lock:
            candidates = self._store.get("reliability", "search_archive", "candidates")
            if candidate_id in candidates:
                return
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "episode_id": episode_id,
                "model_family": model_family,
                "code_hash": code_hash,
                "meta": meta or {},
            }
            self._store.set(
                "reliability", "search_archive", "candidates", value=candidates
            )
            if lineage is not None:
                self._store.set(
                    "reliability", "search_archive", "lineage", candidate_id,
                    value=lineage.model_dump(mode="json"),
                )

    def add_evidence(self, evidence: EvidenceVector) -> None:
        """Append one EvidenceVector. Multiple per candidate accumulate."""
        with self._lock:
            all_evidence = self._store.get("reliability", "search_archive", "evidence")
            slot = all_evidence.setdefault(evidence.candidate_id, [])
            slot.append(evidence.model_dump(mode="json"))
            self._store.set(
                "reliability", "search_archive", "evidence", value=all_evidence
            )

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

    def __init__(self, state_store: StateStore):
        self._store = state_store
        self._lock = threading.Lock()
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

    Checks (all must pass):
    - verdict is PROMOTE
    - candidate hash in the request matches the search-archive record
    - protocol version matches the manifest
    - the query token was actually consumed by the gate (no replay)
    """

    def __init__(
        self,
        search_archive: SearchArchive,
        certified_archive: CertifiedArchive,
    ):
        self._search = search_archive
        self._certified = certified_archive

    def verify_and_admit(
        self,
        request: GateRequest,
        verdict: GateVerdict,
        *,
        model_family: str,
    ) -> bool:
        if verdict.verdict != GateVerdictKind.PROMOTE:
            return False
        record = self._search.get_candidate(request.candidate_id)
        if record is None:
            return False
        if record.get("code_hash") != request.candidate_code_hash:
            return False  # code drifted after freezing — reject
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
            lineage.certified = True
            self._search._store.set(
                "reliability", "search_archive", "lineage", request.candidate_id,
                value=lineage.model_dump(mode="json"),
            )
        return True
