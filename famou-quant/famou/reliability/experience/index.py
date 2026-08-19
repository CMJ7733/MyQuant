"""ExperienceIndex — version-aware store of ExperienceRecords.

Backed by the StateStore so it checkpoints with the experiment, and guarded by
the same ``CommitGuard`` as the archives: writes are legal only inside a
barrier window (invariant E2). Reads are unguarded but must name the version
they are reading at (invariant E1) — there is deliberately no "give me
everything current" accessor on the retrieval path, because that is precisely
the call that would silently break trajectory replay.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from famou.core.state import StateStore
from famou.reliability.archives import CommitGuard, permissive_guard
from famou.reliability.experience.types import ExperienceRecord

_INDEX_PATH = ("reliability", "experience_index")
_RECORDS = "records"
#: candidate_id -> failure kind, so a later candidate can be recognised as a
#: repair attempt on its parent's failure without rescanning the archive.
_CANDIDATE_FAILURES = "candidate_failures"


class ExperienceIndex:
    """Append/upsert store of experience, readable only at a stated version."""

    def __init__(self, state_store: StateStore, guard: Optional[CommitGuard] = None):
        self._store = state_store
        self._lock = threading.Lock()
        self._guard = guard or permissive_guard()
        if self._store.get(*_INDEX_PATH, default=None) is None:
            self._store.set(
                *_INDEX_PATH, value={_RECORDS: {}, _CANDIDATE_FAILURES: {}}
            )

    # ------------------------------------------------------------------
    # writes (guarded)
    # ------------------------------------------------------------------

    def upsert(self, record: ExperienceRecord) -> None:
        """Insert or replace a record by ``experience_id``.

        Upsert rather than append because aggregate memories (a failure
        pattern, a semantic regularity) accumulate evidence over time and must
        stay one record. ``valid_from_state_version`` is NOT advanced on
        update: it marks when the pattern first became retrievable, and moving
        it forward would retroactively hide the record from decisions that
        legitimately could have seen it.
        """
        self._guard.check("ExperienceIndex.upsert")
        with self._lock:
            records = self._store.get(*_INDEX_PATH, _RECORDS)
            existing = records.get(record.experience_id)
            payload = record.model_dump(mode="json")
            if existing is not None:
                payload["valid_from_state_version"] = existing[
                    "valid_from_state_version"
                ]
            records[record.experience_id] = payload
            self._store.set(*_INDEX_PATH, _RECORDS, value=records)

    def record_candidate_failure(
        self, candidate_id: str, *, failure_kind: str, model_family: str
    ) -> None:
        """Remember that a candidate failed, and how.

        The family is stored alongside the kind because a repair is credited
        to the aggregate keyed by the *parent's* family — the child that fixes
        an mlp shape error may itself be a different family, and crediting the
        child's family would file the recovery under a pattern that never
        happened.
        """
        self._guard.check("ExperienceIndex.record_candidate_failure")
        with self._lock:
            failures = self._store.get(*_INDEX_PATH, _CANDIDATE_FAILURES)
            failures[candidate_id] = {
                "kind": failure_kind,
                "model_family": model_family,
            }
            self._store.set(*_INDEX_PATH, _CANDIDATE_FAILURES, value=failures)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def candidate_failure(self, candidate_id: str) -> Optional[Dict[str, str]]:
        return self._store.get(
            *_INDEX_PATH, _CANDIDATE_FAILURES, candidate_id, default=None
        )

    def get(self, experience_id: str) -> Optional[ExperienceRecord]:
        raw = self._store.get(*_INDEX_PATH, _RECORDS, experience_id, default=None)
        return ExperienceRecord.model_validate(raw) if raw else None

    def visible_at(self, state_version: int) -> List[ExperienceRecord]:
        """Every non-deprecated record retrievable at ``state_version``.

        The version filter is invariant E1. A record written by the barrier
        that produced version n+1 carries ``valid_from_state_version = n+1``,
        so a decision taken at version n cannot see the results of the batch
        that was still in flight when it was made.
        """
        out: List[ExperienceRecord] = []
        for raw in self._all_raw().values():
            if raw.get("deprecated"):
                continue
            if int(raw.get("valid_from_state_version", 0)) > state_version:
                continue
            out.append(ExperienceRecord.model_validate(raw))
        return out

    def all_records(self) -> List[ExperienceRecord]:
        """Every record regardless of version. For audit and tests only —
        never call this on the retrieval path (see the module docstring)."""
        return [ExperienceRecord.model_validate(r) for r in self._all_raw().values()]

    def size(self) -> int:
        return len(self._all_raw())

    # ------------------------------------------------------------------

    def _all_raw(self) -> Dict[str, dict]:
        return self._store.get(*_INDEX_PATH, _RECORDS, default={})
