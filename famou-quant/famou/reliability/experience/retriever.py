"""MemoryRetriever — the read side, and the only thing the strategy calls.

Every retrieval names the state version it reads at, and every retrieval
produces a ``RetrievalBundle`` that is recorded whether or not anyone uses the
result. In stages 0-1 nothing uses it: the bundle exists so that when
retrieval *is* wired into the policy, there is a matched baseline of decisions
taken without it. Wiring first and measuring later would leave no way to tell
whether retrieval helped.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from famou.reliability.experience.failure import FailureMemory
from famou.reliability.experience.index import ExperienceIndex
from famou.reliability.experience.types import (
    RETRIEVAL_VERSION,
    ExperienceRecord,
    QueryType,
    RetrievalBundle,
)


class MemoryRetriever:
    """Version-scoped retrieval over the experience index."""

    def __init__(
        self,
        index: ExperienceIndex,
        *,
        failure_memory: Optional[FailureMemory] = None,
        retrieval_version: str = RETRIEVAL_VERSION,
    ):
        self._index = index
        self._failure = failure_memory or FailureMemory()
        self.retrieval_version = retrieval_version

    # ------------------------------------------------------------------

    def retrieve(
        self,
        *,
        at_version: int,
        query_type: QueryType = QueryType.POLICY_DECISION,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 4,
        decision_id: Optional[str] = None,
    ) -> RetrievalBundle:
        """Retrieve at ``at_version``. Never reads records newer than that.

        ``at_version`` is not a convenience parameter — passing the live
        version instead of the decision's version is what would break replay
        (invariant E1), so it has no default.
        """
        visible = self._index.visible_at(at_version)
        ranked = self._failure.rank(visible, filters=filters)
        top = ranked[:top_k]

        return RetrievalBundle(
            decision_id=decision_id,
            state_version=at_version,
            query_type=query_type,
            query_filters=dict(filters or {}),
            retrieved_experience_ids=[r.experience_id for r, _ in top],
            retrieval_scores=[score for _, score in top],
            evidence_levels=[r.evidence_level.value for r, _ in top],
            n_candidates_considered=len(visible),
            token_cost=sum(_approx_tokens(r) for r, _ in top),
            retrieval_version=self.retrieval_version,
            consumed_by_policy=False,   # stages 0-1
            created_at=time.time(),
        )

    def records_for(self, bundle: RetrievalBundle) -> List[ExperienceRecord]:
        """Dereference a bundle back into records (for prompts / audit)."""
        out: List[ExperienceRecord] = []
        for experience_id in bundle.retrieved_experience_ids:
            record = self._index.get(experience_id)
            if record is not None:
                out.append(record)
        return out

    def explain(self, bundle: RetrievalBundle) -> List[Tuple[str, float, str]]:
        """(id, score, statement) triples — what a prompt would actually show."""
        records = {r.experience_id: r for r in self.records_for(bundle)}
        return [
            (eid, score, records[eid].statement if eid in records else "")
            for eid, score in zip(
                bundle.retrieved_experience_ids, bundle.retrieval_scores
            )
        ]


def _approx_tokens(record: ExperienceRecord) -> int:
    """Rough context cost of showing this record. ~4 chars per token.

    Approximate on purpose: this feeds the retrieval-cost term of a future
    reward, where the ordering between a two-line and a twenty-line record is
    what matters, not the exact count.
    """
    return max(1, len(record.statement) // 4)
