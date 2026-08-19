"""ExperienceConsolidator — the write side, driven by the barrier.

Called from inside ``BarrierCommit``'s write window so indexed experience and
archive state always share one version number (invariant E2). It is the only
component that turns committed outcomes into retrievable experience, which is
the experience-layer analogue of "the barrier is the only archive writer".

Stage 1 registers one memory (``FailureMemory``). Later stages add semantic
and certified memories here rather than touching the barrier again.
"""

from __future__ import annotations

from typing import Any, List, Optional

from famou.reliability.experience.failure import FailureMemory, ObservedOutcome
from famou.reliability.experience.index import ExperienceIndex


class ExperienceConsolidator:
    """Folds one committed batch into the experience index."""

    def __init__(
        self,
        index: ExperienceIndex,
        *,
        failure_memory: Optional[FailureMemory] = None,
        logger: Optional[Any] = None,
    ):
        self._index = index
        self._failure = failure_memory or FailureMemory()
        self.logger = logger

    def consolidate(
        self,
        outcomes: List[ObservedOutcome],
        *,
        valid_from_state_version: int,
        transition_id: str,
        decision_id: str,
        data_protocol_version: str = "",
        policy_version: str = "",
    ) -> List[str]:
        """Index everything this batch taught us. Returns experience ids touched.

        ``valid_from_state_version`` is the version the barrier is *producing*,
        not the one it decided at: the batch's results become retrievable only
        once they are committed, exactly like archive contents.

        Never raises. Losing an experience record is a degraded memory; letting
        it abort ``BarrierCommit.commit`` would lose the run's actual results,
        which are the thing that cost GPU hours.
        """
        touched: List[str] = []
        for outcome in outcomes:
            try:
                touched.extend(
                    self._failure.observe(
                        self._index,
                        outcome,
                        valid_from_state_version=valid_from_state_version,
                        transition_id=transition_id,
                        decision_id=decision_id,
                        data_protocol_version=data_protocol_version,
                        policy_version=policy_version,
                    )
                )
            except Exception as e:  # pragma: no cover - defensive
                if self.logger:
                    self.logger.warning(
                        f"[Experience] consolidation failed for "
                        f"{outcome.candidate_id}: {e}"
                    )
        return touched
