"""Select module for model-fusion rollouts in MLPipelineStrategy.

This module provides a SelectModule that does not perform any selection logic
itself; instead it relays a parent_id and inspiration_ids that have already
been chosen upstream by the strategy and placed into Context.metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.modules.select.base import SelectModule


class ModelFusionSelect(SelectModule):
    """Select the exact parent and inspirations pre-chosen by the strategy for model fusion.

    The MLPipelineStrategy computes model-fusion ensembles by picking the top-K
    programs upstream and placing the chosen ids into ``context.metadata["fusion_context"]``.
    This SelectModule simply reads those ids back out and forwards them through
    the standard SelectModule interface, so the rest of the rollout pipeline
    (Generate / Evaluate / Judge) can run unchanged.

    Expected ``context.metadata["fusion_context"]`` shape::

        {
            "parent_id": str,                 # required, non-empty
            "inspiration_ids": List[str],     # optional, defaults to []
            ...                                # extra fields are passed through
        }
    """

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that ``fusion_context`` is present and has a usable parent_id.

        Args:
            context: Rollout context; must contain ``metadata["fusion_context"]``.
            result: Current rollout result (forwarded to base validation).

        Raises:
            ValueError: If ``fusion_context`` is missing, not a dict, or its
                ``parent_id`` is blank.
        """
        super().validate_input(context, result)
        fusion_context = context.metadata.get("fusion_context")
        if not isinstance(fusion_context, dict):
            raise ValueError(f"{self.name}: Missing fusion_context in Context.metadata")
        if not str(fusion_context.get("parent_id") or "").strip():
            raise ValueError(f"{self.name}: fusion_context has no parent_id")

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """Return the pre-chosen parent_id from ``fusion_context``.

        The ``population`` argument is ignored — selection is purely metadata-driven.

        Args:
            context: Rollout context carrying ``fusion_context``.
            population: Unused; declared for interface compatibility.

        Returns:
            The parent program id stored in ``fusion_context["parent_id"]``.
        """
        del population
        fusion_context = context.metadata.get("fusion_context") or {}
        return str(fusion_context.get("parent_id"))

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """Return the pre-chosen inspiration ids from ``fusion_context``.

        Args:
            context: Rollout context carrying ``fusion_context``.
            population: Unused; declared for interface compatibility.
            parent_id: Unused; declared for interface compatibility.

        Returns:
            The inspiration program ids stored in ``fusion_context["inspiration_ids"]``,
            or an empty list if absent.
        """
        del population, parent_id
        fusion_context = context.metadata.get("fusion_context") or {}
        return list(fusion_context.get("inspiration_ids") or [])

    def select_state_metadata(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Pass the full ``fusion_context`` forward as selection state metadata.

        Downstream modules (and persisted rollout records) can read this metadata
        to understand which programs participated in the model-fusion ensemble.

        Args:
            context: Rollout context carrying ``fusion_context``.
            population: Unused; declared for interface compatibility.
            parent_id: Unused; declared for interface compatibility.

        Returns:
            A dict with ``fusion_context`` and ``selection_scope`` keys, or
            ``None`` if ``fusion_context`` is missing or malformed.
        """
        del population, parent_id
        fusion_context = context.metadata.get("fusion_context")
        if not isinstance(fusion_context, dict):
            return None
        return {
            "fusion_context": fusion_context,
            "selection_scope": "ml_pipeline_model_fusion",
        }
