"""
PlannerFeedbackJudge - Judge module that provides feedback to Planner.

Flow:
1. Collect evaluation results from generated program
2. Calculate reward signal
3. Update Planner with feedback
4. Save planner state to StateStore

This module should be placed AFTER EvaluateModule in the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module
from famou.modules.planning.planners import BasePlanner

if TYPE_CHECKING:
    pass

logger = logging.getLogger("famou")


class PlannerFeedbackJudge(Module):
    """
    Judge module that provides feedback to Planner.

    Pipeline Role:
    - Reads: result.generated_program (with evaluation results)
    - Reads: result.selection.extra["planner_action"]
    - Writes: Planner state via update() and StateStore

    Should be placed AFTER EvaluateModule in the pipeline so that
    evaluation metrics are available.

    Key Features:
    - Computes reward signal from evaluation metrics
    - Updates Planner with action-outcome feedback
    - Persists planner state to StateStore
    - Supports custom reward functions

    Configuration:
        reward_metric: Metric to use as reward ("combined_score", "validity", etc.)
        use_delta_reward: If True, reward = child_score - parent_score
        planner: Reference to Planner instance (set by Strategy)

    Example:
        >>> judge = PlannerFeedbackJudge(
        ...     reward_metric="combined_score",
        ...     use_delta_reward=True,
        ... )
        >>> result = judge(context, result)
        >>> # Planner is updated with feedback
    """

    def __init__(
        self,
        reward_metric: str = "combined_score",
        use_delta_reward: bool = False,
        planner: Optional[BasePlanner] = None,
        name: Optional[str] = None,
        **config,
    ):
        """
        Initialize PlannerFeedbackJudge.

        Args:
            reward_metric: Which metric to use as reward signal
            use_delta_reward: Use improvement over parent as reward
            planner: Planner instance (optional, can be set later)
            name: Module name
            **config: Additional configuration
        """
        super().__init__(name=name, **config)

        self.reward_metric = reward_metric
        self.use_delta_reward = use_delta_reward
        self._planner = planner

    def set_planner(self, planner: BasePlanner) -> None:
        """
        Set planner reference.

        Called by Strategy when creating the pipeline.

        Args:
            planner: Planner instance
        """
        self._planner = planner

    # -------------------------------------------------------------------------
    # Reward Computation
    # -------------------------------------------------------------------------

    def _compute_reward(
        self,
        context: Context,
        result: RolloutResult,
    ) -> float:
        """
        Compute reward signal from evaluation results.

        Args:
            context: Context
            result: RolloutResult with evaluated program

        Returns:
            Reward signal (float)
        """
        program = result.generated_program
        if not program:
            return 0.0

        # Get metric value
        if self.reward_metric == "combined_score":
            value = program.combined_score or 0.0
        elif self.reward_metric == "validity":
            value = program.validity or 0.0
        elif self.reward_metric in program.metrics:
            value = program.metrics.get(self.reward_metric, 0.0)
        else:
            value = program.combined_score or 0.0

        # Optionally compute delta (improvement over parent)
        if self.use_delta_reward and result.selection:
            parent = context.get_program_by_id(result.selection.parent_id)
            if parent:
                if self.reward_metric == "combined_score":
                    parent_value = parent.combined_score or 0.0
                elif self.reward_metric == "validity":
                    parent_value = parent.validity or 0.0
                elif self.reward_metric in parent.metrics:
                    parent_value = parent.metrics.get(self.reward_metric, 0.0)
                else:
                    parent_value = parent.combined_score or 0.0

                value = value - parent_value

        return float(value)

    def _build_feedback(
        self,
        context: Context,
        result: RolloutResult,
    ) -> Dict[str, Any]:
        """
        Build feedback item for Planner.update().

        Args:
            context: Context
            result: RolloutResult

        Returns:
            Feedback dict for Planner
        """
        program = result.generated_program
        selection = result.selection or {}
        action = selection.extra.get("planner_action", {}) if selection else {}

        reward = self._compute_reward(context, result)

        # Get parent info for enriched history tracking
        parent_id = selection.parent_id if selection else None
        parent_score = None
        if parent_id and context:
            parent = context.get_program_by_id(parent_id)
            if parent:
                parent_score = parent.combined_score

        feedback = {
            "pid": result.rollout_id,
            "action": action,
            "parent_id": parent_id,
            "parent_score": parent_score,
            "outcome": {
                "reward": reward,
                "combined_score": program.combined_score if program else None,
                "validity": program.validity if program else None,
                "crash": program.is_buggy if program else True,
            },
        }

        return feedback

    # -------------------------------------------------------------------------
    # Planner State Management
    # -------------------------------------------------------------------------

    def _get_planner(self, context: Context) -> Optional[BasePlanner]:
        """
        Get Planner instance.

        Tries:
        1. Instance variable (set by Strategy)
        2. Select module reference (set by Strategy via _select_module)
        3. StateStore (for resumed experiments)

        Args:
            context: Context with state store

        Returns:
            Planner instance or None
        """
        if self._planner is not None:
            return self._planner

        # Try to get planner from select module (wired by Strategy)
        select_module = getattr(self, "_select_module", None)
        if select_module is not None:
            planner = getattr(select_module, "_planner", None)
            if planner is not None:
                self._planner = planner  # Cache for future use
                if self.logger:
                    self.log_info(f"Got planner from select module")
                return planner

        # Fallback - normally Strategy sets the planner
        if self.logger:
            self.log_warning(
                f"{self.name}: No planner set. Feedback will not be recorded."
            )
        return None

    def _save_planner_state(self, context: Context, result: RolloutResult) -> None:
        """
        Save planner state via StateUpdateCollector.

        Args:
            context: Context with state accessor
            result: RolloutResult with state_updates collector
        """
        if self._planner is None:
            return

        if hasattr(self._planner, "get_state"):
            state = self._planner.get_state()
            if result.state_updates:
                result.state_updates.set_island("planner_state", value=state)
                logger.debug(f"Saved planner state for island {context.island_id}")

    # -------------------------------------------------------------------------
    # Module Protocol
    # -------------------------------------------------------------------------

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that evaluation results are available."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No generated program. "
                "Make sure EvaluateModule runs before PlannerFeedbackJudge."
            )
        # combined_score may be None for failed evaluations
        # We still want to provide feedback in that case

    def execute(
        self, context: Context, result: RolloutResult, **kwargs
    ) -> RolloutResult:
        """
        Collect feedback and update Planner.

        Flow:
        1. Compute reward from evaluation results
        2. Build feedback item
        3. Update Planner
        4. Save planner state

        Args:
            context: Context
            result: RolloutResult with evaluated program

        Returns:
            RolloutResult (unchanged, side effects only)
        """
        planner = self._get_planner(context)

        if planner is None:
            self.log_warning("No planner available, skipping feedback")
            return result

        # Build feedback
        feedback = self._build_feedback(context, result)

        if self.logger:
            self.log_info(
                f"Updating planner: "
                f"action={feedback['action']}, "
                f"reward={feedback['outcome']['reward']:.4f}, "
                f"parent_id={feedback.get('parent_id')}, "
                f"parent_score={feedback.get('parent_score')}, "
                f"child_score={feedback['outcome'].get('combined_score')}, "
                f"validity={feedback['outcome'].get('validity')}, "
                f"crash={feedback['outcome'].get('crash')}"
            )

        # Update planner
        planner.update([feedback])

        # Save state
        self._save_planner_state(context, result)

        # Store feedback in program meta for debugging
        if result.generated_program:
            result.generated_program.meta["planner_feedback"] = feedback

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """No output validation needed (side effects only)."""
        pass
