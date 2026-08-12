"""
Stochastic Planner - Random target selection.

Used as:
- Fallback when LLM planner fails
- Baseline for comparison
- Simple exploration strategy
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from famou.modules.planning.planners.base import PlannerAction


class StochasticPlanner:
    """
    Stochastic planner that samples target and operation randomly.

    Features:
    - Uniform random target selection
    - Optional operation randomization
    - Deterministic seed support for reproducibility

    Example:
        >>> planner = StochasticPlanner(arms=["A", "B", "C"], seed=42)
        >>> action = planner.propose_action()
        >>> # action = {"op": "modify", "target": "B"}
    """

    DEFAULT_ARMS = ["A", "B", "C"]

    def __init__(
        self,
        arms: Optional[List[str]] = None,
        seed: Optional[int] = None,
        randomize_op: bool = False,
    ):
        """
        Initialize stochastic planner.

        Args:
            arms: Available block labels (default: ["A", "B", "C"])
            seed: Random seed for reproducibility (optional)
            randomize_op: If True, also randomize operation (default: False, always "modify")
        """
        self.arms = list(arms) if arms else list(self.DEFAULT_ARMS)
        self._rng = random.Random(seed)
        self.randomize_op = randomize_op

    def propose_action(self, parent_code: str = "", **kwargs) -> Dict[str, Any]:
        """
        Propose a random action.

        Args:
            parent_code: Ignored (stochastic doesn't use code context)
            **kwargs: May contain 'arms' to override default arms

        Returns:
            Action dict: {"op": str, "target": str}
        """
        # Allow runtime arms override
        arms = kwargs.get("arms") or self.arms
        if arms:
            self.arms = list(arms)

        if not self.arms:
            return {"op": "modify", "target": ""}

        target = self._rng.choice(self.arms)

        if self.randomize_op:
            op = self._rng.choice(list(PlannerAction.ALLOWED_OPS))
        else:
            op = "modify"

        return {"op": op, "target": target}

    def update(self, batch: List[Dict[str, Any]]) -> None:
        """
        No-op for stochastic planner (stateless).

        Args:
            batch: Ignored
        """
        pass
