"""
Planner implementations for Planning-based evolution.

Planners decide which code block to modify and how (modify/add/remove/adjust_params).

Available planners:
- LLMPlannerV2: LLM-based planner with history context
- StochasticPlanner: Random target selection (fallback)
"""

from famou.modules.planning.planners.base import BasePlanner
from famou.modules.planning.planners.llm_planner_v2 import LLMPlannerV2
from famou.modules.planning.planners.stochastic import StochasticPlanner


def create_planner(planner_type: str, **kwargs) -> BasePlanner:
    """
    Factory function to create a planner by type.

    Args:
        planner_type: Planner type name ("llm_v2", "stochastic")
        **kwargs: Planner-specific configuration

    Returns:
        Planner instance

    Raises:
        ValueError: If planner_type is unknown

    Example:
        >>> planner = create_planner("llm_v2", llm_client=client, arms=["A", "B", "C"])
        >>> planner = create_planner("stochastic", arms=["A", "B"], seed=42)
    """
    planner_map = {
        "llm_v2": LLMPlannerV2,
        "stochastic": StochasticPlanner,
    }

    if planner_type not in planner_map:
        available = ", ".join(planner_map.keys())
        raise ValueError(
            f"Unknown planner type: {planner_type}. Available: {available}"
        )

    return planner_map[planner_type](**kwargs)


__all__ = [
    "BasePlanner",
    "LLMPlannerV2",
    "StochasticPlanner",
    "create_planner",
]
