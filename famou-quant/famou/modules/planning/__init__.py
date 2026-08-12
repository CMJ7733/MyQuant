"""
Planning module for Famou 2.0.

Provides block-level code evolution with explicit Planner deciding which code blocks to modify,
and text-plan-driven evolution with LLM-generated exploration/exploitation/crossover plans.

Key components:
- base_module: PlanningModule base class (Module + RequiresLLM, no Program output)
- explore_plan: ExplorePlanModule — propose new directions
- exploit_plan: ExploitPlanModule — deepen the best known direction
- crossover_plan: CrossoverPlanModule — fuse the best plan with dissimilar alternatives
- planners/: Block-level planner implementations (LLMPlannerV2, etc.)
- block_detection: Block detection utilities
- planner_select: PlannerBasedSelect module
- planner_feedback: PlannerFeedbackJudge module
"""

from famou.modules.planning.base import PlanningModule
from famou.modules.planning.explore_plan import ExplorePlanModule
from famou.modules.planning.exploit_plan import ExploitPlanModule
from famou.modules.planning.crossover_plan import CrossoverPlanModule
from famou.modules.planning.planners import (
    BasePlanner,
    LLMPlannerV2,
    StochasticPlanner,
    create_planner,
)
from famou.modules.planning.block_detection import (
    detect_blocks_from_code,
    parse_blocks_with_spans,
    parse_child_blocks,
    replace_blocks,
    uses_start_end_format,
)
from famou.modules.planning.planner_select import PlannerBasedSelect
from famou.modules.planning.planner_feedback import PlannerFeedbackJudge

__all__ = [
    # Text-plan modules
    "PlanningModule",
    "ExplorePlanModule",
    "ExploitPlanModule",
    "CrossoverPlanModule",
    # Block-level planners
    "BasePlanner",
    "LLMPlannerV2",
    "StochasticPlanner",
    "create_planner",
    # Block detection
    "detect_blocks_from_code",
    "parse_blocks_with_spans",
    "parse_child_blocks",
    "replace_blocks",
    "uses_start_end_format",
    # Modules
    "PlannerBasedSelect",
    "PlannerFeedbackJudge",
]
