"""Crossover-mode planning module: fuse the best plan with dissimilar alternatives.

Provides ``CrossoverPlanModule``, which renders the system/user prompt pair for
asking the LLM to combine the focus plan with one or more dissimilar plans to
produce a new hybrid direction.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from famou.core.data import Context
from famou.modules.planning.base import PlanningModule
from famou.utils.program_summary import build_anchor_summary, normalize_plan_summary
from famou.prompts import prompt_registry


class CrossoverPlanModule(PlanningModule):
    """
    Generate a crossover plan that combines the focus plan with dissimilar plans.

    Reads from planner_inputs:
        anchor_program: representative program of the UCB-selected focus plan
        focus_plan_summary: the primary plan to build on
        crossover_plan_summaries: most dissimilar plans for cross-pollination

    Template: planning/plan_strategy_crossover_{system,user}.txt
    """

    def build_prompt(
        self, context: Context, planner_inputs: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Render the (system, user) prompt pair for crossover planning.

        Builds a JSON-serialized summary of the anchor program, the focus plan,
        and the list of dissimilar plans to cross-pollinate with, then
        substitutes them into the registered prompt templates.

        Args:
            context: Rollout context carrying the task description.
            planner_inputs: Dict prepared by the strategy. Required key:
                ``anchor_program``. Optional keys: ``focus_plan_summary``,
                ``crossover_plan_summaries`` (default empty).

        Returns:
            A (system_prompt, user_prompt) tuple ready to be sent to the LLM.
        """
        anchor_program = planner_inputs["anchor_program"]
        focus_plan = planner_inputs.get("focus_plan_summary")
        crossover_plans = planner_inputs.get("crossover_plan_summaries", [])

        anchor_summary = build_anchor_summary(anchor_program)
        focus_payload = (
            normalize_plan_summary(focus_plan)
            if isinstance(focus_plan, dict)
            else None
        )
        crossover_payload = [
            s for s in (normalize_plan_summary(p) for p in crossover_plans)
            if isinstance(s, dict)
        ]

        system = prompt_registry.get("planning/plan_strategy_crossover_system.txt")
        user = prompt_registry.get(
            "planning/plan_strategy_crossover_user.txt",
            task_description=context.task_description,
            anchor_program_summary_json=json.dumps(
                anchor_summary, ensure_ascii=False, indent=2
            ),
            focus_plan_summary_json=(
                json.dumps(focus_payload, ensure_ascii=False, indent=2)
                if isinstance(focus_payload, dict)
                else "(none)"
            ),
            crossover_plan_summaries_json=(
                json.dumps(crossover_payload, ensure_ascii=False, indent=2)
                if crossover_payload
                else "[]"
            ),
        )
        return system, user
