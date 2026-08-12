"""Explore-mode planning module: propose a new direction from plan history."""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

from famou.core.data import Context
from famou.modules.planning.base import PlanningModule
from famou.utils.program_summary import build_anchor_summary, normalize_plan_summary
from famou.prompts import prompt_registry


class ExplorePlanModule(PlanningModule):
    """
    Generate an exploration plan that proposes a meaningfully new direction.

    Reads from planner_inputs:
        anchor_program: best visible program (used as current baseline)
        history_plan_summaries: recent plan outcomes for context

    Template: planning/plan_strategy_explore_{system,user}.txt
    """

    def build_prompt(
        self, context: Context, planner_inputs: Dict[str, Any]
    ) -> Tuple[str, str]:
        anchor_program = planner_inputs["anchor_program"]
        prior_plans = planner_inputs.get("history_plan_summaries", [])
        track_hint = planner_inputs.get("track_hint")

        anchor_summary = build_anchor_summary(anchor_program)
        reference_payload = [
            s for s in (normalize_plan_summary(p) for p in prior_plans)
            if isinstance(s, dict)
        ]

        system = prompt_registry.get("planning/plan_strategy_explore_system.txt")
        user = prompt_registry.get(
            "planning/plan_strategy_explore_user.txt",
            task_description=context.task_description,
            anchor_program_summary_json=json.dumps(
                anchor_summary, ensure_ascii=False, indent=2
            ),
            reference_plan_summaries_json=(
                json.dumps(reference_payload, ensure_ascii=False, indent=2)
                if reference_payload
                else "[]"
            ),
        )
        if track_hint:
            user = (
                f"<ExplorationHint>\n"
                f"Exploration track: {track_hint}\n"
                "This is a loose inspiration label — propose a direction semantically different from prior plans.\n"
                f"</ExplorationHint>\n\n{user}"
            )
        return system, user
