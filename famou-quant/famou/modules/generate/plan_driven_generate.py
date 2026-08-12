"""Generation module that follows a strategy-level text plan."""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule, WRITE_MODE_DIFF
from famou.prompts import prompt_registry
from famou.utils.code_parser import has_single_evolve_block

from famou.utils.program_summary import (
    extract_eval_wall_time as _extract_eval_wall_time,
    get_implementation_plan,
)
from famou.utils.trace_utils import append_named_trace


def _get_visible_programs(context: Context) -> List[Program]:
    if context.island_accessor:
        return list(context.island_accessor)
    return context.get_all_programs()


def _pick_best_program(programs: List[Program]) -> Program:
    return max(
        programs,
        key=lambda p: (
            p.combined_score if p.combined_score is not None else float("-inf"),
            p.validity if p.validity is not None else float("-inf"),
            p.iteration,
            p.generation,
        ),
    )


def _resolve_plan_parent(
    context: Context,
    requested_parent_id: str,
    plan_context: Dict[str, Any],
):
    """Resolve the best visible parent for the active plan, with fallback."""
    visible = _get_visible_programs(context)
    visible_by_id = {p.id: p for p in visible}

    if requested_parent_id and requested_parent_id in visible_by_id:
        program = visible_by_id[requested_parent_id]
        return program, {"requested_parent_id": requested_parent_id,
                         "resolved_parent_id": program.id,
                         "resolution": "requested_parent",
                         "visible_program_ids": [p.id for p in visible]}

    candidate_ids = []
    for cid in [plan_context.get("best_program_id"), *(plan_context.get("plan_program_ids") or [])]:
        if isinstance(cid, str) and cid.strip():
            candidate_ids.append(cid.strip())
    seen = set()
    same_plan = []
    for cid in candidate_ids:
        p = visible_by_id.get(cid)
        if p and p.id not in seen:
            same_plan.append(p)
            seen.add(p.id)
    plan_id = str(plan_context.get("plan_id") or "").strip()
    if plan_id:
        for p in visible:
            if p.id not in seen and str((p.meta or {}).get("plan_id") or "").strip() == plan_id:
                same_plan.append(p)
                seen.add(p.id)

    candidates = same_plan or visible
    if not candidates:
        return None, {"requested_parent_id": requested_parent_id, "resolved_parent_id": None,
                      "resolution": "no_visible_programs", "visible_program_ids": []}
    program = _pick_best_program(candidates)
    resolution = "same_plan_survivor" if same_plan else "island_best_fallback"
    return program, {"requested_parent_id": requested_parent_id,
                     "resolved_parent_id": program.id,
                     "resolution": resolution,
                     "visible_program_ids": [p.id for p in visible]}


class PlanDrivenGenerate(GenerateModule):
    """Generate code by following the currently active plan."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        include_inspiration_code: bool = False,
        plan_history_limit: int = 8,
        **kwargs,
    ) -> None:
        """Configure prompt assembly for plan-guided code generation."""
        super().__init__(name, **kwargs)
        self.include_inspiration_code = include_inspiration_code
        self.plan_history_limit = max(0, int(plan_history_limit))

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """Build the generation prompt from the active plan, best parent, and plan history."""
        plan_context = self._get_plan_context(selection)
        current_best_program = self._resolve_parent_or_raise(context, selection, plan_context)
        initial_anchor_parent_id = str(
            plan_context.get("seed_parent_id")
            or plan_context.get("anchor_program_id")
            or ""
        ).strip()
        initial_anchor_parent = (
            self._maybe_get_program(context, initial_anchor_parent_id)
            if initial_anchor_parent_id
            else None
        )
        if initial_anchor_parent is None:
            initial_anchor_parent = current_best_program
        current_best_summary = self._build_semantic_program_summary(
            current_best_program,
            include_code=True,
        )
        initial_anchor_summary = self._build_semantic_program_summary(
            initial_anchor_parent,
            include_code=True,
        )

        plan_history = self._build_plan_history(
            context,
            plan_context,
            excluded_program_ids={
                initial_anchor_parent.id,
                current_best_program.id,
            },
        )
        current_best_payload = self._format_current_best_payload(
            current_best_program=current_best_program,
            initial_anchor_parent=initial_anchor_parent,
            current_best_summary=current_best_summary,
        )

        return prompt_registry.get(
            "generation/plan_driven.txt",
            language=context.language,
            task_description=context.task_description,
            plan_context=plan_context,
            initial_anchor_parent_program_json=json.dumps(
                initial_anchor_summary, ensure_ascii=False, indent=2
            ),
            current_best_program_json=current_best_payload or "",
            evolution_history_json=(
                "\n".join(
                    json.dumps(item, ensure_ascii=False, indent=2)
                    for item in plan_history
                )
                if plan_history
                else "(no prior rollout history for the current plan yet)"
            ),
            diff_write_mode=self.write_mode == WRITE_MODE_DIFF,
            has_evolve_block=has_single_evolve_block(initial_anchor_parent.code, context.language),
        )

    def execute(
        self, context: Context, result: RolloutResult, **kwargs
    ) -> RolloutResult:
        """Run generation and mirror parsed implementation-plan data into the trace."""
        if result.selection is not None:
            plan_context = self._get_plan_context(result.selection)
            self._resolve_parent_or_raise(context, result.selection, plan_context)
        result = super().execute(context, result, **kwargs)
        program = result.generated_program
        if program is None:
            return result

        generate_trace = program.meta.get("traces", {}).get("generate")
        if isinstance(generate_trace, dict):
            parsed = generate_trace.get("parsed")
            if not isinstance(parsed, dict):
                parsed = {}
                generate_trace["parsed"] = parsed
            parsed["implementation_plan"] = program.meta.get("implementation_plan", "")
        return result

    def post_process(
        self,
        response,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """Enrich the generated program with plan metadata and planner trace information."""
        program = super().post_process(response, context, selection, parent)
        plan_context = self._get_plan_context(selection)
        implementation_plan = self._extract_implementation_plan(response)

        program.meta["plan"] = plan_context["plan_text"]
        program.meta["plan_id"] = plan_context["plan_id"]
        program.meta["plan_text"] = plan_context["plan_text"]
        program.meta["plan_round"] = plan_context["plan_round"]
        program.meta["plan_seed_parent_id"] = plan_context["seed_parent_id"]
        program.meta["plan_current_parent_id_at_dispatch"] = plan_context.get(
            "current_parent_id"
        )
        program.meta["plan_best_program_id_at_dispatch"] = plan_context["best_program_id"]
        program.meta["plan_created_iteration"] = plan_context["created_iteration"]
        program.meta["plan_inspiration_ids"] = list(selection.inspiration_ids or [])
        program.meta["generation_type"] = "plan_driven"
        program.meta["implementation_plan"] = implementation_plan
        if isinstance(parent.meta.get("source_plan_ids"), list):
            program.meta["source_plan_ids"] = list(parent.meta.get("source_plan_ids") or [])
        if isinstance(parent.meta.get("fused_explore_plan_ids"), list):
            program.meta["fused_explore_plan_ids"] = list(
                parent.meta.get("fused_explore_plan_ids") or []
            )
        # Persist the full dispatch-time plan view for traceability and resume debugging.
        program.meta["plan_snapshot"] = copy.deepcopy(plan_context)

        planner_trace = plan_context.get("planner_trace")
        if isinstance(planner_trace, dict):
            append_named_trace(program, "plan_planner", planner_trace)

        return program

    def _build_plan_history(
        self,
        context: Context,
        plan_context: Dict[str, Any],
        *,
        excluded_program_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build structured summaries for prior children produced under the same plan."""
        history_entries: List[Dict[str, Any]] = []
        plan_program_ids = list(plan_context.get("plan_program_ids", []))
        excluded_program_ids = {
            str(program_id).strip()
            for program_id in (excluded_program_ids or set())
            if isinstance(program_id, str) and str(program_id).strip()
        }
        programs: List[Program] = []
        for program_id in plan_program_ids:
            normalized_program_id = str(program_id).strip()
            if normalized_program_id in excluded_program_ids:
                continue
            program = self._maybe_get_program(context, program_id)
            if program is not None:
                programs.append(program)
        programs.sort(key=lambda p: (p.iteration, p.generation, p.id))
        if self.plan_history_limit > 0:
            start_ordinal = max(1, len(programs) - self.plan_history_limit + 1)
            programs = programs[-self.plan_history_limit :]
        else:
            start_ordinal = 1

        for ordinal, program in enumerate(programs, start=start_ordinal):
            history_entries.append(
                self._build_plan_history_entry(
                    program=program,
                    ordinal=ordinal,
                    include_code=self.include_inspiration_code,
                )
            )
        return history_entries

    def _format_current_best_payload(
        self,
        *,
        current_best_program: Program,
        initial_anchor_parent: Program,
        current_best_summary: Dict[str, Any],
    ) -> Optional[str]:
        """Avoid duplicating the same program summary across prompt sections."""
        if current_best_program.id == initial_anchor_parent.id:
            return None
        return json.dumps(current_best_summary, ensure_ascii=False, indent=2)

    def _build_semantic_program_summary(
        self,
        program: Program,
        *,
        include_code: bool,
    ) -> Dict[str, Any]:
        """Build the LLM-facing semantic summary without opaque ids or generation fields."""
        summary: Dict[str, Any] = {
            "implementation_plan": get_implementation_plan(program),
            "combined_score": program.combined_score,
            "metrics": dict(program.metrics or {}),
            "validity": program.validity,
            "eval_time": _extract_eval_wall_time(program),
            "error_info": program.error_info,
        }
        if include_code:
            summary["code"] = program.code
        return summary

    def _build_plan_history_entry(
        self,
        *,
        program: Program,
        ordinal: int,
        include_code: bool,
    ) -> Dict[str, Any]:
        """Build one semantic history row for the active plan."""
        change_summary = ""
        if isinstance(program.llm_feedback, dict):
            change_summary = str(program.llm_feedback.get("change_summary") or "").strip()
        if not change_summary:
            change_summary = get_implementation_plan(program)

        summary = self._build_semantic_program_summary(program, include_code=include_code)
        return {
            "ordinal": ordinal,
            "implementation_plan": summary.get("implementation_plan"),
            "combined_score": summary.get("combined_score"),
            "metrics": summary.get("metrics"),
            "validity": summary.get("validity"),
            "eval_time": summary.get("eval_time"),
            "error_info": summary.get("error_info"),
            "change_summary": change_summary,
            **({"code": summary.get("code")} if include_code else {}),
        }

    def _extract_implementation_plan(self, response: LLMResponse) -> str:
        """Extract the XML implementation plan emitted by the generation model."""
        match = re.search(
            r"<implementation_plan>\s*(.*?)\s*</implementation_plan>",
            response.text,
            re.DOTALL | re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    def _get_program(self, context: Context, program_id: str) -> Program:
        """Resolve a program id from island-local storage or the global archive."""
        program = self._maybe_get_program(context, program_id)
        if program is None:
            raise ValueError(f"{self.name}: Program {program_id} not found")
        return program

    def _maybe_get_program(self, context: Context, program_id: str) -> Optional[Program]:
        """Resolve a program id from island-local storage or the current population snapshot."""
        program = None
        if context.island_accessor:
            program = context.island_accessor.get_by_id(program_id)
        if program is None:
            program = context.get_program_by_id(program_id)
        return program

    def _resolve_parent_or_raise(
        self,
        context: Context,
        selection: SelectionData,
        plan_context: Dict[str, Any],
    ) -> Program:
        """Resolve the actual parent under turnover and persist the decision on selection metadata."""
        parent, resolution = _resolve_plan_parent(context, selection.parent_id, plan_context)
        if parent is None:
            raise ValueError(
                f"{self.name}: No visible parent is available for requested parent "
                f"{selection.parent_id!r}"
            )

        extra = dict(selection.extra or {})
        extra["parent_resolution"] = resolution
        if selection.parent_id != parent.id:
            extra["requested_parent_id"] = selection.parent_id
            selection.parent_id = parent.id
        selection.extra = extra
        return parent

    def _get_plan_context(self, selection: SelectionData) -> Dict[str, Any]:
        """Read and validate plan metadata attached during plan-scoped selection."""
        extra = selection.extra if isinstance(selection.extra, dict) else {}
        raw_plan_context = extra.get("plan_context")
        plan_context = raw_plan_context if isinstance(raw_plan_context, dict) else {}
        return {
            "plan_id": plan_context.get("plan_id"),
            "planner_mode": plan_context.get("planner_mode"),
            "plan_text": plan_context.get("plan_text"),
            "plan_round": plan_context.get("plan_round"),
            "created_iteration": plan_context.get("created_iteration"),
            "seed_parent_id": plan_context.get("seed_parent_id"),
            "anchor_program_id": plan_context.get("anchor_program_id"),
            "anchor_score": plan_context.get("anchor_score"),
            "anchor_metrics": dict(plan_context.get("anchor_metrics") or {}),
            "anchor_validity": plan_context.get("anchor_validity"),
            "anchor_implementation_plan": plan_context.get("anchor_implementation_plan"),
            "anchor_key_features": plan_context.get("anchor_key_features"),
            "anchor_improvement_directions": plan_context.get("anchor_improvement_directions"),
            "current_parent_id": plan_context.get("current_parent_id"),
            "best_program_id": plan_context.get("best_program_id"),
            "best_score": plan_context.get("best_score"),
            "best_metrics": dict(plan_context.get("best_metrics") or {}),
            "best_validity": plan_context.get("best_validity"),
            "best_implementation_plan": plan_context.get("best_implementation_plan"),
            "best_key_features": plan_context.get("best_key_features"),
            "best_improvement_directions": plan_context.get("best_improvement_directions"),
            "island_best_score_at_dispatch": plan_context.get("island_best_score_at_dispatch"),
            "plan_program_ids": list(plan_context.get("plan_program_ids") or []),
            "remaining_budget": plan_context.get("remaining_budget"),
            "planner_trace": plan_context.get("planner_trace"),
            "planner_focus_plan_id": plan_context.get("planner_focus_plan_id"),
            "planner_history_plan_ids": list(plan_context.get("planner_history_plan_ids") or []),
            "planner_crossover_plan_ids": list(
                plan_context.get("planner_crossover_plan_ids") or []
            ),
            "component_key": plan_context.get("component_key"),
            "component_name": plan_context.get("component_name"),
            "component_goal": plan_context.get("component_goal"),
            "component_constraints": plan_context.get("component_constraints"),
            "component_suggested_techniques": plan_context.get(
                "component_suggested_techniques"
            ),
            "component_experience": dict(plan_context.get("component_experience") or {}),
        }
