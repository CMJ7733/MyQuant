"""Component-level summary judge for ML pipeline component-enhancement plans."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import RequiresLLM
from famou.infrastructure.llm.base import (
    get_llm_max_retries,
    get_llm_max_tokens,
    get_llm_temperature,
    get_llm_timeout,
)
from famou.modules.judge.base import JudgeModule
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.utils.program_summary import (
    extract_eval_wall_time as _extract_eval_wall_time,
    get_implementation_plan,
    get_key_features,
)
from famou.utils.trace_utils import append_named_trace, build_llm_trace


def _format_eval_wall_time(program: Optional[Program]) -> str:
    """Format evaluation wall time for prompt rendering."""
    eval_wall_time = _extract_eval_wall_time(program)
    if eval_wall_time is None:
        return "N/A"
    return f"{eval_wall_time:.6f}"


class ComponentJudge(JudgeModule, RequiresLLM):
    """Summarize a completed component-enhancement plan and persist reusable experience."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        component_name: Optional[str] = None,
        bonus_budget_on_improvement: int = 2,
        **kwargs: Any,
    ) -> None:
        """Store component identity and strategy-local bonus-budget policy."""
        del kwargs
        super().__init__(name)
        self.component_name = component_name
        self.bonus_budget_on_improvement = int(bonus_budget_on_improvement)

    def judge(self, context: Context, result: RolloutResult) -> None:
        """Run only on the closing rollout of one component-enhancement plan."""
        program = result.generated_program
        if program is None or result.selection is None:
            return

        plan_context = result.selection.extra.get("plan_context")
        if not isinstance(plan_context, dict):
            return
        if plan_context.get("planner_mode") != "component":
            return
        component_key = str(
            plan_context.get("component_key")
            or plan_context.get("planner_focus_plan_id")
            or ""
        ).strip()
        component_name = str(
            plan_context.get("component_name")
            or self.component_name
            or component_key
        ).strip()
        if not component_key:
            return
        if not self._should_summarize_plan(plan_context, program):
            return

        plan_payload = self._build_plan_payload(context, result, plan_context, program)
        prompt = prompt_registry.get(
            "evaluation/component_ablation_summary.txt",
            component_name=component_name,
            plan_text=plan_payload["plan_text"],
            seed_parent_summary=plan_payload["seed_parent_summary"],
            rollout_history_json=plan_payload["rollout_history_json"],
        )
        system_prompt = prompt_registry.get("evaluation/judge_system.txt")

        summary, trace = self._run_summary_llm(
            program=program,
            prompt=prompt,
            system_prompt=system_prompt,
            request_extra={
                "component_key": component_key,
                "component_name": component_name,
                "plan_id": plan_payload["plan_id"],
                "best_program_id": plan_payload["best_program_id"],
                "best_rollout_id": plan_payload["best_rollout_id"],
            },
        )

        program.meta["component_ablation_summary"] = summary
        program.meta["component_ablation_metadata"] = {
            "component_key": component_key,
            "component_name": component_name,
            "plan_id": plan_payload["plan_id"],
            "best_program_id": plan_payload["best_program_id"],
            "best_rollout_id": plan_payload["best_rollout_id"],
            "program_ids": plan_payload["program_ids"],
            "rollout_ids": plan_payload["rollout_ids"],
        }
        append_named_trace(program, "component_ablation", trace)

        previous = {}
        if context.state:
            previous = context.state.get_island(
                "ml_pipeline",
                "component_experience",
                component_key,
                default={},
            )
        if not isinstance(previous, dict):
            previous = {}

        history = previous.get("history", [])
        if not isinstance(history, list):
            history = []
        history = history + [
            {
                "plan_id": plan_payload["plan_id"],
                "component_key": component_key,
                "component_name": component_name,
                "best_program_id": plan_payload["best_program_id"],
                "best_rollout_id": plan_payload["best_rollout_id"],
                "summary": summary,
            }
        ]

        state_value = {
            "component_key": component_key,
            "component_name": component_name,
            "latest_summary": summary,
            "latest_plan_metadata": {
                "plan_id": plan_payload["plan_id"],
                "best_program_id": plan_payload["best_program_id"],
                "best_rollout_id": plan_payload["best_rollout_id"],
                "program_ids": plan_payload["program_ids"],
                "rollout_ids": plan_payload["rollout_ids"],
            },
            "history": history,
        }
        if result.state_updates:
            result.state_updates.set_island(
                "ml_pipeline",
                "component_experience",
                component_key,
                value=state_value,
            )

    def _should_summarize_plan(
        self,
        plan_context: Dict[str, Any],
        program: Program,
    ) -> bool:
        """Mirror the strategy budget update to detect whether the plan closes now."""
        remaining_budget = int(plan_context.get("remaining_budget") or 0)
        next_remaining_budget = max(0, remaining_budget - 1)

        island_best_score = plan_context.get("island_best_score_at_dispatch")
        child_score = program.combined_score
        improved_island_best = False
        if child_score is not None:
            if island_best_score is None:
                improved_island_best = True
            else:
                improved_island_best = child_score > island_best_score

        if improved_island_best:
            next_remaining_budget += self.bonus_budget_on_improvement
        return next_remaining_budget <= 0

    def _build_plan_payload(
        self,
        context: Context,
        result: RolloutResult,
        plan_context: Dict[str, Any],
        program: Program,
    ) -> Dict[str, Any]:
        """Build the semantic component-plan summary payload for the LLM."""
        plan_program_ids = list(plan_context.get("plan_program_ids") or [])
        plan_rollout_ids = list(plan_context.get("plan_rollout_ids") or [])
        if program.id not in plan_program_ids:
            plan_program_ids.append(program.id)
            plan_rollout_ids.append(result.rollout_id)

        ordered_programs: List[Program] = []
        ordered_rollout_ids: List[Optional[str]] = []
        seen_ids = set()
        for index, program_id in enumerate(plan_program_ids):
            candidate = (
                program if program_id == program.id else context.get_program_by_id(program_id)
            )
            if candidate is None or candidate.id in seen_ids:
                continue
            seen_ids.add(candidate.id)
            ordered_programs.append(candidate)
            ordered_rollout_ids.append(
                str(plan_rollout_ids[index]) if index < len(plan_rollout_ids) else None
            )

        seed_parent = context.get_program_by_id(str(plan_context.get("seed_parent_id") or ""))
        best_program = self._select_best_program(ordered_programs, seed_parent)
        best_rollout_id = None
        ordered_program_ids = [candidate.id for candidate in ordered_programs]
        if best_program is not None and best_program.id in ordered_program_ids:
            best_index = ordered_program_ids.index(best_program.id)
            best_rollout_id = ordered_rollout_ids[best_index]

        rollout_rows = [
            self._build_rollout_row(
                ordinal=index,
                candidate=plan_program,
                context=context,
                seed_parent=seed_parent,
            )
            for index, plan_program in enumerate(ordered_programs, start=1)
        ]

        return {
            "plan_id": str(plan_context.get("plan_id") or ""),
            "plan_text": str(plan_context.get("plan_text") or ""),
            "seed_parent_summary": self._build_seed_parent_summary(seed_parent),
            "rollout_history_json": rollout_rows,
            "best_program_id": best_program.id if best_program is not None else None,
            "best_rollout_id": best_rollout_id,
            "program_ids": ordered_program_ids,
            "rollout_ids": ordered_rollout_ids,
        }

    def _build_seed_parent_summary(self, program: Optional[Program]) -> str:
        """Render a compact semantic summary for the seed parent."""
        if program is None:
            return "N/A"
        return (
            f"Implementation plan: {get_implementation_plan(program) or 'N/A'}\n"
            f"Key features: {get_key_features(program) or 'N/A'}\n"
            f"Combined score: {program.combined_score if program.combined_score is not None else 'N/A'}\n"
            f"Metrics: {dict(program.metrics or {})}\n"
            f"Validity: {program.validity if program.validity is not None else 'N/A'}\n"
            f"Eval wall time: {_format_eval_wall_time(program)}"
        )

    def _build_rollout_row(
        self,
        *,
        ordinal: int,
        candidate: Program,
        context: Context,
        seed_parent: Optional[Program],
    ) -> Dict[str, Any]:
        """Build one rollout summary row without exposing opaque ids to the LLM."""
        parent = context.get_program_by_id(candidate.parent_id) if candidate.parent_id else None
        if parent is None:
            parent = seed_parent

        change_summary = ""
        if isinstance(candidate.llm_feedback, dict):
            change_summary = str(candidate.llm_feedback.get("change_summary") or "").strip()
        if not change_summary:
            change_summary = get_implementation_plan(candidate)

        return {
            "ordinal": ordinal,
            "implementation_plan": get_implementation_plan(candidate),
            "combined_score": candidate.combined_score,
            "metrics": dict(candidate.metrics or {}),
            "validity": candidate.validity,
            "error_info": candidate.error_info,
            "eval_wall_time": _format_eval_wall_time(candidate),
            "parent_eval_wall_time": _format_eval_wall_time(parent),
            "parent_child_diff_summary": change_summary,
        }

    def _select_best_program(
        self,
        plan_programs: List[Program],
        seed_parent: Optional[Program],
    ) -> Optional[Program]:
        """Select the best program in the plan, falling back to the seed parent."""
        candidates = list(plan_programs)
        if seed_parent is not None:
            candidates.append(seed_parent)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda candidate: (
                candidate.validity if candidate.validity is not None else float("-inf"),
                candidate.combined_score
                if candidate.combined_score is not None
                else float("-inf"),
            ),
        )

    def _run_summary_llm(
        self,
        *,
        program: Program,
        prompt: str,
        system_prompt: str,
        request_extra: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Run the LLM summary with parse retries and return both summary and trace."""
        temperature = get_llm_temperature(self.llm_client)
        max_tokens = get_llm_max_tokens(self.llm_client)
        timeout = get_llm_timeout(self.llm_client)
        max_attempts = get_llm_max_retries(self.llm_client)

        last_error: Optional[Exception] = None
        last_response = None
        last_prompt = prompt

        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if attempt > 1:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "RETRY INSTRUCTION:\n"
                    "Return exactly one JSON object that matches the requested schema."
                )
            last_prompt = attempt_prompt
            response = None
            try:
                response = self.llm_client.generate(
                    prompt=attempt_prompt,
                    system=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = extract_json(response.text)
                if not isinstance(parsed, dict):
                    raise ValueError("ComponentJudge expected a JSON object")
                trace = build_llm_trace(
                    module_name=self.name,
                    system=system_prompt,
                    prompt=attempt_prompt,
                    response=response,
                    request_extra={
                        **request_extra,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                    parsed={"component_ablation_summary": parsed},
                )
                return parsed, trace
            except Exception as exc:
                last_error = exc
                last_response = response

        error_trace = build_llm_trace(
            module_name=self.name,
            system=system_prompt,
            prompt=last_prompt,
            response=last_response,
            request_extra={
                **request_extra,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "attempt": max_attempts,
                "max_attempts": max_attempts,
            },
            parsed={"component_ablation_summary": None},
            error=last_error,
        )
        append_named_trace(program, "component_ablation", error_trace)
        if last_error is not None:
            raise last_error
        raise ValueError("ComponentJudge failed without a captured exception")
