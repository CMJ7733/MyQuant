"""LLM-based judging module for Famou 2.0."""

from typing import Any, Dict, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.utils.trace_utils import append_judge_trace, attach_debug_trace, build_llm_trace
from famou.modules.judge.base import JudgeModule
from famou.core.protocol import RequiresLLM
from famou.infrastructure.llm import LLMClient
from famou.infrastructure.llm.base import (
    get_llm_max_tokens,
    get_llm_temperature,
    get_llm_timeout,
)
from famou.utils.program_summary import extract_eval_wall_time as _extract_eval_wall_time


def _format_eval_wall_time(program: Optional[Program]) -> str:
    """Format evaluation wall time for prompt rendering."""
    eval_wall_time = _extract_eval_wall_time(program)
    if eval_wall_time is None:
        return "N/A"
    return f"{eval_wall_time:.6f}"


class LLMJudge(JudgeModule, RequiresLLM):
    """
    LLM-based code evaluator providing structured feedback.

    Uses LLM to generate structured evaluation feedback comparing the current
    program with its parent. Returns JSON with:
    - change_summary: Summary of changes from parent to current
    - error_eval: Error analysis and fix suggestions
    - key_features: Key features of the implementation
    - improvements: Potential improvements

    Args:
        name: Module name (default: class name)
    """
    llm_client: LLMClient  # Injected by RolloutEngine

    def judge(self, context: Context, result: RolloutResult) -> Dict[str, Any]:
        """Generate structured feedback using LLM."""
        program = result.generated_program
        parent = None
        if program.parent_id: 
            parent = context.get_program_by_id(program.parent_id)

        system_prompt = prompt_registry.get("evaluation/judge_system.txt")
        prompt = ""
        response = None

        try:
            # Build evaluation prompt
            prompt = self._build_feedback_prompt(program, parent, context)

            # Call LLM
            self.log_info(f"Generating evaluation for {program.id}")

            # Only pass overrides if explicitly configured (uses LLMClient defaults otherwise)
            llm_kwargs = {}
            temperature = get_llm_temperature(self.llm_client)
            max_tokens = get_llm_max_tokens(self.llm_client)
            timeout = get_llm_timeout(self.llm_client)

            response = self.llm_client.generate(
                prompt=prompt,
                system=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **llm_kwargs,
            )

            # Extract JSON feedback
            feedback = extract_json(response.text)
            if isinstance(feedback, dict):
                raw_improvement = (
                    feedback.get("improvement_directions")
                    or feedback.get("improvements")
                    or ""
                )
                feedback["improvement_directions"] = raw_improvement
                feedback.setdefault("improvements", raw_improvement)

            judge_trace = build_llm_trace(
                module_name=self.name,
                system=system_prompt,
                prompt=prompt,
                response=response,
                request_extra={
                    "program_id": program.id,
                    "parent_id": program.parent_id,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                },
                parsed={"llm_feedback": feedback},
            )
            append_judge_trace(program, judge_trace)
            debug_attempt = program.meta.get("debug_attempts")
            if isinstance(debug_attempt, int):
                attach_debug_trace(
                    program,
                    attempt=debug_attempt,
                    field_name="judge_trace",
                    trace=judge_trace,
                )

            self.log_info(
                f"Generated evaluation for {program.id}",
                program_id=program.id,
                feedback_length=len(response.text),
            )

            return feedback  # Return the feedback dict

        except Exception as e:
            temperature = get_llm_temperature(self.llm_client)
            max_tokens = get_llm_max_tokens(self.llm_client)
            timeout = get_llm_timeout(self.llm_client)
            judge_trace = build_llm_trace(
                module_name=self.name,
                system=system_prompt,
                prompt=prompt,
                response=response,
                request_extra={
                    "program_id": program.id,
                    "parent_id": program.parent_id,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                },
                parsed={"llm_feedback": None, "parse_error": str(e)},
                error=e,
            )
            append_judge_trace(program, judge_trace)
            debug_attempt = program.meta.get("debug_attempts")
            if isinstance(debug_attempt, int):
                attach_debug_trace(
                    program,
                    attempt=debug_attempt,
                    field_name="judge_trace",
                    trace=judge_trace,
                )
            self.log_error(
                f"Evaluation failed for {program.id}: {e}",
                program_id=program.id,
            )
            raise e

    def _build_feedback_prompt(
        self,
        program: Program,
        parent: Optional[Program],
        context: Context,
    ) -> str:
        """
        Build prompt for LLM evaluation.

        Uses template from famou/prompts/templates/evaluation/judge_feedback.txt

        Template expects:
        - language: Programming language
        - parent_program: Parent program code (or "N/A")
        - current_program: Current program code
        - error_info: Error messages (or "None")

        Args:
            program: Program to evaluate
            parent: Parent program for comparison (if available)
            context: Experiment context

        Returns:
            Formatted evaluation prompt
        """
        template_name = "evaluation/judge_feedback.txt"

        # Get parent code (or N/A if no parent)
        parent_code = parent.code if parent else "N/A (no parent available)"

        # Get error info
        error_info = program.normalized_error_info or "None"
        parent_error_info = (
            parent.normalized_error_info if parent else None
        ) or "None"

        return prompt_registry.get(
            template_name,
            language=program.language,
            parent_program=parent_code,
            current_program=program.code,
            error_info=error_info,
            parent_combined_score=(
                parent.combined_score if parent and parent.combined_score is not None else "N/A"
            ),
            current_combined_score=(
                program.combined_score if program.combined_score is not None else "N/A"
            ),
            parent_validity=(
                parent.validity if parent and parent.validity is not None else "N/A"
            ),
            current_validity=(
                program.validity if program.validity is not None else "N/A"
            ),
            parent_metrics=(dict(parent.metrics or {}) if parent else {}),
            current_metrics=dict(program.metrics or {}),
            parent_eval_wall_time=_format_eval_wall_time(parent),
            current_eval_wall_time=_format_eval_wall_time(program),
            parent_error_info=parent_error_info,
        )
