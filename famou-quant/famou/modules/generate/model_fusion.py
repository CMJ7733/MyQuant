"""Model fusion generation module for Famou 2.0.

Creates ensemble solutions by combining multiple historical models.
"""

import json
from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule, WRITE_MODE_DIFF
from famou.prompts import prompt_registry
from famou.utils.code_parser import evolve_code, has_single_evolve_block
from famou.utils.id_gen import generate_program_id
from famou.utils.program_summary import build_program_summary
import time


class ModelFusionGenerate(GenerateModule):
    """
    Generate ensemble programs by combining multiple historical models.

    This module is designed for machine learning tasks where you want to
    create ensemble solutions from a repository of existing models without
    retraining from scratch.

    The template asks the LLM to:
    1. Analyze available models and their complementary strengths
    2. Design an optimal ensemble strategy
    3. Implement the ensemble combining selected models

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.7 for creative ensemble design)
        max_tokens: Max tokens (default: 3000 for longer implementations)
        max_models: Maximum number of models to include in ensemble (default: 5)

    Example:
        >>> generator = ModelFusionGenerate(max_models=5)
        >>> # Creates ensemble from top models in population
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 3000,
        max_models: int = 5,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.max_models = max_models

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build model fusion prompt.

        Uses template from famou/prompts/templates/generation/model_fusion.txt

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and inspiration_ids

        Returns:
            Formatted prompt string
        """
        fusion_context = self._get_fusion_context(selection)
        models_to_fuse = self._resolve_models_to_fuse(context, selection, fusion_context)
        anchor_summary, candidate_summaries = self._resolve_prompt_summaries(
            models_to_fuse,
            fusion_context,
        )

        anchor_code = models_to_fuse[0].code if models_to_fuse else ""

        # Render model fusion template
        prompt = prompt_registry.get(
            "generation/model_fusion.txt",
            language=context.language,
            task_description=context.task_description,
            strategy_guardrails=(
                fusion_context.get("strategy_guardrails")
                if isinstance(fusion_context, dict)
                else None
            ),
            anchor_model_block=self._format_model_prompt_block(
                title="AnchorModel",
                summary=anchor_summary,
            ),
            candidate_models_block="\n\n".join(
                self._format_model_prompt_block(
                    title=f"CandidateModel{index}",
                    summary=summary,
                )
                for index, summary in enumerate(candidate_summaries, start=1)
            )
            if candidate_summaries
            else "No candidate models available.",
            diff_write_mode=self.write_mode == WRITE_MODE_DIFF,
            has_evolve_block=has_single_evolve_block(anchor_code, context.language),
        )

        self.log_info(
            f"Model fusion: combining {len(models_to_fuse)} models "
            f"(scores: {[p.combined_score for p in models_to_fuse[:3]]}...)"
        )

        return prompt

    def _get_fusion_context(self, selection: SelectionData) -> Optional[Dict[str, Any]]:
        """Read optional strategy-provided fusion metadata from selection.extra."""
        extra = selection.extra if isinstance(selection.extra, dict) else {}
        fusion_context = extra.get("fusion_context")
        return fusion_context if isinstance(fusion_context, dict) else None

    def _resolve_models_to_fuse(
        self,
        context: Context,
        selection: SelectionData,
        fusion_context: Optional[Dict[str, Any]],
    ) -> List[Program]:
        """Resolve the exact model list either from strategy metadata or score-based fallback."""
        if fusion_context is not None:
            ordered_ids = [
                str(fusion_context.get("parent_id") or "").strip(),
                *[
                    str(program_id).strip()
                    for program_id in list(fusion_context.get("inspiration_ids") or [])
                ],
            ]
            resolved: List[Program] = []
            for program_id in ordered_ids:
                if not program_id:
                    continue
                program = context.get_program_by_id(program_id)
                if program is not None:
                    resolved.append(program)
            if resolved:
                return resolved[: self.max_models]

        parent = context.get_program_by_id(selection.parent_id)
        inspiration_programs = [
            context.get_program_by_id(program_id)
            for program_id in list(selection.inspiration_ids or [])
        ]
        resolved = [program for program in [parent, *inspiration_programs] if program is not None]
        if resolved:
            return resolved[: self.max_models]

        all_programs = context.accessor.get_all()
        return self._select_models(all_programs)

    def _select_models(self, programs: List[Program]) -> List[Program]:
        """
        Select top models for ensemble fusion.

        Args:
            programs: All available programs

        Returns:
            List of top programs sorted by performance
        """
        # Filter programs with valid scores
        valid_programs = [
            p for p in programs
            if p.combined_score is not None and p.combined_score > 0
        ]

        # Sort by score descending
        valid_programs.sort(key=lambda p: p.combined_score or 0, reverse=True)

        # Return top N models
        return valid_programs[:self.max_models]

    def _resolve_prompt_summaries(
        self,
        models: List[Program],
        fusion_context: Optional[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Resolve semantic summaries for the anchor model and candidate models."""
        if fusion_context is not None:
            parent_summary = fusion_context.get("parent_program_summary")
            inspiration_summaries = fusion_context.get("inspiration_program_summaries")
            if isinstance(parent_summary, dict) and isinstance(inspiration_summaries, list):
                normalized_candidates = [
                    summary
                    for summary in inspiration_summaries
                    if isinstance(summary, dict)
                ]
                return parent_summary, normalized_candidates

        fallback_summaries = [
            self._build_prompt_summary_from_program(program)
            for program in models
        ]
        if not fallback_summaries:
            return {}, []
        return fallback_summaries[0], fallback_summaries[1:]

    def _build_prompt_summary_from_program(self, program: Program) -> Dict[str, Any]:
        """Build a semantic fallback summary when strategy metadata is unavailable."""
        summary = build_program_summary(
            program,
            include_code=True,
        )
        return {
            "plan_direction": "",
            "implementation_plan": summary.get("implementation_plan"),
            "combined_score": summary.get("combined_score"),
            "metrics": dict(summary.get("metrics") or {}),
            "eval_time": (
                summary.get("metrics", {}).get("eval_wall_time")
                if isinstance(summary.get("metrics"), dict)
                else None
            ),
            "error_info": summary.get("error_info"),
            "code": summary.get("code") or program.code,
        }

    def _format_model_prompt_block(
        self,
        *,
        title: str,
        summary: Dict[str, Any],
    ) -> str:
        """Format one semantic model block without exposing opaque internal ids."""
        if not isinstance(summary, dict) or not summary:
            return f"<{title}>\nN/A\n</{title}>"

        return (
            f"<{title}>\n"
            f"Plan direction:\n{summary.get('plan_direction') or 'N/A'}\n\n"
            f"Implementation plan:\n{summary.get('implementation_plan') or 'N/A'}\n\n"
            f"Combined score:\n{self._format_scalar(summary.get('combined_score'))}\n\n"
            f"Metrics:\n{json.dumps(dict(summary.get('metrics') or {}), ensure_ascii=False, indent=2)}\n\n"
            f"Eval wall time:\n{self._format_scalar(summary.get('eval_time'))}\n\n"
            f"Error info:\n{summary.get('error_info') or 'None'}\n\n"
            f"Code:\n```python\n{summary.get('code') or ''}\n```\n"
            f"</{title}>"
        )

    def _format_scalar(self, value: Any) -> str:
        """Render optional scalar values for prompt readability."""
        if value is None:
            return "N/A"
        return str(value)

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Process LLM JSON response into a Program.

        The model_fusion template returns JSON with structure:
        {
            "thought_process": {...},
            "code": "complete ensemble implementation",
            "required_packages": ["pkg1", "pkg2"]
        }

        This method extracts and parses the JSON, creating ensemble programs
        that combine multiple models.

        Args:
            response: LLM response with JSON text
            context: Experiment context
            selection: Selection data used for generation
            parent: Parent program

        Returns:
            Program instance with ensemble code and metadata
        """
        try:
            # Parse JSON response
            response_text = response.text.strip()

            # Try to find JSON in the response (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON object found in LLM response")

            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)

            # Extract fields
            code = data.get("code", "")
            thought_process = data.get("thought_process", {})
            required_packages = data.get("required_packages", None)

            if not code:
                raise ValueError("JSON response missing 'code' field")

            # Validate code
            if not code.strip():
                raise RuntimeError("No code extracted from JSON response")

            # diff_write: splice the fused EVOLVE-BLOCK into the init skeleton so
            # the out-of-block suffix (e.g. __main__) is always preserved.
            if self.write_mode == WRITE_MODE_DIFF:
                skeleton = self._get_init_code(context) or parent.code
                code = evolve_code(code, skeleton, language=context.language)

            # Handle required_packages
            required_packages = self._sanitize_required_packages(required_packages)

            # Get ensemble strategy info
            ensemble_strategy = thought_process.get("reasoning", "Ensemble of models")

            # Create program with ensemble-specific metadata
            created_at = time.time()
            program = Program(
                id=generate_program_id(
                    generation=parent.generation + 1,
                    iteration=context.iteration,
                    island_id=context.island_id,
                    created_at=created_at,
                ),
                code=code,
                generation=parent.generation + 1,
                iteration=context.iteration,
                language=context.language,
                parent_id=parent.id,
                system_prompt=context.task_description,
                prompt=response.text,  # Will be overwritten by execute() with actual prompt
                response=response.text,
                thinking=response.thinking,
                required_packages=required_packages,
                meta={
                    "experiment_id": context.experiment_id,
                    "parent_score": parent.combined_score,
                    "selection": selection,
                    "thought_process": thought_process,
                    "generation_type": "model_ensemble",  # Mark as ensemble generation
                    "ensemble_strategy": ensemble_strategy,
                    "ensemble_size": self.max_models,  # Number of models in ensemble
                },
                created_at=created_at,
            )

            fusion_context = self._get_fusion_context(selection)
            if fusion_context is not None:
                program.meta["plan_id"] = fusion_context.get("plan_id")
                program.meta["plan_kind"] = fusion_context.get("plan_kind", "model_fusion")
                program.meta["source_plan_ids"] = list(fusion_context.get("source_plan_ids") or [])
                program.meta["fused_explore_plan_ids"] = list(
                    fusion_context.get("fused_explore_plan_ids") or []
                )
                program.meta["fusion_parent_id"] = fusion_context.get("parent_id")
                program.meta["fusion_inspiration_ids"] = list(
                    fusion_context.get("inspiration_ids") or []
                )

            self.log_info(
                f"Model ensemble generation completed: "
                f"strategy={ensemble_strategy[:50]}, "
                f"models_in_ensemble={self.max_models}, "
                f"packages={required_packages}"
            )

            return program

        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON response from LLM: {e}\n"
                f"Response preview: {response.text[:200]}"
            ) from e
        except KeyError as e:
            raise ValueError(
                f"Missing required field in JSON response: {e}\n"
                f"Available keys: {list(data.keys())}"
            ) from e

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate ensemble program. "
                "Check LLM client and prompt configuration."
            )
