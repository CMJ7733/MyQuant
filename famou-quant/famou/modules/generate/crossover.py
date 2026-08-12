"""Crossover-based generation module for Famou 2.0.

Combines one anchor program with multiple candidate programs using LLM.
"""

import json
import random
from typing import Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.modules.generate.base import GenerateModule, WRITE_MODE_DIFF
from famou.prompts import prompt_registry
from famou.utils.code_parser import has_single_evolve_block
from famou.utils.program_summary import get_implementation_plan


class CrossoverGenerate(GenerateModule):
    """
    Generate programs by combining one anchor program with multiple candidates.

    Uses LLM to intelligently merge code from several programs.
    The primary parent provides the base structure, while candidate
    programs contribute complementary ideas and techniques.

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: None, uses client default)
        max_tokens: Max tokens (default: None, uses client default)
        prefer_diverse_secondary: If True, prefer secondary parent with different
            feature vector (default: True)

    Example:
        >>> generator = CrossoverGenerate()
        >>> # Combines primary parent with first inspiration
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        prefer_diverse_secondary: bool = True,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.prefer_diverse_secondary = prefer_diverse_secondary

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build crossover prompt combining the anchor program with multiple candidates.

        Uses template from famou/prompts/templates/generation/crossover.txt

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and inspiration_ids

        Returns:
            Formatted prompt string
        """
        parent = context.get_program_by_id(selection.parent_id)
        candidate_programs = self._get_candidate_programs(context, selection)

        anchor_program_block = self._format_program_block(
            title="AnchorProgram",
            program=parent,
        )
        candidate_programs_block = "\n\n".join(
            self._format_program_block(
                title=f"CandidateProgram{index}",
                program=program,
            )
            for index, program in enumerate(candidate_programs, start=1)
        ) if candidate_programs else "No candidate programs available."

        prompt = prompt_registry.get(
            "generation/crossover.txt",
            language=context.language,
            task_description=context.task_description,
            strategy_guardrails=(
                selection.extra.get("plan_context", {}).get("strategy_guardrails")
                if isinstance(selection.extra, dict)
                and isinstance(selection.extra.get("plan_context"), dict)
                else None
            ),
            anchor_program_block=anchor_program_block,
            candidate_programs_block=candidate_programs_block,
            diff_write_mode=self.write_mode == WRITE_MODE_DIFF,
            has_evolve_block=has_single_evolve_block(parent.code, context.language),
        )

        self.log_info(
            f"Crossover: anchor={parent.id}, candidates={len(candidate_programs)} "
            f"(anchor_score: {parent.combined_score or 0:.3f})"
        )

        return prompt

    def _get_candidate_programs(
        self, context: Context, selection: SelectionData
    ) -> List[Program]:
        """
        Get candidate programs for crossover.

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and inspiration_ids

        Returns:
            Candidate programs to borrow ideas from
        """
        parent_id = selection.parent_id
        parent = context.get_program_by_id(parent_id)
        candidates: List[Program] = []
        seen_ids = {parent_id}

        for program_id in list(selection.inspiration_ids or []):
            candidate = context.get_program_by_id(program_id)
            if candidate is None or candidate.id in seen_ids:
                continue
            candidates.append(candidate)
            seen_ids.add(candidate.id)

        # Get all programs excluding primary parent
        if candidates:
            return candidates

        all_programs = context.accessor.get_all()
        fallback_candidates = [p for p in all_programs if p.id != parent_id]

        if not fallback_candidates:
            self.log_warning("No crossover candidates, using the anchor program only")
            return []

        # Prefer diverse secondary if enabled and feature vectors available
        if self.prefer_diverse_secondary and parent.feature_vector:
            from famou.utils.math import cosine_distance

            candidates_with_features = [
                p for p in fallback_candidates if p.feature_vector is not None
            ]

            if candidates_with_features:
                distances = [
                    (p, cosine_distance(parent.feature_vector, p.feature_vector))
                    for p in candidates_with_features
                ]
                distances.sort(key=lambda x: x[1], reverse=True)
                return [distances[0][0]]

        scores = [max(p.combined_score or 0.0, 0.001) for p in fallback_candidates]
        total = sum(scores)
        weights = [s / total for s in scores]

        return random.choices(fallback_candidates, weights=weights, k=1)

    def _format_program_block(
        self,
        *,
        title: str,
        program: Program,
    ) -> str:
        """Format one semantic program block without exposing opaque ids."""
        payload: Dict[str, object] = {
            "implementation_plan": get_implementation_plan(program),
            "combined_score": program.combined_score,
            "metrics": dict(program.metrics or {}),
            "validity": program.validity,
            "error_info": program.error_info,
        }
        return (
            f"<{title}>\n"
            f"Implementation plan:\n{payload['implementation_plan'] or 'N/A'}\n\n"
            f"Combined score:\n{payload['combined_score'] if payload['combined_score'] is not None else 'N/A'}\n\n"
            f"Metrics:\n{json.dumps(payload['metrics'], ensure_ascii=False, indent=2)}\n\n"
            f"Validity:\n{payload['validity'] if payload['validity'] is not None else 'N/A'}\n\n"
            f"Error info:\n{payload['error_info'] or 'None'}\n\n"
            f"Code:\n```{program.language}\n{program.code}\n```\n"
            f"</{title}>"
        )

    def post_process(
        self,
        response,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """Persist crossover lineage metadata onto the generated child."""
        program = super().post_process(response, context, selection, parent)

        inherited_source_plan_ids = []
        if isinstance(parent.meta.get("source_plan_ids"), list):
            inherited_source_plan_ids.extend(
                str(plan_id).strip()
                for plan_id in list(parent.meta.get("source_plan_ids") or [])
                if str(plan_id).strip()
            )

        extra = selection.extra if isinstance(selection.extra, dict) else {}
        plan_context = extra.get("plan_context") if isinstance(extra, dict) else None
        crossover_source_plan_ids = []
        if isinstance(plan_context, dict):
            crossover_source_plan_ids.extend(
                str(plan_id).strip()
                for plan_id in list(plan_context.get("planner_crossover_plan_ids") or [])
                if str(plan_id).strip()
            )

        merged_source_plan_ids: List[str] = []
        for plan_id in [*inherited_source_plan_ids, *crossover_source_plan_ids]:
            if plan_id not in merged_source_plan_ids:
                merged_source_plan_ids.append(plan_id)
        if merged_source_plan_ids:
            program.meta["source_plan_ids"] = merged_source_plan_ids

        inherited_fused_explore_plan_ids = []
        if isinstance(parent.meta.get("fused_explore_plan_ids"), list):
            inherited_fused_explore_plan_ids.extend(
                str(plan_id).strip()
                for plan_id in list(parent.meta.get("fused_explore_plan_ids") or [])
                if str(plan_id).strip()
            )
        if inherited_fused_explore_plan_ids:
            program.meta["fused_explore_plan_ids"] = inherited_fused_explore_plan_ids

        return program

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a program via crossover. "
                "Check LLM client and prompt configuration."
            )
