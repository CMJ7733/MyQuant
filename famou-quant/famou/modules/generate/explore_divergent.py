"""Explore generation: produce architecturally divergent new solutions.

This is the generation half of the explore operation in the ``pipeline``
strategy. It anchors on the current best program (full code, so the LLM has a
concrete strong baseline) but asks for a *fundamentally different* approach,
and shows the best program of every existing branch — by key_features only —
as negative references ("these directions already exist; produce something
architecturally different").

The produced program is tagged ``meta["operation"]="explore"`` so the branch
graph treats it as a new branch root.
"""

from __future__ import annotations

import time
from typing import List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule, WRITE_MODE_DIFF
from famou.prompts import prompt_registry
from famou.utils.code_parser import has_single_evolve_block


class ExploreDivergentGenerate(GenerateModule):
    """Generate an architecturally divergent program from the current best.

    Args:
        name: Module name (default: class name).
        temperature: LLM temperature override (default: higher for diversity).
        max_tokens: Max tokens override.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """Build the divergent-exploration prompt.

        Uses ``generation/explore_divergent.txt``. Parent (current best) is
        rendered with full code; inspirations (other branch bests) are rendered
        with key_features only.
        """
        parent = context.get_program_by_id(selection.parent_id)

        current_best_program_block = prompt_registry.get(
            "base/program.txt",
            language=parent.language,
            program_code=parent.code,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )

        existing_branches_block = self._format_branch_references(
            context, selection.inspiration_ids or []
        )

        return prompt_registry.get(
            "generation/explore_divergent.txt",
            language=context.language,
            task_description=context.task_description,
            current_best_program_block=current_best_program_block,
            existing_branches_block=existing_branches_block,
            diff_write_mode=self.write_mode == WRITE_MODE_DIFF,
            has_evolve_block=has_single_evolve_block(parent.code, context.language),
        )

    def _format_branch_references(
        self, context: Context, inspiration_ids: List[str]
    ) -> str:
        """Render existing branch bests as code-less summaries (negative references).

        Aligned field-for-field with ``base/program.txt`` (Combined Score,
        Error Info, Evaluation Metrics, Feedback) but WITHOUT the code block —
        explore only needs a sense of direction to diverge from, not the code.
        """
        lines: List[str] = []
        index = 0
        for program_id in inspiration_ids:
            program = context.get_program_by_id(program_id)
            if program is None:
                continue
            index += 1
            summary = prompt_registry.get(
                "base/program_summary.txt",
                combined_score=program.combined_score,
                error_info=program.error_info,
                metrics=program.metrics,
                llm_feedback=program.llm_feedback,
            )
            lines.append(f"## Existing direction {index}\n{summary.strip()}")
        if not lines:
            return "(no existing branches yet — you are free to choose any approach)"
        return "\n\n".join(lines)

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """Tag the generated program as an explore branch root."""
        program = super().post_process(response, context, selection, parent)
        program.meta["operation"] = "explore"
        program.meta["generation_type"] = "explore_divergent"
        return program
