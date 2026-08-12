"""Mutation-based generation module for Famou 2.0."""

from typing import List

from famou.core.data import Context, RolloutResult, SelectionData
from famou.prompts import prompt_registry
from famou.modules.generate.base import GenerateModule, WRITE_MODE_DIFF
from famou.utils.code_parser import has_single_evolve_block


class MutationGenerate(GenerateModule):
    """
    Generate improved programs through evolutionary mutation.

    Uses a simple prompt template that asks the LLM to improve the parent code.
    Optionally includes error feedback if the parent had execution errors.

    This is the basic generation strategy for evolutionary code improvement,
    analogous to mutation in genetic algorithms.

    Args:
        name: Module name (default: class name)
    """

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build improvement prompt for parent program.

        Uses template from famou/prompts/templates/generation/mutation.txt

        Template expects:
        - language: Programming language
        - inspirations: Formatted evolution history string
        - parent_program: Current program code

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and inspiration_ids

        Returns:
            Formatted prompt string
        """
        parent = context.get_program_by_id(selection.parent_id)
        template_name = "generation/mutation.txt"

        parent_program = prompt_registry.get(
            "base/program.txt",
            language=parent.language,
            program_code=parent.code,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )
        # Format inspirations as evolution history
        inspirations_text = self._format_inspirations(context, selection.inspiration_ids)

        # Render template with variables
        prompt = prompt_registry.get(
            template_name,
            language=context.language,
            inspirations=inspirations_text,
            parent_program=parent_program,
            diff_write_mode=self.write_mode == WRITE_MODE_DIFF,
            has_evolve_block=has_single_evolve_block(parent.code, context.language),
        )

        return prompt

    def _format_inspirations(self, context: Context, inspiration_ids: List[str]) -> str:
        """
        Format evolution history as inspiration text using base/program.txt template.

        Args:
            context: Experiment context
            parent: Current parent program
            max_count: Maximum number of inspirations to include

        Returns:
            Formatted string of evolution history
        """
        inspirations = [context.get_program_by_id(id) for id in inspiration_ids]
        formatted_programs = []
        for i, prog in enumerate(inspirations):
            program_text = prompt_registry.get(
                "base/program.txt",
                language=prog.language,
                program_code=prog.code,
                combined_score=prog.combined_score,
                error_info=prog.error_info,
                metrics=prog.metrics,
            )
            formatted_programs.append(f"## Program {i} (Generation {prog.generation})\n{program_text}")

        return "\n".join(formatted_programs)

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a program. "
                "Check LLM client and prompt configuration."
            )
