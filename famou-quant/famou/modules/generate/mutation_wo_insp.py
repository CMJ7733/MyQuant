"""Mutation-based generation module without inspirations (ablation variant)."""

from famou.core.data import Context, RolloutResult, SelectionData
from famou.prompts import prompt_registry
from famou.modules.generate.base import GenerateModule


class MutationGenerateWoInsp(GenerateModule):
    """
    Ablation variant of MutationGenerate that omits inspiration programs.

    Identical to MutationGenerate except that the evolution-history /
    inspirations section is removed from the prompt entirely. Used to isolate
    the contribution of inspiration signals in adaptive_cluster_wo_insp.

    Args:
        name: Module name (default: class name)
    """

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """Build improvement prompt for parent program without inspirations."""
        parent = context.get_program_by_id(selection.parent_id)
        template_name = "generation/mutation_wo_insp.txt"

        parent_program = prompt_registry.get(
            "base/program.txt",
            language=parent.language,
            program_code=parent.code,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )

        prompt = prompt_registry.get(
            template_name,
            language=context.language,
            parent_program=parent_program,
        )

        return prompt

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a program. "
                "Check LLM client and prompt configuration."
            )
