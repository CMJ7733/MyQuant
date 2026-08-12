"""Experience-guided generation module for Famou 2.0.

This module demonstrates StateStore usage by using LLM-analyzed experiences
to inform code generation. Experiences are retrieved and selected by
ExperienceGuidedSelect and passed via SelectionData.experiences.

Key concepts:
- Receives experiences from SelectionData (selected by ExperienceGuidedSelect)
- Includes experience insights in generation prompt
- No inspirations needed - experiences provide the context
"""

from typing import Optional

from famou.core.data import Context, SelectionData
from famou.modules.generate.base import GenerateModule
from famou.prompts.registry import prompt_registry


class ExperienceGuidedGenerate(GenerateModule):
    """
    Generates new programs informed by past successful transformations.

    Retrieves experiences from SelectionData.experiences
    (populated by ExperienceGuidedSelect) and includes the LLM-generated
    insights in the generation prompt.

    The LLM learns from:
    - What changes were made in successful transformations
    - Why those changes improved performance
    - General patterns extracted from past successes

    This creates a learning loop where successful strategies are captured
    and reused in future generations.

    Args:
        name: Module name (default: class name)
        temperature: LLM sampling temperature (default: 0.7)
        max_tokens: Maximum tokens to generate (default: 2000)

    Example:
        >>> # In your pipeline (works with ExperienceCollector)
        >>> rollout = Rollout(
        ...     modules=[
        ...         ExperienceGuidedSelect(
        ...             selection_strategy="diverse",
        ...             max_experiences=5
        ...         ),
        ...         ExperienceGuidedGenerate(),
        ...         EvaluateModule(evaluate_fn=my_eval),
        ...         ExperienceCollector(min_improvement=0.05),  # Stores insights
        ...     ]
        ... )
        >>>
        >>> # Early iterations: No experiences, generates normally
        >>> # Later iterations: Uses accumulated insights to guide generation
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build generation prompt with experience insights.

        Retrieves experiences from selection.experiences
        (populated by ExperienceGuidedSelect) and includes them in the prompt.

        Uses template from famou/prompts/templates/experience/guided_generation.txt

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and experiences

        Returns:
            Formatted prompt string
        """
        # Get parent program
        parent = context.get_program_by_id(selection.parent_id)

        # Get experiences from selection (populated by ExperienceGuidedSelect)
        experiences = selection.experiences or []

        # Get language from context or default to "python"
        language = getattr(context, 'language', 'python')

        # Build parent program section
        parent_program = prompt_registry.get(
            "base/program.txt",
            language=parent.language,
            program_code=parent.code,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )

        # Render template with experiences
        prompt = prompt_registry.get(
            "experience/guided_generation.txt",
            language=language,
            task_description=context.task_description,
            parent_program=parent_program,
            experiences=experiences,
        )

        return prompt
