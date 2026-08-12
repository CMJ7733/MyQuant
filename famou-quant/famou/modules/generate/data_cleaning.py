"""Data cleaning and preprocessing generation module for Famou 2.0.

Enhances programs through systematic data cleaning and preprocessing.
"""

from typing import List, Optional

from famou.core.data import Context, RolloutResult, SelectionData
from famou.prompts import prompt_registry
from famou.modules.generate.base import GenerateModule


class DataCleaningGenerate(GenerateModule):
    """
    Generate improved programs through data cleaning and preprocessing.

    This module specializes in improving data quality through systematic
    cleaning and preprocessing techniques. It adapts its strategy based on complexity level:

    - Simple: Essential cleaning (missing values, basic outliers, scaling)
    - Medium: Diverse cleaning strategies (advanced imputation, outlier detection, balancing)
    - High: Advanced preprocessing pipeline with automated cleaning and optimization

    The template asks the LLM to:
    1. Identify data quality issues (missing values, outliers, imbalance)
    2. Implement appropriate cleaning and preprocessing techniques
    3. Ensure proper fit/transform pattern to avoid data leakage
    4. Validate cleaning impact on model performance

    This is useful for:
    - Handling messy real-world data
    - Dealing with imbalanced datasets
    - Improving model robustness to data issues
    - Preventing overfitting to noise

    Args:
        name: Module name (default: class name)
        complexity: Strategy complexity - "simple", "medium", or "high" (default: "medium")
        temperature: LLM temperature (default: 0.7)
        max_tokens: Max tokens (default: 4000)
        include_memory: Include previously explored cleaning (default: True)
        include_experience: Include verified effective cleaning (default: True)

    Example:
        >>> # Basic data cleaning
        >>> generator = DataCleaningGenerate(complexity="simple")
        >>>
        >>> # Advanced preprocessing pipeline
        >>> generator = DataCleaningGenerate(complexity="high")
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        complexity: str = "medium",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        include_memory: bool = True,
        include_experience: bool = True,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)

        # Validate complexity
        if complexity not in ["simple", "medium", "high"]:
            raise ValueError(
                f"complexity must be 'simple', 'medium', or 'high', got '{complexity}'"
            )

        self.complexity = complexity
        self.include_memory = include_memory
        self.include_experience = include_experience

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build data cleaning prompt.

        Uses template from famou/prompts/templates/generation/data_cleaning.txt

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and inspirations

        Returns:
            Formatted prompt string
        """
        parent = context.get_program_by_id(selection.parent_id)

        # Format parent program
        parent_program = prompt_registry.get(
            "base/program.txt",
            language=parent.language,
            program_code=parent.code,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )

        # Get data preview from context if available
        data_preview = getattr(context, 'data_preview', None)

        # Get packages from context if available
        packages = getattr(context, 'packages', None)

        # Format memory (previously explored cleaning)
        memory = None
        if self.include_memory and selection.inspiration_ids:
            memory = self._format_memory(context, selection.inspiration_ids)

        # Format experience (verified effective cleaning)
        experience = None
        if self.include_experience and selection.experiences:
            experience = self._format_experiences(context, selection.experiences)

        # Render template
        prompt = prompt_registry.get(
            "generation/data_cleaning.txt",
            language=context.language,
            task_description=context.task_description,
            data_preview=data_preview,
            parent_program=parent_program,
            memory=memory,
            experience=experience,
            packages=packages,
            complexity=self.complexity,
        )

        self.log_info(
            f"Data cleaning (complexity={self.complexity}): "
            f"memory={bool(memory)}, experience={bool(experience)}"
        )

        return prompt

    def _format_memory(self, context: Context, inspiration_ids: List[str]) -> str:
        """
        Format previously explored cleaning as memory.

        Args:
            context: Experiment context
            inspiration_ids: List of program IDs to include

        Returns:
            Formatted memory string
        """
        if not inspiration_ids:
            return ""

        inspirations = []
        for prog_id in inspiration_ids:
            program = context.get_program_by_id(prog_id)
            if program:
                inspirations.append(program)

        if not inspirations:
            return ""

        formatted = []
        for i, prog in enumerate(inspirations):
            program_text = prompt_registry.get(
                "base/program.txt",
                language=prog.language,
                program_code=prog.code,
                combined_score=prog.combined_score,
                error_info=prog.error_info,
                metrics=prog.metrics,
            )
            formatted.append(
                f"## Previous Cleaning {i + 1} (Generation {prog.generation})\n"
                f"{program_text}"
            )

        return "\n\n".join(formatted)

    def _format_experiences(self, context: Context, experiences: List) -> str:
        """
        Format verified effective cleaning experiences.

        Args:
            context: Experiment context
            experiences: List of experience objects/IDs

        Returns:
            Formatted experiences string
        """
        if not experiences:
            return ""

        formatted_experiences = []

        for exp in experiences[:5]:  # Limit to 5 experiences
            if isinstance(exp, str):
                formatted_experiences.append(f"Experience ID: {exp}")
            elif isinstance(exp, dict):
                import json
                formatted = json.dumps(exp, indent=2)
                formatted_experiences.append(formatted)
            else:
                formatted_experiences.append(str(exp))

        return "\n\n".join(formatted_experiences)

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate program with data cleaning. "
                "Check LLM client and prompt configuration."
            )
