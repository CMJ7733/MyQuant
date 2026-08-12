"""Draft generation module for Famou 2.0.

Proposes novel and effective solutions distinct from previously tried ideas.
"""

import json
from typing import List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
import time


class DraftGenerate(GenerateModule):
    """
    Generate novel solutions by exploring diverse approaches.

    This module is designed to propose creative and distinct solutions that
    differ from previously explored ideas. It adapts its strategy based on
    the iteration number:

    - First draft (iteration 0): Focus on understanding and correctness
    - Early exploration (iterations 1-2): Try diverse algorithms and strategies
    - Refined draft (iterations 3+): Build upon successful elements

    The template asks the LLM to:
    1. Propose novel ideas distinct from previous explorations
    2. Implement complete, runnable solutions
    3. Leverage verified experiences when available
    4. Adapt strategy based on iteration stage

    This is useful for:
    - Exploring diverse solution spaces
    - Avoiding getting stuck in local optima
    - Creative problem solving

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.8 for creative exploration)
        max_tokens: Max tokens (default: 3000)
        include_memory: Include previously explored ideas (default: True)
        include_experience: Include verified experiences (default: True)

    Example:
        >>> generator = DraftGenerate()
        >>> # Explores novel approaches across iterations
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.8,
        max_tokens: int = 3000,
        include_memory: bool = True,
        include_experience: bool = True,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.include_memory = include_memory
        self.include_experience = include_experience

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build draft prompt with iteration-aware strategy.

        Uses template from famou/prompts/templates/generation/draft.txt

        Args:
            context: Experiment context
            selection: SelectionData with inspirations and experiences

        Returns:
            Formatted prompt string
        """
        # Get data preview from context if available
        data_preview = getattr(context, 'data_preview', None)

        # Get init code from context if available
        init_code = getattr(context, 'init_code', None)

        # Get packages from context if available
        packages = getattr(context, 'packages', None)

        # Format memory (previously explored ideas)
        memory = None
        if self.include_memory and selection.inspiration_ids:
            memory = self._format_memory(context, selection.inspiration_ids)

        # Format experience (verified effective experiences)
        experience = None
        if self.include_experience and selection.experiences:
            experience = self._format_experiences(context, selection.experiences)

        # Get current iteration
        current_iteration = getattr(context, 'iteration', 0)

        # Render draft template with iteration-aware instructions
        prompt = prompt_registry.get(
            "generation/draft.txt",
            language=context.language,
            task_description=context.task_description,
            data_preview=data_preview,
            init_code=init_code,
            memory=memory,
            experience=experience,
            packages=packages,
            current_iteration=current_iteration,
        )

        self.log_info(
            f"Draft generation (iteration {current_iteration}): "
            f"memory={bool(memory)}, experience={bool(experience)}"
        )

        return prompt

    def _format_memory(self, context: Context, inspiration_ids: List[str]) -> str:
        """
        Format previously explored ideas as memory.

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
            # Use program template for consistent formatting
            program_text = prompt_registry.get(
                "base/program.txt",
                language=prog.language,
                program_code=prog.code,
                combined_score=prog.combined_score,
                error_info=prog.error_info,
                metrics=prog.metrics,
            )
            formatted.append(
                f"## Explored Idea {i + 1} (Generation {prog.generation})\n"
                f"{program_text}"
            )

        return "\n\n".join(formatted)

    def _format_experiences(self, context: Context, experiences: List) -> str:
        """
        Format verified effective experiences.

        Args:
            context: Experiment context
            experiences: List of experience objects/IDs

        Returns:
            Formatted experiences string
        """
        if not experiences:
            return ""

        # If experiences are strings (IDs), look them up
        # If they're objects with content, format them directly
        formatted_experiences = []

        for exp in experiences[:5]:  # Limit to 5 experiences
            if isinstance(exp, str):
                # Experience ID - would need to lookup from experience store
                # For now, just note the ID
                formatted_experiences.append(f"Experience ID: {exp}")
            elif isinstance(exp, dict):
                # Experience object with fields
                formatted = json.dumps(exp, indent=2)
                formatted_experiences.append(formatted)
            else:
                # Try to convert to string
                formatted_experiences.append(str(exp))

        return "\n\n".join(formatted_experiences)

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Process LLM JSON response into a Program.

        The draft template returns JSON with structure:
        {
            "thought_process": {...},
            "code": "complete code string",
            "required_packages": ["pkg1", "pkg2"]
        }

        This method extracts and parses the JSON, unlike the base class which
        expects markdown-formatted code.

        Args:
            response: LLM response with JSON text
            context: Experiment context
            selection: Selection data used for generation
            parent: Parent program

        Returns:
            Program instance with code, thought_process, and packages
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

            # Handle required_packages
            required_packages = self._sanitize_required_packages(required_packages)

            # Create program with draft-specific metadata
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
                    "thought_process": thought_process,  # Store thinking process
                    "generation_type": "draft",  # Mark as draft generation
                },
                created_at=created_at,
            )

            self.log_info(
                f"Draft generation completed: thought_process keys={list(thought_process.keys())}, "
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
                f"{self.name}: Failed to generate draft solution. "
                "Check LLM client and prompt configuration."
            )
