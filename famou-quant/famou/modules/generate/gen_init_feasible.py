"""Initial feasible solution generation module for Famou 2.0.

Generates simple, correct baseline solutions for tasks.
"""

import json
from typing import Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
import time


class GenInitFeasibleGenerate(GenerateModule):
    """
    Generate simple, correct baseline solutions.

    This module is designed to create initial feasible solutions that are:
    - Simple and straightforward
    - Correct and directly runnable
    - Focused on reliability over complexity

    The template asks the LLM to:
    1. Analyze the task and data
    2. Implement a basic but working solution
    3. Prioritize approaches that don't require model training when possible
    4. Ensure the code is complete and self-contained

    This is useful for:
    - Creating seed programs for evolution
    - Establishing performance baselines
    - Tasks where simple heuristics work well

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.3 for focused, reliable output)
        max_tokens: Max tokens (default: 2000)
        data_preview: Optional data preview string to include
        init_code: Optional initial code scaffold

    Example:
        >>> generator = GenInitFeasibleGenerate()
        >>> # Creates simple baseline solution
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        data_preview: Optional[str] = None,
        init_code: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.data_preview = data_preview
        self.init_code = init_code

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build initial feasible solution prompt.

        Uses template from famou/prompts/templates/generation/gen_init_feasible.txt

        Args:
            context: Experiment context
            selection: SelectionData (may be None for initial generation)

        Returns:
            Formatted prompt string
        """
        # Get data preview from context if not provided
        data_preview = self.data_preview
        if data_preview is None and hasattr(context, 'data_preview'):
            data_preview = context.data_preview

        # Get init code from context if not provided
        init_code = self.init_code
        if init_code is None and hasattr(context, 'init_code'):
            init_code = context.init_code

        # Render gen_init_feasible template
        prompt = prompt_registry.get(
            "generation/gen_init_feasible.txt",
            language=context.language,
            task_description=context.task_description,
            data_preview=data_preview or "No data preview available.",
            init_code=init_code or "# No initial code scaffold provided",
        )

        self.log_info("Generating initial feasible solution")

        return prompt

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Process LLM JSON response into a Program.

        The gen_init_feasible template returns JSON with structure:
        {
            "thought_process": {...},
            "code": "complete baseline code",
            "required_packages": ["pkg1", "pkg2"]
        }

        This method extracts and parses the JSON, focusing on simple,
        reliable baseline solutions.

        Args:
            response: LLM response with JSON text
            context: Experiment context
            selection: Selection data used for generation
            parent: Parent program

        Returns:
            Program instance with baseline code and metadata
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

            # Create program with baseline-specific metadata
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
                    "generation_type": "initial_baseline",  # Mark as baseline generation
                    "baseline_strategy": thought_process.get("reasoning", "Simple baseline"),
                },
                created_at=created_at,
            )

            self.log_info(
                f"Initial baseline generation completed: "
                f"strategy={thought_process.get('reasoning', 'N/A')[:50]}, "
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
                f"{self.name}: Failed to generate initial feasible solution. "
                "Check LLM client and prompt configuration."
            )
