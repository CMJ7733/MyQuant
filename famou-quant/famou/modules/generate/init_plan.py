"""Initial plan generation module for Famou 2.0.

Proposes multiple distinct solution approaches to guide exploration.
"""

import json
from typing import Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
import time


class InitPlanGenerate(GenerateModule):
    """
    Generate multiple distinct solution approaches for initial exploration.

    This module analyzes the task and proposes a diverse set of solution plans
    that span different complexity levels and algorithmic approaches. The plans
    are ordered from simple baselines to sophisticated techniques to guide
    systematic exploration.

    The template asks the LLM to:
    1. Analyze task requirements and constraints
    2. Propose diverse approaches (baseline → intermediate → advanced)
    3. Ensure substantive differences between approaches
    4. Estimate resource needs and difficulty

    This is useful for:
    - Initial exploration of the solution space
    - Understanding the breadth of possible approaches
    - Establishing baseline and target performance levels
    - Guiding subsequent refinement efforts

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.9 for diverse thinking)
        max_tokens: Max tokens (default: 4000 for multiple plans)
        num_plans: Number of plans to generate (default: 3)

    Example:
        >>> generator = InitPlanGenerate(num_plans=5)
        >>> # Generates 5 distinct solution approaches
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.9,
        max_tokens: int = 4000,
        num_plans: int = 3,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.num_plans = num_plans

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build initial plan prompt.

        Uses template from famou/prompts/templates/generation/init_plan.txt

        Args:
            context: Experiment context
            selection: SelectionData (typically minimal for initial plans)

        Returns:
            Formatted prompt string
        """
        # Get data preview from context if available
        data_preview = getattr(context, 'data_preview', None)

        # Get packages from context if available
        packages = getattr(context, 'packages', None)

        # Get time constraints if available
        time_remaining = getattr(context, 'time_remaining', None)
        execution_timeout = getattr(context, 'execution_timeout', None)

        # Render init plan template
        prompt = prompt_registry.get(
            "generation/init_plan.txt",
            language=context.language,
            task_description=context.task_description,
            data_preview=data_preview,
            packages=packages,
            time_remaining=time_remaining,
            execution_timeout=execution_timeout,
            num_plans=self.num_plans,
        )

        self.log_info(
            f"Init plan generation: num_plans={self.num_plans}, "
            f"data_preview={bool(data_preview)}, packages={bool(packages)}"
        )

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

        The init plan template returns JSON with structure:
        {
            "thought_process": {...},
            "plans": [
                {
                    "name": "descriptive name",
                    "description": "detailed explanation",
                    "core_method": "algorithm or technique",
                    "expected_advantages": "why it works",
                    "resource_needs": "requirements",
                    "implementation_difficulty": "Easy/Medium/Hard"
                }
            ]
        }

        Args:
            response: LLM response with JSON text
            context: Experiment context
            selection: Selection data used for generation
            parent: Parent program

        Returns:
            Program instance with plans and thought process
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
            thought_process = data.get("thought_process", {})
            plans = data.get("plans", [])

            if not plans:
                raise ValueError("JSON response missing 'plans' field")

            # Validate plans
            if not isinstance(plans, list) or len(plans) == 0:
                raise ValueError("plans must be a non-empty list")

            # Create program with init plan metadata
            created_at = time.time()
            program = Program(
                id=generate_program_id(
                    generation=parent.generation + 1,
                    iteration=context.iteration,
                    island_id=context.island_id,
                    created_at=created_at,
                ),
                code="",  # Init plans don't generate new code
                generation=parent.generation + 1,
                iteration=context.iteration,
                language=context.language,
                parent_id=parent.id,
                system_prompt=context.task_description,
                prompt=response.text,
                response=response.text,
                thinking=response.thinking,
                required_packages=None,
                meta={
                    "experiment_id": context.experiment_id,
                    "parent_score": parent.combined_score,
                    "selection": selection,
                    "thought_process": thought_process,
                    "plans": plans,  # Store init plans
                    "generation_type": "init_plan",
                },
                created_at=created_at,
            )

            self.log_info(
                f"Init plan generated: {len(plans)} plans, "
                f"thought_process keys={list(thought_process.keys())}"
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
        """
        Validate that initial plans were generated.

        Args:
            context: Experiment context
            result: RolloutResult with generated program

        Raises:
            ValueError: If validation fails
        """
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate initial plans. "
                "Check LLM client and prompt configuration."
            )

        # Check that plans are present in metadata
        if not result.generated_program.meta:
            raise ValueError("No metadata found in generated program")

        plans = result.generated_program.meta.get("plans")
        if not plans:
            raise ValueError("No plans found in generated program metadata")

        self.log_info(f"Validation passed: {len(plans)} initial plans generated")
