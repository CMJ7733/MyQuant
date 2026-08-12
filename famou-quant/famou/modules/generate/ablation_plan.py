"""Ablation plan generation module for Famou 2.0.

Analyzes existing solutions and proposes systematic ablation studies
to understand component contributions.
"""

import json
from typing import Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
import time


class AblationPlanGenerate(GenerateModule):
    """
    Generate systematic ablation studies to analyze solution components.

    This module analyzes a given solution and proposes diverse, high-impact
    ablation experiments that systematically test different components.

    The template asks the LLM to:
    1. Analyze key components and assumptions in the solution
    2. Design ablation experiments with clear hypotheses
    3. Propose specific modifications for each ablation
    4. Estimate expected impact on performance

    This is useful for:
    - Understanding which components drive performance
    - Identifying critical vs. optional features
    - Guiding future experimental directions
    - Building robust solutions through testing

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.7 for analytical thinking)
        max_tokens: Max tokens (default: 3000)
        include_code: Include current code in analysis (default: True)
        include_features: Include feature breakdown (default: True)

    Example:
        >>> generator = AblationPlanGenerate()
        >>> # Analyzes solution and proposes ablation studies
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.7,
        max_tokens: int = 3000,
        include_code: bool = True,
        include_features: bool = True,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.include_code = include_code
        self.include_features = include_features

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build ablation plan prompt.

        Uses template from famou/prompts/templates/generation/ablation_plan.txt

        Args:
            context: Experiment context
            selection: SelectionData with current solution info

        Returns:
            Formatted prompt string
        """
        # Get current solution info from selection
        current_approach = None
        current_code = None
        current_features = None
        current_metrics = None

        if selection.parent_id:
            # Get parent program to analyze
            parent_program = context.get_program_by_id(selection.parent_id)
            if parent_program:
                # Extract approach from thought process if available
                if hasattr(parent_program, 'meta') and parent_program.meta:
                    meta = parent_program.meta
                    current_approach = meta.get('thought_process', {}).get('reasoning', None)

                current_code = parent_program.code
                current_features = self._extract_features(parent_program)
                current_metrics = self._format_metrics(parent_program)

        # Render ablation plan template
        prompt = prompt_registry.get(
            "generation/ablation_plan.txt",
            language=context.language,
            task_description=context.task_description,
            current_approach=current_approach,
            current_code=current_code,
            current_features=current_features,
            current_metrics=current_metrics,
        )

        self.log_info(
            f"Ablation plan generation: "
            f"include_code={self.include_code}, include_features={self.include_features}"
        )

        return prompt

    def _extract_features(self, program: Program) -> Optional[str]:
        """
        Extract key features from a program.

        Args:
            program: Program object

        Returns:
            Formatted feature description string
        """
        features = []

        # Extract from metrics
        if program.metrics:
            score = (
                f"{program.combined_score:.4f}"
                if program.combined_score is not None
                else "N/A"
            )
            features.append(f"Performance: score={score}")

        # Extract from generation info
        features.append(f"Generation: {program.generation}")
        features.append(f"Iteration: {program.iteration}")

        # Extract from code (basic analysis)
        if program.code:
            # Count lines of code
            lines = [line for line in program.code.split('\n') if line.strip() and not line.strip().startswith('#')]
            features.append(f"Lines of code: {len(lines)}")

            # Check for key programming constructs
            constructs = []
            if 'for ' in program.code or 'while ' in program.code:
                constructs.append("loops")
            if 'def ' in program.code:
                constructs.append("functions")
            if 'class ' in program.code:
                constructs.append("classes")
            if constructs:
                features.append(f"Constructs: {', '.join(constructs)}")

        return "\n".join(features) if features else None

    def _format_metrics(self, program: Program) -> Optional[str]:
        """
        Format program metrics as a string.

        Args:
            program: Program object

        Returns:
            Formatted metrics string
        """
        if not program.metrics:
            return None

        metrics = []
        for key, value in program.metrics.items():
            if isinstance(value, float):
                metrics.append(f"{key}: {value:.4f}")
            else:
                metrics.append(f"{key}: {value}")

        return "\n".join(metrics) if metrics else None

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Process LLM JSON response into a Program.

        The ablation plan template returns JSON with structure:
        {
            "thought_process": {...},
            "ablation_plans": [
                {
                    "name": "descriptive name",
                    "hypothesis": "what we expect to learn",
                    "modifications": ["change 1", "change 2"],
                    "expected_impact": "High/Medium/Low"
                }
            ]
        }

        Args:
            response: LLM response with JSON text
            context: Experiment context
            selection: Selection data used for generation
            parent: Parent program

        Returns:
            Program instance with ablation plans and thought process
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
            ablation_plans = data.get("ablation_plans", [])

            if not ablation_plans:
                raise ValueError("JSON response missing 'ablation_plans' field")

            # Validate ablation plans
            if not isinstance(ablation_plans, list) or len(ablation_plans) == 0:
                raise ValueError("ablation_plans must be a non-empty list")

            # Create program with ablation-specific metadata
            created_at = time.time()
            program = Program(
                id=generate_program_id(
                    generation=parent.generation + 1,
                    iteration=context.iteration,
                    island_id=context.island_id,
                    created_at=created_at,
                ),
                code="",  # Ablation plans don't generate new code
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
                    "ablation_plans": ablation_plans,  # Store ablation plans
                    "generation_type": "ablation_plan",
                },
                created_at=created_at,
            )

            self.log_info(
                f"Ablation plan generated: {len(ablation_plans)} plans, "
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
        Validate that ablation plans were generated.

        Args:
            context: Experiment context
            result: RolloutResult with generated program

        Raises:
            ValueError: If validation fails
        """
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate ablation plans. "
                "Check LLM client and prompt configuration."
            )

        # Check that ablation plans are present in metadata
        if not result.generated_program.meta:
            raise ValueError("No metadata found in generated program")

        ablation_plans = result.generated_program.meta.get("ablation_plans")
        if not ablation_plans:
            raise ValueError("No ablation plans found in generated program metadata")

        self.log_info(f"Validation passed: {len(ablation_plans)} ablation plans generated")
