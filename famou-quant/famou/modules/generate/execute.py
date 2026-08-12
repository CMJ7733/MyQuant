"""Execute generation module for Famou 2.0.

Implements a specific plan or approach as executable code.
"""

import json
from typing import Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
from famou.utils.program_summary import (
    extract_eval_wall_time as _extract_eval_wall_time,
    get_implementation_plan,
)
import time


class ExecuteGenerate(GenerateModule):
    """
    Generate code by implementing a specific plan or approach.

    This module takes a well-defined plan or approach and implements it
    as complete, runnable code. It focuses on faithful implementation of
    the provided plan while ensuring the solution is distinct from
    previously explored ideas.

    The template asks the LLM to:
    1. Analyze the plan requirements and constraints
    2. Implement the plan faithfully and accurately
    3. Ensure the implementation is complete and runnable
    4. Identify potential implementation challenges

    This is useful for:
    - Implementing specific approaches identified during planning
    - Executing ablation studies with precise modifications
    - Realizing specific architectural decisions
    - Creating concrete implementations from abstract plans

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.6 for faithful implementation)
        max_tokens: Max tokens (default: 3000)

    Example:
        >>> generator = ExecuteGenerate()
        >>> # Implements provided plan as executable code
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.6,
        max_tokens: int = 3000,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build execute prompt.

        Uses template from famou/prompts/templates/generation/execute.txt

        Args:
            context: Experiment context
            selection: SelectionData with plan and reference context

        Returns:
            Formatted prompt string
        """
        # Extract plan from selection metadata if available
        plan = None
        constraints = None
        reference_context = None

        if selection.parent_id:
            parent_program = context.get_program_by_id(selection.parent_id)
            if parent_program and hasattr(parent_program, 'meta') and parent_program.meta:
                plan = self._resolve_plan_payload(parent_program, selection)

                # Extract constraints from context
                if hasattr(context, 'constraints'):
                    constraints = context.constraints

                # Extract reference context from inspirations
                if selection.inspiration_ids:
                    reference_context = self._format_reference_context(
                        context, selection.inspiration_ids
                    )

        # Get packages from context if available
        packages = getattr(context, 'packages', None)
        if packages:
            if constraints:
                constraints = f"{constraints}\n\nAvailable Packages:\n{packages}"
            else:
                constraints = f"Available Packages:\n{packages}"

        # Render execute template
        prompt = prompt_registry.get(
            "generation/execute.txt",
            language=context.language,
            task_description=context.task_description,
            plan=plan,
            constraints=constraints,
            context=reference_context,
        )

        self.log_info(
            f"Execute generation: plan={bool(plan)}, "
            f"constraints={bool(constraints)}, context={bool(reference_context)}"
        )

        return prompt

    def _format_reference_context(self, context: Context, inspiration_ids: list) -> str:
        """
        Format reference context from inspiration programs.

        Args:
            context: Experiment context
            inspiration_ids: List of program IDs to reference

        Returns:
            Formatted reference context string
        """
        if not inspiration_ids:
            return ""

        inspirations = []
        for prog_id in inspiration_ids[:3]:  # Limit to 3 references
            program = context.get_program_by_id(prog_id)
            if program and program.code:
                inspirations.append(
                    {
                        "implementation_plan": get_implementation_plan(program),
                        "implementation_code": program.code,
                        "metrics": dict(program.metrics or {}),
                        "combined_score": program.combined_score,
                        "error_info": program.error_info,
                        "eval_time": _extract_eval_wall_time(program),
                    }
                )

        if not inspirations:
            return None
        return json.dumps(inspirations, ensure_ascii=False, indent=2)

    def _resolve_plan_payload(
        self, parent_program: Program, selection: SelectionData
    ) -> Optional[str]:
        """Resolve the selected plan payload from the parent planning artifact."""
        meta = parent_program.meta or {}
        generation_type = meta.get("generation_type", "")
        plan_index = self._get_plan_index(selection)

        if generation_type == "init_plan":
            plans = meta.get("plans", [])
            if not isinstance(plans, list) or not plans:
                self.log_warning(
                    f"{self.name}: Parent init_plan is missing plans, falling back to plan=None"
                )
                return None
            if not 0 <= plan_index < len(plans):
                self.log_warning(
                    f"{self.name}: plan_index {plan_index} out of range for {len(plans)} plans; "
                    "falling back to index 0"
                )
                plan_index = 0
            return json.dumps(plans[plan_index], indent=2)

        if generation_type == "ablation_plan":
            ablation_plans = meta.get("ablation_plans", [])
            if not isinstance(ablation_plans, list) or not ablation_plans:
                self.log_warning(
                    f"{self.name}: Parent ablation_plan is missing ablation_plans, "
                    "falling back to plan=None"
                )
                return None
            if not 0 <= plan_index < len(ablation_plans):
                self.log_warning(
                    f"{self.name}: plan_index {plan_index} out of range for {len(ablation_plans)} "
                    "ablation plans; falling back to index 0"
                )
                plan_index = 0
            return json.dumps(ablation_plans[plan_index], indent=2)

        return None

    def _get_plan_index(self, selection: SelectionData) -> int:
        """Read plan_index from SelectionData.extra with validation."""
        raw_plan_index = selection.extra.get("plan_index", 0)
        if not isinstance(raw_plan_index, int) or raw_plan_index < 0:
            raise ValueError(f"{self.name}: plan_index must be a non-negative int")
        return raw_plan_index

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Process LLM JSON response into a Program.

        The execute template returns JSON with structure:
        {
            "thought_process": {...},
            "code": "complete code string",
            "required_packages": ["pkg1", "pkg2"]
        }

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

            # Create program with execute-specific metadata
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
                prompt=response.text,
                response=response.text,
                thinking=response.thinking,
                required_packages=required_packages,
                meta={
                    "experiment_id": context.experiment_id,
                    "parent_score": parent.combined_score,
                    "selection": selection,
                    "thought_process": thought_process,
                    "generation_type": "execute",
                },
                created_at=created_at,
            )

            self.log_info(
                f"Execute generation completed: thought_process keys={list(thought_process.keys())}, "
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

