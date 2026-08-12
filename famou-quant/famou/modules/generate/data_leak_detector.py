"""Data leakage detection and correction generation module for Famou 2.0."""

from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.infrastructure.llm.base import LLMResponse
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry


class DataLeakageDetectorGenerate(GenerateModule):
    """
    Generate leakage-free programs by detecting and fixing data leakage.

    This module analyzes programs for common data leakage patterns and generates
    corrected versions that maintain proper train-test separation.

    Detectable leakage types:
    - **Time series leakage**: Using future information to predict past values
    - **Train-test contamination**: Preprocessing before train-test split
    - **Target leakage**: Including target variable or derived values as features
    - **CV leakage**: Improper cross-validation strategy (e.g., random CV on time series)
    - **File leakage**: Reading test data files (test.csv, submission.csv) during training

    The template asks the LLM to:
    1. Analyze the code for potential leakage sources
    2. Provide corrected code with proper train-test separation
    3. Use appropriate CV strategy for the data type

    This is useful for:
    - Time series forecasting competitions
    - Kaggle competitions with strict train-test separation
    - Preventing accidental data leakage in experiments
    - Ensuring model generalizability

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: 0.3 - lower for precise analysis)
        include_memory: Include previous leakage checks (default: True)
        include_experience: Include verified effective fixes (default: True)

    Example:
        >>> detector = DataLeakageDetectorGenerate()
    """

    LEAKAGE_TYPES = [
        "train_test_contamination",
        "target_leakage",
        "future_info",
        "cv_leakage",
        "file_leakage",
    ]

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 0.3,
        include_memory: bool = True,
        include_experience: bool = True,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, **kwargs)
        self.include_memory = include_memory
        self.include_experience = include_experience

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build data leakage detection prompt.

        Uses template from famou/prompts/templates/generation/data_leak_detector.txt

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

        # Format memory (previous leakage checks)
        memory = None
        if self.include_memory and selection.inspiration_ids:
            memory = self._format_memory(context, selection.inspiration_ids)

        # Format experience (verified effective fixes)
        experience = None
        if self.include_experience and selection.experiences:
            experience = self._format_experiences(context, selection.experiences)

        # Render template
        prompt = prompt_registry.get(
            "generation/data_leak_detector.txt",
            language=context.language,
            task_description=context.task_description,
            data_preview=data_preview,
            parent_program=parent_program,
            memory=memory,
            experience=experience,
            packages=packages,
        )

        self.log_info(
            f"Data leakage detection: "
            f"memory={bool(memory)}, experience={bool(experience)}"
        )

        return prompt

    def _format_memory(self, context: Context, inspiration_ids: List[str]) -> str:
        """
        Format previous leakage checks as memory.

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
            # Extract leakage info from metrics if available
            leakage_info = ""
            if prog.metrics and "has_leakage" in prog.metrics:
                leakage_info += f" (Leakage detected: {prog.metrics.get('has_leakage')})"

            program_text = prompt_registry.get(
                "base/program.txt",
                language=prog.language,
                program_code=prog.code,
                combined_score=prog.combined_score,
                error_info=prog.error_info,
                metrics=prog.metrics,
            )
            formatted.append(
                f"## Previous Check {i + 1} (Generation {prog.generation}){leakage_info}\n"
                f"{program_text}"
            )

        return "\n\n".join(formatted)

    def _format_experiences(self, context: Context, experiences: List) -> str:
        """
        Format verified effective leakage fixes.

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

    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for leakage analysis and corrected code.

        Args:
            response: Raw LLM response string

        Returns:
            Dictionary with leakage_analysis, corrected_code, and required_packages
        """
        import json
        import re

        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group(0))

                # Validate structure
                if "leakage_analysis" in result and "corrected_code" in result:
                    return result
                elif "code" in result:
                    # Fallback to simpler structure
                    return {
                        "leakage_analysis": {"has_leakage": "unknown"},
                        "corrected_code": result["code"],
                        "required_packages": result.get("required_packages", []),
                    }
            except json.JSONDecodeError:
                pass

        # Fallback: try to extract code directly
        code_match = re.search(r'```(?:python|{{language}})?\s*(.*?)```', response, re.DOTALL)
        if code_match:
            return {
                "leakage_analysis": {"has_leakage": "unknown - parsing failed", "issues_found": []},
                "corrected_code": code_match.group(1).strip(),
                "required_packages": [],
            }

        # Last resort: return full response as code
        return {
            "leakage_analysis": {"has_leakage": "unknown - parsing failed", "issues_found": []},
            "corrected_code": response,
            "required_packages": [],
        }

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Run leak detection on the already-generated program (Option A pipeline position).

        The generated program is registered as transient and pointed to as
        selection.parent_id so that build_prompt / post_process receives it as
        the program to inspect.

        post_process handles the decision:
          - false: returns the same Program object (passthrough, original_code stored in meta)
          - true:  returns a model_copy inheriting all attributes, with corrected code
        """
        original_program = result.generated_program
        if original_program is None:
            return result

        # Register so context.get_program_by_id works inside build_prompt
        if context.accessor:
            context.accessor.register_transient(original_program)

        # Point selection at the generated program as the inspection target
        original_parent_id = result.selection.parent_id if result.selection else None
        if result.selection:
            result.selection.parent_id = original_program.id

        result = super().execute(context, result, **kwargs)

        # Restore original parent_id for lineage tracking downstream
        if result.selection and original_parent_id is not None:
            result.selection.parent_id = original_parent_id

        # If post_process returned None for some reason, fall back to original
        if result.generated_program is None:
            result.generated_program = original_program

        return result

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Parse LLM response.

        - false: return parent unchanged (no code generated)
        - true:  build a new Program that inherits ALL attributes of parent
                 but replaces code/required_packages with the corrected version
        """
        parsed = self.parse_llm_response(response.text)
        leakage_analysis = parsed.get("leakage_analysis", {})
        has_leakage = leakage_analysis.get("has_leakage", False)
        issues_found = leakage_analysis.get("issues_found", [])

        if not has_leakage or has_leakage == "false":
            self.log_info("No leakage detected — passthrough")
            # Attach analysis to parent meta for traceability, return parent unchanged
            parent.meta["leakage_analysis"] = leakage_analysis
            return parent

        # Leakage detected: build corrected program inheriting all parent attributes
        corrected_code = parsed.get("corrected_code", "")
        required_packages = parsed.get("required_packages")

        if not isinstance(corrected_code, str) or not corrected_code.strip():
            raise RuntimeError("Leakage detected but no corrected_code in LLM response")

        self.log_warning(
            f"Leakage detected: {issues_found} — replacing code"
        )

        required_packages = self._sanitize_required_packages(required_packages)

        # Inherit ALL attributes from parent, only override code and required_packages
        corrected = parent.model_copy(
            update={
                "code": corrected_code.strip(),
                "required_packages": required_packages,
                "meta": {
                    **parent.meta,
                    "generation_type": "data_leak_detector",
                    "leakage_analysis": leakage_analysis,
                    "original_code": parent.code,
                    "original_required_packages": parent.required_packages,
                },
            }
        )
        return corrected

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate leakage-free program. "
                "Check LLM client and prompt configuration."
            )
