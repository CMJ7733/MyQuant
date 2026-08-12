"""
Base generation modules for Famou 2.0.

All LLM-based generation modules:
- Read from result.selection.parent_id (str - single parent ID)
- Look up parent Program from context.accessor
- Write ONE Program to result.generated_program
"""

import logging
import time
from abc import abstractmethod
from typing import Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.core.protocol import Module, RequiresLLM
from famou.core.types import FatalRolloutError
from famou.infrastructure.llm.base import LLMClient, LLMResponse, get_llm_max_tokens
from famou.utils.code_parser import (
    extract_code_from_markdown,
    evolve_code,
    extract_required_packages,
    sanitize_required_packages,
)
from famou.utils.id_gen import generate_program_id
from famou.utils.trace_utils import append_named_trace, build_llm_trace, set_generate_trace


logger = logging.getLogger(__name__)

# Code write modes (see GenerateModule.post_process):
#   full_write (default) — use the LLM's full output as-is.
#   diff_write           — splice only the EVOLVE-BLOCK back into the parent.
WRITE_MODE_FULL = "full_write"
WRITE_MODE_DIFF = "diff_write"
_VALID_WRITE_MODES = (WRITE_MODE_FULL, WRITE_MODE_DIFF)


class GenerateModule(Module, RequiresLLM):
    """
    Base class for LLM-based code generation strategies.

    **SIMPLIFIED: One Program Per Rollout**

    All generation uses LLM to create exactly one new program from a selected parent.
    This class handles the common LLM calling logic, while subclasses customize
    how they extract information from context and build prompts.

    Data Flow:
    1. Read parent ID from result.selection.parent_id (str)
    2. Look up parent Program from context.accessor
    3. Generate ONE new Program (with code, prompts, responses)
    4. Write to result.generated_program (single Program)

    Design:
    - Subclasses implement build_prompt() to return a prompt string
    - Base class execute() handles:
      - Parent lookup from population
      - LLM calling and response handling
      - Code extraction and syntax validation
      - Program creation

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature override (default: None, uses LLMClient default)
        max_tokens: Max tokens override (default: None, uses LLMClient default)

    === DATA CONTRACT ===
    READS from Context:
        - context.accessor: For parent program lookup
        - context.task_description: For system prompt

    READS from RolloutResult:
        - result.selection: SelectionData (REQUIRED)
            - selection.parent_id (str): ID of parent program
            - selection.inspiration_ids (List[str]): Reference programs
            - selection.experiences (Optional[List[str]]): Experience IDs
            - selection.skills (Optional[List[str]]): Agent skills

    WRITES to RolloutResult:
        - result.generated_program (Program): NEW program with code, generation+1

    This makes it easy to implement new generation strategies by
    just changing the prompt building logic.
    """

    llm_client: LLMClient  # Injected by RolloutEngine

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Code write mode. Read from generate module params (modules.generate.write_mode).
        #   full_write (default): use the LLM's full output as-is — keeps imports /
        #     helpers the LLM adds outside the EVOLVE-BLOCK.
        #   diff_write: merge only the EVOLVE-BLOCK content back into the parent.
        write_mode = str(kwargs.get("write_mode") or WRITE_MODE_FULL).lower()
        if write_mode not in _VALID_WRITE_MODES:
            logger.warning(
                "Unknown generate write_mode %r; falling back to %r",
                write_mode, WRITE_MODE_FULL,
            )
            write_mode = WRITE_MODE_FULL
        self.write_mode = write_mode

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that selection data is available."""
        if not result.selection or not result.selection.parent_id:
            raise ValueError(
                f"{self.name}: Cannot generate without selection.parent_id. "
                "Make sure SelectModule has run first."
            )

    def _sanitize_required_packages(self, required_packages):
        """Keep only third-party package names that are safe to install."""
        return sanitize_required_packages(required_packages)

    @abstractmethod
    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build generation prompt from selection data.

        This is the core method that subclasses must implement.
        Different generation strategies differ in how they build prompts:
        - Mutation: Ask to improve the parent code
        - Error-driven: Include error messages and ask to fix
        - Multi-objective: Ask for specific improvements (speed, readability, etc.)
        - Crossover: Combine multiple parents using inspirations

        The selection provides access to:
        - selection.parent_id: The program to modify (use context.get_program_by_id())
        - selection.inspiration_ids: Additional programs for reference/examples
        - selection.experiences: Experience IDs for advanced strategies
        - selection.knowledge: Knowledge IDs for advanced strategies

        Args:
            context: Experiment context (contains population, task_description, etc.)
            selection: SelectionData with parent_id, inspiration_ids, experiences, knowledge

        Returns:
            Formatted prompt string to send to LLM

        Example:
            >>> def build_prompt(self, context, selection):
            ...     parent = context.get_program_by_id(selection.parent_id)
            ...     return f"Improve this code:\\n{parent.code}"
        """
        pass

    def post_process(
        self,
        response: LLMResponse,
        context: Context,
        selection: SelectionData,
        parent: Program,
    ) -> Program:
        """
        Process LLM response into a Program.

        Default implementation:
        1. Extracts code from markdown
        2. Evolves code with parent code (fills in missing parts)
        3. Extracts required packages (for Python)
        4. Creates Program instance

        Override for custom processing logic (e.g., skip evolve_code, custom extraction).

        Args:
            response: LLM response with text, thinking, etc.
            context: Experiment context
            selection: Selection data used for generation
            parent: Parent program

        Returns:
            Program instance

        Example (skip evolve_code):
            >>> def post_process(self, response, context, selection, parent):
            ...     code = extract_code_from_markdown(response.text, context.language)
            ...     # Don't evolve, use raw code
            ...     return super().post_process(
            ...         LLMResponse(text=code, thinking=response.thinking),
            ...         context, selection, parent
            ...     )

        Example (custom metadata):
            >>> def post_process(self, response, context, selection, parent):
            ...     program = super().post_process(response, context, selection, parent)
            ...     program.meta["custom_field"] = "custom_value"
            ...     return program
        """
        # Extract code from markdown
        code = extract_code_from_markdown(response.text, context.language)

        # Merge strategy depends on write_mode:
        #   diff_write — splice the child's EVOLVE-BLOCK into the parent, keeping the
        #     parent's out-of-block code (classic evolve-block diff editing).
        #   full_write (default) — use the LLM's full output as-is. Avoids silently
        #     dropping imports / helpers the LLM places outside the block.
        if self.write_mode == WRITE_MODE_DIFF:
            skeleton = self._get_init_code(context) or parent.code
            code = evolve_code(code, skeleton, language=context.language)

        if not code:
            raise RuntimeError("No code extracted from LLM response")

        # Extract required packages (for Python programs)
        required_packages = None
        if context.language == "python":
            required_packages = self._sanitize_required_packages(
                extract_required_packages(response.text)
            )

        created_at = time.time()
        return Program(
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
            },
            file_extension=parent.file_extension,
            created_at=created_at,
        )

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Generate a single new program by prompting LLM.

        This method handles the common LLM calling logic.
        Subclasses customize behavior via build_prompt() and optionally post_process().

        Args:
            context: Experiment context (contains population, task_description, etc.)
            result: Rollout result (contains selection with parent_id, inspirations, etc.)

        Returns:
            Updated result with generated_program populated
        """
        # Validate parent exists
        parent = context.get_program_by_id(result.selection.parent_id)
        if parent is None:
            raise RuntimeError(
                f"Parent program {result.selection.parent_id} not found in population. "
                "Cannot generate without valid parent."
            )

        try:
            # Build prompt using subclass strategy
            prompt = self.build_prompt(context, result.selection)
            response = None
        except FatalRolloutError:
            raise
        except Exception as e:
            raise RuntimeError(
                f"{self.name}: build_prompt failed unexpectedly: {e}"
            ) from e

        # Only pass overrides if explicitly configured (uses LLMClient defaults otherwise)
        llm_kwargs = {}
        if self.temperature is not None:
            llm_kwargs["temperature"] = self.temperature
        request_extra = {
            "temperature": llm_kwargs.get("temperature", getattr(self.llm_client, "temperature", None)),
            "max_tokens": get_llm_max_tokens(self.llm_client),
            "parent_id": parent.id,
            "selection": result.selection,
        }

        try:
            # Call LLM
            self.log_info(f"Generating program from parent {parent.id}")

            response = self.llm_client.generate(
                prompt=prompt,
                system=context.task_description,
                **llm_kwargs,
            )
        except Exception as e:
            append_named_trace(
                parent,
                "generate_failures",
                build_llm_trace(
                    module_name=self.name,
                    system=context.task_description,
                    prompt=prompt,
                    response=None,
                    request_extra=request_extra,
                    error=e,
                ),
            )
            self.log_error(f"Failed to generate program from parent {parent.id}: {e}")
            raise RuntimeError(
                f"Generation failed for parent {parent.id}: {e}"
            ) from e

        try:
            # Create program using post_process hook (handles code extraction, evolution, etc.)
            new_program = self.post_process(
                response=response,
                context=context,
                selection=result.selection,
                parent=parent,
            )
        except FatalRolloutError:
            raise
        except Exception as e:
            append_named_trace(
                parent,
                "generate_failures",
                build_llm_trace(
                    module_name=self.name,
                    system=context.task_description,
                    prompt=prompt,
                    response=response,
                    request_extra=request_extra,
                    error=e,
                ),
            )
            if self._is_retryable_post_process_error(e):
                self.log_error(f"Failed to generate program from parent {parent.id}: {e}")
                raise RuntimeError(
                    f"Generation failed for parent {parent.id}: {e}"
                ) from e
            raise FatalRolloutError(
                f"{self.name}: post_process failed unexpectedly: {e}"
            ) from e

        # Update prompt field with actual prompt used
        new_program.prompt = prompt

        # Tag which module generated this program (for bug chain tracing)
        new_program.meta["operator"] = self.name

        parsed = {
            "program_id": new_program.id,
            "parent_id": new_program.parent_id,
            "required_packages": new_program.required_packages,
        }
        set_generate_trace(
            new_program,
            build_llm_trace(
                module_name=self.name,
                system=context.task_description,
                prompt=prompt,
                response=response,
                request_extra=request_extra,
                parsed=parsed,
            ),
        )

        result.generated_program = new_program

        self.log_info(
            f"Generated program {new_program.id}",
            parent_id=parent.id,
        )

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Enforce that execute() produced a non-empty program."""
        if result.generated_program is None:
            raise ValueError(
                f"{self.name}: execute() did not write result.generated_program"
            )
        if not result.generated_program.code:
            raise ValueError(
                f"{self.name}: Generated program has no code"
            )

    def _is_retryable_post_process_error(self, error: Exception) -> bool:
        """Return True when post-processing failure is likely due to LLM output quality."""
        return isinstance(error, (ValueError, RuntimeError))

    def _get_init_code(self, context: Context) -> Optional[str]:
        """Return the generation-0 seed code to use as diff_write skeleton.

        Walks island_accessor then accessor; returns None when no gen-0 program exists.
        """
        for accessor in (context.island_accessor, context.accessor):
            if accessor is None:
                continue
            try:
                programs = accessor.get_by_generation(0)
            except Exception:
                continue
            if programs:
                seed = programs[0] if isinstance(programs, list) else programs
                code = getattr(seed, "code", None)
                if code:
                    return code
        return None
