"""Error-driven generation module for Famou 2.0.

Focuses on fixing programs that have execution errors.
V2: Includes bug code chain tracing and debug loop bookkeeping.
"""

from copy import deepcopy
from typing import List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.modules.generate.base import GenerateModule
from famou.prompts import prompt_registry
from famou.utils.trace_utils import copy_debug_history, snapshot_program

_INHERITED_PLAN_META_KEYS = (
    "plan",
    "plan_id",
    "plan_text",
    "plan_round",
    "plan_seed_parent_id",
    "plan_best_program_id_at_dispatch",
    "plan_created_iteration",
    "plan_inspiration_ids",
    "generation_type",
    "implementation_plan",
    "plan_snapshot",
)


class ErrorDrivenGenerate(GenerateModule):
    """
    Generate programs by focusing on fixing errors with bug code chain context.

    When the parent program has errors, this module builds a comprehensive debug prompt:
    1. Bug Code Chain: traces back through parent_id links to show the full debug history
       (original plan -> original code/error -> debug attempt 1 -> ... -> current error)
    2. Working examples from inspirations as reference

    If the parent has no errors, falls back to standard mutation.

    In the debug loop context (auto-injected by Evolver), the execute() override
    handles iteration bookkeeping:
    - Registers buggy program as transient so get_by_id() works
    - Saves pre_debug_parent_id for restoration after debug loop
    - Tracks debug attempt count in program meta

    Args:
        name: Module name (default: class name)
        temperature: LLM temperature (default: None, uses client default)
        max_tokens: Max tokens (default: None, uses client default)
        include_working_examples: Include working programs as reference (default: True)
        max_working_examples: Maximum working examples to include (default: 2)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_working_examples: bool = True,
        max_working_examples: int = 2,
        **kwargs,
    ):
        super().__init__(name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.include_working_examples = include_working_examples
        self.max_working_examples = max_working_examples

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Override execute to handle debug loop iteration bookkeeping.

        Before super().execute():
        1. Register the current buggy program in accessor (so get_by_id works)
        2. Update selection.parent_id to point to the current buggy program
        3. Track the original (pre-debug) program for downstream consumption

        After super().execute():
        4. Set debug meta on the new program (attempt count, original info)
        """
        buggy_program = result.generated_program
        prior_debug_history = []
        original_program_id = None
        original_error = None
        original_score = None
        original_stdout = None
        debug_attempt = 0
        if buggy_program and buggy_program.is_buggy and result.selection:
            prior_debug_history = copy_debug_history(buggy_program)
            # Register so get_by_id() can find it during prompt building
            if context.accessor:
                context.accessor.register_transient(buggy_program)

            # Track the original (first buggy) program across iterations
            # First iteration: buggy_program is from upstream Generate
            # Subsequent iterations: carry forward the original info
            if "original_program_id" not in buggy_program.meta:
                # First debug iteration — buggy_program is the original
                original_program_id = buggy_program.id
                original_error = buggy_program.error_info
                original_score = buggy_program.combined_score
                original_stdout = buggy_program.stdout
                debug_attempt = 0

                # Preserve the strategy-selected parent_id for downstream
                if result.selection.extra is None:
                    result.selection.extra = {}
                if "pre_debug_parent_id" not in result.selection.extra:
                    result.selection.extra["pre_debug_parent_id"] = result.selection.parent_id
            else:
                # Subsequent iterations — carry forward from previous
                original_program_id = buggy_program.meta["original_program_id"]
                original_error = buggy_program.meta.get("original_error")
                original_score = buggy_program.meta.get("original_score")
                original_stdout = buggy_program.meta.get("original_stdout")
                debug_attempt = buggy_program.meta.get("debug_attempts", 0)

            # Update parent_id to point to the current buggy program
            result.selection.parent_id = buggy_program.id

            self.log_info(
                f"Debug iteration {debug_attempt + 1}: "
                f"parent_id -> {buggy_program.id}"
            )

        result = super().execute(context, result, **kwargs)

        # Set debug meta on the newly generated program
        new_program = result.generated_program
        if new_program and original_program_id is not None:
            self._inherit_plan_metadata(buggy_program, new_program)
            next_attempt = debug_attempt + 1
            new_program.meta["original_program_id"] = original_program_id
            new_program.meta["original_error"] = original_error
            new_program.meta["original_score"] = original_score
            new_program.meta["original_stdout"] = original_stdout
            new_program.meta["debug_attempts"] = next_attempt
            new_program.meta["was_buggy"] = True
            new_program.meta["debug_history"] = prior_debug_history

            debug_entry = {
                "attempt": next_attempt,
                "input_program": snapshot_program(buggy_program),
                "generate_trace": new_program.meta.get("traces", {}).get("generate"),
            }
            new_program.meta["debug_history"].append(debug_entry)
            new_program.meta["_pending_debug_attempt"] = next_attempt

        return result

    def _inherit_plan_metadata(self, source: Program, target: Program) -> None:
        """Preserve plan lineage when a plan-driven child enters the debug loop."""
        source_meta = source.meta if isinstance(source.meta, dict) else {}
        target_meta = target.meta if isinstance(target.meta, dict) else {}

        for key in _INHERITED_PLAN_META_KEYS:
            if key in source_meta and key not in target_meta:
                target_meta[key] = deepcopy(source_meta[key])

        source_traces = source_meta.get("traces")
        if not isinstance(source_traces, dict):
            return

        target_traces = target_meta.get("traces")
        if not isinstance(target_traces, dict):
            target_traces = {}
            target_meta["traces"] = target_traces

        if "plan_planner" in source_traces and "plan_planner" not in target_traces:
            target_traces["plan_planner"] = deepcopy(source_traces["plan_planner"])

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build error-fixing prompt with bug code chain.

        Args:
            context: Experiment context
            selection: SelectionData with parent_id and inspiration_ids

        Returns:
            Formatted prompt string
        """
        parent = context.get_program_by_id(selection.parent_id)

        if parent.is_buggy:
            return self._build_error_fix_prompt(context, selection, parent)
        else:
            return self._build_mutation_prompt(context, selection, parent)

    def _build_error_fix_prompt(
        self, context: Context, selection: SelectionData, parent: Program
    ) -> str:
        """Build prompt with bug code chain for fixing errors."""
        sections = []

        # 1. Bug Code Chain — show the LLM the full debug history
        bug_chain = self._build_bug_code_chain(context, parent)
        if bug_chain:
            sections.append(bug_chain)

        # 2. Current buggy program (always shown)
        sections.append(self._format_current_bug(context, parent))

        # 3. Working examples from inspirations
        if self.include_working_examples and selection.inspiration_ids:
            working = self._get_working_examples(context, selection.inspiration_ids)
            if working:
                sections.append(f"# Working Reference Programs\n{working}")

        # 4. Task instruction
        sections.append(
            f"# Task\n"
            f"Fix the error in the current buggy program above.\n"
            f"Output a complete, runnable {context.language} program that resolves the error.\n"
            f"Learn from the bug code chain history if provided.\n"
            f"Do NOT repeat previously failed strategies."
        )

        prompt = "\n\n".join(sections)

        self.log_info(
            f"Error-driven generation for {parent.id}: "
            f"chain_depth={self._get_chain_depth(context, parent)}"
        )

        return prompt

    def _build_bug_code_chain(self, context: Context, current: Program) -> str:
        """
        Trace back through parent_id links to build the buggy program chain.

        Only includes buggy programs (validity < 1.0). Stops at the first
        non-buggy ancestor WITHOUT including it.

        Debug iteration 1 sees: [first_buggy]
        Debug iteration 2 sees: [first_buggy, debug_fix_1]
        Debug iteration 3 sees: [first_buggy, debug_fix_1, debug_fix_2]
        """
        chain = []
        prog = current

        # Walk back through parent chain, collecting only buggy programs
        while prog.parent_id:
            parent = context.get_program_by_id(prog.parent_id)
            if not parent:
                break
            # Stop before non-buggy ancestor (don't include it)
            if not parent.is_buggy:
                break
            chain.append(parent)
            prog = parent

        if not chain:
            return ""

        # Reverse to show oldest first
        chain.reverse()

        parts = ["# Bug Code Chain (previous failed attempts)"]

        for i, prog in enumerate(chain):
            error_str = (
                self._format_error_details(prog) if prog.has_error_details else "No error"
            )
            score_str = f"{prog.combined_score:.4f}" if prog.combined_score is not None else "N/A"
            validity_str = f"{prog.validity:.4f}" if prog.validity is not None else "N/A"
            stdout_str = prog.stdout if prog.stdout else "No output"

            entry = (
                f"## Attempt {i + 1}\n"
                f"Score: {score_str}, Validity: {validity_str}\n"
                f"```{context.language}\n{prog.code}\n```\n"
                f"Stdout: {stdout_str}\n"
                f"Error: {error_str}"
            )
            parts.append(entry)

        return "\n\n".join(parts)

    def _format_current_bug(self, context: Context, parent: Program) -> str:
        """Format the current buggy program that needs fixing."""
        error_details = self._format_error_details(parent)
        score_str = f"{parent.combined_score:.4f}" if parent.combined_score is not None else "N/A"
        stdout_str = parent.stdout if parent.stdout else "No output"

        return (
            f"# Current Buggy Program (TO FIX)\n"
            f"Score: {score_str}\n"
            f"```{context.language}\n{parent.code}\n```\n"
            f"Stdout: {stdout_str}\n"
            f"Error:\n{error_details}"
        )

    def _get_chain_depth(self, context: Context, current: Program) -> int:
        """Count the depth of the debug chain (number of buggy ancestors)."""
        depth = 0
        prog = current
        while prog.parent_id:
            parent = context.get_program_by_id(prog.parent_id)
            if not parent:
                break
            if not parent.is_buggy:
                break
            depth += 1
            prog = parent
        return depth

    def _build_mutation_prompt(
        self, context: Context, selection: SelectionData, parent: Program
    ) -> str:
        """Fallback to mutation prompt when no errors."""
        parent_program = prompt_registry.get(
            "base/program.txt",
            language=parent.language,
            program_code=parent.code,
            combined_score=parent.combined_score,
            error_info=parent.error_info,
            metrics=parent.metrics,
            llm_feedback=parent.llm_feedback,
        )

        inspirations_text = self._format_inspirations(context, selection.inspiration_ids)

        prompt = prompt_registry.get(
            "generation/mutation.txt",
            language=context.language,
            inspirations=inspirations_text,
            parent_program=parent_program,
        )

        self.log_info(f"No errors in {parent.id}, using mutation prompt")
        return prompt

    def _format_error_details(self, program: Program) -> str:
        """Format error information for the prompt."""
        error_info = program.normalized_error_info

        if isinstance(error_info, dict):
            parts = []
            if "error_type" in error_info:
                parts.append(f"Error Type: {error_info['error_type']}")
            if "error_message" in error_info:
                parts.append(f"Message: {error_info['error_message']}")
            if "traceback" in error_info:
                parts.append(f"Traceback:\n{error_info['traceback']}")
            if "stderr" in error_info:
                parts.append(f"Stderr:\n{error_info['stderr']}")
            return "\n".join(parts) if parts else str(error_info)
        else:
            return str(error_info)

    def _get_working_examples(
        self, context: Context, inspiration_ids: List[str]
    ) -> str:
        """Get working programs from inspirations as examples."""
        working = []

        for prog_id in inspiration_ids:
            program = context.get_program_by_id(prog_id)
            if program and not program.is_buggy:
                working.append(program)
                if len(working) >= self.max_working_examples:
                    break

        if not working:
            return ""

        formatted = []
        for i, prog in enumerate(working):
            program_text = prompt_registry.get(
                "base/program.txt",
                language=prog.language,
                program_code=prog.code,
                combined_score=prog.combined_score,
                metrics=prog.metrics,
            )
            formatted.append(f"## Working Example {i + 1}\n{program_text}")

        return "\n\n".join(formatted)

    def _format_inspirations(
        self, context: Context, inspiration_ids: Optional[List[str]]
    ) -> str:
        """Format inspirations for mutation fallback."""
        if not inspiration_ids:
            return ""

        inspirations = [context.get_program_by_id(id) for id in inspiration_ids]
        inspirations = [p for p in inspirations if p is not None]

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
                f"## Program {i} (Generation {prog.generation})\n{program_text}"
            )

        return "\n".join(formatted)

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program was generated."""
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: Failed to generate a program. "
                "Check LLM client and prompt configuration."
            )
