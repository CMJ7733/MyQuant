"""
BlockTargetGenerate - Generation module for block-level code evolution.

Flow:
1. Read planner_action from SelectionData.extra
2. Build prompt based on action (modify/add/remove/adjust_params)
3. Call LLM to generate modified block
4. Parse BLOCK X START...END from response
5. Replace block in parent code
6. Create new Program with modified code

Key Design:
- LLM returns ONLY the modified block wrapped in # BLOCK X START ... # BLOCK X END
- BlockTargetGenerate parses and replaces in parent code
- Supports fallback to full code generation if block parsing fails
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.core.protocol import Module, RequiresLLM
from famou.infrastructure.llm.base import get_llm_max_tokens
from famou.modules.generate.base import GenerateModule
from famou.modules.planning.block_detection import (
    parse_blocks_with_spans,
    parse_child_blocks,
    replace_blocks,
)
from famou.utils.code_parser import evolve_code, extract_required_packages
from famou.prompts import prompt_registry
from famou.utils.id_gen import generate_program_id
from famou.utils.trace_utils import append_named_trace, build_llm_trace, set_generate_trace

if TYPE_CHECKING:
    from famou.infrastructure.llm.base import LLMClient

logger = logging.getLogger("famou")


# Operation to template mapping (only modify and adjust_params supported)
OPERATION_TEMPLATES = {
    "modify": "planning/modify_block.txt",
    "adjust_params": "planning/adjust_params.txt",
}


class BlockTargetGenerate(GenerateModule):
    """
    Generation module for block-level code evolution.

    Inherits from GenerateModule to be recognized in Rollout pipeline validation.

    Pipeline Role:
    - Reads: result.selection (parent_id, planner_action)
    - Writes: result.generated_program

    Key Features:
    - Operation-specific prompts (modify/add/remove/adjust_params)
    - Block-only output parsing (# BLOCK X START ... # BLOCK X END)
    - Automatic block replacement in parent code
    - Inspiration program inclusion
    - Fallback to full code if block parsing fails

    Configuration:
        include_inspirations: Whether to include inspiration code in prompt
        temperature: LLM temperature for generation
        max_tokens: Maximum tokens for LLM response
        fallback_to_full: Fallback to full code if block parsing fails

    Example:
        >>> generate = BlockTargetGenerate(
        ...     include_inspirations=True,
        ...     temperature=0.8,
        ... )
        >>> result = generate(context, result)
        >>> # result.generated_program contains modified code
    """

    def __init__(
        self,
        include_inspirations: bool = True,
        temperature: float = 0.8,
        max_tokens: int = 4096,
        fallback_to_full: bool = True,
        custom_templates: Optional[Dict[str, str]] = None,
        name: Optional[str] = None,
        **config,
    ):
        """
        Initialize BlockTargetGenerate.

        Args:
            include_inspirations: Include inspiration code in prompt
            temperature: LLM temperature
            max_tokens: Maximum tokens for response
            fallback_to_full: Fallback to full code if block parsing fails
            custom_templates: Custom prompt templates (keyed by operation)
            name: Module name
            **config: Additional configuration
        """
        # Call GenerateModule's __init__ with temperature, max_tokens, and shared
        # generate config such as write_mode.
        super().__init__(
            name=name,
            temperature=temperature,
            max_tokens=max_tokens,
            **config,
        )

        self.include_inspirations = include_inspirations
        self.fallback_to_full = fallback_to_full

        # Custom templates override registry templates
        self.custom_templates = custom_templates or {}

    # -------------------------------------------------------------------------
    # Prompt Building (Abstract Method Implementation)
    # -------------------------------------------------------------------------

    def build_prompt(self, context: Context, selection: SelectionData) -> str:
        """
        Build generation prompt from selection data.

        Required by GenerateModule abstract base class.
        This method extracts info from selection and delegates to _build_prompt.

        Args:
            context: Experiment context
            selection: SelectionData with parent_id, inspiration_ids, and planner_action

        Returns:
            Formatted prompt string
        """
        action = selection.extra.get("planner_action", {})
        language = context.language

        # Get parent program
        parent = context.get_program_by_id(selection.parent_id)
        if not parent:
            raise ValueError(f"Parent {selection.parent_id} not found")

        # Get inspiration programs
        inspirations = []
        if selection.inspiration_ids:
            for insp_id in selection.inspiration_ids:
                insp = context.get_program_by_id(insp_id)
                if insp:
                    inspirations.append(insp)

        return self._build_prompt(context, parent, action, inspirations, language)

    def _get_comment_prefix(self, language: str) -> str:
        """Get comment prefix for language."""
        lang = language.lower().strip()
        if lang in ("python", "py"):
            return "#"
        return "//"

    def _build_prompt(
        self,
        context: Context,
        parent: Program,
        action: Dict[str, Any],
        inspirations: List[Program],
        language: str,
    ) -> str:
        """
        Build prompt based on operation type.

        Args:
            context: Context with task description
            parent: Parent program
            action: Planner action {op, target}
            inspirations: Inspiration programs
            language: Programming language

        Returns:
            Formatted prompt string
        """
        op = action.get("op", "modify")
        target = action.get("target", "A")
        comment = self._get_comment_prefix(language)

        # Format inspirations
        inspirations_text = ""
        if self.include_inspirations and inspirations:
            insp_parts = []
            for i, insp in enumerate(inspirations, 1):
                insp_parts.append(f"INSPIRATION {i}:\n```\n{insp.code}\n```")
            inspirations_text = "REFERENCE CODE:\n" + "\n\n".join(insp_parts)

        # Check for custom template override
        if op in self.custom_templates:
            # Use custom inline template
            template = self.custom_templates[op]
            prompt = template.format(
                task_description=context.task_description,
                target_block=target,
                parent_code=parent.code,
                comment=comment,
                inspirations=inspirations_text,
                language=language,
            )
        else:
            # Use template from PromptRegistry
            template_name = OPERATION_TEMPLATES.get(op, OPERATION_TEMPLATES["modify"])
            prompt = prompt_registry.get(
                template_name,
                task_description=context.task_description,
                target_block=target,
                parent_code=parent.code,
                comment=comment,
                inspirations=inspirations_text,
                language=language,
            )

        return prompt

    # -------------------------------------------------------------------------
    # Code Processing
    # -------------------------------------------------------------------------

    def _process_response(
        self,
        response: str,
        parent: Program,
        action: Dict[str, Any],
        language: str,
    ) -> str:
        """
        Process LLM response and create modified code.

        Args:
            response: LLM response
            parent: Parent program
            action: Planner action
            language: Programming language

        Returns:
            Modified code

        Raises:
            ValueError: If block parsing fails and fallback is disabled
        """
        target = action.get("target", "A")

        # EVOLVE-BLOCK path: bridge LLM output format to evolve_code()
        if target == "EVOLVE":
            child_blocks = parse_child_blocks(response, language)
            if target in child_blocks:
                content = child_blocks["EVOLVE"]
                comment = self._get_comment_prefix(language)
                temp_child = f"{comment} EVOLVE-BLOCK-START\n{content}\n{comment} EVOLVE-BLOCK-END"
                new_code = evolve_code(temp_child, parent.code, language=language)
                logger.debug("Successfully replaced EVOLVE-BLOCK")
                return new_code

        # Parse child blocks from response
        child_blocks = parse_child_blocks(response, language)

        if target in child_blocks:
            # Got the target block - replace in parent
            parent_blocks = parse_blocks_with_spans(parent.code, language)

            if target in parent_blocks:
                new_code = replace_blocks(
                    parent.code,
                    parent_blocks,
                    {target: child_blocks[target]},
                    language,
                )
                logger.debug(f"Successfully replaced block {target}")
                return new_code
            else:
                logger.warning(
                    f"Target block {target} not found in parent code"
                )

        # Fallback: try to use response as-is (if it looks like complete code)
        if self.fallback_to_full:
            # Check if response contains code (simple heuristic)
            stripped = response.strip()
            if stripped.startswith("```"):
                # Extract from code block
                import re
                match = re.search(r"```(?:\w+)?\n(.*?)```", stripped, re.DOTALL)
                if match:
                    code = match.group(1).strip()
                    if code:
                        logger.warning(
                            f"Block {target} not found, using full code from response"
                        )
                        return code

            # Use as-is if it looks like code
            if "def " in stripped or "class " in stripped or "#" in stripped:
                logger.warning(
                    f"Block {target} not found, using response as full code"
                )
                return stripped

        raise ValueError(
            f"Failed to parse block {target} from LLM response. "
            f"Response head: {response[:200]}"
        )

    # -------------------------------------------------------------------------
    # Module Protocol
    # -------------------------------------------------------------------------

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that selection is available."""
        if not result.selection:
            raise ValueError(f"{self.name}: No selection available")
        if not result.selection.parent_id:
            raise ValueError(f"{self.name}: No parent_id in selection")
        if "planner_action" not in result.selection.extra:
            raise ValueError(f"{self.name}: No planner_action in selection.extra")

    def execute(
        self, context: Context, result: RolloutResult, **kwargs
    ) -> RolloutResult:
        """
        Execute block-targeted generation.

        Flow:
        1. Get parent and action from selection
        2. Build operation-specific prompt
        3. Call LLM
        4. Parse and replace block
        5. Create new Program

        Args:
            context: Context with task description
            result: RolloutResult with selection

        Returns:
            Updated RolloutResult with generated_program
        """
        selection = result.selection
        action = selection.extra.get("planner_action", {})
        language = context.language

        # Get parent program
        parent = context.get_program_by_id(selection.parent_id)
        if not parent:
            raise ValueError(
                f"{self.name}: Parent {selection.parent_id} not found"
            )

        # Get inspiration programs
        inspirations = []
        if selection.inspiration_ids:
            for insp_id in selection.inspiration_ids:
                insp = context.get_program_by_id(insp_id)
                if insp:
                    inspirations.append(insp)
        if self.logger:
            self.log_info(
                f"Generating: op={action.get('op')}, "
                f"target=BLOCK {action.get('target')}, parent={parent.id}"
            )

        # Build prompt
        prompt = self._build_prompt(
            context, parent, action, inspirations, language
        )

        llm_response = None
        try:
            # Call LLM
            llm_response = self.llm_client.generate(
                prompt=prompt,
                system=context.task_description,
                temperature=self.temperature,
                max_tokens=get_llm_max_tokens(self.llm_client),
            )
            response = llm_response.text
        except Exception as e:
            append_named_trace(
                parent,
                "generate_failures",
                build_llm_trace(
                    module_name=self.name,
                    system=context.task_description,
                    prompt=prompt,
                    response=llm_response,
                    request_extra={
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "parent_id": parent.id,
                        "selection": selection,
                        "planner_action": action,
                    },
                    error=e,
                ),
            )
            raise

        if self.logger:
            self.log_info(f"LLM response: {len(response)} chars")

        # Process response to get modified code
        try:
            new_code = self._process_response(
                response, parent, action, language
            )
        except ValueError as e:
            self.log_error(f"Failed to process response: {e}")
            raise

        required_packages = None
        if context.language == "python":
            required_packages = self._sanitize_required_packages(
                extract_required_packages(response)
            )

        # Create new Program
        created_at = time.time()
        new_program = Program(
            id=generate_program_id(
                parent.generation + 1,
                context.iteration,
                island_id=context.island_id,
                created_at=created_at,
            ),
            code=new_code,
            generation=parent.generation + 1,
            iteration=context.iteration,
            parent_id=parent.id,
            language=parent.language,
            file_extension=parent.file_extension,
            system_prompt=context.task_description,
            prompt=prompt,
            response=response,
            thinking=llm_response.thinking,
            required_packages=required_packages,
            meta={
                "experiment_id": context.experiment_id,
                "planner_action": action,
                "target_block": action.get("target"),
                "operation": action.get("op"),
            },
            created_at=created_at,
        )

        set_generate_trace(
            new_program,
            build_llm_trace(
                module_name=self.name,
                system=context.task_description,
                prompt=prompt,
                response=llm_response,
                request_extra={
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "parent_id": parent.id,
                    "selection": selection,
                    "planner_action": action,
                },
                parsed={
                    "program_id": new_program.id,
                    "parent_id": new_program.parent_id,
                    "operation": action.get("op"),
                    "target_block": action.get("target"),
                },
            ),
        )

        result.generated_program = new_program

        self.log_info(
            f"Generated program {new_program.id} "
            f"(gen={new_program.generation}, op={action.get('op')}, "
            f"target={action.get('target')})"
        )

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that program was generated."""
        if not result.generated_program:
            raise ValueError(f"{self.name}: Failed to generate program")
        if not result.generated_program.code:
            raise ValueError(f"{self.name}: Generated program has no code")
