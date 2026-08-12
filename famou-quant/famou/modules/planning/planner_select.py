"""
PlannerBasedSelect - Selection module that uses Planner for action decisions.

Flow:
1. Delegate parent/inspiration selection to a composed SelectModule
2. Detect blocks from parent's code
3. Use Planner to propose action (op + target block)
4. Inject planner_action into SelectionData.extra for GenerateModule

Uses composition pattern:
- Holds a SelectModule instance for parent/inspiration selection
- Adds planning-specific logic (block detection + planner action) on top

Uses StateStore for:
- Caching detected blocks per program
- Storing/retrieving Planner state for persistence
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module, RequiresLLM
from famou.modules.planning.block_detection import detect_blocks_from_code
from famou.modules.planning.planners import BasePlanner, create_planner
from famou.modules.select.base import SelectModule
from famou.modules.select.elite import EliteSelect
from famou.modules.select.random import RandomSelect
from famou.modules.select.tournament import TournamentSelect

if TYPE_CHECKING:
    from famou.infrastructure.llm.base import LLMClient

logger = logging.getLogger("famou")


# Registry for creating SelectModule from string type
_SELECT_MODULE_REGISTRY = {
    "best": EliteSelect,
    "elite": EliteSelect,
    "random": RandomSelect,
    "tournament": TournamentSelect,
}


def _create_select_module(
    select_module: Optional[SelectModule] = None,
    selection_strategy: Optional[str] = None,
    num_inspirations: int = 2,
    select_module_config: Optional[Dict[str, Any]] = None,
) -> SelectModule:
    """
    Create a SelectModule instance from various input formats.

    Supports:
    1. Direct SelectModule instance (passthrough)
    2. String strategy name (backwards compatible)
    3. Config dict with type + params

    Args:
        select_module: Pre-built SelectModule instance
        selection_strategy: Strategy name for backwards compatibility
        num_inspirations: Number of inspirations (used with selection_strategy)
        select_module_config: Config dict with type + params

    Returns:
        SelectModule instance
    """
    # 1. Direct instance
    if select_module is not None:
        return select_module

    # 2. Config dict
    if select_module_config:
        module_type = select_module_config.pop("type", "elite")
        cls = _SELECT_MODULE_REGISTRY.get(module_type)
        if cls is None:
            raise ValueError(
                f"Unknown select module type: {module_type}. "
                f"Available: {list(_SELECT_MODULE_REGISTRY.keys())}"
            )
        return cls(**select_module_config)

    # 3. Backwards compatible: string strategy name
    strategy = selection_strategy or "best"
    cls = _SELECT_MODULE_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown selection strategy: {strategy}. "
            f"Available: {list(_SELECT_MODULE_REGISTRY.keys())}"
        )
    return cls(num_inspirations=num_inspirations)


class PlannerBasedSelect(Module, RequiresLLM):
    """
    Selection module that uses Planner to decide which block to modify.

    Uses composition: delegates parent/inspiration selection to a SelectModule,
    then adds planning-specific logic (block detection + planner action).

    Pipeline Role:
    - Reads: context.accessor (population), context.state (planner state)
    - Writes: result.selection with parent_id and extra["planner_action"]

    Key Features:
    - Composable SelectModule for parent/inspiration selection
    - Block detection from code (auto-detects based on language)
    - Planner-based action proposal (op + target)
    - StateStore integration for planner persistence

    Configuration:
        planner_type: Planner type ("llm_v2", "stochastic")
        select_module: SelectModule instance for parent selection
        selection_strategy: String shortcut ("best", "random", "tournament")
        default_blocks: Default blocks if detection fails
        block_pattern: Custom block detection regex (optional)

    Example:
        >>> # Using a SelectModule instance directly
        >>> select = PlannerBasedSelect(
        ...     planner_type="llm_v2",
        ...     select_module=TournamentSelect(tournament_size=5, num_inspirations=2),
        ... )

        >>> # Backwards compatible: string strategy
        >>> select = PlannerBasedSelect(
        ...     planner_type="stochastic",
        ...     selection_strategy="best",
        ...     num_inspirations=1,
        ... )
    """

    # LLM client (injected by RolloutEngine)
    llm_client: "LLMClient"

    def __init__(
        self,
        planner_type: str = "llm_v2",
        select_module: Optional[SelectModule] = None,
        selection_strategy: Optional[str] = None,
        num_inspirations: int = 2,
        select_module_config: Optional[Dict[str, Any]] = None,
        default_blocks: Optional[List[str]] = None,
        block_pattern: Optional[str] = None,
        planner_config: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **config,
    ):
        """
        Initialize PlannerBasedSelect.

        Args:
            planner_type: Planner type ("llm_v2", "stochastic")
            select_module: SelectModule instance for parent/inspiration selection
            selection_strategy: String shortcut for backwards compatibility
                ("best", "random", "tournament")
            num_inspirations: Number of inspirations (used with selection_strategy)
            select_module_config: Config dict to create SelectModule
                e.g. {"type": "tournament", "tournament_size": 5}
            default_blocks: Default blocks if detection fails
            block_pattern: Custom regex for block detection
            planner_config: Additional planner configuration
            name: Module name
            **config: Additional configuration
        """
        super().__init__(name=name, **config)

        self.planner_type = planner_type
        self.default_blocks = default_blocks or ["A", "B", "C"]
        self.block_pattern = block_pattern
        self.planner_config = planner_config or {}

        # Create SelectModule via composition
        self._select_module = _create_select_module(
            select_module=select_module,
            selection_strategy=selection_strategy,
            num_inspirations=num_inspirations,
            select_module_config=select_module_config,
        )

        # Planner instance (created lazily when llm_client is available)
        self._planner: Optional[BasePlanner] = None

    # -------------------------------------------------------------------------
    # Planner Management
    # -------------------------------------------------------------------------

    def _get_or_create_planner(self, context: Context) -> BasePlanner:
        """
        Get or create Planner instance.

        Uses StateStore to persist planner state across rollouts.

        Args:
            context: Context with state store

        Returns:
            Planner instance
        """
        if self._planner is not None:
            return self._planner

        # Create planner with LLM client
        planner_kwargs = {
            "arms": self.default_blocks,
            **self.planner_config,
        }

        if self.planner_type == "llm_v2":
            planner_kwargs["llm_client"] = self.llm_client

        self._planner = create_planner(self.planner_type, **planner_kwargs)

        # Restore state from StateStore if available
        if context.state:
            planner_state = context.state.get_island(
                "planner_state", default=None
            )
            if planner_state and hasattr(self._planner, "set_state"):
                self._planner.set_state(planner_state)
                logger.debug(f"Restored planner state for island {context.island_id}")

        return self._planner

    def _save_planner_state(self, context: Context, result: RolloutResult) -> None:
        """
        Save planner state via StateUpdateCollector.

        Args:
            context: Context with state accessor
            result: RolloutResult with state_updates collector
        """
        if self._planner is None:
            return

        if hasattr(self._planner, "get_state"):
            state = self._planner.get_state()
            if result.state_updates:
                result.state_updates.set_island("planner_state", value=state)
                logger.debug(f"Saved planner state for island {context.island_id}")

    # -------------------------------------------------------------------------
    # Block Detection
    # -------------------------------------------------------------------------

    def _detect_blocks(self, code: str, language: str, context: Context, result: RolloutResult = None) -> List[str]:
        """
        Detect blocks from code, using cache if available.

        Args:
            code: Source code
            language: Programming language
            context: Context with state accessor
            result: RolloutResult with state_updates collector (optional)

        Returns:
            List of block labels
        """
        # Try cache first (keyed by code hash)
        code_hash = str(hash(code))
        if context.state:
            cached = context.state.get_island(
                "block_cache", code_hash, default=None
            )
            if cached:
                return cached

        # Detect blocks
        blocks = detect_blocks_from_code(
            code,
            language=language,
            pattern=self.block_pattern,
            default_blocks=self.default_blocks,
        )

        # Cache result
        if result and result.state_updates and blocks:
            result.state_updates.set_island("block_cache", code_hash, value=blocks)

        return blocks

    # -------------------------------------------------------------------------
    # Module Protocol
    # -------------------------------------------------------------------------

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that population is available."""
        if not context.accessor:
            raise ValueError(
                f"{self.name}: Context has no accessor. "
                "Make sure Evolver creates Context with accessor."
            )
        all_programs = context.accessor.get_all()
        if not all_programs:
            raise ValueError(
                f"{self.name}: Cannot select from empty population."
            )

    def execute(
        self, context: Context, result: RolloutResult, **kwargs
    ) -> RolloutResult:
        """
        Execute selection with Planner action proposal.

        Flow:
        1. Delegate to SelectModule for parent/inspiration selection
        2. Detect blocks from parent's code
        3. Use Planner to propose action
        4. Inject planner_action into SelectionData.extra

        Args:
            context: Context with population and state
            result: RolloutResult to populate

        Returns:
            Updated RolloutResult
        """
        language = context.language

        # 1. Delegate to SelectModule for parent + inspiration selection
        result = self._select_module.execute(context, result)
        parent_id = result.selection.parent_id

        # Get parent program object (need code for block detection)
        parent = context.get_program_by_id(parent_id)
        if not parent:
            raise ValueError(f"{self.name}: Parent {parent_id} not found")

        if self.logger:
            self.log_info(
                f"[{self.name}] Selected parent: {parent_id} "
                f"(score={parent.combined_score})"
            )

        # 2. Detect blocks
        blocks = self._detect_blocks(parent.code, language, context, result)
        if self.logger:
            self.log_info(f"[{self.name}] Detected blocks: {blocks}")

        # 3. Get Planner and propose action
        planner = self._get_or_create_planner(context)

        # Compute best_score from population for planner context
        population = context.accessor.get_all()
        best_score = max(
            (p.combined_score or 0.0 for p in population), default=0.0
        )

        parent_score = parent.combined_score or 0.0
        iteration = context.iteration
        task_desc = context.task_description or ""

        if self.logger:
            self.log_info(
                f"[{self.name}] Planner context: parent_score={parent_score:.4f}, "
                f"best_score={best_score:.4f}, iteration={iteration}, "
                f"task_description={task_desc[:80]}{'...' if len(task_desc) > 80 else ''}"
            )

        action = planner.propose_action(
            parent_code=parent.code,
            pid=result.rollout_id,
            arms=blocks,
            # Additional context for LLMPlannerV2
            parent_id=parent_id,
            parent_score=parent_score,
            best_score=best_score,
            iteration=iteration,
            task_description=task_desc,
        )
        if self.logger:
            self.log_info(
                f"[{self.name}] Planner action: op={action.get('op')}, "
                f"target={action.get('target')}, "
                f"source={getattr(self._planner, 'last_decision_source', 'unknown')}"
            )

        # Save planner state
        self._save_planner_state(context, result)

        # 4. Inject planning-specific data into selection.extra
        planner_llm_info = {
            "prompt": getattr(self._planner, "last_prompt", None),
            "response": getattr(self._planner, "last_response", None),
            "trace": getattr(self._planner, "last_trace", None),
            "source": getattr(self._planner, "last_decision_source", "unknown"),
        }
        result.selection.extra.update({
            "planner_action": action,
            "planner_llm": planner_llm_info,
            "detected_blocks": blocks,
            "language": language,
        })

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that selection was created."""
        if not result.selection:
            raise ValueError(f"{self.name}: Failed to create selection")
        if not result.selection.parent_id:
            raise ValueError(f"{self.name}: No parent selected")
        if "planner_action" not in result.selection.extra:
            raise ValueError(f"{self.name}: No planner action in selection")
