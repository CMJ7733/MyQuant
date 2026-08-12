"""
Selection modules for Famou 2.0.

**SIMPLIFIED: One Program Per Rollout**

Implements strategies for selecting:
- Parent program: The program to modify/mutate
- Inspiration programs: Other programs that serve as examples/reference for the LLM
- Experience/knowledge (future): Additional selection data for advanced strategies
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.core.protocol import Module

if TYPE_CHECKING:
    pass


class SelectModule(Module):
    """
    Base class for selection strategies.

    Selection modules produce a SelectionData containing:
    - parent_id: The program to modify/mutate (str)
    - inspiration_ids: Other programs for LLM examples (List[str])
    - experiences: Experience IDs for reuse (Optional[List[str]])
    - knowledge: Knowledge IDs to apply (Optional[List[str]])
    - extra: Metadata from StateStore or other sources (Dict[str, Any])

    Design:
    - Subclasses implement select_parent() to return str (single program ID)
    - Subclasses can override select_inspirations() to return List[str] (program IDs)
    - Subclasses can override select_experiences() to return List[str] (experience IDs)
    - Subclasses can override select_knowledges() to return List[str] (knowledge IDs)
    - Subclasses can override select_state_metadata() to return Dict (state metadata)
    - Base class execute() handles the module protocol and calls all methods
    - Creates SelectionData and assigns to result.selection

    StateStore Integration:
    - Access context.state directly to retrieve/store data from StateStore
    - Use context.state.get(key, default) for global state
    - Use context.state.get_island(island_id, key, default) for island-specific state
    - Override select_state_metadata() to include state data in SelectionData.extra

    === DATA CONTRACT ===
    READS from Context:
        - context.accessor: PopulationAccessor (read-only population queries)
        - context.state: StateStore (optional, for stateful strategies)
        - context.island_id: Current island ID

    WRITES to RolloutResult:
        - result.selection: SelectionData (REQUIRED)
            - selection.parent_id (str): ID of program to modify
            - selection.inspiration_ids (List[str]): Reference programs
            - selection.experiences (Optional[List[str]]): Experience IDs
            - selection.knowledge (Optional[List[str]]): Knowledge IDs
            - selection.extra (Dict): Metadata from state or other sources

    Developer implements:
        Required:
        - select_parent() -> str

        Optional (override for advanced strategies):
        - select_inspirations() -> List[str]
        - select_experiences() -> Optional[List[str]]
        - select_knowledges() -> Optional[List[str]]
        - select_state_metadata() -> Optional[Dict[str, Any]]

    Framework handles:
        - Calling execute() and setting result.selection
        - Validation that selection was created
    """

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
                f"{self.name}: Cannot select from empty population. "
                "Make sure Experiment has programs in the current population."
            )

    @abstractmethod
    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select a single parent program from population.

        Args:
            context: Rollout context with population structure, diversity, iteration, etc.
            population: List of available programs (flattened from all buckets)

        Returns:
            Selected program ID (single string)
        """
        pass

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[str]]:
        """
        Select inspiration programs from population.

        Inspirations are high-quality programs that serve as examples/reference
        for the LLM during generation. They are NOT the program being modified.

        Default implementation selects top-K programs by score, excluding parent.
        Subclasses can override for different strategies.

        Args:
            context: Rollout context with population structure, diversity, iteration, etc.
            population: List of available programs (flattened from all buckets)
            parent_id: ID of the selected parent (excluded from inspirations)

        Returns:
            List of inspiration program IDs (may be empty)
        """
        return None

    def select_experiences(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[str]]:
        """
        Select experience IDs for reuse in generation.

        Experiences are previously learned patterns/solutions that can guide
        the LLM during generation. This is an advanced feature - override this
        method in subclasses to provide experience selection logic.

        Default implementation returns None (no experiences).

        Args:
            context: Rollout context with population structure, diversity, iteration, etc.
            population: List of available programs (flattened from all buckets)
            parent_id: ID of the selected parent

        Returns:
            List of experience IDs, or None if no experiences selected

        Example:
            >>> def select_experiences(self, context, population, parent_id):
            ...     # Use StateStore to retrieve experiences
            ...     return context.state.get_island(context.island_id, 'experiences', [])
        """
        return None

    def select_knowledges(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[str]]:
        """
        Select knowledge IDs to apply during generation.

        Knowledge represents domain-specific information, techniques, or
        strategies that should guide the LLM during generation. This is an
        advanced feature - override this method in subclasses to provide
        knowledge selection logic.

        Default implementation returns None (no knowledge).

        Args:
            context: Rollout context with population structure, diversity, iteration, etc.
            population: List of available programs (flattened from all buckets)
            parent_id: ID of the selected parent

        Returns:
            List of knowledge IDs, or None if no knowledge selected

        Example:
            >>> def select_knowledges(self, context, population, parent_id):
            ...     # Select knowledge based on task characteristics
            ...     return ['knowledge_debugging', 'knowledge_optimization']
        """
        return None

    def select_state_metadata(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Select arbitrary metadata from StateStore to include in SelectionData.

        This method allows subclasses to retrieve and include any custom data
        from the StateStore (global or island-specific) in the selection output.
        The returned dictionary will be stored in SelectionData.extra.

        Default implementation returns None (no state metadata).

        Args:
            context: Rollout context with access to state store
            population: List of available programs (flattened from all buckets)
            parent_id: ID of the selected parent

        Returns:
            Dictionary of metadata to include in SelectionData.extra, or None

        Example:
            >>> def select_state_metadata(self, context, population, parent_id):
            ...     # Retrieve adaptive parameters from state
            ...     if not context.state:
            ...         return None
            ...     return {
            ...         'temperature': context.state.get('adaptive_temperature', 1.0),
            ...         'exploration_bonus': context.state.get_island(
            ...             context.island_id, 'exploration_bonus', 0.0
            ...         ),
            ...         'phase': context.state.get('current_phase', 'explore')
            ...     }
        """
        return None


    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Execute selection logic.

        Reads population, calls all select_* methods,
        and creates SelectionData with the results.

        Creates result.selection with:
            - parent_id from select_parent()
            - inspiration_ids from select_inspirations()
            - experiences from select_experiences() (optional)
            - knowledge from select_knowledges() (optional)
            - extra from select_state_metadata() (optional)
        """
        # Get all programs from accessor
        all_programs = context.accessor.get_all()

        # Select parent (required)
        selected_id = self.select_parent(context, all_programs)

        # Select inspirations (optional)
        inspiration_ids = self.select_inspirations(context, all_programs, selected_id)

        # Select experiences (optional)
        experiences = self.select_experiences(context, all_programs, selected_id)

        # Select knowledge (optional)
        knowledge = self.select_knowledges(context, all_programs, selected_id)

        # Select state metadata (optional)
        state_metadata = self.select_state_metadata(context, all_programs, selected_id)

        # Create SelectionData and assign to result
        result.selection = SelectionData(
            parent_id=selected_id,
            inspiration_ids=inspiration_ids,
            experiences=experiences,
            knowledge=knowledge,
            extra=state_metadata if state_metadata else {},
        )

        self.log_info(f"Selected parent: {selected_id}")
        if inspiration_ids:
            self.log_info(f"Selected inspirations: {inspiration_ids}")
        if experiences:
            self.log_info(f"Selected experiences: {experiences}")
        if knowledge:
            self.log_info(f"Selected knowledge: {knowledge}")
        if state_metadata:
            self.log_info(f"Selected state metadata: {list(state_metadata.keys())}")

        return result
