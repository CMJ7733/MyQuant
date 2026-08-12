"""
State accessors for controlled access to evolution state.

This module provides read-only accessor classes for querying island populations.
These accessors:
- Are scoped to a single island (created per-rollout as snapshot)
- Provide O(1) ID lookups via indexing
- Immutable snapshot semantics (thread-safe by design)
- Prevent accidental mutations (modules should read, Controller should write)
- Offer convenient query methods (get_top_k, get_valid, etc.)
"""

from __future__ import annotations

from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Any,
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from famou.core.data import Program


class PopulationAccessor:
    """
    Read-only accessor for a SINGLE island's population snapshot.

    Provides a unified interface for accessing programs in the population
    without risk of modifying the original population. The accessor is
    created per-rollout and contains an immutable snapshot of the population
    at the time the rollout started.

    Programs returned by this accessor are shared references used across
    archive and island populations. Do not modify programs retrieved from
    this accessor.

    Features:
    - O(1) program lookups by ID via internal index
    - Immutable snapshot semantics (thread-safe by design)
    - Convenient query methods (get_top_k, get_valid, get_by_generation)
    - Iterator support for easy iteration over programs

    Usage in modules:
        >>> # Get program by ID (O(1))
        >>> program = accessor.get_by_id("prog_123")
        >>>
        >>> # Get top-k programs by score (no island_id needed - uses current island)
        >>> top_5 = accessor.get_top_k(k=5)
        >>>
        >>> # Get all valid programs
        >>> valid = accessor.get_valid(min_validity=1.0)
        >>>
        >>> # Get all programs from a specific bucket
        >>> cluster_0 = accessor.get_bucket("cluster_0")
        >>>
        >>> # Check which island this accessor is for
        >>> print(f"Current island: {accessor.island_id}")

    Note:
        This accessor is READ-ONLY and immutable after construction.
        It contains a snapshot of the population at the time it was created.
        Population updates are handled by the Controller through Experiment.island_populations.
    """

    def __init__(
        self,
        island_id: int,
        island_data: Dict[str, List["Program"]],
    ):
        """
        Initialize accessor with snapshot of island data.

        Args:
            island_id: Island ID this accessor is scoped to
            island_data: Snapshot of island population data
                        Format: {bucket_id: [programs]}

        Note:
            The accessor stores a reference to the snapshot data.
            It is immutable after construction - no locks or rebuilds needed.
        """
        self._island_id = island_id
        self._buckets = island_data  # Store the snapshot
        self._id_index: Dict[str, "Program"] = {}

        # Build index once during initialization
        for bucket_progs in self._buckets.values():
            for program in bucket_progs:
                self._id_index[program.id] = program

    @property
    def island_id(self) -> int:
        """Get the island ID this accessor is scoped to."""
        return self._island_id

    # ==========================================================================
    # Internal Methods
    # ==========================================================================

    def _get_island_data(self) -> Dict[str, List["Program"]]:
        """Get this island's population data (snapshot)."""
        return self._buckets

    # ==========================================================================
    # Transient Registration (for debug loop intermediate programs)
    # ==========================================================================

    def register_transient(self, program: "Program") -> None:
        """
        Register a transient program for ID-based lookup.

        Used by debug loops to make intermediate programs (not yet in population)
        findable via get_by_id(). Does NOT add to any bucket or affect
        population queries (get_top_k, get_all, etc.).

        Args:
            program: Program to register for lookup
        """
        self._id_index[program.id] = program

    # ==========================================================================
    # ID-based Queries (O(1) via index)
    # ==========================================================================

    def get_by_id(self, program_id: str) -> Optional["Program"]:
        """
        Get program by ID (O(1) lookup via index).

        Returns a shared reference - the same Program object used in archive
        and other island populations. Do not modify the returned program.

        Args:
            program_id: Program identifier

        Returns:
            Program if found, None otherwise

        Example:
            >>> program = accessor.get_by_id("0_0_a1b2c3d4")
            >>> if program:
            ...     print(f"Score: {program.combined_score}")
        """
        return self._id_index.get(program_id)

    def contains(self, program_id: str) -> bool:
        """
        Check if a program ID exists in this island.

        Args:
            program_id: Program identifier to check

        Returns:
            True if program exists in this island's population, False otherwise

        Example:
            >>> if accessor.contains("prog_123"):
            ...     print("Program is in population")
        """
        return program_id in self._id_index

    # ==========================================================================
    # Island-level Queries
    # ==========================================================================

    def get_all(self) -> List["Program"]:
        """
        Get all programs in this island (flattened across buckets).

        Returns:
            List of all programs in this island (flattened across buckets)

        Example:
            >>> all_progs = accessor.get_all()
            >>> print(f"Total programs: {len(all_progs)}")
        """
        island_pop = self._get_island_data()
        programs = []
        for bucket_progs in island_pop.values():
            programs.extend(bucket_progs)
        return programs

    def get_bucket(self, bucket_id: str) -> List["Program"]:
        """
        Get programs from a specific bucket.

        Args:
            bucket_id: Bucket identifier (e.g., "population", "cluster_0")

        Returns:
            List of programs in the bucket (empty list if not found)

        Example:
            >>> # Get cluster 0 programs
            >>> cluster_progs = accessor.get_bucket("cluster_0")
        """
        return list(self._get_island_data().get(bucket_id, []))

    def get_all_buckets(self) -> Dict[str, List["Program"]]:
        """
        Get all buckets in this island as a dictionary.

        This is a convenience method for modules that need to work with
        the bucket structure directly (e.g., ClusterAdaptiveSelect).

        Returns:
            Dictionary mapping bucket_id to list of programs in that bucket.
            Returns a copy to prevent accidental mutation of the original data.

        Example:
            >>> # Get all buckets (e.g., for cluster-based selection)
            >>> buckets = accessor.get_all_buckets()
            >>> for bucket_id, programs in buckets.items():
            ...     print(f"{bucket_id}: {len(programs)} programs")
        """
        island_data = self._get_island_data()
        return {bid: list(progs) for bid, progs in island_data.items()}

    def count(self) -> int:
        """
        Count total programs in this island.

        Returns:
            Total number of programs across all buckets in this island

        Example:
            >>> count = accessor.count()
            >>> print(f"Island has {count} programs")
        """
        return len(self.get_all())

    # ==========================================================================
    # Filtered Queries
    # ==========================================================================

    def get_top_k(
        self,
        k: int,
        key: Optional[Callable[["Program"], float]] = None,
    ) -> List["Program"]:
        """
        Get top-k programs by score (or custom key) from this island.

        Args:
            k: Number of programs to return
            key: Optional scoring function (default: combined_score descending)

        Returns:
            Top-k programs sorted by score descending

        Example:
            >>> # Get top 5 by combined_score
            >>> top_5 = accessor.get_top_k(k=5)
            >>>
            >>> # Get top 10 by validity
            >>> top_10 = accessor.get_top_k(
            ...     k=10,
            ...     key=lambda p: p.validity or 0.0
            ... )
        """
        if key is None:
            key = lambda p: p.combined_score if p.combined_score is not None else float("-inf")

        programs = self.get_all()
        sorted_progs = sorted(programs, key=key, reverse=True)
        return sorted_progs[:k]

    def get_valid(
        self,
        min_validity: float = 1.0,
    ) -> List["Program"]:
        """
        Get programs with validity >= threshold from this island.

        Args:
            min_validity: Minimum validity threshold (0.0 to 1.0)

        Returns:
            List of programs with validity >= min_validity

        Example:
            >>> # Get all fully valid programs
            >>> valid = accessor.get_valid(min_validity=1.0)
            >>>
            >>> # Get programs with validity >= 0.5
            >>> semi_valid = accessor.get_valid(min_validity=0.5)
        """
        programs = self.get_all()
        return [
            p
            for p in programs
            if p.validity is not None and p.validity >= min_validity
        ]

    def get_by_generation(
        self,
        generation: int,
    ) -> List["Program"]:
        """
        Get all programs of a specific generation from this island.

        Args:
            generation: Generation number (0=seed, 1=first gen, etc.)

        Returns:
            List of programs from the specified generation

        Example:
            >>> # Get all seed programs (generation 0)
            >>> seeds = accessor.get_by_generation(generation=0)
        """
        programs = self.get_all()
        return [p for p in programs if p.generation == generation]

    def get_by_iteration(
        self,
        iteration: int,
    ) -> List["Program"]:
        """
        Get all programs created at a specific iteration from this island.

        Args:
            iteration: Iteration number

        Returns:
            List of programs created at the specified iteration

        Example:
            >>> # Get all programs created in iteration 5
            >>> iter_5_progs = accessor.get_by_iteration(iteration=5)
        """
        programs = self.get_all()
        return [p for p in programs if p.iteration == iteration]

    # ==========================================================================
    # Metadata Queries
    # ==========================================================================

    def bucket_ids(self) -> List[str]:
        """
        Get list of bucket IDs for this island.

        Returns:
            List of bucket IDs in this island (empty list if island is empty)

        Example:
            >>> buckets = accessor.bucket_ids()
            >>> print(f"Buckets: {buckets}")  # e.g., ["population", "cluster_0", "cluster_1"]
        """
        return list(self._get_island_data().keys())

    # ==========================================================================
    # Iteration and Collection Interface
    # ==========================================================================

    def __iter__(self) -> Iterator["Program"]:
        """
        Iterate over all programs in this island.

        Example:
            >>> for program in accessor:
            ...     print(f"{program.id}: {program.combined_score}")
        """
        for program in self._id_index.values():
            yield program

    def __len__(self) -> int:
        """
        Total program count in this island.

        Example:
            >>> total = len(accessor)
            >>> print(f"Total programs: {total}")
        """
        return len(self._id_index)

    def __contains__(self, program_id: str) -> bool:
        """
        Check if program ID exists in this island (supports 'in' operator).

        Example:
            >>> if "prog_123" in accessor:
            ...     print("Found!")
        """
        return program_id in self._id_index

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return f"PopulationAccessor(island={self._island_id}, programs={len(self._id_index)})"


class IslandAccessor:
    """
    Read-only accessor for island-visible programs with lineage tracking.

    Provides island-scoped access to programs, enforcing visibility boundaries
    defined by island_index. This prevents modules from accessing programs
    outside their assigned island, ensuring proper island isolation.

    Programs returned by this accessor are shared references from the global
    archive. Do not modify programs retrieved from this accessor.

    Features:
    - O(1) program lookups by ID (uses archive dict directly)
    - Visibility filtering (only sees programs in island_index)
    - Lineage tracking within island boundaries (parent/children/ancestors/descendants)
    - Immutable snapshot semantics (thread-safe by design)

    Usage in modules:
        >>> # Get program by ID (only if visible in this island)
        >>> program = island_accessor.get_by_id("prog_123")
        >>>
        >>> # Get top-k programs in this island
        >>> top_5 = island_accessor.get_top_k(5)
        >>>
        >>> # Get parent (only if parent is visible in this island)
        >>> parent = island_accessor.get_parent("prog_123")
        >>>
        >>> # Get ancestors (stops when ancestor not visible)
        >>> lineage = island_accessor.get_ancestors("prog_123")
    """

    def __init__(
        self,
        island_id: int,
        visible_ids: Set[str],
        archive: Dict[str, "Program"]
    ):
        """
        Initialize accessor with island's visible program IDs.

        Args:
            island_id: Island this accessor is scoped to
            visible_ids: Set of program IDs visible to this island (from island_index)
            archive: Global archive dict (ID -> Program mapping)

        Note:
            The accessor stores references to visible_ids and archive.
            It is immutable after construction for thread safety.
        """
        self.island_id = island_id
        self._visible_ids = visible_ids
        self._archive = archive

        # Build children index for lineage queries (only visible programs)
        self._children_index: Dict[str, List[str]] = {}
        for pid in visible_ids:
            program = archive.get(pid)
            if program and program.parent_id:
                if program.parent_id not in self._children_index:
                    self._children_index[program.parent_id] = []
                self._children_index[program.parent_id].append(pid)

    # ==========================================================================
    # Core Queries (O(1) via archive dict with visibility check)
    # ==========================================================================

    def get_by_id(self, program_id: str) -> Optional["Program"]:
        """
        Get program by ID if visible in this island (O(1)).

        Returns a shared reference from archive. Do not modify the returned program.

        Args:
            program_id: Program identifier

        Returns:
            Program if found and visible, None otherwise

        Example:
            >>> program = island_accessor.get_by_id("prog_123")
            >>> if program:
            ...     print(f"Score: {program.combined_score}")
        """
        if program_id not in self._visible_ids:
            return None
        return self._archive.get(program_id)

    def contains(self, program_id: str) -> bool:
        """
        Check if program is visible in this island.

        Args:
            program_id: Program identifier to check

        Returns:
            True if program exists and is visible in this island, False otherwise

        Example:
            >>> if island_accessor.contains("prog_123"):
            ...     print("Program is visible in this island")
        """
        return program_id in self._visible_ids

    def get_all(self) -> List["Program"]:
        """
        Get all programs visible in this island.

        Returns:
            List of all programs visible in this island

        Example:
            >>> all_progs = island_accessor.get_all()
            >>> print(f"Total visible programs: {len(all_progs)}")
        """
        return [self._archive[pid] for pid in self._visible_ids if pid in self._archive]

    # ==========================================================================
    # Filtering Queries
    # ==========================================================================

    def get_top_k(
        self,
        k: int,
        key: Callable[["Program"], float] = lambda p: p.combined_score or 0.0
    ) -> List["Program"]:
        """
        Get top-k programs by custom key (default: combined_score).

        Only considers programs visible in this island.

        Args:
            k: Number of top programs to return
            key: Function to extract sort key from program (default: combined_score)

        Returns:
            List of top-k programs, sorted by key in descending order

        Example:
            >>> # Get top 5 by score
            >>> top_5 = island_accessor.get_top_k(5)
            >>>
            >>> # Get top 3 by validity
            >>> top_valid = island_accessor.get_top_k(3, key=lambda p: p.validity or 0.0)
        """
        programs = self.get_all()
        # Filter programs where key is not None
        scored = [p for p in programs if key(p) is not None]
        return sorted(scored, key=key, reverse=True)[:k]

    def get_valid(self, min_validity: float = 1.0) -> List["Program"]:
        """
        Get all valid programs in this island.

        Args:
            min_validity: Minimum validity score (default: 1.0 for fully valid)

        Returns:
            List of programs with validity >= min_validity

        Example:
            >>> valid_progs = island_accessor.get_valid(min_validity=1.0)
        """
        programs = self.get_all()
        return [
            p for p in programs
            if p.validity is not None and p.validity >= min_validity
        ]

    def get_by_generation(self, generation: int) -> List["Program"]:
        """
        Get all programs at specific generation in this island.

        Args:
            generation: Generation number (0 for seeds, 1+ for evolved)

        Returns:
            List of programs at the specified generation

        Example:
            >>> gen_0 = island_accessor.get_by_generation(0)  # Seed programs
        """
        programs = self.get_all()
        return [p for p in programs if p.generation == generation]

    def get_by_iteration(self, iteration: int) -> List["Program"]:
        """
        Get all programs created at specific iteration in this island.

        Args:
            iteration: Iteration number

        Returns:
            List of programs created at the specified iteration

        Example:
            >>> iter_10 = island_accessor.get_by_iteration(10)
        """
        programs = self.get_all()
        return [p for p in programs if p.iteration == iteration]

    # ==========================================================================
    # Lineage Queries (filtered by island visibility)
    # ==========================================================================

    def get_parent(self, program_id: str) -> Optional["Program"]:
        """
        Get parent program if visible in this island.

        Returns None if:
        - Program not found
        - Program has no parent
        - Parent not visible in this island (outside island boundary)

        Args:
            program_id: Program identifier

        Returns:
            Parent program if visible, None otherwise

        Example:
            >>> parent = island_accessor.get_parent("prog_123")
            >>> if parent:
            ...     print(f"Parent: {parent.id}")
            >>> else:
            ...     print("Parent not visible (or doesn't exist)")
        """
        program = self.get_by_id(program_id)
        if not program or not program.parent_id:
            return None

        # Only return parent if visible in this island
        if program.parent_id in self._visible_ids:
            return self._archive.get(program.parent_id)
        return None

    def get_children(self, program_id: str) -> List["Program"]:
        """
        Get all children of program that are visible in this island.

        Args:
            program_id: Program identifier

        Returns:
            List of children programs (only those visible in this island)

        Example:
            >>> children = island_accessor.get_children("init")
            >>> print(f"Visible children: {len(children)}")
        """
        if program_id not in self._visible_ids:
            return []

        child_ids = self._children_index.get(program_id, [])
        return [self._archive[cid] for cid in child_ids if cid in self._archive]

    def get_ancestors(self, program_id: str) -> List["Program"]:
        """
        Get chain of ancestors from root to immediate parent.

        Stops when ancestor not visible in island (island boundary).
        Returns list from oldest ancestor to immediate parent.

        Args:
            program_id: Program identifier

        Returns:
            List of ancestor programs (root to parent), within island boundaries

        Example:
            >>> lineage = island_accessor.get_ancestors("prog_123")
            >>> # Returns [init, prog_1, prog_5] if all visible
            >>> # May return [] or partial chain if ancestors outside island
        """
        ancestors = []
        current = self.get_by_id(program_id)

        while current and current.parent_id:
            parent = self.get_parent(current.id)
            if not parent:
                # Parent not visible - stop here (island boundary)
                break
            ancestors.append(parent)
            current = parent

        return list(reversed(ancestors))  # Root to immediate parent

    def get_descendants(self, program_id: str) -> List["Program"]:
        """
        Get all descendants (subtree) visible in this island.

        Uses BFS traversal to find all programs descended from the given program.
        Only includes descendants that are visible in this island.

        Args:
            program_id: Program identifier (root of subtree)

        Returns:
            List of all descendant programs visible in this island

        Example:
            >>> descendants = island_accessor.get_descendants("init")
            >>> print(f"Visible descendants: {len(descendants)}")
        """
        if program_id not in self._visible_ids:
            return []

        descendants = []
        queue = [program_id]
        visited = set()

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            children = self.get_children(current_id)
            descendants.extend(children)
            queue.extend(c.id for c in children)

        return descendants

    # ==========================================================================
    # Collection Interface
    # ==========================================================================

    def __iter__(self) -> Iterator["Program"]:
        """
        Iterate over all programs visible in this island.

        Example:
            >>> for program in island_accessor:
            ...     print(f"{program.id}: {program.combined_score}")
        """
        for pid in self._visible_ids:
            program = self._archive.get(pid)
            if program:
                yield program

    def __len__(self) -> int:
        """
        Total program count visible in this island.

        Example:
            >>> total = len(island_accessor)
            >>> print(f"Visible programs: {total}")
        """
        return len(self._visible_ids)

    def __contains__(self, program_id: str) -> bool:
        """
        Check if program visible (supports 'in' operator).

        Example:
            >>> if "prog_123" in island_accessor:
            ...     print("Visible!")
        """
        return program_id in self._visible_ids

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return f"IslandAccessor(island={self.island_id}, programs={len(self._visible_ids)})"


class StateAccessor:
    """
    Immutable snapshot of StateStore for a specific island.

    Created per-rollout by StateStore.create_accessor(). Provides read-only
    access to both global and island-scoped state at the point in time the
    snapshot was taken.

    Features:
    - No locks (data is deep-copied at creation time)
    - Pickle-safe (plain dicts + int, no threading primitives)
    - Thread-safe by immutability
    - island_id is bound at creation — get_island() needs no island_id arg

    API mirrors StateStore.get / StateStore.get_island for easy migration:
        old: context.state.get("current_phase", default="unknown")
        new: context.state.get("current_phase", default="unknown")

        old: context.state.get_island(context.island_id, "plans", default={})
        new: context.state.get_island("plans", default={})
    """

    __slots__ = ("_island_id", "_global", "_island")

    def __init__(
        self,
        island_id: int,
        global_snapshot: Dict[str, Any],
        island_snapshot: Dict[str, Any],
    ):
        object.__setattr__(self, "_island_id", island_id)
        object.__setattr__(self, "_global", global_snapshot)
        object.__setattr__(self, "_island", island_snapshot)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("StateAccessor is immutable")

    def __getstate__(self):
        return (self._island_id, self._global, self._island)

    def __setstate__(self, state):
        island_id, global_snap, island_snap = state
        object.__setattr__(self, "_island_id", island_id)
        object.__setattr__(self, "_global", global_snap)
        object.__setattr__(self, "_island", island_snap)

    @property
    def island_id(self) -> int:
        return self._island_id

    # ------------------------------------------------------------------
    # Global reads
    # ------------------------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        """
        Read a value from global state snapshot.

        Args:
            *keys: Nested key path (e.g., "current_phase")
            default: Fallback if path not found

        Returns:
            The value (deep-copied if mutable) or default
        """
        if not keys:
            raise ValueError("At least one key required")
        return self._get_nested(self._global, keys, default)

    # ------------------------------------------------------------------
    # Island reads (island_id already bound)
    # ------------------------------------------------------------------

    def get_island(self, *keys: str, default: Any = None) -> Any:
        """
        Read a value from this island's state snapshot.

        Unlike StateStore.get_island(), no island_id argument is needed —
        it was fixed when the accessor was created.

        Args:
            *keys: Nested key path (e.g., "plan_execution_chains")
            default: Fallback if path not found

        Returns:
            The value (deep-copied if mutable) or default
        """
        if not keys:
            raise ValueError("At least one key required")
        return self._get_nested(self._island, keys, default)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _get_nested(data: Dict, keys: tuple, default: Any) -> Any:
        """Walk nested dicts; return deep-copied mutable or scalar."""
        import copy as _copy

        value = data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        if isinstance(value, (dict, list)):
            return _copy.deepcopy(value)
        return value

    def __repr__(self) -> str:
        return (
            f"StateAccessor(island_id={self._island_id}, "
            f"global_keys={len(self._global)}, "
            f"island_keys={len(self._island)})"
        )

