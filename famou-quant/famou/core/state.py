"""
Thread-safe state store for Famou 2.0.

Provides persistent state management for stateful strategies that need to
track information across rollouts (e.g., component modification history,
cluster statistics, learned experiences).

Key features:
- Thread-safe with per-island locking
- Serializable for checkpoint persistence
- Supports both global and per-island state
- Atomic update operations for read-modify-write patterns
"""

from __future__ import annotations

import copy
import threading
from typing import Any, Callable, Dict, List, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from famou.core.accessors import StateAccessor
    from famou.core.data import StateUpdate

from pydantic import BaseModel, Field, PrivateAttr


T = TypeVar("T")


class StateStore(BaseModel):
    """
    Thread-safe central store for cross-module state.

    Supports two types of state:
    - Global: Shared across all islands (e.g., total iteration count)
    - Per-island: Isolated per island (e.g., component modification history)

    Thread Safety:
    - All operations are protected by locks
    - Global state uses a single lock
    - Per-island state uses per-island locks for better concurrency
    - Use update_* methods for atomic read-modify-write operations

    Persistence:
    - Inherits from Pydantic BaseModel for automatic serialization
    - Use model_dump() to serialize, model_validate() to deserialize
    - Private lock attributes are excluded from serialization
    """
    # Serializable state
    global_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global state shared across all islands"
    )
    island_state: Dict[int, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-island state: {island_id: {key: value}}"
    )

    # Private lock attributes (not serialized)
    _global_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _island_locks: Dict[int, threading.Lock] = PrivateAttr(default_factory=dict)
    _locks_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    model_config = {"validate_assignment": True}

    def __init__(self, **data):
        """Initialize StateRegistry with optional pre-loaded state."""
        super().__init__(**data)
        # Reinitialize locks (they're not serialized)
        self._global_lock = threading.Lock()
        self._island_locks = {}
        self._locks_lock = threading.Lock()

    def _get_island_lock(self, island_id: int) -> threading.Lock:
        """
        Get or create a lock for a specific island.

        Uses double-checked locking pattern for efficiency.
        """
        # Fast path: lock already exists
        if island_id in self._island_locks:
            return self._island_locks[island_id]

        # Slow path: create lock under protection
        with self._locks_lock:
            if island_id not in self._island_locks:
                self._island_locks[island_id] = threading.Lock()
            return self._island_locks[island_id]

    # =========================================================================
    # Global State Operations (support nested paths)
    # =========================================================================

    def _get_nested(self, data: Dict, keys: tuple, default: T) -> T:
        """Internal helper to get nested value from a dict."""
        value = data
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        # Deep copy mutable values to prevent external modification
        if isinstance(value, (dict, list)):
            return copy.deepcopy(value)
        return value

    def _set_nested(self, data: Dict, keys: tuple, value: Any) -> None:
        """Internal helper to set nested value in a dict, creating intermediate dicts."""
        # Deep copy mutable values
        if isinstance(value, (dict, list)):
            value = copy.deepcopy(value)

        # Navigate to parent, creating dicts as needed
        current = data
        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        # Set the final value
        current[keys[-1]] = value

    def _delete_nested(self, data: Dict, keys: tuple) -> bool:
        """Internal helper to delete nested value from a dict."""
        # Navigate to parent
        current = data
        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return False
            current = current[key]

        # Delete the final key
        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]
            return True
        return False

    def get(self, *keys: str, default: T = None) -> T:
        """
        Get a value from global state, supports nested paths.

        Args:
            *keys: State key path (e.g., "experiences", "strategy", "id")
            default: Default value if key not found

        Returns:
            The value, or default if not found

        Example:
            >>> state.get("counter")  # single key
            >>> state.get("experiences", "strategy", "id")  # nested path
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._global_lock:
            return self._get_nested(self.global_state, keys, default)

    def set(self, *keys: str, value: Any) -> None:
        """
        Set a value in global state, supports nested paths.

        Creates intermediate dicts as needed for nested paths.

        Args:
            *keys: State key path (e.g., "experiences", "strategy", "id")
            value: Value to store (keyword-only, will be deep copied if mutable)

        Example:
            >>> state.set("counter", value=0)  # single key
            >>> state.set("experiences", "strategy", "id", value=data)  # nested
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._global_lock:
            self._set_nested(self.global_state, keys, value)

    def update(
        self,
        *keys: str,
        updater: Callable[[T], T],
        default: T = None
    ) -> T:
        """
        Atomically update a global value, supports nested paths.

        Performs read-modify-write in a single lock acquisition,
        preventing race conditions.

        Args:
            *keys: State key path (e.g., "experiences", "strategy", "id")
            updater: Function that takes old value and returns new value
            default: Default value if key not found

        Returns:
            The new value after update

        Example:
            >>> state.update("counter", updater=lambda x: x + 1, default=0)
            >>> state.update("exp", "strat", "count", updater=lambda x: x + 1, default=0)
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._global_lock:
            old_value = self._get_nested(self.global_state, keys, default)
            new_value = updater(old_value)
            self._set_nested(self.global_state, keys, new_value)
            return new_value

    def delete(self, *keys: str) -> bool:
        """
        Delete a key from global state, supports nested paths.

        Args:
            *keys: State key path to delete

        Returns:
            True if key existed and was deleted, False otherwise

        Example:
            >>> state.delete("counter")  # single key
            >>> state.delete("experiences", "strategy", "id")  # nested
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._global_lock:
            return self._delete_nested(self.global_state, keys)

    # =========================================================================
    # Per-Island State Operations (support nested paths)
    # =========================================================================

    def get_island(self, island_id: int, *keys: str, default: T = None) -> T:
        """
        Get a value from per-island state, supports nested paths.

        Args:
            island_id: Island identifier
            *keys: State key path (e.g., "experiences", "strategy", "id")
            default: Default value if key not found

        Returns:
            The value, or default if not found

        Example:
            >>> state.get_island(0, "counter")  # single key
            >>> state.get_island(0, "exp", "strat", "id")  # nested path
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._get_island_lock(island_id):
            island_data = self.island_state.get(island_id, {})
            return self._get_nested(island_data, keys, default)

    def set_island(self, island_id: int, *keys: str, value: Any) -> None:
        """
        Set a value in per-island state, supports nested paths.

        Creates intermediate dicts as needed for nested paths.

        Args:
            island_id: Island identifier
            *keys: State key path (e.g., "experiences", "strategy", "id")
            value: Value to store (keyword-only, will be deep copied if mutable)

        Example:
            >>> state.set_island(0, "counter", value=0)  # single key
            >>> state.set_island(0, "exp", "strat", "id", value=data)  # nested
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._get_island_lock(island_id):
            if island_id not in self.island_state:
                self.island_state[island_id] = {}
            self._set_nested(self.island_state[island_id], keys, value)

    def update_island(
        self,
        island_id: int,
        *keys: str,
        updater: Callable[[T], T],
        default: T = None
    ) -> T:
        """
        Atomically update a per-island value, supports nested paths.

        Performs read-modify-write in a single lock acquisition,
        preventing race conditions between concurrent rollouts.

        Args:
            island_id: Island identifier
            *keys: State key path (e.g., "experiences", "strategy", "id")
            updater: Function that takes old value and returns new value
            default: Default value if key not found

        Returns:
            The new value after update

        Example:
            >>> # Atomic increment
            >>> state.update_island(0, "count", updater=lambda x: x + 1, default=0)

            >>> # Nested atomic update
            >>> state.update_island(0, "exp", "strat", "count",
            ...     updater=lambda x: x + 1, default=0)
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._get_island_lock(island_id):
            if island_id not in self.island_state:
                self.island_state[island_id] = {}

            old_value = self._get_nested(self.island_state[island_id], keys, default)
            new_value = updater(old_value)
            self._set_nested(self.island_state[island_id], keys, new_value)
            return new_value

    def delete_island(self, island_id: int, *keys: str) -> bool:
        """
        Delete a key from per-island state, supports nested paths.

        Args:
            island_id: Island identifier
            *keys: State key path to delete

        Returns:
            True if key existed and was deleted, False otherwise

        Example:
            >>> state.delete_island(0, "counter")  # single key
            >>> state.delete_island(0, "exp", "strat", "id")  # nested
        """
        if not keys:
            raise ValueError("At least one key required")

        with self._get_island_lock(island_id):
            if island_id not in self.island_state:
                return False
            return self._delete_nested(self.island_state[island_id], keys)

    def get_all_island_keys(self, island_id: int) -> list[str]:
        """
        Get all keys for a specific island.

        Args:
            island_id: Island identifier

        Returns:
            List of keys in the island's state
        """
        with self._get_island_lock(island_id):
            return list(self.island_state.get(island_id, {}).keys())

    def clear_island(self, island_id: int) -> None:
        """
        Clear all state for a specific island.

        Args:
            island_id: Island identifier
        """
        with self._get_island_lock(island_id):
            self.island_state[island_id] = {}

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_all(self) -> None:
        """Clear all state (global and per-island)."""
        with self._global_lock:
            self.global_state.clear()

        with self._locks_lock:
            for island_id in list(self.island_state.keys()):
                with self._get_island_lock(island_id):
                    self.island_state[island_id] = {}

    def get_island_ids(self) -> list[int]:
        """Get list of island IDs that have state."""
        with self._locks_lock:
            return list(self.island_state.keys())

    # =========================================================================
    # Snapshot & Batch Update
    # =========================================================================

    def create_accessor(self, island_id: int) -> "StateAccessor":
        """
        Create an immutable snapshot for a specific island.

        Deep copies both global state and the specified island's state,
        producing a StateAccessor that is lock-free, serialization-safe, and
        thread-safe by immutability.

        Args:
            island_id: Island to snapshot

        Returns:
            StateAccessor with frozen copies of global and island data
        """
        from famou.core.accessors import StateAccessor

        with self._global_lock:
            global_snapshot = copy.deepcopy(self.global_state)

        with self._get_island_lock(island_id):
            island_snapshot = copy.deepcopy(
                self.island_state.get(island_id, {})
            )

        return StateAccessor(
            island_id=island_id,
            global_snapshot=global_snapshot,
            island_snapshot=island_snapshot,
        )

    def apply_updates(self, updates: List["StateUpdate"]) -> None:
        """
        Batch-apply a list of StateUpdate objects.

        Intended to be called from the main thread (Evolver) after a rollout
        completes. Each update is applied in order using the existing
        set/set_island/delete/delete_island methods.

        Args:
            updates: Ordered list of StateUpdate mutations
        """
        for upd in updates:
            if upd.kind == "set_global":
                self.set(*upd.keys, value=upd.value)
            elif upd.kind == "set_island":
                self.set_island(upd.island_id, *upd.keys, value=upd.value)
            elif upd.kind == "delete_global":
                self.delete(*upd.keys)
            elif upd.kind == "delete_island":
                self.delete_island(upd.island_id, *upd.keys)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        num_global = len(self.global_state)
        num_islands = len(self.island_state)
        return f"StateStore(global_keys={num_global}, islands={num_islands})"
