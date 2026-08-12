"""
ID generation and seeding utilities for Famou 2.0.

Provides functions for generating unique program IDs, managing random seeds
for reproducibility, and timestamp utilities.
"""

import random
import time
import uuid
from datetime import datetime, timezone
from typing import Optional


# Global seed for reproducibility
_GLOBAL_SEED: Optional[int] = None


def set_global_seed(seed: int = 42) -> None:
    """
    Set global seed for reproducibility.

    This sets the seed for:
    - Python's random module
    - UUID generation (via random)

    Args:
        seed: The seed value (default: 42, inspired by openevolve)
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    random.seed(seed)


def get_global_seed() -> Optional[int]:
    """Get the current global seed."""
    return _GLOBAL_SEED


def generate_short_uuid() -> str:
    """
    Generate a short UUID (8 characters).

    Returns:
        8-character hex string
    """
    return uuid.uuid4().hex[:8]


def generate_initial_program_id(index: int = 0) -> str:
    """Generate the canonical ID for an initial program."""
    return "init" if index == 0 else f"init_{index}"


def is_initial_program_id(program_id: str) -> bool:
    """Whether a program ID represents an initial user-provided seed."""
    return program_id == "init" or program_id.startswith("init_")


def generate_island_initial_program_id(base_id: str, island_id: int) -> str:
    """Generate the island-specific ID for an initial program.

    Appends ``_island_{island_id}`` to the base init ID so that island
    membership is directly encoded in the program ID.

    Args:
        base_id: The original init program ID (e.g. "init", "init_1")
        island_id: The island this program is assigned to

    Returns:
        Island-specific ID, e.g. "init_island_0", "init_1_island_2"

    Example:
        >>> generate_island_initial_program_id("init", 0)
        'init_island_0'
        >>> generate_island_initial_program_id("init_1", 2)
        'init_1_island_2'
    """
    return f"{base_id}_island_{island_id}"


def generate_program_id(
    generation: int,
    iteration: int,
    island_id: int = 0,
    created_at: Optional[float] = None,
) -> str:
    """
    Generate a unique program ID in the format:
    {iteration}_{island}_{generation}_{YYYYMMDDHHmm}.

    Args:
        generation: The genealogical generation (0=seed, 1=child, etc.)
        iteration: The iteration/round when this program was created
        island_id: Island where this program was created
        created_at: Optional unix timestamp for deterministic ID formatting

    Returns:
        Unique program ID string

    Example:
        >>> generate_program_id(1, 7, island_id=3, created_at=1712304300.0)
        '7_3_1_202404050805'
    """
    if created_at is None:
        created_at = time.time()
    timestamp = datetime.fromtimestamp(created_at).strftime("%Y%m%d%H%M")
    return f"{iteration}_{island_id}_{generation}_{timestamp}"


def generate_experiment_id(name: Optional[str] = None) -> str:
    """
    Generate a unique experiment ID with datetime for readability.

    Format: {name}_{YYYYMMDD_HHMMSS}_{uuid4}

    The datetime prefix makes experiments:
    - Human-readable: You can tell when an experiment was created
    - Sortable: Experiments sort chronologically in file browsers
    - Unique: The UUID suffix guarantees uniqueness even if same second

    Args:
        name: Optional experiment name to include in ID

    Returns:
        Unique experiment ID

    Example:
        >>> generate_experiment_id("circle_packing")
        'circle_packing_20240115_143022_a1b2'
        >>> generate_experiment_id()
        'exp_20240115_143022_a1b2'
    """
    from datetime import datetime, timezone

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_id = generate_short_uuid()[:4]  # 4 chars for uniqueness

    if name:
        # Sanitize name (replace spaces with underscores, lowercase)
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        return f"{safe_name}_{timestamp}_{short_id}"
    return f"exp_{timestamp}_{short_id}"


def generate_rollout_id(
    experiment_id: str,
    iteration: int,
    island_id: Optional[int] = None,
    attempt: int = 1,
) -> str:
    """
    Generate a rollout ID.

    Args:
        experiment_id: The experiment this rollout belongs to
        iteration: The iteration number (global iteration counter)
        island_id: Optional island ID for multi-island experiments

    Returns:
        Rollout ID string

    Example:
        >>> generate_rollout_id("exp_12345678", 3)
        'exp_12345678_rollout_3'
        >>> generate_rollout_id("exp_12345678", 3, island_id=1)
        'exp_12345678_island_1_rollout_3'
    """
    if island_id is not None:
        rollout_id = f"{experiment_id}_island_{island_id}_rollout_{iteration}"
    else:
        rollout_id = f"{experiment_id}_rollout_{iteration}"

    if attempt > 1:
        rollout_id = f"{rollout_id}_attempt_{attempt}"
    return rollout_id


def get_timestamp() -> float:
    """
    Get current UTC timestamp.

    Returns:
        Unix timestamp (seconds since epoch)
    """
    return time.time()


def timestamp_to_str(timestamp: float, format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Convert timestamp to human-readable string.

    Args:
        timestamp: Unix timestamp
        format: strftime format string

    Returns:
        Formatted timestamp string
    """
    return time.strftime(format, time.gmtime(timestamp))


# Initialize with default seed on module import
# set_global_seed(42)
