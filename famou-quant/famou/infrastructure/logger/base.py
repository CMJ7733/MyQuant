"""
Base logger protocol for Famou 2.0.

Defines the minimal interface that all loggers must implement.

This protocol keeps the Logger interface minimal and reusable, with only
core logging methods. Framework-specific formatting logic is handled by
the calling code (e.g., Evolver has helper methods for formatting).
"""

from typing import Any, Dict, Optional, Protocol

from famou.core.data import Program


class Logger(Protocol):
    """
    Minimal protocol for loggers.

    All logger implementations must implement the four core logging methods:
    info(), warning(), error(), debug(). Additional utility methods like
    program() and stats() are optional but recommended for convenience.

    Design Philosophy:
    - Core methods handle basic logging with structured context
    - Utility methods (program, stats) are general-purpose, not framework-specific
    - Framework-specific formatting (e.g., iteration results) is handled by
      the calling code using helper methods
    - No logger-specific methods like iteration_result() or new_best_program()
      that would bloat the interface

    Example:
        >>> logger = LocalLogger(name="experiment")
        >>> logger.info("Starting iteration", iteration=1, experiment_id="exp_123")
        >>> logger.program(best_program)
        >>> logger.stats({"best_score": 0.85, "avg_score": 0.67})
    """

    # ==========================================================================
    # Core Logging Methods (Required)
    # ==========================================================================

    def info(self, message: str, **context: Any) -> None:
        """
        Log info-level message with optional structured context.

        Args:
            message: The message to log
            **context: Additional context (e.g., iteration=1, score=0.85)
        """
        ...

    def warning(self, message: str, system_only: bool = False, **context: Any) -> None:
        """
        Log warning-level message with optional structured context.

        Args:
            message: The message to log
            system_only: If True, skip writing to extra_log_dirs (user-visible dirs)
            **context: Additional context
        """
        ...

    def error(self, message: str, **context: Any) -> None:
        """
        Log error-level message with optional structured context.

        Args:
            message: The message to log
            **context: Additional context
        """
        ...

    def debug(self, message: str, **context: Any) -> None:
        """
        Log debug-level message with optional structured context.

        Args:
            message: The message to log
            **context: Additional context
        """
        ...

    # ==========================================================================
    # Utility Methods (Optional but Recommended)
    # ==========================================================================

    def program(self, program: Program, title: Optional[str] = None) -> None:
        """
        Display a Program object with its metadata.

        This is a general-purpose utility method for displaying program
        information. Implementations can format this however they like.

        Args:
            program: The program to display
            title: Optional custom title for the display
        """
        ...

    def stats(self, stats: Dict[str, Any], title: str = "Statistics") -> None:
        """
        Display statistics in a formatted way.

        This is a general-purpose utility method for displaying statistics.
        Implementations can format this however they like (table, JSON, etc).

        Args:
            stats: Dictionary of statistics to display
            title: Title for the stats display
        """
        ...
