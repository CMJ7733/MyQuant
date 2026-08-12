"""
Stderr filter to suppress unwanted messages.

This module provides utilities to filter out unwanted stderr messages,
such as MallocStackLogging warnings from wandb-core subprocesses.
"""

import sys
import os
import re
from typing import Optional, TextIO


class FilteredStderr:
    """
    A file-like object that filters stderr output.

    This suppresses unwanted messages like MallocStackLogging warnings
    from wandb-core subprocesses on macOS.

    Usage:
        >>> # Install filter
        >>> sys.stderr = FilteredStderr(sys.stderr)
        >>>
        >>> # All stderr writes will be filtered
        >>> print("error", file=sys.stderr)
    """

    # Patterns to filter out
    FILTER_PATTERNS = [
        re.compile(r'MallocStackLogging.*'),
        re.compile(r'wandb-core\(\d+\) MallocStackLogging.*'),
    ]

    def __init__(self, original_stderr: TextIO):
        """
        Initialize the filtered stderr.

        Args:
            original_stderr: The original stderr stream to wrap
        """
        self._original_stderr = original_stderr

    def write(self, text: str) -> int:
        """
        Write text to stderr (with filtering).

        Args:
            text: Text to write

        Returns:
            Number of characters written
        """
        if not text:
            return 0

        # Filter each line
        lines = text.split('\n')
        filtered_lines = [
            line for line in lines
            if not self._should_filter(line)
        ]
        filtered_text = '\n'.join(filtered_lines)

        # Write to original stderr
        if filtered_text:
            self._original_stderr.write(filtered_text)

        return len(text)

    def _should_filter(self, line: str) -> bool:
        """
        Check if a line should be filtered out.

        Args:
            line: Line to check

        Returns:
            True if line should be filtered, False otherwise
        """
        for pattern in self.FILTER_PATTERNS:
            if pattern.search(line):
                return True
        return False

    def flush(self) -> None:
        """Flush the stderr buffer."""
        self._original_stderr.flush()

    def __getattr__(self, name: str):
        """Delegate any other attributes to the original stderr."""
        return getattr(self._original_stderr, name)


def install_stderr_filter() -> None:
    """
    Install the stderr filter to suppress unwanted messages.

    This replaces sys.stderr with a filtered version that suppresses
    MallocStackLogging warnings from wandb-core subprocesses.

    Example:
        >>> from famou.infrastructure.monitor.stderr_filter import install_stderr_filter
        >>> install_stderr_filter()
        >>> main()
    """
    if not isinstance(sys.stderr, FilteredStderr):
        sys.stderr = FilteredStderr(sys.stderr)


def uninstall_stderr_filter() -> None:
    """
    Uninstall the stderr filter and restore original stderr.

    Example:
        >>> from famou.infrastructure.monitor.stderr_filter import uninstall_stderr_filter
        >>> main()
        >>> uninstall_stderr_filter()
    """
    if isinstance(sys.stderr, FilteredStderr):
        sys.stderr = sys.stderr._original_stderr


# Convenience alias for backward compatibility
StderrFilter = FilteredStderr
