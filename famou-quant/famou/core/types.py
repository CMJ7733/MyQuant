"""
Core type definitions for Famou 2.0.

Defines enums and type aliases used throughout the framework.
"""

from enum import Enum


class RolloutStatus(str, Enum):
    """
    Status of a rollout pipeline execution.

    Set by RolloutEngine based on module execution results.

    SUCCESS: All modules in the pipeline completed normally. The produced
             program itself may be buggy (validity=0, score=0) — that is a
             valid EvaluateModule output, NOT a failure.
    FAILED:  At least one module did not complete normally (generate failed,
             evaluate crashed, judge errored, etc.). The rollout is discarded
             entirely and the iteration is retried; iteration budget is NOT
             consumed.
    FATAL:   Framework or strategy bug; fail fast and stop the experiment.
    """

    SUCCESS = "success"
    FAILED = "failed"
    FATAL = "fatal"

    def __str__(self) -> str:
        return self.value


class ModuleValidationError(ValueError):
    """
    Exception raised when a module's validation fails.

    This is raised by Module.validate_input() and Module.validate_output()
    for logical validation errors that should NOT be retried:
    - Empty population when selection is required
    - Missing parent program for generation
    - Invalid program structure after generation

    Note: This inherits from ValueError but is explicitly excluded from
    retry logic in RolloutEngine. Other ValueErrors (like JSONDecodeError)
    will still be retried.
    """

    pass


class FatalRolloutError(RuntimeError):
    """Non-recoverable framework/strategy error that should stop evolution."""

    pass


class Language(str, Enum):
    """
    Supported programming languages for code evolution.

    Currently focused on Python, but designed for future extension.
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"

    def __str__(self) -> str:
        return self.value
