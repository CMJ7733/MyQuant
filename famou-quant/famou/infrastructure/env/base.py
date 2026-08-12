"""
Base execution environment protocol.

Defines the interface for executing programs via user-provided evaluator functions.
"""

from typing import Any, Dict, List, Optional, Protocol


class ExecutionEnvironment(Protocol):
    """
    Protocol for execution environments.

    Execution environments run user programs via custom evaluator functions.
    Implementations can be local (subprocess), containerized (Docker),
    or remote (API-based).

    Example Usage:
        >>> from examples.circle_packing.evaluator import evaluate
        >>> env: ExecutionEnvironment = LocalEnv()
        >>> result = env.execute_with_evaluator(
        ...     program_code=my_program.code,
        ...     evaluate_fn=evaluate,
        ...     language="python",
        ...     timeout=600,
        ... )
        >>> print(result["combined_score"])
        0.95
    """

    def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn: Any,  # Callable[[str], Dict[str, Any]]
        language: str = "python",
        extension: str = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Execute a program using user-provided evaluator function.

        The evaluator function receives a program file path and returns
        evaluation results in flat format (all metrics as top-level keys).

        Args:
            program_code: Source code to execute
            evaluate_fn: User's evaluator function(program_path) -> results_dict
            language: Programming language (default: "python")
            timeout: Optional execution timeout (passed to evaluator if supported)

        Returns:
            Dictionary with evaluation results (flat format):
            {
                "combined_score": float,      # Required - overall fitness
                "validity": float,             # Required - 0.0-1.0 validity score
                "error_info": str,             # Optional - error details
                "metric1": value1,             # Optional - any custom metrics
                "metric2": value2,             # Optional - any custom metrics
            }

        Example:
            >>> def my_evaluator(program_path: str) -> Dict[str, Any]:
            ...     # Run program, validate, compute metrics
            ...     return {
            ...         "combined_score": 0.95,
            ...         "validity": 1.0,
            ...         "accuracy": 0.98,
            ...         "speed": 1.2,
            ...     }
            >>> env = LocalEnv()
            >>> result = env.execute_with_evaluator("def f(): pass", my_evaluator)
        """
        ...

    def prepare_dependencies(
        self,
        required_packages: List[str],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Prepare a reusable dependency environment for evaluator execution.

        Args:
            required_packages: Third-party packages to install
            timeout: Optional preparation timeout

        Returns:
            Dictionary describing the preparation result
        """
        ...
