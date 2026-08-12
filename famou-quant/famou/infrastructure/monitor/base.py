"""
Abstract interfaces and constants for monitoring backends.

Defines the MonitoringBackend interface that all monitoring implementations must follow,
along with constants for metric prefixes and log types.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional


class MetricPrefix:
    """Metric name prefixes for organization in WandB."""

    # Performance metrics
    PERFORMANCE = "performance"
    SCORE = f"{PERFORMANCE}/score"

    # Population metrics
    POPULATION = "population"
    SIZE = f"{POPULATION}/size"

    # Cost metrics
    COST = "cost"
    TOKENS = f"{COST}/tokens"
    USD = f"{COST}/usd"

    # Diversity metrics
    DIVERSITY = "diversity"

    # Lineage metrics
    LINEAGE = "lineage"

    # Iteration tracking
    ITERATION = "iteration"


class LogType(Enum):
    """Types of log entries for categorization."""

    METRIC = "metric"  # Numerical metrics
    CONFIG = "config"  # Configuration/hyperparameters
    ARTIFACT = "artifact"  # Files (code, visualizations)
    LINEAGE_TREE = "lineage_tree"  # Lineage tree visualization


class MonitoringBackend(ABC):
    """
    Abstract base class for monitoring backends.

    All monitoring implementations (WandB, TensorBoard, etc.) must inherit
    from this class and implement the abstract methods.

    The interface is designed for:
    - Real-time metric logging during evolution
    - Configuration and hyperparameter tracking
    - Artifact logging (code files, visualizations)
    - Graceful shutdown

    Example:
        >>> monitor = WandBMonitor(project="famou-experiments")
        >>> monitor.log_config({"strategy": "standard", "max_iterations": 10})
        >>> monitor.log_metrics({"score": 0.85}, step=0)
        >>> monitor.log_artifact("best.py", "best_program")
        >>> monitor.finish()
    """

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        commit: bool = True
    ) -> None:
        """
        Log metrics to the monitoring backend.

        Args:
            metrics: Dictionary of metric names to values
            step: Current iteration/step number
            commit: Whether to commit immediately (async backends may use this)

        Example:
            >>> monitor.log_metrics({
            ...     "performance/best_score": 0.95,
            ...     "performance/avg_score": 0.82,
            ...     "cost/total_cost_usd": 0.50,
            ... }, step=5)
        """
        pass

    @abstractmethod
    def log_config(
        self,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Log experiment configuration and hyperparameters.

        Should be called once at experiment start.

        Args:
            config: Configuration dictionary (will be logged)
            tags: Optional list of tags for categorization

        Example:
            >>> monitor.log_config({
            ...     "strategy": "creative_exploration",
            ...     "max_iterations": 10,
            ...     "temperature": 0.7,
            ... }, tags=["baseline", "test"])
        """
        pass

    @abstractmethod
    def log_artifact(
        self,
        path: str,
        name: str,
        type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an artifact (file) to the monitoring backend.

        Used for:
        - Best program code
        - Visualizations (lineage trees, heatmaps)
        - Configuration files

        Args:
            path: Path to the file to log
            name: Human-readable name
            type: Artifact type (e.g., "code", "visualization")
            metadata: Optional metadata dictionary

        Example:
            >>> monitor.log_artifact(
            ...     "best_program_iter_5.py",
            ...     "Best Program - Iteration 5",
            ...     "code",
            ...     metadata={"score": 0.95, "generation": 3}
            ... )
        """
        pass

    @abstractmethod
    def log_lineage_tree(
        self,
        tree_data: Dict[str, Any],
        island_id: Optional[int] = None
    ) -> None:
        """
        Log lineage tree visualization.

        Args:
            tree_data: Tree structure data
            island_id: Optional island ID for multi-island experiments

        Example:
            >>> monitor.log_lineage_tree({
            ...     "nodes": [{"id": "prog1", "parent": None, "score": 0.8}],
            ...     "edges": [("prog1", "prog2")]
            ... }, island_id=0)
        """
        pass

    @abstractmethod
    def log_table(
        self,
        data: Dict[str, List[Any]],
        name: str
    ) -> None:
        """
        Log a table (for structured data).

        Args:
            data: Dictionary with column names as keys and lists as values
            name: Table name

        Example:
            >>> monitor.log_table({
            ...     "ID": ["prog1", "prog2"],
            ...     "Score": [0.85, 0.92],
            ...     "Generation": [1, 2]
            ... }, "top_programs")
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """
        Finish monitoring and cleanup resources.

        Should be called at the end of experiment.
        Ensures all pending logs are flushed.
        """
        pass

    @abstractmethod
    def is_async(self) -> bool:
        """
        Check if this backend uses async logging.

        Returns:
            True if logging is asynchronous (non-blocking)
        """
        pass

    def log_metric(
        self,
        name: str,
        value: float,
        step: int
    ) -> None:
        """
        Convenience method to log a single metric.

        Args:
            name: Metric name
            value: Metric value
            step: Current step
        """
        self.log_metrics({name: value}, step=step)

    def log_info(
        self,
        message: str,
        level: str = "info"
    ) -> None:
        """
        Log an informational message (if supported).

        Args:
            message: Log message
            level: Log level (info, warning, error)
        """
        # Default: do nothing (can be overridden)
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures finish() is called."""
        self.finish()
        return False
