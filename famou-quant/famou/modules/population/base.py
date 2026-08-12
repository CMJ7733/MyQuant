"""
Population management base modules for Famou 2.0.

**SIMPLIFIED: One Program Per Rollout**

These modules handle how a new program is merged into the population.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from famou.core.data import Program

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig
    from famou.infrastructure.logger.base import Logger


class PopulationModule(ABC):
    """
    Base class for population management strategies.

    **Core Responsibility:**
    Decide whether the new program from the current rollout should be added to
    the population and which existing programs should be kept/removed.

    **Design Pattern:**
    PopulationModule.update_population() is called by Evolver after a rollout completes,
    not as part of the Rollout pipeline. This is because population updates happen
    between rollouts, not during them.

    **Multi-Island Support:**
    The island_id parameter in update_population() allows strategies to maintain
    per-island state if needed (e.g., ClusterPopulation tracks centroids per island).
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize PopulationModule with optional name for logging."""
        self.name = name or self.__class__.__name__
        self.logger: Optional[Logger] = None

    def set_logger(self, logger: Logger) -> None:
        """Set logger for this module (called by Evolver during initialization)."""
        self.logger = logger

    def log_info(self, message: str) -> None:
        """Log info message if logger is available."""
        if self.logger:
            self.logger.log(message, level="info")

    def log_warning(self, message: str) -> None:
        """Log warning message if logger is available."""
        if self.logger:
            self.logger.log(message, level="warning")

    @abstractmethod
    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Update population with a new program.

        Implements the strategy for merging the new program into the existing
        population (e.g., Top-K, diversity-based, MAP-Elites, etc.)

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: Newly generated program (with score)
            experiment_config: Experiment configuration for accessing settings
                like island_size. Optional for backwards compatibility.
            island_id: Island identifier for multi-island evolution (default: 0).

        Returns:
            Updated population {bucket_id: [programs]}
        """
        pass
