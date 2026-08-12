"""Age-layered population management module for Famou 2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from famou.core.data import Program
from famou.modules.population.base import PopulationModule

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig


class AgeLayeredPopulation(PopulationModule):
    """
    Age-Layered Population Structure (ALPS-inspired).

    Organizes programs into age layers to prevent premature convergence and
    maintain exploration. Young programs compete in separate layers, giving
    them a chance to evolve without being dominated by older, highly-optimized
    programs.

    Use Cases:
    - Long-running evolution (prevent stagnation)
    - Continuous exploration while exploiting
    - Problems prone to premature convergence
    - Maintain diversity across program ages

    Population Structure:
        {
            "layer_0": [Program1, ...],  # Age 0-10 iterations
            "layer_1": [Program2, ...],  # Age 11-50 iterations
            "layer_2": [Program3, ...],  # Age 51+ iterations
        }

    Age Calculation:
        age = current_iteration - program.iteration

    Example:
        >>> # At iteration 100
        >>> # Program created at iteration 95 has age 5 → layer_0
        >>> # Program created at iteration 40 has age 60 → layer_2
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        age_thresholds: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Initialize AgeLayeredPopulation.

        Args:
            name: Module name (default: class name)
            age_thresholds: Age boundaries for layers (e.g., [10, 50] creates 3 layers:
                           layer_0: age 0-10, layer_1: age 11-50, layer_2: age 51+)
                           Default: [10, 50]
            programs_per_layer: Number of top programs to keep per layer
                               If None, calculated as island_size // num_layers
                               (default: island_size // 3, fallback to 10)
        """
        super().__init__(name)
        self.age_thresholds = age_thresholds or [10, 50]

        # Sort thresholds to ensure consistent layering
        self.age_thresholds.sort()

    def _get_layer_for_age(self, age: int) -> str:
        """
        Determine which layer a program belongs to based on its age.

        Args:
            age: Program age (current_iteration - program.iteration)

        Returns:
            Layer name (e.g., "layer_0", "layer_1", ...)
        """
        for i, threshold in enumerate(self.age_thresholds):
            if age <= threshold:
                return f"layer_{i}"

        # Oldest layer (beyond all thresholds)
        return f"layer_{len(self.age_thresholds)}"

    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Update population by assigning programs to age layers and keeping top-K per layer.

        Programs are re-assigned to layers based on their current age, so programs
        can "age out" and move to older layers over time.

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: New program to consider
            experiment_config: Experiment configuration for accessing island_size
            island_id: Island identifier (unused)

        Returns:
            Updated population with age-layered structure
        """
        # Calculate programs_per_layer from island_size if not set
        # Default: island_size // num_layers
        num_layers = len(self.age_thresholds) + 1
        island_size = experiment_config.get_island_size(island_id) if experiment_config else 30
        programs_per_layer = max(1, island_size // num_layers)

        # Skip if new program has no score
        if new_program.combined_score is None:
            self.log_warning(
                f"Program {new_program.id} has no score, skipping population update"
            )
            return current_population

        # Get current iteration from new program (most recent)
        current_iteration = new_program.iteration

        # Flatten current population
        all_programs = []
        for bucket_progs in current_population.values():
            all_programs.extend(bucket_progs)

        # Add new program
        all_programs.append(new_program)

        # Re-assign all programs to layers based on current age
        layers: Dict[str, List[Program]] = {}

        for program in all_programs:
            # Skip programs without scores
            if program.combined_score is None:
                continue

            # Calculate age
            age = current_iteration - program.iteration

            # Get layer for this age
            layer_name = self._get_layer_for_age(age)

            # Initialize layer if needed
            if layer_name not in layers:
                layers[layer_name] = []

            layers[layer_name].append(program)

        # Keep top-K per layer
        updated_population = {}
        for layer_name, programs in layers.items():
            # Sort by score descending and take top-K
            sorted_programs = sorted(
                programs,
                key=lambda p: p.combined_score,
                reverse=True,
            )
            top_k = sorted_programs[:programs_per_layer]
            updated_population[layer_name] = top_k

        # Ensure all layers exist (even if empty)
        num_layers = len(self.age_thresholds) + 1
        for i in range(num_layers):
            layer_name = f"layer_{i}"
            if layer_name not in updated_population:
                updated_population[layer_name] = []

        # Log statistics
        self._log_layer_stats(
            updated_population, current_iteration, new_program
        )

        return updated_population

    def _log_layer_stats(
        self,
        population: Dict[str, List[Program]],
        current_iteration: int,
        new_program: Program,
    ) -> None:
        """Log statistics about layer distribution."""
        if not self.logger:
            return

        new_age = current_iteration - new_program.iteration
        new_layer = self._get_layer_for_age(new_age)

        layer_stats = []
        for layer_name in sorted(population.keys()):
            programs = population[layer_name]
            if not programs:
                layer_stats.append(f"{layer_name}=0")
                continue

            count = len(programs)
            scores = [p.combined_score for p in programs]
            best = max(scores)
            ages = [current_iteration - p.iteration for p in programs]
            min_age = min(ages)
            max_age = max(ages)

            layer_stats.append(
                f"{layer_name}={count} (ages={min_age}-{max_age}, best={best:.4f})"
            )

        total = sum(len(progs) for progs in population.values())

        self.log_info(
            f"Added to {new_layer} (age={new_age}, score={new_program.combined_score:.4f}). "
            f"Total={total}, {', '.join(layer_stats)}"
        )
