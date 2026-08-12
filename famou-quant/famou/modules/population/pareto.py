"""Pareto Front population management module for Famou 2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from famou.core.data import Program
from famou.modules.population.base import PopulationModule

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig


class ParetoPopulation(PopulationModule):
    """
    Pareto Front for multi-objective optimization.

    Maintains the Pareto front: the set of non-dominated solutions where
    a solution dominates another if it's better or equal on all objectives
    and strictly better on at least one.

    Use Cases:
    - Multi-objective optimization (accuracy vs efficiency, precision vs recall)
    - Trade-off analysis (performance vs resource usage)
    - Diverse high-quality solutions across multiple metrics
    - Exploring solution space comprehensively

    Population Structure:
        {
            "pareto_front": [Program1, Program2, ...]  # All non-dominated programs
        }

    Pareto Dominance:
        Program A dominates Program B if:
        - A is better or equal on all objectives
        - A is strictly better on at least one objective

    Example:
        >>> # Objectives: ["accuracy", "efficiency"], both maximize
        >>> # Program A: accuracy=0.9, efficiency=0.8
        >>> # Program B: accuracy=0.85, efficiency=0.85
        >>> # Program C: accuracy=0.95, efficiency=0.7
        >>> # A dominates B (better accuracy, slightly worse efficiency)
        >>> # A and C are non-dominated (tradeoff)
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        objectives: List[str],
        maximize: Optional[List[bool]] = None,
        **kwargs,
    ):
        """
        Initialize ParetoPopulation.

        Args:
            name: Module name (default: class name)
            objectives: Names of objective metrics (must exist in program.metrics)
                       e.g., ["accuracy", "efficiency"]
            maximize: Whether to maximize (True) or minimize (False) each objective
                     Default: all True (maximize all)
            max_size: Maximum Pareto front size. If exceeded, uses crowding distance
                     to maintain diversity. If None, uses island_size from config
                     (default: island_size, fallback to 20).
        """
        super().__init__(name)
        self.objectives = objectives
        self.maximize = maximize or [True] * len(objectives)

        if len(self.maximize) != len(self.objectives):
            raise ValueError(
                f"maximize length ({len(self.maximize)}) must match "
                f"objectives length ({len(self.objectives)})"
            )

    def _get_objective_value(
        self, program: Program, objective: str
    ) -> Optional[float]:
        """
        Extract objective value from program.

        First checks program.metrics, then falls back to combined_score
        if objective is "score" or "fitness".

        Args:
            program: Program to extract objective from
            objective: Objective name

        Returns:
            Objective value or None if not found
        """
        # Special case: combined_score
        if objective in ["score", "fitness", "combined_score"]:
            return program.combined_score

        # Check metrics dict
        if objective in program.metrics:
            value = program.metrics[objective]
            # Convert to float if possible
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, bool):
                return float(value)
            else:
                return None

        return None

    def _dominates(self, prog_a: Program, prog_b: Program) -> bool:
        """
        Check if program A dominates program B.

        A dominates B if:
        - A is better or equal on all objectives
        - A is strictly better on at least one objective

        Args:
            prog_a: First program
            prog_b: Second program

        Returns:
            True if A dominates B, False otherwise
        """
        better_or_equal = True
        strictly_better_on_one = False

        for i, objective in enumerate(self.objectives):
            value_a = self._get_objective_value(prog_a, objective)
            value_b = self._get_objective_value(prog_b, objective)

            # If either value is missing, cannot determine dominance
            if value_a is None or value_b is None:
                return False

            # Adjust for minimize vs maximize
            if not self.maximize[i]:
                value_a, value_b = -value_a, -value_b

            # Check dominance conditions
            if value_a < value_b:
                better_or_equal = False
                break
            elif value_a > value_b:
                strictly_better_on_one = True

        return better_or_equal and strictly_better_on_one

    def _compute_pareto_front(self, programs: List[Program]) -> List[Program]:
        """
        Compute Pareto front from a list of programs.

        Args:
            programs: List of programs to filter

        Returns:
            List of non-dominated programs (Pareto front)
        """
        pareto_front = []

        for candidate in programs:
            # Check if candidate is dominated by any program in current front
            is_dominated = False
            programs_to_remove = []

            for i, front_program in enumerate(pareto_front):
                if self._dominates(front_program, candidate):
                    # Candidate is dominated, skip it
                    is_dominated = True
                    break
                elif self._dominates(candidate, front_program):
                    # Candidate dominates this front program, mark for removal
                    programs_to_remove.append(i)

            if not is_dominated:
                # Remove dominated programs from front
                pareto_front = [
                    p for i, p in enumerate(pareto_front) if i not in programs_to_remove
                ]
                # Add candidate to front
                pareto_front.append(candidate)

        return pareto_front

    def _apply_crowding_distance(
        self, programs: List[Program], max_size: int
    ) -> List[Program]:
        """
        Apply crowding distance to reduce Pareto front size.

        Keeps programs with highest crowding distance (most isolated/diverse).

        Args:
            programs: Pareto front programs
            max_size: Maximum number of programs to keep

        Returns:
            Reduced Pareto front with max_size programs
        """
        if len(programs) <= max_size:
            return programs

        # Compute crowding distance for each program
        crowding_distances = [0.0] * len(programs)

        for obj_idx, objective in enumerate(self.objectives):
            # Get objective values
            values = []
            for prog in programs:
                value = self._get_objective_value(prog, objective)
                if value is None:
                    value = 0.0
                values.append(value)

            # Sort by objective value
            sorted_indices = sorted(
                range(len(programs)), key=lambda i: values[i]
            )

            # Boundary programs get infinite distance
            crowding_distances[sorted_indices[0]] = float("inf")
            crowding_distances[sorted_indices[-1]] = float("inf")

            # Compute crowding distance for middle programs
            value_range = values[sorted_indices[-1]] - values[sorted_indices[0]]
            if value_range > 0:
                for i in range(1, len(programs) - 1):
                    idx = sorted_indices[i]
                    idx_prev = sorted_indices[i - 1]
                    idx_next = sorted_indices[i + 1]

                    crowding_distances[idx] += (
                        values[idx_next] - values[idx_prev]
                    ) / value_range

        # Sort by crowding distance (descending) and take top max_size
        sorted_indices = sorted(
            range(len(programs)),
            key=lambda i: crowding_distances[i],
            reverse=True,
        )
        return [programs[i] for i in sorted_indices[:max_size]]

    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Update Pareto front with new program.

        Recomputes the Pareto front from all programs (current + new).
        If max_size is set and exceeded, applies crowding distance pruning.

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: New program to consider
            experiment_config: Experiment configuration for accessing island_size
            island_id: Island identifier (unused)

        Returns:
            Updated population with Pareto front in "pareto_front" bucket
        """
        # Use per-island capacity as max_size if not explicitly set
        max_size = experiment_config.get_island_size(island_id) if experiment_config else 20

        # Check if new program has all required objectives
        missing_objectives = []
        for objective in self.objectives:
            value = self._get_objective_value(new_program, objective)
            if value is None:
                missing_objectives.append(objective)

        if missing_objectives:
            self.log_warning(
                f"Program {new_program.id} missing objectives {missing_objectives}, "
                "skipping population update"
            )
            return current_population

        # Flatten current population
        all_programs = []
        for bucket_progs in current_population.values():
            all_programs.extend(bucket_progs)

        # Add new program
        all_programs.append(new_program)

        # Compute Pareto front
        pareto_front = self._compute_pareto_front(all_programs)

        # Apply crowding distance if needed
        if len(pareto_front) > max_size:
            pareto_front = self._apply_crowding_distance(
                pareto_front, max_size
            )

        # Log statistics
        self._log_pareto_stats(pareto_front, new_program)

        return {"pareto_front": pareto_front}

    def _log_pareto_stats(
        self, pareto_front: List[Program], new_program: Program
    ) -> None:
        """Log statistics about Pareto front."""
        if not self.logger:
            return

        # Check if new program is on the front
        on_front = new_program in pareto_front

        # Get objective ranges
        obj_ranges = []
        for objective in self.objectives:
            values = [
                self._get_objective_value(p, objective)
                for p in pareto_front
                if self._get_objective_value(p, objective) is not None
            ]
            if values:
                min_val = min(values)
                max_val = max(values)
                obj_ranges.append(f"{objective}=[{min_val:.4f}, {max_val:.4f}]")
            else:
                obj_ranges.append(f"{objective}=N/A")

        status = "ON FRONT" if on_front else "dominated"
        self.log_info(
            f"Pareto front size={len(pareto_front)}, new program {status}. "
            f"Ranges: {', '.join(obj_ranges)}"
        )
