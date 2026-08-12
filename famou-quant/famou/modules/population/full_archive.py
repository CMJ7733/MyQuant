"""Full archive population module - keeps all programs without pruning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Any, Literal, Optional

from famou.core.data import Program
from famou.modules.population.base import PopulationModule

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig


class FullArchivePopulation(PopulationModule):
    """
    Keep all programs without pruning, organized by generation, lineage, or flat.

    This population management strategy never removes programs, just accumulates
    all programs that have been added. Programs can be organized in different ways:

    Organization Options:
    - "generation" (default): {"gen_0": [...], "gen_1": [...], "gen_2": [...]}
      → Organize by evolutionary depth (generation number)
      → Most intuitive for tree structure
      → Easy to query "all programs at depth N"

    - "flat": {"population": [all programs]}
      → Single bucket with all programs in chronological order
      → Simplest structure
      → No hierarchy

    - "lineage": {"lineage_{parent_id}": [...], "lineage_no_parent": [...]}
      → Organize by immediate parent ID
      → Programs with same parent_id grouped together (sibling programs)
      → Seed programs (generation 0) or programs without parent_id go to "lineage_no_parent"
      → Good for analyzing mutations from the same parent

    This is useful when:
    - You want to keep complete lineage for later analysis
    - You have sufficient memory
    - You want to explore the full search space without pruning

    Args:
        organize_by: Organization strategy - "generation", "flat", or "lineage"
                    (default: "generation")
    """

    def __init__(
        self,
        name: Optional[str] = None,
        organize_by: Literal["generation", "flat", "lineage"] = "generation",
        **kwargs: Dict[str, Any],
    ):
        """
        Initialize FullArchivePopulation with organization strategy.

        Args:
            name: Module name (default: class name)
            organize_by: How to organize programs in buckets (default: "generation")
        """
        super().__init__(name)
        self.organize_by = organize_by

    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Add new program to population without removing any existing programs.

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: New program to add
            experiment_config: Experiment configuration (unused, for API compatibility)
            island_id: Island identifier (unused, for API compatibility)

        Returns:
            Updated population organized according to organize_by strategy
        """
        # Suppress unused parameter warnings (kept for API compatibility)
        _ = experiment_config
        _ = island_id
        # Check if program already exists (deduplication by ID across all buckets)
        existing_ids = set()
        for bucket_progs in current_population.values():
            for p in bucket_progs:
                existing_ids.add(p.id)

        if new_program.id in existing_ids:
            self.log_info(f"Program {new_program.id} already in population, skipping")
            return current_population

        # Add program to appropriate bucket based on organization strategy
        updated_population = dict(current_population)  # Shallow copy of bucket structure

        if self.organize_by == "flat":
            # Single bucket with all programs
            bucket_id = "population"
            if bucket_id not in updated_population:
                updated_population[bucket_id] = []
            updated_population[bucket_id] = updated_population[bucket_id] + [new_program]

        elif self.organize_by == "generation":
            # Organize by generation depth
            bucket_id = f"gen_{new_program.generation}"
            if bucket_id not in updated_population:
                updated_population[bucket_id] = []
            updated_population[bucket_id] = updated_population[bucket_id] + [new_program]

        elif self.organize_by == "lineage":
            # Organize by lineage root (find root ancestor)
            lineage_root = self._get_lineage_root(new_program)
            bucket_id = f"lineage_{lineage_root}"
            if bucket_id not in updated_population:
                updated_population[bucket_id] = []
            updated_population[bucket_id] = updated_population[bucket_id] + [new_program]

        else:
            raise ValueError(
                f"Invalid organize_by value: {self.organize_by}. "
                f"Must be 'generation', 'flat', or 'lineage'."
            )

        # Count total programs across all buckets
        total_programs = sum(len(progs) for progs in updated_population.values())

        if self.logger:
            self.log_info(
                f"Added program {new_program.id} to bucket '{bucket_id}' "
                f"(total: {total_programs} programs across {len(updated_population)} buckets)"
            )

        return updated_population

    def _get_lineage_root(self, program: Program) -> str:
        """
        Get the lineage identifier for organizing programs by parent.

        Programs are grouped by their immediate parent_id:
        - Programs with a parent_id → grouped by parent_id
        - Seed programs (generation 0) → special "no_parent" bucket
        - Programs without parent_id → special "no_parent" bucket

        Args:
            program: Program to get lineage identifier for

        Returns:
            Parent ID for grouping, or "no_parent" for seed programs
        """
        # For generation 0 (seed programs), use special bucket
        if program.generation == 0:
            return "no_parent"

        # For generated programs with parent_id, use parent as lineage group
        if program.parent_id:
            return program.parent_id

        # Fallback: programs without parent info go to special bucket
        return "no_parent"
