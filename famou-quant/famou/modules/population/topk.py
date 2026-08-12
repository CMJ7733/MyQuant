"""Top-K population management module for Famou 2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from famou.core.data import Program
from famou.modules.population.base import PopulationModule

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig


class TopKPopulation(PopulationModule):
    """
    Keep top-K programs by combined_score.

    This is the most common population management strategy, maintaining
    a fixed-size population of the best programs seen so far.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        """Initialize TopKPopulation."""
        super().__init__(name)

    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Keep top-K programs by score in a single "population" bucket.

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: New program to consider
            experiment_config: Experiment configuration for accessing island_size
            island_id: Island identifier (unused in TopKPopulation, for API compatibility)

        Returns:
            Updated population with top-K in "population" bucket
        """
        # Priority: per-island configured capacity > default 20
        k = experiment_config.get_island_size(island_id) if experiment_config else 20

        # Flatten current population
        current_progs = []
        for bucket_progs in current_population.values():
            current_progs.extend(bucket_progs)

        # Add new program to the pool
        all_programs = current_progs + [new_program]

        # Filter out programs without scores
        scored_programs = [p for p in all_programs if p.combined_score is not None]

        if not scored_programs:
            self.log_warning("No programs with scores available!")
            return {"population": []}

        # Sort by score (descending) and take top K
        sorted_programs = sorted(
            scored_programs,
            key=lambda p: p.combined_score,
            reverse=True,
        )
        top_k = sorted_programs[:k]

        # Return in single "population" bucket
        if self.logger:
            scores = [p.combined_score for p in top_k]
            self.log_info(
                f"Selected top-{k} programs: "
                f"best={max(scores):.4f}, worst={min(scores):.4f}, avg={sum(scores)/len(scores):.4f}"
            )

        return {"population": top_k}
