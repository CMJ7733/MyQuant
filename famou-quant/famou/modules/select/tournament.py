"""Tournament selection strategy for Famou 2.0.

Standard evolutionary algorithm operator that selects parents through competition.
"""

import random
from typing import List, Optional

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


class TournamentSelect(SelectModule):
    """
    Tournament selection strategy - select best from random subset.

    Standard EA operator that:
    1. Randomly samples k programs from population (tournament)
    2. Returns the best program from the tournament as parent

    This provides selection pressure while maintaining diversity better than
    pure elite selection. Smaller tournament_size = more exploration,
    larger tournament_size = more exploitation.

    Args:
        name: Module name (default: class name)
        tournament_size: Number of programs in each tournament (default: 3)
        num_inspirations: Number of programs to select as inspirations (default: 2)
        inspiration_tournament_size: Tournament size for inspiration selection (default: 2)

    Example:
        >>> selector = TournamentSelect(tournament_size=5)
        >>> # Runs 5-way tournament, winner becomes parent
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        tournament_size: int = 3,
        num_inspirations: int = 2,
        inspiration_tournament_size: int = 2,
        **kwargs,
    ):
        super().__init__(name)
        self.tournament_size = tournament_size
        self.num_inspirations = num_inspirations
        self.inspiration_tournament_size = inspiration_tournament_size

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select parent via tournament selection.

        Randomly samples tournament_size programs and returns the best one.

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs

        Returns:
            Selected program ID (tournament winner)
        """
        # Ensure tournament size doesn't exceed population
        k = min(self.tournament_size, len(population))

        # Sample random tournament
        tournament = random.sample(population, k)

        # Select best from tournament
        winner = max(
            tournament,
            key=lambda p: p.combined_score if p.combined_score is not None else 0.0,
        )

        self.log_info(
            f"Tournament (k={k}): selected {winner.id} "
            f"(score={winner.combined_score or 0:.4f})"
        )

        return winner.id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Select inspirations via multiple smaller tournaments.

        Runs num_inspirations tournaments (excluding parent) to select
        diverse high-quality inspirations.

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs
            parent_id: ID of the selected parent (excluded from inspirations)

        Returns:
            List of inspiration program IDs
        """
        if self.num_inspirations <= 0:
            return []

        # Exclude parent from candidates
        candidates = [p for p in population if p.id != parent_id]

        if not candidates:
            return []

        # Run multiple tournaments to select inspirations
        inspirations = []
        selected_ids = set()

        for _ in range(self.num_inspirations):
            # Filter out already selected programs
            available = [p for p in candidates if p.id not in selected_ids]
            if not available:
                break

            # Run tournament
            k = min(self.inspiration_tournament_size, len(available))
            tournament = random.sample(available, k)

            winner = max(
                tournament,
                key=lambda p: p.combined_score if p.combined_score is not None else 0.0,
            )

            inspirations.append(winner.id)
            selected_ids.add(winner.id)

        if inspirations:
            self.log_info(f"Selected {len(inspirations)} inspirations via tournaments")

        return inspirations
