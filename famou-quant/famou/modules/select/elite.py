"""Elite selection strategy for Famou 2.0.

Selects the single best program as parent and top programs as inspirations.
"""

from typing import List, Optional

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


class EliteSelect(SelectModule):
    """
    Elite selection strategy - selects best program by score.

    Selects:
    - Parent: The single best program by combined_score
    - Inspirations: Next best programs (excluding parent)

    This is a greedy exploitation strategy that always picks the highest-scoring
    program as the parent to mutate.

    Args:
        name: Module name (default: class name)
        num_inspirations: Number of top programs to include as inspirations (default: 2)
        exclude_parent: If True, exclude parent from inspirations (default: True)

    Example:
        >>> selector = EliteSelect(num_inspirations=3)
        >>> # Selects best program as parent, next 3 best as inspirations
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        num_inspirations: int = 2,
        exclude_parent: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self.num_inspirations = num_inspirations
        self.exclude_parent = exclude_parent

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select the best program ID by score.

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs

        Returns:
            Selected program ID (best by score)
        """
        # Sort programs by score descending (None scores treated as 0)
        sorted_programs = sorted(
            population,
            key=lambda p: p.combined_score if p.combined_score is not None else 0.0,
            reverse=True,
        )

        # Select best program (top of sorted list)
        best_program = sorted_programs[0]
        selected_id = best_program.id

        # Log selection info
        self.log_info(
            f"Selected best program {selected_id} with score: "
            f"{best_program.combined_score if best_program.combined_score is not None else 'None'}"
        )

        return selected_id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Select next best programs as inspirations (excluding parent).

        Uses the same ordering as parent selection (best scores first).

        Args:
            context: Rollout context (unused in this implementation)
            population: List of available programs
            parent_id: ID of the selected parent (excluded from inspirations)

        Returns:
            List of inspiration program IDs (next best programs after parent)
        """
        # Filter out parent if needed
        if self.exclude_parent:
            available = [p for p in population if p.id != parent_id]
        else:
            available = population

        # No inspirations requested
        if self.num_inspirations <= 0 or not available:
            return []

        # Sort by score descending (best first)
        sorted_programs = sorted(
            available,
            key=lambda p: p.combined_score if p.combined_score is not None else 0.0,
            reverse=True,
        )

        # Select top programs as inspirations
        selected = sorted_programs[: self.num_inspirations]
        inspiration_ids = [p.id for p in selected]

        if inspiration_ids:
            scores_str = ", ".join(
                [
                    f"{pid}({self._get_score(population, pid):.3f})"
                    for pid in inspiration_ids
                ]
            )
            self.log_info(f"Selected {len(inspiration_ids)} inspirations: {scores_str}")

        return inspiration_ids

    def _get_score(self, population: List[Program], program_id: str) -> float:
        """Helper to get score for a program ID."""
        for p in population:
            if p.id == program_id:
                return p.combined_score if p.combined_score is not None else 0.0
        return 0.0

