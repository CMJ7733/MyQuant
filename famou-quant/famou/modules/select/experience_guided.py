"""Experience-guided selection strategy for Famou 2.0.

Selects parent programs and retrieves relevant experiences from StateStore.
The selected experiences are passed to ExperienceGuidedGenerate via SelectionData.

Key concepts:
- Score-based parent selection (weighted random by score)
- Experience selection from StateStore using multiple strategies
- Experiences passed via SelectionData.experiences field
"""

from typing import List, Optional

import numpy as np

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule
from famou.utils.math import stable_softmax


class ExperienceGuidedSelect(SelectModule):
    """
    Selection module that retrieves experiences from StateStore.

    Selects a parent program by score and retrieves relevant experiences
    from StateStore to pass to the generation module.

    Selection Strategies:
    - "top_improvement": Select experiences with highest improvement (default)
    - "diverse": Select diverse experiences (greedy max-min diversity)
    - "similar_to_parent": Select experiences similar to current parent
    - "recent": Select most recent experiences

    Args:
        name: Module name (default: class name)
        temperature: Softmax temperature for parent sampling (default: 1.0)
                     Lower = more deterministic (favor best)
                     Higher = more random (explore more)
        num_inspirations: Number of inspiration programs (default: 0)
        max_experiences: Maximum experiences to select (default: 3)
        selection_strategy: How to select experiences (default: "top_improvement")
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        temperature: float = 1.0,
        num_inspirations: int = 0,
        max_experiences: int = 3,
        selection_strategy: str = "top_improvement",
        **kwargs,
    ):
        super().__init__(name)
        self.temperature = temperature
        self.num_inspirations = num_inspirations
        self.max_experiences = max_experiences
        self.selection_strategy = selection_strategy

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select parent using score-weighted random sampling.

        Args:
            context: Rollout context (unused)
            population: List of available programs

        Returns:
            Selected program ID
        """
        if not population:
            raise ValueError("Cannot select from empty population")

        # Extract scores
        scores = np.array([p.combined_score or 0.0 for p in population])
        scores = np.nan_to_num(scores, nan=0.0)

        # Handle all-zero scores
        if scores.sum() == 0:
            # Random selection if all scores are zero
            selected = population[np.random.randint(len(population))]
            self.log_info(f"Selected random parent {selected.id} (all scores zero)")
            return selected.id

        # Normalize to prevent overflow
        if scores.max() > 0:
            scores = scores / scores.max()

        # Softmax sampling with temperature
        probabilities = stable_softmax(scores, self.temperature)
        selected_idx = np.random.choice(len(population), p=probabilities)
        selected = population[selected_idx]

        self.log_info(
            f"Selected parent {selected.id} (score={selected.combined_score:.4f})"
        )

        return selected.id

    def select_experiences(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[dict]]:
        """
        Retrieve and select most relevant experiences from StateStore.

        Selection Strategies:
        - "top_improvement": Select experiences with highest improvement (default)
        - "diverse": Select diverse experiences (greedy max-min diversity)
        - "similar_to_parent": Select experiences similar to current parent
        - "recent": Select most recent experiences

        Args:
            context: Rollout context with state store
            population: List of available programs
            parent_id: ID of the selected parent

        Returns:
            List of experience dictionaries (up to max_experiences), or None if no experiences
        """
        # Get parent program for similarity-based selection
        parent = context.get_program_by_id(parent_id)
        if not parent:
            return None

        # Retrieve all experiences for this island from StateAccessor
        all_experiences = context.state.get_island(
            "experiences",
            default=[]
        )

        if not all_experiences:
            self.log_info("No experiences available for guidance")
            return None

        # Apply selection strategy
        if self.selection_strategy == "top_improvement":
            selected = self._select_top_improvement(all_experiences)
        elif self.selection_strategy == "diverse":
            selected = self._select_diverse(all_experiences)
        elif self.selection_strategy == "similar_to_parent":
            selected = self._select_similar_to_parent(all_experiences, parent)
        elif self.selection_strategy == "recent":
            selected = self._select_recent(all_experiences)
        else:
            self.log_warning(f"Unknown strategy '{self.selection_strategy}', using top_improvement")
            selected = self._select_top_improvement(all_experiences)

        self.log_info(
            f"Selected {len(selected)} experiences using '{self.selection_strategy}' strategy"
        )
        self.log_info(selected)

        return selected

    def _select_top_improvement(self, experiences: List[dict]) -> List[dict]:
        """Select experiences with highest improvement."""
        sorted_exp = sorted(
            experiences,
            key=lambda e: e.get("improvement", 0.0),
            reverse=True
        )
        return sorted_exp[:self.max_experiences]

    def _select_diverse(self, experiences: List[dict]) -> List[dict]:
        """Select diverse experiences using greedy max-min text diversity."""
        if len(experiences) <= self.max_experiences:
            return experiences

        # Start with best experience
        selected = [max(experiences, key=lambda e: e.get("improvement", 0.0))]
        remaining = [e for e in experiences if e != selected[0]]

        # Greedily select most diverse
        while len(selected) < self.max_experiences and remaining:
            best_candidate = None
            best_min_dist = -1

            for candidate in remaining:
                # Calculate minimum distance to already selected
                candidate_text = f"{candidate.get('changes_made', '')} {candidate.get('general_pattern', '')}"
                candidate_tokens = set(candidate_text.split())

                min_dist = float('inf')
                for sel in selected:
                    sel_text = f"{sel.get('changes_made', '')} {sel.get('general_pattern', '')}"
                    sel_tokens = set(sel_text.split())

                    # Jaccard distance
                    intersection = len(candidate_tokens & sel_tokens)
                    union = len(candidate_tokens | sel_tokens)
                    similarity = intersection / union if union > 0 else 0.0
                    distance = 1.0 - similarity

                    min_dist = min(min_dist, distance)

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        return selected

    def _select_similar_to_parent(self, experiences: List[dict], parent: Program) -> List[dict]:
        """Select experiences most similar to current parent program."""
        if not parent.code:
            return self._select_top_improvement(experiences)

        parent_tokens = set(parent.code.split())

        # Score each experience by similarity to parent
        scored = []
        for exp in experiences:
            exp_parent_code = exp.get("parent_code", "")
            exp_tokens = set(exp_parent_code.split())

            # Jaccard similarity
            intersection = len(parent_tokens & exp_tokens)
            union = len(parent_tokens | exp_tokens)
            similarity = intersection / union if union > 0 else 0.0

            scored.append((similarity, exp))

        # Sort by similarity (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scored[:self.max_experiences]]

    def _select_recent(self, experiences: List[dict]) -> List[dict]:
        """Select most recent experiences (already ordered by collection time)."""
        # Experiences are stored with most recent first
        return experiences[:self.max_experiences]
