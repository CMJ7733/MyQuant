"""Diversity-based selection strategy for Famou 2.0.

Selects parents to maximize population diversity using feature vectors.
"""

import random
from typing import List, Optional

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule
from famou.utils.math import cosine_distance


class DiversitySelect(SelectModule):
    """
    Diversity-based selection strategy using feature vectors.

    Selects parents that are dissimilar to recently selected programs,
    promoting exploration of the search space. Useful when population
    diversity is low and you want to explore different regions.

    Selection modes:
    - "max_min": Select program with maximum minimum distance to recent selections
    - "weighted": Combine diversity score with quality score

    Args:
        name: Module name (default: class name)
        mode: Selection mode - "max_min" or "weighted" (default: "weighted")
        diversity_weight: Weight for diversity vs quality in "weighted" mode (default: 0.5)
        num_inspirations: Number of inspirations to select (default: 2)
        quality_threshold: Minimum score percentile to consider (default: 0.0)

    Example:
        >>> selector = DiversitySelect(diversity_weight=0.7)
        >>> # Heavily favors diversity over pure quality
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        mode: str = "weighted",
        diversity_weight: float = 0.5,
        num_inspirations: int = 2,
        quality_threshold: float = 0.0,
        **kwargs,
    ):
        super().__init__(name)
        self.mode = mode
        self.diversity_weight = diversity_weight
        self.num_inspirations = num_inspirations
        self.quality_threshold = quality_threshold

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """
        Select parent that maximizes diversity-quality tradeoff.

        Args:
            context: Rollout context with diversity info
            population: List of available programs

        Returns:
            Selected program ID
        """
        # Filter programs with feature vectors
        with_features = [p for p in population if p.feature_vector is not None]

        if not with_features:
            # Fallback to random selection if no feature vectors
            self.log_warning("No feature vectors available, using random selection")
            selected = random.choice(population)
            return selected.id

        # Apply quality threshold if specified
        if self.quality_threshold > 0:
            scores = [p.combined_score or 0.0 for p in with_features]
            if scores:
                min_score = sorted(scores)[int(len(scores) * self.quality_threshold)]
                with_features = [
                    p for p in with_features
                    if (p.combined_score or 0.0) >= min_score
                ]

        if not with_features:
            with_features = [p for p in population if p.feature_vector is not None]

        if self.mode == "max_min":
            selected = self._select_max_min_diversity(with_features)
        else:  # weighted
            selected = self._select_weighted(with_features)

        self.log_info(
            f"Diversity selection ({self.mode}): {selected.id} "
            f"(score={selected.combined_score or 0:.4f})"
        )

        return selected.id

    def _select_max_min_diversity(self, candidates: List[Program]) -> Program:
        """
        Select program with maximum minimum distance to others.

        Greedy selection that picks the most isolated program.
        """
        if len(candidates) == 1:
            return candidates[0]

        best_program = None
        best_min_dist = -1.0

        for candidate in candidates:
            # Compute minimum distance to all other programs
            distances = [
                cosine_distance(candidate.feature_vector, other.feature_vector)
                for other in candidates
                if other.id != candidate.id
            ]

            if distances:
                min_dist = min(distances)
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_program = candidate

        return best_program or candidates[0]

    def _select_weighted(self, candidates: List[Program]) -> Program:
        """
        Select using weighted combination of diversity and quality.

        Diversity score = average distance to other programs
        Quality score = normalized combined_score
        Final score = diversity_weight * diversity + (1 - diversity_weight) * quality
        """
        if len(candidates) == 1:
            return candidates[0]

        # Compute diversity scores (average distance to others)
        diversity_scores = []
        for candidate in candidates:
            distances = [
                cosine_distance(candidate.feature_vector, other.feature_vector)
                for other in candidates
                if other.id != candidate.id
            ]
            avg_dist = sum(distances) / len(distances) if distances else 0.0
            diversity_scores.append(avg_dist)

        # Normalize diversity scores to [0, 1]
        max_div = max(diversity_scores) if diversity_scores else 1.0
        min_div = min(diversity_scores) if diversity_scores else 0.0
        div_range = max_div - min_div if max_div > min_div else 1.0
        diversity_scores = [(d - min_div) / div_range for d in diversity_scores]

        # Compute quality scores (normalized)
        quality_scores = [p.combined_score or 0.0 for p in candidates]
        max_qual = max(quality_scores) if quality_scores else 1.0
        min_qual = min(quality_scores) if quality_scores else 0.0
        qual_range = max_qual - min_qual if max_qual > min_qual else 1.0
        quality_scores = [(q - min_qual) / qual_range for q in quality_scores]

        # Compute weighted scores
        weighted_scores = [
            self.diversity_weight * div + (1 - self.diversity_weight) * qual
            for div, qual in zip(diversity_scores, quality_scores)
        ]

        # Select best weighted score
        best_idx = weighted_scores.index(max(weighted_scores))
        return candidates[best_idx]

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """
        Select diverse inspirations using greedy max-min selection.

        Selects programs that are maximally different from each other
        and from the parent.

        Args:
            context: Rollout context
            population: List of available programs
            parent_id: ID of the selected parent

        Returns:
            List of diverse inspiration program IDs
        """
        if self.num_inspirations <= 0:
            return []

        # Filter candidates (exclude parent, require feature vectors)
        candidates = [
            p for p in population
            if p.id != parent_id and p.feature_vector is not None
        ]

        if not candidates:
            # Fallback: return top programs by score
            available = [p for p in population if p.id != parent_id]
            sorted_by_score = sorted(
                available,
                key=lambda p: p.combined_score or 0.0,
                reverse=True,
            )
            return [p.id for p in sorted_by_score[: self.num_inspirations]]

        # Get parent for distance calculations
        parent = next((p for p in population if p.id == parent_id), None)

        # Greedy max-min diversity selection
        selected: List[Program] = []

        # Start with program most different from parent
        if parent and parent.feature_vector:
            distances = [
                cosine_distance(p.feature_vector, parent.feature_vector)
                for p in candidates
            ]
            first_idx = distances.index(max(distances))
            selected.append(candidates[first_idx])
        else:
            # Start with highest scoring program
            candidates_sorted = sorted(
                candidates, key=lambda p: p.combined_score or 0.0, reverse=True
            )
            selected.append(candidates_sorted[0])

        # Greedily add most diverse programs
        remaining = [p for p in candidates if p.id != selected[0].id]

        while len(selected) < self.num_inspirations and remaining:
            best_candidate = None
            best_min_dist = -1.0

            for candidate in remaining:
                # Min distance to already selected
                min_dist = min(
                    cosine_distance(candidate.feature_vector, s.feature_vector)
                    for s in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break

        inspiration_ids = [p.id for p in selected]

        if inspiration_ids:
            self.log_info(f"Selected {len(inspiration_ids)} diverse inspirations")

        return inspiration_ids
