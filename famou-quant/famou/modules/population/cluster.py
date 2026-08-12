"""Cluster-based population management module for Famou 2.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from famou.core.data import Program
from famou.modules.population.base import PopulationModule
from famou.utils.math import (
    cosine_distance,
    normalize_vector,
    compute_mean_vector,
    compute_sum_vector,
    vectors_equal,
)

if TYPE_CHECKING:
    from famou.config.settings import ExperimentConfig


class ClusterPopulation(PopulationModule):
    """
    Online incremental clustering + periodic offline re-clustering (per-island state).

    This module maintains population diversity using a hybrid approach:
    - **Online**: Each new program is assigned to nearest cluster or creates a new
      cluster based on a dynamic distance threshold. Centroids are updated incrementally.
    - **Offline**: Periodically runs full k-means to correct drift from greedy online
      assignment.

    State is tracked per-island to support multi-island evolution.

    Args:
        name: Module name (default: class name)
        offline_interval: Run offline k-means every N updates (default: 10)

    Population structure:
        {
            "cluster_0": [Program, ...],
            "cluster_1": [Program, ...],
            ...
        }
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        offline_interval: int = 10,
        **kwargs,
    ):
        """Initialize ClusterPopulation with per-island state tracking."""
        super().__init__(name)
        self.offline_interval = offline_interval
        # Per-island state for incremental centroid updates
        self._island_centroids: Dict[int, List[List[float]]] = {}
        self._island_cluster_sums: Dict[int, List[List[float]]] = {}
        self._island_cluster_counts: Dict[int, List[int]] = {}
        self._island_update_counts: Dict[int, int] = {}

    def _get_island_state(self, island_id: int) -> tuple:
        """Get per-island state, initializing if needed."""
        if island_id not in self._island_centroids:
            self._island_centroids[island_id] = []
            self._island_cluster_sums[island_id] = []
            self._island_cluster_counts[island_id] = []
            self._island_update_counts[island_id] = 0
        return (
            self._island_centroids[island_id],
            self._island_cluster_sums[island_id],
            self._island_cluster_counts[island_id],
            self._island_update_counts[island_id],
        )

    def _reset_island_state(self, island_id: int) -> None:
        """Reset per-island clustering state for a fresh population."""
        self._island_centroids[island_id] = []
        self._island_cluster_sums[island_id] = []
        self._island_cluster_counts[island_id] = []
        self._island_update_counts[island_id] = 0

    def update_population(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        experiment_config: Optional[ExperimentConfig] = None,
        island_id: int = 0,
    ) -> Dict[str, List[Program]]:
        """
        Update population using online clustering with periodic offline re-clustering.

        Args:
            current_population: Current population {bucket_id: [programs]}
            new_program: New program to consider
            experiment_config: Experiment configuration for accessing island_size
            island_id: Island identifier for state tracking (default: 0)

        Returns:
            Updated population with cluster structure
        """
        # Get configuration
        island_size = experiment_config.get_island_size(island_id) if experiment_config else 20
        num_clusters = max(1, island_size // 10)

        # Flatten current population to get all programs
        all_programs = self._flatten_population(current_population)

        if not all_programs:
            self._reset_island_state(island_id)
        else:
            self._get_island_state(island_id)

        # Increment update count for this island
        self._island_update_counts[island_id] += 1
        current_update_count = self._island_update_counts[island_id]

        # Check if we need offline re-clustering
        if current_update_count % self.offline_interval == 0 and len(all_programs) > num_clusters:
            clusters = self._offline_kmeans(
                all_programs, num_clusters, island_id
            )
        else:
            # Online incremental update
            clusters = self._online_update(
                current_population, new_program, num_clusters, island_id
            )

        # Enforce overall capacity
        clusters = self._enforce_capacity(clusters, island_size)

        # Log cluster statistics
        self._log_cluster_stats(clusters, island_id)

        return clusters

    def _online_update(
        self,
        current_population: Dict[str, List[Program]],
        new_program: Program,
        num_clusters: int,
        island_id: int,
    ) -> Dict[str, List[Program]]:
        """
        Online incremental clustering update.

        Assigns new program to nearest cluster or creates new cluster based on
        dynamic threshold.
        """
        # Get per-island state
        centroids, cluster_sums, cluster_counts, _ = self._get_island_state(island_id)

        # Extract existing clusters
        clusters = self._extract_clusters(current_population)

        # Initialize state from existing clusters if needed
        if not centroids and clusters:
            self._initialize_state_from_clusters(clusters, island_id)
            centroids, cluster_sums, cluster_counts, _ = self._get_island_state(island_id)

        # Skip if new program has no score or no feature vector
        if new_program.combined_score is None or new_program.feature_vector is None:
            # Ensure we have all cluster buckets
            for i in range(num_clusters):
                if f"cluster_{i}" not in clusters:
                    clusters[f"cluster_{i}"] = []
            return clusters

        x = new_program.feature_vector

        # Case 1: No clusters yet - create first cluster
        if not centroids:
            clusters["cluster_0"] = [new_program]
            self._island_centroids[island_id] = [normalize_vector(x)]
            self._island_cluster_sums[island_id] = [list(x)]
            self._island_cluster_counts[island_id] = [1]
            # Ensure empty clusters for remaining
            for i in range(1, num_clusters):
                clusters[f"cluster_{i}"] = []
            return clusters

        # Compute distances to existing centroids
        distances = [cosine_distance(c, x) for c in centroids]

        # Compute dynamic threshold from existing clusters
        if len(centroids) >= 2:
            # Use average inter-centroid distance
            inter_distances = []
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    inter_distances.append(
                        cosine_distance(centroids[i], centroids[j])
                    )
            threshold = sum(inter_distances) / len(inter_distances)
        else:
            # Single cluster: compute diversity from programs in the cluster
            cluster_programs = []
            for prog_list in clusters.values():
                cluster_programs.extend(prog_list)
            threshold = self._compute_cluster_diversity(cluster_programs)

        # Decide: create new cluster or assign to nearest
        mean_dist = sum(distances) / len(distances)

        if mean_dist > threshold and len(centroids) < num_clusters:
            # Create new cluster
            new_idx = len(centroids)
            clusters[f"cluster_{new_idx}"] = [new_program]
            self._island_centroids[island_id].append(normalize_vector(x))
            self._island_cluster_sums[island_id].append(list(x))
            self._island_cluster_counts[island_id].append(1)
        else:
            # Assign to nearest cluster
            nearest_idx = distances.index(min(distances))
            cluster_key = f"cluster_{nearest_idx}"

            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(new_program)

            # Incremental centroid update (guard against state/centroid index mismatch)
            if nearest_idx < len(cluster_sums) and nearest_idx < len(cluster_counts):
                for i in range(len(x)):
                    if i < len(cluster_sums[nearest_idx]):
                        cluster_sums[nearest_idx][i] += x[i]
                    else:
                        cluster_sums[nearest_idx].append(x[i])
                cluster_counts[nearest_idx] += 1
                self._island_centroids[island_id][nearest_idx] = normalize_vector(cluster_sums[nearest_idx])

        # Ensure all cluster buckets exist
        for i in range(num_clusters):
            if f"cluster_{i}" not in clusters:
                clusters[f"cluster_{i}"] = []

        return clusters

    def _offline_kmeans(
        self,
        programs: List[Program],
        num_clusters: int,
        island_id: int,
        max_iterations: int = 10,
    ) -> Dict[str, List[Program]]:
        """
        Run full cosine k-means to re-cluster all programs.

        This corrects drift from online greedy assignment.
        """
        self.log_info(
            f"Island {island_id}: Running offline k-means re-clustering "
            f"on {len(programs)} programs"
        )

        # Filter programs with feature vectors
        programs_with_features = [p for p in programs if p.feature_vector is not None]
        if len(programs_with_features) <= num_clusters:
            # Not enough programs for k-means
            clusters = {}
            for i, p in enumerate(programs_with_features):
                clusters[f"cluster_{i}"] = [p]
            for i in range(len(programs_with_features), num_clusters):
                clusters[f"cluster_{i}"] = []
            self._update_state_from_clusters(clusters, island_id)
            return clusters

        # Initialize centroids using k-means++ style
        centroids = self._kmeans_plus_plus_init(programs_with_features, num_clusters)

        # Run k-means iterations
        # Note: centroids may be fewer than num_clusters when all feature vectors
        # are identical (kmeans++ returns only 1 centroid in that case).
        actual_clusters = len(centroids)
        for _ in range(max_iterations):
            # Assign programs to nearest centroid
            assignments: List[List[Program]] = [[] for _ in range(actual_clusters)]
            for program in programs_with_features:
                distances = [
                    cosine_distance(c, program.feature_vector)
                    for c in centroids
                ]
                nearest_idx = distances.index(min(distances))
                assignments[nearest_idx].append(program)

            # Update centroids
            new_centroids = []
            for i, cluster_programs in enumerate(assignments):
                if cluster_programs:
                    centroid = compute_mean_vector(
                        [p.feature_vector for p in cluster_programs]
                    )
                    new_centroids.append(normalize_vector(centroid))
                else:
                    new_centroids.append(centroids[i])

            # Check convergence
            if new_centroids == centroids:
                break
            centroids = new_centroids

        # Build clusters dict (pad with empty buckets up to num_clusters)
        clusters = {f"cluster_{i}": assignments[i] for i in range(actual_clusters)}
        for i in range(actual_clusters, num_clusters):
            clusters[f"cluster_{i}"] = []

        # Update internal state
        self._update_state_from_clusters(clusters, island_id)

        return clusters

    def _kmeans_plus_plus_init(
        self, programs: List[Program], num_clusters: int
    ) -> List[List[float]]:
        """Initialize centroids using k-means++ strategy."""
        # Start with best-scoring program
        sorted_programs = sorted(
            programs, key=lambda p: p.combined_score or 0.0, reverse=True
        )
        centroids = [normalize_vector(sorted_programs[0].feature_vector)]

        for _ in range(1, min(num_clusters, len(programs))):
            max_min_dist = -1
            best_vec = None

            for program in programs:
                vec = program.feature_vector
                # Skip if already a centroid
                if any(vectors_equal(vec, c) for c in centroids):
                    continue
                # Find min distance to existing centroids
                min_dist = min(cosine_distance(c, vec) for c in centroids)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_vec = vec

            if best_vec:
                centroids.append(normalize_vector(best_vec))

        return centroids

    def _initialize_state_from_clusters(
        self, clusters: Dict[str, List[Program]], island_id: int
    ) -> None:
        """Initialize centroid state from existing cluster structure."""
        self._island_centroids[island_id] = []
        self._island_cluster_sums[island_id] = []
        self._island_cluster_counts[island_id] = []

        for cluster_id in sorted(clusters.keys()):
            programs = clusters[cluster_id]
            if programs:
                vectors = [p.feature_vector for p in programs if p.feature_vector]
                if vectors:
                    sum_vec = compute_sum_vector(vectors)
                    self._island_cluster_sums[island_id].append(sum_vec)
                    self._island_centroids[island_id].append(normalize_vector(sum_vec))
                    self._island_cluster_counts[island_id].append(len(vectors))
                else:
                    self._island_cluster_sums[island_id].append([])
                    self._island_centroids[island_id].append([])
                    self._island_cluster_counts[island_id].append(0)
            else:
                self._island_cluster_sums[island_id].append([])
                self._island_centroids[island_id].append([])
                self._island_cluster_counts[island_id].append(0)

    def _update_state_from_clusters(
        self, clusters: Dict[str, List[Program]], island_id: int
    ) -> None:
        """Update centroid state after offline re-clustering."""
        self._initialize_state_from_clusters(clusters, island_id)

    def _flatten_population(
        self, population: Dict[str, List[Program]]
    ) -> List[Program]:
        """Flatten population dict into a list of unique programs."""
        seen_ids = set()
        programs = []
        for bucket_programs in population.values():
            for program in bucket_programs:
                if program.id not in seen_ids:
                    seen_ids.add(program.id)
                    programs.append(program)
        return programs

    def _extract_clusters(
        self, population: Dict[str, List[Program]]
    ) -> Dict[str, List[Program]]:
        """Extract existing cluster structure from population."""
        clusters = {}
        for bucket_id, programs in population.items():
            if bucket_id.startswith("cluster_"):
                clusters[bucket_id] = list(programs)
        return clusters

    def _enforce_capacity(
        self, clusters: Dict[str, List[Program]], island_size: int
    ) -> Dict[str, List[Program]]:
        """
        Enforce overall capacity by keeping top island_size programs globally.

        Removes lowest-scoring programs across all clusters until total count
        equals island_size, while maintaining cluster assignments.
        """
        # Collect all programs with their cluster assignments
        all_programs = []
        for cluster_id, programs in clusters.items():
            for program in programs:
                all_programs.append((program, cluster_id))

        total = len(all_programs)
        if total <= island_size:
            return clusters

        # Sort by score descending and keep top island_size
        all_programs.sort(key=lambda x: x[0].combined_score or 0.0, reverse=True)
        keep_programs = all_programs[:island_size]

        # Rebuild clusters with kept programs
        new_clusters: Dict[str, List[Program]] = {
            cluster_id: [] for cluster_id in clusters.keys()
        }
        for program, cluster_id in keep_programs:
            new_clusters[cluster_id].append(program)

        return new_clusters

    def _log_cluster_stats(
        self, clusters: Dict[str, List[Program]], island_id: int
    ) -> None:
        """Log statistics about cluster distribution."""
        if not self.logger:
            return

        cluster_sizes = []
        cluster_scores = []
        for _, programs in sorted(clusters.items()):
            cluster_sizes.append(len(programs))
            if programs:
                scores = [p.combined_score for p in programs if p.combined_score]
                if scores:
                    cluster_scores.append(max(scores))

        total = sum(cluster_sizes)
        non_empty = sum(1 for s in cluster_sizes if s > 0)
        best_score = max(cluster_scores) if cluster_scores else 0

        self.log_info(
            f"Island {island_id}: Clusters {non_empty}/{len(clusters)} non-empty, "
            f"total={total}, sizes={cluster_sizes}, best={best_score:.4f}"
        )

    def _compute_cluster_diversity(self, programs: List[Program]) -> float:
        """
        Compute diversity (average pairwise cosine distance) within a cluster.

        Args:
            programs: List of programs in the cluster

        Returns:
            Average pairwise cosine distance, or 0.0 if insufficient programs
        """
        if len(programs) < 2:
            return 0.0

        # Extract feature vectors
        vectors = [p.feature_vector for p in programs if p.feature_vector]
        if len(vectors) < 2:
            return 0.0

        # Compute average pairwise cosine distance
        total_dist = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_dist += cosine_distance(vectors[i], vectors[j])
                count += 1

        return total_dist / count if count > 0 else 0.0
