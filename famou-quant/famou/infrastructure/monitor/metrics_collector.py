"""
Metrics collector for evolutionary experiments.

This module provides utilities to collect metrics from evolutionary experiments
in a decoupled, non-invasive way. It works with any experiment that provides
basic population data structures.

Key design principles:
- Decoupled: No direct dependencies on Experiment class
- Type-agnostic: Works with any population-like structure
- Extensible: Easy to add new metrics
- Efficient: Minimal overhead on experiment execution
"""

import logging
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class ProgramData:
    """
    Lightweight program data container.

    This class provides a simple interface for metrics collection
    without requiring dependency on the full Program class.
    """
    id: str
    score: Optional[float] = None
    generation: int = 0
    parent_id: Optional[str] = None
    code: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    feature_vector: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgramData':
        """Create from dictionary."""
        return cls(
            id=data.get('id', ''),
            score=data.get('combined_score') or data.get('score'),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id'),
            code=data.get('code'),
            metadata=data.get('meta') or data.get('metadata')
        )

    @classmethod
    def from_program(cls, program: Any) -> 'ProgramData':
        """
        Create from a Program object (duck typing).

        Works with any object that has the expected attributes.
        """
        return cls(
            id=str(getattr(program, 'id', '')),
            score=getattr(program, 'combined_score', None),
            generation=getattr(program, 'generation', 0),
            parent_id=getattr(program, 'parent_id', None),
            code=getattr(program, 'code', None),
            metadata=getattr(program, 'meta', None),
            feature_vector=getattr(program, 'feature_vector', None)
        )


class PerformanceMetrics:
    """Performance metrics collector."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def collect(
        self,
        programs: Union[List[ProgramData], List[Any]],
        iteration: int
    ) -> Dict[str, float]:
        """
        Collect performance metrics from a list of programs.

        Args:
            programs: List of ProgramData or Program-like objects
            iteration: Current iteration number

        Returns:
            Dictionary of metrics with MetricPrefix-style naming
        """
        # Convert to ProgramData if needed
        if programs and not isinstance(programs[0], ProgramData):
            programs = [ProgramData.from_program(p) for p in programs]

        # Extract scores (filter out None)
        scores = [p.score for p in programs if p.score is not None]

        if not scores:
            self._logger.warning("No valid scores found for performance metrics")
            return {
                "iteration/iteration": iteration,
                "performance/best_score": 0.0,
                "performance/avg_score": 0.0,
                "performance/median_score": 0.0,
                "performance/std_score": 0.0,
                "performance/p10_score": 0.0,
                "performance/p90_score": 0.0,
            }

        # Calculate statistics
        import numpy as np

        metrics = {
            "iteration/iteration": iteration,
            "performance/best_score": float(np.max(scores)),
            "performance/avg_score": float(np.mean(scores)),
            "performance/median_score": float(np.median(scores)),
            "performance/std_score": float(np.std(scores)),
            "performance/p10_score": float(np.percentile(scores, 10)),
            "performance/p90_score": float(np.percentile(scores, 90)),
            "performance/min_score": float(np.min(scores)),
        }

        return metrics


class PopulationMetrics:
    """Population metrics collector."""

    def collect(
        self,
        programs: Union[List[ProgramData], List[Any]],
        iteration: int
    ) -> Dict[str, float]:
        """
        Collect population metrics.

        Args:
            programs: List of ProgramData or Program-like objects
            iteration: Current iteration number

        Returns:
            Dictionary of population metrics
        """
        # Convert to ProgramData if needed
        if programs and not isinstance(programs[0], ProgramData):
            programs = [ProgramData.from_program(p) for p in programs]

        # Basic stats
        total_size = len(programs)
        generations = [p.generation for p in programs]

        import numpy as np

        metrics = {
            "iteration/iteration": iteration,
            "population/size": total_size,
            "population/avg_generation": float(np.mean(generations)) if generations else 0.0,
            "population/max_generation": float(np.max(generations)) if generations else 0.0,
            "population/min_generation": float(np.min(generations)) if generations else 0.0,
        }

        # Count unique programs (by code)
        unique_codes = set()
        for p in programs:
            if p.code:
                unique_codes.add(p.code)

        metrics["population/unique_programs"] = len(unique_codes)

        # Count unique IDs (should equal size)
        unique_ids = set(p.id for p in programs)
        metrics["population/unique_ids"] = len(unique_ids)

        return metrics


class DiversityMetrics:
    """Diversity metrics collector."""

    def collect(
        self,
        programs: Union[List[ProgramData], List[Any]],
        iteration: int
    ) -> Dict[str, float]:
        """
        Collect diversity metrics.

        Args:
            programs: List of ProgramData or Program-like objects
            iteration: Current iteration number

        Returns:
            Dictionary of diversity metrics
        """
        # Convert to ProgramData if needed
        if programs and not isinstance(programs[0], ProgramData):
            programs = [ProgramData.from_program(p) for p in programs]

        if not programs:
            return {
                "iteration/iteration": iteration,
                "diversity/genetic_diversity": 0.0,
                "diversity/code_diversity": 0.0,
                "diversity/semantic_diversity": 0.0,
            }

        metrics = {"iteration/iteration": iteration}

        # Genetic diversity (different generations)
        generations = [p.generation for p in programs]
        unique_generations = len(set(generations))
        genetic_diversity = unique_generations / len(generations) if generations else 0.0
        metrics["diversity/genetic_diversity"] = genetic_diversity

        # Parent diversity (different parents)
        parents = [p.parent_id for p in programs if p.parent_id]
        if parents:
            unique_parents = len(set(parents))
            parent_diversity = unique_parents / len(parents)
            metrics["diversity/parent_diversity"] = parent_diversity
        else:
            metrics["diversity/parent_diversity"] = 0.0

        # Semantic diversity (average pairwise cosine distance of feature vectors)
        semantic_diversity = self._compute_semantic_diversity(programs)
        metrics["diversity/semantic_diversity"] = semantic_diversity

        return metrics

    def _compute_semantic_diversity(self, programs: List[Any]) -> float:
        """
        Compute semantic diversity as average pairwise cosine distance.

        Uses feature vectors of programs to measure diversity. Higher values
        indicate more diverse populations in terms of semantic/behavioral differences.

        Args:
            programs: List of programs to analyze

        Returns:
            Average pairwise cosine distance in range [0, 2]:
            - 0 = all programs are semantically identical
            - 1 = moderately diverse
            - 2 = maximally diverse (opposite directions)
        """
        if len(programs) < 2:
            return 0.0

        # Extract feature vectors from programs
        vectors = [p.feature_vector for p in programs if hasattr(p, 'feature_vector') and p.feature_vector]

        if len(vectors) < 2:
            return 0.0

        # Compute average pairwise cosine distance
        total_dist = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_dist += self._cosine_distance(vectors[i], vectors[j])
                count += 1

        return total_dist / count if count > 0 else 0.0

    def _cosine_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine distance between two vectors.

        Cosine distance = 1 - cosine_similarity
        Cosine similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine distance in range [0, 2], where 0 means identical direction
        """
        import math

        if not vec1 or not vec2:
            return 1.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 1.0

        return 1.0 - (dot / (norm1 * norm2))


class CostTracker:
    """
    LLM API cost tracker.

    Tracks token usage and calculates costs for LLM API calls.
    Completely decoupled from any specific LLM client implementation.
    """

    # Default costs per million tokens (as of 2024)
    # These can be overridden on initialization
    DEFAULT_COSTS = {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "default": {"input": 1.0, "output": 2.0},
    }

    def __init__(
        self,
        model_name: str = "default",
        input_cost_per_1m: Optional[float] = None,
        output_cost_per_1m: Optional[float] = None
    ):
        """
        Initialize cost tracker.

        Args:
            model_name: Model name for cost lookup
            input_cost_per_1m: Override input cost per million tokens
            output_cost_per_1m: Override output cost per million tokens
        """
        self.model_name = model_name

        # Get costs from defaults or use overrides
        default_costs = self.DEFAULT_COSTS.get(model_name, self.DEFAULT_COSTS["default"])
        self.input_cost_per_1m = input_cost_per_1m or default_costs["input"]
        self.output_cost_per_1m = output_cost_per_1m or default_costs["output"]

        # Tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

        self._logger = logging.getLogger(__name__)

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
        model_name: Optional[str] = None
    ) -> None:
        """
        Record a single LLM API call.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_name: Optional model name (for cost calculation)

        Example:
            >>> tracker = CostTracker(model_name="gpt-4")
            >>> tracker.record_call(input_tokens=100, output_tokens=50)
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_calls += 1

        self._logger.debug(
            f"LLM call recorded: {input_tokens} input + {output_tokens} output tokens"
        )

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current cost metrics.

        Returns:
            Dictionary with cost metrics
        """
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_cost_per_1m
        total_cost = input_cost + output_cost

        return {
            "cost/total_calls": self.total_calls,
            "cost/input_tokens": self.total_input_tokens,
            "cost/output_tokens": self.total_output_tokens,
            "cost/total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost/input_cost_usd": input_cost,
            "cost/output_cost_usd": output_cost,
            "cost/total_cost_usd": total_cost,
            "cost/avg_tokens_per_call": (
                (self.total_input_tokens + self.total_output_tokens) / self.total_calls
                if self.total_calls > 0 else 0
            ),
            "cost/avg_cost_per_call": total_cost / self.total_calls if self.total_calls > 0 else 0,
        }

    def reset(self) -> None:
        """Reset all tracking counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    @property
    def total_cost_usd(self) -> float:
        """Get total cost in USD."""
        input_cost = (self.total_input_tokens / 1_000_000) * self.input_cost_per_1m
        output_cost = (self.total_output_tokens / 1_000_000) * self.output_cost_per_1m
        return input_cost + output_cost


class EvolutionMetricsCollector:
    """
    Unified metrics collector for evolutionary experiments.

    Combines all metric collectors into a single convenient interface.

    Usage:
        >>> collector = EvolutionMetricsCollector()
        >>>
        >>> # Collect all metrics at once
        >>> metrics = collector.collect_all(programs, iteration=5)
        >>> monitor.log_metrics(metrics, step=5)
        >>>
        >>> # Or collect specific metric types
        >>> perf_metrics = collector.performance.collect(programs, iteration=5)
    """

    def __init__(self):
        """Initialize all metric collectors."""
        self.performance = PerformanceMetrics()
        self.population = PopulationMetrics()
        self.diversity = DiversityMetrics()

        self._logger = logging.getLogger(__name__)

    def collect_all(
        self,
        programs: Union[List[ProgramData], List[Any]],
        iteration: int
    ) -> Dict[str, float]:
        """
        Collect all metrics at once.

        Args:
            programs: List of ProgramData or Program-like objects
            iteration: Current iteration number

        Returns:
            Combined dictionary of all metrics
        """
        all_metrics = {}

        # Collect performance metrics
        perf_metrics = self.performance.collect(programs, iteration)
        all_metrics.update(perf_metrics)

        # Collect population metrics (don't duplicate iteration)
        pop_metrics = self.population.collect(programs, iteration)
        pop_metrics.pop("iteration/iteration", None)  # Remove duplicate
        all_metrics.update(pop_metrics)

        # Collect diversity metrics (don't duplicate iteration)
        div_metrics = self.diversity.collect(programs, iteration)
        div_metrics.pop("iteration/iteration", None)  # Remove duplicate
        all_metrics.update(div_metrics)

        self._logger.debug(
            f"Collected {len(all_metrics)} metrics for iteration {iteration}"
        )

        return all_metrics

    def collect_from_experiment(
        self,
        experiment: Any,
        iteration: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Collect metrics from an Experiment object (duck typing).

        This method is decoupled and works with any experiment-like object
        that has a get_all_programs() method returning a list of programs.

        Args:
            experiment: Experiment object with get_all_programs() method
            iteration: Iteration number (extracted from experiment if not provided)

        Returns:
            Dictionary of metrics
        """
        # Get programs from experiment (duck typing)
        if hasattr(experiment, 'get_all_programs'):
            programs = experiment.get_all_programs()
        else:
            self._logger.warning("Experiment has no get_all_programs() method")
            return {}

        # Get iteration number
        if iteration is None:
            if hasattr(experiment, 'current_iteration'):
                iteration = experiment.current_iteration
            else:
                iteration = 0
                self._logger.warning("Cannot determine iteration, using 0")

        return self.collect_all(programs, iteration)


class IslandMetrics:
    """
    Multi-island metrics collector.

    Collects per-island metrics for multi-island experiments.
    """

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def collect_per_island_metrics(
        self,
        experiment,
        iteration: int
    ) -> Dict[str, float]:
        """
        Collect per-island metrics.

        Args:
            experiment: Experiment object with island populations
            iteration: Current iteration number

        Returns:
            Dictionary with per-island metrics
        """
        num_islands = experiment.num_islands
        metrics = {
            "iteration/iteration": iteration,
            f"islands/num_islands": num_islands,
        }

        # Collect metrics for each island
        for island_id in range(num_islands):
            island_pop = experiment.get_island_population(island_id)

            # Flatten population: get all programs from all buckets
            island_programs = [
                p for bucket_progs in island_pop.values()
                for p in bucket_progs
            ]

            if not island_programs:
                # Empty island
                metrics[f"islands/island_{island_id}/size"] = 0
                metrics[f"islands/island_{island_id}/best_score"] = 0.0
                metrics[f"islands/island_{island_id}/avg_score"] = 0.0
                metrics[f"islands/island_{island_id}/best_generation"] = 0
                continue

            # Extract scores
            scores = [p.combined_score for p in island_programs if p.combined_score is not None]
            generations = [p.generation for p in island_programs]

            import numpy as np

            # Basic metrics
            metrics[f"islands/island_{island_id}/size"] = len(island_programs)
            metrics[f"islands/island_{island_id}/best_score"] = float(np.max(scores)) if scores else 0.0
            metrics[f"islands/island_{island_id}/avg_score"] = float(np.mean(scores)) if scores else 0.0
            metrics[f"islands/island_{island_id}/median_score"] = float(np.median(scores)) if scores else 0.0
            metrics[f"islands/island_{island_id}/std_score"] = float(np.std(scores)) if scores else 0.0
            metrics[f"islands/island_{island_id}/best_generation"] = int(np.max(generations)) if generations else 0
            metrics[f"islands/island_{island_id}/avg_generation"] = float(np.mean(generations)) if generations else 0.0

            # Find best program for this island
            if scores:
                best_idx = np.argmax(scores)
                best_program = island_programs[best_idx]
                metrics[f"islands/island_{island_id}/best_program_id"] = best_program.id
                metrics[f"islands/island_{island_id}_id"] = best_program.id  # For easy access

        # Add global best across all islands
        all_best_scores = [
            metrics.get(f"islands/island_{i}/best_score", 0.0)
            for i in range(num_islands)
        ]
        metrics["islands/global_best_score"] = float(np.max(all_best_scores))
        metrics["islands/best_island"] = int(np.argmax(all_best_scores))

        # Total population across all islands
        total_population = sum(
            metrics.get(f"islands/island_{i}/size", 0)
            for i in range(num_islands)
        )
        metrics["islands/total_population"] = total_population

        self._logger.debug(
            f"Collected per-island metrics for {num_islands} islands",
            iteration=iteration,
            total_population=total_population
        )

        return metrics
