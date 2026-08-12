"""
Population management modules for Famou 2.0.

Base class:
- PopulationModule: Inherit to create custom population management strategies

Built-in implementations (import from files directly):
- full_archive.FullArchivePopulation: Keep all programs without pruning (default)
- topk.TopKPopulation: Keep top-K programs by combined_score
- cluster.ClusterPopulation: Cluster-based diversity maintenance with k-means
- age_layered.AgeLayeredPopulation: Age-based layers (ALPS-inspired)
- pareto.ParetoPopulation: Multi-objective Pareto front optimization
"""

from famou.modules.population.base import PopulationModule

__all__ = ["PopulationModule"]
