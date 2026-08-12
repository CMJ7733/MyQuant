"""
Selection modules for Famou 2.0.

Base class:
- SelectModule: Inherit to create custom selection strategies

Built-in implementations:
- elite.EliteSelect: Select best program by score (recommended)
- random.RandomSelect: Random parent selection
- tournament.TournamentSelect: Tournament selection (k-way competition)
- diversity.DiversitySelect: Diversity-maximizing selection
- cluster_adaptive.ClusterAdaptiveSelect: Cluster-aware adaptive selection

Backwards compatibility:
- TopKSelect is an alias for EliteSelect
"""

from famou.modules.select.base import SelectModule

__all__ = ["SelectModule"]
