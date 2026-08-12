"""Thinking Evolution, Adaptive Generation, pools, and the search loop (§3.5-§3.6)."""

from cogalpha.evolution.operators import (  # noqa: F401
    OPERATORS,
    EvolutionStats,
    ThinkingEvolution,
    format_metrics,
)
from cogalpha.evolution.adaptive import AdaptiveGeneration, Feedback  # noqa: F401
from cogalpha.evolution.pool import (  # noqa: F401
    CandidatePool,
    PlateauStopper,
    build_parent_pool,
    generation_elite_score,
)
from cogalpha.evolution.loop import CogAlphaSearch, SearchResult  # noqa: F401

__all__ = [
    "OPERATORS",
    "EvolutionStats",
    "ThinkingEvolution",
    "format_metrics",
    "AdaptiveGeneration",
    "Feedback",
    "CandidatePool",
    "PlateauStopper",
    "build_parent_pool",
    "generation_elite_score",
    "CogAlphaSearch",
    "SearchResult",
]
