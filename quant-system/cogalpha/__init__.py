"""CogAlpha: cognitive alpha mining via LLM-driven code-based evolution.

Implementation of the ACL 2026 paper *Cognitive Alpha Mining via LLM-Driven
Code-Based Evolution* (Liu et al.), archived under ``/root/quant/cogalpha``.

The five components of the paper map onto packages as follows:

======================================  ==================================
Paper component                         Package
======================================  ==================================
Seven-Level Agent Hierarchy (§3.1)      :mod:`cogalpha.agents.hierarchy`
Diversified Guidance (§3.2)             :mod:`cogalpha.agents.guidance`
Multi-Agent Quality Checker (§3.3)      :mod:`cogalpha.quality`
Fitness Evaluation, 5 metrics (§3.4)    :mod:`cogalpha.fitness`
Adaptive Generation (§3.5)              :mod:`cogalpha.evolution.adaptive`
Thinking Evolution (§3.6)               :mod:`cogalpha.evolution.operators`
======================================  ==================================

The working stream of Figure 1 is driven by :func:`cogalpha.evolution.loop.run_search`.
"""

from cogalpha.types import (  # noqa: F401
    Alpha,
    AlphaTier,
    CheckReport,
    CheckStage,
    Fitness,
    GenerationRecord,
    Lineage,
)
from cogalpha.config import CogAlphaConfig, load_config  # noqa: F401

__version__ = "0.1.0"

__all__ = [
    "Alpha",
    "AlphaTier",
    "CheckReport",
    "CheckStage",
    "Fitness",
    "GenerationRecord",
    "Lineage",
    "CogAlphaConfig",
    "load_config",
    "__version__",
]
