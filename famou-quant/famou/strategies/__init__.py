"""
Strategies module for Famou 2.0.

Strategies combine rollout pipelines with population modules.
For type hints, import Strategy ABC from famou.core.protocol.
"""

from famou.strategies._registry import StrategyRegistry, create_strategy

__all__ = ["StrategyRegistry", "create_strategy"]
