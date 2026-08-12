"""
Execution environment infrastructure for Famou 2.0.

Provides:
- ExecutionEnvironment protocol (base interface)
- EnvFactory: Factory for creating famou_sdk environments

Execution environments are created via EnvFactory.create(config), which
directly uses famou_sdk's ProcessEnv and ThreadEnv.
"""

from famou.infrastructure.env.base import ExecutionEnvironment
from famou.infrastructure.env.env_factory import EnvFactory

__all__ = [
    "ExecutionEnvironment",
    "EnvFactory",
]
