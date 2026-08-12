"""
Logger infrastructure for Famou 2.0.

Provides logging interfaces and implementations:
- Logger: Protocol defining the logger interface
- LocalLogger: Multi-output logger (console + .log + .jsonl)
"""

from famou.infrastructure.logger.base import Logger
from famou.infrastructure.logger.local_logger import LocalLogger

__all__ = [
    "Logger",
    "LocalLogger",
]
