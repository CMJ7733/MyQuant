"""Backend infrastructure module for FAMOU system.

This module provides backend execution capabilities including:
- ThreadPoolBackend: Thread pool based backend executor
- Other backend implementations as needed
"""

from .base import BackendExecutor, ResultHandle, BackendTask
from .threadpool_backend import ThreadPoolBackend

try:
    from .ray_backend import RayBackend
except ImportError:
    RayBackend = None

__all__ = [
    'BackendExecutor',
    'ResultHandle',
    'BackendTask',
    'ThreadPoolBackend',
    'RayBackend',
]