"""
Storage infrastructure for Famou 2.0.

Provides:
- DataService: Protocol for data persistence
- LocalStorage: Local file-based implementation
- DualWriteLocalStorage: Dual-write to both data and system paths
"""

from famou.infrastructure.storage.base import DataService
from famou.infrastructure.storage.local_storage import LocalStorage
from famou.infrastructure.storage.dual_write_storage import DualWriteLocalStorage

__all__ = [
    "DataService",
    "LocalStorage",
    "DualWriteLocalStorage",
]
