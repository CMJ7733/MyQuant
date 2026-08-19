"""Evaluation modules for Famou 2.0."""

from typing import TYPE_CHECKING

from famou.modules.evaluate.base import EvaluateModule

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from famou.modules.evaluate.hybrid import HybridEvaluateModule

__all__ = [
    "EvaluateModule",
    "HybridEvaluateModule",
]


def __getattr__(name: str):
    """Resolve the remote evaluator on first access (PEP 562).

    ``hybrid`` imports ``famou_sdk``, which only exists inside the internal
    environment and is only needed when ``infrastructure.evaluator.type == "hybrid"``.
    Importing it eagerly made the whole ``famou.modules`` package -- and therefore
    every local-mode run, including the qlib experiments -- fail to import on a
    machine without the SDK. ``from famou.modules.evaluate import HybridEvaluateModule``
    still works wherever the SDK is installed; only the moment of the import moved.
    """
    if name == "HybridEvaluateModule":
        from famou.modules.evaluate.hybrid import HybridEvaluateModule

        return HybridEvaluateModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
