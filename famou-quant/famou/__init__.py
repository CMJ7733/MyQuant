"""
Famou 2.0 - Framework for Automated Machine-guided Optimization Unconstrained

Main exports for the Famou framework.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checkers and IDEs only
    from famou.strategies import StrategyRegistry

__all__ = [
    "StrategyRegistry",
]

__version__ = "2.0.0"


def __getattr__(name: str):
    """Resolve framework exports on first access (PEP 562).

    ``StrategyRegistry`` transitively imports every module, strategy, LLM client and
    execution environment -- effectively the whole framework and its dependency tree.
    Doing that eagerly at package import meant ``import famou.monitor`` could not
    succeed unless all of it imported cleanly, which defeats the monitor's reason to
    exist: reading a finished run's artifacts on a machine that has the artifacts and
    little else.

    ``from famou import StrategyRegistry`` and ``famou.StrategyRegistry`` both still
    work and still import the same object; only the moment of the import moved.
    """
    if name == "StrategyRegistry":
        from famou.strategies import StrategyRegistry

        return StrategyRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
