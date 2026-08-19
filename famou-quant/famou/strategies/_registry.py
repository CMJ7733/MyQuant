"""
Strategy Registry for Famou 2.0.

Pure file-based registry with auto-discovery from directories.
Strategies are shared via Git commit/push/pull.

Supports two patterns for strategy creation:
1. Old style (dict with "rollout" key) - backward compatible
2. New style (dict with "strategy" key) - Strategy ABC objects

Usage:
    # Get a strategy (works with both old and new style)
    >>> from famou.strategies import StrategyRegistry
    >>> strategy = StrategyRegistry.get("standard", evaluate_fn=my_eval_fn)
    >>> strategy.forward(ctx)  # Get rollout for current iteration
    >>> strategy.population_module  # PopulationModule

    # List all available strategies
    >>> StrategyRegistry.list_strategies()
    ['standard', 'adaptive', ...]

    # Use in experiment
    >>> evolver = Evolver(experiment=experiment, config=config, ...)
    >>> evolver.run(strategy)  # Pass strategy directly
"""

import importlib.util
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from famou.config.settings import ModulesConfig, EvaluatorConfig
from famou.core.data import RolloutResult, WorkBatch
from famou.core.protocol import Rollout, Context
from famou.modules import PopulationModule

# Avoid circular imports
if TYPE_CHECKING:
    from famou.core.protocol import Strategy as StrategyABC

# Default strategy directory (absolute path based on this file's location)
_DEFAULT_STRATEGY_DIR = Path(__file__).parent


@dataclass
class StrategyMetadata:
    """
    Metadata about a strategy (for discovery and documentation).

    Metadata is stored separately from Strategy objects to maintain
    clean separation of concerns. Strategy objects contain only
    execution logic, while metadata is managed by Registry.

    Attributes:
        name: Strategy name
        description: Human-readable description
        tags: Categorization tags (e.g., ["exploitation", "adaptive"])
        author: Optional author name
    """
    name: str
    description: str = ""
    tags: List[str] = None
    author: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


# Import Strategy ABC at runtime to avoid circular imports
def _get_strategy_abc() -> type:
    from famou.core.protocol import Strategy as StrategyABC
    return StrategyABC


class StaticStrategy:
    """
    Wrapper that converts old dict-based strategies to Strategy ABC.

    Always returns the same rollout (static decision).
    This is used internally by StrategyRegistry for backward compatibility.

    Attributes:
        name: Strategy name
        rollout: Rollout pipeline to execute
        population_module: Population management strategy
        evaluate_fn: Evaluator function for program execution
        description: Human-readable description
        tags: Categorization tags
        author: Author name
    """

    def __init__(
        self,
        name: str,
        rollout: Rollout,
        population_module: PopulationModule,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        description: str = "",
        tags: List[str] = None,
        author: str = ""
    ):
        self.name = name
        self.rollout = rollout
        self.population_module = population_module
        self.evaluate_fn = evaluate_fn
        self.description = description
        self.tags = tags or []
        self.author = author
        self.state_store = None

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        """
        Always return the same rollout (static strategy).

        Args:
            ctx: Current context (ignored for static strategies)
            rollout_history: Recent rollout history (ignored for static strategies)

        Returns:
            The configured rollout
        """
        return self.rollout

    def forward_batch(
        self,
        ctx: Context,
        rollout_history: List[RolloutResult],
        max_batch_size: int = 1,
    ) -> WorkBatch:
        """Wrap :meth:`forward` as a batch of one.

        ``Evolver._plan_rollout_tasks`` dispatches exclusively through
        ``forward_batch``. The Strategy protocol documents this default ("the
        default implementation wraps ``forward()`` as a batch of one, so every
        existing Strategy keeps working unchanged"), but ``Strategy`` is a
        ``typing.Protocol`` whose bodies are ``...`` and ``StaticStrategy``
        inherits from nothing, so the promised default did not exist anywhere
        and every registered strategy raised ``AttributeError`` on the first
        iteration.

        A static strategy returns the same rollout regardless of context, so
        there is nothing meaningful to fan out: honouring ``max_batch_size`` by
        emitting N copies would run the identical decision N times. One rollout
        per decision is both correct and what these strategies already assumed.
        """
        return WorkBatch.single(
            self.forward(ctx, rollout_history), strategy=self.name
        )

    def dump_state(self) -> Dict[str, Any]:
        """Dump strategy-level state for checkpoint persistence."""
        return {}


    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy-level state from checkpoint."""
        pass

    # -- rollout lifecycle ---------------------------------------------------
    #
    # The Evolver calls these on every dispatched rollout. They exist to let a
    # stateful strategy commit or roll back whatever ``forward()`` reserved. A
    # static strategy reserves nothing -- it returns the same rollout every
    # time -- so the protocol's documented no-op is the correct behaviour, and
    # the only reason to spell it out here is that ``Strategy`` is a
    # ``typing.Protocol`` (bodies are ``...``) that ``StaticStrategy`` does not
    # inherit from, so nothing supplied these defaults and the Evolver raised
    # ``AttributeError`` the first time a rollout finished.

    def on_rollout_failed(self, result: RolloutResult) -> None:
        """No-op: a static strategy holds no per-rollout state to roll back."""
        del result

    def on_rollout_complete(self, result: RolloutResult) -> None:
        """No-op: a static strategy holds no per-rollout state to commit."""
        del result

    def finalize_experiment(self, contexts: Dict[int, Context]) -> None:
        """No-op: a static strategy has no runtime state to flush."""
        del contexts

    def __str__(self) -> str:
        """String representation."""
        modules_str = " → ".join(m.name for m in self.rollout.modules)
        return f"{self.name}: {modules_str} + {self.population_module.name}"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"StaticStrategy(name='{self.name}', "
            f"rollout={len(self.rollout.modules)} modules, "
            f"population={self.population_module.name})"
        )


class StrategyRegistry:
    """
    Pure file-based strategy registry with auto-discovery.

    Scans strategy directory for .py files with create_strategy() function.
    Supports two patterns for strategy creation:
    - Old style: {"rollout": ..., "population_module": ...} → StaticStrategy
    - New style: {"strategy": ...} → Strategy ABC object

    No in-memory caching - always reads from disk (simple & reliable).

    Default strategy directory: famou/strategies/ (resolved via __file__)

    To add a new strategy:
    1. Create file: famou/strategies/my_strategy.py
    2. Define create_strategy() function that returns strategy dict
    3. Commit to Git: git add, git commit, git push
    4. Others pull: git pull
    5. Use: StrategyRegistry.get("my_strategy")
    """

    # Strategy directories to scan (use absolute paths)
    _strategy_dirs: List[Path] = [_DEFAULT_STRATEGY_DIR]

    # Metadata storage (separate from Strategy objects)
    _metadata: Dict[str, StrategyMetadata] = {}

    @classmethod
    def get(
        cls,
        name: str,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
    ) -> "StrategyABC":
        """
        Load strategy from file with optional evaluator and params.

        Searches strategy directories for {name}.py file and loads it.
        The strategy's create_strategy() function receives evaluate_fn and params
        to configure modules.

        Supports two patterns (auto-detected):
        1. Old style: Returns {"rollout": ..., "population_module": ...}
           → Wrapped in StaticStrategy
        2. New style: Returns {"strategy": StrategyABC(...)}
           → Returns Strategy ABC object directly

        Args:
            name: Strategy name (filename without .py extension)
            evaluate_fn: Evaluator function to inject into EvaluateModule
            params: Module parameters from config (overrides strategy defaults)
            evaluator_config: EvaluatorConfig for HybridEvaluateModule (remote evaluation)

        Returns:
            Strategy ABC instance (either StaticStrategy or custom Strategy)

        Raises:
            KeyError: If strategy not found
            TypeError: If strategy doesn't implement required interface

        Example:
            >>> # Works with both old and new style strategies
            >>> strategy = StrategyRegistry.get(
            ...     "standard",
            ...     evaluate_fn=my_evaluator,
            ...     params=config.modules,
            ... )
            >>> rollout = strategy.forward(ctx)  # Get rollout for current iteration
        """
        for directory in cls._strategy_dirs:
            strategy_file = Path(directory) / f"{name}.py"
            if strategy_file.exists():
                return cls._load_from_file(strategy_file, name, evaluate_fn, params, evaluator_config)

        # Not found - provide helpful error message
        available = cls.list_strategies()
        available_str = ", ".join(available) if available else "(none found)"
        raise KeyError(
            f"Strategy '{name}' not found.\n"
            f"Available strategies: {available_str}\n"
            f"Searched in: {cls._strategy_dirs}\n\n"
            f"To create a new strategy, see: famou/strategies/README.md"
        )

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        List all available strategies by scanning directories.

        Returns:
            List of strategy names (sorted)

        Example:
            >>> StrategyRegistry.list_strategies()
            ['standard', 'explore', 'my_custom']
        """
        strategies = []
        for directory in cls._strategy_dirs:
            path = Path(directory)
            if not path.exists():
                continue

            for strategy_file in path.glob("*.py"):
                # Skip __init__.py and private files
                if strategy_file.name.startswith("_"):
                    continue

                # Strategy name is filename without .py
                strategy_name = strategy_file.stem
                strategies.append(strategy_name)

        # Remove duplicates (custom overrides builtin), sort
        return sorted(set(strategies))

    @classmethod
    def list_all(cls) -> Dict[str, "StrategyABC"]:
        """
        Load all strategies with full details.

        Returns:
            Dictionary mapping names to Strategy ABC instances

        Example:
            >>> strategies = StrategyRegistry.list_all()
            >>> for name, strategy in strategies.items():
            ...     metadata = StrategyRegistry._metadata.get(name)
            ...     print(f"{name}: {metadata.description if metadata else 'N/A'}")
        """
        all_strategies = {}
        for name in cls.list_strategies():
            try:
                all_strategies[name] = cls.get(name)
            except Exception as e:
                # Skip strategies that fail to load
                print(f"Warning: Failed to load strategy '{name}': {e}")
        return all_strategies

    @classmethod
    def add_strategy_directory(cls, directory: str | Path) -> None:
        """
        Add a custom directory to scan for strategies.

        Useful for loading strategies from external locations.

        Args:
            directory: Path to directory containing strategy files

        Example:
            >>> StrategyRegistry.add_strategy_directory("/shared/team-strategies")
            >>> strategy = StrategyRegistry.get("team_best")
        """
        path = Path(directory).resolve()
        if path not in cls._strategy_dirs:
            cls._strategy_dirs.append(path)

    @classmethod
    def reset_strategy_directories(cls) -> None:
        """
        Reset to default strategy directory.

        Useful for testing or resetting state.
        """
        cls._strategy_dirs = [_DEFAULT_STRATEGY_DIR]

    @classmethod
    def get_metadata(cls, name: str) -> Optional[StrategyMetadata]:
        """
        Get metadata for a strategy without loading it.

        Args:
            name: Strategy name

        Returns:
            StrategyMetadata if available, None otherwise

        Example:
            >>> metadata = StrategyRegistry.get_metadata("standard")
            >>> if metadata:
            ...     print(f"{metadata.name}: {metadata.description}")
        """
        return cls._metadata.get(name)

    @classmethod
    def _load_from_file(
        cls,
        file_path: Path,
        name: str,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
        evaluator_config: Optional[EvaluatorConfig] = None,
    ) -> "StrategyABC":
        """
        Load strategy from Python file with auto-detection.

        The file must define a create_strategy() function that accepts
        evaluate_fn and params. Supports two patterns:

        Pattern 1 (Old style): Returns {"rollout": ..., "population_module": ...}
        Pattern 2 (New style): Returns {"strategy": StrategyABC(...)}

        Args:
            file_path: Path to strategy file
            name: Strategy name
            evaluate_fn: Evaluator function to inject
            params: Module parameters from config
            evaluator_config: EvaluatorConfig for HybridEvaluateModule

        Returns:
            Strategy ABC instance

        Raises:
            ValueError: If file doesn't define create_strategy()
            TypeError: If strategy doesn't implement required interface
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location(f"strategy_{name}", file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Failed to load strategy file: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check for create_strategy() function
        if not hasattr(module, "create_strategy"):
            raise ValueError(
                f"Strategy file {file_path} must define create_strategy() function.\n"
                f"See famou/strategies/README.md for template."
            )

        # Call factory function
        # Build evaluate module based on evaluator_config (dynamic selection in one place)
        evaluate_module = cls._build_evaluate_module(evaluator_config)

        result = module.create_strategy(
            evaluate_fn=evaluate_fn,
            params=params or ModulesConfig(),
            evaluate_module=evaluate_module,
        )

        # Must be a dict
        if not isinstance(result, dict):
            raise TypeError(
                f"create_strategy() must return dict. "
                f"Got {type(result).__name__}"
            )

        # Extract author from module docstring if not provided
        author = result.get("author", "")
        if not author and module.__doc__:
            for line in module.__doc__.split("\n"):
                if line.strip().startswith("Author:"):
                    author = line.split("Author:")[1].strip()
                    break
            result["author"] = author

        # Pattern 1: New style - dict contains "strategy" key (lowercase)
        if "strategy" in result:
            strategy_obj = result["strategy"]

            # Get Strategy ABC class
            StrategyABC = _get_strategy_abc()

            # Validate it's a Strategy ABC instance
            if not isinstance(strategy_obj, StrategyABC):
                raise TypeError(
                    f"'strategy' value must be a Strategy ABC instance. "
                    f"Got {type(strategy_obj).__name__}"
                )

            # Validate Strategy interface
            cls._validate_strategy(strategy_obj)

            # Store metadata separately (NOT on strategy object)
            metadata = StrategyMetadata(
                name=name,
                description=result.get("description", ""),
                tags=result.get("tags", []),
                author=result.get("author", "")
            )
            cls._metadata[name] = metadata

            # evaluate_fn handling
            if evaluate_fn is not None and not hasattr(strategy_obj, 'evaluate_fn'):
                strategy_obj.evaluate_fn = evaluate_fn

            return strategy_obj

        # Pattern 2: Old style - dict contains "rollout" key
        elif "rollout" in result:
            # Wrap in StaticStrategy for backward compatibility
            rollout = result["rollout"]
            population_module = result["population_module"]

            if not isinstance(rollout, Rollout):
                raise TypeError(
                    f"'rollout' must be a Rollout instance. "
                    f"Got {type(rollout).__name__}"
                )

            strategy_obj = StaticStrategy(
                name=name,
                rollout=rollout,
                population_module=population_module,
                evaluate_fn=evaluate_fn,
                description=result.get("description", ""),
                tags=result.get("tags", []),
                author=result.get("author", "")
            )

            # Store metadata for StaticStrategy too
            metadata = StrategyMetadata(
                name=name,
                description=result.get("description", ""),
                tags=result.get("tags", []),
                author=result.get("author", "")
            )
            cls._metadata[name] = metadata

            return strategy_obj

        else:
            raise ValueError(
                f"Invalid strategy dict. Must contain either 'strategy' or 'rollout' key.\n"
                f"Got keys: {list(result.keys())}"
            )

    @classmethod
    def _build_evaluate_module(
        cls,
        evaluator_config: Optional[EvaluatorConfig],
    ):
        """
        Build and return the appropriate evaluate module based on evaluator_config.

        Returns HybridEvaluateModule for hybrid mode, None for local mode.
        When None is returned, each strategy is responsible for creating
        EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params) itself.
        """
        if evaluator_config and evaluator_config.type == "hybrid":
            from famou.modules.evaluate.hybrid import HybridEvaluateModule
            return HybridEvaluateModule(config=evaluator_config, name="hybrid_evaluate")
        return None

    @classmethod
    def _validate_strategy(cls, strategy: "StrategyABC") -> None:
        """
        Validate that strategy implements required interface.

        Args:
            strategy: Strategy instance to validate

        Raises:
            TypeError: If strategy doesn't meet requirements
        """
        # Get Strategy ABC class
        StrategyABC = _get_strategy_abc()

        # Check it's a Strategy ABC subclass instance
        if not isinstance(strategy, StrategyABC):
            raise TypeError(
                f"Strategy must inherit from Strategy ABC. "
                f"Got {type(strategy).__name__}"
            )

        # Check required attributes
        if not hasattr(strategy, 'population_module'):
            raise TypeError(
                f"Strategy must have 'population_module' attribute. "
                f"Missing in {type(strategy).__name__}"
            )

        if not hasattr(strategy, 'forward'):
            raise TypeError(
                f"Strategy must have 'forward()' method. "
                f"Missing in {type(strategy).__name__}"
            )

        if not callable(strategy.forward):
            raise TypeError(
                f"Strategy's 'forward' must be callable. "
                f"Got type: {type(strategy.forward).__name__}"
            )


# Convenience function for creating strategies (old style)
def create_strategy(
    rollout_modules: List,
    population_module: PopulationModule,
    description: str,
    tags: Optional[List[str]] = None,
    author: Optional[str] = None,
) -> Dict:
    """
    Helper to create a strategy dictionary from components (old style).

    This creates a static strategy using the old pattern with "rollout" key.
    For new strategies, consider implementing Strategy ABC directly.

    Args:
        rollout_modules: List of modules for the rollout pipeline
        population_module: Population management module
        description: Human-readable description
        tags: Optional tags for categorization
        author: Optional author name

    Returns:
        Dictionary with "rollout" key (old style pattern)

    Example:
        >>> def create_strategy(evaluate_fn=None, params=None):
        ...     from famou.strategies import create_strategy as make_strategy
        ...     return make_strategy(
        ...         rollout_modules=[TopKSelect(k=1), MutationGenerate()],
        ...         population_module=TopKPopulation(),
        ...         description="My custom strategy",
        ...         tags=["exploitation"]
        ...     )
    """
    from famou.core.protocol import Rollout

    rollout = Rollout(modules=rollout_modules, name="custom_rollout")

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": description,
        "tags": tags or [],
        "author": author or "",
    }
