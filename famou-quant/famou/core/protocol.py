"""
Module protocol and dependency injection for Famou 2.0.

Defines:
- Module: Base class for all algorithm components
- Protocol mixins for dependency injection (RequiresLLM, RequiresEnv)

Key Design Pattern: Program Enrichment Pipeline (Simplified)
-------------------------------------------------------------
**One Program Per Rollout**

Programs flow through modules and are gradually enriched with data:

1. SelectModule: population → selected_parent_id (single ID, str)
2. GenerateModule: selected_parent_id → generated_program (ONE Program)
3. EvaluateModule: generated_program (ENRICHES IN PLACE with metrics, validity, combined_score, error_info)
4. JudgeModule: generated_program (ENRICHES IN PLACE with llm_feedback)

Each module either:
- Routes Programs (SelectModule): Selects one parent ID, stores in selected_parent_id
- Creates new Program (GenerateModule): Looks up parent, creates one child
- Enriches existing Program IN PLACE (EvaluateModule, JudgeModule): No separate output

Inspired by gigaevo's Stage abstraction - modules are composable, reorderable
components that transform RolloutResult through the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Protocol, TYPE_CHECKING, runtime_checkable

from famou.core.data import Context, Program, Rollout, RolloutResult, WorkBatch
from famou.core.types import ModuleValidationError


# Avoid circular imports
if TYPE_CHECKING:
    from famou.infrastructure.llm.base import LLMClient
    from famou.infrastructure.env.base import ExecutionEnvironment
    from famou.infrastructure.embedding.base import EmbeddingClient
    from famou.infrastructure.logger.base import Logger
    from famou.core.state import StateStore


class Module(ABC):
    """
    Base class for all algorithm modules.

    Modules are the building blocks of the evolutionary pipeline.
    Each module transforms RolloutContext by either:
    1. Creating new Programs
    2. Enriching existing Programs
    3. Routing Programs between context fields

    Pipeline Data Flow Pattern:
    ---------------------------
    context.accessor
        ↓
    [SelectModule] → result.selected_parent_id (str) + result.inspiration_ids (List[str])
        ↓
    [GenerateModule] → result.generated_program (Program - NEW)
        ↓
    [EvaluateModule] → result.generated_program (ENRICHED IN PLACE with metrics)
        ↓
    [JudgeModule] → result.generated_program (ENRICHED IN PLACE with feedback)
        ↓
    Final: result.generated_program fully enriched

    Attributes:
        name: Module name (for logging and debugging)
        config: Configuration dictionary
        logger: Logger instance (injected by RolloutEngine)
    """

    def __init__(self, name: Optional[str] = None, **config):
        """
        Initialize module.

        Args:
            name: Optional module name (defaults to class name)
            **config: Configuration parameters specific to this module
        """
        self.name = name or self.__class__.__name__
        self.config = config
        self.logger: Optional["Logger"] = None  # Injected by RolloutEngine

    def __deepcopy__(self, memo):
        """Deep-copy this module WITHOUT its injected per-execution dependencies.

        Injected deps (env / llm_client / embedding_client / logger) are shared,
        per-execution objects — some hold un-copyable state (e.g. an env wraps a
        threading.Lock). More importantly, sharing them across concurrent
        rollouts is exactly what serializes evaluation: RolloutEngine injects the
        env by assigning ``module.env = env`` on the (strategy-shared) module
        instances, so concurrent workers clobber each other and converge onto a
        single env whose per-instance lock then serializes them.

        Copies produced here start with those deps reset to None; RolloutEngine
        re-injects the correct per-worker deps before executing each copy. This
        lets the ThreadPool backend give every task its own module instances
        (the isolation the Ray backend already gets for free via per-task
        serialization).
        """
        import copy as _copy

        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        _injected = {"env", "llm_client", "embedding_client", "logger"}
        for key, value in self.__dict__.items():
            if key in _injected:
                new.__dict__[key] = None
            else:
                new.__dict__[key] = _copy.deepcopy(value, memo)
        return new

    def __call__(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Entry point for module execution.

        Flow:
        1. Validate inputs (check prerequisites)
        2. Execute module logic (read from context, write to result)
        3. Post-validate output (check results)
        4. Return updated result

        **Key Design:**
        - **context**: Experiment-level data (READ-ONLY) - prompts, population, config
        - **result**: Rollout-level data (MUTABLE) - selected/generated/evaluated programs

        **Thread-Safety:**
        Each rollout has its own RolloutResult, so parallel rollouts are naturally
        isolated. Context is shared (read-only), which is safe.

        Args:
            context: Experiment context (read-only for modules)
            result: Rollout result (modules write to this)
            **kwargs: Additional runtime arguments

        Returns:
            Updated rollout result

        Raises:
            ModuleValidationError: If validation fails (not retried by RolloutEngine)
        """
        # Pre-execution validation
        # Wrap ValueError in ModuleValidationError so it's not retried
        try:
            self.validate_input(context, result)
        except ValueError as e:
            raise ModuleValidationError(str(e)) from e

        # Execute module logic
        # Errors here (including JSONDecodeError) WILL be retried
        result = self.execute(context, result, **kwargs)

        # Post-execution validation
        # Wrap ValueError in ModuleValidationError so it's not retried
        try:
            self.validate_output(context, result)
        except ValueError as e:
            raise ModuleValidationError(str(e)) from e

        return result

    @abstractmethod
    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """
        Execute module logic.

        This is the core method that subclasses must implement.
        Modules READ from context (experiment data) and WRITE to result (rollout data).

        Args:
            context: Experiment context (read-only)
            result: Rollout result (mutable)
            **kwargs: Additional runtime arguments

        Returns:
            Updated rollout result
        """
        pass

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """
        Validate that inputs have required data.

        Override this method to check prerequisites before execution.
        Raise ValueError with a clear message if validation fails.

        Args:
            context: Experiment context (for checking population, config, etc.)
            result: Rollout result (for checking previous module outputs)

        Raises:
            ValueError: If inputs are missing required data
        """
        pass

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """
        Validate that module produced valid output.

        Override this method to check postconditions after execution.
        Useful for catching bugs and ensuring data integrity.

        Args:
            context: Experiment context
            result: Rollout result (after execution)

        Raises:
            ValueError: If output is invalid

        Example:
            >>> class GenerateModule(Module):
            ...     def validate_output(self, context, result):
            ...         if not result.generated_program:
            ...             raise ValueError(
            ...                 f"{self.name}: Failed to generate program"
            ...             )
            ...         if not result.generated_program.code:
            ...             raise ValueError(
            ...                 f"{self.name}: Generated program has no code"
            ...             )
        """
        pass

    def log_info(self, message: str, **context_data):
        """Helper to log debug message with context (module-level logs are DEBUG)."""
        if self.logger:
            self.logger.debug(f"[{self.name}] {message}", **context_data)

    def log_error(self, message: str, **context_data):
        """Helper to log error message with context."""
        if self.logger:
            self.logger.error(f"[{self.name}] {message}", **context_data)

    def log_warning(self, message: str, **context_data):
        """Helper to log warning message with context."""
        if self.logger:
            self.logger.warning(f"[{self.name}] {message}", **context_data)

    def __repr__(self) -> str:
        """Concise representation for debugging."""
        return f"{self.__class__.__name__}(name={self.name})"


# ============================================================================
# Strategy: Dynamic Decision-Making Base Class
# ============================================================================

if TYPE_CHECKING:
    from famou.modules.population.base import PopulationModule


class Strategy(ABC):
    """
    Base class for all execution strategies.

    A Strategy decides which rollout to execute based on current context.
    This enables adaptive strategies that change behavior based on:
    - Current population state (diversity, fitness distribution)
    - Iteration phase (warmup, explore, exploit)
    - Custom logic (time of day, resource availability, etc.)

    Design Philosophy:
    ------------------
    - PopulationModule stays experiment-level (maintains boundary with current design)
    - forward() is called per-iteration (before rollout execution)
    - Strategies can switch between different rollouts adaptively
    - Context provides read-only snapshot for decision-making
    - Minimal changes to Evolver (single rollout execution)

    Required Implementation:
    ------------------------
    All Strategy subclasses must:

    1. Accept evaluate_fn in __init__:
       def __init__(self, evaluate_fn: Callable[[str], Dict[str, Any]]):
           self.evaluate_fn = evaluate_fn
           self.population_module = ...

    2. Set population_module attribute (class or instance):
       self.population_module = TopKPopulation(k=100)

    3. Implement forward() method:
       def forward(self, ctx: Context) -> Rollout:
           ...

    Execution Attributes (required):
    --------------------------------
    - population_module: How to manage population (TopK, Cluster, etc.)
    - evaluate_fn: Evaluator function for program execution

    Note: Metadata (name, description, tags, author) is managed by
    StrategyRegistry, not part of Strategy interface.
    """

    # Required attributes (will be set by subclasses)
    population_module: "PopulationModule"
    evaluate_fn: Callable[[str], Dict[str, Any]]

    # Optional: reference to the StateStore (set by Evolver before run)
    state_store: Optional[Any] = None

    @abstractmethod
    def forward(self, ctx: Context, rollout_history: List["RolloutResult"]) -> Rollout:
        """
        Forward pass to determine next rollout to execute based on current context and execution history.

        This method is called by Evolver before each rollout execution (including
        initial seed enrichment). Strategies inspect the context (population state,
        diversity, iteration phase) and execution history to return the appropriate
        rollout to execute.

        Args:
            ctx: Current context with population snapshot and metadata
                - ctx.iteration: Current iteration number (0 = initial enrichment)
                - ctx.population: Single-island population buckets (empty at iteration 0)
                - ctx.accessor: O(1) read-only population queries
                - ctx.diversity: Population diversity metric (0.0 at iteration 0)
                - ctx.state: Thread-safe cross-rollout state storage
                - ctx.experiment_config: Full experiment configuration

            rollout_history: Recent rollout history for THIS island only
                - Filtered to current island (in multi-island mode)
                - Typically last 50-100 rollouts for this island
                - Empty during initial enrichment (ctx.iteration == 0)
                - Includes both successful and failed rollouts
                - Each RolloutResult contains: status, generated_program, selection, stats

        Returns:
            Rollout: The rollout pipeline to execute for this iteration.
                Contains modules (Select, Generate, Evaluate, Judge) that define
                how to create and enrich one new program.

        Initial Enrichment (ctx.iteration == 0):
            When ctx.iteration == 0, forward() is called for initial seed program
            enrichment before evolution begins. At this point:
            - ctx.population is empty (no programs yet)
            - ctx.diversity is 0.0 (no population diversity)
            - Only ENRICHMENT modules (EvaluateModule, JudgeModule) from the returned
              rollout will be executed on seed programs
            - Select/Generate modules are IGNORED during initial enrichment

        Best Practices:
            - Context is READ-ONLY (do not mutate ctx fields)
            - Use ctx.state for persistent cross-rollout state (thread-safe)
            - Return different rollouts based on adaptive logic (diversity, iteration phase)
            - Keep forward() fast (it's called on the hot path before each rollout)
            - Store rollouts as instance attributes (avoid recreating them each call)

        See Also:
            - Evolver._enrich_initial_programs() for initial enrichment execution
            - RolloutEngine.execute_enrichment() for enrichment-only module execution
            - Context for available decision-making information
        """
        ...

    def forward_batch(
        self,
        ctx: Context,
        rollout_history: List["RolloutResult"],
        max_batch_size: int = 1,
    ) -> "WorkBatch":
        """Decide a whole batch of rollouts in one decision. Optional.

        The default implementation wraps ``forward()`` as a batch of one, so
        every existing Strategy keeps working unchanged and the Evolver has a
        single dispatch path.

        Override this when one decision should fan out into several rollouts
        that are executed concurrently and then committed together (e.g. "try
        these 4 mutations, then update state once"). The Evolver guarantees:

        - ``max_batch_size`` is how many rollouts it can actually dispatch
          right now (bounded by the worker pool and the remaining iteration
          budget). Returning more than that is an error, not a hint — the
          extra rollouts have nowhere to run.
        - Every dispatched rollout produces exactly one
          ``on_rollout_complete`` or ``on_rollout_failed`` callback, so a
          strategy that waits for a batch to finish can count them and will
          not hang on a failed member.

        Args:
            ctx: Current context (same object ``forward()`` would receive)
            rollout_history: Recent rollout history for this island
            max_batch_size: Upper bound on ``len(batch.rollouts)``, >= 1

        Returns:
            WorkBatch with 1..max_batch_size rollouts.
        """
        from famou.core.data import WorkBatch

        return WorkBatch.single(self.forward(ctx, rollout_history))

    def dump_state(self) -> Dict[str, Any]:
        """Dump strategy-level state for checkpoint persistence. Override in subclass."""
        return {}
    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy-level state from checkpoint. Override in subclass."""
        pass

    def on_rollout_failed(self, result: "RolloutResult") -> None:
        """Called when a rollout has FAILED (any module in the pipeline did not
        complete normally). The rollout is discarded and will be retried; the
        iteration is NOT consumed.

        Override in subclass to release any locks/resources acquired in forward().
        Paired with on_rollout_complete for commit."""
        pass

    def on_rollout_complete(self, result: "RolloutResult") -> None:
        """Called when a rollout finishes normally (whether the produced program is
        valid or buggy — as long as a Program is recorded and iteration is consumed).

        Override in subclass to commit transactional state (release locks, update
        plan counters, record statistics). Paired with on_rollout_failed for rollback."""
        pass

    def finalize_experiment(self, contexts: Dict[int, Context]) -> None:
        """Finalize strategy-managed runtime state before the last checkpoint."""
        del contexts


# ============================================================================
# Dependency Injection Protocol Mixins
# ============================================================================

@runtime_checkable
class RequiresLLM(Protocol):
    """
    Protocol mixin for modules that require LLM client.

    Modules that need to call LLMs should inherit from this protocol.
    The RolloutEngine will automatically inject the LLM client.

    Example:
        >>> class MutationGenerate(Module, RequiresLLM):
        ...     llm_client: LLMClient
        ...
        ...     def execute(self, context, **kwargs):
        ...         response = self.llm_client.generate(...)
        ...         # ...
    """

    llm_client: "LLMClient"

@runtime_checkable
class RequiresEnv(Protocol):
    """
    Protocol mixin for modules that require execution environment.

    Modules that need to execute code should inherit from this protocol.
    The RolloutEngine will automatically inject the execution environment.

    Example:
        >>> class LocalEvaluate(Module, RequiresEnv):
        ...     env: ExecutionEnvironment
        ...
        ...     def execute(self, context, **kwargs):
        ...         result = self.env.execute(program)
        ...         # ...
    """

    env: "ExecutionEnvironment"

@runtime_checkable
class RequiresEmbedding(Protocol):
    """
    Protocol mixin for modules that require embedding client.

    Modules that need to generate embeddings should inherit from this protocol.
    The RolloutEngine will automatically inject the embedding client.
    """

    embedding_client: "EmbeddingClient"


# ============================================================================
# Helper functions for dependency injection
# ============================================================================

def has_dependency(module: Module, protocol: type) -> bool:
    """
    Check if a module requires a specific dependency protocol.

    Args:
        module: Module instance
        protocol: Protocol class (RequiresLLM, RequiresEnv)

    Returns:
        True if module implements the protocol

    Example:
        >>> class MyModule(Module, RequiresLLM):
        ...     llm_client: LLMClient
        >>> module = MyModule()
        >>> has_dependency(module, RequiresLLM)
        True
    """
    return isinstance(module, protocol)



def inject_llm(module: Module, llm_client: "LLMClient") -> None:
    """
    Inject LLM client into module if it requires one.

    Args:
        module: Module instance
        llm_client: LLM client to inject
    """
    if has_dependency(module, RequiresLLM):
        module.llm_client = llm_client  # type: ignore


def inject_env(module: Module, env: "ExecutionEnvironment") -> None:
    """
    Inject execution environment into module if it requires one.

    Args:
        module: Module instance
        env: Execution environment to inject
    """
    if has_dependency(module, RequiresEnv):
        module.env = env  # type: ignore

def inject_embedding(module: Module, embedding_client: "EmbeddingClient") -> None:
    """
    Inject embedding client into module if it requires one.

    Args:
        module: Module instance
        embedding_client: Embedding client to inject
    """
    if has_dependency(module, RequiresEmbedding):
        module.embedding_client = embedding_client  # type: ignore


# Rebuild Pydantic models after all classes are defined
# Rollout references Module which is defined in this module
# WorkBatch references Rollout (imported from data)
Rollout.model_rebuild()
WorkBatch.model_rebuild()
