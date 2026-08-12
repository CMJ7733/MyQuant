"""
RolloutEngine for Famou 2.0.

Orchestrates module execution in a pipeline.
Handles dependency injection and error recovery.
"""

import time
from typing import Any, Optional, Tuple, Dict
from langfuse import Langfuse

from famou.core.data import Context, ModuleStat, Program, RolloutResult, StateUpdateCollector
from famou.core.state import StateStore
from famou.core.types import FatalRolloutError, Language, ModuleValidationError, RolloutStatus
from famou.core.protocol import Module, Rollout, inject_embedding, inject_env, inject_llm
from famou.infrastructure.embedding import EmbeddingClient
from famou.infrastructure.env import ExecutionEnvironment
from famou.infrastructure.llm import LLMClient
from famou.infrastructure.llm.base import get_llm_max_retries
from famou.infrastructure.logger import Logger
from famou.utils import generate_rollout_id

# Langfuse v3+ - no direct import needed, using get_client() from factory


class RolloutEngine:
    """
    Reusable execution engine for running Rollout pipelines.

    The RolloutEngine:
    1. Provides reusable infrastructure (LLM, logger, env)
    2. Accepts different Rollout objects at runtime
    3. Injects dependencies and caches injected rollouts for efficiency
    4. Executes modules sequentially in the pipeline
    5. Handles errors according to strategy

    **Key Improvement:**
    The engine is now decoupled from specific rollouts - you can reuse the same
    engine to execute different rollout strategies without recreating infrastructure.

    **Pipeline Pattern:**
    ```
    Context (experiment-level, read-only)
        ↓
    [SelectModule] → result.selected_programs (List[str] - IDs)
        ↓
    [GenerateModule] → result.generated_programs (List[Program] - NEW)
        ↓
    [EvaluateModule] → result.generated_programs (ENRICHED with metrics)
        ↓
    [JudgeModule] → result.generated_programs (ENRICHED with scores)
        ↓
    RolloutResult (finalized)
    ```

    Example:
        >>> # Create engine once with infrastructure
        >>> engine = RolloutEngine(
        ...     llm_client=openai_client,
        ...     env=local_env,
        ...     logger=logger,
        ... )
        >>>
        >>> # Run different rollouts with same engine (efficient!)
        >>> exploit_rollout = Rollout(modules=[TopKSelect(), MutationGenerate(), ...])
        >>> explore_rollout = Rollout(modules=[RandomSelect(), MutationGenerate(), ...])
        >>>
        >>> result1 = engine.execute_rollout(exploit_rollout, context, iteration=1)
        >>> result2 = engine.execute_rollout(explore_rollout, context, iteration=2)
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        env: Optional[ExecutionEnvironment] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        logger: Optional[Logger] = None,
        strategy_state: Optional[StateStore] = None,
        langfuse: Optional[Langfuse] = None,  
        retry_enabled: bool = True,
        max_retries: Optional[int] = None,
        retry_wait_min: float = 2.0,
        retry_wait_max: float = 10.0,
        retry_multiplier: float = 1.5,
    ):
        """
        Initialize RolloutEngine with infrastructure.

        Args:
            llm_client: LLM client for generation/judging modules
            env: Execution environment for evaluation modules
            embedding_client: Embedding client for feature extraction modules
            logger: Logger for output
            strategy_state: StateStore for stateful modules (created if not provided)
            langfuse: Optional Langfuse client for observability (trace rollouts/modules)
            retry_enabled: Enable retry logic for module execution (default: True)
            max_retries: Maximum retry attempts for any module
            retry_wait_min: Minimum wait between retries in seconds (default: 2.0)
            retry_wait_max: Maximum wait between retries in seconds (default: 10.0)
            retry_multiplier: Exponential backoff multiplier (default: 1.5)

        Failure Strategy:
            When a module fails after all retries, the rollout is marked as failed
            and the iteration continues with the next rollout. Failed rollouts do
            not update the population.

        Note:
            LLM client already has langfuse injected via InfraFactory.
            This langfuse parameter is for RolloutEngine to create trace/span hierarchy.
        """
        self.llm_client = llm_client
        self.env = env
        self.embedding_client = embedding_client
        self.logger = logger
        self.strategy_state = strategy_state if strategy_state is not None else StateStore()
        self.langfuse = langfuse
        self.retry_enabled = retry_enabled
        self.max_retries = (
            max_retries if max_retries is not None else get_llm_max_retries(llm_client, default=5)
        )
        self.retry_wait_min = retry_wait_min
        self.retry_wait_max = retry_wait_max
        self.retry_multiplier = retry_multiplier

    def execute_rollout(
        self,
        rollout: Rollout,
        context: Context,
        iteration: int,
        env: ExecutionEnvironment,
        attempt: int = 1,
        llm_client: Optional[LLMClient] = None,
    ) -> RolloutResult:
        """
        Execute complete evolutionary rollout.

        Pipeline: Select → Generate → Evaluate → Judge
        Used during evolution iterations.

        The rollout is validated and dependencies are injected (with caching).
        Then all modules are executed sequentially with retry logic.

        Failure Strategy (Fail Rollout, Continue Iteration):
            - Modules are retried on transient failures (not validation errors)
            - If a module fails after all retries, the rollout is marked as failed
            - Failed rollouts are returned with status="failed"
            - The iteration continues (failed rollouts don't update population)

        Args:
            rollout: Rollout pipeline to execute
            context: Experiment context (shared, read-only)
            iteration: Current iteration number
            env: Execution environment for this rollout
            llm_client: LLM client to use for this rollout. If provided, overrides
                self.llm_client. Enables worker-resident LLM clients to be passed
                per-call rather than baked into the engine at construction time.

        Returns:
            Finalized RolloutResult (status="success" or "failed")
        """
        # Inject dependencies directly into rollout modules
        # Note: inject operations are idempotent - each call overwrites previous values
        # This ensures the correct env is always used without needing to copy modules
        self._inject_dependencies(rollout, env, llm_client=llm_client)

        # Create fresh RolloutResult for this rollout
        # Include island_id in rollout_id for multi-island experiments (for uniqueness)
        island_id_for_rollout = None
        if context.experiment_config and context.experiment_config.num_islands > 1:
            island_id_for_rollout = context.island_id

        result = RolloutResult(
            rollout_id=generate_rollout_id(
                experiment_id=context.experiment_id,
                iteration=iteration,
                island_id=island_id_for_rollout,
                attempt=attempt,
            ),
            experiment_id=context.experiment_id,
            iteration=iteration,
            island_id=context.island_id,
            rollout_attempt=attempt,
            rollout_name=rollout.name,
            created_at=time.time(),
        )
        result.state_updates = StateUpdateCollector(island_id=context.island_id)

        if self.logger:
            self.logger.debug(
                f"Starting rollout {result.rollout_id} (strategy: {rollout.name})",
                iteration=iteration,
                experiment_id=context.experiment_id,
                rollout_name=rollout.name,
            )

        # Execute pipeline
        return self._execute_pipeline(rollout.modules, context, result, iteration)

    def _inject_dependencies(
        self,
        rollout: Rollout,
        env: ExecutionEnvironment,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """
        Inject infrastructure dependencies into rollout's modules.

        Uses protocol checking to inject only what each module needs:
        - RequiresLLM → llm_client (caller-supplied takes priority over self.llm_client)
        - RequiresEnv → env
        - RequiresEmbedding → embedding_client
        - All modules → logger

        Recursively walks into pipeline modules (Sequence, Conditional, Cycle)
        via the get_children() method to inject dependencies into nested modules.

        Note: StateStore is passed via context.state (no injection needed)

        Args:
            rollout: Rollout whose modules need dependency injection
            env: Execution environment
            llm_client: Override LLM client. When provided (e.g., worker-resident
                client passed per-call), it takes priority over self.llm_client.
        """
        effective_llm = llm_client or self.llm_client
        for module in rollout.modules:
            self._inject_into_module(module, effective_llm, env)

    def _inject_into_module(
        self, module: Module, effective_llm: Optional[LLMClient],
        env: ExecutionEnvironment
    ) -> None:
        """Inject dependencies into a single module, recursing into children."""
        # Inject logger when available
        if self.logger:
            module.logger = self.logger

        # Inject LLM client if module requires it
        if effective_llm:
            inject_llm(module, effective_llm)

        # Inject execution environment if module requires it
        inject_env(module, env)

        # Inject embedding client if module requires it
        if self.embedding_client:
            inject_embedding(module, self.embedding_client)

        # Recurse into pipeline module children (Sequence, Conditional, Cycle)
        if hasattr(module, 'get_children'):
            for child in module.get_children():
                self._inject_into_module(child, effective_llm, env)

    def _execute_module_with_retry(
        self,
        module: Module,
        context: Context,
        result: RolloutResult,
    ) -> Tuple[RolloutResult, int, bool, Optional[str], float, Optional[Exception]]:
        """
        Execute module with retry logic.

        Retries on all exceptions EXCEPT ValueError (validation errors).
        Uses exponential backoff between retry attempts.

        Args:
            module: Module to execute
            context: Experiment context
            result: Rollout result

        Returns:
            Tuple of (result, retry_count, success, error_msg, elapsed_time)
        """
        start_time = time.time()

        if not self.retry_enabled:
            # No retry, execute directly
            try:
                result = module(context, result)
                elapsed = time.time() - start_time
                return result, 0, True, None, elapsed, None
            except Exception as e:
                elapsed = time.time() - start_time
                return result, 0, False, str(e), elapsed, e

        # Import tenacity for retry logic
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_exponential,
            retry_if_not_exception_type,
            RetryError,
        )

        # Track retry count
        retry_count = 0

        def _track_retry(retry_state):
            nonlocal retry_count
            # attempt_number is the attempt that just failed (1-based)
            # This equals the number of retries made so far
            retry_count = retry_state.attempt_number
            self._log_retry_attempt(module, retry_state)

        # Create retry decorator (retry all except validation errors and fatal errors)
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(
                multiplier=self.retry_multiplier,
                min=self.retry_wait_min,
                max=self.retry_wait_max,
            ),
            retry=retry_if_not_exception_type((ModuleValidationError, FatalRolloutError)),
            before_sleep=_track_retry,
            reraise=True,
        )

        # Wrap module execution
        @retry_decorator
        def _execute():
            return module(context, result)

        try:
            result = _execute()
            elapsed = time.time() - start_time
            return result, retry_count, True, None, elapsed, None
        except RetryError as e:
            # All retries exhausted - retry_count already updated by _track_retry
            retry_count = e.last_attempt.attempt_number
            original_exception = e.last_attempt.exception()
            elapsed = time.time() - start_time
            error_msg = str(original_exception)
            return result, retry_count, False, error_msg, elapsed, original_exception
        except Exception as e:
            # Non-retryable error (e.g., ModuleValidationError)
            elapsed = time.time() - start_time
            return result, 0, False, str(e), elapsed, e

    def execute_enrichment(
        self,
        program,
        modules: list,
        context: Context,
        rollout_id: str,
        env: Optional[ExecutionEnvironment] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> RolloutResult:
        """
        Enrich existing program with enrichment modules.

        Pipeline: Evaluate → Judge (no Select/Generate)
        Used for initial programs and re-evaluation.

        This method is designed for enriching initial programs or re-evaluating
        existing programs. It wraps the program in a RolloutResult and executes
        enrichment modules (EvaluateModule, LLMJudge, etc.) with retry logic.

        Args:
            program: Program to enrich
            modules: List of enrichment modules to apply
            context: Experiment context
            rollout_id: Unique ID for this enrichment operation
            env: Optional ExecutionEnvironment instance. When called via
                 BackendTask/ThreadWorker, this is the worker's env instance.
                 Falls back to self.env if not provided.
            llm_client: LLM client to use for injection. If provided, overrides
                self.llm_client. Enables worker-resident LLM clients to be passed
                per-call rather than baked into the engine at construction time.

        Returns:
            RolloutResult with enriched program

        Example:
            >>> enrichment_modules = [EvaluateModule(), LLMJudge()]
            >>> result = engine.execute_enrichment(
            ...     program=initial_program,
            ...     modules=enrichment_modules,
            ...     context=context,
            ...     rollout_id="init_0",
            ... )
            >>> enriched_program = result.generated_program
        """
        # Use provided env, fall back to self.env
        effective_env = env or self.env
        effective_llm = llm_client or self.llm_client

        # Inject dependencies into modules
        for module in modules:
            # Always inject logger
            if self.logger:
                module.logger = self.logger

            # Inject LLM client if module requires it
            if effective_llm:
                inject_llm(module, effective_llm)

            # Inject execution environment if module requires it
            if effective_env:
                inject_env(module, effective_env)

            # Inject embedding client if module requires it
            if self.embedding_client:
                inject_embedding(module, self.embedding_client)

            # Note: StateStore is now passed via context.state (no injection needed)

        # Create RolloutResult with the program
        result = RolloutResult(
            rollout_id=rollout_id,
            iteration=0,
            island_id=context.island_id,
            rollout_name="enrichment",  # Special name for initial program enrichment
            status="success",
            generated_program=program,
            created_at=time.time(),
        )
        result.state_updates = StateUpdateCollector(island_id=context.island_id)

        if self.logger:
            self.logger.debug(
                f"Starting enrichment {rollout_id}",
                program_id=program.id,
                num_modules=len(modules),
            )

        # Execute pipeline
        return self._execute_pipeline(modules, context, result, iteration=0)

    def _execute_pipeline(
        self,
        modules: list,
        context: Context,
        result: RolloutResult,
        iteration: int,
    ) -> RolloutResult:
        """
        Core execution: run modules with retry logic using Langfuse v3+ observation API.

        Shared by both execute_rollout() and execute_enrichment().
        Executes modules sequentially, handles retries and errors,
        finalizes result.

        Args:
            modules: List of modules to execute
            context: Experiment context
            result: RolloutResult to update
            iteration: Current iteration number (for logging)

        Returns:
            Finalized RolloutResult
        """
        # Create Langfuse root span for this rollout using v3+ API
        if self.langfuse:
            trace_id = self.langfuse.create_trace_id(seed=result.rollout_id)  # rollout_id 必须每轮不同
            with self.langfuse.start_as_current_observation(
                as_type="span",
                name="famou.rollout",
                trace_context={"trace_id": trace_id},
                metadata={
                    "experiment_id": context.experiment_id,
                    "iteration": iteration,
                    "island_id": result.island_id,
                    "rollout_name": result.rollout_name,
                    "rollout_id": result.rollout_id,
                },
            ) as rollout_obs:
                # Get TRACE ID (not observation ID!)
                trace_id = rollout_obs.trace_id
                url = self.langfuse.get_trace_url()
                
                # Store trace information in result
                result.langfuse_trace_id = trace_id
                result.langfuse_trace_url = url

                # Set trace-level attributes (session_id, tags)
                # In Langfuse v3, these must be set on the trace, not the observation
                # session_id = experiment (all rollouts in experiment form one session)
                # tags = for filtering by iteration, island, rollout type
                try:
                    self.langfuse.update_current_trace(
                        session_id=context.experiment_id,
                        tags=["famou", result.rollout_name, context.experiment_id],
                    )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to set trace attributes: {e}")

                try:
                    # Execute module pipeline with retry and failure tracking
                    result = self._execute_modules_with_retries(
                        modules, context, result, iteration, rollout_obs
                    )
                    rollout_input = {
                        "experiment_id": context.experiment_id,
                        "island_id": result.island_id,
                        "iteration": iteration,
                    }

                    # Update rollout observation with final status
                    rollout_obs.update(
                        input=rollout_input,
                        output=self._format_rollout_output(result),
                        level="ERROR" if result.status != RolloutStatus.SUCCESS else "DEFAULT",
                        status_message=result.error_message if result.error_message else "success",
                    )
                except KeyboardInterrupt:
                    rollout_obs.update(input=rollout_input,
                                       output={"status": "cancelled"},
                                       level="ERROR",
                                       status_message="KeyboardInterrupt")
                    raise
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to update Langfuse observation: {e}")
        else:
            # No Langfuse, execute modules directly
            result = self._execute_modules_with_retries(
                modules, context, result, iteration, rollout_obs=None
            )

        return result

    def _execute_modules_with_retries(
        self,
        modules: list,
        context: Context,
        result: RolloutResult,
        iteration: int,
        rollout_obs: Optional[Any] = None,
    ) -> RolloutResult:
        """
        Execute modules with retry logic (inner loop, no Langfuse context management).

        Args:
            modules: List of modules to execute
            context: Experiment context
            result: RolloutResult to update
            iteration: Current iteration number
            rollout_obs: Parent rollout observation (for creating nested spans)

        Returns:
            Updated RolloutResult
        """
        # Execute module pipeline with retry and failure tracking
        rollout_failed = False
        failed_module = None
        failed_module_obj = None
        failure_error = None
        failure_exception: Optional[Exception] = None

        for module_index, module in enumerate(modules):
            start_time = time.time()
            retry_count = 0
            success = True
            error_msg = None
            module_obs_id = None


            # Create module span as child of rollout span
            if self.langfuse and rollout_obs:
                try:
                    with self.langfuse.start_as_current_observation(
                        as_type="span",
                        name=f"module.{module.name}",
                        metadata={
                            "module_name": module.name,
                            "module_type": type(module).__name__,
                            "module_index": module_index,
                        },
                    ) as module_obs:
                        module_obs_id = module_obs.id


                        # Execute module with retry
                        result, retry_count, success, error_msg, elapsed, failure_exception = self._execute_module_with_retry(
                            module, context, result
                        )

                        # Update module span with completion status
                        module_obs.update(
                            level="ERROR" if not success else "DEFAULT",
                            status_message=error_msg if error_msg else "success",
                            metadata={
                                "success": success,
                                "retry_count": retry_count,
                                "execution_time_ms": int(elapsed * 1000),
                            },
                        )
                except Exception as span_error:
                    if self.logger:
                        self.logger.warning(f"Failed to create module span: {span_error}")
                    # Execute without tracing
                    result, retry_count, success, error_msg, elapsed, failure_exception = self._execute_module_with_retry(
                        module, context, result
                    )
            else:
                # No Langfuse, execute directly
                result, retry_count, success, error_msg, elapsed, failure_exception = self._execute_module_with_retry(
                    module, context, result
                )

            # Track rollout failure
            if not success:
                rollout_failed = True
                failed_module = module.name
                failed_module_obj = module
                failure_error = error_msg

            # Record module stats regardless of success/failure
            result.module_stats.append(ModuleStat(
                module_name=module.name,
                execution_time=elapsed,
                success=success,
                error_message=error_msg,
                retry_count=retry_count,
                langfuse_observation_id=module_obs_id,
            ))

            # Stop processing if module failed
            if not success:
                break

        # Set rollout status and error information
        if rollout_failed:
            from famou.modules.generate.base import GenerateModule

            if (
                isinstance(failure_exception, FatalRolloutError)
                or (
                    isinstance(failure_exception, ModuleValidationError)
                    and isinstance(failed_module_obj, GenerateModule)
                )
            ):
                result.status = RolloutStatus.FATAL
            else:
                # 任意模块未能正常完成 -> rollout 整体作废,统一 FAILED
                result.status = RolloutStatus.FAILED
            result.failed_module = failed_module
            result.error_message = failure_error
        else:
            result.status = RolloutStatus.SUCCESS

        # Finalize result (compute stats)
        result.finalize()

        # Mark completion time
        result.completed_at = time.time()

        # Log completion
        if self.logger:
            if rollout_failed:
                self.logger.error(
                    f"Execution {result.rollout_id} FAILED",
                    iteration=iteration,
                    failed_module=failed_module,
                    error=failure_error,
                    status=str(result.status),
                )
            else:
                self.logger.debug(
                    f"Execution {result.rollout_id} completed successfully",
                    iteration=iteration,
                    score=(
                        f"{result.generated_program.combined_score:.4f}"
                        if result.generated_program and result.generated_program.combined_score is not None
                        else "N/A"
                    ),
                    validity=(
                        f"{result.generated_program.validity:.2f}"
                        if result.generated_program and result.generated_program.validity is not None
                        else "N/A"
                    ),
                    status="success",
                )

        return result

    def _log_retry_attempt(self, module: Module, retry_state):
        """Log retry attempt with details."""
        if not self.logger:
            return

        attempt = retry_state.attempt_number
        wait = retry_state.next_action.sleep if retry_state.next_action else 0
        exception = retry_state.outcome.exception()

        self.logger.warning(
            f"Module {module.name} failed (attempt {attempt}/{self.max_retries}), "
            f"retrying in {wait:.1f}s: {exception}",
            system_only=True,
            module=module.name,
            attempt=attempt,
            max_retries=self.max_retries,
            wait_time=f"{wait:.1f}s",
            error=str(exception),
            error_type=type(exception).__name__,
        )

    def _format_rollout_output(self, result: RolloutResult) -> Dict[str, Any]:
        """
        Format rollout result for Langfuse output.

        Args:
            result: Completed RolloutResult

        Returns:
            Dictionary with rollout summary
        """
        output = {
            "status": result.status,
            "execution_time": result.stats.get("execution_time", 0),
        }

        if result.generated_program:
            output["program_id"] = result.generated_program.id
            output["score"] = result.generated_program.combined_score
            output["validity"] = result.generated_program.validity

        if result.failed_module:
            output["failed_module"] = result.failed_module
            output["error_message"] = result.error_message

        return output

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"RolloutEngine("
            f"llm={self.llm_client is not None}, "
            f"env={self.env is not None}, "
            f"retry={self.retry_enabled}, "
            f"max_retries={self.max_retries})"
        )
