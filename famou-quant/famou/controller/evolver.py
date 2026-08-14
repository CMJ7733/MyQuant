"""
Evolver for Famou 2.0.

Main controller for experiment lifecycle.
Uses streaming evolution: maintains worker pool, updates population as rollouts complete.
"""

import threading
import time
import json
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from langfuse import Langfuse

from famou.config import ExperimentConfig
from famou.controller.engine import RolloutEngine
from famou.controller.evaluator_dependency import (
    install_packages_into_current_env,
    resolve_init_program_dependencies,
)
from famou.core.accessors import IslandAccessor, PopulationAccessor
from famou.core.data import Context, Experiment, Program, RolloutResult
from famou.core.protocol import Rollout
from famou.core.state import StateStore
from famou.core.types import RolloutStatus

if TYPE_CHECKING:
    from famou.core.protocol import Strategy
from famou.infrastructure.env import EnvFactory
from famou.infrastructure.llm import LLMClient
from famou.infrastructure.embedding import EmbeddingClient
from famou.infrastructure.logger import Logger
from famou.infrastructure.storage import DataService
from famou.infrastructure.backend import BackendTask
from famou.utils import generate_experiment_id, generate_island_initial_program_id
from famou.utils.math import (
    cosine_distance,
    cosine_similarity,
)

from famou.modules.population import PopulationModule

Context.model_rebuild()

@dataclass
class _IslandTracker:
    """
    Tracks iteration assignment across islands using global iteration counter.

    Uses a global iteration counter that increments across all islands in round-robin fashion.
    This ensures:
    - Each program gets a unique iteration number
    - Iteration=0 remains reserved for seed programs
    - Rollout IDs can be unique per island-iteration pair
    - Archive ordering reflects chronological program creation

    The tracker automatically computes remaining iterations based on start_iteration
    and max_iterations. This correctly handles resume scenarios where start_iteration > 1.

    Usage (new experiment):
        tracker = _IslandTracker(num_islands=1, start_iteration=1, max_iterations=10)
        # remaining_iterations = 10 - 1 + 1 = 10
        # -> dispatches iterations 1, 2, ..., 10

    Usage (resume from iteration 5):
        tracker = _IslandTracker(num_islands=1, start_iteration=6, max_iterations=10)
        # remaining_iterations = 10 - 6 + 1 = 5
        # -> dispatches iterations 6, 7, 8, 9, 10

    Usage (multi-island):
        tracker = _IslandTracker(num_islands=3, start_iteration=1, max_iterations=10)
        # Round-robin with global counter
        # -> (0, 1), (1, 2), (2, 3), (0, 4), (1, 5), (2, 6), (0, 7), (1, 8), (2, 9), (0, 10)
    """

    num_islands: int
    start_iteration: int
    max_iterations: int  # Target total iterations (from config)
    current_island_index: int = 0
    current_global_iteration: int = field(init=False)
    total_iterations_dispatched: int = 0
    remaining_iterations: int = field(init=False)  # Computed from start and max

    def __post_init__(self):
        # Global iteration counter starts from start_iteration
        self.current_global_iteration = self.start_iteration
        # Compute remaining iterations: max_iterations is the target total,
        # start_iteration - 1 is how many have been completed
        self.remaining_iterations = self.max_iterations - self.start_iteration + 1
        # Ensure non-negative (in case of misconfiguration)
        if self.remaining_iterations < 0:
            self.remaining_iterations = 0

    def get_next(self) -> Optional[Tuple[int, int]]:
        """
        Get the next (island_id, iteration) to execute using global iteration counter.

        Returns:
            (island_id, iteration) tuple, or None if all remaining iterations dispatched
        """
        # Check if we've dispatched all remaining iterations
        if self.total_iterations_dispatched >= self.remaining_iterations:
            return None

        # Get current island and global iteration
        island_id = self.current_island_index
        iteration = self.current_global_iteration

        # Advance counters
        self.current_global_iteration += 1
        self.total_iterations_dispatched += 1
        self.current_island_index = (self.current_island_index + 1) % self.num_islands

        return island_id, iteration

    def has_pending(self, island_id: int) -> bool:
        """Check if there are pending iterations."""
        return self.total_iterations_dispatched < self.remaining_iterations


class Evolver:
    """
    Main controller for evolutionary experiment.

    The Evolver:
    1. Manages the experiment lifecycle (iterations, population updates)
    2. Creates Context for each iteration (experiment-level, shared)
    3. Delegates rollout execution to RolloutEngine
    4. Updates Experiment state after each rollout
    5. Persists checkpoints (if storage provided)
    """

    # =============================================================================
    # Initialization
    # =============================================================================
    @staticmethod
    def _assign_initial_programs_to_islands(
        initial_programs: List[Program],
        num_islands: int,
    ) -> Dict[int, List[Program]]:
        """
        Assign initial programs to islands using round-robin distribution.

        Each assigned program is copied with an island-specific ID in the form
        ``{base_id}_{island_id}`` (e.g. "init_0", "init_1_2") so that island
        membership is directly encoded in the program ID.

        Args:
            initial_programs: List of initial programs to distribute
            num_islands: Number of islands to distribute across

        Returns:
            Dictionary mapping island_id to list of island-specific program copies
        """
        # Initialize empty lists for each island
        island_assignments: Dict[int, List[Program]] = {i: [] for i in range(num_islands)}

        # Distribute programs using round-robin
        for i, program in enumerate(initial_programs):
            island_id = i % num_islands
            island_assignments[island_id].append(program)
        for i in range(num_islands):
            if len(island_assignments[i]) == 0:
                island_assignments[i].append(initial_programs[i % len(initial_programs)])

        # Create island-specific copies with renamed IDs
        result: Dict[int, List[Program]] = {}
        for island_id, programs in island_assignments.items():
            renamed = []
            for program in programs:
                new_id = generate_island_initial_program_id(program.id, island_id)
                copy = program.model_copy(update={
                    "id": new_id,
                    "meta": {**program.meta, "island_id": island_id},
                })
                renamed.append(copy)
            result[island_id] = renamed
        return result

    def _initialize_islands_from_initial_programs(
        self,
        initial_programs: List[Program],
        population_module: PopulationModule,
        num_islands: int,
        config: ExperimentConfig,
    ) -> Tuple[Dict[int, Dict[str, List[Program]]], Dict[str, Program]]:
        """
        Initialize all islands from initial programs through PopulationModule.

        Works for both single island (num_islands=1) and multi-island cases.
        Initial programs are assigned to islands using round-robin distribution.
        Each program is copied with an island-specific ID (``{base_id}_{island_id}``)
        so island membership is encoded directly in the program ID.

        Args:
            initial_programs: Initial programs to populate the islands
            population_module: Module that manages population structure
            num_islands: Number of islands to create
            config: Experiment configuration

        Returns:
            Tuple of:
            - island_populations: Dict mapping island_id to population dict
            - new_archive: Updated archive using island-specific program IDs
        """
        # Assign initial programs to islands (round-robin), creating renamed copies
        island_program_assignments = self._assign_initial_programs_to_islands(
            initial_programs=initial_programs,
            num_islands=num_islands,
        )

        # Initialize each island by adding programs through PopulationModule
        island_populations = {}
        new_archive: Dict[str, Program] = {}

        for island_id in range(num_islands):
            assigned_programs = island_program_assignments[island_id]
            population = {}  # Start with empty population

            # Add each program through PopulationModule
            for program in assigned_programs:
                population = population_module.update_population(
                    current_population=population,
                    new_program=program,
                    experiment_config=config,
                    island_id=island_id,
                )
                # Accumulate renamed programs into new archive
                new_archive[program.id] = program

            island_populations[island_id] = population

        return island_populations, new_archive

    def __init__(
        self,
        config: ExperimentConfig,
        llm_client: LLMClient,
        logger: Logger,
        data_service: DataService,
        env: EnvFactory,
        population_module: PopulationModule,
        initial_programs: Optional[List[Program]] = None,
        experiment: Optional[Experiment] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        langfuse: Optional["Langfuse"] = None,  # type: ignore
        experiment_id: Optional[str] = None,
        monitor: Optional['MonitoringBackend'] = None,
        backend: Optional['Backend'] = None,
        evaluator_path: Optional[str] = None,
    ):
        """
        Initialize Evolver for new experiment or resume from checkpoint.

        Two usage modes:
        1. New experiment: Provide config + initial_programs
        2. Resume: Provide config + experiment

        Args:
            config: Experiment configuration (contains name, task_description, etc.)
            population_module: Population management strategy (REQUIRED)
            initial_programs: Initial programs for new experiments (unenriched)
            experiment: Loaded experiment for resume
            logger: Logger for output
            data_service: Storage service for checkpoints
            llm_client: LLM client for code generation
            env: Execution environment Factory
            embedding_client: Optional embedding client for semantic operations
            langfuse: Optional Langfuse client for observability (passed to RolloutEngine)
            experiment_id: Optional experiment ID (only for new experiments)
            monitor: Optional monitoring backend for real-time metrics (WandB, etc.)

        Raises:
            ValueError: If both or neither of initial_programs/experiment provided
            ValueError: If initial_programs is empty

        Examples:
            New experiment:
                >>> evolver = Evolver(
                ...     config=config,  # config.name used for experiment
                ...     population_module=TopKPopulation(),
                ...     initial_programs=[program1, program2],
                ...     ...
                ... )

            Resume from checkpoint:
                >>> experiment = storage.load_experiment("data/exp_123")
                >>> config = ExperimentConfig.from_yaml("data/exp_123/config.yaml")
                >>> evolver = Evolver(
                ...     config=config,
                ...     population_module=TopKPopulation(),
                ...     experiment=experiment,
                ...     ...
                ... )
        """
        # Validate usage mode
        if experiment is not None and initial_programs is not None:
            raise ValueError(
                "Cannot provide both 'experiment' and 'initial_programs'. "
                "Use 'initial_programs' for new experiments or 'experiment' for resume."
            )

        if experiment is None and initial_programs is None:
            raise ValueError(
                "Must provide either 'initial_programs' (new experiment) "
                "or 'experiment' (resume from checkpoint)"
            )

        # Store common attributes
        self.config = config
        self.logger = logger
        self.data_service = data_service
        self.population_module = population_module
        self.llm_client = llm_client
        self.env = env
        self.embedding_client = embedding_client
        self.monitor = monitor  # Monitoring backend (optional)
        self.langfuse = langfuse
        self.backend = backend
        self.evaluator_path = evaluator_path

        # Mode 1: New experiment from initial programs
        if initial_programs is not None:
            if not initial_programs:
                raise ValueError("initial_programs cannot be empty")

            # Generate experiment ID
            if experiment_id is None:
                experiment_id = generate_experiment_id(config.name)

            # Create new experiment with empty island populations
            empty_island_populations = {i: {} for i in range(config.num_islands)}

            # Initialize empty island_index for each island
            empty_island_index = {i: set() for i in range(config.num_islands)}

            # Create new experiment
            self.experiment = Experiment(
                id=experiment_id,
                name=config.name,
                island_populations=empty_island_populations,
                island_index=empty_island_index,
                num_islands=config.num_islands,
                archive={p.id: p for p in initial_programs},
                current_iteration=0,
            )

            if logger:
                logger.info(
                    f"[Init] Evolver initialized: experiment_id={self.experiment.id}, "
                    f"num_programs={len(initial_programs)}, num_islands={config.num_islands}"
                )

        # Mode 2: Resume from loaded experiment
        else:
            self.experiment = experiment

            if logger:
                logger.info(
                    f"[Init] Evolver resumed from checkpoint: experiment_id={self.experiment.id}, "
                    f"current_iteration={self.experiment.current_iteration}, "
                    f"archive_size={len(self.experiment.archive)}"
                )

        # Initialize or restore StateStore
        if self.experiment.strategy_state:
            # Resume: restore state from checkpoint
            self.strategy_state = StateStore(**self.experiment.strategy_state)
            if logger:
                logger.info(
                    "Restored strategy state from checkpoint",
                    global_keys=len(self.strategy_state.global_state),
                    island_count=len(self.strategy_state.island_state),
                )
        else:
            # New experiment: fresh state
            self.strategy_state = StateStore()

        # Create RolloutEngine with state registry
        self.engine = RolloutEngine(
            llm_client=llm_client,
            env=env,
            logger=logger,
            embedding_client=embedding_client,
            strategy_state=self.strategy_state,
            langfuse=langfuse,
        )

        # Enrichment 列表:Evolver 生命周期可挂载的功能增强,
        # 自身持有状态/缓存/统计,Evolver 只在合适时点遍历调用。
        from famou.infrastructure.enrichment import (
            DebugEnrichment,
            EmbeddingFeatureEnrichment,
            LeakDetectionEnrichment,
        )
        self.enrichments = [
            LeakDetectionEnrichment(
                config=getattr(self.config, "data_leak_detector", None) or {},
            ),
            DebugEnrichment(config=self.config.debug or {}),
            EmbeddingFeatureEnrichment(embedding_client=self.embedding_client),
        ]
        # Cooperative stop signal: set by stop_evolve() to interrupt the evolution loop
        self._stop_event = threading.Event()

        # Augment task_description with time constraint and diagnostic hint (new experiments only)
        if initial_programs is not None:
            self._augment_task_description(initial_programs)

    def stop_evolve(self) -> None:
        """Signal the evolver to stop at the next iteration boundary.

        Thread-safe. Sets the stop event so that _run_evolution_with_backend()
        breaks out of its loop on the next iteration check.
        Also initiates a non-blocking backend shutdown to release worker threads.
        """
        self._stop_event.set()
        if self.backend is not None:
            try:
                self.backend.shutdown(wait=False)
            except Exception:
                pass

    def _save_initial_programs(self) -> None:
        """
        Save initial programs to storage.

        This should be called before evolution starts to ensure all programs
        in the initial archive are saved to individual files.

        Adds experiment_id to program metadata if not already present.
        At this point all init programs already carry ``meta["island_id"]``
        (set during island assignment), so no special handling is needed.
        """
        for program in list(self.experiment.archive.values()):
            # Add experiment_id if not present
            if "experiment_id" not in program.meta:
                program.meta["experiment_id"] = self.experiment.id

            # Save program to file
            self.data_service.save_program(program)

    def _resolve_initial_program_dependencies(self) -> None:
        """Resolve and install packages required by initial programs.

        Uses static AST analysis + batched LLM extraction (max 5 files per call).
        Sets ``program.required_packages`` on each program and pre-installs any
        packages not already covered by the evaluator environment.
        Persists the resolution trace to ``init_dependency_resolution.json``.
        """
        programs = list(self.experiment.archive.values())
        if not programs:
            return

        # Use source_file from meta as path hint for LLM prompt (filename only)
        program_paths = [
            str(p.meta.get("source_file", p.id)) for p in programs
        ]

        if self.logger:
            self.logger.info(
                "[Init] Resolving initial program dependencies",
                num_programs=len(programs),
            )

        trace = resolve_init_program_dependencies(
            programs=programs,
            program_paths=program_paths,
            llm_client=self.llm_client,
        )

        if self.logger:
            self.logger.info(
                "[Init] Initial program dependencies resolved",
                final_packages=trace["final_packages"],
                batch_count=trace["batch_count"],
                llm_success=trace["llm_success"],
            )

        # Persist trace alongside evaluator_dependency_resolution.json
        if self.data_service:
            self.data_service.save_json_artifact(
                self.experiment.id,
                "init_dependency_resolution.json",
                trace,
            )

        # Install packages not already covered by the evaluator environment
        if trace["final_packages"]:
            install_packages_into_current_env(trace["final_packages"])

    def _validate_initial_program_feasibility(self) -> None:
        """
        Validate that the initial program set passes the feasibility gate.

        When `experiment.feasibility_check` is enabled, at least one enriched
        initial program must have `validity == 1.0`. Otherwise evolution is aborted.
        """
        if not self.config.feasibility_check:
            return

        programs = list(self.experiment.archive.values())
        if not programs:
            raise ValueError(
                "[Init] feasibility_check is enabled, but no initial programs were found"
            )

        passed_programs = []
        failed_programs = []
        for program in programs:
            validity = program.validity
            is_feasible = isinstance(validity, (int, float)) and abs(float(validity) - 1.0) <= 1e-9
            if is_feasible:
                passed_programs.append(program)
                continue

            label = program.meta.get("source_file") or program.id
            failed_programs.append(
                {
                    "label": label,
                    "program_id": program.id,
                    "validity": validity,
                    "combined_score": program.combined_score,
                    "error_info": program.error_info,
                }
            )

        if passed_programs:
            if self.logger:
                self.logger.info(
                    "[Init] Feasibility check passed",
                    total_initial_programs=len(programs),
                    feasible_initial_programs=len(passed_programs),
                    infeasible_initial_programs=len(failed_programs),
                )
            return

        if self.logger:
            for item in failed_programs:
                self.logger.error(
                    "[Init] Feasibility check failed",
                    source_file=item["label"],
                    program_id=item["program_id"],
                    validity=item["validity"],
                    combined_score=item["combined_score"],
                    error_info=item["error_info"],
                )

        failure_summary = "; ".join(
            (
                f"{item['label']} (program_id={item['program_id']}, "
                f"validity={item['validity']}, combined_score={item['combined_score']}, "
                f"error_info={item['error_info'] or 'None'})"
            )
            for item in failed_programs
        )
        raise ValueError(
            "[Init] feasibility_check failed: initial program(s) must achieve "
            "at least one validity=1.0 before evolution starts. "
            f"Failed programs: {failure_summary}"
        )

    def _extract_enrichment_modules(self, rollout: Rollout) -> List:
        """
        Extract enrichment modules from rollout.

        Enrichment modules are those that add data to programs:
        - EvaluateModule: Adds scores/metrics
        - LLMJudge: Adds LLM analysis
        - (Future: VerifyModule, ProfileModule, etc.)

        Selection/Generation modules are skipped (they create new programs).

        Args:
            rollout: Rollout pipeline

        Returns:
            List of enrichment modules
        """
        from famou.modules.evaluate import EvaluateModule
        from famou.modules.judge import JudgeModule

        enrichment_modules = []

        for module in rollout.modules:
            if isinstance(module, (EvaluateModule, JudgeModule)):
                enrichment_modules.append(module)

        return enrichment_modules

    def _enrich_initial_programs(self, rollout: Rollout) -> None:
        """
        Enrich initial programs using enrichment modules from rollout.

        Extracts enrichment modules (EvaluateModule, LLMJudge, etc.) from
        the rollout and applies them to initial programs in archive using
        backend for parallel execution.

        This ensures initial programs go through the same enrichment pipeline
        as generated programs, providing consistent treatment.

        Args:
            rollout: Rollout pipeline containing enrichment modules
        """
        from famou.infrastructure.backend import BackendTask

        enrichment_modules = self._extract_enrichment_modules(rollout)
        programs = list(self.experiment.archive.values())

        if self.logger:
            self.logger.info(
                f"[Init] Enriching {len(programs)} initial programs "
                f"with {len(enrichment_modules)} modules"
            )

        # Build enrichment tasks
        tasks = []
        for i, program in enumerate(programs):
            context = Context(
                experiment_id=self.experiment.id,
                task_description=self.config.task_description,
                population={},  # Empty for initial enrichment
                island_id=0,
                config={},
                experiment_config=self.config,
                state=self.strategy_state.create_accessor(0),
            )

            task = BackendTask.create_enrichment(
                experiment_id=self.experiment.id,
                context=context,
                engine=self.engine,
                program=program,
                enrichment_modules=enrichment_modules,
                enrichment_rollout_id=f"{self.experiment.id}_init_{i}",
                island_id=0,
            )
            tasks.append(task)

        # Submit to backend for parallel execution
        batch_handle = self.backend.submit_batch(tasks)

        completed_count = 0
        total = len(tasks)

        while batch_handle.has_pending():
            completed = batch_handle.wait_first_completed(timeout=300)

            if completed.get("status") == "timeout":
                if self.logger:
                    self.logger.warning("[Init] Enrichment task timed out, continuing...")
                continue

            completed_task_id = completed.get("completed_task")
            error = completed.get("error")
            result = completed.get("result")

            completed_count += 1

            if error:
                if self.logger:
                    self.logger.error(
                        f"[Init] Enrichment task {completed_task_id} failed: {error}"
                    )
            elif result and result.generated_program:
                # Update archive with enriched program
                self.experiment.archive[result.generated_program.id] = result.generated_program
                # Apply any state updates collected during enrichment
                if result.state_updates is not None:
                    self.strategy_state.apply_updates(result.state_updates.updates)
                # Update best program
                if result.generated_program.combined_score:
                    self.experiment.update_best_program(result.generated_program)

                if self.logger:
                    score = result.generated_program.combined_score or 0.0
                    validity = result.generated_program.validity or 0.0
                    status = result.status
                    status_symbol = "✓" if status == "success" else "✗"
                    self.logger.info(
                        f"[Init] Enriched {completed_count}/{total}: "
                        f"{result.generated_program.id} {status_symbol} "
                        f"(score={score:.4f}, validity={validity:.2f})"
                    )

            batch_handle.remove_completed(completed_task_id)

        # Update island populations with enriched programs; also get the renamed archive
        island_populations, new_archive = self._initialize_islands_from_initial_programs(
            initial_programs=list(self.experiment.archive.values()),
            population_module=self.population_module,
            num_islands=self.config.num_islands,
            config=self.config,
        )
        self.experiment.island_populations = island_populations
        # Replace archive with island-specific IDs (e.g. "init_0", "init_1_2")
        self.experiment.archive = new_archive
        # Refresh best_program tracking: old IDs (e.g. "init") are no longer in
        # archive after rename, so best_program_id would go stale without this.
        self.experiment.best_program_id = None
        self.experiment.best_program_score = None
        for program in new_archive.values():
            self.experiment.update_best_program(program)

        # Sync island_index from island_populations (initial setup)
        self.experiment.sync_island_index()

        if self.logger:
            self.logger.info("[Init] Initial programs enrichment complete")

    # =============================================================================
    # Main Control Flow
    # =============================================================================
    def run(self, strategy: "Strategy", run_mode: Optional[str] = None) -> Experiment:
        """
        Run the evolutionary experiment with streaming execution.

        Uses the configured experiment, config, logger, and storage.
        Takes a Strategy that decides which rollout to execute per iteration.

        Works for both new and resumed experiments:
        - New (iteration=0): Enriches initial programs first
        - Resume (iteration>0): Continues evolution directly

        Args:
            strategy: Strategy object that decides which rollout to execute per iteration.
                     Must implement Strategy ABC with forward() method.
            run_mode: Optional mode to specify what to run:
                     - "init": Only run initial program enrichment
                     - "evolution": Only run the evolution loop (skip enrichment)
                     - None or "full": Run full pipeline (init then evolution)

        Returns:
            Updated Experiment with final population
        """
        # Store strategy for dynamic rollout selection
        self.current_strategy = strategy
        self.current_strategy.state_store = self.strategy_state
        if hasattr(self.current_strategy, "llm_client"):
            self.current_strategy.llm_client = self.llm_client
        if hasattr(self.current_strategy, "embedding_client"):
            self.current_strategy.embedding_client = self.embedding_client
        if hasattr(self.current_strategy, "logger"):
            self.current_strategy.logger = self.logger
        if self.experiment.strategy_checkpoint:
            self.current_strategy.load_state(self.experiment.strategy_checkpoint)

        # For initial enrichment, get a rollout from strategy
        # Create minimal context for strategy decision
        # Note: iteration=0 signals initial enrichment phase to strategy
        # Strategies can check `if ctx.iteration == 0:` for custom initial behavior
        enrichment_context = Context(
            experiment_id=self.experiment.id,
            task_description=self.config.task_description,
            population={},
            island_id=0,
            experiment_config=self.config,
            diversity=0.0,
            iteration=0,  # Important: signals initial enrichment
        )
        # No rollout history during initial enrichment
        enrichment_rollout = strategy.forward(enrichment_context, rollout_history=[])
        # 初始 enrichment 同样走 enrichment 链:各 enrichment 自身判定是否适用
        # (leak/debug 在无 Generate/Evaluate 模块时自动跳过,embedding 在有 Judge 时注入)
        for enr in self.enrichments:
            enr.apply_to_rollout(enrichment_rollout)

        # Determine what to run based on run_mode
        # Default behavior: run full pipeline (init + evolution)
        valid_modes = {None, "full", "init", "evolution"}
        if run_mode not in valid_modes:
            if self.logger:
                self.logger.warning(f"[Run] Invalid run_mode '{run_mode}', treating as 'full'")
            run_mode = "full"

        should_enrich = run_mode is None or run_mode == "full" or run_mode == "init"
        should_evolve = run_mode is None or run_mode == "full" or run_mode == "evolution"

        # Start backend workers once for the entire run (enrichment + evolution)
        max_workers = self.config.max_workers
        self.backend.start_workers(max_workers, self.env, llm_client=self.llm_client, embedding_client=self.embedding_client)
        try:
            # Run initial enrichment only for new experiments and if requested
            if should_enrich and self.experiment.current_iteration == 0:
                if self.logger:
                    self.logger.info("[Init] New experiment detected, enriching initial programs")
                self._resolve_initial_program_dependencies()
                self._enrich_initial_programs(enrichment_rollout)

                # Save enriched programs
                if self.data_service:
                    self._save_initial_programs()

                self._validate_initial_program_feasibility()

                # Record initial metrics (iteration 0) after enrichment
                if self.monitor:
                    if self.logger:
                        self.logger.info("Recording initial metrics (iteration 0)")
                    self._record_metrics(iteration=0)

            # Run evolution if requested
            if should_evolve:
                return self._run_evolution_with_backend()

            # If only enrichment was requested, return current experiment state
            return self.experiment
        finally:
            self.backend.shutdown(wait=True)

    def _run_evolution_with_backend(self) -> Experiment:
        """
        Unified streaming evolution for both single and multi-island modes.

        This method replaces _run_single_island() and _run_multi_island(),
        handling both cases through the _IslandTracker abstraction.

        Returns:
            Updated Experiment with final population
        """
        max_iterations = self.config.max_iterations
        # Start from current_iteration + 1 (iteration 0 is reserved for seeds)
        # For new experiments: current_iteration=0 → start_iteration=1
        # For resume at iter 5: current_iteration=5 → start_iteration=6
        start_iteration = self.experiment.current_iteration + 1
        max_workers = self.config.max_workers
        num_islands = self.config.num_islands

        # Initialize tracker with global iteration counter
        tracker = _IslandTracker(
            num_islands=num_islands,
            start_iteration=start_iteration,
            max_iterations=max_iterations
        )

        if self.logger:
            self._log_evolution_start()

        retry_queue: deque[Tuple[int, int, int]] = deque()
        # Rollouts already planned by a batch decision but not yet submitted.
        # Draining these before asking the strategy again is what keeps one
        # decision -> N rollouts coherent.
        planned_tasks: deque[BackendTask] = deque()
        terminal_error: Optional[Exception] = None

        def _next_task() -> Optional[BackendTask]:
            """Next task to submit, or None when there is no work left.

            Order: planned batch members -> retries -> a fresh decision.
            Draining the current batch ahead of retries keeps one decision's
            rollouts together; it only ever delays a retry by at most
            max_batch_size slots, and planned_tasks drains monotonically so a
            retry can never be starved.
            """
            if planned_tasks:
                return planned_tasks.popleft()
            if retry_queue:
                island_id, iteration, attempt = retry_queue.popleft()
                # A retry is its own decision (the failed rollout produced no
                # evidence), so it goes through the single-rollout path.
                return self._get_rollout_task(island_id, iteration, attempt=attempt)
            next_slot = tracker.get_next()
            if next_slot is None:
                return None
            island_id, iteration = next_slot
            planned_tasks.extend(self._plan_rollout_tasks(island_id, iteration, tracker))
            return planned_tasks.popleft() if planned_tasks else None

        try:
            # 1. 创建初始批次
            tasks = []
            initial_batch_size = min(max_workers, tracker.remaining_iterations)

            for _ in range(initial_batch_size):
                task = _next_task()
                if task is None:
                    break
                tasks.append(task)
            
            batch_handle = self.backend.submit_batch(tasks)
            
            # 2. 主循环：处理完成的任务并提交新任务
            total_completed = 0
            completed_per_island = {i: 0 for i in range(num_islands)}
            
            while batch_handle.has_pending():
                # 检查外部停止信号（由 stop_evolve() 设置）
                if self._stop_event.is_set():
                    if self.logger:
                        self.logger.info("[Backend] Stop signal received, breaking evolution loop")
                    break

                # 等待第一个完成的任务，使用短超时以便定期检查停止信号
                completed_info = batch_handle.wait_first_completed(timeout=5)
                
                if completed_info.get("status") == "timeout":
                    continue
                
                completed_task_id = completed_info["completed_task"]
                
                # 从批次中移除已完成的任务
                batch_handle.remove_completed(completed_task_id)
                
                if "error" in completed_info:
                    error = completed_info["error"]
                    try:
                        island_id, iteration, attempt = BackendTask.get_task_metadata_from_task_id(
                            completed_task_id
                        )
                        terminal_error = RuntimeError(
                            f"Backend task failed at iteration {iteration} "
                            f"(attempt {attempt}, island {island_id}): {error}"
                        )
                    except Exception:
                        terminal_error = RuntimeError(
                            f"Backend task {completed_task_id} failed: {error}"
                        )
                    if self.logger:
                        self.logger.error(f"[Backend] Task {completed_task_id} failed: {error}")
                    backend_error: Exception = terminal_error
                    raise backend_error
                
                # 3. 处理迁移、检查点等逻辑（与原逻辑相同）
                result = completed_info["result"]
                # 提取任务信息
                island_id, iteration, attempt = BackendTask.get_task_metadata_from_task_id(completed_task_id)

                if result.status == RolloutStatus.FATAL:
                    self._update_experiment(result, island_id)
                    terminal_error = RuntimeError(
                        f"Fatal rollout error at iteration {iteration} "
                        f"(attempt {attempt}, island {island_id}): {result.error_message}"
                    )
                    fatal_error: Exception = terminal_error
                    raise fatal_error
                
                # 更新实验状态
                self._update_experiment(result, island_id)

                counts_iteration = result.status == RolloutStatus.SUCCESS
                if counts_iteration:
                    # 记录指标
                    if self.monitor:
                        self._record_metrics(iteration)
                    completed_per_island[island_id] += 1
                    total_completed += 1
                else:
                    retry_queue.appendleft((island_id, iteration, attempt + 1))
                    if self.logger:
                        self.logger.warning(
                            f"[Failed] island {island_id} iteration {iteration} attempt {attempt} "
                            f"failed (module={result.failed_module}); retrying same formal iteration"
                        )
                    # Strategy 事务回退钩子:让 strategy 释放 forward() 阶段占用的资源
                    if self.current_strategy is not None:
                        self.current_strategy.on_rollout_failed(result)
                
                # 执行迁移（如果需要）
                if counts_iteration and num_islands > 1:
                    migration_interval = self.config.migration_interval
                    migration_size = self.config.migration_size
                    reset_interval = self.config.reset_interval
                    reset_size = self.config.reset_size
                    
                    if reset_interval and total_completed % reset_interval == 0:
                        self._perform_reset(reset_size)
                    elif migration_interval and total_completed % migration_interval == 0:
                        self._perform_migration(migration_size)
                
                # 保存检查点
                if counts_iteration and self.data_service and self.config.checkpoint_interval > 0:
                    if total_completed % self.config.checkpoint_interval == 0:
                        self._save_checkpoint()
                        
                        # 刷新 Langfuse
                        if self.langfuse:
                            try:
                                self.langfuse.flush()
                            except Exception as e:
                                if self.logger:
                                    self.logger.warning(f"Failed to flush Langfuse: {e}")
                
                # 4. 提交新任务（如果有剩余）
                new_task = _next_task()
                if new_task is not None:
                    new_handle = self.backend.submit(new_task)

                    # 添加到批次
                    batch_handle.handles.append(new_handle)
                    batch_handle.task_ids.append(new_task.task_id)
                    batch_handle._handle_dict[new_task.task_id] = new_handle
        except Exception as e:
            terminal_error = terminal_error or e
            if self.logger:
                self.logger.error(f"[Backend] Evolution failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # === 以下为 evolution 结束后必须执行的清理逻辑 ===
            # 放在 finally 中确保 stop/pause/cancel 等场景也能执行

            try:
                self._finalize_strategy_before_shutdown()
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[Finalize] Strategy finalization skipped: {e}")

            # Log completion
            if self.logger:
                if terminal_error is None:
                    self._log_evolution_complete()
                else:
                    self._log_evolution_aborted(terminal_error)

            # Final checkpoint
            if self.data_service:
                self._save_checkpoint()

            # Finish monitoring
            if self.monitor:
                if self.logger:
                    self.logger.info("Finishing monitoring...")
                self.monitor.finish()
                if self.logger:
                    # Log WandB URL if available
                    if hasattr(self.monitor, 'run_url'):
                        self.logger.info(f"Monitor finished", url=self.monitor.run_url)
                    else:
                        self.logger.info("Monitor finished")

            # Final flush of Langfuse observations
            if self.langfuse:
                try:
                    if self.logger:
                        self.logger.debug("Flushing Langfuse observations...")
                    self.langfuse.flush()
                    if self.logger:
                        self.logger.debug("Langfuse flush completed")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Failed to flush Langfuse: {e}")

            # Generate output JSON files (evolve.json & recommend_solution.json)
            self._generate_output_json_files()

        if terminal_error is not None:
            error_to_raise: Exception = terminal_error
            raise error_to_raise

        return self.experiment

    # =========================================================================
    # Debug loop auto-injection
    # =========================================================================

    def _augment_task_description(self, initial_programs: List[Program]) -> None:
        """Augment task_description once with runtime constraint and diagnostic requirement.

        Makes a single LLM call to extract the per-test-case time constraint from
        evaluator code and the initial program, then appends the result plus a
        stderr diagnostic requirement to self.config.task_description.

        Only called for new experiments (resume skips: task_description was already
        augmented in the original run). Silently skips on any failure so the main
        flow is never interrupted.
        """
        if not self.evaluator_path or not self.llm_client:
            return

        runtime_cfg = getattr(self.config, "runtime_context", None) or {}
        if not (isinstance(runtime_cfg, dict) and runtime_cfg.get("enabled", False)):
            return

        try:
            evaluator_code = Path(self.evaluator_path).read_text(encoding="utf-8")
        except Exception:
            return

        init_code = initial_programs[0].code if initial_programs else ""
        init_section = f"\n\n<InitCode>\n{init_code}\n</InitCode>" if init_code else ""

        prompt = (
            f"<TaskDescription>\n{self.config.task_description}\n</TaskDescription>\n\n"
            f"<EvaluatorCode>\n{evaluator_code}\n</EvaluatorCode>"
            f"{init_section}\n\n"
            "Based on the above, output exactly one sentence describing the per-test-case "
            "time constraint the solution must satisfy. Be specific (include the exact value "
            "if visible). Output nothing else."
        )

        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system="You are a concise technical analyst.",
            )
            hint = response.text.strip()
            if not hint:
                return
        except Exception:
            return

        self.config.task_description += (
            f"\n\n## Runtime Constraint\n"
            f"{hint}\n\n"
            f"## Diagnostic Output (Required)\n"
            f"Print 2–3 key metrics to stderr (never stdout) at critical stages using "
            f'`print("[DIAG] ...", file=sys.stderr)`. '
            f"Focus on: iterations completed, best objective, time elapsed. "
            f"Example: [DIAG] iter=500 best=0.82 t=1.4s"
        )

        if self.data_service is not None:
            try:
                exp_dir = Path(self.data_service.get_experiment_dir(self.experiment.id))
                exp_dir.mkdir(parents=True, exist_ok=True)
                (exp_dir / "task_description.txt").write_text(
                    self.config.task_description, encoding="utf-8"
                )
            except Exception:
                pass

        if self.logger:
            self.logger.info(
                "Task description augmented with runtime constraint hint",
                hint=hint,
            )

    def _get_rollout_task(self, island_id: int, iteration: int, attempt: int = 1) -> BackendTask:
        """
        Execute a single rollout for a specific island.

        Unified method that handles both single-island (island_id=0) and
        multi-island execution.

        Args:
            island_id: Island ID (0 for single-island mode)
            iteration: Iteration number

        Returns:
            RolloutResult with evaluated programs

        Thread Safety:
            - Reads island population (thread-safe, no mutations)
            - Does not write to shared state
            - All rollout state is isolated in Context and RolloutResult
        """
        # Log iteration start
        if self.logger:
            if self.config.num_islands > 1:
                self.logger.info(
                    f"[Iter Start] island {island_id} | "
                    f"Iteration {iteration}/{self.config.max_iterations}")
            else:
                self.logger.info(f"[Iter Start] Iteration {iteration}/{self.config.max_iterations}")

        # Create context with appropriate population snapshot
        context = self._create_context(iteration, island_id)

        # Get island-filtered rollout history for strategy decision-making
        rollout_history = self._get_island_rollout_history(island_id)

        # Get rollout from strategy for this iteration
        rollout = self.current_strategy.forward(context, rollout_history)
        rollout = self._prepare_rollout_for_execution(rollout)

        backend_task = BackendTask.create(
            self.experiment.id,
            context,
            rollout,
            self.engine,
            island_id,
            iteration,
            attempt=attempt,
        )

        return backend_task

    def _plan_rollout_tasks(
        self,
        island_id: int,
        iteration: int,
        tracker: "_IslandTracker",
    ) -> List[BackendTask]:
        """Ask the strategy for one decision and turn it into dispatchable tasks.

        A strategy that overrides ``forward_batch()`` may answer with several
        rollouts for a single decision; they are dispatched concurrently and
        the strategy is told about each completion so it can commit them as a
        unit. The default implementation returns a batch of one, which makes
        this path behave exactly like the old single-rollout dispatch.

        Batch size is bounded by what can actually run: the worker pool, and
        the iteration slots still left in this experiment. Slots beyond the
        first are consumed here, so the caller must not also consume one.

        Multi-island note: ``_IslandTracker`` hands out slots round-robin
        across islands, so consecutive slots belong to different islands
        while a batch belongs to exactly one island's context. Batching is
        therefore restricted to single-island runs; multi-island runs keep
        the one-rollout-per-decision behaviour.
        """
        context = self._create_context(iteration, island_id)
        rollout_history = self._get_island_rollout_history(island_id)

        max_batch_size = 1
        if self.config.num_islands == 1:
            remaining_slots = tracker.remaining_iterations - tracker.total_iterations_dispatched
            # +1: the slot for `iteration` was already taken by the caller
            max_batch_size = max(1, min(self.config.max_workers, remaining_slots + 1))

        batch = self.current_strategy.forward_batch(
            context, rollout_history, max_batch_size=max_batch_size
        )
        rollouts = list(batch.rollouts)
        if len(rollouts) > max_batch_size:
            raise RuntimeError(
                f"Strategy returned {len(rollouts)} rollouts but only "
                f"{max_batch_size} could be dispatched. forward_batch() must "
                "honour max_batch_size: the surplus rollouts have no iteration "
                "slot to run in, and a strategy waiting for them would hang."
            )

        tasks: List[BackendTask] = []
        for index, rollout in enumerate(rollouts):
            if index == 0:
                slot_island, slot_iteration = island_id, iteration
            else:
                slot = tracker.get_next()
                if slot is None:  # defensive: max_batch_size should prevent this
                    break
                slot_island, slot_iteration = slot
            tasks.append(
                BackendTask.create(
                    self.experiment.id,
                    context,
                    self._prepare_rollout_for_execution(rollout),
                    self.engine,
                    slot_island,
                    slot_iteration,
                    attempt=1,
                )
            )

        if self.logger and len(tasks) > 1:
            self.logger.info(
                f"[Batch] island {island_id}: one decision -> {len(tasks)} "
                f"concurrent rollouts (barrier={batch.barrier})"
            )
        return tasks

    def _execute_rollout(self, island_id: int, iteration: int) -> RolloutResult:
        """
        Execute a single rollout for a specific island.

        Unified method that handles both single-island (island_id=0) and
        multi-island execution.

        Args:
            island_id: Island ID (0 for single-island mode)
            iteration: Iteration number

        Returns:
            RolloutResult with evaluated programs

        Thread Safety:
            - Reads island population (thread-safe, no mutations)
            - Does not write to shared state
            - All rollout state is isolated in Context and RolloutResult
        """
        # Log iteration start
        if self.logger:
            if self.config.num_islands > 1:
                self.logger.info(
                    f"[Iter Start] island {island_id} | "
                    f"Iteration {iteration}/{self.config.max_iterations}")
            else:
                self.logger.info(f"[Iter Start] Iteration {iteration}/{self.config.max_iterations}")

        # Create context with appropriate population snapshot
        context = self._create_context(iteration, island_id)

        # Get island-filtered rollout history for strategy decision-making
        rollout_history = self._get_island_rollout_history(island_id)

        # Get rollout from strategy for this iteration
        rollout = self.current_strategy.forward(context, rollout_history)
        rollout = self._prepare_rollout_for_execution(rollout)

        # Execute rollout
        result = self.engine.execute_rollout(rollout, context, iteration)

        return result

    def _prepare_rollout_for_execution(self, rollout: Rollout) -> Rollout:
        """Apply Evolver-level rollout rewrites before any execution backend sees it.

        Enrichment 链按 self.enrichments 顺序遍历,每个 enrichment 自行决定是否注入。
        当前顺序:LeakDetection -> Debug -> EmbeddingFeature
        """
        for enr in self.enrichments:
            enr.apply_to_rollout(rollout)
        return rollout

    def _create_context(self, iteration: int, island_id: int = 0) -> Context:
        """
        Create Context for a rollout.

        Context is a snapshot of the experiment used in the rollout and contains:
        - Experiment ID
        - Task description (what problem to solve)
        - Current population (for selection)
        - Population accessor (for efficient O(1) queries)
        - Archive accessor (for global archive access with lineage tracking)
        - Shared state store (for stateful strategies)
        - Config (test cases, etc.)

        Args:
            iteration: Current iteration number
            island_id: Island ID (default: 0). For multi-island mode, only this
                      island's population is included in the context.

        Returns:
            Context instance with accessor and state for module access
        """
        # Get snapshot of island population
        island_population = self.experiment.get_island_population(island_id)

        # Create accessor with population snapshot
        accessor = PopulationAccessor(
            island_id=island_id,
            island_data=island_population,
        )

        # Create island accessor with visibility filtering
        visible_ids = self.experiment.island_index.get(island_id, set())
        island_accessor = IslandAccessor(
            island_id=island_id,
            visible_ids=visible_ids,
            archive=self.experiment.archive,
        )

        # Compute population diversity from feature vectors
        pop_programs = []
        for bucket_progs in island_population.values():
            pop_programs.extend(bucket_progs)
        diversity = self._compute_population_diversity(pop_programs)

        return Context(
            experiment_id=self.experiment.id,
            task_description=self.config.task_description,
            island_id=island_id,
            accessor=accessor,
            island_accessor=island_accessor,
            state=self.strategy_state.create_accessor(island_id),
            experiment_config=self.config,
            diversity=diversity,
            iteration=iteration,
            created_at=time.time(),
        )

    def _get_island_rollout_history(
        self,
        island_id: int,
        window: Optional[int] = None
    ) -> List[RolloutResult]:
        """
        Get rollout history for a specific island.

        Filters global rollout history to only include rollouts from the specified
        island. Optionally limits to the most recent N rollouts.

        Args:
            island_id: Island ID to filter by
            window: Maximum number of recent rollouts to return. If None, returns
                   all rollouts for this island (default: None)

        Returns:
            List of RolloutResult objects for this island (most recent last)
        """
        # Filter global history to this island
        island_history = [
            result for result in self.experiment.rollout_history
            if result.island_id == island_id
        ]
        # Return last N rollouts, or all if window is None
        if window is not None:
            return island_history[-window:]
        return island_history

    # =============================================================================
    # Migration logic
    # =============================================================================
    def _perform_migration(self, migration_size: int) -> None:
        """
        Perform migration between islands using configured topology.

        Programs are sampled based on combined_score (weighted random sampling)
        rather than deterministically selecting top programs. This introduces
        diversity while still favoring higher-scored programs.

        Programs are shared across islands (not copied), as they are immutable.
        The same program object may exist in multiple island populations.

        Args:
            migration_size: Number of programs to migrate per island

        Currently supports ring topology: island i → island (i+1) % num_islands
        """
        num_islands = self.config.num_islands
        topology = self.config.migration_topology

        if topology == "ring":
            # Ring topology: each island sends to next island


            total_added = 0
            to_migrant = []
            for island_id in range(num_islands):
                source_pop = self.experiment.get_island_population(island_id)
                source_programs = self._unique_programs_by_id(
                    self._flatten_population(source_pop)
                )
                # Sample migrants based on combined_score (weighted random)
                migrants = self._sample_by_score(source_programs, migration_size)
                to_migrant.append(migrants)
            for island_id in range(num_islands):
                target_id = (island_id + 1) % num_islands
                migrants = to_migrant[island_id]
                target_pop = self.experiment.get_island_population(target_id)
                updated_pop, added = self._apply_programs_to_population(
                    current_population=target_pop,
                    programs=migrants,
                    island_id=target_id,
                )
                total_added += added
                self.experiment.island_populations[target_id] = updated_pop

            if self.logger:
                self.logger.info(
                    f"[Migration] Migration completed: {migration_size} programs/island, "
                    f"{total_added} total added across {num_islands} islands (ring topology)"
                )

        # TODO: Support other topologies (full, star)

    def _perform_reset(self, reset_size: int) -> None:
        """
        Reset the worst-performing half of islands and reset from top donors.

        Worst-performing is determined by each island's best program score.
        """
        num_islands = self.config.num_islands
        if num_islands < 2:
            return

        worst_count = num_islands // 2
        if worst_count == 0:
            return

        island_scores = []
        for island_id in range(num_islands):
            island_pop = self.experiment.get_island_population(island_id)
            best_program = self._best_program_in_population(island_pop)
            best_score = (
                best_program.combined_score
                if best_program and best_program.combined_score is not None
                else float("-inf")
            )
            island_scores.append((island_id, best_score))

        island_scores.sort(key=lambda item: item[1])
        worst_islands = [island_id for island_id, _ in island_scores[:worst_count]]
        donor_islands = [island_id for island_id, _ in island_scores[worst_count:]]

        donor_programs: List[Program] = []
        for donor_id in donor_islands:
            donor_pop = self.experiment.get_island_population(donor_id)
            donor_programs.extend(self._flatten_population(donor_pop))

        donor_programs = self._unique_programs_by_id(donor_programs)
        donors = self._top_scored_programs(donor_programs, reset_size)
        if not donors:
            return

        for island_id in worst_islands:
            # Clear both population and island_index (visibility layer)
            self.experiment.island_populations[island_id] = {}
            self.experiment.island_index[island_id] = set()

            # Re-populate with donor programs (also updates island_index)
            updated_pop, added = self._apply_programs_to_population(
                current_population={},
                programs=donors,
                island_id=island_id,
            )
            self.experiment.island_populations[island_id] = updated_pop

        self.experiment.updated_at = time.time()

        if self.logger:
            worst_str = ", ".join(str(i) for i in worst_islands)
            donor_str = ", ".join(str(i) for i in donor_islands)
            self.logger.info(
                f"[Reset] Island reset performed: worst=[{worst_str}], donors=[{donor_str}], "
                f"{reset_size} programs per island"
            )

    # =============================================================================
    # Population Helpers
    # =============================================================================

    def _flatten_population(
        self, population: Dict[str, List[Program]]
    ) -> List[Program]:
        """Flatten population buckets into a list of programs."""
        programs: List[Program] = []
        for bucket_progs in population.values():
            programs.extend(bucket_progs)
        return programs

    def _unique_programs_by_id(self, programs: List[Program]) -> List[Program]:
        """Deduplicate programs by ID while preserving order."""
        seen_ids = set()
        unique: List[Program] = []
        for program in programs:
            if program.id in seen_ids:
                continue
            seen_ids.add(program.id)
            unique.append(program)
        return unique

    def _top_scored_programs(
        self, programs: List[Program], limit: int
    ) -> List[Program]:
        """Select top scored programs, skipping those without scores."""
        scored = [p for p in programs if p.combined_score is not None]
        scored.sort(key=lambda p: p.combined_score, reverse=True)
        return scored[:limit]

    def _sample_by_score(
        self, programs: List[Program], count: int
    ) -> List[Program]:
        """
        Sample programs with probability proportional to combined_score.

        Uses softmax-style weighting to convert scores to probabilities.
        Programs without scores are excluded.

        Args:
            programs: List of programs to sample from
            count: Number of programs to sample

        Returns:
            List of sampled programs (without replacement)
        """
        import random
        import math

        # Filter to programs with scores
        scored = [p for p in programs if p.combined_score is not None]
        if not scored:
            return []

        # Limit count to available programs
        count = min(count, len(scored))
        if count == 0:
            return []

        # Extract scores and compute softmax weights
        scores = [p.combined_score for p in scored]
        max_score = max(scores)

        # Use softmax with temperature=1 to convert scores to probabilities
        # Shift by max_score for numerical stability
        exp_scores = [math.exp(s - max_score) for s in scores]
        total = sum(exp_scores)
        weights = [e / total for e in exp_scores]

        # Sample without replacement using weighted random selection
        sampled: List[Program] = []
        remaining = list(zip(scored, weights))

        for _ in range(count):
            if not remaining:
                break

            # Normalize weights for remaining programs
            total_weight = sum(w for _, w in remaining)
            if total_weight == 0:
                break

            # Select one program based on weights
            r = random.random() * total_weight
            cumulative = 0.0
            selected_idx = 0

            for idx, (_, w) in enumerate(remaining):
                cumulative += w
                if r <= cumulative:
                    selected_idx = idx
                    break

            # Add selected program and remove from remaining
            selected_program, _ = remaining.pop(selected_idx)
            sampled.append(selected_program)

        return sampled

    def _best_program_in_population(
        self, population: Dict[str, List[Program]]
    ) -> Optional[Program]:
        """Get the best program in a population (by combined_score)."""
        programs = [p for p in self._flatten_population(population) if p.combined_score is not None]
        if not programs:
            return None
        return max(programs, key=lambda p: p.combined_score)

    def _apply_programs_to_population(
        self,
        current_population: Dict[str, List[Program]],
        programs: List[Program],
        island_id: int,
    ) -> Tuple[Dict[str, List[Program]], int]:
        """
        Add programs to a population via PopulationModule with dedupe.

        Programs are immutable and shared across islands (not copied).
        Also updates island_index to maintain visibility layer.

        Returns updated population and number of programs retained.
        """
        existing_ids = {p.id for p in self._flatten_population(current_population)}
        updated_population = current_population
        added = 0

        # Ensure island_index exists for this island
        if island_id not in self.experiment.island_index:
            self.experiment.island_index[island_id] = set()

        for program in programs:
            if program.id in existing_ids:
                continue
            # Programs are immutable and can be safely shared across islands
            updated_population = self.population_module.update_population(
                current_population=updated_population,
                new_program=program,
                experiment_config=self.config,
                island_id=island_id,
            )
            updated_ids = {p.id for p in self._flatten_population(updated_population)}
            if program.id in updated_ids:
                added += 1
                # Add to island's visible set (island_index)
                self.experiment.island_index[island_id].add(program.id)
            existing_ids = updated_ids

        return updated_population, added

    def _compute_population_diversity(self, programs: List[Program]) -> float:
        """
        Compute population diversity as average pairwise cosine distance.

        Uses feature vectors of programs to measure diversity. Higher values
        indicate more diverse populations.

        Args:
            programs: List of programs to analyze

        Returns:
            Average pairwise cosine distance (0 = all identical, 2 = maximally diverse)
        """
        if len(programs) < 2:
            return 0.0

        # Extract feature vectors from programs
        vectors = [p.feature_vector for p in programs if p.feature_vector]
        if len(vectors) < 2:
            return 0.0

        # Compute average pairwise cosine distance
        total_dist = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total_dist += cosine_distance(vectors[i], vectors[j])
                count += 1

        return total_dist / count if count > 0 else 0.0
        
    # =============================================================================
    # Logging Helpers
    # =============================================================================

    def _log_separator(self, char: str = "=", length: int = 80) -> None:
        """Log a separator line for visual organization."""
        self.logger.info(char * length)

    def _log_box(self, lines: List[str], width: int = 80) -> None:
        """
        Log a boxed message for major events.

        Args:
            lines: List of strings to display in the box
            width: Total width of the box (default: 80)
        """
        top = "╔" + "═" * (width - 2) + "╗"
        bottom = "╚" + "═" * (width - 2) + "╝"

        self.logger.info(top)
        for line in lines:
            # Center-pad the line
            padding = width - 4 - len(line)  # -4 for "║ " and " ║"
            left_pad = padding // 2
            right_pad = padding - left_pad
            formatted = f"║ {' ' * left_pad}{line}{' ' * right_pad} ║"
            self.logger.info(formatted)
        self.logger.info(bottom)

    def _log_evolution_start(self) -> None:
        """Log evolution start with separator."""
        num_islands = self.config.num_islands
        max_iterations = self.config.max_iterations
        start_iteration = self.experiment.current_iteration + 1

        self._log_separator("-", 79)

        if num_islands == 1:
            self.logger.info(
                f"[Start] Evolution: {self.experiment.name}, Experiment ID: {self.experiment.id}, "
                f"Iterations: {start_iteration} → {max_iterations}, Workers: {self.config.max_workers}"
            )
        else:
            self.logger.info(
                f"[Start] Evolution: {self.experiment.name}, Experiment ID: {self.experiment.id}"
            )
            self.logger.info(
                f"[Start] Iterations: {start_iteration} → {max_iterations}, Islands: {num_islands}, "
                f"Workers: {self.config.max_workers}, Island Sizes: {self.config.island_sizes}, "
                f"Population Size: {self.config.population_size}, "
                f"Migration: every {self.config.migration_interval} iters"
            )

        self._log_separator("-", 79)

    def _log_evolution_complete(self) -> None:
        """Log evolution completion with summary."""
        num_islands = self.config.num_islands
        total_population = sum(
            len(bucket_progs)
            for island_pop in self.experiment.island_populations.values()
            for bucket_progs in island_pop.values()
        )

        self._log_separator("-", 79)

        if num_islands == 1:
            self.logger.info(
                f"[Complete] Evolution complete: {self.experiment.name}, "
                f"final_iteration={self.experiment.current_iteration}, "
                f"population={total_population}, archive_size={len(self.experiment.archive)}"
            )
        else:
            self.logger.info(
                f"[Complete] Multi-island evolution complete: {self.experiment.name}, "
                f"final_iteration={self.experiment.current_iteration}, islands={num_islands}, "
                f"total_population={total_population}, archive_size={len(self.experiment.archive)}"
            )

        # Log best program per island
        for island_id in range(num_islands):
            island_pop = self.experiment.get_island_population(island_id)
            all_programs = [
                p for bucket_progs in island_pop.values() for p in bucket_progs
            ]
            if all_programs:
                best = max(
                    all_programs,
                    key=lambda p: p.combined_score if p.combined_score else 0.0,
                )
                if num_islands == 1:
                    self.logger.info(
                        f"[Complete] Best program: {best.id}"
                        f"(score={best.combined_score:.4f}, population={len(all_programs)})"
                    )
                else:
                    self.logger.info(
                        f"[Complete] Island {island_id} best: {best.id}"
                        f"(score={best.combined_score:.4f}, population={len(all_programs)})"
                    )

        self._log_separator("-", 79)

    def _log_evolution_aborted(self, error: Exception) -> None:
        """Log aborted evolution summary when execution terminates early."""
        num_islands = self.config.num_islands
        total_population = sum(
            len(bucket_progs)
            for island_pop in self.experiment.island_populations.values()
            for bucket_progs in island_pop.values()
        )

        self._log_separator("-", 79)

        if num_islands == 1:
            self.logger.error(
                f"[Aborted] Evolution aborted: {self.experiment.name}, "
                f"final_iteration={self.experiment.current_iteration}, "
                f"population={total_population}, archive_size={len(self.experiment.archive)}, "
                f"error={error}"
            )
        else:
            self.logger.error(
                f"[Aborted] Multi-island evolution aborted: {self.experiment.name}, "
                f"final_iteration={self.experiment.current_iteration}, islands={num_islands}, "
                f"total_population={total_population}, archive_size={len(self.experiment.archive)}, "
                f"error={error}"
            )

        self._log_separator("-", 79)

    def _log_iteration_result(
        self,
        iteration: int,
        max_iterations: int,
        child: Program,
        parent_id: str,
        elapsed_time: float,
        island_id: Optional[int] = None,
    ) -> None:
        """
        Log iteration result in multi-line format for better readability.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum iterations
            child: Generated child program
            parent_id: Parent program ID
            elapsed_time: Elapsed time in seconds
            island_id: Island ID (None for single-island)
        """
        # Get parent program to calculate deltas
        parent = None
        if parent_id and parent_id != "unknown":
            island_pop = self.experiment.get_island_population(island_id or 0)
            for bucket_progs in island_pop.values():
                for p in bucket_progs:
                    if p.id == parent_id:
                        parent = p
                        break
                if parent is not None:
                    break

        # Build island-iteration prefix
        if self.config.num_islands > 1 and island_id is not None:
            prefix = f"[Iter Result] Island {island_id} | Iter {iteration}/{max_iterations}"
        else:
            prefix = f"[Iter Result] Iter {iteration}/{max_iterations}"

        # Build header line
        header = f"{prefix} {child.id} ← {parent_id}"

        # Build metrics parts
        metrics_parts = []

        # Child metrics
        if child.metrics:
            for key, value in child.metrics.items():
                if isinstance(value, float):
                    metrics_parts.append(f"{key}={value:.4f}")
                elif isinstance(value, int):
                    metrics_parts.append(f"{key}={value}")
                else:
                    metrics_parts.append(f"{key}={value}")

        # Validity
        if child.validity is not None:
            metrics_parts.append(f"validity={child.validity:.4f}")

        # Combined score (with emoji)
        score_val = child.combined_score if child.combined_score is not None else 0.0
        metrics_parts.append(f"🪙 combined_score={score_val:.4f}")

        # Build delta parts if parent exists
        delta_parts = []
        if parent:
            # Calculate deltas for all metrics
            if child.metrics and parent.metrics:
                for key in child.metrics.keys():
                    if key in parent.metrics:
                        child_val = child.metrics[key]
                        parent_val = parent.metrics[key]
                        if isinstance(child_val, (int, float)) and isinstance(parent_val, (int, float)):
                            delta = child_val - parent_val
                            delta_parts.append(f"{key}={delta:+.4f}")

            # Validity delta
            if child.validity is not None and parent.validity is not None:
                delta = child.validity - parent.validity
                delta_parts.append(f"validity={delta:+.4f}")

            # Combined score delta
            if child.combined_score is not None and parent.combined_score is not None:
                delta = child.combined_score - parent.combined_score
                delta_parts.append(f"combined_score={delta:+.4f}")

        # Build multi-line message (no indentation on continuation lines)
        lines = [header]
        lines.append(f"Metrics: {', '.join(metrics_parts)}")
        if delta_parts:
            lines.append(f"Deltas: {', '.join(delta_parts)}")
        lines.append(f"Time: {elapsed_time:.2f}s")

        # Log as multi-line (LocalLogger will handle splitting on \n)
        log_message = "\n".join(lines)
        self.logger.info(log_message)

    def _log_population_stats_compact(
        self,
        island_id: int,
        all_programs: List[Program],
        diversity: float,
    ) -> None:
        """
        Log population statistics (matching example.log format).

        Args:
            island_id: Island identifier
            all_programs: All programs in the island
            diversity: Population diversity score
        """
        if not all_programs:
            return

        scores = [
            p.combined_score for p in all_programs if p.combined_score is not None
        ]
        if not scores:
            return

        # Calculate statistics
        size = len(all_programs)
        best_score = max(scores)
        avg_score = sum(scores) / len(scores)
        max_gen = max((p.generation for p in all_programs), default=0)

        # Count errors by type (if available)
        error_count = 0
        for p in all_programs:
            if p.is_buggy:
                # Try to extract error type
                error_count += 1

        # Format like example.log with [stats] tag
        if self.config.num_islands > 1:
            prefix = f"[Stats] Island {island_id}:"
        else:
            prefix = "[Stats] Population:"

        stats_line = (
            f"{prefix} {size} programs, best={best_score:.4f}, avg={avg_score:.4f}, "
            f"diversity={diversity:.2f}, gen={max_gen}, error_count={error_count}"
        )

        self.logger.info(stats_line)

    def _log_new_best_program(
        self,
        new_best: Program,
        old_best_id: str,
        old_best_score: float,
        island_id: Optional[int] = None,
    ) -> None:
        """Log when a new best program is found."""
        improvement = new_best.combined_score - old_best_score

        if self.config.num_islands > 1 and island_id is not None:
            log_message = (
                f"[Best] island {island_id}: New best program {new_best.id} replaces {old_best_id} "
                f"(combined_score: {old_best_score:.4f} → {new_best.combined_score:.4f}, {improvement:+.4f})"
            )
        else:
            log_message = (
                f"[Best] New best program {new_best.id} replaces {old_best_id} "
                f"(combined_score: {old_best_score:.4f} → {new_best.combined_score:.4f}, {improvement:+.4f})"
            )

        self.logger.info(log_message)

    # =============================================================================
    # Experiment Status
    # =============================================================================

    def _update_experiment(self, result: RolloutResult, island_id: int = 0) -> None:
        """
        Update experiment state with rollout result.

        Only successful rollouts update the population. Failed rollouts are
        recorded in history but do not add programs to the population.

        Updates:
        - Add new programs to population (only if rollout succeeded)
        - Append result to rollout history
        - Increment iteration counter
        - Update timestamp

        Args:
            result: RolloutResult from completed rollout
            island_id: Island ID to update (default: 0 for single-island mode)
        """
        # Track old best for comparison
        old_best_score = None
        old_best_id = None
        if self.logger:
            island_pop = self.experiment.get_island_population(island_id)
            all_programs_before = [
                p for bucket_progs in island_pop.values() for p in bucket_progs
            ]
            if all_programs_before:
                best_before = max(
                    all_programs_before,
                    key=lambda p: p.combined_score if p.combined_score else 0.0,
                )
                if best_before.combined_score:
                    old_best_score = best_before.combined_score
                    old_best_id = best_before.id

        # Only update population if rollout succeeded
        if result.status == "success" and result.generated_program:
            result.generated_program.meta["island_id"] = island_id
            # Save program to individual file
            if self.data_service:
                self.data_service.save_program(result.generated_program)

            # Sync parent program's island_ids so parent tracks all islands its
            # children have appeared in (useful for migration / lineage tracing)
            parent_id = result.generated_program.parent_id
            if parent_id and parent_id in self.experiment.archive:
                parent = self.experiment.archive[parent_id]
                existing = set(
                    parent.meta.get(
                        "island_ids",
                        [parent.meta["island_id"]] if "island_id" in parent.meta else [],
                    )
                )
                if island_id not in existing:
                    parent.meta["island_ids"] = sorted(existing | {island_id})
                    if self.data_service:
                        self.data_service.save_program(parent)

            # Add new program to archive (all programs ever created)
            self.experiment.archive[result.generated_program.id] = result.generated_program

            # Add to island's visible set (island_index)
            if island_id not in self.experiment.island_index:
                self.experiment.island_index[island_id] = set()
            self.experiment.island_index[island_id].add(result.generated_program.id)

            # Update best program tracking
            self.experiment.update_best_program(result.generated_program)

            # Get current island population
            current_pop = self.experiment.get_island_population(island_id)

            # Update population using PopulationModule (required)
            updated_population = self.population_module.update_population(
                current_population=current_pop,
                new_program=result.generated_program,
                experiment_config=self.config,
                island_id=island_id,
            )
            # Update experiment with new island population
            self.experiment.island_populations[island_id] = updated_population
        elif result.status == RolloutStatus.FAILED:
            # 任意模块未正常完成 -> 整体作废,不入 population、不入 history、不计 iteration
            if self.logger:
                self.logger.warning(
                    "[Failed] Rollout discarded due to module failure. "
                    f"Module: {result.failed_module}, Error: {result.error_message}"
                )
        elif result.status == RolloutStatus.FATAL and self.logger:
            self.logger.error(
                "[Fatal] Rollout hit a non-recoverable framework/strategy error. "
                f"Module: {result.failed_module}, Error: {result.error_message}"
            )

        counts_toward_iteration = result.status == RolloutStatus.SUCCESS

        # Enrichment 事务确认:成功完成的 rollout 触发每个 enrichment 的 commit 钩子
        # (例如 DebugEnrichment 用它来更新动态触发率统计)
        if counts_toward_iteration:
            for enr in self.enrichments:
                enr.on_rollout_complete(result)

        # Store rollout result in history only for formal iterations
        if counts_toward_iteration:
            self.experiment.rollout_history.append(result)

        # Apply deferred state_updates first (only for non-DISCARDED)
        if counts_toward_iteration and result.state_updates is not None:
            self.strategy_state.apply_updates(result.state_updates.updates)

        # Then on_rollout_complete for ALL statuses (including DISCARDED)
        if hasattr(self, "current_strategy") and self.current_strategy is not None:
            on_rollout_complete = getattr(self.current_strategy, "on_rollout_complete", None)
            if callable(on_rollout_complete):
                on_rollout_complete(result)

        result.strategy_state = self._build_strategy_trace_snapshot()
        self._append_rollout_llm_request_logs(result)

        # Save rollout result to individual file
        if self.data_service:
            self.data_service.save_result(result)

        # Increment iteration counter only when a formal rollout was consumed
        if counts_toward_iteration:
            self.experiment.current_iteration += 1

        # Update timestamp
        self.experiment.updated_at = time.time()

        if self.logger and result.status == RolloutStatus.SUCCESS and result.generated_program:
            # Calculate elapsed time
            elapsed_time = 0.0
            if result.stats and "execution_time" in result.stats:
                elapsed_time = result.stats["execution_time"]

            parent_id = result.selection.parent_id if result.selection else "unknown"

            # Log iteration result using helper
            self._log_iteration_result(
                iteration=self.experiment.current_iteration,
                max_iterations=self.config.max_iterations,
                child=result.generated_program,
                parent_id=parent_id,
                elapsed_time=elapsed_time,
                island_id=island_id if self.config.num_islands > 1 else None,
            )

            # Log population statistics
            self._log_population_stats(island_id, old_best_score, old_best_id)

    def _append_rollout_llm_request_logs(self, result: RolloutResult) -> None:
        """Append worker-returned LLM request logs into the driver-local experiment log."""
        if not result.llm_request_logs:
            return

        experiment_id = result.experiment_id
        if not experiment_id:
            return

        log_path: Optional[Path] = None
        # Use dedicated llm_request_log_dir from storage config if configured
        if (
            self.data_service is not None
            and hasattr(self.data_service, "llm_request_log_dir")
            and self.data_service.llm_request_log_dir
        ):
            log_path = Path(self.data_service.llm_request_log_dir) / "llm_requests.log"

        if log_path is None and self.data_service is not None and hasattr(self.data_service, "get_experiment_dir"):
            try:
                exp_dir = self.data_service.get_experiment_dir(experiment_id)
                log_path = Path(exp_dir) / "llm_requests.log"
            except Exception:
                log_path = None

        if log_path is None and self.logger is not None and hasattr(self.logger, "text_log_file"):
            text_log_file = getattr(self.logger, "text_log_file", None)
            if text_log_file:
                log_path = Path(text_log_file).parent / "llm_requests.log"

        if log_path is None:
            return

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            for entry in result.llm_request_logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _log_population_stats(
        self,
        island_id: int,
        old_best_score: Optional[float],
        old_best_id: Optional[str],
    ) -> None:
        """
        Log population statistics for an island.

        Calculates and logs:
        - Population size, best/average scores
        - Diversity (average pairwise cosine distance)
        - Max generation, error count
        - New best program notification if applicable

        Args:
            island_id: Island to log stats for
            old_best_score: Previous best score (for detecting improvement)
            old_best_id: Previous best program ID
        """
        if not self.logger:
            return

        island_pop = self.experiment.get_island_population(island_id)
        all_programs = [
            p for bucket_progs in island_pop.values() for p in bucket_progs
        ]
        scores = [
            p.combined_score for p in all_programs if p.combined_score is not None
        ]

        if not scores:
            return

        best_score = max(scores)
        best_program = max(
            all_programs,
            key=lambda p: p.combined_score if p.combined_score else 0.0,
        )

        # Calculate diversity
        diversity = self._compute_population_diversity(all_programs)

        # Check if this is a new best program
        if (
            old_best_score is not None
            and best_score > old_best_score
            and old_best_id != best_program.id
        ):
            self._log_new_best_program(
                new_best=best_program,
                old_best_id=old_best_id,
                old_best_score=old_best_score,
                island_id=island_id if self.config.num_islands > 1 else None,
            )

        # Log compact population stats
        self._log_population_stats_compact(
            island_id=island_id,
            all_programs=all_programs,
            diversity=diversity,
        )

    def _build_strategy_trace_snapshot(self) -> Dict[str, object]:
        """Build a trace-friendly strategy snapshot without changing resume semantics."""
        snapshot = self.strategy_state.model_dump(mode="json")
        if hasattr(self, "current_strategy") and self.current_strategy is not None:
            snapshot["strategy_checkpoint"] = self.current_strategy.dump_state()
        return snapshot

    def _finalize_strategy_before_shutdown(self) -> None:
        """Let strategies flush active runtime state into the last checkpoint."""
        if not hasattr(self, "current_strategy") or self.current_strategy is None:
            return

        finalize_hook = getattr(self.current_strategy, "finalize_experiment", None)
        if not callable(finalize_hook):
            return

        contexts = {
            island_id: self._create_context(
                iteration=self.experiment.current_iteration,
                island_id=island_id,
            )
            for island_id in self.experiment.island_populations.keys()
        }
        finalize_hook(contexts)

    def _save_checkpoint(self) -> None:
        """
        Save experiment checkpoint to storage.

        Persists:
        - Experiment state (population, history, etc.)
        - Strategy state (from StateStore)
        - All programs in the archive

        Saves to experiment_checkpoint_{iteration}.json where iteration is
        taken from experiment.current_iteration.
        """
        if not self.data_service:
            return

        try:
            # Sync state registry to experiment before saving
            self.experiment.strategy_state = self._build_strategy_trace_snapshot()

            # Save strategy-level checkpoint if strategy is available
            if hasattr(self, 'current_strategy') and self.current_strategy is not None:
                self.experiment.strategy_checkpoint = self.current_strategy.dump_state()

            # Save experiment checkpoint (uses iteration from experiment)
            self.data_service.save_experiment(self.experiment)

            if self.logger:
                self.logger.info(
                    f"[Checkpoint] Saved checkpoint at iteration {self.experiment.current_iteration} "
                    f"(archive_size={len(self.experiment.archive)})"
                )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"[Checkpoint] Failed to save checkpoint at iteration {self.experiment.current_iteration}: {str(e)}"
                )

    def _record_metrics(self, iteration: int) -> None:
        """
        Record metrics to monitoring backend.

        Collects and logs performance, population, diversity, and island metrics.
        Called after each rollout completes.

        Args:
            iteration: Current iteration number
        """
        if not self.monitor:
            return

        try:
            # Import here to avoid circular dependency
            from famou.infrastructure.monitor import EvolutionMetricsCollector, IslandMetrics

            # Collect global metrics
            collector = EvolutionMetricsCollector()
            metrics = collector.collect_from_experiment(self.experiment, iteration)

            # Collect per-island metrics (if multi-island)
            if self.experiment.num_islands > 1:
                island_collector = IslandMetrics()
                island_metrics = island_collector.collect_per_island_metrics(
                    self.experiment, iteration
                )
                # Remove duplicate iteration/iteration
                island_metrics.pop("iteration/iteration", None)
                metrics.update(island_metrics)

            # Debug: log metrics being recorded
            if self.logger:
                self.logger.debug(
                    f"Recording {len(metrics)} metrics to monitoring backend",
                    iteration=iteration,
                    metric_keys=list(metrics.keys())[:5]  # Log first 5 keys
                )

            # Log to monitoring backend (async, non-blocking)
            self.monitor.log_metrics(metrics, step=iteration)

        except Exception as e:
            # Don't let monitoring errors interrupt evolution
            if self.logger:
                self.logger.warning(
                    "Failed to record metrics",
                    iteration=iteration,
                    error=str(e),
                    exc_info=True,  # Include stack trace for debugging
                )

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"Evolver("
            f"experiment={self.experiment.name}, "
            f"current_iteration={self.experiment.current_iteration})"
        )

    # =============================================================================
    # Generate output JSON files (evolve.json & recommend_solution.json)
    # =============================================================================

    def _get_storage_path(self) -> Optional[str]:
        """Get the storage base path from data_service."""
        if hasattr(self.data_service, "get_experiment_dir"):
            try:
                return str(self.data_service.get_experiment_dir(self.experiment.id))
            except Exception:
                pass
        if hasattr(self.data_service, 'base_path'):
            return str(self.data_service.base_path)
        return None

    def _generate_output_json_files(self) -> None:
        """
        Generate evolve.json and recommend_solution.json at the end of evolution.

        evolve.json 是实验级的轻量汇总(best score / iteration / params),由本类
        生成;recommend_solution.json 是从 live population 中选出的代表性解集,
        委托给 RecommendationExporter 处理。
        """
        # Get storage path
        storage_path = self._get_storage_path()
        if not storage_path:
            if self.logger:
                self.logger.warning("[Output] Cannot determine storage path, skipping JSON generation")
            return

        # Output directory defaults to {storage_path}/{experiment_id}/ unless storage resolves it directly.
        output_dir = Path(storage_path)
        if output_dir.name != self.experiment.id and not output_dir.joinpath("config.yaml").exists():
            output_dir = output_dir / self.experiment.id
        output_dir.mkdir(parents=True, exist_ok=True)

        # 轻量产出:experiment-level evolve.json 留在 evolver 内
        self._generate_evolve_json(output_dir)

        # 重头戏:推荐解选择与写文件委托给独立 exporter
        from famou.controller.recommendation import RecommendationExporter
        RecommendationExporter(self.experiment, logger=self.logger).export(output_dir)

        if self.logger:
            self.logger.info(f"[Output] Generated output JSON files in {output_dir}")

    def _generate_evolve_json(self, output_dir: Path) -> None:
        """Generate evolve.json with execution results and parameters."""
        import json

        # Get best program
        best_program = self.experiment.best_program
        best_combined_score = best_program.combined_score if best_program and best_program.combined_score else 0.0
        best_iteration = best_program.iteration if best_program else 0
        best_generation = best_program.generation if best_program else 0

        # Build evolve.json content
        evolve_data = {
            "execution_results": {
                "total_iteration": self.experiment.current_iteration,
                "best_combined_score": best_combined_score,
                "best_iteration": best_iteration,
                "best_generation": best_generation
            },
            "parameters": {
                "population_size": self.config.population_size,
                "island_size": self.config.island_size,
                "island_sizes": self.config.island_sizes,
                "num_islands": self.config.num_islands,
                "max_iterations": self.config.max_iterations,
                "coldstart": False
            },
            "user_requirements": self.config.task_description,
            "evaluation_file": self.config.evaluator
        }

        # Write to file
        evolve_file = output_dir / "evolve.json"
        evolve_file.write_text(json.dumps(evolve_data, indent=4, ensure_ascii=False))
