#!/usr/bin/env python3
"""
Main entry point for running Famou experiments.

Usage (new experiment):
    python -m famou -c CONFIG -p PROGRAM [PROGRAM ...] -e EVALUATOR

Usage (resume from checkpoint):
    python -m famou --resume EXPERIMENT_PATH -e EVALUATOR
    python -m famou --resume EXPERIMENT_PATH -e EVALUATOR -c CONFIG  # override config

Example (single seed):
    python -m famou \\
        --config examples/circle_packing/config.yaml \\
        --programs examples/circle_packing/init.py \\
        --evaluator examples/circle_packing/evaluator.py

Example (multiple seeds):
    python -m famou \\
        --config config.yaml \\
        --programs init1.py init2.py init3.py \\
        --evaluator evaluator.py

Example (resume):
    python -m famou \\
        --resume famou_data/circle_packing_evolution_abc123 \\
        --evaluator examples/circle_packing/evaluator.py
"""

import argparse
import signal
import shutil
import sys
from pathlib import Path
from typing import Callable, List

from famou.config import FamouConfig
from famou.core.data import Program
from famou.controller import Evolver
from famou.controller.evaluator_dependency import (
    DeferredEvaluator,
    prepare_evaluator_environment,
)
from famou.infrastructure.factory import InfraFactory
from famou.infrastructure.logger import LocalLogger
from famou.strategies import StrategyRegistry
from famou.utils import generate_experiment_id
from famou.utils.id_gen import generate_initial_program_id

from famou.core.types import Language


def load_config(yaml_path: str) -> FamouConfig:
    """
    Load experiment configuration from YAML file.

    Args:
        yaml_path: Path to YAML config file

    Returns:
        FamouConfig instance
    """
    return FamouConfig.from_yaml(yaml_path)


def load_initial_programs(
    program_paths: List[str],
    experiment_id: str,
    language: Language
) -> List[Program]:
    """
    Load initial program(s) from one or more files.

    This function ONLY loads the code and creates Program objects.
    Enrichment (evaluation, judging, etc.) happens later in Evolver.

    Args:
        program_paths: List of paths to program files
        experiment_id: Experiment ID to add to program metadata
        language: Programming language of the programs

    Returns:
        List of Program objects with unique IDs (init, init_1, ...)

    Example:
        >>> programs = load_initial_programs(
        ...     ["init1.py", "init2.py"],
        ...     "exp_123",
        ...     Language.PYTHON
        ... )
        >>> [p.id for p in programs]
        ['init', 'init_1']
    """
    programs = []

    for i, program_path in enumerate(program_paths):
        path = Path(program_path)
        if not path.exists():
            raise FileNotFoundError(f"Program file not found: {program_path}")

        code = path.read_text()

        # Create initial program WITHOUT scores/evaluation
        initial_program = Program(
            id=generate_initial_program_id(i),
            code=code,
            generation=0,
            iteration=0,
            language=language,
            file_extension=path.suffix, # Preserve original extension (.h, .cpp, .py, etc.)
        )

        # Add experiment_id and file info to metadata
        initial_program.meta["experiment_id"] = experiment_id
        initial_program.meta["source_file"] = str(path.name)

        programs.append(initial_program)

    return programs


def load_evaluator(
    evaluator_path: str,
    function_name: str = "evaluate",
    required_packages: List[str] | None = None,
) -> Callable:
    """
    Build a deferred evaluator callable from a Python file.

    The evaluator module is imported only when the callable is executed inside
    the execution environment. This avoids importing evaluator code in the main
    process before evaluator dependencies have been prepared.

    Args:
        evaluator_path: Path to Python file containing evaluate function (can be relative)
        function_name: Name of the function to load (default: "evaluate")

    Returns:
        Deferred evaluator callable

    Raises:
        FileNotFoundError: If file doesn't exist
        AttributeError: If file doesn't contain the specified function
        ImportError: If module has import errors or cannot be loaded
        TypeError: If found object is not callable
        ValueError: If any other loading error occurs

    Example:
        >>> evaluate_fn = load_evaluator("examples/circle_packing/evaluator.py")
        >>> result = evaluate_fn("/tmp/program.py")
    """
    path = Path(evaluator_path)
    if not path.exists():
        raise ValueError(f"Failed to load evaluator from {evaluator_path}: file not found")
    return DeferredEvaluator(
        str(path),
        function_name=function_name,
        required_packages=required_packages,
    )


def resolve_evaluator_path(args, config: FamouConfig) -> str:
    """Resolve evaluator path from CLI or config."""
    evaluator_path = args.evaluator
    if not evaluator_path and config and config.experiment.evaluator:
        evaluator_path = config.experiment.evaluator
    if not evaluator_path:
        raise ValueError("--evaluator is required (specify via CLI or in config.yaml)")
    return evaluator_path


def prepare_and_load_evaluator(
    *,
    evaluator_path: str,
    config: FamouConfig,
    infra,
    experiment_id: str,
):
    """Resolve evaluator dependencies, persist trace, then return deferred evaluator callable."""
    trace = prepare_evaluator_environment(
        evaluator_path=evaluator_path,
        llm_client=infra.llm,
        env=infra.env,
        storage=infra.storage,
        experiment_id=experiment_id,
        logger=infra.logger,
        prefer_env_install=str(config.infrastructure.env.type).lower() in {"local", "process"},
    )
    evaluate_fn = load_evaluator(
        evaluator_path,
        required_packages=trace["decision"]["final_packages"],
    )
    infra.logger.info(
        "Evaluator prepared",
        evaluator_path=evaluator_path,
        dependency_count=len(trace["decision"]["final_packages"]),
    )
    return evaluate_fn


def prepare_precise_output_path(config: FamouConfig) -> None:
    """Clear and recreate storage.precise_path for a fresh run when configured."""
    precise_path = getattr(config.infrastructure.storage, "precise_path", None)
    if precise_path:
        output_dir = Path(precise_path)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Also handle precise_user_path for dual_write mode
    if getattr(config.infrastructure.storage, "dual_write", False):
        precise_user_path = getattr(config.infrastructure.storage, "precise_user_path", None)
        if precise_user_path:
            output_dir = Path(precise_user_path)
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(storage, experiment_id: str, config: FamouConfig) -> Path:
    """Resolve the final output directory shown to users."""
    if hasattr(storage, "get_experiment_dir"):
        try:
            return Path(storage.get_experiment_dir(experiment_id))
        except Exception:
            pass
    precise_path = getattr(config.infrastructure.storage, "precise_path", None)
    if precise_path:
        return Path(precise_path)
    return Path(config.infrastructure.storage.base_path) / experiment_id


def resolve_strategy(config, infra, evaluate_fn, persisted_decision=None):
    """
    Load the effective strategy without mutating the user-provided config.

    When strategy_router is enabled, the router decides the effective direction
    and strategy for this run. The original config.experiment.strategy remains
    unchanged so the saved config still reflects what the user wrote.
    """
    selected_strategy_name = config.experiment.strategy
    decision = None

    if config.experiment.strategy_router.get("enabled", False):
        from famou.controller.strategy_router import resolve_strategy_route

        decision = resolve_strategy_route(
            config.experiment,
            llm_client=infra.llm,
            persisted_decision=persisted_decision,
        )
        selected_strategy_name = decision.strategy_name
        print(
            f"\n[5/6] Strategy router selected direction '{decision.direction}' "
            f"via {decision.source} -> strategy '{selected_strategy_name}'"
        )
        infra.logger.info(
            "Strategy router resolved strategy",
            configured_strategy=config.experiment.strategy,
            selected_direction=decision.direction,
            selected_strategy=selected_strategy_name,
            source=decision.source,
        )
    else:
        print(f"\n[5/6] Loading strategy '{selected_strategy_name}'...")

    strategy = StrategyRegistry.get(
        selected_strategy_name,
        evaluate_fn=evaluate_fn,
        params=config.modules,
        evaluator_config=infra.evaluator_config,
    )
    print(f"      Strategy: {strategy}")
    return strategy, decision


def persist_strategy_route_trace(storage, experiment_id, decision, logger=None):
    """Persist the full strategy-router trace as a standalone JSON artifact."""
    if decision is None or not decision.trace:
        return None

    filename = "strategy_router_trace.json"
    if hasattr(storage, "save_json_artifact"):
        storage.save_json_artifact(experiment_id, filename, decision.trace)
        if logger:
            logger.info("Persisted strategy router trace", filename=filename)
        return filename

    return None


def persist_strategy_route_decision(experiment, decision, logger=None, trace_filename=None):
    """Attach the final routing decision to experiment metadata for future resume."""
    if decision is None:
        experiment.strategy_router_decision = {}
        return

    experiment.strategy_router_decision = decision.to_dict()
    if trace_filename:
        experiment.strategy_router_decision["trace_file"] = trace_filename
    if logger:
        logger.info(
            "Persisted strategy router decision",
            selected_direction=decision.direction,
            selected_strategy=decision.strategy_name,
            source=decision.source,
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Famou evolutionary optimization experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # New experiment (single seed)
    python -m famou \\
        --config examples/circle_packing/config.yaml \\
        --programs examples/circle_packing/init.py \\
        --evaluator examples/circle_packing/evaluator.py

    # New experiment (multiple seeds for diversity)
    python -m famou \\
        --config config.yaml \\
        --programs init1.py init2.py init3.py \\
        --evaluator evaluator.py

    # Resume from checkpoint
    python -m famou \\
        --resume famou_data/circle_packing_evolution_abc123 \\
        --evaluator examples/circle_packing/evaluator.py

    # Resume with config override (e.g., increase max_iterations)
    python -m famou \\
        --resume famou_data/circle_packing_evolution_abc123 \\
        --config new_config.yaml \\
        --evaluator examples/circle_packing/evaluator.py
        """,
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to YAML configuration file (required for new, optional for resume)",
    )
    parser.add_argument(
        "--programs",
        "-p",
        nargs="+",
        help=(
            "Path(s) to initial program file(s). Multiple files supported for multi-seed experiments. "
            "Can be specified in config.yaml instead."
        ),
    )
    parser.add_argument(
        "--evaluator",
        "-e",
        help=(
            "Path to evaluator module (Python file with 'evaluate' function). "
            "Can be specified in config.yaml instead."
        ),
    )
    parser.add_argument(
        "--resume",
        "-r",
        help="Path to experiment directory to resume from checkpoint",
    )
    parser.add_argument(
        "--iteration",
        "-i",
        type=int,
        default=-1,
        help="Checkpoint iteration to resume from (default: latest)",
    )

    args = parser.parse_args()
    infra = None
    evolver = None
    saved_config = None

    # Validate arguments based on mode
    if args.resume:
        # Resume mode - config and programs are optional
        pass
    else:
        # New experiment mode - config is required
        if not args.config:
            parser.error("--config is required for new experiments")

    print("\n" + "=" * 70)
    print("  FAMOU 2.0 - Evolutionary Code Optimization")
    print("=" * 70)

    # Create basic logger for early logging
    basic_logger = LocalLogger(name="famou", level="INFO", file_enabled=False)

    # Load config first (to get paths from config if not provided via CLI)
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume path not found: {args.resume}")
            sys.exit(1)
        saved_config_path = resume_path / "config.yaml"
        if saved_config_path.exists():
            saved_config = load_config(str(saved_config_path))
            config = load_config(args.config) if args.config else saved_config
        elif args.config:
            config = load_config(args.config)
        else:
            config = None
    else:
        config = load_config(args.config)

    if config is None:
        parser.error("--config is required when resume path has no saved config")

    evaluator_path = resolve_evaluator_path(args, config)
    
    # Check if using hybrid evaluator (remote mode)
    use_hybrid_evaluator = (
        config and 
        config.infrastructure.evaluator and 
        config.infrastructure.evaluator.type == "hybrid"
    )
    
    if use_hybrid_evaluator:
        # Hybrid mode: no local evaluator needed
        evaluate_fn = None
        print(f"\n[1/6] Using HybridEvaluator (remote mode)")
        print(f"      Service URL: {config.infrastructure.evaluator.service_url}")
        basic_logger.info("Using HybridEvaluator", 
            service_url=config.infrastructure.evaluator.service_url)
    else:
        # Local mode: require evaluator_path
        if not evaluator_path:
            parser.error("--evaluator is required for local mode (specify via CLI or in config.yaml)")
        
        print(f"\n[1/6] Loading evaluator from: {evaluator_path}")
        evaluate_fn = load_evaluator(evaluator_path)
        basic_logger.info(f"Evaluator loaded", evaluator=evaluate_fn.__name__)

    # Branch based on mode: resume or new experiment
    if args.resume:
        # =================================================================
        # RESUME MODE
        # =================================================================
        # Extract original experiment_id from path
        original_experiment_id = resume_path.name

        print(
            f"\n[1/6] Loading resume configuration from: "
            f"{args.config if args.config else resume_path / 'config.yaml'}"
        )
        if args.config:
            print(f"      Config override: {args.config}")
            config_changed = saved_config is None or config.to_dict() != saved_config.to_dict()
        else:
            print(f"      Using saved config from: {resume_path / 'config.yaml'}")
            config_changed = False

        basic_logger.info(
            "Configuration loaded for resume",
            experiment=config.experiment.name,
            max_iterations=config.experiment.max_iterations,
            population_size=config.experiment.population_size,
            island_sizes=config.experiment.island_sizes,
            config_changed=config_changed,
        )

        print(f"\n[2/6] Loading experiment: {original_experiment_id}")
        resume_infra_config = saved_config.infrastructure if saved_config is not None else config.infrastructure
        infra = InfraFactory.create_all(
            resume_infra_config,
            experiment_name=config.experiment.name,
            experiment_id=original_experiment_id,
        )
        print(infra)

        # Load experiment from checkpoint
        print(f"\n[3/6] Loading checkpoint (iteration={args.iteration})...")
        experiment = infra.storage.load_experiment(original_experiment_id, iteration=args.iteration)
        loaded_iteration = experiment.current_iteration
        print(f"      Loaded iteration: {loaded_iteration}")
        print(f"      Archive size: {len(experiment.archive)}")

        # Detect if this is the latest checkpoint
        is_latest = True
        if args.iteration != -1:
            # User specified explicit iteration - check if it's actually the latest
            latest_experiment = infra.storage.load_experiment(original_experiment_id, iteration=-1)
            is_latest = (latest_experiment.current_iteration == loaded_iteration)

        # Determine if we should fork
        should_fork = (not is_latest) or config_changed
        fork_reason = None
        if not is_latest:
            fork_reason = "previous_checkpoint"
        elif config_changed:
            fork_reason = "config_changed"

        if should_fork:
            # =============================================================
            # FORK: Create new experiment from checkpoint
            # =============================================================
            new_experiment_id = generate_experiment_id(config.experiment.name)

            print(f"\n      Forking experiment (reason: {fork_reason})")
            print(f"      Original: {original_experiment_id} (iteration {loaded_iteration})")
            print(f"      New:      {new_experiment_id}")

            # Set resume_from tracking
            experiment.resume_from = {
                "experiment_id": original_experiment_id,
                "iteration": loaded_iteration,
            }
            experiment.id = new_experiment_id

            # Re-create infrastructure with new experiment_id
            prepare_precise_output_path(config)
            infra = InfraFactory.create_all(
                config.infrastructure,
                experiment_name=config.experiment.name,
                experiment_id=new_experiment_id,
            )

            # Save config to new folder
            infra.storage.save_config(new_experiment_id, config)

            # Copy programs to new folder (update experiment_id in meta)
            print(f"      Copying {len(experiment.archive)} programs to new folder...")
            for program in list(experiment.archive.values()):
                program.meta["experiment_id"] = new_experiment_id
                infra.storage.save_program(program)

            infra.logger.info(
                "Forked experiment",
                original_id=original_experiment_id,
                original_iteration=loaded_iteration,
                new_id=new_experiment_id,
                reason=fork_reason,
            )

            experiment_id = new_experiment_id
        else:
            # =============================================================
            # PURE RESUME: Continue same experiment
            # =============================================================
            experiment_id = original_experiment_id
            infra.logger.info(
                "Resuming experiment",
                experiment_id=experiment_id,
                iteration=loaded_iteration,
            )

        print(f"\n[4/6] Preparing evaluator from: {evaluator_path}")
        evaluate_fn = prepare_and_load_evaluator(
            evaluator_path=evaluator_path,
            config=config,
            infra=infra,
            experiment_id=experiment_id,
        )

        persisted_route = None if config_changed else experiment.strategy_router_decision
        strategy, route_decision = resolve_strategy(
            config,
            infra,
            evaluate_fn,
            persisted_decision=persisted_route,
        )

        # Create evolver in resume mode (with experiment, not initial_programs)
        evolver = Evolver(
            config=config.experiment,
            population_module=strategy.population_module,
            experiment=experiment,  # Resume mode
            logger=infra.logger,
            llm_client=infra.llm,
            env=infra.env,
            data_service=infra.storage,
            embedding_client=infra.embedding,
            monitor=infra.monitor,
            langfuse=infra.langfuse,
            backend=infra.backend,
            evaluator_path=evaluator_path,
        )
        trace_filename = persist_strategy_route_trace(
            infra.storage,
            evolver.experiment.id,
            route_decision,
            logger=infra.logger,
        )
        persist_strategy_route_decision(
            evolver.experiment,
            route_decision,
            logger=infra.logger,
            trace_filename=trace_filename,
        )

        if should_fork:
            print(f"\n[6/6] Starting forked evolution from iteration {experiment.current_iteration}...")
        else:
            print(f"\n[6/6] Resuming evolution from iteration {experiment.current_iteration}...")
        print("=" * 70)

    else:
        # =================================================================
        # NEW EXPERIMENT MODE
        # =================================================================
        print(f"\n[1/6] Loading configuration from: {args.config}")
        basic_logger.info(
            "Configuration loaded",
            experiment=config.experiment.name,
            population_size=config.experiment.population_size,
            max_iterations=config.experiment.max_iterations,
            island_sizes=config.experiment.island_sizes,
        )

        # Generate experiment ID
        experiment_id = generate_experiment_id(config.experiment.name)
        basic_logger.info("Experiment ID generated", experiment_id=experiment_id)

        # Setup infrastructure
        prepare_precise_output_path(config)
        print("\n[2/6] Setting up infrastructure...")
        infra = InfraFactory.create_all(
            config.infrastructure,
            experiment_name=config.experiment.name,
            experiment_id=experiment_id,
        )
        print(infra)
        infra.logger.info(
            "Infrastructure initialized",
            llm_provider=config.infrastructure.llm.provider,
            llm_model=config.infrastructure.llm.model,
            storage_type=config.infrastructure.storage.type,
            storage_path=config.infrastructure.storage.base_path,
        )

        # Save config for future resume
        infra.storage.save_config(experiment_id, config)
        infra.logger.info("Config saved for resume support")

        print(f"\n[3/6] Preparing evaluator from: {evaluator_path}")
        evaluate_fn = prepare_and_load_evaluator(
            evaluator_path=evaluator_path,
            config=config,
            infra=infra,
            experiment_id=experiment_id,
        )

        # Resolve programs: CLI args > config > error
        program_paths = args.programs
        if not program_paths and config.experiment.programs:
            # Parse comma-separated programs from config
            program_paths = [p.strip() for p in config.experiment.programs.split(",") if p.strip()]
        if not program_paths:
            parser.error("--programs is required (specify via CLI or in config.yaml)")

        # Load initial programs
        num_files = len(program_paths)
        if num_files == 1:
            print(f"\n[4/6] Loading initial program from: {program_paths[0]}")
        else:
            print(f"\n[4/6] Loading {num_files} initial programs:")
            for p in program_paths:
                print(f"      - {p}")
        initial_programs = load_initial_programs(
            program_paths,
            experiment_id,
            config.experiment.language,
        )
        infra.logger.info(
            "Initial programs loaded",
            num_programs=len(initial_programs),
            files=[str(Path(p).name) for p in program_paths],
        )

        strategy, route_decision = resolve_strategy(config, infra, evaluate_fn)

        # Create evolver with initial programs (new experiment mode)
        evolver = Evolver(
            config=config.experiment,
            population_module=strategy.population_module,
            initial_programs=initial_programs,
            logger=infra.logger,
            llm_client=infra.llm,
            env=infra.env,
            data_service=infra.storage,
            experiment_id=experiment_id,
            embedding_client=infra.embedding,
            monitor=infra.monitor,
            langfuse=infra.langfuse,
            backend=infra.backend,
            evaluator_path=evaluator_path,
        )
        trace_filename = persist_strategy_route_trace(
            infra.storage,
            evolver.experiment.id,
            route_decision,
            logger=infra.logger,
        )
        persist_strategy_route_decision(
            evolver.experiment,
            route_decision,
            logger=infra.logger,
            trace_filename=trace_filename,
        )

        print(f"\n[6/6] Starting evolution...")
        print("=" * 70)

    try:
        # 注册 SIGTERM 处理: 将 SIGTERM 转为 SystemExit，确保 finally 块执行
        # 这在 Ray Job 场景下尤为重要 (ray job stop / 调度器终止 / Docker stop)
        def _sigterm_handler(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, _sigterm_handler)

        final_experiment = evolver.run(strategy)

        # Print results
        print("\n" + "=" * 70)
        print("  EVOLUTION COMPLETE")
        print("=" * 70)

        # Get best program
        population = final_experiment.get_all_programs()
        if population:
            best_program = max(
                population, key=lambda p: p.combined_score if p.combined_score else 0.0
            )

            print(f"\n  Iterations completed: {final_experiment.current_iteration}")
            print(f"  Archive size: {len(final_experiment.archive)}")
            print(f"  Final population size: {len(population)}")
            print(f"\n  Best program:")
            print(f"    ID: {best_program.id}")
            print(f"    Score: {best_program.combined_score:.6f}")
            print(f"    Generation: {best_program.generation}")

            if best_program.metrics:
                print(f"    Metrics: {best_program.metrics}")

            # Print top 5 programs
            print(f"\n  Top 5 programs:")
            sorted_pop = sorted(
                population,
                key=lambda p: p.combined_score if p.combined_score else 0.0,
                reverse=True,
            )
            for i, prog in enumerate(sorted_pop[:5]):
                score = prog.combined_score if prog.combined_score else 0.0
                print(f"    {i+1}. {prog.id}: {score:.6f}")

        # Print storage location
        output_dir = resolve_output_dir(infra.storage, final_experiment.id, config)
        print("\n" + "=" * 70)
        print("  DATA STORAGE")
        print("=" * 70)
        print(f"\n  All results saved to: {output_dir}/")
        print(f"    - Experiment state: experiment.json")
        print(f"    - Programs: programs/{len(final_experiment.archive)} files")
        print(f"    - Rollout results: results/{len(final_experiment.rollout_history)} files")
        print("\n" + "=" * 70)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("  EVOLUTION INTERRUPTED BY USER")
        print("=" * 70)

        infra.logger.warning(
            "Evolution interrupted by user",
            current_iteration=evolver.experiment.current_iteration if evolver else None,
            archive_size=len(evolver.experiment.archive) if evolver else None,
        )

        # Save checkpoint before exiting
        if infra and infra.storage and evolver:
            output_dir = resolve_output_dir(infra.storage, evolver.experiment.id, config)
            infra.logger.info(f"Saving checkpoint to: {output_dir}/")
            try:
                infra.storage.save_experiment(evolver.experiment)
                infra.logger.info(
                    "Checkpoint saved successfully",
                    experiment_id=evolver.experiment.id,
                    num_programs=len(evolver.experiment.archive),
                    num_rollouts=len(evolver.experiment.rollout_history),
                )
                print(f"\n  Checkpoint saved to: {output_dir}/")
                print(f"    - Experiment state: experiment.json")
                print(f"    - Programs: programs/{len(evolver.experiment.archive)} files")
                print(f"    - Rollout results: results/{len(evolver.experiment.rollout_history)} files")
            except Exception as e:
                infra.logger.error(f"Failed to save checkpoint: {e}")
                print(f"  Failed to save checkpoint: {e}")

        print("\n" + "=" * 70)

    except Exception as e:
        print("\n\n" + "=" * 70)
        print("  EVOLUTION FAILED")
        print("=" * 70)

        if infra:
            infra.logger.error(
                "Evolution failed with error",
                error=str(e),
                current_iteration=evolver.experiment.current_iteration if evolver else None,
            )

        import traceback
        traceback.print_exc()

        # Try to save checkpoint even on failure
        if infra and infra.storage and evolver:
            output_dir = resolve_output_dir(infra.storage, evolver.experiment.id, config)
            infra.logger.info("Attempting to save checkpoint after failure")
            try:
                infra.storage.save_experiment(evolver.experiment)
                infra.logger.info(
                    "Checkpoint saved successfully",
                    experiment_id=evolver.experiment.id,
                )
                print(f"\n  Checkpoint saved to: {output_dir}/")
            except Exception as save_error:
                infra.logger.error(f"Failed to save checkpoint: {save_error}")
                print(f"  Failed to save checkpoint: {save_error}")

        print("\n" + "=" * 70)
        sys.exit(1)

    finally:
        # CRITICAL: Flush Langfuse observations before exit
        if infra and infra.langfuse:
            try:
                infra.logger.debug("Flushing Langfuse observations...")
                infra.langfuse.flush()
                # Wait for async flush to complete
                import time
                time.sleep(1)
                infra.logger.debug("Langfuse flush completed")
            except Exception as e:
                infra.logger.warning(f"Failed to flush Langfuse on exit: {e}")


if __name__ == "__main__":
    main()
