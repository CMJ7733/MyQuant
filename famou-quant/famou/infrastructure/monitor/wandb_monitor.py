"""
WandB monitoring backend for real-time experiment tracking.

This module provides a WandB-based implementation of the MonitoringBackend
interface with asynchronous logging support.
"""

import os
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from famou.infrastructure.monitor.async_logger import AsyncLogQueue
from famou.infrastructure.monitor.base import MonitoringBackend, MetricPrefix


class WandBMonitor(MonitoringBackend):
    """
    WandB-based monitoring backend with asynchronous logging.

    Features:
    - Real-time metric logging to WandB
    - Configuration and hyperparameter tracking
    - Artifact logging (code files, visualizations)
    - Asynchronous logging (non-blocking)
    - Graceful shutdown

    Usage:
        >>> monitor = WandBMonitor(
        ...     project="famou-experiments",
        ...     entity="my-team",
        ...     tags=["baseline", "test"],
        ...     notes="Testing new strategy"
        ... )
        >>> monitor.log_config({"learning_rate": 0.01, "max_iterations": 100})
        >>> monitor.log_metrics({"score": 0.95}, step=0)
        >>> monitor.finish()
    """

    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        name: Optional[str] = None,
        group: Optional[str] = None,
        async_mode: bool = True,
        num_workers: int = 2,
        queue_size: int = 100,
    ):
        """
        Initialize WandB monitor.

        Args:
            project: WandB project name
            entity: WandB entity (username or team)
            config: Configuration dictionary (logged to WandB)
            tags: List of tags for categorization
            notes: Run notes
            name: Run name (auto-generated if not provided)
            group: Group name for grouped runs
            async_mode: Enable asynchronous logging (default: True)
            num_workers: Number of async worker threads (default: 2)
            queue_size: Max queue size for async logging (default: 100)
        """
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is not installed. Install it with: pip install wandb"
            )

        self._wandb = wandb

        # Set timeout environment variables to prevent hanging
        os.environ['WANDB_INIT_TIMEOUT'] = '90'  # 90 seconds timeout for init (default)
        os.environ['WANDB_START_METHOD'] = 'thread'  # Use thread instead of fork
        os.environ['WANDB_DISABLE_GIT'] = 'true'  # Disable git collection
        os.environ['WANDB_DISABLE_CODE'] = 'true'  # Disable code collection
        os.environ['WANDB_MODE'] = 'online'  # Force online mode
        os.environ['WANDB_SILENT'] = 'false'  # Keep output

        # Suppress macOS MallocStackLogging warnings from wandb-core
        os.environ['MallocStackLogging'] = '0'
        os.environ['MallocNanoZone'] = '0'

        # Fix Unicode display issues
        if 'LC_CTYPE' in os.environ and os.environ['LC_CTYPE'] == 'C':
            os.environ['LC_CTYPE'] = 'en_US.UTF-8'
        if 'LANG' not in os.environ or os.environ['LANG'] == '':
            os.environ['LANG'] = 'en_US.UTF-8'

        # Initialize logger FIRST (before any other operations)
        self._logger = logging.getLogger(__name__)

        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags
        self.notes = notes
        self._run_name_input = name  # Store input separately from property
        self.group = group
        self.async_mode = async_mode

        # Track last logged step to prevent non-monotonic steps
        self._last_logged_step: int = -1
        self._step_lock = threading.Lock()

        # Initialize async queue if enabled
        self._async_queue: Optional[AsyncLogQueue] = None
        if async_mode:
            self._async_queue = AsyncLogQueue(
                num_workers=num_workers,
                queue_size=queue_size,
                daemon=True
            )
            self._async_queue.start()

        # Initialize WandB run with shorter timeout
        self._logger.info(f"Initializing WandB run for project: {project}...")

        try:
            self._run = self._wandb.init(
                project=project,
                entity=entity,
                config=config,
                tags=tags,
                notes=notes,
                name=name,
                group=group,
                settings=self._wandb.Settings(
                    init_timeout=90,  # 90 seconds timeout for initialization
                ),
            )
        except Exception as e:
            self._logger.warning(f"Failed to initialize WandB in online mode: {e}")
            self._logger.info("Falling back to offline mode (metrics will be saved locally)")
            self._run = self._wandb.init(
                project=project,
                entity=entity,
                config=config,
                tags=tags,
                notes=notes,
                name=name,
                group=group,
                mode="offline",  # Offline mode as fallback
            )

        self._logger.info(
            f"WandBMonitor initialized: {self._run.name}",
            project=project,
            run_id=self._run.id,
            url=self._run.url,
            async_mode=async_mode
        )

        # Print URL prominently to console
        print(f"\n{'='*70}")
        print(f"🎯 WandB Monitor Initialized!")
        print(f"{'='*70}")
        print(f"  Run URL: {self._run.url}")
        print(f"  Project: {project}")
        print(f"  Run Name: {self._run.name}")
        print(f"  Run ID: {self._run.id}")
        print(f"{'='*70}\n")

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        commit: bool = True
    ) -> None:
        """
        Log metrics to WandB.

        Args:
            metrics: Dictionary of metric names to values
            step: Current iteration/step number
            commit: Whether to commit immediately (ignored in async mode)

        Example:
            >>> monitor.log_metrics({
            ...     "performance/best_score": 0.95,
            ...     "performance/avg_score": 0.82,
            ...     "cost/total_cost_usd": 0.50,
            ... }, step=5)
        """
        # Check if step is monotonic to avoid WandB warnings
        with self._step_lock:
            if step < self._last_logged_step:
                self._logger.debug(
                    f"Skipping metrics for step {step} (already logged step {self._last_logged_step})"
                )
                return
            self._last_logged_step = step

        if self.async_mode and self._async_queue:
            # Asynchronous logging - capture step in closure
            task = lambda s=step: self._run.log(metrics, step=s, commit=commit)
            success = self._async_queue.put(task, blocking=False)

            if not success:
                self._logger.warning(
                    f"Failed to queue metrics for step {step} (queue full)"
                )
        else:
            # Synchronous logging
            self._run.log(metrics, step=step, commit=commit)

    def log_config(
        self,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Log experiment configuration to WandB.

        Args:
            config: Configuration dictionary
            tags: Optional list of tags to update

        Note:
            Config is typically logged during initialization.
            This method can be used to update config or tags mid-run.

        Example:
            >>> monitor.log_config({
            ...     "strategy": "creative_exploration",
            ...     "max_iterations": 10,
            ...     "temperature": 0.7,
            ... })
        """
        # Update WandB config
        self._run.config.update(config)

        # Update tags if provided
        if tags:
            # WandB doesn't support updating tags after init
            # So we just log them as a metric
            self.log_metrics({"config/tags": ",".join(tags)}, step=0)

        self._logger.debug(f"Config updated: {len(config)} keys")

    def log_artifact(
        self,
        path: str,
        name: str,
        type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an artifact (file) to WandB.

        Args:
            path: Path to the file to log
            name: Human-readable name
            type: Artifact type (e.g., "code", "visualization")
            metadata: Optional metadata dictionary

        Example:
            >>> monitor.log_artifact(
            ...     "best_program_iter_5.py",
            ...     "Best Program - Iteration 5",
            ...     "code",
            ...     metadata={"score": 0.95, "generation": 3}
            ... )
        """
        def _log_artifact():
            artifact = self._wandb.Artifact(name, type=type)
            artifact.add_file(path)

            if metadata:
                artifact.metadata = metadata

            self._run.log_artifact(artifact)

        if self.async_mode and self._async_queue:
            success = self._async_queue.put(_log_artifact, blocking=False)
            if not success:
                self._logger.warning(f"Failed to queue artifact: {name}")
        else:
            _log_artifact()

        self._logger.debug(f"Artifact logged: {name}")

    def log_lineage_tree(
        self,
        tree_data: Dict[str, Any],
        island_id: Optional[int] = None
    ) -> None:
        """
        Log lineage tree visualization to WandB.

        Args:
            tree_data: Tree structure data with nodes and edges
            island_id: Optional island ID for multi-island experiments

        Example:
            >>> monitor.log_lineage_tree({
            ...     "nodes": [
            ...         {"id": "prog1", "parent": None, "score": 0.8},
            ...         {"id": "prog2", "parent": "prog1", "score": 0.85}
            ...     ],
            ...     "edges": [("prog1", "prog2")]
            ... }, island_id=0)
        """
        def _log_lineage_tree():
            try:
                import plotly.graph_objects as go

                # Create tree visualization
                fig = self._create_lineage_tree_figure(tree_data, island_id)

                # Log to WandB
                plot_name = f"lineage_tree_island_{island_id}" if island_id is not None else "lineage_tree"
                self._run.log({plot_name: self._wandb.Plotly(fig)})

            except ImportError:
                self._logger.warning("plotly not installed, skipping lineage tree visualization")
            except Exception as e:
                self._logger.error(f"Failed to create lineage tree: {e}")

        if self.async_mode and self._async_queue:
            success = self._async_queue.put(_log_lineage_tree, blocking=False)
            if not success:
                self._logger.warning("Failed to queue lineage tree logging")
        else:
            _log_lineage_tree()

    def log_table(
        self,
        data: Dict[str, List[Any]],
        name: str
    ) -> None:
        """
        Log a table to WandB.

        Args:
            data: Dictionary with column names as keys and lists as values
            name: Table name

        Example:
            >>> monitor.log_table({
            ...     "ID": ["prog1", "prog2"],
            ...     "Score": [0.85, 0.92],
            ...     "Generation": [1, 2]
            ... }, "top_programs")
        """
        def _log_table():
            table = self._wandb.Table(data=data or {})
            self._run.log({name: table})

        if self.async_mode and self._async_queue:
            success = self._async_queue.put(_log_table, blocking=False)
            if not success:
                self._logger.warning(f"Failed to queue table: {name}")
        else:
            _log_table()

    def finish(self) -> None:
        """
        Finish monitoring and cleanup resources.

        Ensures all pending logs are flushed before closing WandB run.
        """
        self._logger.info("Finishing WandB monitor...")

        # Wait for async queue to empty
        if self.async_mode and self._async_queue:
            self._logger.info(
                f"Waiting for async queue to empty: "
                f"{self._async_queue.current_queue_size} tasks remaining"
            )

            # Wait for all tasks to complete (increased timeout)
            emptied = self._async_queue.wait_until_empty(timeout=60.0)

            if not emptied:
                self._logger.warning(
                    f"Timeout waiting for queue: "
                    f"{self._async_queue.current_queue_size} tasks remaining"
                )

            # Shutdown async queue IMMEDIATELY to stop new tasks
            self._async_queue.shutdown()

            self._logger.info(
                f"Async queue shutdown: "
                f"completed={self._async_queue.completed_tasks}, "
                f"failed={self._async_queue.failed_tasks}"
            )

        # Finish WandB run with timeout to prevent hanging
        self._logger.info("Finishing WandB run (this may take a moment)...")

        try:
            # Use a separate thread with timeout to prevent indefinite hanging
            finish_thread = threading.Thread(target=self._run.finish, kwargs={"exit_code": 0})
            finish_thread.daemon = True
            finish_thread.start()

            # Wait with shorter timeout (15s instead of 30s)
            finish_thread.join(timeout=15.0)

            if finish_thread.is_alive():
                self._logger.warning(
                    "WandB finish() timed out after 15s. "
                    "Forcing exit. Your metrics have been saved to WandB."
                )
                # Force exit without waiting for wandb-core processes
            else:
                self._logger.info(
                    f"WandB monitor finished: {self._run.name}",
                    url=self._run.url
                )
        except Exception as e:
            self._logger.error(f"Error finishing WandB run: {e}", exc_info=True)

        # Additional cleanup: try to terminate wandb-core processes
        try:
            import subprocess
            import signal
            # Find wandb-core processes
            result = subprocess.run(
                ["pgrep", "-f", "wandb-core"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                if pids and pids[0]:
                    self._logger.info(f"Terminating {len(pids)} wandb-core processes...")
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except (ProcessLookupError, ValueError):
                            pass
        except Exception as e:
            # Silently ignore cleanup errors
            pass

    def is_async(self) -> bool:
        """Check if this backend uses async logging."""
        return self.async_mode

    def _create_lineage_tree_figure(
        self,
        tree_data: Dict[str, Any],
        island_id: Optional[int] = None
    ):
        """
        Create Plotly figure for lineage tree visualization.

        Args:
            tree_data: Dictionary with 'nodes' and 'edges'
            island_id: Optional island ID

        Returns:
            Plotly Figure object
        """
        import plotly.graph_objects as go

        nodes = tree_data.get("nodes", [])
        edges = tree_data.get("edges", [])

        if not nodes:
            # Create empty figure
            fig = go.Figure()
            fig.update_layout(
                title=f"Lineage Tree" + (f" (Island {island_id})" if island_id is not None else ""),
                showlegend=False
            )
            return fig

        # Build node lists
        node_ids = [node.get("id", f"node_{i}") for i, node in enumerate(nodes)]
        node_parents = [node.get("parent") for node in nodes]
        node_scores = [node.get("score", 0.0) for node in nodes]

        # Create treemap
        fig = go.Figure(go.Treemap(
            labels=node_ids,
            parents=node_parents,
            values=[max(score, 0.01) for score in node_scores],  # Ensure positive values
            text=[
                f"ID: {nid}<br>Score: {score:.4f}"
                for nid, score in zip(node_ids, node_scores)
            ],
            textposition="middle center",
            marker=dict(
                colorscale="Viridis",
                cmid=0.0,
                colorbar=dict(title="Score"),
            ),
            hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
        ))

        fig.update_layout(
            title=f"Lineage Tree" + (f" (Island {island_id})" if island_id is not None else ""),
            margin=dict(t=50, l=0, r=0, b=0),
        )

        return fig

    @property
    def run_id(self) -> str:
        """Get WandB run ID."""
        return self._run.id

    @property
    def run_name(self) -> str:
        """Get WandB run name."""
        return self._run.name

    @property
    def run_url(self) -> str:
        """Get WandB run URL."""
        return self._run.url

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"WandBMonitor("
            f"project={self.project}, "
            f"run={self._run.name}, "
            f"async={self.async_mode}"
            f")"
        )
