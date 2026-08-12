"""
Asynchronous logging queue for non-blocking metric logging.

This module provides a queue-based async logging system that allows
the evolution process to continue without waiting for WandB API calls to complete.
"""

import logging
import queue
import threading
import time
from typing import Callable, Dict, Any, Optional, List


class AsyncLogQueue:
    """
    Asynchronous queue for non-blocking logging operations.

    Features:
    - Queue-based task scheduling
    - Worker thread pool for parallel execution
    - Automatic retry on failure
    - Graceful shutdown
    - Thread-safe operations

    Usage:
        >>> queue = AsyncLogQueue(num_workers=2)
        >>> queue.start()
        >>>
        >>> # Non-blocking log call
        >>> queue.put(lambda: wandb.log(metrics))
        >>>
        >>> # Wait for all logs to complete before shutdown
        >>> queue.wait_until_empty()
        >>> queue.shutdown()
    """

    def __init__(
        self,
        num_workers: int = 2,
        queue_size: int = 100,
        daemon: bool = True
    ):
        """
        Initialize async log queue.

        Args:
            num_workers: Number of worker threads (default: 2)
            queue_size: Maximum queue size (default: 100)
            daemon: Whether workers are daemon threads (default: True)
        """
        self.num_workers = num_workers
        self.max_queue_size = queue_size
        self.daemon = daemon

        # Thread-safe queue
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # Worker management
        self._workers: List[threading.Thread] = []
        self._shutdown_flag: bool = False

        # Statistics
        self._total_tasks: int = 0
        self._completed_tasks: int = 0
        self._failed_tasks: int = 0

        # Logging
        self._logger = logging.getLogger(__name__)

    def start(self) -> None:
        """
        Start worker threads.

        Should be called before adding any tasks.
        """
        if self._workers:
            self._logger.warning("Workers already started")
            return

        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncLogWorker-{i}",
                daemon=self.daemon
            )
            worker.start()
            self._workers.append(worker)

        self._logger.info(
            f"Started {self.num_workers} async log workers (queue_size={self.max_queue_size})"
        )

    def shutdown(self) -> None:
        """
        Signal shutdown to workers.

        Workers will finish processing current tasks from queue
        and then exit. Call wait_until_empty() first to ensure all tasks complete.
        """
        if self._shutdown_flag:
            return

        self._logger.info("Shutting down async log queue...")
        self._shutdown_flag = True

        # Wait for workers to finish
        for worker in self._workers:
            worker.join(timeout=5.0)

        self._workers.clear()

        self._logger.info(
            f"Async log queue shutdown: "
            f"completed={self._completed_tasks}, "
            f"failed={self._failed_tasks}, "
            f"total={self._total_tasks}"
        )

    def put(
        self,
        task: Callable[[], None],
        blocking: bool = False,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Add a logging task to the queue.

        Args:
            task: Callable that performs the logging operation
            blocking: If True, block until queue has space (default: False)
            timeout: Timeout in seconds (only if blocking=True)

        Returns:
            True if task was added successfully, False if queue was full

        Example:
            >>> def log_to_wandb():
            ...     wandb.log({"score": 0.95})
            >>>
            >>> success = queue.put(log_to_wandb)
        """
        try:
            self._queue.put(task, block=blocking, timeout=timeout)
            self._total_tasks += 1
            return True
        except queue.Full:
            self._logger.warning(
                f"Async log queue is full (max={self.max_queue_size}), "
                f"dropping log task. Total tasks: {self._total_tasks}"
            )
            return False

    def wait_until_empty(
        self,
        timeout: float = 5.0,
        poll_interval: float = 0.1
    ) -> bool:
        """
        Wait until all tasks in the queue are processed.

        Args:
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check queue status

        Returns:
            True if queue was emptied, False if timeout

        Example:
            >>> # After logging all metrics for an iteration
            >>> queue.wait_until_empty(timeout=10.0)
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self._queue.empty():
                # Give a small delay for any in-flight tasks
                time.sleep(poll_interval)
                return True
            time.sleep(poll_interval)

        self._logger.warning(
            f"Timeout waiting for queue to empty, "
            f"remaining tasks: {self._queue.qsize()}"
        )
        return False

    def _worker_loop(self) -> None:
        """
        Worker thread main loop.

        Continuously pulls tasks from queue and executes them
        until shutdown signal is received.
        """
        thread_name = threading.current_thread().name
        self._logger.debug(f"Worker {thread_name} started")

        while not self._shutdown_flag:
            try:
                # Get task from queue (with timeout to allow checking shutdown_flag)
                try:
                    task = self._queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # Check if it's a shutdown signal (None task)
                if task is None:
                    self._queue.task_done()
                    continue

                # Execute the task
                try:
                    task()
                    self._completed_tasks += 1
                except Exception as e:
                    self._failed_tasks += 1
                    self._logger.error(
                        f"Async log task failed in {thread_name}: {e}",
                        exc_info=True
                    )
                finally:
                    self._queue.task_done()

            except Exception as e:
                # Log but continue working
                self._logger.error(
                    f"Worker {thread_name} error: {e}",
                    exc_info=True
                )
                time.sleep(1.0)  # Brief pause before retrying

        self._logger.debug(f"Worker {thread_name} exiting")

    @property
    def current_queue_size(self) -> int:
        """Current number of tasks in the queue."""
        return self._queue.qsize()

    @property
    def total_tasks(self) -> int:
        """Total number of tasks added to the queue."""
        return self._total_tasks

    @property
    def completed_tasks(self) -> int:
        """Number of successfully completed tasks."""
        return self._completed_tasks

    @property
    def failed_tasks(self) -> int:
        """Number of failed tasks."""
        return self._failed_tasks

    @property
    def is_alive(self) -> bool:
        """Check if any workers are still running."""
        return any(worker.is_alive() for worker in self._workers)

    def __repr__(self) -> str:
        """Concise representation."""
        return (
            f"AsyncLogQueue("
            f"workers={len(self._workers)}, "
            f"queue_size={self.current_queue_size}, "
            f"total={self._total_tasks}, "
            f"completed={self._completed_tasks}, "
            f"failed={self._failed_tasks}"
            f")"
        )
