from concurrent.futures import ThreadPoolExecutor, Future
import math
import queue
import threading
import time
from collections import deque
from typing import Optional, Deque, List

from .base import ResultHandle, BackendExecutor, BackendTask, BatchResultHandle
from famou.core.data import Context, Rollout, RolloutResult

class ThreadWorker:
    """Thread worker implementation with thread-local env and llm_client storage"""

    def __init__(self, env_factory, llm_client=None, gpu_pool=None):
        """
        Initialize worker with env_factory and llm_client.

        Args:
            env_factory: Factory to create per-thread env instances
            llm_client: Pre-created independent LLM client for this worker.
                If None, the worker will not have an LLM client.
            gpu_pool: A shared thread-safe queue of free GPU slices (each slice
                is a list of GPU IDs, e.g. [0] or [0, 1]). A worker acquires a
                slice from the pool right before running a task and returns it
                afterwards, so at most one task runs per slice at a time and no
                card is ever double-booked while another sits idle. None means
                no GPU pooling (CPU-only run). This is shared across ALL workers
                so allocation follows real availability, not submission order.
        """
        self.env_factory = env_factory
        # Use the pre-created llm_client directly (no pickle needed)
        self.llm_client = llm_client
        self.gpu_pool = gpu_pool
        self._thread_local = threading.local()

    def _get_env(self):
        """Get or create env for current thread.

        The env is created without any static GPU pinning. Instead the shared
        gpu_pool is attached to the env so it can acquire a free card around
        each actual evaluation (see _SdkEnvWrapper.execute_with_evaluator).
        Acquiring per-evaluation — rather than per-rollout — means a card is
        never held idle while the rollout is doing LLM generation, and each
        evaluation independently lands on whichever card is free.
        """
        if not hasattr(self._thread_local, 'env'):
            env = self.env_factory.create()
            if self.gpu_pool is not None:
                setattr(env, "gpu_pool", self.gpu_pool)
            self._thread_local.env = env
        return self._thread_local.env

    def execute_rollout(self, task: BackendTask) -> RolloutResult:
        """Execute rollout or enrichment task based on task_type.

        Each thread gets its own env instance via thread-local storage.
        Uses the worker's own llm_client for LLM calls. GPU allocation happens
        inside the env, per evaluation, from the shared pool.

        Args:
            task: BackendTask to execute

        Returns:
            RolloutResult: Result of the execution
        """
        env = self._get_env()  # Thread-local env

        # Give this task its OWN module instances. Strategies hand out shared
        # rollout templates (e.g. self.explore_rollout), so without this every
        # concurrent worker would share the same module objects; RolloutEngine
        # injects env via `module.env = env`, and the shared assignment makes all
        # concurrent evaluations converge onto one env whose per-instance lock
        # then serializes them. Deep-copying here (Module.__deepcopy__ drops the
        # injected deps) mirrors the per-task isolation the Ray backend gets from
        # serialization, so evaluations actually run in parallel across GPUs.
        import copy

        if task.task_type == "enrichment":
            modules = copy.deepcopy(task.enrichment_modules) if task.enrichment_modules else task.enrichment_modules
            return task.engine.execute_enrichment(
                program=task.program,
                modules=modules,
                context=task.context,
                rollout_id=task.enrichment_rollout_id,
                env=env,
                llm_client=self.llm_client,  # Use worker's own llm_client
            )
        rollout = copy.deepcopy(task.rollout) if task.rollout is not None else task.rollout
        return task.engine.execute_rollout(
            rollout, task.context, task.iteration, env,
            attempt=task.attempt,
            llm_client=self.llm_client,  # Use worker's own llm_client
        )
    
    def cleanup(self):
        """Cleanup thread-local env if exists"""
        if hasattr(self._thread_local, 'env'):
            env = self._thread_local.env
            if hasattr(env, 'close'):
                try:
                    env.close()
                except Exception:
                    pass

    def interrupt(self):
        """Interrupt the currently running evaluator subprocess in this worker's env.

        Delegates to env.interrupt() if available (ProcessEnv path).
        Safe to call from any thread at any time.
        """
        env = getattr(self._thread_local, 'env', None)
        if env is not None and hasattr(env, 'interrupt'):
            try:
                env.interrupt()
            except Exception:
                pass


class ThreadPoolResultHandle(ResultHandle[RolloutResult]):
    """Thread pool result handle implementation"""
    
    def __init__(self, future: Future):
        self._future = future
    
    def result(self, timeout: Optional[float] = None) -> RolloutResult:
        try:
            return self._future.result(timeout)
        except Exception as e:
            raise e
    
    def done(self) -> bool:
        return self._future.done()
    
    def cancel(self) -> bool:
        return self._future.cancel()
    
    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        return self._future.exception(timeout)


@BackendExecutor.register("threadpool")
class ThreadPoolBackend(BackendExecutor):
    """Thread pool based implementation to replace ThreadPoolExecutor in _run_evolution"""
    
    def __init__(self, config):
        self.executor = None
        self.max_workers = 0
        self.completed_tasks = deque()
        self.workers = []
        self._submit_index = 0
    
    def submit(self, task: BackendTask) -> ThreadPoolResultHandle:
        # Check if thread pool is initialized
        if self.executor is None:
            raise RuntimeError("ThreadPoolBackend executor not initialized. Call start_workers() first.")
        
        # Select worker (round-robin via monotonic counter)
        worker = self.workers[self._submit_index % len(self.workers)]
        self._submit_index += 1
        
        # Submit task to thread pool
        future = self.executor.submit(worker.execute_rollout, task)
        
        return ThreadPoolResultHandle(future)
    
    def submit_batch(self, tasks: List[BackendTask]) -> BatchResultHandle:
        """Submit batch of tasks to thread pool
        
        Args:
            tasks: List of BackendTasks to execute
            
        Returns:
            BatchResultHandle: Handle to track batch progress
        """
        handles = []
        task_ids = []
        
        for task in tasks:
            handle = self.submit(task)
            handles.append(handle)
            task_ids.append(task.task_id)
        
        return BatchResultHandle(handles, task_ids)
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown thread pool and cleanup all worker resources"""
        if not wait:
            # 1. 先 interrupt 所有正在运行的 evaluator 子进程，使 worker 线程立即返回
            for worker in self.workers:
                try:
                    worker.interrupt()
                except Exception:
                    pass
            # 2. cancel_futures=True 取消所有已排队但尚未开始的任务
            self.executor.shutdown(wait=False, cancel_futures=True)
        else:
            self.executor.shutdown(wait=True)
        
        # 清理每个 worker 持有的 thread-local env 资源
        if hasattr(self, 'workers') and self.workers:
            for worker in self.workers:
                try:
                    worker.cleanup()
                except Exception as e:
                    # Log but don't raise - best effort cleanup
                    import logging
                    logging.warning(f"Failed to cleanup worker: {e}")
    
    def get_max_workers(self) -> int:
        """Get maximum number of worker threads
        
        Returns:
            int: Maximum number of workers
        """
        return self.max_workers
    
    def start_workers(self, num_workers: int, env_factory=None, llm_client=None, embedding_client=None) -> None:
        """Start specified number of worker threads

        Args:
            num_workers: Number of worker threads to start
            env_factory: Factory to create per-thread env objects
            llm_client: LLM client template. A pool of independent clients will be
                pre-created (one per worker) to avoid connection pool contention.
            embedding_client: Embedding client for feature extraction modules (currently unused for thread workers)
        """
        if num_workers <= 0:
            return

        if env_factory is None:
            raise ValueError("env_factory is required for ThreadPoolBackend")

        # Build a SHARED pool of free GPU slices from the env config. Workers
        # acquire a slice per task and return it when done, so allocation tracks
        # real availability (a card is reused only after its task finishes) and
        # concurrency is naturally capped at the number of slices. This replaces
        # the old per-worker round-robin, which pinned a card by submission
        # index and could double-book one card while another sat idle.
        gpu_slices = self._build_gpu_slices(env_factory)
        gpu_pool = None
        if gpu_slices:
            gpu_pool = queue.Queue()
            for slice_ids in gpu_slices:
                gpu_pool.put(slice_ids)
            import logging
            logging.getLogger(__name__).info(
                "[ThreadPoolBackend] GPU pool: %d slices for %d workers -> %s",
                len(gpu_slices), num_workers, gpu_slices,
            )

        # Pre-create a pool of independent LLM clients
        # This is done upfront to avoid pickle overhead in each worker
        llm_client_pool = self._create_llm_client_pool(llm_client, num_workers)

        # Create worker instances with pre-created independent llm_clients.
        # All workers share the SAME gpu_pool so cards are allocated by
        # availability rather than by which worker object was picked.
        self.workers = [
            ThreadWorker(env_factory, llm_client_pool[i], gpu_pool=gpu_pool)
            for i in range(num_workers)
        ]

        # Create new executor
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.max_workers = num_workers

    @staticmethod
    def _build_gpu_slices(env_factory) -> List[List[int]]:
        """Build the pool of GPU slices available for concurrent evaluations.

        Reads gpu_ids (available cards) and gpu_num_each_run (cards per task)
        from the env config and returns a list of non-overlapping slices; each
        slice is handed to exactly one running task at a time.

        - gpu_num_each_run >= 1: whole cards per task. gpu_ids is chunked into
          non-overlapping groups of ceil(gpu_num_each_run) cards, so
          gpu_ids=[0,1,2,3], gpu_num_each_run=1 -> [[0],[1],[2],[3]] (4-way
          concurrency), and gpu_num_each_run=2 -> [[0,1],[2,3]] (2-way).
        - 0 < gpu_num_each_run < 1: cards are shared. Each card appears
          round(1/gpu_num_each_run) times as its own single-card slice, so
          0.5 -> two tasks may share each card (2 slices per card).

        Returns an empty list when no GPU is configured (CPU-only run), in which
        case no pooling is applied.
        """
        if env_factory is None:
            return []

        get_config = getattr(env_factory, "get_config", None)
        env_config = get_config() if callable(get_config) else None
        if env_config is None:
            return []

        gpu_ids = getattr(env_config, "gpu_ids", None) or []
        gpu_ids = [int(g) for g in gpu_ids]
        if not gpu_ids:
            return []

        raw_per_run = getattr(env_config, "gpu_num_each_run", 1) or 1
        try:
            raw_per_run = float(raw_per_run)
        except (TypeError, ValueError):
            raw_per_run = 1.0

        slices: List[List[int]] = []
        if raw_per_run < 1.0:
            # Fractional card: allow multiple tasks to share each card. Emit the
            # copies round-robin across cards ([0],[1],[2],[3],[0],[1],...) — not
            # grouped per card ([0],[0],...,[1],[1],...) — so that when there are
            # fewer worker threads than slots, the first acquisitions still
            # spread across distinct cards instead of piling onto card 0.
            shares = max(1, int(round(1.0 / raw_per_run)))
            for _ in range(shares):
                for g in gpu_ids:
                    slices.append([g])
        else:
            per_run = max(1, int(math.ceil(raw_per_run)))
            n = len(gpu_ids)
            # Non-overlapping chunks; a trailing partial chunk (< per_run) is
            # dropped so every running task gets its full card count.
            for start in range(0, n - per_run + 1, per_run):
                slices.append(gpu_ids[start:start + per_run])
            if not slices:
                # per_run larger than available cards: one slice with all cards.
                slices.append(list(gpu_ids))
        return slices

    def _create_llm_client_pool(self, llm_client_template, pool_size: int) -> List:
        """Pre-create a pool of independent LLM clients.

        Uses pickle serialization/deserialization to create deep copies,
        similar to how Ray handles client copying across process boundaries.
        This ensures each worker has its own httpx connection pool.

        Args:
            llm_client_template: The LLM client to copy
            pool_size: Number of clients to create

        Returns:
            List of independent LLM clients
        """
        if llm_client_template is None:
            return [None] * pool_size

        pool = []
        for i in range(pool_size):
            try:
                import pickle
                # Serialize and deserialize to create a deep copy
                # This works because OpenAIClient implements __getstate__/__setstate__
                copied = pickle.loads(pickle.dumps(llm_client_template))
                pool.append(copied)
            except Exception as e:
                # If serialization fails, fall back to sharing
                import logging
                logging.warning(
                    f"Failed to create independent LLM client {i+1}/{pool_size}, "
                    f"falling back to shared client: {e}"
                )
                pool.append(llm_client_template)

        return pool
