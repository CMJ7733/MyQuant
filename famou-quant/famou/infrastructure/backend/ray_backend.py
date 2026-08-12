import os
import ray
import dataclasses
from collections import deque
from typing import Optional, List

from .base import ResultHandle, BackendExecutor, BackendTask, BatchResultHandle
from famou.core.data import RolloutResult


class SerializableEnvFactory:
    """
    Serializable wrapper for EnvFactory that preserves config across Ray boundaries.
    
    EnvFactory uses class variables which don't get serialized. This wrapper
    stores the config as an instance variable and reapplies it in the worker process.
    """
    def __init__(self, env_factory_class, config):
        """
        Args:
            env_factory_class: The EnvFactory class
            config: EnvConfig to be applied
        """
        self.env_factory_class = env_factory_class
        self.config = config
    
    def create(self):
        """Create env instance after setting config in the worker process."""
        if self.config is not None:
            self.env_factory_class.set_config(self.config)
        return self.env_factory_class.create()
    
    def get_config(self):
        """Get the stored config."""
        return self.config


class RayResultHandle(ResultHandle[RolloutResult]):
    """Ray result handle implementation"""

    def __init__(self, object_ref: ray.ObjectRef):
        self._object_ref = object_ref

    def result(self, timeout: Optional[float] = None) -> RolloutResult:
        try:
            return ray.get(self._object_ref, timeout=timeout)
        except Exception as e:
            raise e

    def done(self) -> bool:
        ready, _ = ray.wait([self._object_ref], timeout=0)
        return len(ready) > 0

    def cancel(self) -> bool:
        try:
            ray.cancel(self._object_ref, force=False)
            return True
        except Exception:
            return False

    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        try:
            ray.get(self._object_ref, timeout=timeout)
            return None
        except Exception as e:
            return e


@ray.remote(max_restarts=3, max_task_retries=3)
class RayWorker:
    """Ray remote worker actor — persistent, one instance per GPU slot.

    Fault tolerance:
        max_restarts=3:     Actor auto-restarts up to 3 times on node/process failure.
        max_task_retries=3: Tasks auto-retry up to 3 times when actor dies mid-execution.

    llm_client is injected once at startup and held as a resident object.
    OpenAIClient implements __getstate__/__setstate__ to handle the httpx
    connection pool (which contains unpicklable thread locks) transparently.
    
    env_factory is also injected at startup, and env is created lazily in the worker process.
    """

    def __init__(self, env_factory=None, llm_client=None, embedding_client=None):
        """
        Initialize Ray worker.
        
        Args:
            env_factory: Environment factory or class to create env instances
            llm_client: LLM client (will be serialized with __getstate__/__setstate__)
            embedding_client: Embedding client for feature extraction modules
        """
        self.env_factory = env_factory
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        # Lazily create env on first use to avoid serialization issues
        self._env = None
        if hasattr(self.llm_client, "add_buffered_request_hook"):
            self.llm_client.add_buffered_request_hook()

    @property
    def env(self):
        """Lazily create and cache env instance in the worker process."""
        if self._env is None and self.env_factory is not None:
            self._env = self.env_factory.create()
        return self._env

    def shutdown(self):
        """清理 worker 持有的资源"""
        if self._env and hasattr(self._env, 'close'):
            self._env.close()

    def execute_rollout(self, task: "BackendTask") -> RolloutResult:
        """Execute rollout or enrichment task via RolloutEngine.

        Args:
            task: BackendTask (engine field is None — stripped before submission)
                  context may be an ObjectRef if submitted via Ray

        Returns:
            RolloutResult from engine execution
        """
        # If context is an ObjectRef (from ray.put()), resolve it first
        if isinstance(task.context, ray.ObjectRef):
            task = dataclasses.replace(task, context=ray.get(task.context))
        
        if task.task_type == "enrichment":
            return self._execute_enrichment(task)
        return self._execute_rollout(task)

    def _build_engine(self):
        from famou.controller.engine import RolloutEngine
        return RolloutEngine(
            llm_client=self.llm_client,
            embedding_client=self.embedding_client,
        )

    def _execute_rollout(self, task: "BackendTask") -> RolloutResult:
        if hasattr(self.llm_client, "clear_buffered_request_entries"):
            self.llm_client.clear_buffered_request_entries()
        engine = self._build_engine()
        # self._mock_gpu_use()
        result = engine.execute_rollout(
            task.rollout, task.context, task.iteration, self.env,
            attempt=task.attempt,
            llm_client=self.llm_client,
        )
        if hasattr(self.llm_client, "consume_buffered_request_entries"):
            result.llm_request_logs = self.llm_client.consume_buffered_request_entries()
        return result

    def _execute_enrichment(self, task: "BackendTask") -> RolloutResult:
        if hasattr(self.llm_client, "clear_buffered_request_entries"):
            self.llm_client.clear_buffered_request_entries()
        engine = self._build_engine()
        # self._mock_gpu_use()
        result = engine.execute_enrichment(
            program=task.program,
            modules=task.enrichment_modules,
            context=task.context,
            rollout_id=task.enrichment_rollout_id,
            env=self.env,
            llm_client=self.llm_client,
        )
        if hasattr(self.llm_client, "consume_buffered_request_entries"):
            result.llm_request_logs = self.llm_client.consume_buffered_request_entries()
        return result

    def _mock_gpu_use(self):
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.rand(100, 100).to(device)
        print(f"Tensor on device {device}: {tensor}, torch.cuda.device_count(): {torch.cuda.device_count()}")


@BackendExecutor.register("ray")
class RayBackend(BackendExecutor):
    """Ray distributed computing framework backend implementation"""

    def __init__(self, config):
        self.ray_host = config.get("ray_host")
        self.workers: List = []
        self.max_workers = 0
        self.completed_tasks = deque()
        self._submit_index = 0

@BackendExecutor.register("ray")
class RayBackend(BackendExecutor):
    """Ray distributed computing framework backend implementation"""

    def __init__(self, config):
        self.ray_host = config.get("ray_host")
        self.workers: List = []
        self.max_workers = 0
        self.completed_tasks = deque()
        self._submit_index = 0

        # Initialize Ray connection
        if not ray.is_initialized():
            if self.ray_host and self.ray_host != "auto":
                # 通过 Ray Client proxy 连接远程集群（ray://host:port）
                # __file__: famou-v2/famou/infrastructure/backend/ray_backend.py
                # 向上3级到项目根目录 famou-v2
                project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..")
                )
                famou_src_dir = os.path.join(project_root, "famou")
                # 从 requirements.txt 读取依赖，过滤 pip 选项行（--）、可编辑安装（-e）
                # 空行和注释行（#），只保留纯包名/版本约束行
                req_path = os.path.join(project_root, "requirements.txt")
                pip_deps = []
                with open(req_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or line.startswith("--") or line.startswith("-"):
                            continue
                        pip_deps.append(line)
                print("pip_deps: ", pip_deps)
                # py_modules: 上传 famou 源码，使 worker 在序列化/反序列化阶段即可 import
                # Ray 对 py_modules 做内容哈希缓存，文件不变则不会重复上传
                # pip:        在远端集群安装 famou 的所有运行时依赖
                ray.init(
                    address=self.ray_host,
                    runtime_env={
                        "py_modules": [famou_src_dir],
                        "pip": pip_deps,
                        "env_vars": {
                            "PIP_INDEX_URL": "https://pip.baidu-int.com/simple/",
                            "PIP_TRUSTED_HOST": "pip.baidu.com"
                        }
                    }
                )
            else:
                # Ray Job 模式或本地模式：
                # ray.init() 不带参数，Ray Jobs 运行时会自动注入连接信息
                # runtime_env 由 ray job submit 时指定，无需在此重复设置
                ray.init(address="auto")

    def submit(self, task: BackendTask) -> RayResultHandle:
        """Submit a task to a pre-created worker via round-robin.

        Args:
            task: BackendTask to execute

        Returns:
            RayResultHandle: Non-blocking handle

        Raises:
            RuntimeError: If start_workers() has not been called
        """
        if not self.workers:
            raise RuntimeError("RayBackend workers not initialized. Call start_workers() first.")

        # Strip engine so Ray can serialize the task (engine holds unpicklable objects)
        task_for_worker = dataclasses.replace(task, engine=None)

        # Put large context object into Ray object store to avoid network overhead
        # Ray will cache this object and workers can access it via lightweight ObjectRef
        context_ref = ray.put(task_for_worker.context)
        
        # Replace context with its ObjectRef before remote call
        task_for_worker = dataclasses.replace(task_for_worker, context=context_ref)

        worker = self.workers[self._submit_index % len(self.workers)]
        self._submit_index += 1

        object_ref = worker.execute_rollout.remote(task_for_worker)
        return RayResultHandle(object_ref)

    def submit_batch(self, tasks: List[BackendTask]) -> BatchResultHandle:
        """Submit batch of tasks to Ray

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
        """Shutdown Ray connection

        Args:
            wait: If True, wait for pending tasks to complete (graceful);
                  If False, kill all worker actors immediately and cancel pending tasks.
        """
        if not wait and hasattr(self, 'workers') and self.workers:
            # 强制终止所有 actor，立即释放资源，不等待当前任务完成
            for worker in self.workers:
                try:
                    ray.kill(worker)
                except Exception:
                    pass
        else:
            # 优雅清理：等待 worker 完成 env 资源释放
            if hasattr(self, 'workers') and self.workers:
                shutdown_refs = []
                for worker in self.workers:
                    try:
                        shutdown_refs.append(worker.shutdown.remote())
                    except Exception:
                        pass
                # 等待所有 worker 完成清理
                if shutdown_refs:
                    try:
                        ray.get(shutdown_refs, timeout=10)
                    except Exception:
                        pass
        ray.shutdown()

    def get_max_workers(self) -> int:
        """Get maximum number of worker actors

        Returns:
            int: Maximum number of workers
        """
        return self.max_workers

    def start_workers(self, num_workers: int, env_factory=None, llm_client=None, embedding_client=None) -> None:
        """Pre-create num_workers persistent Ray worker actors.

        env_factory.create() is called in the main process where EnvFactory._config
        is already set, then the env object is passed into each remote actor.
        Resource requirements (cpu, gpu, mem) are read from EnvConfig.

        The llm_client (if provided) is passed directly to each worker at startup.
        OpenAIClient handles pickle serialization transparently via __getstate__ /
        __setstate__, so no manual config-dict extraction is needed.

        Args:
            num_workers: Number of worker actors to start
            env_factory: Factory to create env objects
            llm_client: LLM client to inject into each worker as a resident object.
            embedding_client: Embedding client to inject into each worker.
        """
        if num_workers <= 0:
            return

        # 从 env_factory 的 config 中获取资源配置
        resources = {}
        if env_factory is not None:
            env_config = env_factory.get_config()
            if env_config is not None:
                env_type = getattr(env_config, 'type', 'local').lower()

                if env_type == "ray_task":
                    # ray_task 模式：actor 仅作轻量调度器，固定 num_cpus=1，
                    # 所有资源（CPU/GPU/Mem）由 RayTaskEnv 在 Ray task 上动态申请
                    resources["num_cpus"] = 1.0
                else:
                    # 原有模式：资源绑定在 actor 全生命周期
                    cpu = getattr(env_config, 'cpu', 1.0)
                    gpu = getattr(env_config, 'gpu', 0.0)
                    mem = getattr(env_config, 'mem', None)
                    if cpu > 0:
                        resources["num_cpus"] = cpu
                    if gpu > 0:
                        resources["num_gpus"] = gpu
                    if mem:
                        resources["memory"] = self._parse_memory(mem)

        self.workers = []
        for _ in range(num_workers):
            # Wrap env_factory in a serializable wrapper that preserves config
            serializable_factory = None
            if env_factory is not None:
                config = env_factory.get_config()
                # env_factory is already a class, not an instance
                serializable_factory = SerializableEnvFactory(env_factory, config)
            
            # Pass serializable_factory instead of env_factory
            worker = RayWorker.options(**resources).remote(serializable_factory, llm_client, embedding_client)
            self.workers.append(worker)

        self._submit_index = 0
        self.max_workers = num_workers

    @staticmethod
    def _parse_memory(mem_str: str) -> int:
        """将内存字符串解析为字节数，供 Ray resource 使用

        Args:
            mem_str: 内存字符串，如 "512M", "2G", "1024K"

        Returns:
            int: 字节数
        """
        mem_str = mem_str.strip().upper()
        units = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
        for suffix, multiplier in units.items():
            if mem_str.endswith(suffix):
                return int(float(mem_str[:-1]) * multiplier)
        return int(mem_str)
