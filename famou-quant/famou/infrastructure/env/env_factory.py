"""
Environment Factory for creating famou_sdk execution environments.

Provides a factory to create ProcessEnv and ThreadEnv instances from EnvConfig.
"""

import asyncio
import inspect
import logging
import os
import tempfile
import threading
import time
from typing import Dict, Any, List, Optional

from famou.infrastructure.env.evaluator_env import ProcessEnv as SdkProcessEnv
from famou.infrastructure.env.evaluator_env import ThreadEnv as SdkThreadEnv
from famou.infrastructure.env.evaluator_env import ProcessConfig

from famou.config.settings import EnvConfig
from famou.infrastructure.env.base import ExecutionEnvironment
from famou.utils.package_manager import install_packages


logger = logging.getLogger(__name__)


class _EvaluatorWrapper:
    """
    Serializable wrapper for user evaluator functions.
    
    Converts SDK's code-string interface to famou's file-path interface.
    Uses cloudpickle to serialize evaluate_fn so that functions defined in
    scripts or interactive sessions can be passed to subprocess workers.
    """
    
    def __init__(
        self,
        evaluate_fn,
        ext: str,
        input_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize evaluator wrapper.
        
        Args:
            evaluate_fn: User evaluator function(program_path) -> results_dict
            ext: File extension (e.g., ".py")
            input_params: Extra keyword arguments forwarded to the evaluator
        """
        import cloudpickle
        self._evaluate_fn_bytes = cloudpickle.dumps(evaluate_fn)
        self.ext = ext
        self.input_params = dict(input_params or {})
        # 在父进程中记录 evaluator 所在目录，子进程执行时切换工作目录
        src_file = None
        # 优先使用 evaluator_path 属性（如 DeferredEvaluator）
        evaluator_path_attr = getattr(evaluate_fn, 'evaluator_path', None)
        if evaluator_path_attr:
            src_file = str(evaluator_path_attr)
        else:
            # fallback: 通过 inspect 获取函数/类定义所在文件
            for candidate in [
                evaluate_fn,
                getattr(evaluate_fn, '__func__', None),
                type(evaluate_fn),
            ]:
                if candidate is None:
                    continue
                try:
                    src_file = inspect.getfile(candidate)
                    break
                except (TypeError, OSError):
                    continue
            # 最后兜底：从 __code__.co_filename 读取
            if src_file is None:
                code = getattr(evaluate_fn, '__code__', None) or getattr(
                    getattr(evaluate_fn, '__func__', None), '__code__', None
                )
                if code:
                    src_file = code.co_filename
        self.source_dir = os.path.dirname(os.path.abspath(src_file)) if src_file else None
    
    def __call__(self, code_str: str) -> Dict[str, Any]:
        """
        Wrap evaluator to accept code string and write to temp file.
        
        Args:
            code_str: Source code string
            
        Returns:
            Dict with evaluation results
        """
        import cloudpickle
        evaluate_fn = cloudpickle.loads(self._evaluate_fn_bytes)

        if self.source_dir:
            os.chdir(self.source_dir)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=self.ext, delete=False, encoding="utf-8"
        ) as f:
            f.write(code_str)
            temp_path = f.name
        
        try:
            return self._call_evaluator(evaluate_fn, temp_path)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def _call_evaluator(self, evaluate_fn, temp_path: str) -> Dict[str, Any]:
        """Invoke evaluator with the configured extra input params when supported."""
        if not self.input_params:
            return evaluate_fn(temp_path)

        signature = inspect.signature(evaluate_fn)
        parameters = signature.parameters.values()
        accepts_var_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if accepts_var_kwargs:
            return evaluate_fn(temp_path, **self.input_params)

        accepted_kwargs = {
            name: value
            for name, value in self.input_params.items()
            if name in signature.parameters
        }
        return evaluate_fn(temp_path, **accepted_kwargs)


class _SdkEnvWrapper:
    """
    Wrapper to adapt famou_sdk environments to famou's sync interface.
    
    Handles:
    - Async to sync conversion
    - Code string to file path conversion (for evaluator functions)
    - Thread-safe serialization of execution calls
    """
    
    def __init__(
        self,
        sdk_env,
        default_timeout: Optional[float] = None,
        input_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize wrapper.
        
        Args:
            sdk_env: famou_sdk environment (ProcessEnv or ThreadEnv)
            default_timeout: Default evaluator execution timeout from infrastructure.env.timeout
            input_params: Extra keyword arguments forwarded to evaluator functions
        """
        self._sdk_env = sdk_env
        self.default_timeout = default_timeout
        self.input_params = dict(input_params or {})
        self._lock = threading.Lock()  # Serialize access to prevent concurrent execution
        # Optional shared GPU pool (queue.Queue of GPU-id slices). When set by
        # the backend, one free card slice is acquired around each evaluation
        # and returned afterwards, so at most one evaluation runs per card and
        # cards are never held idle during LLM generation.
        self.gpu_pool = None

    def _runtime_prepare_dependencies(self, packages: List[str]) -> Dict[str, Any]:
        """Prepare packages inside the process that will execute the evaluator."""
        normalized = [pkg for pkg in dict.fromkeys(packages) if pkg]
        if not normalized:
            return {
                "mode": "runtime",
                "packages": [],
                "success": True,
                "prepared": False,
            }

        uv_enabled = bool(
            hasattr(self._sdk_env, "config")
            and getattr(self._sdk_env.config, "uv_enabled", False)
        )
        started_at = time.time()

        if uv_enabled:
            def _noop_evaluator(_: str) -> Dict[str, Any]:
                return {"combined_score": 0.0, "validity": 1.0}

            try:
                self._run_async(
                    program_code="pass\n",
                    evaluator_fn=_noop_evaluator,
                    timeout=None,
                    deps={"packages": normalized},
                )
                return {
                    "mode": "runtime_uv_env",
                    "packages": normalized,
                    "success": True,
                    "prepared": True,
                    "started_at": started_at,
                    "completed_at": time.time(),
                }
            except Exception as exc:
                return {
                    "mode": "runtime_uv_env",
                    "packages": normalized,
                    "success": False,
                    "prepared": False,
                    "started_at": started_at,
                    "completed_at": time.time(),
                    "error": str(exc),
                }

        success = install_packages(normalized, quiet=True)
        return {
            "mode": "runtime_current_env",
            "packages": normalized,
            "success": success,
            "prepared": success,
            "started_at": started_at,
            "completed_at": time.time(),
        }
    
    def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn,
        language: str = "python",
        extension: str = None,
        timeout: Optional[float] = None,
        required_packages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute program using SDK environment (sync interface).
        
        Args:
            program_code: Source code to execute
            evaluate_fn: User evaluator function(program_path) -> results_dict
            language: Programming language
            extension: File extension (auto-detected if None)
            timeout: Execution timeout in seconds
            required_packages: Python packages for dependency isolation
            
        Returns:
            Dict with evaluation results including combined_score and validity
        """
        evaluator_required_packages = list(getattr(evaluate_fn, "required_packages", []) or [])
        merged_packages = list(
            dict.fromkeys(
                [pkg for pkg in evaluator_required_packages + list(required_packages or []) if pkg]
            )
        )

        # Extension mapping
        extension_map = {
            "python": ".py", "javascript": ".js", "java": ".java",
            "cpp": ".cpp", "c": ".c", "rust": ".rs", "go": ".go",
        }
        ext = extension or extension_map.get(language.lower(), ".txt")

        # Acquire a free GPU card slice for the duration of THIS evaluation only.
        # get() blocks until a card is free, so the number of concurrent
        # evaluations is capped at the number of slices and no card is ever
        # double-booked. The slice is injected as evaluate_fn(..., gpu_ids=...)
        # and returned to the pool in the finally block. Acquiring here (not at
        # the rollout level) keeps cards free during LLM generation.
        gpu_slice = None
        pool = getattr(self, "gpu_pool", None)
        if pool is not None:
            gpu_slice = pool.get()

        try:
            if gpu_slice is not None:
                self.input_params = dict(self.input_params or {})
                self.input_params["gpu_ids"] = list(gpu_slice)

            # Create serializable evaluator wrapper
            evaluator_wrapper = _EvaluatorWrapper(
                evaluate_fn=evaluate_fn,
                ext=ext,
                input_params=self.input_params,
            )

            uv_enabled = bool(
                merged_packages
                and hasattr(self._sdk_env, "config")
                and getattr(self._sdk_env.config, "uv_enabled", False)
            )

            # Prepare dependencies in the actual execution process. This keeps
            # evaluator packages and generated-program packages on one code path.
            dependency_trace = None
            if merged_packages and not uv_enabled:
                dependency_trace = self._runtime_prepare_dependencies(merged_packages)
            deps = None
            if uv_enabled:
                deps = {"packages": merged_packages}
            elif merged_packages and dependency_trace is not None and not dependency_trace.get("success", False):
                logger.warning(
                    "Runtime package preparation failed; continuing with current environment",
                    extra={
                        "packages": merged_packages,
                        "install_mode": dependency_trace.get("mode"),
                        "error": dependency_trace.get("error", "package install failed"),
                    },
                )
            resolved_timeout = self.default_timeout if timeout is None else timeout
            return self._run_async(program_code, evaluator_wrapper, resolved_timeout, deps)
        finally:
            if gpu_slice is not None:
                pool.put(gpu_slice)

    def prepare_dependencies(
        self,
        required_packages: List[str],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Prepare dependency environment eagerly so later evaluator loads can reuse it."""
        packages = [pkg for pkg in dict.fromkeys(required_packages) if pkg]
        if not packages:
            return {
                "mode": "runtime",
                "packages": [],
                "success": True,
                "prepared": False,
            }
        return self._runtime_prepare_dependencies(packages)
    
    def _run_async(
        self, 
        program_code: str, 
        evaluator_fn, 
        timeout: Optional[float],
        deps: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Run async SDK environment in sync context with thread-safe serialization."""
        # Use lock to serialize access and prevent concurrent execution on same instance
        with self._lock:
            async def _execute():
                # Use async with for each execution to ensure proper setup/teardown
                async with self._sdk_env as env:
                    exec_result = await env.execute_with_evaluator(
                        program_code=program_code,
                        evaluate_fn=evaluator_fn,
                        timeout=timeout,
                        deps=deps,
                    )

                    if exec_result.success:
                        return exec_result.result
                    elif exec_result.timeout:
                        raise TimeoutError(f"Execution exceeded timeout of {timeout}s")
                    else:
                        raise RuntimeError(f"Execution failed: {exec_result.error}")
            
            # Always use asyncio.run() to ensure clean event loop
            return asyncio.run(_execute())

    def close(self) -> None:
        """关闭并清理底层 SDK 环境资源（包括 uv 缓存）"""
        with self._lock:
            if hasattr(self._sdk_env, 'close'):
                asyncio.run(self._sdk_env.close())

    def interrupt(self) -> None:
        """立即中断当前正在运行的 evaluator 任务（委托给底层 SDK 环境）。

        ProcessEnv：kill 子进程及整个进程组；ThreadEnv：无操作。
        可从任意线程安全调用，无需 asyncio 和 self._lock。
        """
        self._sdk_env.interrupt()


class EnvFactory:
    """
    Factory for creating famou_sdk execution environments.
    
    Usage:
        1. Set config first: EnvFactory.set_config(config)
        2. Create environment: env = EnvFactory.create()
    
    Example:
        config = EnvConfig(type="process", timeout=30.0, use_cgroup=True)
        EnvFactory.set_config(config)
        env = EnvFactory.create()
        result = env.execute_with_evaluator(...)
    """
    
    _config: Optional[EnvConfig] = None
    
    @classmethod
    def set_config(cls, config: EnvConfig) -> None:
        """
        Set environment configuration.
        
        Args:
            config: Environment configuration
            
        Example:
            >>> config = EnvConfig(type="process", timeout=30.0)
            >>> EnvFactory.set_config(config)
        """
        cls._config = config
    
    @classmethod
    def get_config(cls) -> Optional[EnvConfig]:
        """
        Get the current environment configuration.
        
        Returns:
            Optional[EnvConfig]: Current configuration or None if not set
            
        Example:
            >>> config = EnvConfig(type="process", timeout=30.0)
            >>> EnvFactory.set_config(config)
            >>> current_config = EnvFactory.get_config()
        """
        return cls._config
    
    @classmethod
    def create(cls) -> ExecutionEnvironment:
        """
        Create execution environment from previously set config.

        Directly creates famou_sdk environments (ProcessEnv or ThreadEnv).

        Returns:
            ExecutionEnvironment instance (wrapped famou_sdk env)

        Raises:
            ValueError: If config not set or environment type is not supported
            
        Example:
            >>> config = EnvConfig(type="process", timeout=30.0)
            >>> EnvFactory.set_config(config)
            >>> env = EnvFactory.create()
            >>> result = env.execute_with_evaluator(code, evaluator)
        """
        if cls._config is None:
            raise ValueError(
                "Config not set. Call EnvFactory.set_config(config) first."
            )
        
        return cls._create_from_config(cls._config)

    @classmethod
    def _create_from_config(cls, config: EnvConfig) -> ExecutionEnvironment:
        """
        Stateless creation logic — no dependency on class variables.

        Safe to call from Ray remote tasks where concurrent modification of
        cls._config would cause race conditions.

        Args:
            config: Environment configuration

        Returns:
            ExecutionEnvironment instance

        Raises:
            ValueError: If environment type is not supported
        """
        env_type = config.type.lower()
        
        if env_type == "thread":
            max_workers = getattr(config, 'max_workers', 1)
            sdk_env = SdkThreadEnv(max_workers=max_workers)
            return _SdkEnvWrapper(
                sdk_env,
                default_timeout=config.timeout,
                input_params=config.input_params,
            )
            
        elif env_type in ("local", "process"):
            process_config = ProcessConfig(
                use_cgroup=getattr(config, 'use_cgroup', False),
                memory_mb=getattr(config, 'memory_mb', None),
                cpu_percent=getattr(config, 'cpu_percent', None),
                uv_enabled=getattr(config, 'uv_enabled', False),
            )
            
            sdk_env = SdkProcessEnv(process_config)
            return _SdkEnvWrapper(
                sdk_env,
                default_timeout=config.timeout,
                input_params=config.input_params,
            )

        elif env_type == "ray_task":
            from famou.infrastructure.env.ray_task_env import RayTaskEnv
            return RayTaskEnv(config)
            
        else:
            raise ValueError(
                f"Unknown environment type: {config.type}. "
                f"Supported types: local, process, thread, ray_task"
            )
