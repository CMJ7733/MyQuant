# Copyright 2026 Baidu ACG FM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
进程执行环境模块

提供基于独立进程的代码执行环境实现:
- ProcessEnv: 在独立子进程中执行预加载的 evaluator 函数，支持进程组管理和 cgroup 资源监控

执行流程:
1. 父进程创建 cgroup（setup）并开始采样
2. 父进程创建管道和子进程
3. 子进程启动后:
   - 设置新进程组
   - 将自己 attach 到 cgroup（复用 Manager）
   - 执行 evaluate_fn
4. 父进程等待结果，持续进行资源采样
"""
import asyncio
import multiprocessing
import os
import signal
import time
import traceback
import uuid
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import psutil

from .env_protocol import BaseEnv, DepsSpec, EvaluateFnType, EvaluatorSpec, ExecutionResult
from .logger import get_logger
from .resource.base import ResourceLimits, is_linux
from .resource.manager import CgroupNotAvailableError, Manager
from .utils.uv_manager import DependencyInstallError, UvConfig, UvManager, VenvInfo

logger = get_logger("evaluator_env.process_env")


@dataclass
class ProcessConfig:
    """
    进程环境配置

    Attributes:
        use_cgroup: 是否使用 cgroup 进行资源监控和限制
        memory_mb: 内存限制（MB），仅在 use_cgroup=True 时生效
        cpu_percent: CPU 限制（百分比），仅在 use_cgroup=True 时生效
        sample_interval: 资源采样间隔（秒）
        uv_enabled: 是否启用 uv 进行依赖隔离
        uv_cache_dir: uv 缓存目录，None 使用默认
        uv_venv_base: 虚拟环境存储目录，None 使用临时目录
        uv_python_preference: Python 查找策略
        uv_timeout: uv 操作超时时间（秒）
    """

    use_cgroup: bool = False
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    sample_interval: float = 0.1

    # uv 配置
    uv_enabled: bool = False
    uv_cache_dir: Optional[str] = None
    uv_venv_base: Optional[str] = None
    uv_python_preference: str = "managed"
    uv_timeout: float = 300.0


def _load_evaluator_from_spec(spec: EvaluatorSpec) -> Callable[[str], Dict[str, Any]]:
    """
    从 EvaluatorSpec 动态加载 evaluator

    支持四种加载模式:
        1. 只传 filepath: 加载文件中的 evaluate 函数执行
        2. filepath + method: 加载文件中的 method 函数执行
        3. filepath + classname: 加载文件中 class 的 evaluate 方法执行
        4. filepath + classname + method: 加载文件中 class 的 method 方法执行

    Args:
        spec: 包含 filepath、可选 classname 和可选 method 的字典

    Returns:
        可调用的 evaluator 函数或方法

    Raises:
        ImportError: 当无法加载模块时
        AttributeError: 当函数、类或方法不存在时
    """
    import importlib.util

    filepath = spec["filepath"]
    classname = spec.get("classname")
    method_name = spec.get("method", "evaluate")

    # 动态加载模块
    spec_loader = importlib.util.spec_from_file_location("_evaluator_module", filepath)
    if spec_loader is None or spec_loader.loader is None:
        raise ImportError(f"Cannot load module from {filepath}")

    module = importlib.util.module_from_spec(spec_loader)
    spec_loader.loader.exec_module(module)

    if classname:
        # 场景 3 和 4: 有 classname，加载类的方法
        evaluator_class = getattr(module, classname)
        instance = evaluator_class()
        method = getattr(instance, method_name)
        return method
    else:
        # 场景 1 和 2: 无 classname，直接加载模块级函数
        func = getattr(module, method_name)
        return func


def _child_process_wrapper(
    result_conn: Connection,
    cgroup_name: Optional[str],
    evaluate_fn: EvaluateFnType,
    program_code: str,
    venv_info: Optional[Dict[str, str]] = None,
) -> None:
    """
    子进程包装函数

    执行流程:
    1. 设置新进程组
    2. 将自己 attach 到 cgroup
    3. 激活虚拟环境（如果有）
    4. 加载 evaluator（如果是 EvaluatorSpec）
    5. 执行 evaluate_fn(program_code)
    6. 将结果通过管道发送回父进程

    Args:
        result_conn: 结果管道，用于发送执行结果
        cgroup_name: cgroup 名称，None 表示不使用 cgroup
        evaluate_fn: 评测函数，可以是 Callable 或 EvaluatorSpec
        program_code: 待执行的代码
        venv_info: 虚拟环境信息，用于激活虚拟环境
    """
    # 1. 设置新进程组
    try:
        os.setpgrp()
    except Exception:
        pass

    # 2. attach 到 cgroup（复用 Manager）
    if cgroup_name is not None:
        try:
            # 创建 Manager 使用相同名称，不需要 setup（父进程已创建）
            manager = Manager(name=cgroup_name)
            manager.attach_pid(os.getpid())
        except Exception:
            # attach 失败不影响执行
            pass

    # 3. 激活虚拟环境（如果有）
    if venv_info:
        try:
            # 将 dict 转换为 VenvInfo 对象
            venv_obj = VenvInfo(
                path=venv_info["path"],
                python_path=venv_info["python_path"],
                deps_hash=venv_info.get("deps_hash", ""),
                python_version=venv_info.get("python_version", ""),
            )
            # 直接修改 sys.path，并清理 sys.modules
            UvManager.activate_venv_sys_path(venv_obj)
        except Exception as e:
            result_conn.send(
                {
                    "success": False,
                    "result": None,
                    "error": f"Failed to activate venv: {e}",
                }
            )
            result_conn.close()
            return

    # 4. 加载 evaluator
    try:
        if isinstance(evaluate_fn, dict):
            # EvaluatorSpec: 从文件动态加载
            evaluator = _load_evaluator_from_spec(evaluate_fn)
        else:
            # Callable: 直接使用
            evaluator = evaluate_fn
    except Exception as e:
        result_conn.send(
            {
                "success": False,
                "result": None,
                "error": f"Failed to load evaluator: {e}\n{traceback.format_exc()}",
            }
        )
        result_conn.close()
        return

    # 5. 执行评测函数
    try:
        result = evaluator(program_code)
        result_conn.send({"success": True, "result": result, "error": None})
    except Exception as e:
        result_conn.send(
            {
                "success": False,
                "result": None,
                "error": f"{e}\n{traceback.format_exc()}",
            }
        )
    finally:
        result_conn.close()


class ProcessEnv(BaseEnv):
    """
    基于独立进程的代码执行环境

    在独立子进程中执行预加载的 evaluator 函数，支持:
    - 进程组管理：子进程创建新进程组，超时时杀掉整个进程组
    - cgroup 资源监控：可选的内存和 CPU 监控与限制
    - 强制终止：超时后可以强杀进程组，不会留下孤儿进程
    - uv 依赖隔离：可选的虚拟环境和依赖包隔离

    特点:
        - 进程隔离：与主进程完全隔离，不共享内存
        - 可强杀：超时后通过 SIGKILL 杀掉整个进程组
        - 资源监控：父进程创建 cgroup 后持续采样
        - 支持子进程再 fork：子进程树都在同一进程组内
        - 依赖隔离：通过 uv 创建独立虚拟环境（可选）

    Example:
        def my_evaluator(code: str) -> Dict[str, Any]:
            exec(code, globals())
            return {"score": 100, "passed": True}

        config = ProcessConfig(use_cgroup=True, memory_mb=512, cpu_percent=100)
        async with ProcessEnv(config) as env:
            result = await env.execute_with_evaluator(
                program_code="x = 1 + 1",
                evaluate_fn=my_evaluator,
                timeout=10.0,
            )
            if result.success:
                print(result.result)
            print(f"Memory: {result.memory_mb} MB, CPU: {result.cpu_percent}%")

    Example with dependencies:
        async with ProcessEnv() as env:
            deps = {"packages": ["requests>=2.28.0", "numpy"]}
            result = await env.execute_with_evaluator(
                program_code="import requests; import numpy as np; ...",
                evaluate_fn=my_evaluator,
                deps=deps,
            )
    """

    def __init__(self, config: Optional[ProcessConfig] = None):
        self.config = config or ProcessConfig()
        self._cgroup_manager: Optional[Manager] = None
        self._current_process: Optional[Process] = None
        self._current_pgid: Optional[int] = None
        self._uv_manager: Optional[UvManager] = None
        self._current_venv: Optional[VenvInfo] = None
        self._is_executing = False  # 执行状态标志，检测并发执行

    async def _setup(self) -> None:
        # uv 管理器：仅在尚未初始化时创建（跨 async with 复用）
        if self.config.uv_enabled and self._uv_manager is None:
            uv_config = UvConfig(
                cache_dir=self.config.uv_cache_dir,
                venv_base=self.config.uv_venv_base,
                python_preference=self.config.uv_python_preference,
                timeout=self.config.uv_timeout,
            )
            self._uv_manager = UvManager(uv_config)
            try:
                await self._uv_manager.setup()
            except Exception as e:
                logger.error(f"Failed to setup uv manager: {e}")
                raise

    async def _teardown(self) -> None:
        await self._cleanup_process()
        self._cleanup_cgroup()
        # 注意：不清理 _uv_manager，使其跨 async with 复用
        # UvManager 的生命周期与 ProcessEnv 实例绑定，通过 close() 显式清理

    async def close(self) -> None:
        """
        显式销毁 ProcessEnv 实例，清理所有资源（包括 uv 缓存）

        应在 ProcessEnv 实例不再使用时调用，例如 actor 销毁时
        """
        await self._cleanup_process()
        self._cleanup_cgroup()
        if self._uv_manager:
            await self._uv_manager.teardown()
            self._uv_manager = None
        self._current_venv = None

    def _cleanup_cgroup(self) -> None:
        if self._cgroup_manager:
            try:
                self._cgroup_manager.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup cgroup: {e}")
            finally:
                self._cgroup_manager = None

    async def _cleanup_process(self) -> None:
        # 关键修复：使用局部变量避免 TOCTOU 竞态条件
        # 在检查和使用之间，self._current_process 可能被其他协程设为 None
        process = self._current_process
        pgid = self._current_pgid
        
        if process is None:
            return
        
        # 立即清空实例变量,防止重复清理
        self._current_process = None
        self._current_pgid = None

        try:
            pid = process.pid
        except Exception:
            pid = pgid
        
        # 使用局部变量 process 进行后续操作
        try:
            if process.is_alive():
                logger.debug(f"_cleanup_process: Sending SIGTERM to process {pid}")
                # 临时恢复 pgid 用于 kill
                self._current_pgid = pgid
                await self._kill_process_group(signal.SIGTERM)
                self._current_pgid = None
                # Use time.sleep instead of asyncio.sleep to prevent cancellation during cleanup
                time.sleep(0.1)

            if process.is_alive():
                logger.debug(f"_cleanup_process: Sending SIGKILL to process {pid}")
                self._current_pgid = pgid
                await self._kill_process_group(signal.SIGKILL)
                self._current_pgid = None
                try:
                    process.join(timeout=1.0)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Error during process cleanup: {e}")

    def interrupt(self) -> None:
        """立即终止当前正在运行的子进程及其所有子进程。

        同步方法，可从任意线程安全调用，无需 asyncio。
        - Linux：os.killpg 杀整个进程组，包括孙子进程
        - Mac/Windows：psutil 递归 kill 进程树
        - fallback：直接 kill 直接子进程
        """
        pgid = self._current_pgid
        process = self._current_process
        if pgid is not None:
            try:
                if is_linux():
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    try:
                        parent = psutil.Process(pgid)
                        for child in parent.children(recursive=True):
                            try:
                                child.kill()
                            except psutil.NoSuchProcess:
                                pass
                        parent.kill()
                    except Exception:
                        if process is not None:
                            try:
                                if process.is_alive():
                                    process.kill()
                            except Exception:
                                pass
            except Exception:
                if process is not None:
                    try:
                        if process.is_alive():
                            process.kill()
                    except Exception:
                        pass
        elif process is not None:
            try:
                if process.is_alive():
                    process.kill()
            except Exception:
                pass

    async def _kill_process_group(self, sig: signal.Signals) -> None:
        """
        终止进程及其所有子进程

        Linux 上使用进程组 (killpg)，Mac/Windows 上使用 psutil 扫描子进程树
        """
        if self._current_pgid is None:
            return

        if is_linux():
            # Linux: 使用进程组
            try:
                os.killpg(self._current_pgid, sig)
                logger.debug(f"Sent signal {sig.name} to process group {self._current_pgid}")
            except ProcessLookupError:
                logger.debug(f"Process group {self._current_pgid} already terminated")
            except PermissionError as e:
                logger.warning(f"Permission denied when killing process group: {e}")
            except OSError as e:
                logger.warning(f"Failed to kill process group: {e}")
        else:
            # Mac/Windows: 使用 psutil 扫描子进程树
            await self._kill_process_tree(self._current_pgid, sig)

    async def _kill_process_tree(self, pid: int, sig: signal.Signals) -> None:
        """
        使用 psutil 递归终止进程树（适用于 Mac/Windows）

        Args:
            pid: 根进程 PID
            sig: 要发送的信号
        """
        try:
            parent = psutil.Process(pid)
        except psutil.NoSuchProcess:
            logger.debug(f"Process {pid} already terminated")
            return

        # 获取所有子进程（递归）
        children = []
        try:
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            pass

        # 先终止子进程（从叶子节点开始）
        for child in reversed(children):
            try:
                child.send_signal(sig)
                logger.debug(f"Sent signal {sig.name} to child process {child.pid}")
            except psutil.NoSuchProcess:
                logger.debug(f"Child process {child.pid} already terminated")
            except psutil.AccessDenied as e:
                logger.warning(f"Permission denied when killing child process {child.pid}: {e}")

        # 最后终止父进程
        try:
            parent.send_signal(sig)
            logger.debug(f"Sent signal {sig.name} to parent process {pid}")
        except psutil.NoSuchProcess:
            logger.debug(f"Parent process {pid} already terminated")
        except psutil.AccessDenied as e:
            logger.warning(f"Permission denied when killing parent process {pid}: {e}")

    def _setup_cgroup(self) -> tuple[Optional[Manager], Optional[str]]:
        """
        创建并 setup cgroup

        Returns:
            tuple: (Manager 实例, cgroup 名称)
        """
        if not self.config.use_cgroup:
            return None, None

        if not is_linux():
            logger.debug("Not Linux, cgroup not available")
            return None, None

        try:
            limits = ResourceLimits(
                memory_mb=self.config.memory_mb,
                cpu_percent=self.config.cpu_percent,
            )
            cgroup_name = f"famou_env_{uuid.uuid4().hex[:8]}"
            manager = Manager(name=cgroup_name, limits=limits)
            manager.setup()  # 创建 cgroup 目录
            return manager, cgroup_name

        except CgroupNotAvailableError as e:
            logger.warning(f"Cgroup not available: {e.reason}")
            return None, None
        except Exception as e:
            logger.warning(f"Failed to create cgroup: {e}")
            return None, None

    async def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn: EvaluateFnType,
        timeout: Optional[float] = None,
        deps: Optional[DepsSpec] = None,
    ) -> ExecutionResult:
        """
        在独立子进程中执行预加载的 evaluator 函数

        Args:
            program_code: Python 代码字符串
            evaluate_fn: 评测函数，支持两种形式:
                - Callable[[str], Dict[str, Any]]: 可 pickle 序列化的函数
                - EvaluatorSpec: {"filepath": str, "classname": str}，
                  用于无法 pickle 的类方法，在子进程中动态导入
            timeout: 执行超时时间（秒），None 表示不限制
            deps: 依赖包规格，用于指定运行时依赖。
                  仅在 uv_enabled=True 时生效，会创建独立的虚拟环境

        Returns:
            ExecutionResult: 执行结果
        """
        # 检测并发执行
        if self._is_executing:
            error_msg = (
                "Concurrent execution detected! ProcessEnv does not support "
                "concurrent calls to execute_with_evaluator() on the same instance. "
                "Please create separate ProcessEnv instances for concurrent execution."
            )
            logger.error(error_msg)
            return ExecutionResult(
                duration=0.0,
                success=False,
                error=error_msg,
            )
        
        self._is_executing = True
        
        try:
            return await self._do_execute_with_evaluator(
                program_code, evaluate_fn, timeout, deps
            )
        finally:
            self._is_executing = False
    
    async def _do_execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn: EvaluateFnType,
        timeout: Optional[float],
        deps: Optional[DepsSpec],
    ) -> ExecutionResult:
        """内部执行方法，被 execute_with_evaluator 调用"""
        start_ts = time.monotonic()

        await self._cleanup_process()
        self._cleanup_cgroup()

        # 处理依赖：创建虚拟环境
        venv_info = None
        if deps and self.config.uv_enabled:
            if self._uv_manager is None:
                logger.warning("uv_manager not initialized, uv_enabled=True but _setup was not called")
                return ExecutionResult(
                    duration=time.monotonic() - start_ts,
                    success=False,
                    error="uv_manager not initialized",
                )

            try:
                venv_info = await self._uv_manager.create_venv(deps)
                self._current_venv = venv_info
            except DependencyInstallError as e:
                logger.error(f"Failed to create uv environment: {e}")
                return ExecutionResult(
                    duration=time.monotonic() - start_ts,
                    success=False,
                    error=f"Failed to create dependency environment: {e}",
                )
            except Exception as e:
                logger.error(f"Unexpected error creating uv environment: {e}")
                return ExecutionResult(
                    duration=time.monotonic() - start_ts,
                    success=False,
                    error=f"Failed to create dependency environment: {e}",
                )

        # 父进程先创建并 setup cgroup
        self._cgroup_manager, cgroup_name = self._setup_cgroup()

        # 开始采样（在子进程启动前就开始）
        if self._cgroup_manager:
            self._cgroup_manager.sample()

        parent_conn, child_conn = multiprocessing.Pipe()

        try:
            # 准备虚拟环境信息（传递给子进程）
            venv_dict = None
            if venv_info:
                venv_dict = {
                    "path": venv_info.path,
                    "python_path": venv_info.python_path,
                    "deps_hash": venv_info.deps_hash,
                    "python_version": venv_info.python_version,
                }

            # 如果有虚拟环境，尝试使用 fork 模式以便更好地控制模块加载
            # fork 模式下，子进程会完整继承父进程状态，我们可以在子进程中修改 sys.path 和 sys.modules
            ctx = multiprocessing.get_context()
            if venv_info and 'fork' in multiprocessing.get_all_start_methods():
                try:
                    ctx = multiprocessing.get_context('fork')
                    logger.debug("Using 'fork' start method for better venv isolation")
                except Exception:
                    # 如果 fork 不可用，使用默认模式
                    pass

            process = ctx.Process(
                target=_child_process_wrapper,
                args=(child_conn, cgroup_name, evaluate_fn, program_code, venv_dict),
                daemon=False,
            )

            process.start()
            self._current_process = process
            self._current_pgid = process.pid

            child_conn.close()

            result_data = await self._wait_for_result(
                parent_conn, process, timeout, start_ts
            )

            duration = time.monotonic() - start_ts
            metrics = self._collect_metrics()

            if result_data.get("timeout"):
                return ExecutionResult(
                    duration=duration,
                    success=False,
                    timeout=True,
                    error="timeout",
                    **metrics,
                )

            if result_data.get("success"):
                return ExecutionResult(
                    duration=duration,
                    success=True,
                    result=result_data.get("result"),
                    **metrics,
                )
            else:
                return ExecutionResult(
                    duration=duration,
                    success=False,
                    error=result_data.get("error", "unknown error"),
                    **metrics,
                )

        except Exception as e:
            duration = time.monotonic() - start_ts
            logger.error(f"Exception in execute_with_evaluator: {e}\n{traceback.format_exc()}")
            return ExecutionResult(
                duration=duration,
                success=False,
                error=f"{e}\n{traceback.format_exc()}",
                **self._collect_metrics(),
            )

        finally:
            try:
                parent_conn.close()
            except Exception as e:
                logger.debug(f"Failed to close parent_conn: {e}")
            
            # 清理进程（如果还活着）
            try:
                await self._cleanup_process()
            except Exception as e:
                logger.error(f"Failed to cleanup process: {e}\n{traceback.format_exc()}")

    async def _wait_for_result(
        self,
        conn: Connection,
        process: Process,
        timeout: Optional[float],
        start_ts: float,
    ) -> Dict[str, Any]:
        """等待子进程结果，同时进行资源采样"""
        deadline = None
        if timeout is not None:
            deadline = start_ts + timeout

        while True:
            # 先采样
            if self._cgroup_manager:
                self._cgroup_manager.sample()

            # 检查超时
            if deadline is not None and time.monotonic() >= deadline:
                logger.info(f"Process timeout, killing process group {self._current_pgid}")
                await self._kill_process_group(signal.SIGKILL)
                process.join(timeout=1.0)
                return {"timeout": True, "success": False}

            # 检查管道是否有数据
            if conn.poll(0):
                try:
                    result = conn.recv()
                    process.join(timeout=1.0)
                    return result
                except EOFError:
                    process.join(timeout=1.0)
                    return {
                        "success": False,
                        "error": f"Child process terminated unexpectedly (exit code: {process.exitcode})",
                    }
                except Exception as e:
                    return {"success": False, "error": f"Failed to receive result: {e}"}

            # 检查进程是否已退出
            # 使用局部变量 process 避免并发访问 self._current_process 导致 NoneType 错误
            try:
                is_alive = process.is_alive()
            except Exception as e:
                logger.error(f"Failed to check process status: {e}")
                return {"success": False, "error": f"Process check failed: {e}"}
            
            if not is_alive:
                if conn.poll(0):
                    try:
                        result = conn.recv()
                        return result
                    except Exception:
                        pass
                
                exitcode = process.exitcode
                error_msg = f"Child process terminated unexpectedly (exit code: {exitcode})"
                
                # 提供更详细的错误信息
                if exitcode == -15:
                    error_msg += " - Process received SIGTERM signal (possible timeout or external kill)"
                elif exitcode == -9:
                    error_msg += " - Process received SIGKILL signal (forced termination)"
                elif exitcode and exitcode < 0:
                    error_msg += f" - Process terminated by signal {-exitcode}"
                
                logger.warning(error_msg)
                return {"success": False, "error": error_msg}

            await asyncio.sleep(self.config.sample_interval)

    def _collect_metrics(self) -> Dict[str, Any]:
        """收集资源使用统计"""
        if not self._cgroup_manager:
            return {}

        stats = self._cgroup_manager.get_stats()
        return {
            "memory_mb": stats.peak_memory_mb if stats.peak_memory_mb > 0 else None,
            "cpu_percent": stats.cpu_percent if stats.cpu_percent > 0 else None,
        }
