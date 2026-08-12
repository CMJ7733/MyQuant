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
线程执行环境模块

提供基于线程池的代码执行环境实现:
- ThreadEnv: 使用 ThreadPoolExecutor 在独立线程中执行预加载的 evaluator 函数
"""
import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

from .env_protocol import BaseEnv, DepsSpec, ExecutionResult


class ThreadEnv(BaseEnv):
    """
    基于线程池的代码执行环境

    使用 ThreadPoolExecutor 在独立线程中执行预加载的 evaluator 函数。

    特点:
        - 轻量级：无需启动子进程，开销小
        - 共享内存：与主进程共享地址空间
        - 无法强杀：Python 线程不支持强制终止（语言限制）
        - 超时软限制：超时后任务仍在后台运行，仅返回超时结果

    适用场景:
        - I/O 密集型任务
        - 轻量计算任务
        - 不要求强隔离的场景

    Example:
        def my_evaluator(code: str) -> Dict[str, Any]:
            exec(code, globals())
            return {"score": 100, "passed": True}

        async with ThreadEnv(max_workers=4) as env:
            result = await env.execute_with_evaluator(
                program_code="x = 1 + 1",
                evaluate_fn=my_evaluator,
                timeout=10.0,
            )
            if result.success:
                print(result.result)

    Attributes:
        _max_workers: 线程池最大工作线程数
        _executor: ThreadPoolExecutor 实例，在上下文中创建和销毁
    """

    def __init__(self, max_workers: int = 1):
        """
        初始化线程执行环境

        Args:
            max_workers: 线程池最大工作线程数，默认为 1
        """
        self._max_workers = max_workers
        self._executor: Optional[ThreadPoolExecutor] = None

    async def _setup(self) -> None:
        """创建线程池"""
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

    async def _teardown(self) -> None:
        """关闭线程池，不等待任务完成"""
        if self._executor:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn: Callable[[str], Dict[str, Any]],
        timeout: Optional[float] = None,
        deps: Optional[DepsSpec] = None,
    ) -> ExecutionResult:
        """
        在线程池中执行预加载的 evaluator 函数

        将同步的 evaluator 调用提交到线程池，通过 run_in_executor 转换为
        可 await 的异步操作，从而不阻塞事件循环。

        Args:
            program_code: Python 代码字符串（保存在内存中）
            evaluate_fn: 预加载的评测函数，签名为 (code: str) -> Dict[str, Any]
            timeout: 执行超时时间（秒），None 表示不限制。
                     注意：超时后线程仍在后台运行，无法强制终止
            deps: 依赖包规格，用于指定运行时依赖。
                  ThreadEnv 不支持此参数，会忽略并记录警告

        Returns:
            ExecutionResult: 执行结果，包含:
                - success: 是否成功
                - result: evaluate_fn 返回的字典（成功时）
                - error: 错误信息（失败时）
                - timeout: 是否超时
                - duration: 执行耗时
        """
        start_ts = time.monotonic()

        try:
            # 获取当前事件循环
            loop = asyncio.get_running_loop()

            # 包装同步调用：直接调用预加载的 evaluate_fn
            def _call():
                return evaluate_fn(program_code)

            # 提交到线程池执行，返回可 await 的 Future
            future = loop.run_in_executor(self._executor, _call)

            # 等待执行完成，可选超时
            if timeout is not None:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future

            duration = time.monotonic() - start_ts
            return ExecutionResult(
                duration=duration,
                success=True,
                result=result,
                **self._collect_metrics(),
            )

        except asyncio.TimeoutError:
            # 超时：注意线程仍在后台运行
            duration = time.monotonic() - start_ts
            return ExecutionResult(
                duration=duration,
                success=False,
                timeout=True,
                error="timeout",
                **self._collect_metrics(),
            )

        except Exception as e:
            # 其他异常：evaluator 执行错误等
            duration = time.monotonic() - start_ts
            return ExecutionResult(
                duration=duration,
                success=False,
                error=f"{e}\n{traceback.format_exc()}",
                **self._collect_metrics(),
            )
