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
执行环境协议模块

本模块定义了代码执行环境的标准协议接口，包括:
- ExecutionResult: 执行结果数据类，封装执行状态和性能指标
- DepsSpec: 依赖包规格定义，用于指定运行时依赖
- Env: 执行环境协议，定义异步执行环境的标准接口
- BaseEnv: 抽象执行环境基类，提供通用的生命周期管理
"""
import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict, Union, runtime_checkable


class DepsSpec(TypedDict, total=False):
    """
    依赖包规格定义

    用于指定代码执行时的 Python 依赖包，支持多种声明方式。
    仅 ProcessEnv 支持此功能。

    支持格式:
        - packages: 包列表，支持 pip 格式
            - "requests"
            - "requests>=2.28.0"
            - "requests>=2.28.0,<3.0.0"
        - requirements: requirements.txt 文件路径
        - pyproject: pyproject.toml 文件路径或内联配置
        - python: Python 版本要求，如 "3.10", ">=3.9,<3.12"
        - index: 自定义 PyPI 索引 URL
        - extra_index: 额外的 PyPI 索引 URL
        - prerelease: 是否允许预发布版本

    Example:
        # 简单包列表
        deps = {"packages": ["requests>=2.28.0", "numpy"]}

        # 指定 Python 版本
        deps = {"python": "3.10", "packages": ["pandas"]}

        # 使用 requirements.txt
        deps = {"requirements": "/path/to/requirements.txt"}

        # 使用 pyproject.toml
        deps = {"pyproject": "/path/to/pyproject.toml"}
    """

    packages: List[str]
    requirements: str
    pyproject: Union[str, Dict[str, Any]]
    python: str
    index: str
    extra_index: str
    prerelease: bool


class EvaluatorSpec(TypedDict, total=False):
    """
    Evaluator 规格定义（用于 ProcessEnv）

    当 evaluator 函数无法被 pickle 序列化时（如类方法），
    可以通过指定文件路径和类名，在子进程中动态导入执行。

    支持四种加载模式:
        1. 只传 filepath: 加载文件中的 evaluate 函数执行
        2. filepath + method: 加载文件中的 method 函数执行
        3. filepath + classname: 加载文件中 class 的 evaluate 方法执行
        4. filepath + classname + method: 加载文件中 class 的 method 方法执行

    Attributes:
        filepath: Python 文件路径（必需）
        classname: 类名（可选）
        method: 要调用的方法/函数名，默认为 "evaluate"
    """

    filepath: str
    classname: str
    method: str


# ProcessEnv 支持的 evaluate_fn 类型
EvaluateFnType = Union[Callable[[str], Dict[str, Any]], EvaluatorSpec]


@dataclass
class ExecutionResult:
    """
    代码/命令执行结果

    用于封装代码执行后的各项指标和状态信息。

    Attributes:
        duration: 执行耗时（秒）
        memory_mb: 内存使用（MB），可选
        cpu_percent: CPU 使用率（百分比），可选
        success: 是否执行成功
        timeout: 是否因超时而终止
        result: 执行返回的结果，可为任意类型
        error: 错误信息，执行失败时包含具体错误描述
    """

    # 性能指标
    duration: float
    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None

    # 执行状态
    success: bool = True
    timeout: bool = False
    result: Any = None
    error: Optional[str] = None


@runtime_checkable
class Env(Protocol):
    """
    执行环境协议

    定义了代码执行环境的标准接口，支持异步上下文管理器模式。
    实现此协议的类需要提供代码执行能力。

    evaluator 函数由外部预加载，必须接受 program_code（Python 代码字符串）作为参数，
    返回包含评测结果的字典。

    Example:
        def my_evaluator(code: str) -> Dict[str, Any]:
            exec(code, globals())
            return {"score": 100, "passed": True}

        async with SomeEnv() as env:
            result = await env.execute_with_evaluator(
                program_code="print('hello')",
                evaluate_fn=my_evaluator,
                timeout=10.0,
            )
    """

    async def __aenter__(self) -> "Env":
        """进入执行环境上下文"""
        ...

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """退出执行环境上下文，清理资源"""
        ...

    async def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn: EvaluateFnType,
        timeout: Optional[float] = None,
        deps: Optional[DepsSpec] = None,
    ) -> ExecutionResult:
        """
        使用预加载的 evaluator 函数执行代码

        evaluator 函数由外部提前加载，接收 program_code（内存中的 Python 代码字符串），
        返回包含评测结果的字典。

        Args:
            program_code: Python 代码字符串（保存在内存中）
            evaluate_fn: 预加载的评测函数，签名为 (code: str) -> Dict[str, Any]
            timeout: 执行超时时间（秒），None 表示不限制
            deps: 依赖包规格，用于指定运行时依赖。
                  仅 ProcessEnv 支持，ThreadEnv 会忽略此参数并记录警告

        Returns:
            ExecutionResult: 包含执行结果和性能指标的对象
                - result: evaluate_fn 返回的字典
                - success: 是否执行成功
                - timeout: 是否超时
                - duration: 执行耗时
        """
        ...


class BaseEnv(abc.ABC):
    """
    抽象执行环境基类：
    - 实现 async context manager 生命周期
    - 实现 execute_with_evaluator 的统一调度、超时、异常包装
    - 子类只需要实现 _setup / _teardown / _execute_impl
    """

    # ========= async context manager =========

    async def __aenter__(self):
        await self._setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._teardown()
        except Exception:
            # teardown 阶段必须吞异常，防止资源泄漏
            pass

    # ========= 对外统一入口 =========
    @abc.abstractmethod
    async def execute_with_evaluator(
        self,
        program_code: str,
        evaluate_fn: EvaluateFnType,
        timeout: Optional[float] = None,
        deps: Optional[DepsSpec] = None,
    ) -> ExecutionResult:
        """
        使用预加载的 evaluator 函数执行代码

        Args:
            program_code: Python 代码字符串（保存在内存中）
            evaluate_fn: 预加载的评测函数，签名为 (code: str) -> Dict[str, Any]
            timeout: 执行超时时间（秒），None 表示不限制
            deps: 依赖包规格，用于指定运行时依赖

        Returns:
            ExecutionResult: 包含执行结果和性能指标的对象
        """

    # ========= 子类必须实现 =========

    @abc.abstractmethod
    async def _setup(self) -> None:
        """
        获取资源：启动进程 / 创建 sandbox / 绑定 cgroup 等
        """

    @abc.abstractmethod
    async def _teardown(self) -> None:
        """
        释放资源：kill 进程 / 清理 sandbox等 / 尝试停止线程
        """

    def _collect_metrics(self) -> dict:
        """
        子类可覆盖，用于返回:
        {
            "memory_mb": ...,
            "cpu_percent": ...
        }
        """
        return {}

    def interrupt(self) -> None:
        """
        立即中断当前正在执行的任务（同步，可从任意线程安全调用）。

        默认实现为空操作。子类若支持即时中断（如 ProcessEnv），
        应覆盖此方法以 kill 正在运行的子进程或进程组。
        """
