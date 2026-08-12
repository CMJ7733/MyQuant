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
代码执行环境模块

提供多种代码执行环境实现:
- ThreadEnv: 基于线程池的执行环境，轻量但无法强杀
- ProcessEnv: 基于独立进程的执行环境，支持进程组管理和 cgroup 资源监控
"""
from .env_protocol import BaseEnv, DepsSpec, Env, EvaluateFnType, EvaluatorSpec, ExecutionResult
from .process_env import ProcessConfig, ProcessEnv
from .thread_env import ThreadEnv

__all__ = [
    "Env",
    "BaseEnv",
    "ExecutionResult",
    "EvaluatorSpec",
    "EvaluateFnType",
    "DepsSpec",
    "ThreadEnv",
    "ProcessEnv",
    "ProcessConfig",
]
