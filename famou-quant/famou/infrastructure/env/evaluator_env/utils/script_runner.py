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
脚本运行工具模块

提供动态加载和执行 Python 脚本的工具函数，包括:
- load_module_from_path: 从文件路径加载 Python 模块
- call_function_from_script: 加载脚本并调用指定函数
- ScriptLoadError: 脚本加载失败异常
- FunctionNotFoundError: 函数未找到异常
"""

import importlib.util
import types
from typing import Any


class ScriptLoadError(RuntimeError):
    """脚本加载失败异常"""
    pass


class FunctionNotFoundError(RuntimeError):
    """函数未找到异常"""
    pass


def load_module_from_path(script_path: str, module_name: str) -> types.ModuleType:
    """
    从指定路径加载 python 模块，不污染 sys.modules
    """
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ScriptLoadError(f"Cannot load script: {script_path}")

    module = importlib.util.module_from_spec(spec)

    # 关键点：不要塞进 sys.modules，避免模块缓存污染
    loader = spec.loader
    assert loader is not None
    loader.exec_module(module)

    return module


def call_function_from_script(script_path: str, func_name: str, *args, **kwargs) -> Any:
    """
    加载脚本并调用指定函数
    """
    module_name = f"user_module_{id(script_path)}"
    module = load_module_from_path(script_path, module_name)

    func = getattr(module, func_name, None)
    if func is None:
        raise FunctionNotFoundError(
            f"Function '{func_name}' not found in {script_path}"
        )

    return func(*args, **kwargs)
