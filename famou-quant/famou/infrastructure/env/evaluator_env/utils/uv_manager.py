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
uv 环境管理模块

提供基于 uv 的 Python 虚拟环境和依赖管理功能:
- UvManager: uv 环境管理器，负责虚拟环境创建、依赖安装、缓存管理
- UvConfig: uv 配置
- VenvInfo: 虚拟环境信息
- VenvLock: 文件锁，防止多实例并发冲突

依赖隔离机制:
1. 为每个唯一的依赖规格创建独立的虚拟环境
2. 虚拟环境基于 deps_hash 缓存复用
3. 子进程通过设置环境变量激活虚拟环境

多实例共享机制:
1. 不同的 UvManager 实例可以共享相同 venv_base 下的虚拟环境
2. 使用文件锁（VenvLock）保护关键区域，避免并发创建/修改冲突
3. 当发现目标路径已存在时，会验证并复用（避免重复创建）
4. 验证包括：Python 解释器可用性 + 依赖包完整性检查
5. 如果依赖缺失，自动补充安装；如果安装失败，删除并重建
6. 每个实例只清理自己创建的虚拟环境，不影响其他实例
7. 支持 Unix (fcntl.flock) 和 Windows (独占文件) 平台
8. 适用于容器内多个 env 实例并发执行的场景
"""

import asyncio
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..logger import get_logger

logger = get_logger("evaluator_env.uv_manager")


class VenvLock:
    """
    基于文件的虚拟环境锁
    
    用于防止多个 UvManager 实例同时操作同一个 venv 路径
    使用独占文件锁 (fcntl.flock on Unix, msvcrt.locking on Windows) 确保原子性操作
    """
    
    def __init__(self, venv_path: str, timeout: float = 60.0):
        """
        Args:
            venv_path: 虚拟环境路径
            timeout: 获取锁的超时时间（秒）
        """
        self.venv_path = Path(venv_path)
        self.lock_file_path = self.venv_path.parent / f".{self.venv_path.name}.lock"
        self.timeout = timeout
        self._lock_fd = None
        self._is_windows = sys.platform == "win32"
        
    async def __aenter__(self):
        """获取锁"""
        # 确保锁文件目录存在
        self.lock_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        start_time = time.monotonic()
        while True:
            try:
                if self._is_windows:
                    # Windows: 使用独占打开模式
                    # 如果文件被其他进程打开，会抛出 PermissionError
                    self._lock_fd = open(str(self.lock_file_path), 'w')
                else:
                    # Unix: 使用 fcntl.flock
                    import fcntl
                    self._lock_fd = os.open(str(self.lock_file_path), os.O_CREAT | os.O_RDWR)
                    fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                logger.debug(f"Acquired lock for venv: {self.venv_path}")
                return self
                
            except (IOError, OSError, PermissionError) as e:
                # 锁被其他进程持有
                if self._lock_fd is not None:
                    try:
                        if self._is_windows:
                            self._lock_fd.close()
                        else:
                            os.close(self._lock_fd)
                    except Exception:
                        pass
                    self._lock_fd = None
                
                # 检查超时
                if time.monotonic() - start_time >= self.timeout:
                    raise TimeoutError(
                        f"Failed to acquire lock for {self.venv_path} after {self.timeout}s"
                    )
                
                # 等待一小段时间后重试
                await asyncio.sleep(0.1)
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """释放锁"""
        if self._lock_fd is not None:
            try:
                if self._is_windows:
                    self._lock_fd.close()
                else:
                    import fcntl
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                    os.close(self._lock_fd)
                logger.debug(f"Released lock for venv: {self.venv_path}")
            except Exception as e:
                logger.warning(f"Failed to release lock: {e}")
            finally:
                self._lock_fd = None
                
                # 清理锁文件（忽略错误）
                try:
                    if self.lock_file_path.exists():
                        self.lock_file_path.unlink()
                except Exception:
                    pass


class UvNotAvailableError(Exception):
    """uv 不可用异常"""

    pass


class DependencyInstallError(Exception):
    """依赖安装失败异常"""

    pass


@dataclass
class UvConfig:
    """
    uv 配置

    Attributes:
        cache_dir: uv 缓存目录，None 使用 uv 默认
        venv_base: 虚拟环境存储基目录，None 使用临时目录
        python_preference: Python 查找策略
            - "managed": 优先使用 uv 管理的 Python
            - "system": 优先使用系统 Python
            - "only-managed": 仅使用 uv 管理的 Python
        timeout: uv 操作超时时间（秒）
        cleanup_on_exit: 退出时是否清理创建的虚拟环境
        default_index: 默认 PyPI 镜像源
            默认使用清华源（国内速度快）
            可选的其他国内镜像：
            - "https://mirrors.aliyun.com/pypi/simple"  # 阿里云
            - "https://pypi.mirrors.ustc.edu.cn/simple"  # 中科大
            - None  # 使用 PyPI 官方源
    """

    cache_dir: Optional[str] = None
    venv_base: Optional[str] = None
    python_preference: str = "managed"
    timeout: float = 300.0
    cleanup_on_exit: bool = True
    default_index: str = "https://pypi.tuna.tsinghua.edu.cn/simple"  # 默认使用清华源


@dataclass
class VenvInfo:
    """
    虚拟环境信息

    Attributes:
        path: 虚拟环境路径
        python_path: Python 解释器路径
        deps_hash: 依赖规格的哈希值
        python_version: Python 版本
        created_at: 创建时间戳
    """

    path: str
    python_path: str
    deps_hash: str
    python_version: str = ""
    created_at: float = field(default_factory=time.time)


class UvManager:
    """
    uv 环境管理器

    职责:
    1. 检测 uv 可用性
    2. 创建和管理虚拟环境
    3. 安装依赖包
    4. 环境缓存和复用
    5. 环境清理

    Example:
        config = UvConfig(venv_base="/tmp/evaluator_venvs")
        manager = UvManager(config)
        await manager.setup()

        deps = {"packages": ["requests>=2.28.0"]}
        venv = await manager.create_venv(deps)

        # 使用 venv.python_path 执行代码
        ...

        await manager.teardown()
    """

    def __init__(self, config: Optional[UvConfig] = None):
        self.config = config or UvConfig()
        self._uv_path: Optional[str] = None
        self._venv_cache: Dict[str, VenvInfo] = {}
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None
        self._created_venvs: List[str] = []  # 记录创建的 venv 路径

    @staticmethod
    def is_available() -> bool:
        """检查 uv 是否可用"""
        return shutil.which("uv") is not None

    async def setup(self) -> None:
        """
        初始化 uv 管理器

        Raises:
            UvNotAvailableError: uv 不可用时抛出
        """
        # 检测 uv
        self._uv_path = shutil.which("uv")
        if not self._uv_path:
            raise UvNotAvailableError(
                "uv is not installed. Please install it with: "
                "curl -LsSf https://astral.sh/uv/install.sh | sh"
            )

        # 创建临时目录（如果没有指定 venv_base）
        if not self.config.venv_base:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="uv_venvs_")
            self.config.venv_base = self._temp_dir.name
            logger.debug(f"Using temp directory for venvs: {self.config.venv_base}")

        # 确保目录存在
        Path(self.config.venv_base).mkdir(parents=True, exist_ok=True)

        logger.info(f"UvManager setup complete, uv={self._uv_path}")

    async def teardown(self) -> None:
        """
        清理资源
        
        注意：只清理由当前实例创建的虚拟环境（_created_venvs 列表中的）
        从其他 UvManager 实例复用的 venv 不会被清理，保证多实例场景下的安全性
        """
        # 清理创建的虚拟环境
        if self.config.cleanup_on_exit:
            for venv_path in self._created_venvs:
                try:
                    shutil.rmtree(venv_path, ignore_errors=True)
                    logger.debug(f"Cleaned up venv: {venv_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup venv {venv_path}: {e}")

        # 清理临时目录
        if self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass

        self._venv_cache.clear()
        self._created_venvs.clear()

    def compute_deps_hash(self, deps: Dict[str, Any]) -> str:
        """
        计算依赖规格的哈希值

        用于环境缓存 key 计算

        Args:
            deps: 依赖规格字典

        Returns:
            str: SHA256 哈希值（前 16 位）
        """
        # 序列化时对字典进行排序以确保一致性
        # 过滤空值
        filtered_deps = {k: v for k, v in deps.items() if v is not None}
        deps_json = json.dumps(filtered_deps, sort_keys=True, default=str)
        hash_value = hashlib.sha256(deps_json.encode()).hexdigest()[:16]
        return hash_value

    async def create_venv(
        self,
        deps: Dict[str, Any],
        reuse: bool = True
    ) -> VenvInfo:
        """
        创建或复用虚拟环境

        Args:
            deps: 依赖规格
            reuse: 是否复用已有环境（基于 deps_hash）

        Returns:
            VenvInfo: 虚拟环境信息

        Raises:
            DependencyInstallError: 依赖安装失败时抛出
        """
        deps_hash = self.compute_deps_hash(deps)
        # 检查缓存
        if reuse and deps_hash in self._venv_cache:
            cached_venv = self._venv_cache[deps_hash]
            if Path(cached_venv.path).exists():
                logger.debug(f"Reusing cached venv: {cached_venv.path}")
                return cached_venv

        # 创建新的虚拟环境
        venv_path = str(Path(self.config.venv_base) / f"venv_{deps_hash}")

        # 使用文件锁保护关键区域，避免多实例并发冲突
        async with VenvLock(venv_path, timeout=self.config.timeout):
            # 再次检查缓存（可能在等待锁的过程中被其他实例创建）
            if reuse and deps_hash in self._venv_cache:
                cached_venv = self._venv_cache[deps_hash]
                if Path(cached_venv.path).exists():
                    logger.debug(f"Reusing cached venv (after lock): {cached_venv.path}")
                    return cached_venv
            
            # 检查路径是否已存在（可能由其他 UvManager 实例创建）
            if reuse and Path(venv_path).exists():
                try:
                    # 尝试验证已有环境是否可用
                    python_path = self._get_python_path(venv_path)
                    if Path(python_path).exists():
                        # 获取 Python 版本信息
                        actual_python_version = await self._get_python_version(python_path)
                        
                        # 验证依赖是否已安装
                        deps_verified = await self._verify_dependencies(venv_path, deps)
                        
                        if not deps_verified:
                            # 依赖未完全安装，尝试补充安装
                            logger.info(f"Dependencies not fully installed in {venv_path}, installing...")
                            try:
                                await self._install_dependencies(venv_path, deps)
                                logger.info(f"Successfully installed missing dependencies in {venv_path}")
                            except Exception as install_error:
                                logger.warning(f"Failed to install dependencies: {install_error}, will recreate venv")
                                # 依赖安装失败，删除损坏的 venv 并重新创建
                                shutil.rmtree(venv_path, ignore_errors=True)
                                raise
                        
                        # 构建 VenvInfo 并加入缓存
                        venv_info = VenvInfo(
                            path=venv_path,
                            python_path=python_path,
                            deps_hash=deps_hash,
                            python_version=actual_python_version,
                        )
                        
                        # 记录到缓存（但不添加到 _created_venvs，因为不是本实例创建的）
                        self._venv_cache[deps_hash] = venv_info
                        
                        logger.info(f"Reusing existing venv from disk: {venv_path}, python={actual_python_version}")
                        return venv_info
                except Exception as e:
                    # 如果验证失败，记录警告并继续尝试创建（会使用 uv 的行为）
                    logger.warning(f"Failed to validate existing venv at {venv_path}: {e}, will attempt to create")

            try:
                # 1. 创建虚拟环境
                python_version = deps.get("python")
                await self._create_virtualenv(venv_path, python_version)

                # 2. 安装依赖
                await self._install_dependencies(venv_path, deps)

                # 3. 获取信息
                python_path = self._get_python_path(venv_path)
                actual_python_version = await self._get_python_version(python_path)

                venv_info = VenvInfo(
                    path=venv_path,
                    python_path=python_path,
                    deps_hash=deps_hash,
                    python_version=actual_python_version,
                )

                # 记录缓存
                self._venv_cache[deps_hash] = venv_info
                self._created_venvs.append(venv_path)

                logger.info(f"Created venv: {venv_path}, python={actual_python_version}")

                return venv_info

            except Exception as e:
                # 清理失败的 venv
                shutil.rmtree(venv_path, ignore_errors=True)
                raise DependencyInstallError(f"Failed to create venv: {e}") from e

    async def _create_virtualenv(
        self,
        venv_path: str,
        python_version: Optional[str] = None
    ) -> None:
        """
        创建虚拟环境

        Args:
            venv_path: 虚拟环境路径
            python_version: Python 版本要求
        """
        args = ["venv", venv_path]

        # Python 版本
        if python_version:
            args.extend(["--python", python_version])

        # Python 偏好设置
        if self.config.python_preference:
            args.append(f"--python-preference={self.config.python_preference}")

        await self._run_uv(args)
        logger.debug(f"Created virtualenv: {venv_path}")

    async def _verify_package_installed(self, python_path: str, package_name: str) -> bool:
        """
        验证包是否已安装
        
        Args:
            python_path: Python 解释器路径
            package_name: 包名（可能包含版本约束，如 'pandas>=1.0.0'）
            
        Returns:
            bool: 包是否已安装
        """
        # 提取纯包名（去除版本约束）
        import re
        pure_name = re.split(r'[><=!]', package_name)[0].strip()
        
        try:
            process = await asyncio.create_subprocess_exec(
                python_path, "-c", f"import {pure_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(process.communicate(), timeout=5.0)
            return process.returncode == 0
        except Exception:
            return False

    async def _verify_dependencies(self, venv_path: str, deps: Dict[str, Any]) -> bool:
        """
        验证依赖是否都已安装
        
        Args:
            venv_path: 虚拟环境路径
            deps: 依赖规格
            
        Returns:
            bool: 所有依赖是否都已安装
        """
        if not deps:
            return True
            
        python_path = self._get_python_path(venv_path)
        
        # 检查包列表
        if deps.get("packages"):
            for package in deps["packages"]:
                if not await self._verify_package_installed(python_path, package):
                    logger.debug(f"Package {package} not found in venv {venv_path}")
                    return False
        
        return True

    async def _install_dependencies(
        self,
        venv_path: str,
        deps: Dict[str, Any]
    ) -> None:
        """
        安装依赖

        Args:
            venv_path: 虚拟环境路径
            deps: 依赖规格
        """
        python_exe = self._get_python_path(venv_path)
        args = ["pip", "install", "--python", python_exe]

        # 自定义索引（优先级高于默认配置）
        if deps.get("index"):
            args.extend(["--index-url", deps["index"]])
        elif self.config.default_index:
            # 使用配置中的默认镜像源
            args.extend(["--index-url", self.config.default_index])
            logger.debug(f"Using default index: {self.config.default_index}")
        
        if deps.get("extra_index"):
            args.extend(["--extra-index-url", deps["extra_index"]])

        # 预发布版本
        if deps.get("prerelease"):
            args.append("--prerelease=allow")

        # 缓存目录
        if self.config.cache_dir:
            args.extend(["--cache-dir", self.config.cache_dir])

        has_installed = False

        # 方式 1: 包列表
        if deps.get("packages"):
            package_args = list(deps["packages"])  # 复制列表
            await self._run_uv(args + package_args)
            has_installed = True

        # 方式 2: requirements.txt
        if deps.get("requirements"):
            req_path = deps["requirements"]
            args.extend(["--requirements", req_path])
            await self._run_uv(args)
            has_installed = True

        # 方式 3: pyproject.toml
        elif deps.get("pyproject"):
            pyproject = deps["pyproject"]
            if isinstance(pyproject, str):
                # 文件路径
                args.extend(["--requirements", pyproject])
                await self._run_uv(args)
            else:
                # 内联配置，需要临时写入文件
                await self._install_from_inline_pyproject(venv_path, pyproject)
            has_installed = True

        if has_installed:
            logger.debug(f"Installed dependencies in: {venv_path}")

    async def _install_from_inline_pyproject(
        self,
        venv_path: str,
        pyproject: Dict[str, Any]
    ) -> None:
        """从内联 pyproject 配置安装依赖"""
        try:
            import tomli_w
        except ImportError:
            # 手动构建简单的 requirements 格式
            requirements = []
            if "dependencies" in pyproject:
                for dep in pyproject["dependencies"]:
                    requirements.append(dep)

            if not requirements:
                return

            # 创建临时 requirements.txt
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False
            ) as f:
                f.write("\n".join(requirements))
                temp_path = f.name

            try:
                python_exe = self._get_python_path(venv_path)
                args = [
                    "pip", "install",
                    "--python", python_exe,
                    "--requirements", temp_path
                ]
                await self._run_uv(args)
            finally:
                os.unlink(temp_path)
            return

        # 使用 tomli_w
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".toml",
            delete=False
        ) as f:
            tomli_w.dump(pyproject, f)
            temp_path = f.name

        try:
            python_exe = self._get_python_path(venv_path)
            args = [
                "pip", "install",
                "--python", python_exe,
                "--requirements", temp_path
            ]
            await self._run_uv(args)
        finally:
            os.unlink(temp_path)

    async def _run_uv(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None
    ) -> Tuple[str, str]:
        """
        执行 uv 命令

        Args:
            args: 命令参数
            cwd: 工作目录
            env: 额外环境变量

        Returns:
            Tuple[str, str]: (stdout, stderr)

        Raises:
            RuntimeError: 命令执行失败时抛出
        """
        if not self._uv_path:
            raise UvNotAvailableError("uv not setup")

        cmd = [self._uv_path] + args

        # 合并环境变量
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        logger.info(f"Running uv command: {' '.join(cmd)}")  # 改为 info 级别，更容易看到

        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=run_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            logger.debug(f"Waiting for uv command to complete (timeout={self.config.timeout}s)...")
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout
            )
            logger.debug(f"uv command completed with returncode={process.returncode}")
        except asyncio.TimeoutError:
            process.kill()
            logger.error(f"uv command timed out after {self.config.timeout}s: {' '.join(args)}")
            raise RuntimeError(f"uv command timed out: {' '.join(args)}")

        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "unknown error"
            logger.error(f"uv command failed with returncode={process.returncode}: {error_msg}")
            raise RuntimeError(f"uv command failed: {error_msg}")

        return stdout.decode(), stderr.decode()

    def _get_python_path(self, venv_path: str) -> str:
        """获取虚拟环境中的 Python 路径"""
        # Linux/macOS
        python_path = os.path.join(venv_path, "bin", "python")
        if os.path.exists(python_path):
            return python_path

        # Windows
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.exists(python_path):
            return python_path

        raise RuntimeError(f"Python not found in venv: {venv_path}")

    async def _get_python_version(self, python_path: str) -> str:
        """获取 Python 版本"""
        process = await asyncio.create_subprocess_exec(
            python_path, "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        # 输出格式: "Python 3.10.12"
        return stdout.decode().strip().replace("Python ", "")

    @staticmethod
    def activate_venv_sys_path(venv_info: VenvInfo) -> None:
        """
        在子进程中激活虚拟环境

        直接修改 sys.path 使虚拟环境中的包可用。
        允许复用父进程中已加载的包，不清理 sys.modules。

        Args:
            venv_info: 虚拟环境信息
        """
        import sys

        venv_path = Path(venv_info.path)
        python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site_packages = venv_path / "lib" / python_version / "site-packages"

        # 移除 sys.path 中所有的 site-packages 和 dist-packages 路径
        # 这样可以避免从父进程/系统环境导入包
        paths_to_remove = []
        for path in sys.path:
            path_lower = path.lower()
            # 移除所有 site-packages 和 dist-packages，但保留虚拟环境的
            if ('site-packages' in path_lower or 'dist-packages' in path_lower) and str(site_packages) not in path:
                paths_to_remove.append(path)

        for path in paths_to_remove:
            sys.path.remove(path)

        # 将虚拟环境的 site-packages 添加到 sys.path 开头
        if str(site_packages) not in sys.path:
            sys.path.insert(0, str(site_packages))

        # 同时设置环境变量
        os.environ["VIRTUAL_ENV"] = str(venv_path)
        os.environ["PATH"] = f"{venv_path / 'bin'}:{os.environ.get('PATH', '')}"

        # 清空 PYTHONHOME（避免干扰）
        if "PYTHONHOME" in os.environ:
            del os.environ["PYTHONHOME"]

        # 设置 PYTHONPATH 为 venv 的 site-packages
        # 确保由当前进程再启动的子进程也能找到 venv 中的包
        os.environ["PYTHONPATH"] = str(site_packages)

        # 更新 sys.executable 指向 venv 的 Python
        # 确保 subprocess 等使用 sys.executable 时使用 venv 的解释器
        venv_python = venv_path / "bin" / "python"
        if venv_python.exists():
            sys.executable = str(venv_python)

        # 注意：不再清理 sys.modules
        # 允许复用父进程中已加载的包，提高性能
        # 如果 venv 中有不同版本的包，Python 的 import 机制会优先使用 sys.path 中靠前的路径
        # 因为我们已经将 venv 的 site-packages 插入到 sys.path 开头，新导入的包会使用 venv 中的版本


def activate_venv_in_child(venv_info: VenvInfo) -> Dict[str, str]:
    """
    在子进程中激活虚拟环境

    通过设置环境变量实现，避免 shell 依赖

    Args:
        venv_info: 虚拟环境信息

    Returns:
        Dict[str, str]: 需要设置的环境变量
    """
    venv_path = Path(venv_info.path)
    venv_bin = str(venv_path / "bin")

    env_updates = {
        "VIRTUAL_ENV": str(venv_path),
        "PATH": f"{venv_bin}:{os.environ.get('PATH', '')}",
    }

    # 清空 PYTHONHOME（避免干扰）
    if "PYTHONHOME" in os.environ:
        env_updates["PYTHONHOME"] = ""

    # 更新 sys.path
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = venv_path / "lib" / python_version / "site-packages"

    if site_packages.exists():
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        env_updates["PYTHONPATH"] = f"{site_packages}:{current_pythonpath}"

    return env_updates
