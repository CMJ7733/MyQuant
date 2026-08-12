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
Cgroup v1 实现模块

提供基于 cgroup v1 的资源管理实现:
- CgroupV1: cgroup v1 后端实现
"""
import time
from pathlib import Path
from typing import Optional

from ..logger import get_logger

from .base import CgroupBackend, CgroupMountInfo, ResourceLimits, ResourceSample

logger = get_logger("evaluator_env.resource.v1")


class CgroupV1(CgroupBackend):
    """
    Cgroup v1 后端实现

    cgroup v1 特点:
        - 每种资源控制器独立挂载（memory, cpu, cpuacct 等）
        - 需要分别操作不同控制器的目录
        - 兼容性好，支持较老的 Linux 内核

    文件系统结构:
        /sys/fs/cgroup/memory/<name>/  - 内存控制
        /sys/fs/cgroup/cpu/<name>/     - CPU 控制
    """

    def __init__(
        self,
        name: str,
        limits: ResourceLimits,
        mount_info: Optional[CgroupMountInfo] = None,
    ):
        """
        初始化 cgroup v1 后端

        Args:
            name: cgroup 名称
            limits: 资源限制配置
            mount_info: 挂载信息，包含动态检测的挂载点路径
        """
        super().__init__(name, limits)
        self._mount_info = mount_info
        self._memory_base: Optional[str] = None
        self._cpu_base: Optional[str] = None
        self._cpuacct_base: Optional[str] = None
        self._memory_path: Optional[Path] = None
        self._cpu_path: Optional[Path] = None
        self._cpuacct_path: Optional[Path] = None

        # 从挂载信息中获取路径
        if mount_info:
            self._memory_base = mount_info.memory_path
            self._cpu_base = mount_info.cpu_path
            self._cpuacct_base = mount_info.cpuacct_path

    def setup(self) -> None:
        """
        创建 cgroup v1 目录并设置资源限制

        创建 memory、cpu 和 cpuacct 控制器的 cgroup 目录，
        并写入相应的资源限制配置。

        Raises:
            ValueError: 当挂载路径未配置时抛出
            PermissionError: 当权限不足时抛出
            OSError: 当文件系统操作失败时抛出
        """
        if self._is_setup:
            logger.warning(f"Cgroup v1 already setup: {self.name}")
            return

        if not self._memory_base or not self._cpu_base:
            raise ValueError("Cgroup v1 mount paths not configured")

        # 创建 memory cgroup
        self._memory_path = Path(self._memory_base) / self.name
        self._memory_path.mkdir(exist_ok=True)

        # 创建 cpu cgroup
        self._cpu_path = Path(self._cpu_base) / self.name
        self._cpu_path.mkdir(exist_ok=True)

        # 创建 cpuacct cgroup（用于读取 CPU 统计）
        if self._cpuacct_base and self._cpuacct_base != self._cpu_base:
            self._cpuacct_path = Path(self._cpuacct_base) / self.name
            self._cpuacct_path.mkdir(exist_ok=True)
        else:
            # cpuacct 与 cpu 同路径
            self._cpuacct_path = self._cpu_path

        # 设置内存限制
        if self.limits.memory_mb is not None:
            memory_limit_bytes = int(self.limits.memory_mb * 1024 * 1024)
            (self._memory_path / "memory.limit_in_bytes").write_text(
                str(memory_limit_bytes)
            )

        # 设置 CPU 限制（使用 CFS 带宽控制）
        if self.limits.cpu_percent is not None:
            # cpu.cfs_period_us 默认 100000 (100ms)
            # cpu.cfs_quota_us = period * (cpu_percent / 100)
            period_us = 100000
            quota_us = int(period_us * self.limits.cpu_percent / 100)
            (self._cpu_path / "cpu.cfs_quota_us").write_text(str(quota_us))

        self._is_setup = True
        logger.info(f"Cgroup v1 setup complete: {self.name}")

    def attach_pid(self, pid: int) -> bool:
        """
        将进程加入 cgroup v1

        v1 需要分别将进程加入 memory 和 cpu 控制器。
        如果当前实例未 setup，会尝试检测已存在的 cgroup 目录（由父进程创建）。

        Args:
            pid: 进程 ID

        Returns:
            bool: 是否成功
        """
        # 如果未 setup，尝试检测已存在的 cgroup 目录
        if not self._is_setup:
            if not self._try_attach_existing():
                logger.error("Cgroup v1 not setup and no existing cgroup found")
                return False

        try:
            # 分别加入 memory 和 cpu cgroup
            (self._memory_path / "cgroup.procs").write_text(str(pid))
            (self._cpu_path / "cgroup.procs").write_text(str(pid))
            # 如果 cpuacct 独立挂载，也加入
            if (
                self._cpuacct_path
                and self._cpuacct_path != self._cpu_path
            ):
                (self._cpuacct_path / "cgroup.procs").write_text(str(pid))

            logger.debug(f"Attached PID {pid} to cgroup v1: {self.name}")
            return True

        except PermissionError as e:
            logger.error(f"Permission denied when attaching PID: {e}")
            return False
        except OSError as e:
            logger.error(f"Failed to attach PID {pid}: {e}")
            return False

    def _try_attach_existing(self) -> bool:
        """
        尝试 attach 到已存在的 cgroup（由父进程创建）

        用于子进程场景：父进程已经创建了 cgroup，子进程只需要加入。

        Returns:
            bool: 是否成功找到并 attach 到已存在的 cgroup
        """
        if not self._memory_base or not self._cpu_base:
            return False

        memory_path = Path(self._memory_base) / self.name
        cpu_path = Path(self._cpu_base) / self.name

        # 检查目录是否存在
        if not memory_path.exists() or not cpu_path.exists():
            return False

        self._memory_path = memory_path
        self._cpu_path = cpu_path

        # cpuacct 路径
        if self._cpuacct_base and self._cpuacct_base != self._cpu_base:
            cpuacct_path = Path(self._cpuacct_base) / self.name
            if cpuacct_path.exists():
                self._cpuacct_path = cpuacct_path
        else:
            self._cpuacct_path = self._cpu_path

        self._is_setup = True
        logger.debug(f"Attached to existing cgroup v1: {self.name}")
        return True

    def sample(self) -> Optional[ResourceSample]:
        """
        采样 cgroup v1 资源使用

        读取 memory.usage_in_bytes 和 cpuacct.usage。

        Returns:
            ResourceSample: 采样数据，失败时返回 None
        """
        if not self._is_setup:
            return None

        try:
            sample = ResourceSample(timestamp=time.monotonic())

            # 读取内存使用
            memory_usage = self._memory_path / "memory.usage_in_bytes"
            if memory_usage.exists():
                sample.memory_bytes = int(memory_usage.read_text().strip())

            # 读取 CPU 使用
            # cpuacct.usage 单位是纳秒
            cpuacct_usage = self._cpuacct_path / "cpuacct.usage"
            if cpuacct_usage.exists():
                usage_ns = int(cpuacct_usage.read_text().strip())
                sample.cpu_usage_usec = usage_ns // 1000

            return sample

        except (OSError, ValueError) as e:
            logger.debug(f"Failed to sample cgroup v1: {e}")
            return None

    def cleanup(self) -> None:
        """
        清理 cgroup v1 目录

        删除 memory、cpu 和 cpuacct 控制器的 cgroup 目录。
        注意：必须先确保所有进程已退出 cgroup。
        """
        if not self._is_setup:
            return

        try:
            if self._memory_path and self._memory_path.exists():
                self._memory_path.rmdir()
            if self._cpu_path and self._cpu_path.exists():
                self._cpu_path.rmdir()
            # 只有当 cpuacct 单独挂载时才清理
            if (
                self._cpuacct_path
                and self._cpuacct_path != self._cpu_path
                and self._cpuacct_path.exists()
            ):
                self._cpuacct_path.rmdir()

            logger.info(f"Cgroup v1 cleanup complete: {self.name}")

        except OSError as e:
            logger.warning(f"Failed to cleanup cgroup v1: {e}")

        finally:
            self._is_setup = False
