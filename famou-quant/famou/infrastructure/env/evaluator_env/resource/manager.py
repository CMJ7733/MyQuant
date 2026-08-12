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
Cgroup 资源管理模块

提供基于 Linux cgroup 的资源监控和限制功能:
- Manager: Cgroup 资源监控管理器，自动检测并适配 v1/v2 版本
- CgroupNotAvailableError: cgroup 不可用异常
"""
import uuid
from typing import Optional

from ..logger import get_logger

from .base import (
    CgroupBackend,
    CgroupMountInfo,
    CgroupVersion,
    ResourceLimits,
    ResourceSample,
    ResourceStats,
    detect_cgroup_mount,
    is_linux,
)
from .v1 import CgroupV1
from .v2 import CgroupV2

logger = get_logger("evaluator_env.resource.manager")


class CgroupNotAvailableError(Exception):
    """
    Cgroup 不可用异常

    当系统不支持 cgroup 或 cgroup 不可写时抛出。

    Attributes:
        reason: 不可用的原因
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Cgroup not available: {reason}")


class Manager:
    """
    Cgroup 资源监控管理器

    提供基于 Linux cgroup 的资源监控和限制功能，自动检测并适配 v1/v2 版本。
    采样时动态计算统计值，避免长时间运行占用大量内存。
    """

    def __init__(
        self,
        name: Optional[str] = None,
        limits: Optional[ResourceLimits] = None,
    ):
        """
        初始化 Cgroup 管理器（不会自动 setup）

        Args:
            name: cgroup 名称，None 时自动生成唯一名称
            limits: 资源限制配置，None 表示不设置限制

        Raises:
            CgroupNotAvailableError: 当 cgroup 不可用时抛出
        """
        self.name = name or f"famou_env_{uuid.uuid4().hex[:8]}"
        self.limits = limits or ResourceLimits()

        # 执行可用性检测
        self.mount_info: CgroupMountInfo = self._check_availability()
        self.version: CgroupVersion = self.mount_info.version

        # 采样统计（动态计算，不保存历史数据）
        self._first_sample: Optional[ResourceSample] = None
        self._last_sample: Optional[ResourceSample] = None
        self._peak_memory_bytes: int = 0
        self._avg_memory_bytes: float = 0.0
        self._sample_count: int = 0

        # 创建后端实例（不自动 setup）
        self._backend: CgroupBackend = self._create_backend()

    def _check_availability(self) -> CgroupMountInfo:
        """检查 cgroup 是否可用"""
        if not is_linux():
            raise CgroupNotAvailableError("not a Linux system")

        mount_info = detect_cgroup_mount()

        if mount_info.version == CgroupVersion.UNKNOWN:
            raise CgroupNotAvailableError("cgroup not mounted")

        if not mount_info.writable:
            raise CgroupNotAvailableError(
                f"cgroup {mount_info.version.value} is read-only "
                f"(mount_point={mount_info.mount_point})"
            )

        return mount_info

    def _create_backend(self) -> CgroupBackend:
        """创建后端实例（不执行 setup）"""
        if self.version == CgroupVersion.V2:
            return CgroupV2(self.name, self.limits, self.mount_info)
        else:
            return CgroupV1(self.name, self.limits, self.mount_info)

    def setup(self) -> None:
        """
        创建 cgroup 目录并设置资源限制

        由父进程调用，创建 cgroup 目录并写入限制配置。

        Raises:
            CgroupNotAvailableError: 当 setup 失败时抛出
        """
        if self._backend.is_setup:
            logger.warning(f"Cgroup already setup: {self.name}")
            return

        try:
            self._backend.setup()
        except (ValueError, PermissionError, OSError) as e:
            raise CgroupNotAvailableError(
                f"failed to setup cgroup {self.version.value}: {e}"
            ) from e

        logger.info(
            f"Cgroup {self.version.value} setup at {self.mount_info.mount_point}/{self.name}"
        )

    def attach_pid(self, pid: int) -> bool:
        """
        将指定进程加入 cgroup

        Args:
            pid: 进程 ID

        Returns:
            bool: 是否成功
        """
        return self._backend.attach_pid(pid)

    def sample(self) -> Optional[ResourceSample]:
        """
        采样当前资源使用并动态更新统计值

        Returns:
            ResourceSample: 采样数据，失败时返回 None
        """
        sample = self._backend.sample()
        if sample is None:
            return None

        # 记录首次采样
        if self._first_sample is None:
            self._first_sample = sample

        # 更新最近采样
        self._last_sample = sample

        # 更新峰值内存
        if sample.memory_bytes > self._peak_memory_bytes:
            self._peak_memory_bytes = sample.memory_bytes

        # 增量计算平均内存（避免累加溢出）
        self._sample_count += 1
        self._avg_memory_bytes += (
            sample.memory_bytes - self._avg_memory_bytes
        ) / self._sample_count

        return sample

    def get_stats(self) -> ResourceStats:
        """
        获取资源使用统计

        Returns:
            ResourceStats: 资源统计结果
        """
        stats = ResourceStats()

        if self._sample_count > 0:
            stats.avg_memory_mb = self._avg_memory_bytes / (1024 * 1024)
            stats.peak_memory_mb = self._peak_memory_bytes / (1024 * 1024)

        if self._first_sample and self._last_sample:
            time_delta = self._last_sample.timestamp - self._first_sample.timestamp
            cpu_delta = (
                self._last_sample.cpu_usage_usec - self._first_sample.cpu_usage_usec
            )

            if time_delta > 0:
                stats.cpu_percent = (cpu_delta / 1_000_000) / time_delta * 100

        return stats

    def cleanup(self) -> None:
        """
        清理 cgroup 目录

        删除创建的 cgroup 目录。注意：必须先确保所有进程已退出 cgroup。
        """
        self._backend.cleanup()
        self.reset_samples()

    def reset_samples(self) -> None:
        """清空采样数据"""
        self._first_sample = None
        self._last_sample = None
        self._peak_memory_bytes = 0
        self._avg_memory_bytes = 0.0
        self._sample_count = 0

    @property
    def is_setup(self) -> bool:
        """是否已完成设置"""
        return self._backend.is_setup
