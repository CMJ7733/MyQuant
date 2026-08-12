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
Cgroup 基础类型和抽象基类

定义 cgroup 资源管理的通用数据结构和抽象接口:
- CgroupVersion: cgroup 版本枚举
- ResourceLimits: 资源限制配置
- ResourceSample: 资源采样数据点
- ResourceStats: 资源统计结果
- CgroupBackend: cgroup 操作抽象基类
- CgroupMountInfo: cgroup 挂载信息
- detect_cgroup_mount: 检测 cgroup 挂载点和版本
"""
import abc
import os
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from ..logger import get_logger

logger = get_logger("evaluator_env.resource")


class CgroupVersion(Enum):
    """Cgroup 版本枚举"""

    V1 = "v1"
    V2 = "v2"
    UNKNOWN = "unknown"


@dataclass
class CgroupMountInfo:
    """
    Cgroup 挂载信息

    Attributes:
        version: cgroup 版本
        mount_point: 挂载点路径（v2 为统一路径，v1 为基础路径）
        memory_path: 内存控制器路径（仅 v1）
        cpu_path: CPU 控制器路径（仅 v1，用于设置 CPU 限制）
        cpuacct_path: CPU 统计控制器路径（仅 v1，用于读取 CPU 使用）
        writable: 是否有写权限
    """

    version: CgroupVersion
    mount_point: Optional[str] = None
    memory_path: Optional[str] = None
    cpu_path: Optional[str] = None
    cpuacct_path: Optional[str] = None
    writable: bool = False


def is_linux() -> bool:
    """检查是否为 Linux 系统"""
    return platform.system() == "Linux"


def _parse_mountinfo() -> List[Tuple[str, str, str]]:
    """
    解析 /proc/self/mountinfo 获取挂载信息

    mountinfo 格式示例:
    454 441 0:35 /... /sys/fs/cgroup/memory ro,nosuid,... - cgroup cgroup rw,memory

    各字段含义:
    - 前半部分（- 之前）: ID PARENT_ID MAJOR:MINOR ROOT MOUNT_POINT OPTIONS [OPTIONAL...]
    - 后半部分（- 之后）: FSTYPE SOURCE SUPER_OPTIONS

    对于 cgroup v1，控制器名称在 SUPER_OPTIONS 中（如 rw,memory 或 rw,cpu）

    Returns:
        List[Tuple[str, str, str]]: [(挂载点, 文件系统类型, super_options), ...]
    """
    mounts = []
    mountinfo_path = Path("/proc/self/mountinfo")

    if not mountinfo_path.exists():
        return mounts

    try:
        content = mountinfo_path.read_text()
        for line in content.splitlines():
            # 用 " - " 分割前后两部分
            parts = line.split(" - ")
            if len(parts) != 2:
                continue

            before_sep = parts[0].split()
            after_sep = parts[1].split()

            # before_sep: [ID, PARENT_ID, MAJOR:MINOR, ROOT, MOUNT_POINT, OPTIONS, ...]
            # after_sep: [FSTYPE, SOURCE, SUPER_OPTIONS]
            if len(before_sep) >= 5 and len(after_sep) >= 3:
                mount_point = before_sep[4]
                fstype = after_sep[0]
                # super_options 包含控制器名称，如 "rw,memory" 或 "rw,cpu"
                super_options = after_sep[2]
                mounts.append((mount_point, fstype, super_options))
    except OSError as e:
        logger.debug(f"Failed to read /proc/self/mountinfo: {e}")

    return mounts


def _check_writable(path: str) -> bool:
    """
    检查目录是否可写

    Args:
        path: 目录路径

    Returns:
        bool: 是否可写
    """
    try:
        return os.access(path, os.W_OK)
    except OSError:
        return False


def detect_cgroup_mount() -> CgroupMountInfo:
    """
    检测 cgroup 挂载点和版本

    检测流程:
    1. 检查是否为 Linux 系统
    2. 解析 /proc/self/mountinfo 获取挂载信息
    3. 优先检测 cgroup2（统一层次结构）
    4. 回退检测 cgroup v1（分离的控制器）
    5. 检查挂载点的写权限

    Returns:
        CgroupMountInfo: 挂载信息
    """
    info = CgroupMountInfo(version=CgroupVersion.UNKNOWN)

    # 1. 检查是否为 Linux
    if not is_linux():
        logger.debug("Not a Linux system, cgroup not available")
        return info

    # 2. 解析 mountinfo
    mounts = _parse_mountinfo()
    if not mounts:
        logger.debug("Failed to parse mountinfo")
        return info

    # 3. 优先检测 cgroup2
    for mount_point, fstype, _ in mounts:
        if fstype == "cgroup2":
            info.version = CgroupVersion.V2
            info.mount_point = mount_point
            info.writable = _check_writable(mount_point)
            logger.debug(f"Found cgroup v2 at {mount_point}, writable={info.writable}")
            return info

    # 4. 检测 cgroup v1
    # super_options 格式如 "rw,memory" 或 "rw,cpu"
    memory_mount = None
    cpu_mount = None
    cpuacct_mount = None

    for mount_point, fstype, super_options in mounts:
        if fstype != "cgroup":
            continue

        # 解析 super_options 中的控制器名称
        # 例如: "rw,memory" -> ["rw", "memory"]
        option_parts = super_options.split(",")

        for opt in option_parts:
            if opt == "memory":
                memory_mount = mount_point
            elif opt == "cpu":
                cpu_mount = mount_point
            elif opt == "cpuacct":
                cpuacct_mount = mount_point

    # cpu 和 cpuacct 可能分开挂载，我们需要 cpu 来设置限制
    # cpuacct 用于读取 CPU 使用统计，如果没有单独挂载则回退到 cpu 路径
    if memory_mount and cpu_mount:
        info.version = CgroupVersion.V1
        info.mount_point = str(Path(memory_mount).parent)  # 通常是 /sys/fs/cgroup
        info.memory_path = memory_mount
        info.cpu_path = cpu_mount
        info.cpuacct_path = cpuacct_mount or cpu_mount  # 回退到 cpu 路径
        # 检查写权限
        info.writable = _check_writable(memory_mount) and _check_writable(cpu_mount)
        logger.debug(
            f"Found cgroup v1: memory={memory_mount}, cpu={cpu_mount}, "
            f"cpuacct={info.cpuacct_path}, writable={info.writable}"
        )
        return info

    logger.debug("No usable cgroup mount found")
    return info


@dataclass
class ResourceLimits:
    """
    资源限制配置

    Attributes:
        memory_mb: 内存限制（MB），None 表示不限制
        cpu_percent: CPU 使用率限制（百分比，如 100 表示一个核），None 表示不限制
    """

    memory_mb: Optional[float] = None
    cpu_percent: Optional[float] = None


@dataclass
class ResourceSample:
    """
    资源采样数据点

    Attributes:
        timestamp: 采样时间戳
        memory_bytes: 内存使用量（字节）
        cpu_usage_usec: CPU 累计使用时间（微秒）
    """

    timestamp: float
    memory_bytes: int = 0
    cpu_usage_usec: int = 0


@dataclass
class ResourceStats:
    """
    资源统计结果

    Attributes:
        avg_memory_mb: 平均内存使用（MB）
        peak_memory_mb: 峰值内存使用（MB）
        cpu_percent: CPU 使用率（百分比）
    """

    avg_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    cpu_percent: float = 0.0


class CgroupBackend(abc.ABC):
    """
    Cgroup 操作抽象基类

    定义 cgroup v1/v2 实现需要遵循的接口。

    Attributes:
        name: cgroup 名称
        limits: 资源限制配置
    """

    def __init__(self, name: str, limits: ResourceLimits):
        """
        初始化 Cgroup 后端

        Args:
            name: cgroup 名称
            limits: 资源限制配置
        """
        self.name = name
        self.limits = limits
        self._is_setup: bool = False

    @property
    def is_setup(self) -> bool:
        """是否已完成设置"""
        return self._is_setup

    @abc.abstractmethod
    def setup(self) -> None:
        """
        创建 cgroup 目录并设置资源限制

        Raises:
            ValueError: 当配置无效时抛出
            PermissionError: 当权限不足时抛出
            OSError: 当文件系统操作失败时抛出
        """

    @abc.abstractmethod
    def attach_pid(self, pid: int) -> bool:
        """
        将进程加入 cgroup

        Args:
            pid: 进程 ID

        Returns:
            bool: 是否成功
        """

    @abc.abstractmethod
    def sample(self) -> Optional[ResourceSample]:
        """
        采样当前资源使用

        Returns:
            ResourceSample: 采样数据，失败时返回 None
        """

    @abc.abstractmethod
    def cleanup(self) -> None:
        """清理 cgroup 目录"""


def compute_stats(samples: List[ResourceSample]) -> ResourceStats:
    """
    根据采样数据计算资源统计

    Args:
        samples: 采样数据列表

    Returns:
        ResourceStats: 统计结果
    """
    stats = ResourceStats()

    if not samples:
        return stats

    # 计算内存统计
    memory_values = [s.memory_bytes for s in samples if s.memory_bytes > 0]
    if memory_values:
        stats.avg_memory_mb = sum(memory_values) / len(memory_values) / (1024 * 1024)
        stats.peak_memory_mb = max(memory_values) / (1024 * 1024)

    # 计算 CPU 使用率
    # CPU 使用率 = (cpu_usage_delta / time_delta) * 100
    if len(samples) >= 2:
        first = samples[0]
        last = samples[-1]
        time_delta = last.timestamp - first.timestamp
        cpu_delta = last.cpu_usage_usec - first.cpu_usage_usec

        if time_delta > 0:
            # cpu_delta 是微秒，time_delta 是秒
            # 使用率 = (cpu_delta / 1000000) / time_delta * 100
            stats.cpu_percent = (cpu_delta / 1_000_000) / time_delta * 100

    return stats
