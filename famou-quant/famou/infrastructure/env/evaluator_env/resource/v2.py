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
Cgroup v2 实现模块

提供基于 cgroup v2 的资源管理实现:
- CgroupV2: cgroup v2 后端实现
"""
import time
from pathlib import Path
from typing import Optional

from ..logger import get_logger

from .base import CgroupBackend, CgroupMountInfo, ResourceLimits, ResourceSample

logger = get_logger("evaluator_env.resource.v2")


class CgroupV2(CgroupBackend):
    """
    Cgroup v2 后端实现

    cgroup v2 特点:
        - 统一层次结构，所有控制器在同一目录
        - 更简洁的接口设计
        - 需要较新的 Linux 内核（4.5+）

    文件系统结构:
        <mount_point>/<name>/  - 统一的 cgroup 目录
            - memory.max        - 内存限制
            - memory.current    - 当前内存使用
            - cpu.max           - CPU 限制
            - cpu.stat          - CPU 统计
            - cgroup.procs      - 进程列表
    """

    def __init__(
        self,
        name: str,
        limits: ResourceLimits,
        mount_info: Optional[CgroupMountInfo] = None,
    ):
        """
        初始化 cgroup v2 后端

        Args:
            name: cgroup 名称
            limits: 资源限制配置
            mount_info: 挂载信息，包含动态检测的挂载点路径
        """
        super().__init__(name, limits)
        self._mount_info = mount_info
        self._cgroup_base: Optional[str] = None
        self._cgroup_path: Optional[Path] = None

        # 从挂载信息中获取路径
        if mount_info:
            self._cgroup_base = mount_info.mount_point

    def setup(self) -> None:
        """
        创建 cgroup v2 目录并设置资源限制

        创建统一的 cgroup 目录，启用所需控制器，
        并写入资源限制配置。

        Raises:
            ValueError: 当挂载路径未配置时抛出
            PermissionError: 当权限不足时抛出
            OSError: 当文件系统操作失败时抛出
        """
        if self._is_setup:
            logger.warning(f"Cgroup v2 already setup: {self.name}")
            return

        if not self._cgroup_base:
            raise ValueError("Cgroup v2 mount path not configured")

        self._cgroup_path = Path(self._cgroup_base) / self.name
        self._cgroup_path.mkdir(exist_ok=True)

        # 启用控制器
        self._enable_controllers()

        # 设置内存限制
        if self.limits.memory_mb is not None:
            memory_limit_bytes = int(self.limits.memory_mb * 1024 * 1024)
            memory_max = self._cgroup_path / "memory.max"
            if memory_max.exists():
                memory_max.write_text(str(memory_limit_bytes))

        # 设置 CPU 限制
        if self.limits.cpu_percent is not None:
            # cpu.max 格式: "quota period" 或 "max period"
            period_us = 100000
            quota_us = int(period_us * self.limits.cpu_percent / 100)
            cpu_max = self._cgroup_path / "cpu.max"
            if cpu_max.exists():
                cpu_max.write_text(f"{quota_us} {period_us}")

        self._is_setup = True
        logger.info(f"Cgroup v2 setup complete: {self.name}")

    def _enable_controllers(self) -> None:
        """
        启用 memory 和 cpu 控制器

        通过写入父目录的 cgroup.subtree_control 文件来启用控制器。

        Raises:
            OSError: 当无法启用控制器时抛出
        """
        subtree_control = self._cgroup_path.parent / "cgroup.subtree_control"
        if not subtree_control.exists():
            return

        current = subtree_control.read_text()
        controllers_to_add = []

        if "memory" not in current:
            controllers_to_add.append("+memory")
        if "cpu" not in current:
            controllers_to_add.append("+cpu")

        if controllers_to_add:
            subtree_control.write_text(" ".join(controllers_to_add))

    def attach_pid(self, pid: int) -> bool:
        """
        将进程加入 cgroup v2

        如果当前实例未 setup，会尝试检测已存在的 cgroup 目录（由父进程创建）。

        Args:
            pid: 进程 ID

        Returns:
            bool: 是否成功
        """
        # 如果未 setup，尝试检测已存在的 cgroup 目录
        if not self._is_setup:
            if not self._try_attach_existing():
                logger.error("Cgroup v2 not setup and no existing cgroup found")
                return False

        try:
            procs_file = self._cgroup_path / "cgroup.procs"
            procs_file.write_text(str(pid))

            logger.debug(f"Attached PID {pid} to cgroup v2: {self.name}")
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
        if not self._cgroup_base:
            return False

        cgroup_path = Path(self._cgroup_base) / self.name

        # 检查目录是否存在
        if not cgroup_path.exists():
            return False

        self._cgroup_path = cgroup_path
        self._is_setup = True
        logger.debug(f"Attached to existing cgroup v2: {self.name}")
        return True

    def sample(self) -> Optional[ResourceSample]:
        """
        采样 cgroup v2 资源使用

        读取 memory.current 和 cpu.stat。

        Returns:
            ResourceSample: 采样数据，失败时返回 None
        """
        if not self._is_setup:
            return None

        try:
            sample = ResourceSample(timestamp=time.monotonic())

            # 读取内存使用
            memory_current = self._cgroup_path / "memory.current"
            if memory_current.exists():
                sample.memory_bytes = int(memory_current.read_text().strip())

            # 读取 CPU 使用
            cpu_stat = self._cgroup_path / "cpu.stat"
            if cpu_stat.exists():
                for line in cpu_stat.read_text().splitlines():
                    if line.startswith("usage_usec"):
                        sample.cpu_usage_usec = int(line.split()[1])
                        break

            return sample

        except (OSError, ValueError) as e:
            logger.debug(f"Failed to sample cgroup v2: {e}")
            return None

    def cleanup(self) -> None:
        """
        清理 cgroup v2 目录

        删除 cgroup 目录。
        注意：必须先确保所有进程已退出 cgroup。
        """
        if not self._is_setup:
            return

        try:
            if self._cgroup_path and self._cgroup_path.exists():
                self._cgroup_path.rmdir()

            logger.info(f"Cgroup v2 cleanup complete: {self.name}")

        except OSError as e:
            logger.warning(f"Failed to cleanup cgroup v2: {e}")

        finally:
            self._is_setup = False
