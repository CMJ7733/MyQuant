"""Enrichment 抽象基类。

Enrichment 是可挂载到 Evolver 生命周期的功能增强模块。每个 enrichment 自己持有
状态(缓存、统计、锁等),通过两组钩子和 Evolver 协作:

- apply_to_rollout(rollout):rollout 派发前修改其 modules 列表(注入新模块)
- on_rollout_complete(result):rollout 成功完成后更新内部统计(可选)

Evolver 持有一个 List[Enrichment],在合适生命周期点统一遍历调用,自身不感知
具体有哪些 enrichment、各自做什么。
"""

from abc import ABC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from famou.core.data import Rollout, RolloutResult


class Enrichment(ABC):
    """Evolver 生命周期可挂载的功能增强模块。

    子类按需 override 下列钩子。基类全部默认 no-op,不强制实现。
    """

    name: str = "enrichment"

    def apply_to_rollout(self, rollout: "Rollout") -> None:
        """rollout 派发前修改其 modules 列表。

        典型场景:在合适位置插入额外的 module(debug 修复、leak 检测、
        embedding 抽取等)。默认 no-op。
        """
        pass

    def on_rollout_complete(self, result: "RolloutResult") -> None:
        """rollout 成功完成后更新内部统计。

        只在 status == SUCCESS 时被 Evolver 调用,与 strategy 的同名钩子
        触发条件一致。默认 no-op。
        """
        pass
