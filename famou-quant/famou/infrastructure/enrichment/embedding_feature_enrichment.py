"""Embedding feature enrichment:为有 judge 的 rollout 注入 EmbeddingFeature 模块。

在最后一个顶层 JudgeModule 之后插入 EmbeddingFeature,把它当作系统级行为
(由 infra 配置驱动)而不是每个 strategy 各自写的 rollout 配线问题。

启用条件不在 config 里,而是看 evolver 是否配置了 embedding_client。
"""

import threading
from typing import TYPE_CHECKING, Any, Optional

from famou.infrastructure.enrichment.base import Enrichment

if TYPE_CHECKING:
    from famou.core.data import Rollout
    from famou.infrastructure.embedding import EmbeddingClient


class EmbeddingFeatureEnrichment(Enrichment):
    """在最后一个 JudgeModule 之后注入 EmbeddingFeature。

    没有缓存——EmbeddingFeature 本身是轻量的、可重复构造的。
    """

    name = "embedding_feature"

    def __init__(self, embedding_client: Optional["EmbeddingClient"]):
        """
        Args:
            embedding_client: 若为 None 则该 enrichment 永远不注入
        """
        self._embedding_client = embedding_client
        self._inject_lock = threading.Lock()

    def is_enabled(self) -> bool:
        return self._embedding_client is not None

    def apply_to_rollout(self, rollout: "Rollout") -> None:
        if not self.is_enabled():
            return

        with self._inject_lock:
            from famou.modules.judge.base import JudgeModule
            from famou.modules.judge.embedding_feature import EmbeddingFeature

            # 已经存在则跳过
            for module in rollout.modules:
                if isinstance(module, EmbeddingFeature):
                    return

            # 找到最后一个顶层 JudgeModule
            judge_idx = None
            for i, module in enumerate(rollout.modules):
                if isinstance(module, JudgeModule):
                    judge_idx = i

            if judge_idx is None:
                return

            rollout.modules.insert(judge_idx + 1, EmbeddingFeature())
