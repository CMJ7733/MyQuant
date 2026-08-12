"""Enrichment 模块:Evolver 生命周期可挂载的功能增强。

每个 enrichment 独立持有自己的状态(缓存、统计、锁),通过抽象基类 Enrichment
提供的钩子和 Evolver 协作。
"""

from famou.infrastructure.enrichment.base import Enrichment
from famou.infrastructure.enrichment.debug_enrichment import DebugEnrichment
from famou.infrastructure.enrichment.embedding_feature_enrichment import (
    EmbeddingFeatureEnrichment,
)
from famou.infrastructure.enrichment.leak_detection_enrichment import (
    LeakDetectionEnrichment,
)

__all__ = [
    "Enrichment",
    "DebugEnrichment",
    "EmbeddingFeatureEnrichment",
    "LeakDetectionEnrichment",
]
