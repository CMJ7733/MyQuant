"""Leak detection enrichment:为代码生成型 rollout 注入 data leakage 检测器。

在第一个顶层 GenerateModule 之后插入 DataLeakageDetectorGenerate,
检测器会检查 result.generated_program 是否存在数据泄漏;若无可操作泄漏则透传。

Pipeline position: GenerateModule → [LeakDetector] → EvaluateModule
"""

import threading
from typing import TYPE_CHECKING, Any, Dict

from famou.infrastructure.enrichment.base import Enrichment

if TYPE_CHECKING:
    from famou.core.data import Rollout


class LeakDetectionEnrichment(Enrichment):
    """在 GenerateModule 之后注入 DataLeakageDetectorGenerate。

    一次 evolver 实例对应一个 LeakDetectionEnrichment 实例,
    内部 detector 模块只构建一次,跨 rollout 复用。
    """

    name = "leak_detection"

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: experiment.data_leak_detector 配置,期望含 enabled
        """
        self._config = config if isinstance(config, dict) else {}
        self._detector_cache = None
        self._inject_lock = threading.Lock()

    def is_enabled(self) -> bool:
        return bool(self._config.get("enabled", False))

    def apply_to_rollout(self, rollout: "Rollout") -> None:
        if not self.is_enabled():
            return

        with self._inject_lock:
            from famou.modules.generate.base import GenerateModule
            from famou.modules.generate.data_leak_detector import (
                DataLeakageDetectorGenerate,
            )

            # 已注入过(按 name 判定)则跳过
            for module in rollout.modules:
                if getattr(module, "name", None) == "leak_detection_generate":
                    return

            # 找到第一个顶层 GenerateModule(排除 DataLeakageDetectorGenerate 本身)
            gen_idx = None
            for i, module in enumerate(rollout.modules):
                if isinstance(module, GenerateModule) and not isinstance(
                    module, DataLeakageDetectorGenerate
                ):
                    gen_idx = i
                    break

            if gen_idx is None:
                return  # 非代码生成型 rollout

            if self._detector_cache is None:
                self._detector_cache = DataLeakageDetectorGenerate(
                    name="leak_detection_generate",
                )

            # 紧跟 GenerateModule 之后插入,在 EvaluateModule 之前
            rollout.modules.insert(gen_idx + 1, self._detector_cache)
