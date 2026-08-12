"""Debug enrichment:为代码生成型 rollout 注入"检测到 bug 自动修复"的条件循环。

当 EvaluateModule 发现生成的程序 is_buggy 时,触发 ErrorDrivenGenerate -> Evaluate
的修复循环,最多 max_iterations 次。修复完成后恢复原始 parent_id,确保 archive
中血缘指向真实存在的父程序而非过程中的临时程序。

支持 dynamic 模式:根据历史触发成功率动态调整触发概率,降低无效修复的开销。
"""

import random
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from famou.infrastructure.enrichment.base import Enrichment

if TYPE_CHECKING:
    from famou.core.data import Rollout, RolloutResult


class DebugEnrichment(Enrichment):
    """检测 buggy program 后自动注入修复循环。

    一次 evolver 实例对应一个 DebugEnrichment 实例。内部 conditional 模块
    只构建一次,跨 rollout 复用(其中的子模块是无状态的)。
    """

    name = "debug"

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: experiment.debug 配置,期望含 enabled / max_iterations / dynamic
        """
        self._config = config or {}
        # 一次性构建后跨 rollout 复用的 conditional 模块
        self._conditional_cache = None
        # 多 worker 并发场景下保护 cache 构建和注入逻辑
        self._inject_lock = threading.Lock()
        # 动态触发率所需的全局统计
        self._stats_lock = threading.Lock()
        self._total_cnt = 0
        self._success_cnt = 0

    # -- Ray/pickle 兼容 --------------------------------------------------
    # 注入的 conditional_debug 模块闭包捕获了 self(见 _build_conditional 里的
    # enrichment_ref = self),而 self 持有 threading.Lock。BackendTask.rollout 跨
    # Ray 边界序列化时会连带 pickle 本对象 → "cannot pickle '_thread.lock' object"。
    # 与 OpenAIClient 处理 httpx 连接池同理:序列化时丢锁,反序列化侧重建。
    # worker 侧拿到的是 stats 快照,dynamic 触发率在分布式下本就是每 worker 独立近似。
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_inject_lock"] = None
        state["_stats_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._inject_lock = threading.Lock()
        self._stats_lock = threading.Lock()

    def is_enabled(self) -> bool:
        """Return True iff the enrichment should attempt injection.

        Reads ``enabled`` flag from the debug config; defaults to True so that
        misconfigured experiments still get the repair loop.
        """
        return bool(self._config.get("enabled", True))

    def apply_to_rollout(self, rollout: "Rollout") -> None:
        """在第一个顶层 EvaluateModule 后插入 conditional debug。"""
        if not self.is_enabled():
            return

        with self._inject_lock:
            from famou.modules.evaluate.base import EvaluateModule

            # 已注入过(按 name 判定)则跳过
            for module in rollout.modules:
                if getattr(module, "name", None) == "conditional_debug":
                    return

            # 找到第一个顶层 EvaluateModule
            eval_idx = None
            eval_module = None
            for i, module in enumerate(rollout.modules):
                if isinstance(module, EvaluateModule):
                    eval_idx = i
                    eval_module = module
                    break

            if eval_idx is None:
                return  # 非代码生成型 rollout,跳过

            if self._conditional_cache is None:
                self._conditional_cache = self._build_conditional(
                    evaluate_fn=eval_module.evaluate_fn,
                )

            rollout.modules.insert(eval_idx + 1, self._conditional_cache)

    def on_rollout_complete(self, result: "RolloutResult") -> None:
        """成功完成的 rollout,若触发过 debug 循环,更新触发率统计。

        触发率统计只在 dynamic 模式下需要;其他模式下也可以维护,开销很小。
        """
        if not self._config.get("dynamic", False):
            return
        program = result.generated_program
        if program is None:
            return
        if not (program.meta or {}).get("was_buggy"):
            return
        with self._stats_lock:
            self._total_cnt += 1
            if (program.validity or 0.0) == 1.0:
                self._success_cnt += 1

    def _build_conditional(self, evaluate_fn: Callable):
        """构建 conditional debug 模块,被 apply_to_rollout 延迟调用。"""
        from famou.modules.pipeline import (
            Loop,
            Conditional,
            PassThrough,
            Sequence,
            ModuleList,
        )
        from famou.modules.generate.error_driven import ErrorDrivenGenerate
        from famou.modules.evaluate.base import EvaluateModule
        from famou.core.protocol import Module

        max_iters = self._config.get("max_iterations", 3)
        dynamic = self._config.get("dynamic", False)

        body_modules = [
            ErrorDrivenGenerate(name="debug_generate"),
            EvaluateModule(evaluate_fn=evaluate_fn),
        ]

        debug_loop = Loop(
            body=ModuleList(*body_modules),
            max_iterations=max_iters,
            break_condition=lambda _ctx, result: (
                not result.generated_program or not result.generated_program.is_buggy
            ),
        )

        class _RestoreParentId(Module):
            """修复循环结束后恢复原始 parent_id。

            修复过程中 parent_id 可能临时指向中间产物,但 archive 持久化要求
            parent_id 必须指向真实存在的父程序,因此 commit 前必须恢复。
            同时恢复 selection.parent_id 和 generated_program.parent_id。
            """

            def execute(self, context, result, **kwargs):
                """Restore ``pre_debug_parent_id`` onto both selection and program.

                Reads the saved original parent id from ``result.selection.extra``
                (popping it so subsequent modules see a clean slate) and writes
                it back into both ``result.selection.parent_id`` and
                ``result.generated_program.parent_id`` so the archive lineage
                points to a real, persisted parent program rather than to a
                transient debug-loop intermediate.

                Args:
                    context: Rollout context (unused, present for protocol).
                    result: Rollout result mutated in place.
                    **kwargs: Forwarded extras (unused).

                Returns:
                    The same ``result`` object, mutated in place.
                """
                if result.selection and result.selection.extra:
                    original = result.selection.extra.pop("pre_debug_parent_id", None)
                    if original:
                        result.selection.parent_id = original
                        if result.generated_program:
                            result.generated_program.parent_id = original
                return result

        # 闭包引用 self 来访问触发率统计
        enrichment_ref = self

        def _debug_condition(_ctx, result):
            """Decide whether the conditional debug branch should fire.

            Always returns False if the current program is not buggy. In static
            mode (``dynamic=False``) it returns True for any buggy program. In
            dynamic mode it uses the historical repair success rate to compute
            a trigger probability — after at least 5 observations, the trigger
            rate is ``min(1.0, success_rate + 0.2)``, balancing exploration of
            potential repairs against the cost of failed repair attempts.

            Args:
                _ctx: Pipeline context (unused).
                result: Current rollout result; ``result.generated_program``
                    is inspected for ``is_buggy``.

            Returns:
                True if the repair loop should run, False otherwise.
            """
            if not (result.generated_program and result.generated_program.is_buggy):
                return False
            if not dynamic:
                return True
            with enrichment_ref._stats_lock:
                total = enrichment_ref._total_cnt
                success = enrichment_ref._success_cnt
            if total <= 5:
                return True
            trigger_rate = min(1.0, (success / total) + 0.2)
            return random.random() < trigger_rate

        return Conditional(
            condition=_debug_condition,
            true_branch=Sequence(
                debug_loop,
                _RestoreParentId("_restore_parent_id"),
            ),
            false_branch=PassThrough(),
            name="conditional_debug",
        )
