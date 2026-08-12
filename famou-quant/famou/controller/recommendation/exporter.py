"""RecommendationExporter:从 live population 中选出 K 条代表性推荐解。

把推荐解的"候选收集 → 去重 → 排序 → (可选 MMR)选择 → 写文件"完整流水线
封装在一个类里,Evolver 在实验结束时调用 ``.export(output_dir)`` 即可。

选择策略(按优先级):
  1. 候选只来自当前活跃 population(不读历史 archive)
  2. 按 iteration 指纹硬去重(seed 程序按 id 保留)
  3. 全部候选有 embedding feature_vector → MMR 算法平衡分数与多样性
  4. 否则 → 按分数排序取 Top-K
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

from famou.utils.math import cosine_similarity

if TYPE_CHECKING:
    from famou.core.data import Experiment, Program
    from famou.infrastructure.logger import Logger


class RecommendationExporter:
    """从实验的 live population 中选出代表性推荐解并写入 recommend_solution.json。

    无状态(只持有对 experiment / logger 的引用),可在实验结束时一次性
    构造并调用 ``.export()``。

    Selection policy:
        1. Candidates come from the current live population only.
        2. Hard dedupe by iteration fingerprint (seed programs stay distinct).
        3. If all candidates have embeddings, run MMR to balance
           score and diversity.
        4. Otherwise fall back to score-sorted Top-K.
    """

    # 推荐解数量自适应公式中的 magic numbers
    _MIN_K = 5
    _MAX_K = 15
    _K_FACTOR = 10

    # MMR 算法的 score-vs-diversity 权重(0.5 = 平衡)
    _MMR_LAMBDA = 0.5

    def __init__(
        self,
        experiment: "Experiment",
        logger: Optional["Logger"] = None,
    ):
        """Args:
            experiment: 当前实验对象,推荐解从它的 live population 中选
            logger: 可选,用于打印调试信息(目前未使用,留接口)
        """
        self._experiment = experiment
        self._logger = logger

    # ─────────────────────────── 公开 API ───────────────────────────

    def export(self, output_dir: Path) -> None:
        """选出推荐解集合并把 recommend_solution.json 写到 output_dir。

        Args:
            output_dir: 输出目录,必须已存在(调用方负责创建)。

        Side effects:
            在 ``output_dir / 'recommend_solution.json'`` 写一个 JSON 数组,
            每个元素包含 id / code / metrics / iteration_found / generation /
            parent_id。若没有合格候选,写一个空数组。
        """
        chosen = self._select_programs()
        self._write_solutions(chosen, output_dir)

    # ─────────────────────── 选择主流程 ───────────────────────

    def _select_programs(self) -> List["Program"]:
        """主流程:候选收集 → 过滤 → 去重 → 决定算法 → 返回排好序的推荐解列表。

        Returns:
            按推荐策略排序的程序列表;若无合格候选返回空列表。
        """
        live_programs = self._live_candidates()
        scored_programs = [p for p in live_programs if p.combined_score is not None]
        unique_programs = self._dedupe(scored_programs)
        if not unique_programs:
            return []

        target_k = self._target_k(len(unique_programs))
        if len(unique_programs) <= target_k:
            return self._sort_by_score(unique_programs)

        if not self._can_run_mmr(unique_programs):
            return self._sort_by_score(unique_programs)[:target_k]

        return self._select_with_mmr(unique_programs, target_k)

    # ─────────────────── 候选收集与去重 ───────────────────

    def _live_candidates(self) -> List["Program"]:
        """从 experiment 的 island_populations 收集所有 live 程序,按 id 去重。

        Returns:
            所有 island 中当前存活的程序列表(无重复 id)。
        """
        live_by_id: Dict[str, "Program"] = {}
        for program in self._experiment.get_all_programs():
            if program.id not in live_by_id:
                live_by_id[program.id] = program
        return list(live_by_id.values())

    def _dedupe(self, programs: List["Program"]) -> List["Program"]:
        """按 iteration 指纹硬去重,同指纹下保留分数最高者。

        Formal rollouts use a globally unique iteration counter, so iteration
        is a stable cross-island birth fingerprint. Seed programs all use
        iteration=0, so they are kept distinct by id.

        Args:
            programs: 已过滤过 None score 的候选列表。

        Returns:
            去重后的候选列表(指纹唯一)。
        """
        best_by_fingerprint: Dict[str, "Program"] = {}
        for program in programs:
            fp = self._fingerprint(program)
            existing = best_by_fingerprint.get(fp)
            if existing is None or self._score(program) > self._score(existing):
                best_by_fingerprint[fp] = program
        return list(best_by_fingerprint.values())

    def _fingerprint(self, program: "Program") -> str:
        """计算用于硬去重的指纹字符串。

        seed 程序(iteration=0)按 id 区分,其他程序按全局 iteration 编号区分。

        Args:
            program: 待计算指纹的程序。

        Returns:
            形如 ``"seed:<id>"`` 或 ``"iter:<N>"`` 的字符串。
        """
        if program.iteration == 0:
            return f"seed:{program.id}"
        return f"iter:{program.iteration}"

    # ──────────────────── 排序与分数 ────────────────────

    def _sort_by_score(self, programs: List["Program"]) -> List["Program"]:
        """按分数降序排序,并使用 iteration 和 id 做稳定的 tie-break。

        Args:
            programs: 待排序的程序列表。

        Returns:
            新的有序列表,原列表不变。
        """
        return sorted(
            programs,
            key=lambda p: (-self._score(p), p.iteration, p.id),
        )

    def _score(self, program: "Program") -> float:
        """返回用于排序/MMR 的数值分数,None 兜底为 -inf。

        Args:
            program: 待取分的程序。

        Returns:
            程序的 combined_score,若为 None 则返回 ``float('-inf')``。
        """
        return program.combined_score if program.combined_score is not None else float("-inf")

    def _target_k(self, candidate_count: int) -> int:
        """根据候选数量计算自适应的推荐解数 K。

        公式:``K = clamp(candidate_count // 10, 5, 15)``。

        Args:
            candidate_count: 去重后的候选数量。

        Returns:
            推荐解目标数量。
        """
        return max(self._MIN_K, min(self._MAX_K, candidate_count // self._K_FACTOR))

    # ──────────────────── MMR 多样性选择 ────────────────────

    def _can_run_mmr(self, programs: List["Program"]) -> bool:
        """判断所有候选是否都有形状一致的 embedding,以决定是否启用 MMR。

        If any live candidate lacks a usable embedding, degrade to score-only
        Top-K to avoid silently changing the candidate pool semantics.

        Args:
            programs: 候选程序列表。

        Returns:
            True 当且仅当所有程序都有非空且维度一致的 feature_vector。
        """
        expected_dim: Optional[int] = None
        for program in programs:
            vector = program.feature_vector
            if not vector:
                return False
            if expected_dim is None:
                expected_dim = len(vector)
            elif len(vector) != expected_dim:
                return False
        return True

    def _select_with_mmr(
        self,
        programs: List["Program"],
        target_k: int,
    ) -> List["Program"]:
        """运行 score-normalized MMR 在候选程序中选 target_k 条。

        MMR 公式:``score = lambda * relevance - (1 - lambda) * max_similarity``
        其中 ``relevance`` 是归一化后的程序分数,``max_similarity`` 是该候选
        与已选集合中所有程序的最大 cosine 相似度。

        Args:
            programs: 候选程序列表(已通过 _can_run_mmr 校验)。
            target_k: 目标推荐解数量。

        Returns:
            按 MMR 选出的 target_k 条程序;首条是分数最高者,后续按多样性补充。
        """
        sorted_programs = self._sort_by_score(programs)
        raw_scores = [self._score(p) for p in sorted_programs]
        min_score = min(raw_scores)
        max_score = max(raw_scores)

        if max_score > min_score:
            normalized_scores = [
                (s - min_score) / (max_score - min_score) for s in raw_scores
            ]
        else:
            normalized_scores = [0.0 for _ in raw_scores]

        # 第一条:选归一化分数最高的(即原始分数最高)
        selected_indices = [
            max(range(len(sorted_programs)), key=lambda idx: normalized_scores[idx])
        ]
        candidate_indices = [
            idx for idx in range(len(sorted_programs)) if idx not in selected_indices
        ]

        while candidate_indices and len(selected_indices) < target_k:
            best_candidate_idx = candidate_indices[0]
            best_mmr_score = float("-inf")

            for candidate_idx in candidate_indices:
                relevance = normalized_scores[candidate_idx]
                redundancy = max(
                    cosine_similarity(
                        sorted_programs[candidate_idx].feature_vector,
                        sorted_programs[selected_idx].feature_vector,
                    )
                    for selected_idx in selected_indices
                )
                mmr_score = (
                    self._MMR_LAMBDA * relevance
                    - (1 - self._MMR_LAMBDA) * redundancy
                )
                if mmr_score > best_mmr_score:
                    best_candidate_idx = candidate_idx
                    best_mmr_score = mmr_score

            selected_indices.append(best_candidate_idx)
            candidate_indices.remove(best_candidate_idx)

        return [sorted_programs[idx] for idx in selected_indices]

    # ──────────────────── 文件写入 ────────────────────

    def _write_solutions(
        self,
        programs: List["Program"],
        output_dir: Path,
    ) -> None:
        """把推荐解列表序列化为 recommend_solution.json 并写入 output_dir。

        每条 solution 包含 id / code / metrics(自动补 combined_score)/
        iteration_found / generation / parent_id。

        Args:
            programs: 已经选好序的推荐解列表。
            output_dir: 输出目录,必须已存在。

        Side effects:
            写文件 ``output_dir / 'recommend_solution.json'``。
        """
        solutions = []
        for program in programs:
            metrics = dict(program.metrics)
            if program.combined_score is not None and "combined_score" not in metrics:
                metrics["combined_score"] = program.combined_score

            solutions.append({
                "id": program.id,
                "code": program.code,
                "metrics": metrics,
                "iteration_found": program.iteration,
                "generation": program.generation,
                "parent_id": program.parent_id or "",
            })

        recommend_file = output_dir / "recommend_solution.json"
        recommend_file.write_text(
            json.dumps(solutions, indent=4, ensure_ascii=False),
        )
