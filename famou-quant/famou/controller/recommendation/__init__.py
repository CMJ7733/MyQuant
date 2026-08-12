"""Recommendation 子包:在实验结束时挑选并导出推荐解集合。

主要类:
  RecommendationExporter —— 从 live population 选 K 条代表性解,写入
                             recommend_solution.json。
"""

from famou.controller.recommendation.exporter import RecommendationExporter

__all__ = ["RecommendationExporter"]
