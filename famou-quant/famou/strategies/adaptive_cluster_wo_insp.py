"""
Adaptive Cluster (without inspirations) — ablation strategy.

Description: Single-variable ablation of adaptive_cluster. Parent selection is
    unchanged (ClusterAdaptiveSelect); inspirations are removed by forcing
    num_inspirations=0 and using a generate module whose prompt omits the
    evolution-history section. Used to isolate the contribution of inspirations.
Author: Famou Framework
Tags: ablation, cluster, no-inspirations
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.cluster_adaptive import ClusterAdaptiveSelect
from famou.modules.generate.mutation_wo_insp import MutationGenerateWoInsp
from famou.modules.evaluate import EvaluateModule
from famou.modules.population.cluster import ClusterPopulation
from famou.modules.judge.llm_judge import LLMJudge


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs,
):
    """
    Create the adaptive-cluster-without-inspirations ablation strategy.

    Identical to adaptive_cluster except:
    - select forces num_inspirations=0, so ClusterAdaptiveSelect.select_inspirations
      returns [] (parent selection logic is untouched)
    - generate uses MutationGenerateWoInsp, whose prompt omits the evolution-history
      / inspirations section entirely

    Args:
        evaluate_fn: Evaluator function to inject into EvaluateModule
        params: Module parameters from config (overrides defaults)

    Good for: Ablation studies measuring the contribution of inspirations.
    """
    params = params or ModulesConfig()

    # Default parameters with config overrides. Force num_inspirations=0 to
    # disable inspirations in ClusterAdaptiveSelect (see cluster_adaptive.py).
    select_params = {**params.select, "num_inspirations": 0}
    generate_params = {**params.generate}
    evaluate_params = {**params.evaluate}
    judge_params = {**params.judge}
    population_params = {**params.population}

    # Build rollout pipeline
    rollout = Rollout(
        modules=[
            ClusterAdaptiveSelect(**select_params),
            MutationGenerateWoInsp(**generate_params),
            evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
            LLMJudge(**judge_params),
        ],
        name="AdaptiveClusterWoInsp",
    )

    population_module = ClusterPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Ablation of adaptive_cluster: no inspirations in select/generate",
        "tags": ["ablation", "cluster", "no-inspirations"],
        "author": "Famou Framework",
    }
