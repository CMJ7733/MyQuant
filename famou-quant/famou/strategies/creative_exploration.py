"""
Creative Exploration Strategy

Description: Explores diverse and novel solutions through draft generation
Author: Famou Framework
Tags: exploration, creative, diverse
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.diversity import DiversitySelect
from famou.modules.select.cluster_adaptive import ClusterAdaptiveSelect
from famou.modules.generate.draft import DraftGenerate
from famou.modules.evaluate import EvaluateModule
from famou.modules.population.cluster import ClusterPopulation
from famou.modules.judge.llm_judge import LLMJudge


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
):
    """
    Create the creative exploration strategy.

    This strategy focuses on exploring diverse solution spaces by
    generating novel and distinct approaches across iterations.

    Pipeline:
    - Select diverse programs for broad exploration
    - Generate novel draft solutions (iteration-aware)
    - Evaluate the new solutions
    - Judge solution quality and provide feedback
    - Maintain diverse clusters in population

    Good for:
    - Early exploration phases
    - Problems with multiple viable approaches
    - Avoiding local optima
    - Creative problem solving

    Args:
        evaluate_fn: Evaluator function to inject into EvaluateModule
        params: Module parameters from config (overrides defaults)
    """
    params = params or ModulesConfig()

    # Default parameters with config overrides
    select_params = {**params.select}
    generate_params = {**params.generate}
    evaluate_params = {**params.evaluate}
    judge_params = {**params.judge}
    population_params = {**params.population}

    # Build rollout pipeline
    rollout = Rollout(
        modules=[
                ClusterAdaptiveSelect(**select_params),
                DraftGenerate(**generate_params),
                evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
        name="creative_exploration_rollout",
    )

    population_module = ClusterPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Creative exploration strategy (novel diverse solutions)",
        "tags": ["exploration", "creative", "diverse"],
        "author": "Famou Framework",
    }
