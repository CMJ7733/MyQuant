"""
Model Ensemble Strategy

Description: Creates ensemble solutions from top-performing models
Author: Famou Framework
Tags: ensemble, fusion, machine-learning
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.elite import EliteSelect
from famou.modules.generate.model_fusion import ModelFusionGenerate
from famou.modules.evaluate import EvaluateModule
from famou.modules.population.topk import TopKPopulation
from famou.modules.judge.llm_judge import LLMJudge


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
):
    """
    Create the model ensemble strategy.

    This strategy focuses on creating ensemble solutions by combining
    multiple top-performing models from the population.

    Pipeline:
    - Select top-K programs for context
    - Generate ensemble solution from multiple models
    - Evaluate the new ensemble
    - Judge the ensemble quality
    - Keep top-K programs in population

    Good for:
    - Machine learning tasks
    - Competition tasks where ensembles excel
    - Leveraging diverse model strengths

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
                EliteSelect(**select_params),
                ModelFusionGenerate(**generate_params),
                evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
        name="model_ensemble_rollout",
    )

    population_module = TopKPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Model ensemble strategy (fusion of top models)",
        "tags": ["ensemble", "fusion", "machine-learning"],
        "author": "Famou Framework",
    }
