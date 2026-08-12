"""
Standard Strategy

Description: Standard exploitation strategy used in run_famou.py
Author: Famou Framework
Tags: exploitation, standard, default
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.elite import EliteSelect
from famou.modules.generate.mutation import MutationGenerate
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
    Create the standard strategy.

    This is the default strategy used in run_famou.py:
    - Select best program (k=1) with 2 inspirations for context
    - Mutate the best program
    - Evaluate the new program
    - Judge the program quality
    - Keep top-K programs in population

    Args:
        evaluate_fn: Evaluator function to inject into EvaluateModule
        params: Module parameters from config (overrides defaults)

    Good for: General purpose exploitation, local search, fine-tuning
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
            MutationGenerate(**generate_params),
            evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
            LLMJudge(**judge_params),
        ],
        name="SimpleExperience",
    )

    population_module = ClusterPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Standard exploitation strategy (best parent + mutation)",
        "tags": ["exploitation", "standard", "default"],
        "author": "Famou Framework",
    }
