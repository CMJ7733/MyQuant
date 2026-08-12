"""
Standard Strategy

Description: Standard exploitation strategy used in run_famou.py
Author: Famou Framework
Tags: exploitation, standard, default
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.experience_guided import ExperienceGuidedSelect
from famou.modules.generate.experience_guided import ExperienceGuidedGenerate
from famou.modules.evaluate import EvaluateModule
from famou.modules.population.topk import TopKPopulation
from famou.modules.judge.experience_collector import ExperienceCollector


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
            ExperienceGuidedSelect(**select_params),
            ExperienceGuidedGenerate(**generate_params),
            evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
            ExperienceCollector(**judge_params),

        ],
        name="ExperienceGuidedRollout",
    )

    population_module = TopKPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Simple experience collection and exploitation",
        "tags": ["experience", "simple"],
        "author": "Famou Framework",
    }
