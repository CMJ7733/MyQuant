"""
Initial Baseline Strategy

Description: Generates simple, reliable baseline solutions
Author: Famou Framework
Tags: baseline, simple, initial
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.elite import EliteSelect
from famou.modules.generate.gen_init_feasible import GenInitFeasibleGenerate
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
    Create the initial baseline strategy.

    This strategy focuses on generating simple, correct, and reliable
    baseline solutions that can serve as starting points for evolution.

    Pipeline:
    - Select best program (for improvement)
    - Generate simple feasible baseline
    - Evaluate the new solution
    - Judge solution quality
    - Keep top-K programs in population

    Good for:
    - Creating seed programs
    - Establishing performance baselines
    - Tasks where simple solutions work well
    - Getting started quickly

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
                GenInitFeasibleGenerate(**generate_params),
                evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
        name="initial_baseline_rollout",
    )

    population_module = TopKPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Initial baseline strategy (simple reliable solutions)",
        "tags": ["baseline", "simple", "initial"],
        "author": "Famou Framework",
    }
