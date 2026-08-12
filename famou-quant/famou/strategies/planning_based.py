"""
Planning-Based Strategy

Description: Block-level code evolution with Planner-guided modifications
Author: Famou Framework
Tags: planning, block-evolution, targeted

Pipeline:
1. PlannerBasedSelect: Select parent + propose action (op + target block)
2. BlockTargetGenerate: Generate modified block based on action
3. EvaluateModule: Evaluate modified program
4. PlannerFeedbackJudge: Provide feedback to Planner

Use Cases:
- C++ projects with modular code (ALNS operators, etc.)
- Python modules with clear block structure
- Any code with # BLOCK or // BLOCK markers
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.block_generate import BlockTargetGenerate
from famou.modules.population.topk import TopKPopulation

from famou.modules.planning import (
    PlannerBasedSelect,
    PlannerFeedbackJudge,
    create_planner,
)


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
):
    """
    Create the planning-based strategy.

    This strategy uses a Planner to decide which block to modify:
    - PlannerBasedSelect: Select parent + propose action
    - BlockTargetGenerate: Generate based on action
    - EvaluateModule: Evaluate the modified code
    - PlannerFeedbackJudge: Update Planner with results

    Args:
        evaluate_fn: Evaluator function to inject into EvaluateModule
        params: Module parameters from config (overrides defaults)

    Returns:
        Strategy dict with rollout, population_module, and metadata

    Configuration Example (in YAML):
        modules:
          select:
            planner_type: "llm_v2"
            selection_strategy: "best"
            num_inspirations: 2
            default_blocks: ["A", "B", "C"]
          generate:
            include_inspirations: true
            temperature: 0.8
          evaluate:
            timeout: 30
          judge:
            reward_metric: "combined_score"
          population:
            k: 10
    """
    # Support both dict and ModulesConfig
    if params is None:
        params = {}
    elif hasattr(params, "select"):
        # ModulesConfig object - convert to dict
        params = {
            "select": dict(params.select) if params.select else {},
            "generate": dict(params.generate) if params.generate else {},
            "evaluate": dict(params.evaluate) if params.evaluate else {},
            "judge": dict(params.judge) if params.judge else {},
            "population": dict(params.population) if params.population else {},
            "language": getattr(params, "language", "python"),
        }

    # Extract parameters with defaults
    select_params = {
        "planner_type": "llm_v2",
        "selection_strategy": "best",
        "num_inspirations": 2,
        "default_blocks": ["A", "B", "C"],
        **params.get("select", {}),
    }

    generate_params = {
        "include_inspirations": True,
        "temperature": 0.8,
        **params.get("generate", {}),
    }

    evaluate_params = {**params.get("evaluate", {})}

    judge_params = {
        "reward_metric": "combined_score",
        "use_delta_reward": False,
        **params.get("judge", {}),
    }

    population_params = {
        "k": 10,
        **params.get("population", {}),
    }

    # Add language to select and generate if specified
    if "language" in params:
        select_params.setdefault("language", params["language"])
        generate_params.setdefault("language", params["language"])

    # Create modules
    select_module = PlannerBasedSelect(**select_params)
    generate_module = BlockTargetGenerate(**generate_params)
    evaluate_module = evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params)
    feedback_module = PlannerFeedbackJudge(**judge_params)

    # Wire planner: feedback module gets planner reference from select module
    # The planner is created lazily in select_module, so we pass the reference
    feedback_module._select_module = select_module  # type: ignore

    # Build rollout pipeline
    rollout = Rollout(
        modules=[
                select_module,
                generate_module,
                evaluate_module,
                feedback_module,
            ],
        name="planning_based_rollout",
    )

    population_module = TopKPopulation(**population_params)

    return {
        "rollout": rollout,
        "population_module": population_module,
        "description": "Planning-based block evolution strategy",
        "tags": ["planning", "block-evolution", "targeted"],
        "author": "Famou Framework",
        # Store module references for wiring
        "_modules": {
            "select": select_module,
            "generate": generate_module,
            "evaluate": evaluate_module,
            "feedback": feedback_module,
        },
    }


def wire_planner(strategy_result: Dict[str, Any]) -> None:
    """
    Wire Planner reference between Select and Feedback modules.

    This should be called after the Rollout is created but before execution,
    so that both modules share the same Planner instance.

    Args:
        strategy_result: Result from create_strategy()

    Example:
        >>> strategy = create_strategy(evaluate_fn=my_eval)
        >>> wire_planner(strategy)
        >>> # Now select and feedback modules share the same planner
    """
    modules = strategy_result.get("_modules", {})
    select_module = modules.get("select")
    feedback_module = modules.get("feedback")

    if select_module and feedback_module:
        # The planner is lazily created in PlannerBasedSelect
        # PlannerFeedbackJudge will get the reference after first select
        # This is handled automatically via StateStore

        # Alternative: explicitly wire
        # if select_module._planner:
        #     feedback_module.set_planner(select_module._planner)
        pass
