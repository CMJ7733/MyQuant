"""
Deprecated Strategy Pattern (Old Style - Dict-based)

⚠️ DEPRECATED: This pattern is still supported for backward compatibility but is
NOT recommended for new strategies. Use the Strategy ABC pattern instead (see
example_strategy.py).

Description: Example of the old dict-based strategy pattern that returns a
single static rollout. This pattern is automatically wrapped by StaticStrategy
in StrategyRegistry, which adds the forward() method that always returns the
same rollout.
"""

from typing import Any, Callable, Dict, Optional

from famou.config.settings import ModulesConfig
from famou.core.protocol import Rollout
from famou.modules.select.elite import EliteSelect
from famou.modules.generate.mutation import MutationGenerate
from famou.modules.evaluate import EvaluateModule
from famou.modules.population.topk import TopKPopulation
from famou.modules.judge.llm_judge import LLMJudge


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
):
    """
    Create a deprecated-style strategy (dict-based with static rollout).

    ⚠️ DEPRECATED: This pattern is supported but not recommended.
    Use Strategy ABC pattern instead (see example_strategy.py).

    This function demonstrates the OLD way of defining strategies:
    - Returns a dict with "rollout" key (not "strategy" key)
    - StrategyRegistry automatically wraps this in StaticStrategy
    - The wrapped strategy's forward() method always returns the same rollout
    - Cannot adapt based on context or history

    Args:
        evaluate_fn: Evaluator function to inject into EvaluateModule
        params: Module parameters from config (overrides defaults)

    Returns:
        Dict with keys:
        - rollout: The static Rollout to execute every iteration
        - population_module: Population management strategy
        - description: Human-readable description
        - tags: Categorization tags
        - author: Author name

    Example usage:
        >>> from famou.strategies import StrategyRegistry
        >>> strategy = StrategyRegistry.get("deprecated_strategy", evaluate_fn=my_eval)
        >>> # strategy is a StaticStrategy instance
        >>> rollout = strategy.forward(ctx, history)  # Always returns same rollout
    """
    params = params or ModulesConfig()

    # Extract parameters from config
    select_params = {**params.select}
    generate_params = {**params.generate}
    evaluate_params = {**params.evaluate}
    judge_params = {**params.judge}
    population_params = {**params.population}

    # Build static rollout pipeline
    # This rollout will be used for EVERY iteration (cannot adapt)
    rollout = Rollout(
        modules=[
            # 1. Select best program
            EliteSelect(**select_params),

            # 2. Mutate selected program
            MutationGenerate(**generate_params),

            # 3. Evaluate new program
            EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params) if evaluate_module is None else evaluate_module,

            # 4. Optional: Get LLM feedback
            LLMJudge(**judge_params),
        ],
        name="deprecated_rollout",
    )

    # Population management (keeps top-K programs)
    population_module = TopKPopulation(**population_params)

    # OLD STYLE: Return dict with "rollout" key
    # StrategyRegistry will wrap this in StaticStrategy
    return {
        "rollout": rollout,  # ← Key indicator of old pattern
        "population_module": population_module,
        "description": (
            "Deprecated dict-based strategy pattern (static rollout). "
            "Use Strategy ABC pattern instead for adaptive strategies."
        ),
        "tags": ["deprecated", "example", "static", "old-style"],
        "author": "Famou Framework",
    }


# ============================================================================
# How StrategyRegistry Processes This
# ============================================================================
"""
When you load this strategy:

1. StrategyRegistry.get("deprecated_strategy", evaluate_fn=my_eval)

2. StrategyRegistry calls create_strategy(evaluate_fn, params)

3. Gets dict with "rollout" key (OLD PATTERN detected)

4. Wraps in StaticStrategy:
   ```python
   strategy = StaticStrategy(
       name="deprecated_strategy",
       rollout=result["rollout"],
       population_module=result["population_module"],
       evaluate_fn=evaluate_fn,
       description=result["description"],
       tags=result["tags"],
       author=result.get("author", ""),
   )
   ```

5. StaticStrategy implements Strategy ABC with forward():
   ```python
   class StaticStrategy:
       def forward(self, ctx, rollout_history):
           return self.rollout  # Always same, ignores parameters
   ```

6. Returns the wrapped strategy

So your dict-based strategy becomes a Strategy ABC object automatically!
"""

# ============================================================================
# Why This Pattern is Limited
# ============================================================================
"""
Example of what you CANNOT do with this pattern:

❌ Cannot adapt based on diversity:
   if ctx.diversity < 0.3:
       return explore_rollout
   else:
       return exploit_rollout

❌ Cannot learn from history:
   success_rate = calculate_success(rollout_history[-20:])
   if success_rate < 0.3:
       return simpler_rollout

❌ Cannot switch phases:
   if ctx.iteration < 10:
       return warmup_rollout
   else:
       return normal_rollout

All of these require the Strategy ABC pattern with a proper forward() method.
See example_strategy.py or basic_test.py for how to implement these patterns.
"""
