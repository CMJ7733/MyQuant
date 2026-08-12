"""
Example Strategy Pattern (New Style - Strategy ABC)

✅ RECOMMENDED: This is the recommended pattern for creating new strategies.
Use this as a template when building adaptive, history-aware strategies.

Description: Comprehensive example showing the Strategy ABC pattern with the new
forward(ctx, rollout_history) signature. This pattern enables:
- Adaptive rollout selection based on population state (diversity, fitness, etc.)
- History-aware decision making (success rate, improvement trends, etc.)
- Phase-based strategy switching (warmup, exploration, exploitation, etc.)

This example demonstrates THREE adaptive patterns:

1. **Diversity-based adaptation** (Simple):
   - Low diversity (< 0.3): Exploitation rollout (EliteSelect + MutationGenerate)
   - High diversity (>= 0.3): Exploration rollout (RandomSelect + CrossoverGenerate)

2. **History-aware adaptation** (Advanced - see HistoryAwareStrategy below):
   - Analyzes recent rollout success rate
   - Adapts based on score improvement trends
   - Falls back to simpler approaches when struggling

3. **Phase-based switching** (Advanced):
   - Initial enrichment (iteration 0): Custom enrichment rollout
   - Warmup phase (iterations 1-10): Pure exploration
   - Normal phase (iterations 10+): Adaptive based on diversity/history

## Quick Start

Copy this file and modify the forward() method to implement your strategy:

```python
class MyStrategy(Strategy):
    def __init__(self, evaluate_fn, params):
        self.my_rollout = Rollout([...])
        self.population_module = TopKPopulation()

    def forward(self, ctx, rollout_history):
        # Your decision logic here
        if some_condition(ctx):
            return self.rollout_a
        else:
            return self.rollout_b
```

"""

import random
from typing import Callable, Dict, Any, List, Optional

from famou.config.settings import ModulesConfig
from famou.core.data import RolloutResult
from famou.core.protocol import Strategy, Rollout, Context
from famou.modules.select.elite import EliteSelect
from famou.modules.select.random import RandomSelect
from famou.modules.generate.mutation import MutationGenerate
from famou.modules.generate.crossover import CrossoverGenerate
from famou.modules.evaluate import EvaluateModule
from famou.modules.population.topk import TopKPopulation
from famou.modules.judge.llm_judge import LLMJudge


class BasicTestStrategy(Strategy):
    """
    Example 1: Simple adaptive strategy based on population diversity.

    This is the SIMPLEST example of the Strategy ABC pattern. It demonstrates:
    - How to define multiple rollouts in __init__
    - How to implement forward() to choose between rollouts
    - Adaptive behavior based on ctx.diversity (without using history)

    Decision logic:
    - Low diversity (< 0.3): Exploitation (EliteSelect + MutationGenerate)
    - High diversity (>= 0.3): Exploration (RandomSelect + CrossoverGenerate)

    This is a good starting point for most adaptive strategies. If you need
    history-aware decision making, see HistoryAwareStrategy below.

    Attributes:
        evaluate_fn: Evaluator function for program execution
        population_module: Top-K population management
        exploit_rollout: Exploitation rollout (best parent + mutation)
        explore_rollout: Exploration rollout (random parents + crossover)
    """

    def __init__(
        self,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
        evaluate_module=None,
    ):
        params = params or ModulesConfig()
        self.evaluate_fn = evaluate_fn

        select_params = {**params.select}
        generate_params = {**params.generate}
        evaluate_params = {**params.evaluate}
        judge_params = {**params.judge}
        population_params = {**params.population}

        self.population_module = TopKPopulation(**population_params)

        _evaluate_module = evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params)

        self.exploit_rollout = Rollout(
            modules=[
                EliteSelect(**select_params),
                MutationGenerate(**generate_params),
                _evaluate_module,
                LLMJudge(**judge_params),
            ],
            name="exploit_rollout",
        )

        self.explore_rollout = Rollout(
            modules=[
                RandomSelect(**select_params),
                CrossoverGenerate(**generate_params),
                _evaluate_module,
                LLMJudge(**judge_params),
            ],
            name="explore_rollout",
        )

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        """
        Forward pass to determine which rollout to execute based on population diversity.

        This is a simple example that only uses ctx.diversity for decisions.
        For examples using rollout_history, see protocol.py docstring Pattern 4.

        Adaptive strategy:
        - Low diversity (< 0.3): Use exploration rollout (focus on best programs)
        - High diversity (>= 0.3): Use exploitation rollout (focus on diversity)

        Args:
            ctx: Current context providing diversity metric, iteration, population
            rollout_history: Recent rollout history for this island (available but not used here)

        Returns:
            The rollout pipeline to execute
        """        
        # Simple diversity-based decision (rollout_history available if needed)
        if ctx.diversity < 0.3:
            return self.explore_rollout
        else:
            return self.exploit_rollout


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    Create the example strategy (new style - Strategy ABC pattern).

    ✅ This is the RECOMMENDED pattern for new strategies.

    This follows the new pattern where we return a dict with "strategy" key
    containing the Strategy ABC instance (not "rollout" key like old pattern).

    Args:
        evaluate_fn: Evaluator function to inject into EvaluateModule
        params: Module parameters from config (overrides defaults)
        evaluate_module: Pre-built evaluate module (injected by registry for hybrid mode)

    Returns:
        Dictionary with "strategy" key containing Strategy ABC instance

    Example usage:
        >>> from famou.strategies import StrategyRegistry
        >>> strategy = StrategyRegistry.get("example_strategy", evaluate_fn=my_eval_fn)
        >>> rollout = strategy.forward(ctx, rollout_history)
    """
    strategy = BasicTestStrategy(evaluate_fn=evaluate_fn, params=params, evaluate_module=evaluate_module)

    return {
        "strategy": strategy,  # ← Key difference: "strategy" (not "rollout")
        "description": "Example strategy showing Strategy ABC pattern with forward(ctx, rollout_history)",
        "tags": ["example", "recommended", "adaptive", "new-style"],
        "author": "Famou Framework",
    }
