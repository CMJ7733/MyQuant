"""Pipeline components for Famou 2.0.

Provides composable components for building complex rollout pipelines,
inspired by PyTorch's nn.Module containers.

Core Components:
- Sequence: Sequential execution (like nn.Sequential)
- ModuleList: List container for modules (like nn.ModuleList)
- Conditional: Conditional branch execution (like if/else)
- PassThrough: Placeholder that does nothing (like nn.Identity)
- Cycle: Repeated execution with break conditions (like for/while)
- WhileCycle: Loop while condition is True (like while)
- CountCycle: Loop fixed number of times (like for _ in range(n))

Example - Building a debug pipeline:
    >>> from famou.modules.pipeline import Sequence, ModuleList, Conditional, PassThrough, Cycle
    >>>
    >>> # Simple debug cycle
    >>> debug_cycle = Cycle(
    ...     body=ModuleList(debug_module, evaluate_module),
    ...     max_iterations=3,
    ...     break_condition=lambda ctx, result: not result.generated_program.is_buggy,
    ... )
    >>>
    >>> # Conditional debug (only if program has errors)
    >>> conditional_debug = Conditional(
    ...     condition=lambda ctx, result: result.generated_program.is_buggy,
    ...     true_branch=debug_cycle,
    ...     false_branch=PassThrough(),
    ... )
    >>>
    >>> # Full pipeline
    >>> full_pipeline = Sequence(
    ...     select_module,
    ...     generate_module,
    ...     evaluate_module,
    ...     conditional_debug,
    ...     judge_module,
    ... )
"""

from famou.modules.pipeline.base import PipelineComponent
from famou.modules.pipeline.sequential import Sequence, Pipeline, ModuleList
from famou.modules.pipeline.conditional import Conditional, PassThrough, Identity
from famou.modules.pipeline.cycle import Cycle, WhileCycle, CountCycle, Loop, While, Repeat

__all__ = [
    # Base
    "PipelineComponent",
    # Sequential
    "Sequence",
    "Pipeline",  # Alias for Sequence
    "ModuleList",
    # Conditional
    "Conditional",
    "PassThrough",
    "Identity",  # Alias for PassThrough
    # Cycle (new names)
    "Cycle",
    "WhileCycle",
    "CountCycle",
    # Backward compatibility (old names)
    "Loop",      # Alias for Cycle
    "While",      # Alias for WhileCycle
    "Repeat",     # Alias for CountCycle
]
