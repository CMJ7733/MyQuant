"""Cycle execution components.

Like for/while loops - repeated execution with break conditions.
"""

from typing import Callable, Optional, Union

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module
from famou.modules.pipeline.base import PipelineComponent

# Type alias for condition functions
ConditionFunc = Callable[[Context, RolloutResult], bool]


class Cycle(Module):
    """
    Cycle execution - repeat body until break condition or max iterations.

    Like a for/while loop - executes body repeatedly until:
    - break_condition returns True, or
    - continue_condition returns False, or
    - max_iterations is reached

    Args:
        body: Component to execute repeatedly
        max_iterations: Maximum number of iterations (default: 3)
        break_condition: Stop if this returns True (optional)
        continue_condition: Continue only if this returns True (optional)

    Example:
        >>> from famou.modules.pipeline import Cycle, ModuleList
        >>> # Debug until fixed or max 3 attempts
        >>> debug_cycle = Cycle(
        ...     body=ModuleList(debug_module, evaluate_module),
        ...     max_iterations=3,
        ...     break_condition=lambda ctx, result: not result.generated_program.is_buggy,
        ... )
    """

    def __init__(
        self,
        body: Union[Module, PipelineComponent],
        max_iterations: int = 3,
        break_condition: Optional[ConditionFunc] = None,
        continue_condition: Optional[ConditionFunc] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "cycle")
        self.body = body
        self.max_iterations = max_iterations
        self.break_condition = break_condition
        self.continue_condition = continue_condition

    def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        """Execute body in cycle."""
        for iteration in range(self.max_iterations):
            # Execute body
            result = self._execute_body(context, result)

            # Check break condition
            if self.break_condition and self.break_condition(context, result):
                break

            # Check continue condition (stop if False)
            if self.continue_condition and not self.continue_condition(context, result):
                break

        return result

    def _execute_body(self, context: Context, result: RolloutResult) -> RolloutResult:
        """Execute body component."""
        if hasattr(self.body, 'execute'):
            return self.body.execute(context, result)
        return self.body(context, result)

    def get_children(self) -> list:
        """Return all child modules for recursive dependency injection."""
        return [self.body]

    def __repr__(self) -> str:
        return f"Cycle(max={self.max_iterations}, body={self.body})"


class WhileCycle(Cycle):
    """
    Cycle that continues while condition is True.

    Like Python's while loop - continues executing as long as
    condition remains True, with a safety limit.

    Args:
        condition: Continue while this returns True
        body: Component to execute repeatedly
        max_iterations: Maximum iterations as safety limit (default: 10)

    Example:
        >>> from famou.modules.pipeline import WhileCycle, ModuleList
        >>> # Debug while program is buggy
        >>> debug_while = WhileCycle(
        ...     condition=lambda ctx, result: result.generated_program.is_buggy,
        ...     body=ModuleList(debug_module, evaluate_module),
        ...     max_iterations=5,
        ... )
    """

    def __init__(
        self,
        condition: ConditionFunc,
        body: Union[Module, PipelineComponent],
        max_iterations: int = 10,
        name: Optional[str] = None,
    ):
        super().__init__(
            body=body,
            max_iterations=max_iterations,
            break_condition=lambda ctx, result: not condition(ctx, result),
            name=name or "while_cycle",
        )


class CountCycle(Cycle):
    """
    Cycle that repeats exactly N times.

    Like Python's `for _ in range(n)` - executes body
    a fixed number of times regardless of conditions.

    Args:
        times: Number of times to repeat
        body: Component to execute

    Example:
        >>> from famou.modules.pipeline import CountCycle, ModuleList
        >>> # Try mutation 3 times
        >>> repeat_mutation = CountCycle(
        ...     times=3,
        ...     body=ModuleList(mutation_module, evaluate_module),
        ... )
    """

    def __init__(
        self,
        times: int,
        body: Union[Module, PipelineComponent],
        name: Optional[str] = None,
    ):
        super().__init__(body=body, max_iterations=times, name=name or "count_cycle")


# Backward compatibility aliases
Loop = Cycle
While = WhileCycle
Repeat = CountCycle
