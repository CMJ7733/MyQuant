"""Conditional execution components.

Like if/else statements - execute different branches based on conditions.
"""

from typing import Callable, Optional, Union

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module
from famou.modules.pipeline.base import PipelineComponent

# Type alias for condition functions
ConditionFunc = Callable[[Context, RolloutResult], bool]


class Conditional(Module):
    """
    Conditional execution - execute branch based on condition.

    Like Python's if/else statement - executes one branch when condition
    is True, another (or nothing) when False.

    Args:
        condition: Function that takes (context, result) and returns bool
        true_branch: Component to execute when condition is True
        false_branch: Component to execute when condition is False (default: PassThrough)

    Example:
        >>> from famou.modules.pipeline import Conditional, PassThrough
        >>> # Only debug if program has errors
        >>> conditional = Conditional(
        ...     condition=lambda ctx, result: result.generated_program.is_buggy,
        ...     true_branch=debug_pipeline,
        ...     false_branch=PassThrough(),
        ... )
    """

    def __init__(
        self,
        condition: ConditionFunc,
        true_branch: Union[Module, PipelineComponent],
        false_branch: Optional[Union[Module, PipelineComponent]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name or "conditional")
        self.condition = condition
        self.true_branch = true_branch
        self.false_branch = false_branch or PassThrough()

    def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        """Execute branch based on condition."""
        condition_met = self.condition(context, result)
        branch_name = "true_branch" if condition_met else "false_branch"
        branch = self.true_branch if condition_met else self.false_branch
        self.log_info(
            f"Condition={condition_met}, executing {branch_name}",
            branch=branch_name,
            branch_type=type(branch).__name__,
        )
        return self._execute_branch(branch, context, result)

    def _execute_branch(
        self,
        branch: Union[Module, PipelineComponent],
        context: Context,
        result: RolloutResult,
    ) -> RolloutResult:
        """Execute a single branch."""
        if hasattr(branch, 'execute'):
            return branch.execute(context, result)
        return branch(context, result)

    def get_children(self) -> list:
        """Return all child modules for recursive dependency injection."""
        children = [self.true_branch]
        if self.false_branch:
            children.append(self.false_branch)
        return children

    def __repr__(self) -> str:
        return f"Conditional(condition=..., true_branch={self.true_branch}, false_branch={self.false_branch})"


class PassThrough(Module):
    """
    Placeholder component that does nothing.

    Like nn.Identity in PyTorch - simply returns the input unchanged.

    Useful as:
    - Default branch in Conditional
    - Placeholder in dynamic pipeline construction
    - No-op in optional steps

    Example:
        >>> from famou.modules.pipeline import PassThrough
        >>> no_op = PassThrough()
        >>> result = no_op.execute(context, result)  # result unchanged
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "pass_through")

    def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        """Return result unchanged."""
        return result

    def __repr__(self) -> str:
        return "PassThrough()"


# Alias for compatibility with PyTorch naming
Identity = PassThrough
