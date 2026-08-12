"""Sequential execution component.

Provides Sequence for composing modules into linear pipelines.
"""

from typing import List, Optional, Union

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module
from famou.modules.pipeline.base import PipelineComponent


class Sequence(Module):
    """
    Sequential execution of pipeline components.

    Executes components in order, passing the result from one to the next.
    Nested Sequences are automatically flattened.

    Also serves as a general-purpose module container (replaces ModuleList).

    Args:
        *steps: Variable number of pipeline components to execute in sequence

    Example:
        >>> from famou.modules.pipeline import Sequence
        >>> pipeline = Sequence(
        ...     select_module,
        ...     generate_module,
        ...     evaluate_module,
        ... )
        >>> result = pipeline.execute(context, result)
        >>>
        >>> # Also usable as a loop body (replaces ModuleList)
        >>> debug_body = Sequence(debug_module, evaluate_module)
        >>> debug_loop = Cycle(body=debug_body, max_iterations=3)
    """

    def __init__(self, *steps: Union[Module, PipelineComponent], name: Optional[str] = None):
        super().__init__(name or "sequence")
        # Flatten nested Sequences
        self.steps: List[Union[Module, PipelineComponent]] = []
        for step in steps:
            if isinstance(step, Sequence):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        """Execute all steps in sequence."""
        for step in self.steps:
            if hasattr(step, 'execute'):
                result = step.execute(context, result)
            else:
                result = step(context, result)
        return result

    def __len__(self) -> int:
        """Return number of steps."""
        return len(self.steps)

    def __getitem__(self, index: int) -> Union[Module, PipelineComponent]:
        """Get step by index."""
        return self.steps[index]

    def get_children(self) -> list:
        """Return all child modules for recursive dependency injection."""
        return list(self.steps)

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self.steps]
        return f"Sequence({', '.join(step_names)})"


# Backward compatibility aliases
Pipeline = Sequence
ModuleList = Sequence
