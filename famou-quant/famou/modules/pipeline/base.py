"""Base protocol for pipeline components.

All pipeline components must implement the PipelineComponent protocol.
"""

from typing import Protocol, runtime_checkable

from famou.core.data import Context, RolloutResult


@runtime_checkable
class PipelineComponent(Protocol):
    """
    Protocol for pipeline components.

    Any component that can be used in a Pipeline must implement this protocol
    by providing an `execute` method with the following signature.

    This is a structural type - any class with an `execute` method that matches
    this signature is automatically a PipelineComponent, no inheritance needed.

    Example:
        >>> class MyComponent:
        ...     def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        ...         # Do something
        ...         return result
        >>>
        >>> # MyComponent is now a PipelineComponent
        >>> pipeline = Pipeline(MyComponent(), OtherModule())
    """

    def execute(self, context: Context, result: RolloutResult) -> RolloutResult:
        """
        Execute the pipeline component.

        Args:
            context: Experiment context
            result: Current rollout result

        Returns:
            Updated rollout result
        """
        ...
