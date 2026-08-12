"""
Enrichment base modules for Famou 2.0.

JudgeModule is a generalized base class for all post-generation enrichment.
Modules that augment programs with computed data inherit from this class:
- LLM feedback (llm_feedback field)
- Feature vectors (feature_vector field)
- Metadata (meta field)
- State updates (context.state)

Originally designed for LLM-based code review ("judging"), now generalized
for any enrichment pattern. Name preserved for backward compatibility.
"""

from abc import abstractmethod

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module


class JudgeModule(Module):
    """
    Base class for enrichment modules that augment generated programs.

    GENERALIZED PURPOSE:
    All modules after EvaluateModule that write to program fields or state
    inherit from JudgeModule. This includes:
    - LLM feedback → program.llm_feedback
    - Feature extraction → program.feature_vector
    - Metadata collection → program.meta[key]
    - Experience collection → context.state
    - Custom analysis → any program field or state

    BEHAVIOR:
    - Reads result.generated_program (enriched by EvaluateModule)
    - Computes data via judge() method
    - Writes to program fields or state in-place
    - Optional module (can be skipped)

    NAMING:
    The name "JudgeModule" is preserved from the original LLM feedback use case.
    The method name judge() reflects this history but now serves as the general
    enrichment entry point. Think of it as "judging what data to add."

    Data Flow:
    - Input: result.generated_program (with metrics/scores from EvaluateModule)
    - Process: Compute enrichment data via judge()
    - Output: Same program enriched in-place (or state updated)

    Subclasses must implement judge() to compute and apply enrichment.

    Examples:
        >>> class LLMFeedback(JudgeModule, RequiresLLM):
        ...     def judge(self, context, result):
        ...         feedback = self.llm_client.generate(...)
        ...         result.generated_program.llm_feedback = feedback
        ...
        >>> class EmbeddingFeature(JudgeModule, RequiresEmbedding):
        ...     def judge(self, context, result):
        ...         embedding = self.embedding_client.embed(result.generated_program.code)
        ...         result.generated_program.feature_vector = embedding
        ...
        >>> class ComplexityMetrics(JudgeModule):
        ...     def judge(self, context, result):
        ...         metrics = compute_complexity(result.generated_program.code)
        ...         result.generated_program.meta['complexity'] = metrics
    """

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that a program is available for enrichment."""
        del context  # Unused but required by Module protocol
        if not result.generated_program:
            raise ValueError(
                f"{self.name}: No program to enrich. "
                "Make sure GenerateModule has run first."
            )

    @abstractmethod
    def judge(self, context: Context, result: RolloutResult) -> None:
        """
        Compute and apply enrichment to the program.

        This is the core method that subclasses must implement.
        Subclasses compute data and write directly to:
        - result.generated_program.llm_feedback
        - result.generated_program.feature_vector
        - result.generated_program.meta[key]
        - context.state (for strategy state)

        Args:
            context: Experiment context (read-only)
            result: Rollout result with generated_program to enrich

        Note:
            This method should NOT return a value. It enriches in-place.
            For backward compatibility with legacy implementations that return
            Dict[str, Any], execute() will handle the return value and write
            it to llm_feedback.
        """
        pass

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Execute the enrichment module on the given program."""
        del kwargs  # Unused but required by Module protocol
        try:
            # Call judge() - may return None (new style) or Dict (legacy style)
            result_data = self.judge(context, result)

            # Backward compatibility: if judge() returns data, write to llm_feedback
            # New implementations should write directly in judge() and return None
            if result_data is not None:
                result.generated_program.llm_feedback = result_data

            return result
        except Exception as e:
            self.log_error(f"Enrichment failed: {e}")
            raise
