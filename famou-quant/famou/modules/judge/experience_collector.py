"""Experience collection module for learning from successful programs.

This module uses LLM to extract reusable insights from successful program
transformations and stores them in StateStore for future generation.

Key concepts:
- Experience: A record of successful improvement with LLM-generated insights
- LLM Analysis: Extracts what changed, why it improved, general patterns
- Storage: Per-island experiences in context.state
- Atomic updates: Thread-safe collection during parallel rollouts
"""

from typing import List, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import RequiresLLM
from famou.infrastructure.llm.base import (
    get_llm_max_retries,
    get_llm_max_tokens,
    get_llm_temperature,
    get_llm_timeout,
)
from famou.modules.judge.base import JudgeModule
from famou.prompts.registry import prompt_registry
from famou.utils.trace_utils import append_named_trace, build_llm_trace


class ExperienceCollector(JudgeModule, RequiresLLM):
    """
    Collects successful program improvements with LLM-generated insights.

    After each rollout, if the generated program shows improvement over its parent,
    uses LLM to analyze the transformation and extract reusable insights:
    - What specific changes were made?
    - Why did these changes improve the score?
    - What general pattern or technique can be extracted?

    Stores structured experience containing:
    - Parent/child code snippets
    - Score improvement
    - LLM-generated insights and patterns
    - Metadata (iteration, generation, timestamp)

    Experiences are stored per-island in StateStore and can be retrieved by
    ExperienceGuidedGenerate to inform future code generation.

    Args:
        name: Module name (default: class name)
        min_improvement: Minimum score improvement to collect (default: 0.01)
        max_experiences: Maximum experiences to store per island (default: 100)
        code_context_length: Deprecated compatibility field; code is no longer truncated
        extract_pattern: Whether to extract general patterns (default: True)

        >>> # Experiences automatically stored in context.state per island
        >>> # Later retrieved by ExperienceGuidedGenerate for informed generation
    """

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        min_improvement: float = 0.01,
        max_experiences: int = 100,
        code_context_length: int = 500,
        extract_pattern: bool = True,
        **kwargs,
    ):
        super().__init__(name)
        self.min_improvement = min_improvement
        self.max_experiences = max_experiences
        self.code_context_length = code_context_length
        self.extract_pattern = extract_pattern

    def judge(self, context: Context, result: RolloutResult) -> None:
        """
        Analyze successful transformation using LLM and collect experience.

        If improvement exceeds threshold, uses LLM to analyze:
        1. What specific changes were made (code diff analysis)
        2. Why these changes improved performance
        3. General reusable pattern (if extract_pattern=True)

        Stores structured experience with LLM insights in context.state.
        Does NOT write to program fields (enriches state, not program).

        Args:
            context: Rollout context with state store
            result: Rollout result with generated program
        """
        program = result.generated_program
        if not program:
            return

        # Get parent for comparison
        parent_id = result.selection.parent_id if result.selection else None
        if not parent_id:
            return

        parent = context.accessor.get_by_id(parent_id)
        if not parent or not parent.combined_score:
            return

        # Calculate improvement
        improvement = program.combined_score - parent.combined_score

        # Only collect if improvement exceeds threshold
        if improvement < self.min_improvement:
            self.log_info(
                f"No experience collected: improvement {improvement:.4f} "
                f"< threshold {self.min_improvement}"
            )
            return

        # Use LLM to analyze the transformation
        insights = self._analyze_transformation_with_llm(
            context=context,
            parent=parent,
            child=program,
            improvement=improvement
        )

        # Create experience record with LLM insights
        experience = {
            # Identification
            "parent_id": parent_id,
            "child_id": program.id,

            # Code context
            "parent_code": parent.code,
            "child_code": program.code,

            # Performance metrics
            "parent_score": parent.combined_score,
            "child_score": program.combined_score,
            "improvement": improvement,

            # LLM-generated insights
            "changes_made": insights.get("changes_made", ""),
            "why_improved": insights.get("why_improved", ""),
            "general_pattern": insights.get("general_pattern", "") if self.extract_pattern else "",

            # Metadata
            "iteration": context.iteration,
            "generation": program.generation,
            "timestamp": context.created_at,
        }

        # Read current experiences from snapshot, modify, and collect update
        def add_experience(experiences: Optional[List]) -> List:
            """Updater function for append with size limit."""
            if experiences is None:
                experiences = []

            # Add new experience at the front (most recent first)
            experiences = [experience] + experiences

            # Limit size to prevent unbounded growth
            if len(experiences) > self.max_experiences:
                experiences = experiences[:self.max_experiences]

            return experiences

        # Read from snapshot, modify locally, write via collector
        current_experiences = context.state.get_island(
            "experiences", default=[]
        ) if context.state else []
        updated = add_experience(current_experiences)

        if result.state_updates:
            result.state_updates.set_island("experiences", value=updated)

        self.log_info(
            f"Collected experience: improvement={improvement:.4f}, "
            f"pattern='{insights.get('general_pattern', 'N/A')[:50]}...', "
            f"total_experiences={len(updated)}"
        )

        # Note: Insights stored in StateStore (context.state), not in program fields

    def _analyze_transformation_with_llm(
        self,
        context: Context,
        parent: Program,
        child: Program,
        improvement: float
    ) -> dict:
        """
        Use LLM to analyze a successful parent→child transformation.

        Extracts structured insights about what changed and why it improved.

        Args:
            context: Rollout context
            parent: Parent program
            child: Child program (with better score)
            improvement: Score improvement (child - parent)

        Returns:
            Dictionary with keys: changes_made, why_improved, general_pattern
        """
        # Build analysis prompt
        prompt = self._build_analysis_prompt(context, parent, child, improvement)

        attempt_traces = []
        last_response = None
        last_error = None
        last_prompt = prompt
        insights = None
        max_attempts = get_llm_max_retries(self.llm_client)
        temperature = get_llm_temperature(self.llm_client)
        max_tokens = get_llm_max_tokens(self.llm_client)
        timeout = get_llm_timeout(self.llm_client)

        for attempt in range(1, max_attempts + 1):
            attempt_prompt = prompt
            if attempt > 1:
                attempt_prompt = (
                    f"{prompt}\n\n"
                    "RETRY INSTRUCTION:\n"
                    "Return exactly the required labeled sections. "
                    "Do not omit CHANGES MADE or WHY IMPROVED."
                )
            last_prompt = attempt_prompt
            response = None
            try:
                response = self.llm_client.generate(
                    prompt=attempt_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                candidate = self._parse_llm_analysis(response.text)
                self._validate_llm_analysis(candidate)
                success_trace = build_llm_trace(
                    module_name=self.name,
                    system=None,
                    prompt=attempt_prompt,
                    response=response,
                    request_extra={
                        "parent_id": parent.id,
                        "child_id": child.id,
                        "improvement": improvement,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                    parsed={"insights": candidate},
                )
                attempt_traces.append(success_trace)
                insights = candidate
                last_response = response
                break
            except Exception as e:
                last_response = response
                last_error = e
                attempt_traces.append(
                    build_llm_trace(
                        module_name=self.name,
                        system=None,
                        prompt=attempt_prompt,
                        response=response,
                        request_extra={
                            "parent_id": parent.id,
                            "child_id": child.id,
                            "improvement": improvement,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "timeout": timeout,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                        },
                        parsed={"insights": None, "parse_error": str(e)},
                        error=e,
                    )
                )

        if insights is not None:
            summary_trace = build_llm_trace(
                module_name=self.name,
                system=None,
                prompt=last_prompt,
                response=last_response,
                request_extra={
                    "parent_id": parent.id,
                    "child_id": child.id,
                    "improvement": improvement,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                    "attempt_count": len(attempt_traces),
                },
                parsed={"insights": insights},
            )
            summary_trace["attempts"] = attempt_traces
            append_named_trace(child, "experience_collector", summary_trace)
        else:
            fallback_trace = build_llm_trace(
                module_name=self.name,
                system=None,
                prompt=last_prompt,
                response=last_response,
                request_extra={
                    "parent_id": parent.id,
                    "child_id": child.id,
                    "improvement": improvement,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timeout": timeout,
                    "attempt_count": len(attempt_traces),
                },
                parsed={"insights": None, "source": "fallback_error"},
                error=last_error,
            )
            fallback_trace["attempts"] = attempt_traces
            append_named_trace(child, "experience_collector", fallback_trace)
            self.log_error(f"LLM analysis failed after {max_attempts} attempts: {last_error}")
            # Fallback: basic insights without LLM
            insights = {
                "changes_made": "Code transformation (LLM analysis failed)",
                "why_improved": f"Score improved by {improvement:.4f}",
                "general_pattern": "Apply similar transformations"
            }

        return insights

    def _build_analysis_prompt(
        self,
        context: Context,
        parent: Program,
        child: Program,
        improvement: float
    ) -> str:
        """Build prompt for LLM to analyze transformation using prompt registry."""
        return prompt_registry.get(
            "experience/analysis.txt",
            task_description=context.task_description,
            parent_code=parent.code,
            child_code=child.code,
            parent_score=parent.combined_score,
            child_score=child.combined_score,
            improvement=improvement,
            extract_pattern=self.extract_pattern,
        )

    def _parse_llm_analysis(self, llm_response: str) -> dict:
        """
        Parse LLM response into structured insights.

        Extracts:
        - changes_made
        - why_improved
        - general_pattern (if present)

        Args:
            llm_response: Raw LLM response text

        Returns:
            Dictionary with extracted insights
        """
        insights = {
            "changes_made": "",
            "why_improved": "",
            "general_pattern": ""
        }

        # Parse structured response
        lines = llm_response.strip().split('\n')
        current_key = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headers
            if line.startswith("CHANGES MADE:"):
                current_key = "changes_made"
                content = line.replace("CHANGES MADE:", "").strip()
                if content:
                    insights[current_key] = content
            elif line.startswith("WHY IMPROVED:"):
                current_key = "why_improved"
                content = line.replace("WHY IMPROVED:", "").strip()
                if content:
                    insights[current_key] = content
            elif line.startswith("GENERAL PATTERN:"):
                current_key = "general_pattern"
                content = line.replace("GENERAL PATTERN:", "").strip()
                if content:
                    insights[current_key] = content
            elif current_key:
                # Continuation of current section
                insights[current_key] += " " + line

        # Clean up
        for key in insights:
            insights[key] = insights[key].strip()

        # Fallback if parsing failed
        if not insights["changes_made"] and not insights["why_improved"]:
            insights["changes_made"] = llm_response

        return insights

    def _validate_llm_analysis(self, insights: dict) -> None:
        """Raise when the parsed analysis is missing required structured sections."""
        missing = []
        if not insights.get("changes_made", "").strip():
            missing.append("changes_made")
        if not insights.get("why_improved", "").strip():
            missing.append("why_improved")
        if self.extract_pattern and not insights.get("general_pattern", "").strip():
            missing.append("general_pattern")
        if missing:
            raise ValueError(f"missing required analysis sections: {missing}")

    def get_experience_summary(self, context: Context) -> dict:
        """
        Get summary statistics of collected experiences.

        Useful for monitoring and debugging.

        Args:
            context: Context with state store

        Returns:
            Dictionary with summary statistics
        """
        experiences = context.state.get_island(
            "experiences",
            default=[]
        ) if context.state else []

        if not experiences:
            return {
                "count": 0,
                "avg_improvement": 0.0,
                "max_improvement": 0.0,
            }

        improvements = [exp["improvement"] for exp in experiences]

        return {
            "count": len(experiences),
            "avg_improvement": sum(improvements) / len(improvements),
            "max_improvement": max(improvements),
            "min_improvement": min(improvements),
            "latest_iteration": experiences[0]["iteration"] if experiences else None,
        }
