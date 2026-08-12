"""Base class for LLM-backed planning modules."""

from __future__ import annotations

import re
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module, RequiresLLM
from famou.infrastructure.llm.base import LLMClient, get_llm_max_tokens, get_llm_temperature
from famou.utils.trace_utils import build_llm_trace


class PlanningModule(Module, RequiresLLM):
    """
    Base class for LLM-backed planning modules.

    Planning modules generate a text plan that guides downstream generation.
    Unlike GenerateModule, they do not produce a Program.

    The RolloutEngine injects llm_client via RequiresLLM.

    Data flow:
        READS:  ctx.metadata["planner_inputs"]  (written by strategy.forward())
        WRITES: ctx.metadata["plan_context"]["plan_text"]      (intra-rollout)
                ctx.metadata["plan_context"]["planner_trace"]
                StateStore island_state[island_id]["plans"][plan_id]["plan_text"]  (cross-rollout)
                StateStore island_state[island_id]["plans"][plan_id]["planner_trace"]

    Subclasses must implement:
        build_prompt(context, planner_inputs) -> (system, user)

    Subclasses may override:
        post_process(response, context, planner_inputs) -> {"plan_text": str}
    """

    llm_client: LLMClient

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        if "planner_inputs" not in context.metadata:
            raise ValueError(
                f"{self.name}: Missing planner_inputs in context.metadata. "
                "Ensure strategy.forward() writes planner_inputs before dispatching."
            )
        if "plan_context" not in context.metadata:
            raise ValueError(
                f"{self.name}: Missing plan_context in context.metadata."
            )

    @abstractmethod
    def build_prompt(
        self, context: Context, planner_inputs: Dict[str, Any]
    ) -> Tuple[str, str]:
        """Return (system_prompt, user_prompt) for this planning mode."""

    def post_process(
        self,
        response: Any,
        context: Context,
        planner_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Parse raw LLM response into a plan output dict.

        Default implementation extracts <plan>...</plan>.
        Subclasses can override for custom parsing.

        Returns:
            Dict with at least {"plan_text": str}
        """
        text = response.text
        match = re.search(r"<plan>\s*(.*?)\s*</plan>", text, re.DOTALL | re.IGNORECASE)
        if match:
            plan_text = match.group(1).strip()
            if plan_text:
                return {"plan_text": plan_text}
        raise ValueError("missing or empty <plan> element in planner response")

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        """Call LLM → post_process → write to ctx.metadata and StateStore."""
        planner_inputs = context.metadata.get("planner_inputs", {})
        system, user = self.build_prompt(context, planner_inputs)

        response = self.llm_client.generate(
            prompt=user,
            system=system,
            temperature=get_llm_temperature(self.llm_client),
            max_tokens=get_llm_max_tokens(self.llm_client),
        )
        plan_output = self.post_process(response, context, planner_inputs)
        plan_text = plan_output["plan_text"]

        trace = build_llm_trace(
            module_name=self.name,
            system=system,
            prompt=user,
            response=response,
            parsed={"plan_text": plan_text},
        )

        # Intra-rollout: downstream modules read from ctx.metadata
        context.metadata["plan_context"]["plan_text"] = plan_text
        context.metadata["plan_context"]["planner_trace"] = trace

        # Cross-rollout: persist to StateStore via deferred StateUpdateCollector
        plan_id = context.metadata.get("plan_context", {}).get("plan_id")
        if plan_id and result.state_updates:
            result.state_updates.set_island(
                "plans", plan_id, "plan_text",
                value=plan_text,
            )
            result.state_updates.set_island(
                "plans", plan_id, "planner_trace",
                value=trace,
            )

        self.log_info(f"Generated plan ({len(plan_text)} chars)")
        return result
