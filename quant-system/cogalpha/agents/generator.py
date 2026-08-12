"""Task-specific generation: a hierarchy agent proposes alphas.

This is the "Raw(OHLCV) -> 7-L Hierarchy" edge of Figure 1.  Adaptive Generation
(§3.5) enters here as the ``feedback`` block: the analysis of the previous
generation's best and worst alphas is prepended to the prompt, so an agent is
never generating from a blank slate after the first round.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from cogalpha.agents.guidance import GuidanceMode, build_guidance, pick_mode
from cogalpha.agents.hierarchy import AgentSpec, LAYER_DESCRIPTIONS
from cogalpha.agents.parse import parse_alphas
from cogalpha.llm.base import LLMClient, LLMError
from cogalpha.prompts import ALPHA_CONTRACT, GENERATE_PROMPT, SYSTEM_PROMPT
from cogalpha.types import Alpha, EvolutionOp, Lineage


class AlphaGenerator:
    """Wraps one LLM client as the whole hierarchy's generation front-end."""

    def __init__(
        self,
        llm: LLMClient,
        task_temperatures: Sequence[float] = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
        rewrite_guidance_with_llm: bool = False,
        seed: int = 42,
    ) -> None:
        self.llm = llm
        self.task_temperatures = tuple(task_temperatures)
        self.rewrite_guidance_with_llm = rewrite_guidance_with_llm
        self.rng = random.Random(seed)

    def generate(
        self,
        agent: AgentSpec,
        count: int,
        generation: int = 0,
        cycle: int = 0,
        feedback: str = "",
        mode: Optional[GuidanceMode] = None,
        allowed_modes: Sequence[str] = ("light", "moderate", "creative", "divergent", "concrete"),
    ) -> List[Alpha]:
        """Ask ``agent`` for ``count`` alphas.

        Returns whatever parsed cleanly; an empty list is a normal outcome (the
        model may return prose only) and is counted by the caller rather than
        raised, so one bad response cannot abort a 24-generation run.
        """
        guidance_mode = mode or pick_mode(self.rng, allowed_modes)
        guidance = build_guidance(
            agent,
            guidance_mode,
            llm=self.llm,
            rewrite_with_llm=self.rewrite_guidance_with_llm,
        )

        prompt = GENERATE_PROMPT.format(
            count=count,
            agent_name=agent.name,
            level=agent.level,
            layer=agent.layer,
            layer_description=LAYER_DESCRIPTIONS[agent.level],
            guidance=guidance,
            feedback=_feedback_block(feedback),
            contract=ALPHA_CONTRACT,
        )

        temperature = self.rng.choice(self.task_temperatures)
        try:
            response = self.llm.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=temperature,
                tags={
                    "role": "generate",
                    "agent": agent.name,
                    "level": agent.level,
                    "mode": guidance_mode.name,
                    "generation": generation,
                    "cycle": cycle,
                },
            )
        except LLMError:
            # Budget exhaustion and hard transport failures propagate; the loop
            # decides whether to stop the run.
            raise

        lineage = Lineage(
            op=EvolutionOp.HIERARCHY,
            agent=agent.name,
            level=agent.level,
            guidance_mode=guidance_mode.name,
            generation=generation,
            cycle=cycle,
        )
        alphas = parse_alphas(response.text, lineage, max_alphas=count * 2)
        for alpha in alphas:
            alpha.meta["temperature"] = temperature
            alpha.meta["source"] = "hierarchy"
        return alphas


def _feedback_block(feedback: str) -> str:
    """Render the Adaptive Generation feedback, or nothing on the first round."""
    if not feedback.strip():
        return ""
    return (
        "LESSONS FROM THE PREVIOUS GENERATION\n"
        f"{feedback.strip()}\n\n"
        "Use these lessons: pursue what worked, avoid what did not.\n"
    )
