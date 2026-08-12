"""Diversified Guidance: five paraphrasing modes (§3.2, Appendix A.2).

Each exploration direction is expanded into five rewrites so the semantic
coverage of a single agent widens without drifting off its assigned direction.
The mode definitions below are the paper's own; each carries an *instruction* that
is injected into the generation prompt.

Two implementations of a mode are possible: rewrite the guidance text with the
LLM, or attach the mode instruction and let the generator internalise it.  We do
the latter by default (``rewrite_with_llm=False``) because it costs zero extra
calls per generation and produces the same downstream effect — the paper's own
ablation only shows that *having* diversified guidance helps (Agent_E vs Agent_EG
in Table 3), not that a separate rewrite call is required.  The LLM rewrite path
is available for anyone reproducing the exact prompt chain.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from cogalpha.agents.hierarchy import AgentSpec
from cogalpha.llm.base import LLMClient


@dataclass(frozen=True)
class GuidanceMode:
    """One paraphrasing mode: the paper's definition plus the prompt directive.

    ``description`` is quoted from Appendix A.2 and is what the LLM is told the style
    *means*; ``instruction`` is the imperative actually injected into the prompt.
    """

    name: str
    description: str
    instruction: str


MODES: Dict[str, GuidanceMode] = {
    "light": GuidanceMode(
        name="light",
        description=(
            "Minimal rewording that maintains nearly identical meaning while improving "
            "clarity and linguistic fluency; serves as a baseline for consistency "
            "testing across linguistic variations."
        ),
        instruction=(
            "Stay close to the stated direction. Implement the most direct, canonical "
            "reading of it, with no embellishment."
        ),
    ),
    "moderate": GuidanceMode(
        name="moderate",
        description=(
            "Rephrases the content naturally with mild enrichment or stylistic "
            "variation, capturing nuanced semantic differences and testing factor "
            "robustness under slightly altered descriptive framing."
        ),
        instruction=(
            "Vary the framing moderately: keep the analytical focus, but choose a "
            "different natural formulation of it than the most obvious one."
        ),
    ),
    "creative": GuidanceMode(
        name="creative",
        description=(
            "Expressive, research-oriented rewording that adds interpretative depth, "
            "aiming to inspire novel analytical angles or alternative reasoning patterns "
            "that remain aligned with the original domain."
        ),
        instruction=(
            "Add interpretative depth. Name the market mechanism you believe produces "
            "the effect and let that mechanism dictate the construction, even if the "
            "result is less conventional."
        ),
    ),
    "divergent": GuidanceMode(
        name="divergent",
        description=(
            "Exploratory rewrites from new but relevant analytical viewpoints, often "
            "shifting emphasis toward different sub-mechanisms within the same "
            "conceptual framework, encouraging broader hypothesis generation."
        ),
        instruction=(
            "Shift emphasis to a different sub-mechanism inside the same direction. "
            "Deliberately avoid the formulation a practitioner would reach for first."
        ),
    ),
    "concrete": GuidanceMode(
        name="concrete",
        description=(
            "Makes the guidance more specific and implementation-oriented by "
            "introducing measurable quantities such as statistical formulas, ratios, or "
            "example computations, bridging conceptual factor ideas with practical "
            "implementation cues."
        ),
        instruction=(
            "Be concrete and implementation-oriented: commit to explicit windows, "
            "ratios and statistical operations, and state the formula in the docstring "
            "before writing the code."
        ),
    ),
}

DEFAULT_ORDER = ("light", "moderate", "creative", "divergent", "concrete")

_REWRITE_PROMPT = """[ROLE: rewrite]
Rewrite the following alpha-research direction in the '{mode}' style.

Style definition: {description}

Original direction:
{focus}

{probe}

Return only the rewritten direction as a single paragraph. Do not write code.
"""


def get_mode(name: str) -> GuidanceMode:
    """Look up a mode by name; raises KeyError listing the valid names."""
    if name not in MODES:
        raise KeyError(f"unknown guidance mode '{name}'; have {sorted(MODES)}")
    return MODES[name]


def pick_mode(rng: random.Random, allowed: Sequence[str] = DEFAULT_ORDER) -> GuidanceMode:
    """Sample one mode uniformly. Called per generation, so all five get exercised."""
    return get_mode(rng.choice(list(allowed)))


def build_guidance(
    agent: AgentSpec,
    mode: GuidanceMode,
    llm: Optional[LLMClient] = None,
    rewrite_with_llm: bool = False,
) -> str:
    """Produce the guidance paragraph handed to the generator.

    With ``rewrite_with_llm=True`` the direction is paraphrased by the model
    first (one extra call per generation); otherwise the mode instruction is
    appended to the agent's own focus and probe.
    """
    base = f"{agent.focus}\n\n{agent.probe}"

    if rewrite_with_llm and llm is not None:
        prompt = _REWRITE_PROMPT.format(
            mode=mode.name,
            description=mode.description,
            focus=agent.focus,
            probe=agent.probe,
        )
        response = llm.generate(
            prompt=prompt,
            temperature=1.0,
            tags={"role": "rewrite", "agent": agent.name, "mode": mode.name},
        )
        rewritten = response.text.strip()
        if rewritten:
            return f"{rewritten}\n\nStyle directive ({mode.name}): {mode.instruction}"

    return f"{base}\n\nStyle directive ({mode.name}): {mode.instruction}"


def all_variants(
    agent: AgentSpec,
    llm: Optional[LLMClient] = None,
    rewrite_with_llm: bool = False,
    modes: Sequence[str] = DEFAULT_ORDER,
) -> List[str]:
    """All five guidance variants for one agent, in declared order."""
    return [
        build_guidance(agent, get_mode(m), llm=llm, rewrite_with_llm=rewrite_with_llm)
        for m in modes
    ]
