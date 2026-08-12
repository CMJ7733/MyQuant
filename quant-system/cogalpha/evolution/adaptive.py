"""Adaptive Generation (§3.5): learn from the previous generation's extremes.

After each fitness evaluation there are two populations — valid alphas and invalid
ones.  The paper randomly selects two valid and the two worst-performing invalid
alphas per generation, has each analysed and summarised to explain why it is valid
or invalid, and folds the combined fitness results and analytical summaries into
the next generation's prompts.

The invalid samples are the load-bearing half.  Telling a model "this worked" only
narrows it toward what it already produced; telling it "this failed, and here is
the reason" removes a region of the search space.  So a failure that never reached
scoring — rejected by the checker for leakage, or for being constant — is included
alongside the scored-but-weak, with its rejection reason attached.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from cogalpha.config import EvolutionConfig, FitnessConfig
from cogalpha.evolution.operators import format_metrics
from cogalpha.fitness.thresholds import worst_invalid
from cogalpha.llm.base import LLMClient, LLMError
from cogalpha.prompts import ANALYSE_PROMPT, SYSTEM_PROMPT
from cogalpha.types import Alpha, AlphaTier

#: How much of an alpha's source to include in a feedback block.  Full source for
#: four alphas would crowd out the contract in a 4096-token budget.
_CODE_CHARS = 900


@dataclass
class Feedback:
    """The text folded into the next generation's prompts, plus its provenance."""

    text: str = ""
    valid_ids: List[str] = None  # type: ignore[assignment]
    invalid_ids: List[str] = None  # type: ignore[assignment]
    llm_calls: int = 0

    def __post_init__(self) -> None:
        if self.valid_ids is None:
            self.valid_ids = []
        if self.invalid_ids is None:
            self.invalid_ids = []

    def to_dict(self) -> dict:
        """Feedback text plus which alphas it was derived from, for the archive."""
        return {
            "text": self.text,
            "valid_ids": list(self.valid_ids),
            "invalid_ids": list(self.invalid_ids),
            "llm_calls": self.llm_calls,
        }


class AdaptiveGeneration:
    """Builds the per-generation feedback block."""

    def __init__(
        self,
        llm: LLMClient,
        cfg: EvolutionConfig,
        fitness_cfg: FitnessConfig,
        temperature: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.llm = llm
        self.cfg = cfg
        self.fitness_cfg = fitness_cfg
        self.temperature = temperature
        self.rng = random.Random(seed)

    def build(
        self,
        scored: Sequence[Alpha],
        rejected: Sequence[Alpha],
        generation: int,
        analyse: bool = True,
        agent: Optional[str] = None,
    ) -> Feedback:
        """Sample guiding alphas and summarise why they succeeded or failed.

        ``scored`` are alphas that reached fitness evaluation; ``rejected`` are the
        ones the quality checker discarded.  Both are needed: the invalid
        population spans failures of *both* kinds.
        """
        valid_pool = [
            a for a in scored
            if a.tier in (AlphaTier.QUALIFIED, AlphaTier.ELITE) and a.fitness is not None
        ]
        # Random selection among the valid, per §3.5 -- not the top two. Always
        # feeding back the best would narrow the prompt toward one lineage; the
        # paper's wording ("randomly select two valid alphas") avoids that.
        n_valid = min(self.cfg.adaptive_valid_samples, len(valid_pool))
        valid = self.rng.sample(valid_pool, n_valid) if n_valid else []

        invalid = worst_invalid(
            list(scored) + list(rejected),
            n=self.cfg.adaptive_invalid_samples,
            cfg=self.fitness_cfg,
        )

        if not valid and not invalid:
            return Feedback()

        valid_block = _render(valid, kind="valid") or "(none this generation)"
        invalid_block = _render(invalid, kind="invalid") or "(none this generation)"

        summary = ""
        calls = 0
        if analyse:
            prompt = ANALYSE_PROMPT.format(
                valid_block=valid_block,
                invalid_block=invalid_block,
            )
            try:
                response = self.llm.generate(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    tags={"role": "analyse", "agent": agent, "generation": generation},
                )
                summary = response.text.strip()
                calls = 1
            except LLMError:
                # Losing the summary degrades feedback to the raw measurements,
                # which is still usable; it must not end the run.
                summary = ""

        text = _compose(valid_block, invalid_block, summary)
        return Feedback(
            text=text,
            valid_ids=[a.alpha_id for a in valid],
            invalid_ids=[a.alpha_id for a in invalid],
            llm_calls=calls,
        )


def _render(alphas: Sequence[Alpha], kind: str) -> str:
    """Render sampled alphas with their measurements or rejection reason."""
    parts: List[str] = []
    for alpha in alphas:
        header = f"[{alpha.name}]"
        if alpha.fitness is not None:
            header += f" {format_metrics(alpha.fitness)}"
        if alpha.rejected_at is not None:
            header += (
                f" REJECTED at {alpha.rejected_at.value}: "
                f"{' '.join(alpha.reject_reason.split())[:240]}"
            )
        elif kind == "invalid" and alpha.fitness is not None:
            header += " (executed cleanly but below the qualifying threshold)"

        code = alpha.code.strip()
        if len(code) > _CODE_CHARS:
            code = code[:_CODE_CHARS] + "\n    # ...truncated"
        parts.append(f"{header}\n```python\n{code}\n```")
    return "\n\n".join(parts)


def _compose(valid_block: str, invalid_block: str, summary: str) -> str:
    """Assemble the feedback text, summary first so it survives truncation."""
    sections: List[str] = []
    if summary:
        sections.append(f"What the last generation taught us:\n{summary}")
    sections.append(f"Alphas that worked:\n{valid_block}")
    sections.append(f"Alphas that did not:\n{invalid_block}")
    return "\n\n".join(sections)
