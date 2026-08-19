"""Experience guidance — the proposal-side view of retrieved experience.

Stage 2. Retrieval for the *policy* (stage 3) answers "what should I do next";
retrieval for *generation* answers "how should I build this candidate". They
are separate because their consumers differ: the policy needs a bounded
summary it can encode as features, while an expert needs concrete constraints
it can act on.

Constraints, not prose
----------------------
A deterministic expert cannot read a sentence. So guidance carries both:

- ``records``: the raw experience, for LLM-driven experts and for prompts
- ``constraints``: a small structured dict the deterministic experts honour

Only failure memory contributes constraints today, and only where the mapping
is unambiguous — an OOM or TIMEOUT pattern for this family means "do not grow
the model". Anything more speculative belongs in a semantic memory with
supporting cases, not in a hard constraint that silently shrinks the search
space.

Deliberately NOT here: any notion of "this worked, do it again". Copying past
winners is what makes a search collapse onto one lineage, and the reliability
layer exists precisely because a single good visible score is weak evidence.
Guidance steers away from known defects; it does not steer toward past highs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from famou.reliability.experience.types import ExperienceRecord, RetrievalBundle
from famou.reliability.judge import FailureKind

#: Failure kinds whose fix is unambiguously "use a smaller model".
_SIZE_FAILURES = {FailureKind.OOM.value, FailureKind.TIMEOUT.value}

#: A pattern must have been seen at least this often before it constrains
#: generation. One OOM is an accident; three is a property of the family at
#: this scale.
_MIN_OCCURRENCES_FOR_CONSTRAINT = 3


@dataclass
class ExperienceGuidance:
    """What an expert is told about the past. May be empty."""

    records: List[ExperienceRecord] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    bundle_id: Optional[str] = None

    def __bool__(self) -> bool:
        return bool(self.records)

    @property
    def avoid_growth(self) -> bool:
        """Repeated OOM/timeout for this family: mutations should not scale up."""
        return bool(self.constraints.get("avoid_growth"))

    def experience_ids(self) -> List[str]:
        return [r.experience_id for r in self.records]

    def as_prompt_block(self, max_records: int = 4) -> str:
        """Rendering for LLM-driven experts. Statements only — no ids, no
        scores, nothing the model would be tempted to pattern-match on."""
        if not self.records:
            return ""
        lines = [
            f"- {r.statement}" for r in self.records[:max_records] if r.statement
        ]
        if not lines:
            return ""
        return (
            "Lessons from earlier attempts in this run (avoid repeating "
            "these failures):\n" + "\n".join(lines)
        )


def derive_constraints(records: List[ExperienceRecord]) -> Dict[str, Any]:
    """Turn failure patterns into constraints a deterministic expert honours.

    Conservative by construction: a constraint only fires on a pattern seen
    ``_MIN_OCCURRENCES_FOR_CONSTRAINT`` times whose repair direction is not in
    doubt. Everything else stays advisory text.
    """
    constraints: Dict[str, Any] = {}
    reasons: List[str] = []

    for record in records:
        kind = record.applicability.get("failure_kind")
        occurrences = int(record.outcome_summary.get("occurrences", 0))
        if kind in _SIZE_FAILURES and occurrences >= _MIN_OCCURRENCES_FOR_CONSTRAINT:
            constraints["avoid_growth"] = True
            reasons.append(f"{kind} x{occurrences}")

    if reasons:
        constraints["reasons"] = reasons
    return constraints


def build_guidance(
    records: List[ExperienceRecord], *, bundle: Optional[RetrievalBundle] = None
) -> ExperienceGuidance:
    return ExperienceGuidance(
        records=list(records),
        constraints=derive_constraints(records),
        bundle_id=bundle.query_id if bundle is not None else None,
    )
