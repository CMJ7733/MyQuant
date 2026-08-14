"""Judge / Failure Analyzer — turn raw errors into a learnable signal.

An EvidenceVector for a failed candidate carries ``error_info``: a free-text
string like ``ValueError: shape mismatch in cross_sectional_encoder``. That is
useless to a policy. Two runs of the same underlying mistake produce different
strings, so the agent can neither count them nor learn from them.

This module maps those strings onto a small, STABLE taxonomy. Stability is the
whole point: ``FailureKind`` values are what land in
``AgentObservation.recent_failure_patterns`` and become features for the RL
encoder, so adding or renaming one changes the observation space and
invalidates existing policy checkpoints. Treat the enum as part of the frozen
protocol.

Scope, per the architecture: the judge produces feedback and classification.
It does NOT decide admission — only a sealed-gate PROMOTE routed through
``CertifiedAdmission`` puts anything in the Certified Archive. A judge that
could promote would be a second, unaudited trust path.

The rule-based classifier is deterministic and needs no LLM. ``llm_review_fn``
is an optional hook for a natural-language repair suggestion; it runs only for
candidates that actually failed, and its output is advisory text attached to
the evidence, never a score.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from famou.reliability.types import EvidenceVector, Fidelity


class FailureKind(str, Enum):
    """Stable failure taxonomy. Part of the observation space — see module doc."""

    NONE = "none"                       # succeeded
    SYNTAX = "syntax"                   # does not parse
    CONTRACT = "contract"               # missing HYPERPARAMS / FAMOU_RESULT / --split-config
    FORBIDDEN_EDIT = "forbidden_edit"   # touched a protected protocol symbol
    LEAKAGE = "leakage"                 # future-reference idiom in a candidate expression
    DEPENDENCY = "dependency"           # import outside the TaskSpec allow-list
    SHAPE = "shape"                     # tensor/frame shape mismatch
    NAN_OUTPUT = "nan_output"           # produced NaN/inf predictions
    DEGENERATE = "degenerate"           # ran fine but predictions carry no signal
    DATA = "data"                       # dataset/segment construction failed
    TIMEOUT = "timeout"
    OOM = "oom"
    CRASH = "crash"                     # non-zero exit with no better classification
    NO_IMPROVEMENT = "no_improvement"   # valid, but not better than the incumbent
    UNKNOWN = "unknown"


#: Which failures are worth a repair attempt (Debug expert) vs. which mean the
#: candidate family/approach itself was a dead end. Repairing a leakage or a
#: forbidden edit is pointless — the proposal was invalid by construction, and
#: the policy should learn to stop producing it rather than patch it.
REPAIRABLE = {
    FailureKind.SYNTAX,
    FailureKind.CONTRACT,
    FailureKind.SHAPE,
    FailureKind.NAN_OUTPUT,
    FailureKind.CRASH,
    FailureKind.DEPENDENCY,
}

#: Failures that indicate the *action* was wrong, not the code. These should
#: push the policy away from the (expert, family) pair that produced them.
POLICY_LEVEL = {
    FailureKind.FORBIDDEN_EDIT,
    FailureKind.LEAKAGE,
    FailureKind.DEGENERATE,
    FailureKind.NO_IMPROVEMENT,
}


_PATTERNS: List[tuple] = [
    # (regex, kind) — first match wins, so order matters: specific before broad.
    (r"syntaxerror|invalid syntax|unexpected indent", FailureKind.SYNTAX),
    (r"missing hyperparams|missing famou_result|missing --split-config",
     FailureKind.CONTRACT),
    (r"protected name|forbidden", FailureKind.FORBIDDEN_EDIT),
    (r"future-reference|lookahead|leak", FailureKind.LEAKAGE),
    (r"allow-list|modulenotfounderror|importerror|no module named",
     FailureKind.DEPENDENCY),
    (r"out of memory|oom|cannot allocate|memoryerror", FailureKind.OOM),
    (r"timeout|timed out|exceeded \d+s", FailureKind.TIMEOUT),
    (r"shape|dimension|size mismatch|broadcast|reshape", FailureKind.SHAPE),
    (r"nan|inf\b|not finite", FailureKind.NAN_OUTPUT),
    (r"no valid ic days|empty|no data|calendar|segment .* trading days",
     FailureKind.DATA),
]


class JudgeVerdict(BaseModel):
    """Diagnosis of one candidate. Advisory: never gates admission."""

    candidate_id: str
    kind: FailureKind
    repairable: bool = False
    policy_level: bool = Field(
        default=False,
        description="the action was wrong, not the code: steer the policy away",
    )
    summary: str = ""
    repair_hint: str = Field(
        default="", description="actionable suggestion for a Debug expert"
    )
    llm_feedback: str = Field(
        default="", description="optional natural-language review; advisory only"
    )
    signals: Dict[str, Any] = Field(default_factory=dict)


#: Below this, a run that "succeeded" carries no usable signal. A model whose
#: cross-sectional predictions are near-constant produces a RankIC that is
#: numerically fine and completely meaningless, and without this check it would
#: enter the archive as a valid low-scoring candidate rather than a failure.
DEGENERATE_ABS_IC = 0.002


class FailureAnalyzer:
    """Rule-based classifier over EvidenceVector. Deterministic and testable."""

    def __init__(
        self,
        *,
        degenerate_abs_ic: float = DEGENERATE_ABS_IC,
        llm_review_fn: Optional[Callable[[str, JudgeVerdict], str]] = None,
    ):
        self.degenerate_abs_ic = degenerate_abs_ic
        self._llm_review_fn = llm_review_fn

    # ------------------------------------------------------------------

    def classify(self, evidence: EvidenceVector) -> FailureKind:
        """Map one EvidenceVector onto the taxonomy."""
        static = evidence.static_check
        # Static findings are authoritative: they are structured, whereas
        # error_info is a string we are pattern-matching against.
        if static is not None:
            if not static.compiles:
                return FailureKind.SYNTAX
            if static.forbidden_edits:
                return FailureKind.FORBIDDEN_EDIT
            if static.leakage_flags:
                return FailureKind.LEAKAGE
            if static.schema_errors:
                joined = " ".join(static.schema_errors).lower()
                if "allow-list" in joined:
                    return FailureKind.DEPENDENCY
                return FailureKind.CONTRACT

        stage = (evidence.failure_stage or "").lower()
        if stage == "timeout":
            return FailureKind.TIMEOUT
        if stage == "oom":
            return FailureKind.OOM

        text = (evidence.error_info or "").lower()
        if text:
            for pattern, kind in _PATTERNS:
                if re.search(pattern, text):
                    return kind
            return FailureKind.CRASH if stage else FailureKind.UNKNOWN

        if evidence.validity < 1.0 or stage:
            return FailureKind.CRASH

        # Ran clean — but is there any signal?
        if evidence.rank_ic is not None:
            if abs(evidence.rank_ic.mean) < self.degenerate_abs_ic:
                return FailureKind.DEGENERATE
        return FailureKind.NONE

    def judge(
        self,
        evidence: EvidenceVector,
        *,
        incumbent_rank_ic: Optional[float] = None,
        candidate_code: Optional[str] = None,
    ) -> JudgeVerdict:
        """Classify and, when useful, say what to do about it."""
        kind = self.classify(evidence)

        # A clean run that simply lost to the incumbent is not a defect, but it
        # IS a distinct outcome the policy should be able to count.
        if (
            kind == FailureKind.NONE
            and incumbent_rank_ic is not None
            and evidence.rank_ic is not None
            and evidence.rank_ic.mean <= incumbent_rank_ic
        ):
            kind = FailureKind.NO_IMPROVEMENT

        verdict = JudgeVerdict(
            candidate_id=evidence.candidate_id,
            kind=kind,
            repairable=kind in REPAIRABLE,
            policy_level=kind in POLICY_LEVEL,
            summary=self._summarise(kind, evidence),
            repair_hint=self._hint(kind, evidence),
            signals=self._signals(evidence, incumbent_rank_ic),
        )
        if self._llm_review_fn is not None and kind != FailureKind.NONE and candidate_code:
            try:
                verdict.llm_feedback = self._llm_review_fn(candidate_code, verdict)
            except Exception as e:  # advisory only — never fail the pipeline
                verdict.llm_feedback = f"(llm review unavailable: {type(e).__name__})"
        return verdict

    # ------------------------------------------------------------------

    @staticmethod
    def _summarise(kind: FailureKind, evidence: EvidenceVector) -> str:
        if kind == FailureKind.NONE:
            ic = evidence.rank_ic.mean if evidence.rank_ic else float("nan")
            return f"valid at F{int(evidence.fidelity.value)}, RankIC {ic:.4f}"
        detail = (evidence.error_info or "").strip().splitlines()
        head = detail[0][:160] if detail else "no error text"
        return f"{kind.value}: {head}"

    @staticmethod
    def _hint(kind: FailureKind, evidence: EvidenceVector) -> str:
        hints = {
            FailureKind.SYNTAX:
                "fix the parse error; the candidate never ran",
            FailureKind.CONTRACT:
                "restore the candidate contract: HYPERPARAMS assignment, a "
                "--split-config argument, and exactly one FAMOU_RESULT line",
            FailureKind.FORBIDDEN_EDIT:
                "do not assign to protocol symbols (splits, costs, backtester); "
                "vary the model and its training config instead",
            FailureKind.LEAKAGE:
                "remove the forward-looking expression; only the frozen label "
                "may reference future bars",
            FailureKind.DEPENDENCY:
                "use only packages in the TaskSpec allow-list",
            FailureKind.SHAPE:
                "align the feature matrix with the model's expected input width",
            FailureKind.NAN_OUTPUT:
                "guard against NaN/inf in features and predictions",
            FailureKind.DEGENERATE:
                "predictions are near-constant across the cross-section; the "
                "model is not differentiating names",
            FailureKind.DATA:
                "the segment produced no usable rows; check the window length",
            FailureKind.TIMEOUT:
                "reduce model size or training rounds to fit the compute budget",
            FailureKind.OOM:
                "reduce batch size / num_leaves to fit memory",
            FailureKind.NO_IMPROVEMENT:
                "valid but not better than the incumbent; try a different "
                "family or a larger edit",
        }
        return hints.get(kind, "")

    @staticmethod
    def _signals(
        evidence: EvidenceVector, incumbent_rank_ic: Optional[float]
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "fidelity": int(evidence.fidelity.value),
            "validity": evidence.validity,
            "wall_seconds": evidence.cost.wall_seconds,
        }
        if evidence.rank_ic is not None:
            out["rank_ic"] = evidence.rank_ic.mean
            out["rank_ic_std"] = evidence.rank_ic.std
            out["n_seeds"] = evidence.rank_ic.n_seeds
            if incumbent_rank_ic is not None:
                out["gain_vs_incumbent"] = evidence.rank_ic.mean - incumbent_rank_ic
        if evidence.icir is not None:
            out["icir"] = evidence.icir.mean
        return out


def summarise_failures(verdicts: List[JudgeVerdict]) -> Dict[str, int]:
    """Counts per FailureKind — the shape AgentObservation wants."""
    counts: Dict[str, int] = {}
    for v in verdicts:
        if v.kind == FailureKind.NONE:
            continue
        counts[v.kind.value] = counts.get(v.kind.value, 0) + 1
    return counts
