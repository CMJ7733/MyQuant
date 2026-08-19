"""Failure memory — stage 1.

The first memory type on purpose: it is the only one that needs no LLM, reads
no sealed data, and does not touch the observation space. ``FailureAnalyzer``
already maps raw errors onto a stable 13-member taxonomy, so the "error
signature" a normal RAG would have to learn by embedding free text is simply
``FailureKind``. Everything here is counting.

What it accumulates, per (failure kind, model family):

- how often the failure has been seen, and at which stage
- whether the taxonomy marks it repairable or policy-level
- the deterministic repair hint the judge already produces
- **recovery statistics**: among candidates whose parent exhibited this
  failure, how many produced valid evidence, and via which expert

The recovery half is what makes this worth retrieving rather than just
counting. "shape failures on mlp were repaired 4/6 times, usually by
local_hpo" is actionable; "shape failed 7 times" is not. Both are derived from
lineage the Search Archive already records, so nothing is asserted that the
evidence does not support.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from famou.reliability.experience.index import ExperienceIndex
from famou.reliability.experience.types import (
    EvidenceLevel,
    ExperienceRecord,
    ExperienceType,
    reliability_weight,
)
from famou.reliability.judge import (
    POLICY_LEVEL,
    REPAIRABLE,
    FailureAnalyzer,
    FailureKind,
)
from famou.reliability.types import EvidenceVector, Fidelity


@dataclass
class ObservedOutcome:
    """One committed candidate, in the shape the experience layer needs.

    Deliberately not ``CandidateOutcome``: the barrier builds these and passes
    them in, so the dependency runs barrier -> experience and never back.
    """

    candidate_id: str
    episode_id: str
    model_family: str = "unknown"
    evidence: List[EvidenceVector] = field(default_factory=list)
    parent_ids: List[str] = field(default_factory=list)
    expert: str = "unknown"


#: Minimum seeds before an F2 observation counts as multi-seed evidence.
_MULTISEED_MIN = 3


def failure_experience_id(kind: FailureKind, model_family: str) -> str:
    return f"fail::{kind.value}::{model_family}"


class FailureMemory:
    """Builds and retrieves the (failure kind, model family) aggregates."""

    experience_type = ExperienceType.FAILURE

    def __init__(self, analyzer: Optional[FailureAnalyzer] = None):
        self._analyzer = analyzer or FailureAnalyzer()

    # ------------------------------------------------------------------
    # write path — called from inside the barrier window
    # ------------------------------------------------------------------

    def observe(
        self,
        index: ExperienceIndex,
        outcome: ObservedOutcome,
        *,
        valid_from_state_version: int,
        transition_id: str,
        decision_id: str,
        data_protocol_version: str = "",
        policy_version: str = "",
    ) -> List[str]:
        """Fold one committed candidate into the failure aggregates.

        Returns the experience ids touched. A candidate contributes to at most
        two records: the failure it exhibited, and the failure it repaired.
        """
        touched: List[str] = []
        if not outcome.evidence:
            return touched

        best = outcome.evidence[-1]
        # One judge() call gives both the taxonomy class and the deterministic
        # repair hint. incumbent_rank_ic is deliberately not passed: this
        # memory is about defects, and NO_IMPROVEMENT is a policy signal that
        # already reaches the observation through recent_failure_patterns.
        verdict = self._analyzer.judge(best)
        kind = verdict.kind

        if kind != FailureKind.NONE:
            touched.append(
                self._record_failure(
                    index,
                    outcome,
                    kind=kind,
                    evidence=best,
                    repair_hint=verdict.repair_hint,
                    valid_from_state_version=valid_from_state_version,
                    transition_id=transition_id,
                    decision_id=decision_id,
                    data_protocol_version=data_protocol_version,
                    policy_version=policy_version,
                )
            )
            index.record_candidate_failure(
                outcome.candidate_id,
                failure_kind=kind.value,
                model_family=outcome.model_family,
            )
            # This candidate itself may have been a repair attempt that failed.
            self.note_failed_repair(index, outcome)
            return touched

        # A repair is only meaningful if this candidate actually worked.
        if best.is_valid:
            touched.extend(
                self._record_recovery(
                    index,
                    outcome,
                    valid_from_state_version=valid_from_state_version,
                    transition_id=transition_id,
                    decision_id=decision_id,
                )
            )

        return touched

    # ------------------------------------------------------------------

    def _record_failure(
        self,
        index: ExperienceIndex,
        outcome: ObservedOutcome,
        *,
        kind: FailureKind,
        evidence: EvidenceVector,
        repair_hint: str,
        valid_from_state_version: int,
        transition_id: str,
        decision_id: str,
        data_protocol_version: str,
        policy_version: str,
    ) -> str:
        experience_id = failure_experience_id(kind, outcome.model_family)
        record = index.get(experience_id) or ExperienceRecord(
            experience_id=experience_id,
            experience_type=ExperienceType.FAILURE,
            applicability={
                "failure_kind": kind.value,
                "model_family": outcome.model_family,
            },
            outcome_summary={
                "occurrences": 0,
                "repairable": kind in REPAIRABLE,
                "policy_level": kind in POLICY_LEVEL,
                "repair_hint": repair_hint,
                "failure_stages": {},
                "recovery_attempts": 0,
                "recoveries": 0,
                "recovered_via": {},
            },
            valid_from_state_version=valid_from_state_version,
            created_at=time.time(),
        )

        summary = record.outcome_summary
        summary["occurrences"] = int(summary.get("occurrences", 0)) + 1
        stage = evidence.failure_stage or "unspecified"
        stages = summary.setdefault("failure_stages", {})
        stages[stage] = int(stages.get(stage, 0)) + 1

        _append_capped(record.candidate_ids, outcome.candidate_id)
        _append_capped(record.evidence_ids, evidence.eval_id)
        _append_capped(record.transition_ids, transition_id)
        _append_capped(record.decision_ids, decision_id)
        _append_capped(record.episode_ids, outcome.episode_id)
        if policy_version:
            _append_capped(record.policy_versions, policy_version)
        if data_protocol_version:
            record.data_protocol_version = data_protocol_version

        record.sample_count = int(summary["occurrences"])
        record.evidence_level = _max_level(
            record.evidence_level, _level_of(evidence)
        )
        record.updated_at = time.time()
        record.statement = _render_statement(record)
        record.refresh_weight()
        index.upsert(record)
        return experience_id

    def _record_recovery(
        self,
        index: ExperienceIndex,
        outcome: ObservedOutcome,
        *,
        valid_from_state_version: int,
        transition_id: str,
        decision_id: str,
    ) -> List[str]:
        """Credit this candidate as a repair of whatever its parents failed at."""
        touched: List[str] = []
        for parent_id in outcome.parent_ids:
            parent_failure = index.candidate_failure(parent_id)
            if not parent_failure:
                continue
            kind_value = parent_failure.get("kind")
            family = parent_failure.get("model_family", "unknown")
            if not kind_value:
                continue
            experience_id = f"fail::{kind_value}::{family}"
            record = index.get(experience_id)
            if record is None:
                continue

            summary = record.outcome_summary
            summary["recovery_attempts"] = int(summary.get("recovery_attempts", 0)) + 1
            summary["recoveries"] = int(summary.get("recoveries", 0)) + 1
            via = summary.setdefault("recovered_via", {})
            via[outcome.expert] = int(via.get(outcome.expert, 0)) + 1

            _append_capped(record.transition_ids, transition_id)
            _append_capped(record.decision_ids, decision_id)
            record.updated_at = time.time()
            record.statement = _render_statement(record)
            record.refresh_weight()
            index.upsert(record)
            touched.append(experience_id)
        return touched

    def note_failed_repair(
        self, index: ExperienceIndex, outcome: ObservedOutcome
    ) -> None:
        """A candidate whose parent had failed, and which failed too.

        Counted as an attempt without a recovery, so ``recovery_rate`` is not
        silently optimistic — only counting successes would make every pattern
        look 100% repairable.
        """
        for parent_id in outcome.parent_ids:
            parent_failure = index.candidate_failure(parent_id)
            if not parent_failure or not parent_failure.get("kind"):
                continue
            experience_id = (
                f"fail::{parent_failure['kind']}::"
                f"{parent_failure.get('model_family', 'unknown')}"
            )
            record = index.get(experience_id)
            if record is None:
                continue
            summary = record.outcome_summary
            summary["recovery_attempts"] = int(summary.get("recovery_attempts", 0)) + 1
            record.updated_at = time.time()
            record.statement = _render_statement(record)
            index.upsert(record)

    # ------------------------------------------------------------------
    # read path
    # ------------------------------------------------------------------

    def rank(
        self,
        records: List[ExperienceRecord],
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ExperienceRecord, float]]:
        """Hard-filter on structured keys, then rank. No text matching (E3).

        Ranking is reliability weight first, occurrence count as tie-break, and
        a small bonus for patterns that are known to be repairable — a
        retrievable failure the agent can act on is worth more context budget
        than one that just says "this approach is a dead end".
        """
        filters = filters or {}
        wanted_kind = filters.get("failure_kind")
        wanted_family = filters.get("model_family")

        scored: List[Tuple[ExperienceRecord, float]] = []
        for record in records:
            if record.experience_type != ExperienceType.FAILURE:
                continue
            app = record.applicability
            if wanted_kind and app.get("failure_kind") != wanted_kind:
                continue
            if wanted_family and app.get("model_family") != wanted_family:
                continue

            summary = record.outcome_summary
            score = record.reliability_weight
            score += 0.01 * min(20, int(summary.get("occurrences", 0)))
            if summary.get("repairable"):
                score += 0.05
            scored.append((record, round(score, 6)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

#: Provenance lists are references, but an aggregate seen thousands of times
#: would still grow without bound inside the checkpoint. Keep a bounded sample.
_PROVENANCE_CAP = 25


def _append_capped(target: List[str], value: Optional[str]) -> None:
    if not value or value in target:
        return
    target.append(value)
    if len(target) > _PROVENANCE_CAP:
        del target[0]


def _level_of(evidence: EvidenceVector) -> EvidenceLevel:
    if (
        evidence.fidelity == Fidelity.F2_FULL
        and evidence.rank_ic is not None
        and evidence.rank_ic.n_seeds >= _MULTISEED_MIN
    ):
        return EvidenceLevel.VISIBLE_MULTISEED
    return EvidenceLevel.PROVISIONAL


_LEVEL_ORDER = {
    EvidenceLevel.PROVISIONAL: 0,
    EvidenceLevel.VISIBLE_MULTISEED: 1,
    EvidenceLevel.SEALED_CERTIFIED: 2,
}


def _max_level(a: EvidenceLevel, b: EvidenceLevel) -> EvidenceLevel:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


def _render_statement(record: ExperienceRecord) -> str:
    """Deterministic rendering. No LLM in stage 1 — see the package docstring."""
    app = record.applicability
    summary = record.outcome_summary
    kind = app.get("failure_kind", "unknown")
    family = app.get("model_family", "unknown")
    occurrences = int(summary.get("occurrences", 0))

    stages = summary.get("failure_stages") or {}
    stage_text = ", ".join(
        f"{name} x{count}" for name, count in sorted(stages.items())
    ) or "unspecified"

    parts = [
        f"{family} candidates hit '{kind}' {occurrences}x (stages: {stage_text})."
    ]

    attempts = int(summary.get("recovery_attempts", 0))
    if attempts:
        recoveries = int(summary.get("recoveries", 0))
        rate = recoveries / attempts
        via = summary.get("recovered_via") or {}
        best_expert = max(via, key=via.get) if via else None
        via_text = f", most often via {best_expert}" if best_expert else ""
        parts.append(
            f"Repair attempts {recoveries}/{attempts} succeeded "
            f"({rate:.0%}){via_text}."
        )
    elif summary.get("repairable"):
        parts.append("Marked repairable but never yet repaired.")
    if summary.get("policy_level"):
        parts.append(
            "Policy-level: the action was wrong, not the code — steer away "
            "from this (expert, family) pair rather than patching."
        )

    hint = summary.get("repair_hint")
    if hint:
        parts.append(f"Hint: {hint}")
    return " ".join(parts)
