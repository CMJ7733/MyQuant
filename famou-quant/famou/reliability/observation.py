"""AgentObservation: the compressed, read-only view the policy sees.

Built by ``ObservationBuilder`` from Search Archive + Certified Archive +
BudgetLedger + lineage. Deliberately *summarized* (family stats, top-k
evidence, failure patterns) rather than raw dumps, so the observation stays
short enough for an LLM context window or an RL belief encoder.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from famou.reliability.archives import CertifiedArchive, SearchArchive
from famou.reliability.budget import BudgetLedger
from famou.reliability.judge import FailureAnalyzer, FailureKind
from famou.reliability.types import EvidenceVector, Fidelity


class CandidateSummary(BaseModel):
    candidate_id: str
    model_family: str
    certified: bool = False
    best_rank_ic: Optional[float] = None
    rank_ic_std: Optional[float] = None
    n_f2_seeds: int = 0
    highest_fidelity: int = 0
    failure_stage: Optional[str] = None
    novelty: Optional[float] = None
    gate_attempts: int = Field(
        default=0,
        description="sealed queries already spent on this candidate",
    )
    n_evidence: int = Field(default=0, description="evidence items accumulated")
    promotion_checked_at: int = Field(
        default=0,
        description="evidence count when promotion was last judged; equal to "
        "n_evidence means there is nothing new to reconsider",
    )


class AgentObservation(BaseModel):
    """The meta-controller's input. Compressed by construction."""

    state_version: int
    policy_version: str = "heuristic_v0"
    episode_id: str

    certified_candidates: List[CandidateSummary] = Field(default_factory=list)
    incumbent_rank_ic: Optional[float] = Field(
        default=None, description="best certified visible RankIC (reference point)"
    )
    search_archive_summary: Dict[str, Any] = Field(default_factory=dict)
    top_visible_evidence: List[CandidateSummary] = Field(default_factory=list)
    model_family_stats: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    recent_failure_patterns: Dict[str, int] = Field(default_factory=dict)
    remaining_budget: Dict[str, float] = Field(default_factory=dict)
    in_flight_tasks: int = 0
    lineage_summary: Dict[str, Any] = Field(default_factory=dict)

    def digest(self) -> str:
        blob = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ObservationBuilder:
    """Constructs AgentObservation from the reliability stores."""

    def __init__(
        self,
        search_archive: SearchArchive,
        certified_archive: CertifiedArchive,
        ledger: BudgetLedger,
        analyzer: Optional[FailureAnalyzer] = None,
    ):
        self._search = search_archive
        self._certified = certified_archive
        self._ledger = ledger
        # Failure patterns are reported as FailureKind, not raw error strings:
        # the taxonomy is stable across runs, so it can be counted and learned
        # from. See famou.reliability.judge.
        self._analyzer = analyzer or FailureAnalyzer()

    def build(
        self,
        *,
        episode_id: str,
        state_version: int,
        policy_version: str = "heuristic_v0",
        in_flight_tasks: int = 0,
        top_k: int = 8,
        failure_window: int = 50,
    ) -> AgentObservation:
        certified = self._certified.members()
        certified_summaries: List[CandidateSummary] = []
        incumbent: Optional[float] = None
        for cid, meta in certified.items():
            summary = self._summarize(cid, meta, certified=True)
            certified_summaries.append(summary)
            if summary.best_rank_ic is not None:
                incumbent = (
                    summary.best_rank_ic
                    if incumbent is None
                    else max(incumbent, summary.best_rank_ic)
                )

        all_cands = self._search.all_candidates()
        uncertified = [
            (cid, meta) for cid, meta in all_cands.items() if cid not in certified
        ]
        visible_summaries = [
            s for s in (self._summarize(cid, m, certified=False) for cid, m in uncertified)
            if s.best_rank_ic is not None
        ]
        visible_summaries.sort(
            key=lambda s: s.best_rank_ic or float("-inf"), reverse=True
        )

        failures: Dict[str, int] = {}
        recent = list(uncertified)[-failure_window:]
        for cid, _ in recent:
            for ev in self._search.get_evidence(cid):
                kind = self._analyzer.classify(ev)
                if kind != FailureKind.NONE:
                    failures[kind.value] = failures.get(kind.value, 0) + 1

        return AgentObservation(
            state_version=state_version,
            policy_version=policy_version,
            episode_id=episode_id,
            certified_candidates=certified_summaries,
            incumbent_rank_ic=incumbent,
            search_archive_summary={
                "n_candidates": len(all_cands),
                "n_certified": len(certified),
                "families": sorted(
                    {m.get("model_family", "unknown") for m in all_cands.values()}
                ),
            },
            top_visible_evidence=visible_summaries[:top_k],
            model_family_stats=self._search.family_stats(),
            recent_failure_patterns=failures,
            remaining_budget=self._ledger.remaining(episode_id=episode_id),
            in_flight_tasks=in_flight_tasks,
        )

    def _summarize(
        self, candidate_id: str, meta: Dict[str, Any], *, certified: bool
    ) -> CandidateSummary:
        evidence = self._search.get_evidence(candidate_id)
        best = self._search.best_evidence(candidate_id)
        n_f2 = 0
        samples: List[float] = []
        for e in evidence:
            if e.fidelity == Fidelity.F2_FULL and e.rank_ic is not None:
                samples.extend(e.rank_ic.per_seed or [e.rank_ic.mean])
        n_f2 = len(samples)
        import statistics

        return CandidateSummary(
            candidate_id=candidate_id,
            model_family=meta.get("model_family", "unknown"),
            certified=certified,
            best_rank_ic=best.rank_ic.mean if best and best.rank_ic else None,
            rank_ic_std=(statistics.stdev(samples) if len(samples) > 1 else None),
            n_f2_seeds=n_f2,
            highest_fidelity=max((int(e.fidelity.value) for e in evidence), default=0),
            failure_stage=next(
                (e.failure_stage for e in reversed(evidence) if e.failure_stage), None
            ),
            novelty=best.novelty if best else None,
            gate_attempts=int(meta.get("gate_attempts", 0)),
            n_evidence=len(evidence),
            promotion_checked_at=int(meta.get("promotion_checked_at", 0)),
        )
