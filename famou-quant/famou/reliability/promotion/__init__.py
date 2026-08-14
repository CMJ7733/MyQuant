"""Promotion policy and sealed gate.

Two halves of the trust boundary:

- ``PromotionPolicy`` (agent side): decides whether a candidate's *visible*
  evidence justifies spending one sealed query. Pure function of
  EvidenceVector + budget + novelty — fully testable.

- ``SealedGateService`` (sealed side): the ONLY component allowed to touch
  sealed_promotion data. Returns a GateVerdict with NO numeric metrics —
  only verdict + reason_code + a three-level margin band.

- ``BudgetedGate``: wraps the service with token consumption so a
  GateRequest cannot be replayed.
"""

from __future__ import annotations

import hashlib
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from famou.reliability.budget import BudgetLedger
from famou.reliability.types import (
    EvidenceVector,
    Fidelity,
    FrozenSplitManifest,
    GateConfig,
    GateReasonCode,
    GateRequest,
    GateVerdict,
    GateVerdictKind,
    MarginBand,
)


# =============================================================================
# Promotion policy (agent side)
# =============================================================================


class PromotionDecision:
    """Outcome of PromotionPolicy.evaluate()."""

    def __init__(self, action: str, reason: str):
        #: one of: "request_gate" | "more_seeds" | "raise_fidelity" | "skip"
        self.action = action
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover
        return f"PromotionDecision({self.action}: {self.reason})"


class PromotionPolicy:
    """Rule-based promotion policy (heuristic_v0; later replaced by RL policy).

    Decision ladder:
    1. No valid F2 evidence            -> raise_fidelity
    2. Too few seeds / high variance   -> more_seeds
    3. Mean RankIC below incumbent     -> skip
    4. Too similar to certified member -> skip (don't waste sealed queries)
    5. Otherwise                       -> request_gate

    Rule 0 sits above all of them: a candidate that has already been through
    the gate is never re-submitted on the same evidence. Re-submission is the
    single easiest way to burn a per-episode sealed budget, because a REJECTed
    candidate keeps satisfying rules 1-5 forever.
    """

    def __init__(
        self,
        *,
        min_f2_seeds: int = 3,
        max_rank_ic_cv: float = 1.0,
        min_improvement_over_incumbent: float = 0.0,
        min_novelty: float = 0.05,
        max_gate_attempts: int = 1,
    ):
        self.min_f2_seeds = min_f2_seeds
        self.max_rank_ic_cv = max_rank_ic_cv
        self.min_improvement_over_incumbent = min_improvement_over_incumbent
        self.min_novelty = min_novelty
        self.max_gate_attempts = max_gate_attempts

    def evaluate(
        self,
        evidence: List[EvidenceVector],
        *,
        incumbent_rank_ic: Optional[float],
        sealed_queries_remaining: float,
        novelty: Optional[float] = None,
        prior_gate_attempts: int = 0,
    ) -> PromotionDecision:
        if sealed_queries_remaining < 1:
            return PromotionDecision("skip", "no sealed query budget remaining")

        if prior_gate_attempts >= self.max_gate_attempts:
            return PromotionDecision(
                "skip",
                f"already submitted to the gate {prior_gate_attempts}x; "
                "re-submission needs materially new evidence",
            )

        f2 = [e for e in evidence if e.fidelity == Fidelity.F2_FULL and e.is_valid]
        if not f2:
            f1 = [e for e in evidence if e.fidelity == Fidelity.F1_CHEAP and e.is_valid]
            if not f1:
                return PromotionDecision("raise_fidelity", "no valid evidence yet")
            return PromotionDecision("raise_fidelity", "only F1 evidence; need F2")

        # Merge stability evidence across F2 evaluations.
        #
        # Trained candidates: dispersion across SEEDS.
        # Deterministic candidates (formulas): re-running with another seed
        # returns the identical number, so seed dispersion is structurally zero
        # and would sail through any variance check while proving nothing. For
        # those, contiguous SUBPERIODS carry the real question — does it hold
        # up across time, or only in one regime?
        samples: List[float] = []
        deterministic = any(e.complexity.get("deterministic") for e in f2)
        if deterministic:
            for e in f2:
                samples.extend(e.regime_stability.values())
            evidence_unit = "subperiods"
        else:
            for e in f2:
                if e.rank_ic is not None:
                    samples.extend(e.rank_ic.per_seed or [e.rank_ic.mean])
            evidence_unit = "F2 seeds"

        if len(samples) < self.min_f2_seeds:
            return PromotionDecision(
                "more_seeds",
                f"only {len(samples)} {evidence_unit} (< {self.min_f2_seeds})",
            )

        import statistics

        mean_ic = statistics.fmean(samples)
        std_ic = statistics.stdev(samples) if len(samples) > 1 else 0.0
        if mean_ic > 0 and std_ic / abs(mean_ic) > self.max_rank_ic_cv:
            return PromotionDecision(
                "more_seeds",
                f"RankIC CV {std_ic / abs(mean_ic):.2f} across {evidence_unit} "
                "too high",
            )

        # Compare on the candidate's own headline RankIC, not the stability
        # sample: subperiod means and the full-window mean are not the same
        # quantity, and the incumbent is measured on the full window.
        headline = statistics.fmean(
            [e.rank_ic.mean for e in f2 if e.rank_ic is not None] or [mean_ic]
        )
        if incumbent_rank_ic is not None:
            if headline < incumbent_rank_ic + self.min_improvement_over_incumbent:
                return PromotionDecision(
                    "skip",
                    f"mean RankIC {headline:.4f} not above incumbent "
                    f"{incumbent_rank_ic:.4f}",
                )

        eff_novelty = novelty
        if eff_novelty is None:
            eff_novelty = max((e.novelty or 1.0) for e in f2)
        if eff_novelty < self.min_novelty:
            return PromotionDecision(
                "skip", f"novelty {eff_novelty:.3f} below threshold; near-duplicate"
            )

        return PromotionDecision(
            "request_gate",
            f"stable F2 evidence (n={len(samples)} {evidence_unit}, "
            f"mean={headline:.4f})",
        )


# =============================================================================
# Gate request construction
# =============================================================================


def build_gate_request(
    *,
    candidate_id: str,
    candidate_code: str,
    manifest: FrozenSplitManifest,
    ledger: BudgetLedger,
    seed_list: Optional[List[int]] = None,
) -> GateRequest:
    """Freeze the candidate and spend one sealed query (via token issuance).

    After this returns, the candidate code/config must be treated as
    immutable — CertifiedAdmission will reject hash mismatches.
    """
    token = ledger.issue_gate_token(manifest.episode_id)
    return GateRequest(
        request_id=f"gate_req_{uuid.uuid4().hex[:12]}",
        candidate_id=candidate_id,
        episode_id=manifest.episode_id,
        candidate_code_hash=hashlib.sha256(candidate_code.encode("utf-8")).hexdigest(),
        data_contract_hash=manifest.compute_hash(),
        protocol_version=manifest.protocol_version,
        query_token=token,
        seed_list=seed_list or [],
        frozen_at=time.time(),
    )


# =============================================================================
# Sealed gate service (sealed side)
# =============================================================================


class SealedGateService:
    """Independent sealed evaluator.

    ``sealed_eval_fn`` is the only user-supplied hook. Signature::

        sealed_eval_fn(candidate_code, manifest, seed_list) -> {
            "rank_ic_per_seed": [float, ...],
            "incumbent_rank_ic": float,          # sealed incumbent reference
            "regime_improvements": {regime: float},  # optional
        }

    The service converts raw sealed numbers into a discrete verdict. Raw
    numbers NEVER leave this object: no logging of values, no numeric fields
    on the returned GateVerdict.
    """

    def __init__(
        self,
        manifest: FrozenSplitManifest,
        sealed_eval_fn: Callable[..., Dict[str, Any]],
        config: Optional[GateConfig] = None,
    ):
        self.manifest = manifest
        self._eval_fn = sealed_eval_fn
        self.config = config or GateConfig()

    def evaluate(self, request: GateRequest, candidate_code: str) -> GateVerdict:
        # Protocol integrity checks (no data touched yet)
        if request.data_contract_hash != self.manifest.compute_hash():
            return GateVerdict(
                verdict=GateVerdictKind.REJECT,
                reason_code=GateReasonCode.PROTOCOL_VIOLATION,
                margin_band=MarginBand.UNKNOWN,
            )
        code_hash = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()
        if code_hash != request.candidate_code_hash:
            return GateVerdict(
                verdict=GateVerdictKind.REJECT,
                reason_code=GateReasonCode.PROTOCOL_VIOLATION,
                margin_band=MarginBand.UNKNOWN,
            )

        seeds = request.seed_list or self.config.seeds
        try:
            raw = self._eval_fn(candidate_code, self.manifest, seeds)
        except Exception:
            # Deliberately swallow the exception detail: error text can leak
            # information about the sealed split.
            return GateVerdict(
                verdict=GateVerdictKind.INCONCLUSIVE,
                reason_code=GateReasonCode.EVALUATION_ERROR,
                margin_band=MarginBand.UNKNOWN,
            )

        per_seed = [float(v) for v in raw.get("rank_ic_per_seed", [])]
        incumbent = raw.get("incumbent_rank_ic")
        if not per_seed or incumbent is None:
            return GateVerdict(
                verdict=GateVerdictKind.INCONCLUSIVE,
                reason_code=GateReasonCode.INSUFFICIENT_POWER,
                margin_band=MarginBand.UNKNOWN,
            )

        import statistics

        mean_ic = statistics.fmean(per_seed)
        std_ic = statistics.stdev(per_seed) if len(per_seed) > 1 else 0.0
        margin = mean_ic - float(incumbent)

        # Stability gate first: high variance across seeds -> reject
        if mean_ic != 0 and abs(std_ic / mean_ic) > self.config.max_seed_cv:
            return GateVerdict(
                verdict=GateVerdictKind.REJECT,
                reason_code=GateReasonCode.UNSTABLE_ACROSS_SEEDS,
                margin_band=self._band(margin),
            )

        # Regime fragility (optional signal from the sealed evaluator)
        regime_impr = raw.get("regime_improvements") or {}
        if regime_impr:
            frac_pos = sum(1 for v in regime_impr.values() if v >= 0) / len(regime_impr)
            if frac_pos < self.config.min_regimes_positive:
                return GateVerdict(
                    verdict=GateVerdictKind.REJECT,
                    reason_code=GateReasonCode.REGIME_FRAGILE,
                    margin_band=self._band(margin),
                )

        if margin >= self.config.delta_min:
            return GateVerdict(
                verdict=GateVerdictKind.PROMOTE,
                reason_code=GateReasonCode.ROBUST_IMPROVEMENT,
                margin_band=self._band(margin),
            )
        if margin >= 0:
            # Positive but below the powered threshold: cannot distinguish
            # from noise at this sample size.
            return GateVerdict(
                verdict=GateVerdictKind.INCONCLUSIVE,
                reason_code=GateReasonCode.INSUFFICIENT_POWER,
                margin_band=self._band(margin),
            )
        return GateVerdict(
            verdict=GateVerdictKind.REJECT,
            reason_code=GateReasonCode.NO_IMPROVEMENT,
            margin_band=self._band(margin),
        )

    def _band(self, margin: float) -> MarginBand:
        half = self.config.marginal_band_width
        if margin >= self.config.delta_min + half:
            return MarginBand.CLEAR_PASS
        if margin >= self.config.delta_min - half:
            return MarginBand.MARGINAL
        if margin >= 0:
            return MarginBand.MARGINAL
        return MarginBand.CLEAR_FAIL


class BudgetedGate:
    """Wraps SealedGateService with one-time token consumption.

    The token was issued (and the query spent) at GateRequest build time;
    consuming it here prevents replaying the same request for extra looks
    at the sealed split.
    """

    def __init__(self, service: SealedGateService, ledger: BudgetLedger):
        self._service = service
        self._ledger = ledger

    def evaluate(self, request: GateRequest, candidate_code: str) -> GateVerdict:
        if not self._ledger.consume_gate_token(request.query_token):
            return GateVerdict(
                verdict=GateVerdictKind.REJECT,
                reason_code=GateReasonCode.PROTOCOL_VIOLATION,
                margin_band=MarginBand.UNKNOWN,
            )
        return self._service.evaluate(request, candidate_code)
