"""Delayed reward construction for the Agentic RL loop.

The reward deliberately does NOT equal the visible score — that would teach
the policy to overfit visible-dev. Instead it combines:

    + certified-archive gain (sealed PROMOTE with margin band weighting)
    + visible evidence improvement (small weight, discounted by instability)
    - compute / token / sealed-query costs
    - penalties: invalid candidates, high seed variance, gate rejection

Rewards are *delayed*: ``RewardBuilder.mature()`` is called when the
long-horizon outcome of a transition is known (gate verdict returned, or
the candidate has been fully evaluated at its final fidelity).
"""

from __future__ import annotations

from typing import Dict, Optional

from famou.reliability.archives import SearchArchive
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import (
    Fidelity,
    GateVerdictKind,
    MarginBand,
    Transition,
)


class RewardConfig:
    """Weights for the delayed reward (tunable; frozen per experiment)."""

    def __init__(
        self,
        w_promote: float = 1.0,
        w_marginal_pass: float = 0.6,
        w_visible_gain: float = 0.2,
        w_inconclusive: float = -0.1,
        w_reject: float = -0.3,
        w_gpu_second: float = 1e-4,
        w_llm_token: float = 1e-6,
        w_sealed_query: float = 0.05,
        w_invalid: float = 0.2,
        w_instability: float = 0.3,
        instability_cv_threshold: float = 1.0,
    ):
        self.w_promote = w_promote
        self.w_marginal_pass = w_marginal_pass
        self.w_visible_gain = w_visible_gain
        self.w_inconclusive = w_inconclusive
        self.w_reject = w_reject
        self.w_gpu_second = w_gpu_second
        self.w_llm_token = w_llm_token
        self.w_sealed_query = w_sealed_query
        self.w_invalid = w_invalid
        self.w_instability = w_instability
        self.instability_cv_threshold = instability_cv_threshold


class RewardBuilder:
    """Computes delayed rewards and writes them back to the TrajectoryStore."""

    def __init__(
        self,
        search_archive: SearchArchive,
        config: Optional[RewardConfig] = None,
    ):
        self._search = search_archive
        self.config = config or RewardConfig()

    def mature(
        self,
        store: TrajectoryStore,
        transition: Transition,
        *,
        incumbent_rank_ic: Optional[float],
    ) -> float:
        """Compute and record the delayed reward for one transition."""
        cfg = self.config
        reward = 0.0

        # --- outcome component -------------------------------------------
        verdict = transition.gate_verdict
        if verdict is not None:
            if verdict.verdict == GateVerdictKind.PROMOTE:
                reward += (
                    cfg.w_marginal_pass
                    if verdict.margin_band == MarginBand.MARGINAL
                    else cfg.w_promote
                )
            elif verdict.verdict == GateVerdictKind.REJECT:
                reward += cfg.w_reject
            else:
                reward += cfg.w_inconclusive
            reward -= cfg.w_sealed_query * verdict.query_cost

        # --- visible evidence component (small, stability-discounted) -----
        best_gain = 0.0
        unstable = False
        any_invalid = False
        for cid in transition.candidate_ids:
            ev = self._search.best_evidence(cid)
            if ev is None:
                any_invalid = True
                continue
            if ev.fidelity != Fidelity.F2_FULL or ev.rank_ic is None:
                continue
            gain = ev.rank_ic.mean - (incumbent_rank_ic or 0.0)
            best_gain = max(best_gain, gain)
            if ev.rank_ic.mean != 0 and abs(ev.rank_ic.std / ev.rank_ic.mean) > cfg.instability_cv_threshold:
                unstable = True
        reward += cfg.w_visible_gain * best_gain
        if unstable:
            reward -= cfg.w_instability
        if any_invalid:
            reward -= cfg.w_invalid

        # --- cost component ------------------------------------------------
        reward -= cfg.w_gpu_second * transition.costs.gpu_seconds
        reward -= cfg.w_llm_token * transition.costs.llm_tokens

        store.set_reward(transition.transition_id, reward)
        return reward
