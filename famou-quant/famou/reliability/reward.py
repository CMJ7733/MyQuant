"""Delayed reward construction for the Agentic RL loop.

The reward deliberately does NOT equal the visible score — that would teach
the policy to overfit visible-dev. Instead it combines:

    + certified-archive gain (sealed PROMOTE with margin band weighting)
    + visible evidence improvement (small weight, discounted by instability)
    - compute / token / sealed-query costs
    - retrieval context cost (stage 4: the retrieval head must pay for what
      it asks for, or it will always ask for the maximum)
    - penalties: invalid candidates, high seed variance, gate rejection

Rewards are *delayed*: ``RewardBuilder.mature()`` is called when the
long-horizon outcome of a transition is known (gate verdict returned, or
the candidate has been fully evaluated at its final fidelity).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from famou.reliability.archives import CertifiedArchive, SearchArchive
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
        w_retrieval_token: float = 2e-5,
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
        # Deliberately ~20x the llm_token weight. Retrieved context is not just
        # tokens: it biases what the expert produces, so an indiscriminate
        # policy that always buys the maximum narrows the search while looking
        # cheap. The weight has to be big enough for "retrieve nothing" to win
        # when memory has nothing useful to say.
        self.w_retrieval_token = w_retrieval_token
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
        reward -= cfg.w_retrieval_token * transition.costs.retrieval_tokens

        store.set_reward(transition.transition_id, reward)
        return reward

    # ------------------------------------------------------------------
    # whole-run maturation
    # ------------------------------------------------------------------

    def mature_all(
        self,
        store: TrajectoryStore,
        *,
        certified_archive: Optional[CertifiedArchive] = None,
        transitions: Optional[List[Transition]] = None,
        overwrite: bool = False,
    ) -> List[Tuple[str, float]]:
        """Fill in the delayed reward for a whole run, in commit order.

        Two things here are deliberate and easy to get wrong.

        **Scope.** Defaults to ``store.session_transitions()``, not
        ``transitions()``. The JSONL is cumulative across runs (that is the
        point — it is the training corpus), but each run starts from a fresh
        StateStore, so this run's archives hold no evidence for a previous
        run's candidates. Scoring those would mark every one of them invalid
        and write a fabricated penalty over real training data.

        **Incumbent.** The bar a candidate is measured against is replayed
        forward rather than read off the final archive. Using the end-state
        incumbent would score early transitions against a bar that did not
        exist when they were taken — lookahead bias, and it would punish the
        very decisions that raised the bar. Replaying keeps the reward a
        function of what was knowable at commit time, which is the same
        property ``state_version`` gives the (s, a, s') boundary.

        Already-rewarded transitions are skipped unless ``overwrite`` is set,
        so calling this twice is a no-op. They still advance the incumbent:
        the bar moved whether or not we are re-scoring that step.

        Returns the (transition_id, reward) pairs it wrote.
        """
        pending = (
            transitions if transitions is not None else store.session_transitions()
        )
        incumbent = self._baseline_incumbent(certified_archive)
        filled: List[Tuple[str, float]] = []

        for transition in pending:
            if transition.reward is None or overwrite:
                reward = self.mature(
                    store, transition, incumbent_rank_ic=incumbent
                )
                filled.append((transition.transition_id, reward))
            incumbent = self._advance_incumbent(
                incumbent, transition, certified_archive
            )

        return filled

    # ------------------------------------------------------------------

    def _best_rank_ic(self, candidate_id: str) -> Optional[float]:
        evidence = self._search.best_evidence(candidate_id)
        if evidence is None or evidence.rank_ic is None:
            return None
        return evidence.rank_ic.mean

    def _baseline_incumbent(
        self, certified: Optional[CertifiedArchive]
    ) -> Optional[float]:
        """The bar before any transition ran: the seeded baselines only."""
        if certified is None:
            return None
        best: Optional[float] = None
        for candidate_id, meta in certified.members().items():
            if meta.get("admission") != "baseline":
                continue
            value = self._best_rank_ic(candidate_id)
            if value is not None:
                best = value if best is None else max(best, value)
        return best

    def _advance_incumbent(
        self,
        incumbent: Optional[float],
        transition: Transition,
        certified: Optional[CertifiedArchive],
    ) -> Optional[float]:
        """Raise the bar once this transition's promotion actually landed.

        A PROMOTE verdict is not enough: ``CertifiedAdmission`` can still
        refuse it (hash drift, unconsumed token, protocol mismatch). Only an
        entry that reached the Certified Archive via ``admission == "gate"``
        moves the incumbent.
        """
        if certified is None or transition.gate_verdict is None:
            return incumbent
        if transition.gate_verdict.verdict != GateVerdictKind.PROMOTE:
            return incumbent

        members = certified.members()
        for candidate_id in transition.candidate_ids:
            meta = members.get(candidate_id)
            if meta is None or meta.get("admission") != "gate":
                continue
            value = self._best_rank_ic(candidate_id)
            if value is not None:
                incumbent = value if incumbent is None else max(incumbent, value)
        return incumbent
