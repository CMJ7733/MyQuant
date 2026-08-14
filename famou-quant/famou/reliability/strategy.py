"""ReliabilityAwareQuantStrategy — the meta-controller.

Implements the famou ``Strategy`` protocol: ``forward(ctx, history)`` is
called per iteration and returns a Rollout. Internally it:

1. builds an AgentObservation from Search/Certified archives + budget,
2. picks a StructuredAction with a rule-based policy (heuristic_v0 —
   the BC/offline-RL policy plugs in later via ``policy_fn``),
3. records a DecisionRecord in the TrajectoryStore,
4. translates the action into a candidate via a ProposalExpert,
5. evaluates it through the multi-fidelity evaluator (F0 always; F1/F2
   according to the action), writing EvidenceVectors into the Search
   Archive,
6. runs the PromotionPolicy and, when warranted, spends a sealed query
   through the BudgetedGate and routes PROMOTE verdicts through
   CertifiedAdmission into the Certified Archive,
7. appends a Transition to the TrajectoryStore.

The rollout returned to the Evolver is a *materialized* candidate: the
generation already happened inside forward(), so the rollout's generate
module is a trivial replay (PreGeneratedGenerate) and the famou
EvaluateModule re-scores it with the user evaluator for framework-level
compat (population, logging, checkpoints). The reliability evidence —
not the scalar combined_score — is what drives promotion.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData
from famou.core.protocol import Module, Rollout, Strategy
from famou.core.state import StateStore
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.base import GenerateModule
from famou.modules.population.full_archive import FullArchivePopulation
from famou.reliability.archives import (
    CertifiedAdmission,
    CertifiedArchive,
    SearchArchive,
)
from famou.reliability.budget import BudgetExhausted, BudgetLedger
from famou.reliability.evaluator import FidelityEvaluator
from famou.reliability.experts import ProposalExpert, default_expert_registry
from famou.reliability.observation import AgentObservation, ObservationBuilder
from famou.reliability.promotion import (
    BudgetedGate,
    PromotionPolicy,
    build_gate_request,
)
from famou.reliability.trajectory import TrajectoryStore, build_transition
from famou.reliability.types import (
    DecisionRecord,
    EvaluationCost,
    ExpertKind,
    Fidelity,
    FrozenSplitManifest,
    GateVerdictKind,
    StructuredAction,
)


# =============================================================================
# Trivial modules for the materialized rollout
# =============================================================================
#
# These subclass the framework's GenerateModule/EvaluateModule so the
# Rollout pipeline validator (isinstance checks) accepts them, but their
# execute() is a replay: the candidate was already produced and scored
# inside forward(), so no LLM call or heavy evaluation happens in the worker.


class _FixedSelect(Module):
    """Selects the parent recorded in the action (no search)."""

    def __init__(self, parent_id: str):
        super().__init__()
        self._parent_id = parent_id

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        result.selection = SelectionData(parent_id=self._parent_id)
        return result


class _PreGeneratedGenerate(GenerateModule):
    """Replays an already-generated candidate (generation happened in the
    strategy's forward(), so the LLM/expert call is not repeated inside the
    rollout worker)."""

    def __init__(self, program: Program):
        super().__init__()
        self._program = program

    def build_prompt(self, context: Context, selection: SelectionData) -> str:  # pragma: no cover
        return ""

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        result.generated_program = self._program
        return result


class _ReplayEvaluate(EvaluateModule):
    """Replays precomputed reliability evidence instead of re-running the
    evaluator in the worker (which would double-charge the visible budget)."""

    def __init__(self, *, combined_score: float, validity: float, metrics: Dict[str, Any]):
        super().__init__(evaluate_fn=None)
        self._combined_score = combined_score
        self._validity = validity
        self._metrics = metrics

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        program = result.generated_program
        program.combined_score = self._combined_score
        program.validity = self._validity
        program.metrics = dict(self._metrics)
        return result

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        if not result.generated_program:
            raise ValueError(f"{self.name}: no generated program to replay")


# =============================================================================
# Rule-based meta policy (heuristic_v0)
# =============================================================================


class HeuristicMetaPolicy:
    """Rule-based stand-in for the learned policy.

    Decision logic (deliberately simple and auditable):
    - nothing certified yet / early iterations -> explore (alternate families)
    - a valid F1 candidate with no F2 evidence -> raise fidelity (local_hpo
      at F2 with more seeds)
    - stable F2 candidate -> request promotion
    - otherwise -> mutate the incumbent (exploit)
    """

    policy_version = "heuristic_v0"

    def __init__(self, family_cycle: Optional[List[str]] = None):
        self._family_cycle = family_cycle or ["gbdt", "mlp"]

    def act(self, obs: AgentObservation) -> StructuredAction:
        # 1. Promotion check first: any top visible candidate with stable F2?
        for cand in obs.top_visible_evidence:
            if cand.n_f2_seeds >= 3 and not cand.certified:
                return StructuredAction(
                    expert=ExpertKind.LOCAL_HPO,
                    parent_ids=[cand.candidate_id],
                    model_family=cand.model_family,
                    fidelity=Fidelity.F2_FULL,
                    seed_list=[11, 29, 47],
                    promotion_requested=True,
                    rationale="stable F2 evidence; request sealed promotion",
                )

        # 2. Raise fidelity for promising F1-only candidates
        for cand in obs.top_visible_evidence:
            if cand.highest_fidelity == 1 and cand.best_rank_ic is not None:
                return StructuredAction(
                    expert=ExpertKind.LOCAL_HPO,
                    parent_ids=[cand.candidate_id],
                    model_family=cand.model_family,
                    fidelity=Fidelity.F2_FULL,
                    seed_list=[11, 29, 47],
                    rationale="promising F1 candidate; escalate to F2 multi-seed",
                )

        # 3. Explore vs exploit
        n_candidates = int(obs.search_archive_summary.get("n_candidates", 0))
        if n_candidates == 0 or obs.incumbent_rank_ic is None:
            family = self._family_cycle[n_candidates % len(self._family_cycle)]
            return StructuredAction(
                expert=ExpertKind.EXPLORE,
                parent_ids=[],
                model_family=family,
                fidelity=Fidelity.F1_CHEAP,
                seed_list=[11],
                rationale="cold start: cheap exploration",
            )

        # Exploit: mutate the incumbent at F1
        incumbent_id = (
            obs.certified_candidates[0].candidate_id
            if obs.certified_candidates
            else obs.top_visible_evidence[0].candidate_id
        )
        family = (
            obs.certified_candidates[0].model_family
            if obs.certified_candidates
            else obs.top_visible_evidence[0].model_family
        )
        return StructuredAction(
            expert=ExpertKind.MUTATE,
            parent_ids=[incumbent_id],
            model_family=family,
            fidelity=Fidelity.F1_CHEAP,
            seed_list=[11],
            rationale="exploit incumbent with cheap mutation",
        )


# =============================================================================
# The strategy
# =============================================================================


class ReliabilityAwareQuantStrategy(Strategy):
    """Meta-controller implementing the reliability-gated search loop."""

    llm_client: Any = None

    def __init__(
        self,
        evaluate_fn: Optional[Callable] = None,
        *,
        manifest: FrozenSplitManifest,
        fidelity_evaluator: FidelityEvaluator,
        ledger: BudgetLedger,
        state_store: Optional[StateStore] = None,
        trajectory_store: Optional[TrajectoryStore] = None,
        experts: Optional[Dict[str, ProposalExpert]] = None,
        meta_policy: Optional[Any] = None,
        promotion_policy: Optional[PromotionPolicy] = None,
        gate: Optional[BudgetedGate] = None,
        params: Optional[Any] = None,
    ):
        self.evaluate_fn = evaluate_fn
        self.manifest = manifest
        self.evaluator = fidelity_evaluator
        self.ledger = ledger
        self.gate = gate
        self.population_module = FullArchivePopulation()

        # StateStore: use the injected one (Evolver overwrites via
        # ``strategy.state_store`` before run) or a private one for tests.
        self._own_state_store = state_store or StateStore()
        self._archives_ready = False
        self._search: Optional[SearchArchive] = None
        self._certified: Optional[CertifiedArchive] = None
        self._admission: Optional[CertifiedAdmission] = None
        self._obs_builder: Optional[ObservationBuilder] = None

        self.trajectory = trajectory_store or TrajectoryStore()
        self.experts = experts or default_expert_registry()
        self.meta_policy = meta_policy or HeuristicMetaPolicy()
        self.promotion_policy = promotion_policy or PromotionPolicy()

        self._pending: Dict[str, Dict[str, Any]] = {}  # rollout bookkeeping
        self._state_version = 0
        self.logger: Optional[Any] = None  # injected by Evolver at run()

    # ------------------------------------------------------------------
    # archive wiring (lazy: state_store is injected by Evolver at run())
    # ------------------------------------------------------------------

    def _ensure_archives(self) -> None:
        store = self.state_store or self._own_state_store
        if self._archives_ready and self._search is not None:
            return
        self._search = SearchArchive(store)
        self._certified = CertifiedArchive(store)
        self._admission = CertifiedAdmission(self._search, self._certified)
        self._obs_builder = ObservationBuilder(self._search, self._certified, self.ledger)
        self.ledger.configure_episode(
            self.manifest.episode_id, sealed_limit=self._sealed_limit()
        )
        self._archives_ready = True

    def _sealed_limit(self) -> int:
        return int(self.manifest.__dict__.get("sealed_query_limit", 20))

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        self._ensure_archives()
        self._state_version += 1

        obs = self._obs_builder.build(
            episode_id=self.manifest.episode_id,
            state_version=self._state_version,
            policy_version=self.meta_policy.policy_version,
        )
        action = self.meta_policy.act(obs)

        decision = DecisionRecord(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            observation_digest=obs.digest(),
            structured_action=action,
            state_version=self._state_version,
            policy_version=self.meta_policy.policy_version,
            timestamp=time.time(),
        )
        self.trajectory.record_decision(decision)

        # --- translate action into a candidate --------------------------
        parents = [p for p in (ctx.get_program_by_id(pid) for pid in action.parent_ids) if p]
        # Cold start: no visible parents yet — reseat onto the enriched
        # archive baselines so non-explore experts still have a parent.
        if not parents:
            parents = list(self.experiment_archive_values(ctx))[:1]
        expert = self._pick_expert(action)
        candidate = expert.propose(action, parents, iteration=ctx.iteration)
        candidate.meta["decision_id"] = decision.decision_id

        # --- evaluate through the fidelity ladder -----------------------
        evidence_ids: List[str] = []
        total_cost = EvaluationCost()
        try:
            evidence = self.evaluator.evaluate(
                candidate, action.fidelity, seed_list=action.seed_list
            )
        except BudgetExhausted as e:
            # Budget exhaustion is a *policy-relevant* outcome, not a crash:
            # record empty evidence and let the policy see zero budget next round.
            evidence = None
            if self.logger:
                self.logger.warning(f"[Reliability] budget exhausted: {e}")

        if evidence is not None:
            self._search.add_candidate(
                candidate.id,
                episode_id=self.manifest.episode_id,
                model_family=candidate.meta.get("model_family", "unknown"),
                code_hash=self._code_hash(candidate),
                meta={"expert": action.expert.value},
            )
            from famou.reliability.types import CandidateLineage

            self._search.add_candidate(  # idempotent; ensures lineage recorded
                candidate.id,
                episode_id=self.manifest.episode_id,
                model_family=candidate.meta.get("model_family", "unknown"),
                code_hash=self._code_hash(candidate),
                lineage=CandidateLineage(
                    candidate_id=candidate.id,
                    parent_ids=[p.id for p in parents],
                    expert=action.expert.value,
                    decision_id=decision.decision_id,
                ),
            )
            self._search.add_evidence(evidence)
            evidence_ids.append(evidence.eval_id)
            total_cost = evidence.cost

            # Mirror headline metrics onto the Program so the famou
            # population/logging layer stays meaningful.
            if evidence.rank_ic is not None:
                candidate.combined_score = evidence.rank_ic.mean
                candidate.metrics = {
                    "rank_ic": evidence.rank_ic.mean,
                    "rank_ic_std": evidence.rank_ic.std,
                    "icir": evidence.icir.mean if evidence.icir else 0.0,
                    "fidelity": int(evidence.fidelity.value),
                }
            candidate.validity = evidence.validity
            candidate.error_info = evidence.error_info

        # --- promotion path ----------------------------------------------
        verdict = None
        if evidence is not None and action.promotion_requested and self.gate is not None:
            verdict = self._maybe_promote(candidate, decision)

        # --- transition ---------------------------------------------------
        transition = build_transition(
            decision=decision,
            candidate_ids=[candidate.id],
            evidence_ids=evidence_ids,
            costs=total_cost,
            gate_verdict=verdict,
        )
        self.trajectory.record_transition(transition)

        # --- materialize the rollout for the Evolver ----------------------
        # The candidate is already generated AND evaluated (reliability
        # evidence). The rollout replays both so the Evolver's
        # population/checkpoint machinery sees a normal pipeline result.
        parent_id = parents[0].id if parents else candidate.id
        replay_score = candidate.combined_score or 0.0
        replay_validity = candidate.validity if candidate.validity is not None else 0.0
        rollout = Rollout(
            modules=[
                _FixedSelect(parent_id),
                _PreGeneratedGenerate(candidate),
                _ReplayEvaluate(
                    combined_score=replay_score,
                    validity=replay_validity,
                    metrics=dict(candidate.metrics or {}),
                ),
            ],
            name=f"reliability_{action.expert.value}",
        )
        self._pending[candidate.id] = {
            "decision_id": decision.decision_id,
            "transition_id": transition.transition_id,
        }
        return rollout

    # ------------------------------------------------------------------

    def _pick_expert(self, action: StructuredAction) -> ProposalExpert:
        key = action.model_family
        if action.expert == ExpertKind.EXPLORE and f"{key}_explore" in self.experts:
            return self.experts[f"{key}_explore"]
        if action.expert == ExpertKind.LOCAL_HPO and f"{key}_hpo" in self.experts:
            return self.experts[f"{key}_hpo"]
        if key in self.experts:
            return self.experts[key]
        # fall back to any expert of the requested kind
        for expert in self.experts.values():
            if expert.kind == action.expert:
                return expert
        return next(iter(self.experts.values()))

    def _maybe_promote(self, candidate: Program, decision: DecisionRecord):
        assert self._search is not None and self._obs_builder is not None
        evidence_list = self._search.get_evidence(candidate.id)
        remaining = self.ledger.remaining(episode_id=self.manifest.episode_id)
        incumbent = self._current_incumbent_rank_ic()
        best = self._search.best_evidence(candidate.id)
        novelty = best.novelty if best else None

        decision_out = self.promotion_policy.evaluate(
            evidence_list,
            incumbent_rank_ic=incumbent,
            sealed_queries_remaining=remaining.get("sealed_queries", 0.0),
            novelty=novelty,
        )
        if decision_out.action != "request_gate":
            if self.logger:
                self.logger.info(
                    f"[Promotion] {candidate.id}: {decision_out.action} — {decision_out.reason}"
                )
            return None

        try:
            request = build_gate_request(
                candidate_id=candidate.id,
                candidate_code=candidate.code,
                manifest=self.manifest,
                ledger=self.ledger,
                seed_list=decision.structured_action.seed_list,
            )
        except BudgetExhausted:
            return None

        verdict = self.gate.evaluate(request, candidate.code)
        if self.logger:
            self.logger.info(
                f"[Gate] {candidate.id}: {verdict.verdict.value}/{verdict.reason_code.value} "
                f"(margin={verdict.margin_band.value})"
            )
        if verdict.verdict == GateVerdictKind.PROMOTE:
            admitted = self._admission.verify_and_admit(
                request,
                verdict,
                model_family=candidate.meta.get("model_family", "unknown"),
            )
            if self.logger:
                self.logger.info(f"[Certified] {candidate.id} admitted={admitted}")
        return verdict

    def _current_incumbent_rank_ic(self) -> Optional[float]:
        assert self._search is not None and self._certified is not None
        best: Optional[float] = None
        for cid in self._certified.members():
            ev = self._search.best_evidence(cid)
            if ev and ev.rank_ic is not None:
                best = ev.rank_ic.mean if best is None else max(best, ev.rank_ic.mean)
        return best

    @staticmethod
    def _code_hash(program: Program) -> str:
        import hashlib

        return hashlib.sha256(program.code.encode("utf-8")).hexdigest()

    @staticmethod
    def experiment_archive_values(ctx: Context):
        """Baseline programs visible to this island (cold-start parents)."""
        if ctx.island_accessor is not None:
            progs = ctx.island_accessor.get_all()
            if progs:
                return progs
        if ctx.accessor is not None:
            return ctx.accessor.get_all()
        return ctx.get_all_programs()

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def dump_state(self) -> Dict[str, Any]:
        return {"state_version": self._state_version}

    def load_state(self, state: Dict[str, Any]) -> None:
        self._state_version = int(state.get("state_version", 0))


def _noop_evaluate(program_path: str, **kwargs) -> Dict[str, Any]:
    """Fallback evaluator when no user evaluate_fn is wired (tests)."""
    return {"combined_score": 0.0, "validity": 1.0}
