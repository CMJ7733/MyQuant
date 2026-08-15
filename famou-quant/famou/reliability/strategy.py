"""ReliabilityAwareQuantStrategy — the meta-controller.

Implements the famou ``Strategy`` protocol. ``forward_batch()`` turns ONE
decision into N concurrently-executed rollouts:

1. builds an AgentObservation from Search/Certified archives + budget,
   at the last *committed* state version,
2. picks a StructuredAction with a rule-based policy (heuristic_v0 —
   the BC/offline-RL policy plugs in later via ``meta_policy``),
3. records a DecisionRecord in the TrajectoryStore,
4. asks a ProposalExpert for ``batch_size`` candidates,
5. returns a WorkBatch — one rollout per candidate, each carrying a
   ``_ReliabilityEvaluate`` module.

The expensive part (train / predict / backtest) then runs in the rollout
workers, in parallel. Each worker attaches its EvidenceVector to the
program's meta; the Evolver reports every finished rollout back through
``on_rollout_complete()`` / ``on_rollout_failed()``. When the last member of
a batch reports:

6. the PromotionPolicy runs against the whole batch's evidence and, when
   warranted, spends a sealed query through the BudgetedGate,
7. ``BarrierCommit`` writes Search Archive + Certified Archive + the new
   state version in ONE atomic step and emits the Transition.

Invariant C1: nothing in this class writes to the archives. Results are
staged and applied only by the barrier, so an observation can never see a
half-finished batch. ``forward()`` remains available as a batch of one.

A promotion-only decision (``promotion_target_id`` set) skips steps 4-5
entirely: it spends its sealed query on an existing candidate's accumulated
evidence rather than on a freshly generated mutation of it.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult, SelectionData, WorkBatch
from famou.core.protocol import Module, Rollout, Strategy
from famou.core.state import StateStore
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.base import GenerateModule
from famou.modules.population.full_archive import FullArchivePopulation
from famou.reliability.archives import (
    CertifiedAdmission,
    CertifiedArchive,
    CommitGuard,
    SearchArchive,
)
from famou.reliability.barrier import (
    BarrierCommit,
    BatchIntegrityError,
    CandidateOutcome,
)
from famou.reliability.budget import BudgetExhausted, BudgetLedger
from famou.reliability.evaluator import FidelityEvaluator
from famou.reliability.experts import ProposalExpert, default_expert_registry
from famou.reliability.final_test import SearchClosed, is_frozen
from famou.reliability.observation import AgentObservation, ObservationBuilder
from famou.reliability.promotion import (
    BudgetedGate,
    PromotionPolicy,
    build_gate_request,
)
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import (
    CandidateLineage,
    CandidatePackage,
    DecisionRecord,
    EvaluationCost,
    EvidenceVector,
    ExpertKind,
    Fidelity,
    FrozenSplitManifest,
    StructuredAction,
    TaskSpec,
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
        # inspiration_ids is declared with default_factory=None, which makes it
        # required in practice — every SelectModule passes it explicitly.
        result.selection = SelectionData(parent_id=self._parent_id, inspiration_ids=[])
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


#: key under which the worker hands its EvidenceVector back to the strategy
EVIDENCE_META_KEY = "_reliability_evidence"


class _ReliabilityEvaluate(EvaluateModule):
    """Runs the multi-fidelity evaluation inside the rollout worker.

    This is what makes the batch actually parallel: ``forward_batch()`` only
    decides and generates (cheap), and the expensive train/backtest work
    happens here, concurrently, one instance per worker.

    The resulting EvidenceVector is passed back to the strategy through
    ``program.meta[EVIDENCE_META_KEY]`` because RolloutResult has no typed
    slot for it; the strategy re-validates it in ``on_rollout_complete()``.
    """

    def __init__(
        self,
        evaluator: FidelityEvaluator,
        *,
        fidelity: Fidelity,
        seed_list: List[int],
        decision_id: str,
        max_gpu_seconds: Optional[float] = None,
    ):
        super().__init__(evaluate_fn=None)
        self._evaluator = evaluator
        self._fidelity = fidelity
        self._seed_list = list(seed_list)
        self._decision_id = decision_id
        self._max_gpu_seconds = max_gpu_seconds

    def __deepcopy__(self, memo):
        """Keep the evaluator SHARED across worker copies.

        The ThreadPool backend deep-copies each rollout so concurrent workers
        get isolated modules. The default Module.__deepcopy__ would happily
        deep-copy ``self._evaluator`` too — and with it the BudgetLedger it
        holds. Every worker would then charge its own private copy of the
        ledger, so budget limits would silently stop being enforced and the
        spent totals would be wrong. The evaluator is deliberately
        thread-safe (the ledger serialises check-and-charge under one lock),
        so sharing the reference is both correct and required.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        new.__dict__.update(self.__dict__)
        # injected per-execution deps are re-injected by the engine
        for key in ("env", "llm_client", "embedding_client", "logger"):
            if key in new.__dict__:
                new.__dict__[key] = None
        return new

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        program = result.generated_program
        try:
            evidence = self._evaluator.evaluate(
                program,
                self._fidelity,
                seed_list=self._seed_list,
                max_gpu_seconds=self._max_gpu_seconds,
            )
        except BudgetExhausted as e:
            # Budget exhaustion is a policy-relevant outcome, not a rollout
            # failure: report it as an invalid program so the batch still
            # balances and the policy sees an empty budget next round.
            program.validity = 0.0
            program.combined_score = 0.0
            program.error_info = f"budget exhausted: {e}"
            program.meta[EVIDENCE_META_KEY] = None
            return result

        program.meta[EVIDENCE_META_KEY] = evidence.model_dump(mode="json")
        program.validity = evidence.validity
        program.error_info = evidence.error_info
        if evidence.rank_ic is not None:
            program.combined_score = evidence.rank_ic.mean
            program.metrics = {
                "rank_ic": evidence.rank_ic.mean,
                "rank_ic_std": evidence.rank_ic.std,
                "icir": evidence.icir.mean if evidence.icir else 0.0,
                "fidelity": int(evidence.fidelity.value),
            }
        else:
            program.combined_score = 0.0
        return result

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        if not result.generated_program:
            raise ValueError(f"{self.name}: no generated program to evaluate")


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

    @classmethod
    def for_task(cls, task_spec: TaskSpec) -> "HeuristicMetaPolicy":
        """Cycle only through families the frozen contract allows.

        Without this the default cycle would propose ``gbdt`` in a
        formula-only run, the expert lookup would fall back to whatever is
        registered, and the recorded action would not describe what actually
        ran.
        """
        families = [f for f in task_spec.allowed_model_families
                    if f in ("formula", "gbdt", "mlp", "temporal_transformer")]
        return cls(family_cycle=families or ["gbdt"])

    def act(self, obs: AgentObservation) -> StructuredAction:
        # 1. Promotion check first: any top visible candidate with stable F2?
        #    The action targets THAT candidate directly (promotion_target_id).
        #    Mutating it first and gating the mutation would spend the sealed
        #    query on a candidate that has one evaluation to its name.
        #    ``promotion_checked_at`` is what stops this branch re-firing: a
        #    candidate the policy already declined spends no sealed query, so
        #    a guard on gate_attempts alone would re-propose it forever.
        for cand in obs.top_visible_evidence:
            if (
                cand.n_f2_seeds >= 3
                and not cand.certified
                and cand.gate_attempts == 0
                and cand.promotion_checked_at < cand.n_evidence
            ):
                return StructuredAction(
                    expert=ExpertKind.LOCAL_HPO,
                    parent_ids=[cand.candidate_id],
                    model_family=cand.model_family,
                    fidelity=Fidelity.F2_FULL,
                    seed_list=[11, 29, 47],
                    promotion_requested=True,
                    promotion_target_id=cand.candidate_id,
                    rationale="stable F2 evidence; request sealed promotion",
                )

        # 2. Raise fidelity for promising F1-only candidates.
        #    Targets the candidate ITSELF (reeval_target_id): mutating it would
        #    leave the original at F1, so this branch would re-fire forever and
        #    the search would never move on.
        for cand in obs.top_visible_evidence:
            if cand.highest_fidelity == 1 and cand.best_rank_ic is not None:
                return StructuredAction(
                    expert=ExpertKind.LOCAL_HPO,
                    parent_ids=[cand.candidate_id],
                    model_family=cand.model_family,
                    fidelity=Fidelity.F2_FULL,
                    seed_list=[11, 29, 47],
                    reeval_target_id=cand.candidate_id,
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
        task_spec: Optional[TaskSpec] = None,
        state_store: Optional[StateStore] = None,
        trajectory_store: Optional[TrajectoryStore] = None,
        experts: Optional[Dict[str, ProposalExpert]] = None,
        meta_policy: Optional[Any] = None,
        promotion_policy: Optional[PromotionPolicy] = None,
        gate: Optional[BudgetedGate] = None,
        sealed_query_limit: int = 20,
        observation_encoder: Optional[Any] = None,
        params: Optional[Any] = None,
    ):
        self.evaluate_fn = evaluate_fn
        self.manifest = manifest
        self.evaluator = fidelity_evaluator
        self.ledger = ledger
        self.gate = gate
        # The evaluator already owns a TaskSpec; share it so the static check
        # and the strategy cannot drift onto different contracts.
        self.task_spec = task_spec or getattr(fidelity_evaluator, "task_spec", None) or TaskSpec()
        self.sealed_query_limit = int(sealed_query_limit)
        self.population_module = FullArchivePopulation()

        # StateStore: use the injected one (Evolver overwrites via
        # ``strategy.state_store`` before run) or a private one for tests.
        self._own_state_store = state_store or StateStore()
        self._archives_ready = False
        self._guard = CommitGuard(enforce=True)
        self._search: Optional[SearchArchive] = None
        self._certified: Optional[CertifiedArchive] = None
        self._admission: Optional[CertifiedAdmission] = None
        self._obs_builder: Optional[ObservationBuilder] = None
        self._barrier: Optional[BarrierCommit] = None

        self.trajectory = trajectory_store or TrajectoryStore()
        # Experts are restricted to the families the frozen contract allows, so
        # the reachable search space and the declared one cannot drift apart.
        self.experts = experts or default_expert_registry(
            families=self.task_spec.allowed_model_families
        )
        self.meta_policy = meta_policy or HeuristicMetaPolicy.for_task(self.task_spec)
        self.promotion_policy = promotion_policy or PromotionPolicy()
        # Observation encoder: records learnable features on every decision so
        # heuristic runs already produce a behaviour-cloning corpus. Imported
        # lazily — the RL subpackage pulls in torch, which the search loop
        # itself does not need.
        if observation_encoder is not None:
            self._encoder = observation_encoder
        else:
            try:
                from famou.reliability.rl.encoding import ObservationEncoder

                self._encoder = ObservationEncoder()
            except Exception:  # pragma: no cover - optional
                self._encoder = None

        self._pending_state_version: Optional[int] = None  # set by load_state()
        # decision_id -> {decision, accumulator, action, reported}
        self._batches: Dict[str, Dict[str, Any]] = {}
        # candidate_id -> generation context, needed when the worker reports back
        self._candidate_context: Dict[str, Dict[str, Any]] = {}
        self._batch_lock = threading.Lock()
        self.logger: Optional[Any] = None  # injected by Evolver at run()

    # ------------------------------------------------------------------
    # archive wiring (lazy: state_store is injected by Evolver at run())
    # ------------------------------------------------------------------

    def _ensure_archives(self) -> None:
        store = self.state_store or self._own_state_store
        if self._archives_ready and self._search is not None:
            return
        # All four share ONE guard: the barrier opens the only window in which
        # any of them may be written (invariant C1).
        self._search = SearchArchive(store, guard=self._guard)
        self._certified = CertifiedArchive(store, guard=self._guard)
        self._admission = CertifiedAdmission(
            self._search, self._certified, self.manifest, self.ledger
        )
        self._obs_builder = ObservationBuilder(self._search, self._certified, self.ledger)
        self._barrier = BarrierCommit(
            search_archive=self._search,
            certified_archive=self._certified,
            admission=self._admission,
            trajectory_store=self.trajectory,
            state_store=store,
            guard=self._guard,
            logger=self.logger,
        )
        if self._pending_state_version is not None:
            self._barrier.set_state_version(self._pending_state_version)
            self._pending_state_version = None
        self.ledger.configure_episode(
            self.manifest.episode_id, sealed_limit=self.sealed_query_limit
        )
        self._archives_ready = True

    def seed_certified_baseline(
        self,
        candidate_id: str,
        *,
        model_family: str,
        code_hash: str,
        note: str = "initial baseline",
    ) -> None:
        """Register a trusted starting point before the search begins.

        Bootstrap writes are legitimate — they precede any decision — but they
        still go through the barrier's write window so no code path mutates
        the archives unguarded.
        """
        self._ensure_archives()
        with self._barrier.bootstrap():
            self._certified.add_baseline(
                candidate_id,
                episode_id=self.manifest.episode_id,
                model_family=model_family,
                code_hash=code_hash,
                note=note,
            )
            self._search.add_candidate(
                candidate_id,
                episode_id=self.manifest.episode_id,
                model_family=model_family,
                code_hash=code_hash,
            )

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        """Single-rollout entry point (Strategy protocol).

        Kept for Evolvers/tests that call forward() directly; it is a batch of
        one routed through the same decision path.
        """
        batch = self.forward_batch(ctx, rollout_history, max_batch_size=1)
        return batch.rollouts[0]

    def forward_batch(
        self,
        ctx: Context,
        rollout_history: List[RolloutResult],
        max_batch_size: int = 1,
    ) -> WorkBatch:
        """One decision -> N candidates dispatched concurrently.

        Nothing here evaluates or writes: the candidates are generated and
        handed to the workers, which run ``_ReliabilityEvaluate``. Results
        come back through ``on_rollout_complete()``, and the batch is applied
        to the archives only when every member has reported (invariant C1).
        """
        self._ensure_archives()
        # Freezing an episode ends its search. Without this check a stray
        # iteration after the freeze would mutate the certified archive that
        # the FreezeManifest claims to have snapshotted.
        if is_frozen(self.state_store or self._own_state_store, self.manifest.episode_id):
            raise SearchClosed(
                f"episode {self.manifest.episode_id} is frozen; no further "
                "decisions may be made. The final test evaluates the frozen "
                "certified archive."
            )
        # The observation is built at the last COMMITTED version. Nothing that
        # this batch produces will be visible until its barrier commits.
        state_version = self._barrier.state_version

        obs = self._obs_builder.build(
            episode_id=self.manifest.episode_id,
            state_version=state_version,
            policy_version=self.meta_policy.policy_version,
            in_flight_tasks=self._in_flight(),
        )
        action = self.meta_policy.act(obs)

        # Resolve the promotion target BEFORE recording the decision, so the
        # DecisionRecord describes what actually ran.
        promotion_target = None
        if action.promotion_target_id:
            promotion_target = ctx.get_program_by_id(action.promotion_target_id)
            if promotion_target is None:
                # The intent was "gate that existing candidate". Falling back to
                # generation must NOT inherit promotion_requested: gating a
                # brand-new candidate on its first evaluation is exactly the
                # sealed-budget drain that promotion_target_id exists to stop.
                if self.logger:
                    self.logger.warning(
                        f"[Reliability] promotion target {action.promotion_target_id} "
                        "not in context; generating instead and NOT gating"
                    )
                action = action.model_copy(
                    update={"promotion_requested": False, "promotion_target_id": None}
                )

        # A learned policy exposes the log-prob and value it acted with; a
        # heuristic one has neither. Recording them is what lets a later
        # on-policy correction know how surprised the policy was, rather than
        # limiting the trajectory to behaviour cloning.
        decision = DecisionRecord(
            decision_id=f"dec_{uuid.uuid4().hex[:12]}",
            observation_digest=obs.digest(),
            observation_features=self._encode_observation(obs),
            encoding_version=getattr(self._encoder, "version", None),
            structured_action=action,
            state_version=state_version,
            policy_version=self.meta_policy.policy_version,
            action_log_prob=getattr(self.meta_policy, "last_log_prob", None),
            predicted_value=getattr(self.meta_policy, "last_value", None),
            timestamp=time.time(),
        )
        self.trajectory.record_decision(decision)

        # --- promotion-only decision -------------------------------------
        # The action names an existing candidate: spend the sealed query on
        # the evidence it already has instead of generating (and gating) a
        # fresh mutation of it. No generation, no evaluation, no visible charge.
        if promotion_target is not None:
            batch = self._barrier.open(decision, expected=1)
            self._register_batch(decision, batch, action)
            batch.stage(self._execute_promotion(promotion_target, action))
            self._note_completion(decision.decision_id)
            return WorkBatch.single(
                self._replay_rollout(
                    promotion_target,
                    parent_id=promotion_target.id,
                    name="reliability_promotion",
                ),
                decision_id=decision.decision_id,
                phase="promotion",
            )

        # --- translate action into candidates ---------------------------
        n = max(1, min(int(action.batch_size), max_batch_size))
        parents = [p for p in (ctx.get_program_by_id(pid) for pid in action.parent_ids) if p]
        # Cold start: no visible parents yet — reseat onto the enriched
        # archive baselines so non-explore experts still have a parent.
        if not parents:
            parents = list(self.experiment_archive_values(ctx))[:1]
        expert = self._pick_expert(action)

        # --- re-evaluation decision --------------------------------------
        # Escalate an existing candidate's fidelity instead of generating a
        # variant of it. Runs in a worker like any other evaluation, but
        # registers no new candidate: the evidence attaches to the original.
        reeval_target = None
        if action.reeval_target_id:
            reeval_target = ctx.get_program_by_id(action.reeval_target_id)
            if reeval_target is None and self.logger:
                self.logger.warning(
                    f"[Reliability] reeval target {action.reeval_target_id} "
                    "not in context; generating instead"
                )
        if reeval_target is not None:
            self._register_batch(decision, self._barrier.open(decision, expected=1),
                                 action)
            self._candidate_context[reeval_target.id] = {
                "decision_id": decision.decision_id,
                "parent_ids": [p.id for p in parents],
                "model_family": reeval_target.meta.get("model_family", "unknown"),
                "code_hash": self._code_hash(reeval_target),
                "code": reeval_target.code,
                "expert": action.expert.value,
                "package": None,
                "register": False,      # candidate already in the archive
            }
            return WorkBatch.single(
                Rollout(
                    modules=[
                        _FixedSelect(reeval_target.id),
                        _PreGeneratedGenerate(reeval_target),
                        _ReliabilityEvaluate(
                            self.evaluator,
                            fidelity=action.fidelity,
                            seed_list=action.seed_list,
                            decision_id=decision.decision_id,
                            max_gpu_seconds=action.max_gpu_seconds,
                        ),
                    ],
                    name="reliability_reeval",
                ),
                decision_id=decision.decision_id,
                phase="reeval",
            )

        rollouts: List[Rollout] = []
        accumulator = self._barrier.open(decision, expected=n)
        self._register_batch(decision, accumulator, action)
        for _ in range(n):
            candidate = expert.propose(action, parents, iteration=ctx.iteration)
            candidate.meta["decision_id"] = decision.decision_id
            package = expert.package(
                candidate,
                episode_id=self.manifest.episode_id,
                data_contract_hash=self.manifest.compute_hash(),
                task_spec_hash=self.task_spec.compute_hash(),
            )
            self._candidate_context[candidate.id] = {
                "decision_id": decision.decision_id,
                "parent_ids": [p.id for p in parents],
                "model_family": candidate.meta.get("model_family", "unknown"),
                "code_hash": package.code_hash,
                "code": candidate.code,
                "expert": action.expert.value,
                "package": package,
            }
            rollouts.append(
                Rollout(
                    modules=[
                        _FixedSelect(parents[0].id if parents else candidate.id),
                        _PreGeneratedGenerate(candidate),
                        _ReliabilityEvaluate(
                            self.evaluator,
                            fidelity=action.fidelity,
                            seed_list=action.seed_list,
                            decision_id=decision.decision_id,
                            max_gpu_seconds=action.max_gpu_seconds,
                        ),
                    ],
                    name=f"reliability_{action.expert.value}",
                )
            )

        return WorkBatch(
            rollouts=rollouts,
            concurrency_hint=len(rollouts),
            barrier=True,
            meta={
                "decision_id": decision.decision_id,
                "expert": action.expert.value,
                "fidelity": int(action.fidelity.value),
            },
        )

    # ------------------------------------------------------------------
    # batch bookkeeping (barrier lifecycle)
    # ------------------------------------------------------------------

    def _register_batch(self, decision, accumulator, action) -> None:
        with self._batch_lock:
            self._batches[decision.decision_id] = {
                "decision": decision,
                "accumulator": accumulator,
                "action": action,
                "reported": 0,
            }

    def _in_flight(self) -> int:
        with self._batch_lock:
            return sum(
                entry["accumulator"].expected - entry["reported"]
                for entry in self._batches.values()
            )

    def _note_completion(self, decision_id: str) -> None:
        """Count one reported rollout; commit the batch once all have landed."""
        with self._batch_lock:
            entry = self._batches.get(decision_id)
            if entry is None:
                return
            entry["reported"] += 1
            if entry["reported"] < entry["accumulator"].expected:
                return
            self._batches.pop(decision_id, None)

        self._commit_batch(entry)

    def _commit_batch(self, entry: Dict[str, Any]) -> None:
        accumulator = entry["accumulator"]
        action = entry["action"]
        # Promotion runs here, not in forward(): by now the whole batch's
        # evidence exists, so the policy judges a candidate on everything it
        # produced rather than on a partial view.
        if action.promotion_requested and self.gate is not None:
            for outcome in accumulator.outcomes:
                if outcome.gate_verdict is not None:
                    continue  # promotion-only path already gated
                code = self._candidate_context.get(outcome.candidate_id, {}).get("code")
                if code is None or not outcome.evidence:
                    continue
                self._run_gate_for_outcome(outcome, code, action, staged=outcome.evidence)
        try:
            self._barrier.commit(accumulator, allow_incomplete=True)
        finally:
            for outcome in accumulator.outcomes:
                self._candidate_context.pop(outcome.candidate_id, None)

    def on_rollout_complete(self, result: RolloutResult) -> None:
        """Stage a finished rollout's evidence and maybe close the batch."""
        self._stage_result(result)

    def on_rollout_failed(self, result: RolloutResult) -> None:
        """A failed rollout still has to be accounted for.

        Otherwise its batch never reaches ``expected`` and the barrier waits
        forever. It is staged as an unregistered outcome — no candidate, no
        evidence — and the Evolver retries the iteration as a fresh decision.
        """
        self._stage_result(result, failed=True)

    def _stage_result(self, result: RolloutResult, *, failed: bool = False) -> None:
        program = getattr(result, "generated_program", None)
        if program is None:
            return
        info = self._candidate_context.get(program.id)
        if info is None:
            return  # not one of ours (or already committed)
        decision_id = info["decision_id"]

        evidence = None
        if not failed:
            raw = (program.meta or {}).get(EVIDENCE_META_KEY)
            if raw:
                evidence = EvidenceVector.model_validate(raw)

        outcome = CandidateOutcome(
            candidate_id=program.id,
            episode_id=self.manifest.episode_id,
            model_family=info["model_family"],
            code_hash=info["code_hash"],
            # A re-evaluation attaches evidence to a candidate that is already
            # in the archive; only a genuinely new candidate registers.
            register=evidence is not None and info.get("register", True),
            meta={
                "expert": info["expert"],
                "package": (
                    info["package"].model_dump(mode="json")
                    if info.get("package") is not None
                    else None
                ),
            },
            lineage=CandidateLineage(
                candidate_id=program.id,
                parent_ids=info["parent_ids"],
                expert=info["expert"],
                decision_id=decision_id,
            ),
            evidence=[evidence] if evidence is not None else [],
            # copy, not alias: the gate adds its sealed query to the outcome's
            # cost, and that must not leak into the EvidenceVector's record of
            # what the *visible* evaluation cost.
            cost=evidence.cost.model_copy() if evidence is not None else EvaluationCost(),
        )
        with self._batch_lock:
            entry = self._batches.get(decision_id)
        if entry is None:
            return
        try:
            entry["accumulator"].stage(outcome)
        except BatchIntegrityError:
            return  # duplicate report; ignore rather than corrupt the batch
        self._note_completion(decision_id)

    def finalize_experiment(self, contexts: Dict[int, Context]) -> None:
        """Commit any batch left open by a stop/crash so nothing is lost."""
        del contexts
        with self._batch_lock:
            leftovers = list(self._batches.values())
            self._batches.clear()
        for entry in leftovers:
            if not entry["accumulator"].outcomes:
                continue
            if self.logger:
                self.logger.warning(
                    f"[Barrier] committing incomplete batch "
                    f"{entry['decision'].decision_id}: "
                    f"{len(entry['accumulator'].outcomes)}/"
                    f"{entry['accumulator'].expected} results"
                )
            try:
                self._commit_batch(entry)
            except Exception as e:  # pragma: no cover - shutdown best effort
                if self.logger:
                    self.logger.error(f"[Barrier] final commit failed: {e}")

    # ------------------------------------------------------------------

    def _replay_rollout(self, program: Program, *, parent_id: str, name: str) -> Rollout:
        """Wrap an already-produced candidate as a replay-only Rollout."""
        return Rollout(
            modules=[
                _FixedSelect(parent_id),
                _PreGeneratedGenerate(program),
                _ReplayEvaluate(
                    combined_score=program.combined_score or 0.0,
                    validity=program.validity if program.validity is not None else 0.0,
                    metrics=dict(program.metrics or {}),
                ),
            ],
            name=name,
        )

    def _execute_promotion(
        self, target: Program, action: StructuredAction
    ) -> CandidateOutcome:
        """Spend a sealed query on an already-committed candidate.

        Produces an outcome (so the sealed query is attributable to a
        decision) but registers no new candidate and runs no new evaluation.
        """
        best = self._search.best_evidence(target.id)
        outcome = CandidateOutcome(
            candidate_id=target.id,
            episode_id=self.manifest.episode_id,
            model_family=target.meta.get("model_family", "unknown"),
            code_hash=self._code_hash(target),
            register=False,
            prior_evidence_refs=[best.eval_id] if best else [],
        )
        if self.gate is not None:
            self._run_gate_for_outcome(outcome, target.code, action, staged=[])
        if best is not None and best.rank_ic is not None:
            target.combined_score = best.rank_ic.mean
        if target.validity is None:
            target.validity = 1.0
        return outcome

    def _encode_observation(self, obs: AgentObservation) -> Optional[List[float]]:
        """Feature vector recorded alongside the decision, for offline RL.

        Encoding failures must never break the search loop: this is training
        telemetry, not control flow.
        """
        if self._encoder is None:
            return None
        try:
            return self._encoder.encode(obs)
        except Exception as e:  # pragma: no cover - defensive
            if self.logger:
                self.logger.warning(f"[RL] observation encoding failed: {e}")
            return None

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

    def _run_gate_for_outcome(
        self,
        outcome: CandidateOutcome,
        candidate_code: str,
        action: StructuredAction,
        *,
        staged: List[Any],
    ) -> None:
        """Decide on promotion and, if warranted, query the sealed gate.

        Writes nothing: the request/verdict are attached to ``outcome`` and
        applied by the barrier at commit time. The promotion decision reads
        committed evidence PLUS this batch's staged evidence — a candidate is
        allowed to reason about the results it just produced, it just isn't
        allowed to make them visible to the next decision early.
        """
        assert self._search is not None
        candidate_id = outcome.candidate_id
        evidence_list = list(self._search.get_evidence(candidate_id)) + list(staged)
        # Record the check regardless of the verdict — see
        # SearchArchive.record_promotion_check for why a declined promotion
        # still has to leave a trace.
        outcome.promotion_checked_at = len(evidence_list)
        remaining = self.ledger.remaining(episode_id=self.manifest.episode_id)
        incumbent = self._current_incumbent_rank_ic()
        best = self._search.best_evidence(candidate_id)
        novelty = best.novelty if best else None

        decision_out = self.promotion_policy.evaluate(
            evidence_list,
            incumbent_rank_ic=incumbent,
            sealed_queries_remaining=remaining.get("sealed_queries", 0.0),
            novelty=novelty,
            prior_gate_attempts=self._search.gate_attempts(candidate_id),
        )
        if decision_out.action != "request_gate":
            if self.logger:
                self.logger.info(
                    f"[Promotion] {candidate_id}: {decision_out.action} — {decision_out.reason}"
                )
            return

        try:
            request = build_gate_request(
                candidate_id=candidate_id,
                candidate_code=candidate_code,
                manifest=self.manifest,
                ledger=self.ledger,
                seed_list=action.seed_list,
            )
        except BudgetExhausted:
            return

        verdict = self.gate.evaluate(request, candidate_code)
        if self.logger:
            self.logger.info(
                f"[Gate] {candidate_id}: {verdict.verdict.value}/{verdict.reason_code.value} "
                f"(margin={verdict.margin_band.value})"
            )
        outcome.gate_request = request
        outcome.gate_verdict = verdict
        outcome.cost.sealed_query_count += verdict.query_cost

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
        return {"state_version": self._barrier.state_version if self._barrier else 0}

    def load_state(self, state: Dict[str, Any]) -> None:
        version = int(state.get("state_version", 0))
        if self._barrier is not None:
            self._barrier.set_state_version(version)
        else:
            # Archives are wired lazily at the first forward(); remember the
            # checkpointed version until the barrier exists.
            self._pending_state_version = version


def _noop_evaluate(program_path: str, **kwargs) -> Dict[str, Any]:
    """Fallback evaluator when no user evaluate_fn is wired (tests)."""
    return {"combined_score": 0.0, "validity": 1.0}
