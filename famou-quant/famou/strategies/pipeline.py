"""Pipeline strategy: population-centric evolution with UCB node selection.

Design (see docs/pipeline_strategy_design.md):

- explore / exploit / crossover / model_fusion are SINGLE-ROUND and STATELESS.
  Evolution memory lives in the population (the branch tree), not in plans.
- Only component ablation keeps multi-round stateful plans (ported from
  ml_pipeline; wired in a later step).
- Scheduling (which operation this round) lives here in the strategy.
  Sampling (which parent/inspirations) lives in the SampleSelect modules.
- UCB visit counts live in the StateStore (read by SampleSelect, written here).

Phases (by global rollout count = iteration):
    iter 1-6    : explore (each opens a new branch)
    iter 7-12   : exploit (UCB; early high-C sweeps each new branch once)
    iter 13+    : steady, priority:
        plateau(>12 no-improve) > fusion > crossover(>6 gap) > component(50) > exploit
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from famou.config.settings import ModulesConfig
from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Rollout, Strategy
from famou.infrastructure.llm.base import (
    get_llm_max_retries,
    get_llm_max_tokens,
    get_llm_temperature,
)
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.crossover import CrossoverGenerate
from famou.modules.generate.explore_divergent import ExploreDivergentGenerate
from famou.modules.generate.model_fusion import ModelFusionGenerate
from famou.modules.generate.mutation import MutationGenerate
from famou.modules.generate.plan_driven_generate import PlanDrivenGenerate
from famou.modules.judge.component_ablation import ComponentJudge
from famou.modules.judge.llm_judge import LLMJudge
from famou.modules.population.full_archive import FullArchivePopulation
from famou.modules.select.explore_sample import ExploreSampleSelect
from famou.modules.select.model_fusion_select import ModelFusionSelect
from famou.modules.select.plan_scoped_select import PlanScopedSelect
from famou.modules.select.ucb_sample import CrossoverSampleSelect, ExploitSampleSelect
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.utils.id_gen import generate_short_uuid
from famou.utils.program_summary import (
    extract_eval_wall_time,
    get_implementation_plan,
    get_improvement_directions,
    get_key_features,
)
from famou.utils.trace_utils import build_llm_trace

logger = logging.getLogger("famou.strategy.pipeline")


# ── branch-graph helpers (inlined; branch semantics are specific to this
# strategy and intentionally not shared via a framework util — duplicated in
# the SampleSelect modules that also need them). An explore node roots a new
# branch; every other operation inherits its parent's branch. ──
def _is_explore(program: Program) -> bool:
    meta = program.meta or {}
    return str(meta.get("operation") or "").strip() == "explore"


def _branch_root_of(program: Program, accessor) -> str:
    """Branch-root id: walk up parent_id to the first explore node (or topmost)."""
    current = program
    visited: set = set()
    while True:
        if _is_explore(current) or current.id in visited:
            return current.id
        visited.add(current.id)
        parent_id = current.parent_id
        if not parent_id:
            return current.id
        parent = accessor.get_by_id(parent_id)
        if parent is None:
            return current.id
        current = parent


def _alive_branch_bests(accessor) -> Dict[str, Program]:
    """Best (highest combined_score) program per live branch."""
    def _score(p: Program) -> float:
        s = p.combined_score
        return float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else float("-inf")

    bests: Dict[str, Program] = {}
    for program in accessor.get_all():
        branch_id = _branch_root_of(program, accessor)
        incumbent = bests.get(branch_id)
        if incumbent is None or _score(program) > _score(incumbent):
            bests[branch_id] = program
    return bests


class PipelineStrategy(Strategy):
    """Population-centric strategy with UCB node selection and branch tree."""

    llm_client: Any = None
    embedding_client: Any = None

    def __init__(
        self,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
        evaluate_module=None,
    ) -> None:
        params = params or ModulesConfig()
        self.evaluate_fn = evaluate_fn

        # ── operation tags (also written into program.meta["operation"]) ──
        self.FUSION = "model_fusion"
        self.COMPONENT = "component"
        self.COMPONENT_CROSSOVER = "component_crossover"
        # StateStore namespace for component experience. ComponentJudge hardcodes
        # the "ml_pipeline" namespace when reading/writing experience, so we use
        # the same namespace to stay compatible without modifying that module.
        self.COMPONENT_EXPERIENCE_NS = "ml_pipeline"

        select_params = {**params.select}
        generate_params = {**params.generate}
        evaluate_params = {**params.evaluate}
        judge_params = {**params.judge}
        population_params = {**params.population}

        # ── scheduling constants ──
        self.bootstrap_explore_count = 6
        self.initial_exploit_count = 6
        self.crossover_cooldown = 6
        self.component_cooldown = 50
        self.plateau_window = 20
        self.plateau_forced_explore = 4
        self.fusion_cooldown = 16
        self.fusion_min_candidates = 5
        # crossover inspiration count: honor config override if present.
        self.crossover_num_inspirations = int(select_params.pop("num_inspirations", 2))

        # ── component-ablation campaign constants (ported from ml_pipeline) ──
        self.component_batch_size = 4          # components discovered per campaign
        self.component_base_budget = 3         # exec rounds per component plan
        self.component_bonus_on_improvement = 2
        self.component_min_prior_programs = 2  # need a best to anchor on
        self.component_crossover_min_plans = 2 # min completed plans to cross

        self.population_module = FullArchivePopulation(**population_params)
        self.logger = None
        self._islands: Dict[int, Dict[str, Any]] = {}

        # ── single-round main-track rollouts ──
        self.explore_rollout = Rollout(
            modules=[
                ExploreSampleSelect(**select_params),
                ExploreDivergentGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="pipeline_explore",
        )
        self.exploit_rollout = Rollout(
            modules=[
                ExploitSampleSelect(**select_params),
                MutationGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="pipeline_exploit",
        )
        self.crossover_rollout = Rollout(
            modules=[
                CrossoverSampleSelect(
                    num_inspirations=self.crossover_num_inspirations,
                    **select_params,
                ),
                CrossoverGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="pipeline_crossover",
        )
        self.fusion_rollout = Rollout(
            modules=[
                ModelFusionSelect(),
                ModelFusionGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="pipeline_fusion",
        )

        # ── multi-round component-ablation rollouts (stateful) ──
        self.component_exec_rollout = Rollout(
            modules=[
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
                ComponentJudge(
                    bonus_budget_on_improvement=self.component_bonus_on_improvement,
                ),
            ],
            name="pipeline_component_exec",
        )
        self.component_crossover_rollout = Rollout(
            modules=[
                # num_inspirations = batch size so ALL completed component-branch
                # bests are fused (the generic default of 3 would drop environments
                # when a campaign has 4+ components). Only this rollout is widened;
                # single-track crossover keeps its 2 inspirations.
                PlanScopedSelect(num_inspirations=self.component_batch_size, **select_params),
                CrossoverGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="pipeline_component_crossover",
        )

    # ------------------------------------------------------------------
    # Strategy lifecycle
    # ------------------------------------------------------------------

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        """Decide the operation for this rollout and return its pipeline."""
        if ctx.iteration == 0 or not self._get_visible_programs(ctx):
            return self.explore_rollout  # enrichment: only Evaluate/Judge run

        island_state = self._ensure_island_state(ctx.island_id)
        self._reconcile_rollout_history(island_state, rollout_history)
        ctx.metadata.pop("fusion_context", None)
        ctx.metadata.pop("plan_context", None)

        self._sync_state_store(ctx, island_state)
        iteration = int(island_state["global_dispatch_count"])

        # ── Phase 2: bootstrap explore (iter 1-6) ──
        if island_state["explore_count_done"] < self.bootstrap_explore_count:
            island_state["explore_count_done"] += 1
            island_state["global_dispatch_count"] += 1
            return self.explore_rollout

        # ── Phase 3: initial exploit (iter 7-12) ──
        if island_state["exploit_phase_done"] < self.initial_exploit_count:
            island_state["exploit_phase_done"] += 1
            island_state["global_dispatch_count"] += 1
            return self.exploit_rollout

        # ── Phase 4: steady (priority order) ──
        rollout = self._decide_steady(ctx, island_state, iteration)
        island_state["global_dispatch_count"] += 1
        return rollout

    def _decide_steady(
        self, ctx: Context, island_state: Dict[str, Any], iteration: int
    ) -> Rollout:
        """Steady-state priority: active-campaign > plateau > fusion > crossover > component > exploit."""
        # 4.0 active component campaign: always drain a pending plan task first.
        campaign_rollout = self._dispatch_active_campaign(ctx, island_state, iteration)
        if campaign_rollout is not None:
            return campaign_rollout

        # 4.1 plateau (highest among single-track): finish forced explores, then trigger.
        if island_state["forced_explore_remaining"] > 0:
            island_state["forced_explore_remaining"] -= 1
            return self.explore_rollout
        if island_state["plateau_counter"] >= self.plateau_window:
            island_state["plateau_counter"] = 0
            island_state["forced_explore_remaining"] = self.plateau_forced_explore - 1
            return self.explore_rollout

        # 4.2 model fusion
        if self._should_fusion(ctx, island_state, iteration):
            fusion_context = self._build_fusion_context(ctx)
            if fusion_context is not None:
                island_state["last_fusion_iter"] = iteration
                ctx.metadata["fusion_context"] = fusion_context
                return self.fusion_rollout

        # 4.3 crossover (gap > cooldown)
        if self._should_crossover(island_state, iteration):
            island_state["last_crossover_iter"] = iteration
            return self.crossover_rollout

        # 4.4 component ablation (every component_cooldown iters): start a campaign.
        if self._should_start_component_campaign(ctx, island_state, iteration):
            started = self._start_component_campaign(ctx, island_state, iteration)
            if started is not None:
                island_state["last_component_iter"] = iteration
                return started

        # 4.5 default: exploit
        return self.exploit_rollout

    def on_rollout_complete(self, result: RolloutResult) -> None:
        """Commit UCB visit increment + refresh plateau counter."""
        island_state = self._ensure_island_state(result.island_id)
        self._apply_rollout_result(island_state, result)
        island_state["processed_rollout_count"] = (
            int(island_state.get("processed_rollout_count", 0)) + 1
        )

    def on_rollout_failed(self, result: RolloutResult) -> None:
        """Release any inflight component plan lock; single-track ops are stateless."""
        island_state = self._ensure_island_state(result.island_id)
        extra = result.selection.extra if result.selection else None
        plan_id = self._component_plan_id_of(extra if isinstance(extra, dict) else {})
        if plan_id:
            island_state["inflight_component_plans"].pop(plan_id, None)

    def dump_state(self) -> Dict[str, Any]:
        return {"islands": self._islands}

    def load_state(self, state: Dict[str, Any]) -> None:
        islands = state.get("islands", {})
        if not isinstance(islands, dict):
            return
        # Merge each restored island onto a fresh default template so a
        # checkpoint written by an older schema (missing later-added keys such
        # as the component-campaign fields) is self-healed: restored values win,
        # absent keys fall back to defaults. Replacing _islands wholesale would
        # otherwise drop new keys and crash on the first direct-subscript access.
        restored: Dict[int, Dict[str, Any]] = {}
        for key, value in islands.items():
            island = self._default_island_state()
            if isinstance(value, dict):
                island.update(value)
            # inflight_component_plans is a *transient* anti-concurrency lock
            # holding rollouts dispatched-but-not-yet-finished. On resume nothing
            # is actually in flight (those rollouts' results were never persisted),
            # so any restored lock is stale. Clearing it lets the campaign
            # re-dispatch those plans instead of skipping them forever — otherwise
            # the plan stays active+locked, _campaign_exec_done never turns true,
            # the campaign never closes, and component ablation silently stalls.
            island["inflight_component_plans"] = {}
            # Normalize nested campaign/plan dicts so direct-subscript access in
            # the campaign state machine is safe even for checkpoints written by
            # an older plan/campaign schema (same self-healing as the island).
            island["active_campaign"] = self._normalize_campaign(island.get("active_campaign"))
            plans = island.get("component_plans")
            island["component_plans"] = {
                pid: self._normalize_plan(pid, p)
                for pid, p in (plans.items() if isinstance(plans, dict) else [])
                if isinstance(p, dict)
            }
            restored[int(key)] = island
        self._islands = restored

    def _normalize_campaign(self, campaign: Any) -> Optional[Dict[str, Any]]:
        """Backfill missing keys on a restored active_campaign (None stays None)."""
        if not isinstance(campaign, dict):
            return None
        defaults = {"batch_id": None, "plan_ids": [], "status": "running", "crossover_plan_id": None}
        return {**defaults, **campaign}

    def _normalize_plan(self, plan_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Backfill missing keys on a restored component-plan dict.

        Restored values always win; only absent keys take a default. ``plan_id``
        and ``kind`` are identity fields — ``plan_id`` is forced from the dict key,
        and a plan missing ``kind`` is assumed to be an exec component plan.
        """
        defaults = {
            "plan_id": plan_id,
            "kind": self.COMPONENT,
            "status": "active",
            "plan_text": "",
            "created_iteration": None,
            "seed_parent_id": None,
            "best_program_id": None,
            "best_score": None,
            "best_metrics": {},
            "best_validity": None,
            "best_implementation_plan": None,
            "best_key_features": None,
            "best_improvement_directions": None,
            "remaining_budget": 0,
            "round_count": 0,
            "child_program_ids": [],
            "rollout_ids": [],
            "component_key": None,
            "component_name": None,
            "component_goal": None,
            "component_constraints": None,
            "component_suggested_techniques": None,
            "reference_program_ids": [],
        }
        merged = {**defaults, **plan}
        merged["plan_id"] = plan_id  # the dict key is the source of truth
        return merged

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _ensure_island_state(self, island_id: int) -> Dict[str, Any]:
        if island_id not in self._islands:
            self._islands[island_id] = self._default_island_state()
        return self._islands[island_id]

    @staticmethod
    def _default_island_state() -> Dict[str, Any]:
        """Fresh per-island state template (single source of truth for keys).

        Used both when first touching an island and when merging a restored
        checkpoint in :meth:`load_state`, so the two can never drift apart.
        """
        return {
            "processed_rollout_count": 0,
            "global_dispatch_count": 0,
            "best_score_seen": None,
            "plateau_counter": 0,
            "forced_explore_remaining": 0,
            "last_crossover_iter": None,
            "last_fusion_iter": None,
            "last_component_iter": None,
            "explore_count_done": 0,
            "exploit_phase_done": 0,
            # ── component-ablation campaign (single active campaign) ──
            "component_plans": {},          # plan_id -> plan dict
            "component_plan_order": [],
            "known_component_keys": [],
            "inflight_component_plans": {},  # plan_id -> task (anti-concurrency)
            "active_campaign": None,         # {batch_id, plan_ids, status, crossover_plan_id}
        }

    def _sync_state_store(self, ctx: Context, island_state: Dict[str, Any]) -> None:
        """Publish global_iter so SampleSelect can read the UCB decay schedule."""
        if self.state_store is None:
            return
        self.state_store.set_island(
            ctx.island_id, "global_iter", value=int(island_state["global_dispatch_count"])
        )

    def _reconcile_rollout_history(
        self, island_state: Dict[str, Any], rollout_history: List[RolloutResult]
    ) -> None:
        """Replay unseen results into plateau counter + visit commits."""
        start = min(int(island_state.get("processed_rollout_count", 0)), len(rollout_history))
        for result in rollout_history[start:]:
            self._apply_rollout_result(island_state, result)
        island_state["processed_rollout_count"] = len(rollout_history)

    def _apply_rollout_result(
        self, island_state: Dict[str, Any], result: RolloutResult
    ) -> None:
        """Commit UCB visit increment / component plan state + plateau counter."""
        extra = result.selection.extra if result.selection else None
        extra = extra if isinstance(extra, dict) else {}

        # Component-plan results are routed to the campaign state machine.
        plan_id = self._component_plan_id_of(extra)
        if plan_id:
            self._apply_component_result(island_state, result, plan_id)
        else:
            # Single-track op: commit UCB visit increment to StateStore.
            visit_target = extra.get("ucb_visit_increment")
            if visit_target and self.state_store is not None:
                visits = self.state_store.get_island(result.island_id, "ucb_visits", default={})
                visits = dict(visits) if isinstance(visits, dict) else {}
                visits[visit_target] = int(visits.get(visit_target, 0)) + 1
                self.state_store.set_island(result.island_id, "ucb_visits", value=visits)

        # Plateau counter (best-score improvement detection) — all ops count.
        child = result.generated_program
        child_score = child.combined_score if child is not None else None
        if child_score is not None:
            best = island_state.get("best_score_seen")
            if best is None or child_score > best:
                island_state["best_score_seen"] = child_score
                island_state["plateau_counter"] = 0
            else:
                island_state["plateau_counter"] = int(island_state.get("plateau_counter", 0)) + 1
        else:
            island_state["plateau_counter"] = int(island_state.get("plateau_counter", 0)) + 1

    # ------------------------------------------------------------------
    # Trigger predicates
    # ------------------------------------------------------------------

    def _should_crossover(self, island_state: Dict[str, Any], iteration: int) -> bool:
        last = island_state.get("last_crossover_iter")
        return last is None or (iteration - int(last)) > self.crossover_cooldown

    def _should_fusion(
        self, ctx: Context, island_state: Dict[str, Any], iteration: int
    ) -> bool:
        last = island_state.get("last_fusion_iter")
        if last is not None and (iteration - int(last)) < self.fusion_cooldown:
            return False
        evaluated = [
            p for p in self._get_visible_programs(ctx)
            if p.combined_score is not None and p.combined_score > 0
        ]
        return len(evaluated) >= self.fusion_min_candidates

    # ------------------------------------------------------------------
    # Fusion context
    # ------------------------------------------------------------------

    def _build_fusion_context(self, ctx: Context) -> Optional[Dict[str, Any]]:
        """Minimal fusion context: best as parent + top-k others as inspirations."""
        accessor = ctx.island_accessor or ctx.accessor
        evaluated = [
            p for p in accessor.get_all()
            if p.combined_score is not None and p.combined_score > 0
        ]
        if not evaluated:
            return None
        evaluated.sort(key=lambda p: p.combined_score or 0.0, reverse=True)
        parent = evaluated[0]
        # Prefer distinct branch bests as fusion partners.
        branch_bests = _alive_branch_bests(accessor)
        parent_branch = _branch_root_of(parent, accessor)
        inspirations = [
            p for p in sorted(branch_bests.values(), key=lambda p: p.combined_score or 0.0, reverse=True)
            if p.id != parent.id and _branch_root_of(p, accessor) != parent_branch
        ]
        if not inspirations:
            inspirations = [p for p in evaluated[1:]]
        if not inspirations:
            return None
        return {
            "parent_id": parent.id,
            "inspiration_ids": [p.id for p in inspirations[: self.fusion_min_candidates]],
            "plan_kind": self.FUSION,
        }

    # ------------------------------------------------------------------
    # Component-ablation campaign (single active campaign, stateful)
    # ------------------------------------------------------------------

    def _component_plan_id_of(self, extra: Dict[str, Any]) -> Optional[str]:
        """Extract the component plan_id from a rollout's selection.extra (or None)."""
        if not isinstance(extra, dict):
            return None
        plan_context = extra.get("plan_context")
        if isinstance(plan_context, dict) and plan_context.get("planner_mode") == self.COMPONENT:
            pid = plan_context.get("plan_id")
            return str(pid) if pid else None
        return None

    def _should_start_component_campaign(
        self, ctx: Context, island_state: Dict[str, Any], iteration: int
    ) -> bool:
        """Whether to launch a new component campaign now."""
        if island_state.get("active_campaign") is not None:
            return False  # one campaign at a time
        last = island_state.get("last_component_iter")
        if last is not None and (iteration - int(last)) < self.component_cooldown:
            return False
        evaluated = [
            p for p in self._get_visible_programs(ctx)
            if p.combined_score is not None and p.combined_score > 0
        ]
        return len(evaluated) >= self.component_min_prior_programs

    def _start_component_campaign(
        self, ctx: Context, island_state: Dict[str, Any], iteration: int
    ) -> Optional[Rollout]:
        """Discover components, register their plans, and dispatch the first exec."""
        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            return None
        components = self._discover_component_specs(ctx, island_state, best_program)
        if not components:
            return None

        plan_ids: List[str] = []
        for component in components:
            key = component["component_key"]
            if key in island_state["known_component_keys"]:
                continue
            plan = self._register_component_plan(ctx, island_state, component, best_program)
            island_state["known_component_keys"].append(key)
            plan_ids.append(plan["plan_id"])
        if not plan_ids:
            return None

        batch_id = f"campaign_{ctx.island_id}_{iteration}_{generate_short_uuid()}"
        island_state["active_campaign"] = {
            "batch_id": batch_id,
            "plan_ids": plan_ids,
            "status": "running",
            "crossover_plan_id": None,
        }
        # Dispatch the first plan's first exec round immediately.
        return self._dispatch_active_campaign(ctx, island_state, iteration)

    def _dispatch_active_campaign(
        self, ctx: Context, island_state: Dict[str, Any], iteration: int
    ) -> Optional[Rollout]:
        """If a campaign is active, dispatch its next ready exec / crossover task."""
        campaign = island_state.get("active_campaign")
        if not isinstance(campaign, dict):
            return None

        # Island reset (reset_interval, multi-island only) wipes the island
        # population out from under an in-flight campaign and reseeds it with
        # donor copies, so every program the campaign's plans anchor on vanishes
        # at once. Continuing would silently re-anchor each round onto unrelated
        # donors with a stale best_score baseline (corrupting bonus logic and the
        # persisted component experience). Detect the wipe and abandon the
        # campaign so a fresh one can later start on the reseeded population.
        if not self._campaign_anchor_alive(ctx, island_state, campaign):
            self._abort_campaign(island_state)
            return None

        plans = island_state["component_plans"]
        inflight = island_state["inflight_component_plans"]

        # 1. Dispatch next pending exec round for any non-exhausted component plan.
        for plan_id in campaign["plan_ids"]:
            plan = plans.get(plan_id)
            if not isinstance(plan, dict):
                continue
            if plan.get("status") != "active":
                continue
            if plan_id in inflight:
                continue
            if int(plan.get("remaining_budget", 0)) <= 0:
                continue
            inflight[plan_id] = {"plan_id": plan_id, "iteration": iteration}
            ctx.metadata["plan_context"] = self._build_component_plan_context(ctx, island_state, plan)
            return self.component_exec_rollout

        # 2. Re-dispatch an interrupted crossover round. The crossover plan is
        # NOT in campaign["plan_ids"], so the exec loop above never revisits it.
        # If its rollout was lost — resume cleared the inflight lock, or the
        # rollout failed/was discarded (on_rollout_failed only releases the lock,
        # it does not advance campaign status) — the campaign would otherwise
        # stall in "crossing" forever, permanently disabling component ablation.
        # Re-issue it whenever it is active and not currently in flight.
        crossover_plan_id = campaign.get("crossover_plan_id")
        if crossover_plan_id and crossover_plan_id not in inflight:
            crossover_plan = plans.get(crossover_plan_id)
            if (
                isinstance(crossover_plan, dict)
                and crossover_plan.get("status") == "active"
                and int(crossover_plan.get("remaining_budget", 0)) > 0
            ):
                inflight[crossover_plan_id] = {"plan_id": crossover_plan_id, "iteration": iteration}
                ctx.metadata["plan_context"] = self._build_component_plan_context(
                    ctx, island_state, crossover_plan
                )
                return self.component_crossover_rollout

        # 3. All exec plans done (none active/inflight). Try crossover over the batch.
        if self._campaign_exec_done(island_state, campaign):
            if campaign["status"] == "running":
                crossover_plan = self._create_component_crossover_plan(ctx, island_state, campaign)
                if crossover_plan is not None:
                    campaign["status"] = "crossing"
                    campaign["crossover_plan_id"] = crossover_plan["plan_id"]
                    inflight[crossover_plan["plan_id"]] = {
                        "plan_id": crossover_plan["plan_id"], "iteration": iteration
                    }
                    ctx.metadata["plan_context"] = self._build_component_plan_context(
                        ctx, island_state, crossover_plan
                    )
                    return self.component_crossover_rollout
                # Not enough completed plans to cross → close campaign.
                island_state["active_campaign"] = None
            # crossing in flight, or already done → nothing to dispatch.
        return None

    def _campaign_exec_done(self, island_state: Dict[str, Any], campaign: Dict[str, Any]) -> bool:
        """True when every exec component plan is completed and none inflight."""
        plans = island_state["component_plans"]
        inflight = island_state["inflight_component_plans"]
        for plan_id in campaign["plan_ids"]:
            plan = plans.get(plan_id)
            if not isinstance(plan, dict):
                continue
            if plan.get("status") == "active" or plan_id in inflight:
                return False
        return True

    def _campaign_anchor_alive(
        self, ctx: Context, island_state: Dict[str, Any], campaign: Dict[str, Any]
    ) -> bool:
        """Whether any campaign plan still has a visible program to anchor on.

        A plan's anchors are its seed parent, its rolling best, and its produced
        children. If *every* such program (across every plan, exec + crossover)
        is gone from the island, the population was reset/wiped and the campaign
        is orphaned. With the default FullArchivePopulation nothing is ever
        evicted, so this is a no-op outside an actual island reset.
        """
        accessor = ctx.island_accessor or ctx.accessor
        if accessor is None:
            return True  # cannot tell — never abort on missing accessor
        visible = {p.id for p in accessor.get_all()}
        if not visible:
            return False
        plans = island_state.get("component_plans") or {}
        plan_ids = list(campaign.get("plan_ids") or [])
        crossover_id = campaign.get("crossover_plan_id")
        if crossover_id:
            plan_ids.append(crossover_id)
        for plan_id in plan_ids:
            plan = plans.get(plan_id)
            if not isinstance(plan, dict):
                continue
            anchors = [
                plan.get("seed_parent_id"),
                plan.get("best_program_id"),
                *(plan.get("child_program_ids") or []),
                *(plan.get("reference_program_ids") or []),
            ]
            if any(a and a in visible for a in anchors):
                return True
        return False

    def _abort_campaign(self, island_state: Dict[str, Any]) -> None:
        """Tear down an orphaned campaign and release all its locks/plan state.

        Marks the plans abandoned (so any late-arriving rollout result for them
        is a no-op), drops the active campaign, and clears inflight locks for its
        plans. last_component_iter is left untouched so the normal cooldown
        governs when the next campaign may start on the reseeded population.
        """
        campaign = island_state.get("active_campaign")
        if not isinstance(campaign, dict):
            return
        plan_ids = list(campaign.get("plan_ids") or [])
        crossover_id = campaign.get("crossover_plan_id")
        if crossover_id:
            plan_ids.append(crossover_id)
        plans = island_state.get("component_plans") or {}
        inflight = island_state.get("inflight_component_plans") or {}
        for plan_id in plan_ids:
            inflight.pop(plan_id, None)
            plan = plans.get(plan_id)
            if isinstance(plan, dict) and plan.get("status") == "active":
                plan["status"] = "abandoned"
        island_state["active_campaign"] = None

    def _register_component_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        component: Dict[str, str],
        seed_parent: Program,
    ) -> Dict[str, Any]:
        """Create and store one component-enhancement plan."""
        plan_id = f"comp_{ctx.island_id}_{ctx.iteration}_{generate_short_uuid()}"
        plan_text = (
            "Component enhancement plan.\n"
            f"Component name: {component['component_name']}\n"
            f"Goal: {component['component_goal']}\n"
            f"Constraints: {component['component_constraints']}\n"
            f"Suggested techniques: {component['suggested_techniques']}\n"
            "Instruction: treat this as a controlled component-focused improvement, "
            "and avoid rewriting unrelated core logic."
        )
        plan = {
            "plan_id": plan_id,
            "kind": self.COMPONENT,
            "status": "active",
            "plan_text": plan_text,
            "created_iteration": ctx.iteration,
            "seed_parent_id": seed_parent.id,
            "best_program_id": seed_parent.id,
            "best_score": seed_parent.combined_score,
            "best_metrics": dict(seed_parent.metrics or {}),
            "best_validity": seed_parent.validity,
            "best_implementation_plan": get_implementation_plan(seed_parent),
            "best_key_features": get_key_features(seed_parent),
            "best_improvement_directions": get_improvement_directions(seed_parent),
            "remaining_budget": self.component_base_budget,
            "round_count": 0,
            "child_program_ids": [],
            "rollout_ids": [],
            "component_key": component["component_key"],
            "component_name": component["component_name"],
            "component_goal": component["component_goal"],
            "component_constraints": component["component_constraints"],
            "component_suggested_techniques": component["suggested_techniques"],
            "reference_program_ids": [],
        }
        island_state["component_plans"][plan_id] = plan
        island_state["component_plan_order"].append(plan_id)
        return plan

    def _create_component_crossover_plan(
        self, ctx: Context, island_state: Dict[str, Any], campaign: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a crossover plan fusing the completed component plans' bests."""
        plans = island_state["component_plans"]
        completed = [
            plans[pid] for pid in campaign["plan_ids"]
            if isinstance(plans.get(pid), dict)
            and plans[pid].get("status") == "completed"
            and plans[pid].get("best_program_id")
        ]
        if len(completed) < self.component_crossover_min_plans:
            return None
        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            return None
        completed.sort(key=lambda p: p.get("best_score") or 0.0, reverse=True)
        source_program_ids = [str(p["best_program_id"]) for p in completed]

        plan_id = f"compx_{ctx.island_id}_{ctx.iteration}_{generate_short_uuid()}"
        plan = {
            "plan_id": plan_id,
            "kind": self.COMPONENT_CROSSOVER,
            "status": "active",
            "plan_text": (
                "Component crossover plan.\n"
                "Goal: fuse the strongest component-enhancement variants into one "
                "coherent implementation, keeping the current best as the anchor."
            ),
            "created_iteration": ctx.iteration,
            "seed_parent_id": best_program.id,
            "best_program_id": best_program.id,
            "best_score": best_program.combined_score,
            "best_metrics": dict(best_program.metrics or {}),
            "best_validity": best_program.validity,
            "best_implementation_plan": get_implementation_plan(best_program),
            "best_key_features": get_key_features(best_program),
            "best_improvement_directions": get_improvement_directions(best_program),
            "remaining_budget": 1,
            "round_count": 0,
            "child_program_ids": [],
            "rollout_ids": [],
            "component_key": None,
            "component_name": None,
            "component_goal": None,
            "component_constraints": None,
            "component_suggested_techniques": None,
            "reference_program_ids": source_program_ids,
        }
        island_state["component_plans"][plan_id] = plan
        island_state["component_plan_order"].append(plan_id)
        return plan

    def _apply_component_result(
        self, island_state: Dict[str, Any], result: RolloutResult, plan_id: str
    ) -> None:
        """Update a component plan from a finished rollout; advance campaign state."""
        island_state["inflight_component_plans"].pop(plan_id, None)
        plan = island_state["component_plans"].get(plan_id)
        if not isinstance(plan, dict):
            return
        # A plan abandoned by _abort_campaign (island reset orphaned it) must not
        # be revived by a rollout dispatched before the abort. Drop the lock
        # (done above) and ignore the stale result.
        if plan.get("status") == "abandoned":
            return

        if result.rollout_id not in plan["rollout_ids"]:
            plan["rollout_ids"].append(result.rollout_id)
            plan["round_count"] += 1
            plan["remaining_budget"] = max(0, int(plan.get("remaining_budget", 0)) - 1)

        child = result.generated_program
        if child is not None:
            if child.id not in plan["child_program_ids"]:
                plan["child_program_ids"].append(child.id)
            child_score = child.combined_score
            prev_best = plan.get("best_score")
            improved = child_score is not None and (prev_best is None or child_score > prev_best)
            if improved:
                plan["best_program_id"] = child.id
                plan["best_score"] = child_score
                plan["best_metrics"] = dict(child.metrics or {})
                plan["best_validity"] = child.validity
                plan["best_implementation_plan"] = get_implementation_plan(child)
                plan["best_key_features"] = get_key_features(child)
                plan["best_improvement_directions"] = get_improvement_directions(child)
                # Reward improvement with bonus rounds (exec plans only). This
                # mirrors ComponentJudge._should_summarize_plan, which uses
                # island_best_score_at_dispatch == the plan's pre-round best_score.
                if plan["kind"] == self.COMPONENT:
                    plan["remaining_budget"] += self.component_bonus_on_improvement

        if int(plan.get("remaining_budget", 0)) <= 0:
            plan["status"] = "completed"
            self._maybe_close_campaign(island_state, plan)

    def _maybe_close_campaign(self, island_state: Dict[str, Any], plan: Dict[str, Any]) -> None:
        """Close the active campaign once its crossover plan completes."""
        campaign = island_state.get("active_campaign")
        if not isinstance(campaign, dict):
            return
        if plan["kind"] == self.COMPONENT_CROSSOVER and plan["plan_id"] == campaign.get("crossover_plan_id"):
            campaign["status"] = "crossed"
            island_state["active_campaign"] = None

    def _build_component_plan_context(
        self, ctx: Context, island_state: Dict[str, Any], plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build plan_context consumed by PlanScopedSelect / PlanDrivenGenerate / ComponentJudge."""
        current_parent = self._resolve_component_parent(ctx, plan)
        visible_program_ids = list(plan.get("reference_program_ids", [])) + list(
            plan.get("child_program_ids", [])
        )
        component_key = plan.get("component_key")
        component_experience = self._read_component_experience(ctx, component_key)
        # ComponentJudge recomputes plan closure from remaining_budget +
        # island_best_score_at_dispatch. We pass the plan's current best_score as
        # that baseline so the judge's bonus logic matches our _apply_component_result
        # (bonus only when the child beats the plan's incumbent best).
        island_best_at_dispatch = plan.get("best_score")

        return {
            "plan_id": plan["plan_id"],
            "planner_mode": self.COMPONENT,  # ComponentJudge requires this == "component"
            "plan_text": plan["plan_text"],
            "plan_round": int(plan.get("round_count", 0)) + 1,
            "created_iteration": plan.get("created_iteration"),
            "seed_parent_id": plan.get("seed_parent_id"),
            "current_parent_id": current_parent.id if current_parent else plan.get("seed_parent_id"),
            "best_program_id": plan.get("best_program_id") or plan.get("seed_parent_id"),
            "best_score": plan.get("best_score"),
            "best_metrics": dict(plan.get("best_metrics") or {}),
            "best_validity": plan.get("best_validity"),
            "best_implementation_plan": plan.get("best_implementation_plan"),
            "best_key_features": plan.get("best_key_features"),
            "best_improvement_directions": plan.get("best_improvement_directions"),
            "anchor_program_id": plan.get("seed_parent_id"),
            "island_best_score_at_dispatch": island_best_at_dispatch,
            "plan_program_ids": list(dict.fromkeys(visible_program_ids)),
            "plan_rollout_ids": list(plan.get("rollout_ids") or []),
            "remaining_budget": plan.get("remaining_budget"),
            "planner_focus_plan_id": plan.get("component_key"),
            "planner_history_plan_ids": list(plan.get("reference_program_ids") or []),
            "planner_crossover_plan_ids": list(plan.get("reference_program_ids") or []),
            "component_key": plan.get("component_key"),
            "component_name": plan.get("component_name"),
            "component_goal": plan.get("component_goal"),
            "component_constraints": plan.get("component_constraints"),
            "component_suggested_techniques": plan.get("component_suggested_techniques"),
            "component_experience": component_experience,
            "strategy_guardrails": self._component_guardrails(plan["kind"]),
        }

    def _resolve_component_parent(self, ctx: Context, plan: Dict[str, Any]) -> Optional[Program]:
        """Pick the best visible parent for a component plan, with fallback."""
        accessor = ctx.island_accessor or ctx.accessor
        for pid in [
            str(plan.get("best_program_id") or ""),
            *[str(c) for c in reversed(plan.get("child_program_ids", []))],
            str(plan.get("seed_parent_id") or ""),
        ]:
            if pid:
                program = accessor.get_by_id(pid)
                if program is not None:
                    return program
        return self._get_current_best_program(ctx)

    def _read_component_experience(self, ctx: Context, component_key: Optional[str]) -> Dict[str, Any]:
        """Read accumulated experience for this component (compatible with ComponentJudge)."""
        if not component_key or ctx.state is None:
            return {}
        exp = ctx.state.get_island(
            self.COMPONENT_EXPERIENCE_NS, "component_experience", str(component_key), default={}
        )
        if not isinstance(exp, dict):
            return {}
        history = exp.get("history")
        semantic_history = []
        if isinstance(history, list):
            for item in history[-3:]:
                if isinstance(item, dict):
                    semantic_history.append(
                        {"component_name": item.get("component_name"), "summary": item.get("summary")}
                    )
        return {
            "component_name": exp.get("component_name"),
            "latest_summary": exp.get("latest_summary"),
            "history": semantic_history,
        }

    def _component_guardrails(self, kind: str) -> str:
        """Strategy-level generation constraints for component operations."""
        rules = [
            "严格保持 evaluator 期望的函数签名、输入参数和输出文件约定,不能私自改接口。",
            "优先在当前代码骨架基础上做有针对性的修改,不要无故整段重写。",
            "输出必须是一个自洽、可运行、端到端完整的实现。",
        ]
        if kind == self.COMPONENT_CROSSOVER:
            rules.append("当前是 component crossover:以 anchor 为主干,只吸收互补组件改进,不要拼接多个冲突主流程。")
        else:
            rules.append("当前是 component enhancement:只围绕当前组件方向做受控修改,核心算法主干应尽量稳定。")
        return "\n".join(f"- {r}" for r in rules)

    def _discover_component_specs(
        self, ctx: Context, island_state: Dict[str, Any], best_program: Program
    ) -> List[Dict[str, str]]:
        """Discover component directions via the LLM, with parse retries + fallback."""
        best_summary = {
            "implementation_plan": get_implementation_plan(best_program),
            "combined_score": best_program.combined_score,
            "metrics": dict(best_program.metrics or {}),
            "validity": best_program.validity,
            "eval_time": extract_eval_wall_time(best_program),
            "error_info": best_program.error_info,
        }
        existing_labels = list(island_state.get("known_component_keys", []))
        system_prompt = prompt_registry.get("planning/component_discovery_system.txt")
        base_prompt = prompt_registry.get(
            "planning/component_discovery.txt",
            task_description=ctx.task_description,
            best_program_json=json.dumps(best_summary, ensure_ascii=False, indent=2),
            prior_plans_json=json.dumps([], ensure_ascii=False),
            existing_component_labels_json=json.dumps(existing_labels, ensure_ascii=False),
            target_component_count=self.component_batch_size,
        )
        temperature = get_llm_temperature(self.llm_client)
        max_tokens = get_llm_max_tokens(self.llm_client)
        max_attempts = max(1, get_llm_max_retries(self.llm_client))

        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt if attempt == 1 else (
                base_prompt + "\n\nRETRY: Return exactly one JSON object with a 'components' "
                "array; each item needs component_key, component_name, component_goal, "
                "component_constraints, suggested_techniques."
            )
            try:
                response = self.llm_client.generate(
                    prompt=prompt, system=system_prompt,
                    temperature=temperature, max_tokens=max_tokens,
                )
                components = self._normalize_component_specs(extract_json(response.text))
                if components:
                    return components
            except Exception as exc:  # noqa: BLE001 — discovery is best-effort
                logger.warning("component discovery attempt %d failed: %s", attempt, exc)
        return self._fallback_component_specs(existing_labels)

    def _normalize_component_specs(self, parsed: Any) -> List[Dict[str, str]]:
        """Validate the discovery payload into clean component dicts."""
        if not isinstance(parsed, dict):
            return []
        raw = parsed.get("components")
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, str]] = []
        seen = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("component_name") or "").strip()
            goal = str(item.get("component_goal") or "").strip()
            constraints = str(item.get("component_constraints") or "").strip()
            techniques = str(item.get("suggested_techniques") or "").strip()
            raw_key = str(item.get("component_key") or name).strip().lower()
            key = "".join(c if c.isalnum() else "_" for c in raw_key).strip("_")[:64]
            if not (key and name and goal and constraints and techniques) or key in seen:
                continue
            out.append({
                "component_key": key,
                "component_name": name,
                "component_goal": goal,
                "component_constraints": constraints,
                "suggested_techniques": techniques,
            })
            seen.add(key)
        return out[: self.component_batch_size]

    def _fallback_component_specs(self, existing_keys: List[str]) -> List[Dict[str, str]]:
        """Generic component directions when discovery output is unusable."""
        candidates = [
            {
                "component_key": "core_algorithm",
                "component_name": "Core Algorithm",
                "component_goal": "Improve the core solving method without changing the I/O contract.",
                "component_constraints": "Keep the interface stable; focus on the main algorithm only.",
                "suggested_techniques": "alternative search, better heuristics, refined update rules",
            },
            {
                "component_key": "post_processing",
                "component_name": "Post Processing",
                "component_goal": "Refine the final output with decision-time logic.",
                "component_constraints": "Do not rewrite the main pipeline; inference-time only.",
                "suggested_techniques": "local refinement, constraint repair, fine-tuning of outputs",
            },
            {
                "component_key": "initialization",
                "component_name": "Initialization",
                "component_goal": "Improve the starting configuration / seeding.",
                "component_constraints": "Keep the main optimization loop stable.",
                "suggested_techniques": "smarter seeding, structured initial layouts, multi-start",
            },
            {
                "component_key": "parameter_tuning",
                "component_name": "Parameter Tuning",
                "component_goal": "Tune key hyperparameters / schedule for better convergence.",
                "component_constraints": "Change parameters only, not the method family.",
                "suggested_techniques": "schedule tuning, adaptive parameters, grid/heuristic search",
            },
        ]
        out = [c for c in candidates if c["component_key"] not in existing_keys]
        return out[: self.component_batch_size]

    # ------------------------------------------------------------------
    # Population helpers
    # ------------------------------------------------------------------

    def _get_visible_programs(self, ctx: Context) -> List[Program]:
        if ctx.island_accessor:
            return ctx.island_accessor.get_all()
        if ctx.accessor:
            return ctx.accessor.get_all()
        return []

    def _get_current_best_program(self, ctx: Context) -> Optional[Program]:
        """Return the highest-combined_score visible program (None if empty)."""
        visible = self._get_visible_programs(ctx)
        if not visible:
            return None
        return max(
            visible,
            key=lambda p: p.combined_score if p.combined_score is not None else float("-inf"),
        )


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """Registrable entrypoint for the pipeline strategy."""
    strategy = PipelineStrategy(
        evaluate_fn=evaluate_fn, params=params, evaluate_module=evaluate_module
    )
    return {
        "strategy": strategy,
        "description": "Population-centric strategy: UCB node selection + branch tree, single-round explore/exploit/crossover/fusion",
        "tags": ["population-centric", "ucb", "branch", "pipeline"],
        "author": "Famou Framework",
    }
