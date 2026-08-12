"""Plan-driven strategy with multi-mode planning and plan-level memory."""

from __future__ import annotations

import hashlib
import logging
import math
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from famou.config.settings import ModulesConfig
from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Context as StrategyContext
from famou.core.protocol import Rollout, Strategy
from famou.infrastructure.embedding.base import EmbeddingClient
from famou.infrastructure.llm.base import LLMClient
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.plan_driven_generate import PlanDrivenGenerate
from famou.modules.judge.llm_judge import LLMJudge
from famou.modules.planning.explore_plan import ExplorePlanModule
from famou.modules.planning.exploit_plan import ExploitPlanModule
from famou.modules.planning.crossover_plan import CrossoverPlanModule
from famou.modules.population.full_archive import FullArchivePopulation
from famou.modules.select.plan_scoped_select import PlanScopedSelect
from famou.utils.id_gen import generate_short_uuid
from famou.utils.math import cosine_similarity, stable_softmax
from famou.utils.program_summary import (
    build_program_summary,
    get_implementation_plan,
    get_improvement_directions,
    get_key_features,
)

logger = logging.getLogger("famou.strategy.plan_driven")

EXPLORE_PLAN = "explore_plan"
EXPLOIT_PLAN = "exploit_plan"
CROSSOVER_PLAN = "crossover_plan"


class PlanDrivenStrategy(Strategy):
    """Strategy that creates text plans and evolves within each plan for multiple rounds."""

    llm_client: Optional[LLMClient] = None
    embedding_client: Optional[EmbeddingClient] = None

    def __init__(
        self,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
        evaluate_module=None,
    ) -> None:
        """Initialize the strategy and its plan-driven rollout pipeline."""
        params = params or ModulesConfig()
        self.evaluate_fn = evaluate_fn

        select_params = {**params.select}
        generate_params = {**params.generate}
        evaluate_params = {**params.evaluate}
        judge_params = {**params.judge}
        population_params = {**params.population}

        self.base_plan_budget = 3
        self.bonus_budget_on_improvement = 2
        self.planner_history_window = 10
        self.exploit_history_window = 5
        self.crossover_history_window = 4
        self.relative_best_temperature = 1.0
        self.crossover_top_k = 4
        self.plan_ucb_exploration_weight = 0.4
        self.exploit_reference_count = 4
        self.explore_probability = 0.3
        self.exploit_probability = 0.5

        self.population_module = FullArchivePopulation(**population_params)

        self.exec_rollout = Rollout(
            modules=[
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                evaluate_module or EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="plan_driven_exploit",
        )
        self.explore_rollout = Rollout(
            modules=[
                ExplorePlanModule(),
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="plan_driven_explore",
        )
        self.exploit_rollout = Rollout(
            modules=[
                ExploitPlanModule(),
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="plan_driven_exploit_plan",
        )
        self.crossover_rollout = Rollout(
            modules=[
                CrossoverPlanModule(),
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="plan_driven_crossover",
        )

        # plan_driven 的内部状态:plans / active_plan_id / plan_order 等都是
        # strategy 私有,不跨边界与 rollout 通信(rollout 拿到的是 ctx.metadata
        # 中的快照)。因此放在实例属性里,通过 dump_state/load_state 持久化。
        self._islands: Dict[int, Dict[str, Any]] = {}
        self.logger = None

    def forward(self, ctx: StrategyContext, rollout_history: List[RolloutResult]) -> Rollout:
        """Prepare plan context for the current island and return the rollout to execute."""
        if ctx.iteration == 0 or not self._get_visible_programs(ctx):
            return self.exec_rollout

        island_state = self._ensure_island_state(ctx.island_id)
        self._reconcile_rollout_history(
            island_state=island_state,
            rollout_history=rollout_history,
        )
        active_plan, is_new, planner_inputs = self._ensure_active_plan(ctx, island_state)
        # 标记该 plan 进入 in-flight 状态,防止并发派发同一 plan
        self._mark_in_flight(active_plan)
        ctx.metadata["plan_context"] = self._build_plan_context(ctx, active_plan)

        if not is_new:
            return self.exec_rollout

        ctx.metadata["planner_inputs"] = planner_inputs
        return self._rollout_for_mode(planner_inputs.get("planner_mode", EXPLORE_PLAN))

    def dump_state(self) -> Dict[str, Any]:
        """Serialize strategy-private state for checkpoint persistence.

        plan_driven 的 plans / active_plan_id / plan_order / processed_rollout_count
        都是 strategy 私有(不与 rollout 通信),所以由本方法负责 dump,而不是放进
        StateStore(StateStore 只承载 strategy ↔ rollout / rollout ↔ rollout 的
        共享通信状态)。
        """
        return {
            "islands": self._islands,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy-private state from a previously saved checkpoint."""
        islands = state.get("islands", {})
        if isinstance(islands, dict):
            self._islands = {
                int(island_id): island_state
                for island_id, island_state in islands.items()
            }

    def on_rollout_complete(self, result: RolloutResult) -> None:
        """Commit: 应用 rollout 结果,释放 plan 占用,更新 round_count。"""
        island_state = self._ensure_island_state(result.island_id)
        self._release_in_flight(island_state, result)
        self._apply_rollout_result(island_state, result)
        island_state["processed_rollout_count"] = int(
            island_state.get("processed_rollout_count", 0)
        ) + 1

    def on_rollout_failed(self, result: RolloutResult) -> None:
        """Rollback: rollout 被 discard,释放 plan 占用,不计入 round_count。"""
        island_state = self._ensure_island_state(result.island_id)
        self._release_in_flight(island_state, result)

    def _mark_in_flight(self, plan: Dict[str, Any]) -> None:
        """标记 plan 为 in-flight 状态,防止并发派发同一 plan。

        与 _release_in_flight 配对使用,构成 plan 级别的事务锁:
          forward()           -> _mark_in_flight()    (占用)
          on_rollout_complete -> _release_in_flight() (确认/释放)
          on_rollout_failed   -> _release_in_flight() (回退/释放)
        """
        plan["in_flight"] = True

    def finalize_experiment(self, contexts: Dict[int, Context]) -> None:
        """Finalize any still-active plans so the final checkpoint contains embeddings."""
        for island_id, island_state in self._islands.items():
            active_plan_id = island_state.get("active_plan_id")
            if not active_plan_id:
                continue
            active_plan = island_state.get("plans", {}).get(active_plan_id)
            context = contexts.get(island_id)
            if not isinstance(active_plan, dict) or context is None:
                continue
            self._finalize_plan(
                context,
                island_state,
                active_plan,
                closed_iteration=context.iteration,
            )
            island_state["active_plan_id"] = None

    def _release_in_flight(self, island_state: Dict[str, Any], result: RolloutResult) -> None:
        """Clear in_flight flag on the plan associated with this rollout result."""
        selection_extra = result.selection.extra if result.selection else {}
        plan_context = (
            selection_extra.get("plan_context") if isinstance(selection_extra, dict) else None
        )
        if not isinstance(plan_context, dict):
            return
        plan_id = plan_context.get("plan_id")
        if not plan_id:
            return
        plan = island_state.get("plans", {}).get(plan_id)
        if isinstance(plan, dict):
            plan["in_flight"] = False

    def _ensure_island_state(self, island_id: int) -> Dict[str, Any]:
        """Return mutable per-island planner state, creating it on first access."""
        if island_id not in self._islands:
            self._islands[island_id] = {
                "active_plan_id": None,
                "plans": {},
                "plan_order": [],
                "processed_rollout_count": 0,
            }
        return self._islands[island_id]

    def _reconcile_rollout_history(
        self,
        *,
        island_state: Dict[str, Any],
        rollout_history: List[RolloutResult],
    ) -> None:
        """Replay unseen rollout results into island state before dispatching new work."""
        start_index = min(
            int(island_state.get("processed_rollout_count", 0)),
            len(rollout_history),
        )
        for result in rollout_history[start_index:]:
            self._apply_rollout_result(island_state, result)
        island_state["processed_rollout_count"] = len(rollout_history)

    def _apply_rollout_result(
        self,
        island_state: Dict[str, Any],
        result: RolloutResult,
    ) -> None:
        """Update one plan with the outcome of a completed rollout."""
        selection_extra = result.selection.extra if result.selection else {}
        plan_context = (
            selection_extra.get("plan_context") if isinstance(selection_extra, dict) else None
        )
        if not isinstance(plan_context, dict):
            return

        plan_id = plan_context.get("plan_id")
        if not plan_id:
            return

        plan = island_state["plans"].get(plan_id)
        if plan is None:
            plan = {
                "plan_id": plan_id,
                "planner_mode": plan_context.get("planner_mode", EXPLORE_PLAN),
                "plan_text": plan_context.get("plan_text", ""),
                "status": "active",
                "created_iteration": plan_context.get("created_iteration"),
                "closed_iteration": None,
                "seed_parent_id": plan_context.get("seed_parent_id"),
                "anchor_program_id": plan_context.get("anchor_program_id"),
                "anchor_score": plan_context.get("anchor_score"),
                "anchor_metrics": dict(plan_context.get("anchor_metrics") or {}),
                "anchor_validity": plan_context.get("anchor_validity"),
                "anchor_implementation_plan": plan_context.get(
                    "anchor_implementation_plan", ""
                ),
                "anchor_key_features": plan_context.get("anchor_key_features", ""),
                "anchor_improvement_directions": plan_context.get(
                    "anchor_improvement_directions", ""
                ),
                "best_program_id": plan_context.get("best_program_id"),
                "best_score": plan_context.get("best_score"),
                "best_metrics": dict(plan_context.get("best_metrics") or {}),
                "best_validity": plan_context.get("best_validity"),
                "best_implementation_plan": plan_context.get("best_implementation_plan", ""),
                "best_key_features": plan_context.get("best_key_features", ""),
                "best_improvement_directions": plan_context.get(
                    "best_improvement_directions", ""
                ),
                "base_budget": self.base_plan_budget,
                "remaining_budget": self.base_plan_budget,
                "bonus_rounds_granted": 0,
                "round_count": 0,
                "child_program_ids": [],
                "planner_trace": plan_context.get("planner_trace"),
                "planner_focus_plan_id": plan_context.get("planner_focus_plan_id"),
                "planner_history_plan_ids": list(
                    plan_context.get("planner_history_plan_ids", [])
                ),
                "planner_crossover_plan_ids": list(
                    plan_context.get("planner_crossover_plan_ids", [])
                ),
                "rollout_ids": [],
                "best_feasible_rollout": None,
                "last_rollout": None,
                "finalized": False,
                "final_representative_summary": None,
                "plan_embedding": None,
                "plan_embedding_trace": None,
                "plan_similarities": {},
                "plan_distances": {},
                "plan_diversity": 0.0,
                "in_flight": False,
            }
            island_state["plans"][plan_id] = plan
            island_state["plan_order"].append(plan_id)

        if result.rollout_id not in plan["rollout_ids"]:
            plan["rollout_ids"].append(result.rollout_id)
            plan["round_count"] += 1
            plan["remaining_budget"] = max(0, int(plan.get("remaining_budget", 0)) - 1)

        # Persist plan_text written by PlanningModule during a planning rollout
        plan_text_from_result = plan_context.get("plan_text", "")
        if plan_text_from_result and not plan.get("plan_text"):
            plan["plan_text"] = plan_text_from_result
            planner_trace = plan_context.get("planner_trace")
            if planner_trace is not None:
                plan["planner_trace"] = planner_trace

        child = result.generated_program
        if child and child.id not in plan["child_program_ids"]:
            plan["child_program_ids"].append(child.id)

        if child is not None:
            child_summary = self._build_plan_rollout_summary(child, result.rollout_id)
            plan["last_rollout"] = child_summary
            if child.validity is not None and child.validity > 0:
                best_feasible = plan.get("best_feasible_rollout")
                best_feasible_score = (
                    best_feasible.get("combined_score")
                    if isinstance(best_feasible, dict)
                    else None
                )
                if (
                    not isinstance(best_feasible, dict)
                    or best_feasible_score is None
                    or (
                        child.combined_score is not None
                        and child.combined_score > best_feasible_score
                    )
                ):
                    plan["best_feasible_rollout"] = child_summary

        child_score = child.combined_score if child is not None else None
        best_score = plan.get("best_score")
        if child is not None and child_score is not None and (
            best_score is None or child_score > best_score
        ):
            plan["best_program_id"] = child.id
            plan["best_score"] = child_score
            plan["best_metrics"] = dict(child.metrics or {})
            plan["best_validity"] = child.validity
            plan["best_implementation_plan"] = get_implementation_plan(child)
            plan["best_key_features"] = get_key_features(child)
            plan["best_improvement_directions"] = get_improvement_directions(child)
            plan["finalized"] = False
            plan["plan_embedding"] = None
            plan["plan_embedding_trace"] = None
            plan["plan_similarities"] = {}
            plan["plan_distances"] = {}
            plan["plan_diversity"] = 0.0
            plan["final_representative_summary"] = None

        island_best_score_at_dispatch = plan_context.get("island_best_score_at_dispatch")
        if (
            child is not None
            and child_score is not None
            and (
                island_best_score_at_dispatch is None
                or child_score > island_best_score_at_dispatch
            )
        ):
            plan["remaining_budget"] += self.bonus_budget_on_improvement
            plan["bonus_rounds_granted"] = int(plan.get("bonus_rounds_granted", 0)) + self.bonus_budget_on_improvement
            plan["last_bonus_iteration"] = result.iteration
            plan["last_bonus_reason"] = "improved_island_best_score"

    def _rollout_for_mode(self, mode: str) -> Rollout:
        """Return the planning rollout that matches the given planner mode."""
        return {
            EXPLORE_PLAN: self.explore_rollout,
            EXPLOIT_PLAN: self.exploit_rollout,
            CROSSOVER_PLAN: self.crossover_rollout,
        }.get(mode, self.explore_rollout)

    def _ensure_active_plan(
        self, ctx: Context, island_state: Dict[str, Any]
    ) -> tuple:
        """Return (plan, is_new, planner_inputs_or_None), creating a skeleton when needed."""
        active_plan_id = island_state.get("active_plan_id")
        if active_plan_id:
            active_plan = island_state["plans"].get(active_plan_id)
            if (active_plan
                    and active_plan.get("remaining_budget", 0) > 0
                    and not active_plan.get("in_flight", False)):
                return active_plan, False, None
            if active_plan and not active_plan.get("in_flight", False):
                self._finalize_plan(
                    ctx,
                    island_state,
                    active_plan,
                    closed_iteration=ctx.iteration - 1,
                )
            island_state["active_plan_id"] = None

        new_plan, planner_inputs = self._create_plan_skeleton(ctx, island_state)
        island_state["plans"][new_plan["plan_id"]] = new_plan
        island_state["plan_order"].append(new_plan["plan_id"])
        island_state["active_plan_id"] = new_plan["plan_id"]
        return new_plan, True, planner_inputs

    def _create_plan_skeleton(
        self, ctx: Context, island_state: Dict[str, Any]
    ) -> tuple:
        """Create a plan dict with empty plan_text and return (plan, planner_inputs).

        The actual plan_text is generated later by PlanningModule during the rollout.
        """
        planner_inputs = self._prepare_planner_inputs(ctx, island_state)
        anchor_program = planner_inputs["anchor_program"]
        return {
            "plan_id": f"plan_{ctx.island_id}_{ctx.iteration}_{generate_short_uuid()}",
            "planner_mode": planner_inputs["planner_mode"],
            "plan_text": "",
            "status": "active",
            "created_iteration": ctx.iteration,
            "closed_iteration": None,
            "seed_parent_id": anchor_program.id,
            "anchor_program_id": anchor_program.id,
            "anchor_score": anchor_program.combined_score,
            "anchor_metrics": dict(anchor_program.metrics or {}),
            "anchor_validity": anchor_program.validity,
            "anchor_implementation_plan": get_implementation_plan(anchor_program),
            "anchor_key_features": get_key_features(anchor_program),
            "anchor_improvement_directions": get_improvement_directions(anchor_program),
            "best_program_id": None,
            "best_score": None,
            "best_metrics": {},
            "best_validity": None,
            "best_implementation_plan": "",
            "best_key_features": "",
            "best_improvement_directions": "",
            "base_budget": self.base_plan_budget,
            "remaining_budget": self.base_plan_budget,
            "bonus_rounds_granted": 0,
            "round_count": 0,
            "child_program_ids": [],
            "planner_trace": None,
            "planner_focus_plan_id": planner_inputs.get("focus_plan_id"),
            "planner_history_plan_ids": list(planner_inputs.get("history_plan_ids", [])),
            "planner_crossover_plan_ids": list(
                planner_inputs.get("crossover_plan_ids", [])
            ),
            "rollout_ids": [],
            "best_feasible_rollout": None,
            "last_rollout": None,
            "finalized": False,
            "final_representative_summary": None,
            "plan_embedding": None,
            "plan_embedding_trace": None,
            "plan_similarities": {},
            "plan_distances": {},
            "plan_diversity": 0.0,
            "in_flight": False,
        }, planner_inputs

    def _build_plan_context(self, ctx: Context, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build the selection metadata consumed by plan-scoped rollout modules."""
        island_best_program = self._get_current_best_program(ctx)
        current_parent_id = (
            plan.get("best_program_id")
            or plan.get("seed_parent_id")
            or plan.get("anchor_program_id")
        )
        candidate_ids = [
            plan.get("anchor_program_id"),
            plan.get("seed_parent_id"),
            *(plan.get("child_program_ids") or []),
        ]
        plan_program_ids = list(
            dict.fromkeys(
                str(pid).strip()
                for pid in candidate_ids
                if isinstance(pid, str) and str(pid).strip()
            )
        )
        return {
            "plan_id": plan["plan_id"],
            "planner_mode": plan.get("planner_mode", EXPLORE_PLAN),
            "plan_text": plan["plan_text"],
            "plan_round": int(plan.get("round_count", 0)) + 1,
            "created_iteration": plan["created_iteration"],
            "seed_parent_id": plan["seed_parent_id"],
            "anchor_program_id": plan.get("anchor_program_id"),
            "anchor_score": plan.get("anchor_score"),
            "anchor_metrics": dict(plan.get("anchor_metrics") or {}),
            "anchor_validity": plan.get("anchor_validity"),
            "anchor_implementation_plan": plan.get("anchor_implementation_plan", ""),
            "anchor_key_features": plan.get("anchor_key_features", ""),
            "anchor_improvement_directions": plan.get(
                "anchor_improvement_directions", ""
            ),
            "current_parent_id": current_parent_id,
            "best_program_id": plan["best_program_id"],
            "best_score": plan.get("best_score"),
            "best_metrics": dict(plan.get("best_metrics") or {}),
            "best_validity": plan.get("best_validity"),
            "best_implementation_plan": plan.get("best_implementation_plan", ""),
            "best_key_features": plan.get("best_key_features", ""),
            "best_improvement_directions": plan.get("best_improvement_directions", ""),
            "island_best_score_at_dispatch": island_best_program.combined_score,
            "plan_program_ids": plan_program_ids,
            "remaining_budget": plan.get("remaining_budget", 0),
            "planner_trace": plan.get("planner_trace"),
            "planner_focus_plan_id": plan.get("planner_focus_plan_id"),
            "planner_history_plan_ids": list(plan.get("planner_history_plan_ids", [])),
            "planner_crossover_plan_ids": list(
                plan.get("planner_crossover_plan_ids", [])
            ),
        }

    def _prepare_planner_inputs(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assemble planner mode, anchor program, and plan summaries for a new plan."""
        visible_by_plan = self._get_visible_programs_by_plan(ctx)
        historical_plans = [
            island_state["plans"][plan_id]
            for plan_id in island_state.get("plan_order", [])
            if plan_id in island_state["plans"]
        ]
        planner_mode = self._choose_planner_mode(ctx, len(historical_plans))
        current_best_program = self._get_current_best_program(ctx)

        if planner_mode == EXPLORE_PLAN:
            history_summaries = self._select_recent_plan_summaries(
                island_state,
                visible_by_plan=visible_by_plan,
                window=self.planner_history_window,
            )
            return {
                "planner_mode": EXPLORE_PLAN,
                "anchor_program": current_best_program,
                "focus_plan_id": None,
                "focus_plan_summary": None,
                "history_plan_ids": [item.get("plan_id") for item in history_summaries],
                "history_plan_summaries": history_summaries,
                "crossover_plan_ids": [],
                "crossover_plan_summaries": [],
            }

        ucb_parent = self._select_ucb_parent_plan(
            ctx,
            historical_plans=historical_plans,
            visible_by_plan=visible_by_plan,
        )
        if ucb_parent is None:
            history_summaries = self._select_recent_plan_summaries(
                island_state,
                visible_by_plan=visible_by_plan,
                window=self.planner_history_window,
            )
            return {
                "planner_mode": EXPLORE_PLAN,
                "anchor_program": self._get_current_best_program(ctx),
                "focus_plan_id": None,
                "focus_plan_summary": None,
                "history_plan_ids": [item.get("plan_id") for item in history_summaries],
                "history_plan_summaries": history_summaries,
                "crossover_plan_ids": [],
                "crossover_plan_summaries": [],
            }

        focus_plan, anchor_program = ucb_parent
        focus_plan_id = focus_plan["plan_id"]
        focus_plan_summary = self._build_plan_summary(
            focus_plan,
            visible_by_plan=visible_by_plan,
        )

        if planner_mode == EXPLOIT_PLAN:
            reference_plans = self._sample_relative_best_plans(
                ctx,
                historical_plans=historical_plans,
                visible_by_plan=visible_by_plan,
                sample_size=self.exploit_reference_count,
                excluded_plan_ids=[focus_plan_id],
            )
            history_summaries = [
                self._build_plan_summary(plan, visible_by_plan=visible_by_plan)
                for plan in reference_plans
            ]
            return {
                "planner_mode": EXPLOIT_PLAN,
                "anchor_program": anchor_program,
                "focus_plan_id": focus_plan_id,
                "focus_plan_summary": focus_plan_summary,
                "history_plan_ids": [item.get("plan_id") for item in history_summaries],
                "history_plan_summaries": history_summaries,
                "crossover_plan_ids": [],
                "crossover_plan_summaries": [],
            }

        crossover_summaries = self._select_crossover_plan_summaries(
            island_state,
            visible_by_plan=visible_by_plan,
            focus_plan_id=focus_plan_id,
            limit=self.crossover_top_k,
        )
        if not crossover_summaries:
            reference_plans = self._sample_relative_best_plans(
                ctx,
                historical_plans=historical_plans,
                visible_by_plan=visible_by_plan,
                sample_size=self.exploit_reference_count,
                excluded_plan_ids=[focus_plan_id],
            )
            history_summaries = [
                self._build_plan_summary(plan, visible_by_plan=visible_by_plan)
                for plan in reference_plans
            ]
            return {
                "planner_mode": EXPLOIT_PLAN,
                "anchor_program": anchor_program,
                "focus_plan_id": focus_plan_id,
                "focus_plan_summary": focus_plan_summary,
                "history_plan_ids": [item.get("plan_id") for item in history_summaries],
                "history_plan_summaries": history_summaries,
                "crossover_plan_ids": [],
                "crossover_plan_summaries": [],
            }

        return {
            "planner_mode": CROSSOVER_PLAN,
            "anchor_program": anchor_program,
            "focus_plan_id": focus_plan_id,
            "focus_plan_summary": focus_plan_summary,
            "history_plan_ids": [],
            "history_plan_summaries": [],
            "crossover_plan_ids": [item.get("plan_id") for item in crossover_summaries],
            "crossover_plan_summaries": crossover_summaries,
        }

    def _choose_planner_mode(self, ctx: Context, historical_plan_count: int) -> str:
        """Select planner mode using the requested staged probability policy."""
        if historical_plan_count < 5:
            return EXPLORE_PLAN

        rng = self._make_decision_rng(ctx, "planner_mode")
        if rng.random() < self.explore_probability:
            return EXPLORE_PLAN
        return EXPLOIT_PLAN if rng.random() < self.exploit_probability else CROSSOVER_PLAN

    def _select_ucb_parent_plan(
        self,
        ctx: Context,
        *,
        historical_plans: Sequence[Dict[str, Any]],
        visible_by_plan: Dict[str, List[Program]],
    ) -> Optional[Tuple[Dict[str, Any], Program]]:
        """Select one alive focus plan using a UCB score over plan quality and visits."""
        candidates: List[Tuple[Dict[str, Any], Program]] = []
        for plan in historical_plans:
            representative_program = self._get_alive_representative_program(
                plan,
                visible_by_plan,
            )
            if representative_program is None:
                continue
            candidates.append((plan, representative_program))

        if not candidates:
            return None

        total_rounds = sum(max(1, int(plan.get("round_count", 0))) for plan, _ in candidates)
        scored_candidates = []
        for plan, representative_program in candidates:
            plan_score = float(plan.get("best_score") or 0.0)
            rounds = max(1, int(plan.get("round_count", 0)))
            exploration_bonus = self.plan_ucb_exploration_weight * math.sqrt(
                math.log(total_rounds + 1.0) / rounds
            )
            scored_candidates.append(
                (
                    plan_score + exploration_bonus,
                    plan_score,
                    int(plan.get("created_iteration") or -1),
                    plan,
                    representative_program,
                )
            )

        scored_candidates.sort(reverse=True, key=lambda item: (item[0], item[1], item[2]))
        _, _, _, plan, representative_program = scored_candidates[0]
        return plan, representative_program

    def _sample_relative_best_plans(
        self,
        ctx: Context,
        *,
        historical_plans: Sequence[Dict[str, Any]],
        visible_by_plan: Dict[str, List[Program]],
        sample_size: int,
        excluded_plan_ids: Sequence[str] = (),
    ) -> List[Dict[str, Any]]:
        """Sample multiple plans without replacement using softmax over plan best_score."""
        del visible_by_plan
        selected: List[Dict[str, Any]] = []
        excluded = {plan_id for plan_id in excluded_plan_ids if plan_id}
        candidates = [
            plan
            for plan in historical_plans
            if plan.get("plan_id") not in excluded and self._get_plan_score(plan) is not None
        ]
        if not candidates or sample_size <= 0:
            return selected

        rng = self._make_decision_rng(ctx, "relative_best_plan_batch")
        remaining = list(candidates)
        while remaining and len(selected) < sample_size:
            scores = [float(self._get_plan_score(plan) or 0.0) for plan in remaining]
            probabilities = stable_softmax(
                scores,
                temperature=self.relative_best_temperature,
            ).tolist()
            threshold = rng.random()
            cumulative = 0.0
            chosen_index = len(remaining) - 1
            for index, probability in enumerate(probabilities):
                cumulative += probability
                if threshold <= cumulative:
                    chosen_index = index
                    break
            selected.append(remaining.pop(chosen_index))
        return selected

    def _select_recent_plan_summaries(
        self,
        island_state: Dict[str, Any],
        *,
        visible_by_plan: Dict[str, List[Program]],
        window: int,
    ) -> List[Dict[str, Any]]:
        """Return the latest completed plan summaries for explore-mode direction planning."""
        if window <= 0:
            return []
        recent_ids = [
            plan_id
            for plan_id in island_state.get("plan_order", [])
            if plan_id in island_state["plans"]
            and self._get_plan_score(island_state["plans"][plan_id]) is not None
        ][-window:]
        return [
            self._build_plan_summary(island_state["plans"][plan_id], visible_by_plan=visible_by_plan)
            for plan_id in recent_ids
        ]

    def _select_crossover_plan_summaries(
        self,
        island_state: Dict[str, Any],
        *,
        visible_by_plan: Dict[str, List[Program]],
        focus_plan_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return the alive historical plans most dissimilar to the focus plan."""
        focus_plan = island_state["plans"].get(focus_plan_id)
        if not isinstance(focus_plan, dict):
            return []

        focus_similarities = dict(focus_plan.get("plan_similarities") or {})
        candidates: List[Tuple[float, Dict[str, Any]]] = []
        for plan_id in island_state.get("plan_order", []):
            if plan_id == focus_plan_id or plan_id not in island_state["plans"]:
                continue
            if self._get_alive_representative_program(
                island_state["plans"][plan_id],
                visible_by_plan,
            ) is None:
                continue
            similarity = focus_similarities.get(plan_id)
            if similarity is None:
                continue
            candidates.append((1.0 - float(similarity), island_state["plans"][plan_id]))

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = []
        for distance, plan in candidates[: limit or self.crossover_top_k]:
            summary = self._build_plan_summary(plan, visible_by_plan=visible_by_plan)
            summary["similarity_to_focus"] = 1.0 - distance
            summary["distance_to_focus"] = distance
            selected.append(summary)
        return selected

    def _get_plan_score(self, plan: Dict[str, Any]) -> Optional[float]:
        """Return the score used for plan-level sampling."""
        score = plan.get("best_score")
        if score is None:
            return None
        return float(score)

    def _finalize_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        plan: Dict[str, Any],
        *,
        closed_iteration: int,
    ) -> None:
        """Close a plan and update its plan-level embedding/similarity cache once."""
        plan["status"] = "completed"
        if plan.get("closed_iteration") is None:
            plan["closed_iteration"] = closed_iteration
        if plan.get("finalized"):
            return

        visible_by_plan = self._get_visible_programs_by_plan(ctx)
        representative_summary = self._build_plan_summary(
            plan,
            visible_by_plan=visible_by_plan,
        )
        plan["final_representative_summary"] = representative_summary
        implementation_plan = str(plan.get("best_implementation_plan", "")).strip()

        if self.embedding_client is not None and implementation_plan:
            embedding, embedding_trace = self._embed_plan_with_fallback(
                plan_id=str(plan.get("plan_id") or ""),
                implementation_plan=implementation_plan,
            )
            plan["plan_embedding"] = embedding
            plan["plan_embedding_trace"] = embedding_trace

        plan["finalized"] = True
        self._refresh_plan_similarity_cache(island_state)

    def _embed_plan_with_fallback(
        self,
        *,
        plan_id: str,
        implementation_plan: str,
    ) -> Tuple[List[float], Dict[str, Any]]:
        """Embed one finalized plan with retry and deterministic random fallback."""
        if self.embedding_client is None:
            raise ValueError("embedding_client is required for plan embedding")

        attempts: List[Dict[str, Any]] = []
        sleep_seconds = 1.0
        started_at = time.time()
        max_attempts = 3
        model_name = getattr(self.embedding_client, "model_name", None)
        configured_dimensions = int(getattr(self.embedding_client, "dimensions", 0) or 0)

        for attempt in range(1, max_attempts + 1):
            attempt_started_at = time.time()
            try:
                response = self.embedding_client.embed(implementation_plan)
                completed_at = time.time()
                attempts.append(
                    {
                        "attempt": attempt,
                        "success": True,
                        "started_at": attempt_started_at,
                        "completed_at": completed_at,
                        "latency_ms": int((completed_at - attempt_started_at) * 1000),
                        "response": {
                            "model": response.model,
                            "dimensions": response.dimensions,
                            "usage": response.usage,
                            "raw_response": response.raw_response,
                        },
                    }
                )
                return response.embedding, {
                    "request": {
                        "text": implementation_plan,
                        "model": model_name,
                        "dimensions": configured_dimensions or response.dimensions,
                    },
                    "response": {
                        "model": response.model,
                        "dimensions": response.dimensions,
                        "usage": response.usage,
                        "raw_response": response.raw_response,
                    },
                    "timing": {
                        "started_at": started_at,
                        "completed_at": completed_at,
                    },
                    "attempts": attempts,
                    "fallback_applied": False,
                }
            except Exception as exc:
                completed_at = time.time()
                attempts.append(
                    {
                        "attempt": attempt,
                        "success": False,
                        "started_at": attempt_started_at,
                        "completed_at": completed_at,
                        "latency_ms": int((completed_at - attempt_started_at) * 1000),
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                )
                if attempt < max_attempts:
                    time.sleep(sleep_seconds)
                    sleep_seconds *= 2.0

        dimensions = configured_dimensions or 1536
        fallback_embedding = self._make_fallback_plan_embedding(
            plan_id=plan_id,
            implementation_plan=implementation_plan,
            dimensions=dimensions,
        )
        completed_at = time.time()
        return fallback_embedding, {
            "request": {
                "text": implementation_plan,
                "model": model_name,
                "dimensions": dimensions,
            },
            "response": {
                "model": model_name,
                "dimensions": dimensions,
                "usage": {},
                "raw_response": None,
            },
            "timing": {
                "started_at": started_at,
                "completed_at": completed_at,
            },
            "attempts": attempts,
            "fallback_applied": True,
            "fallback_reason": "embedding_request_failed_after_retries",
            "fallback_vector_source": "deterministic_random",
        }

    def _make_fallback_plan_embedding(
        self,
        *,
        plan_id: str,
        implementation_plan: str,
        dimensions: int,
    ) -> List[float]:
        """Create a deterministic pseudo-random embedding with the requested shape."""
        safe_dimensions = max(1, int(dimensions))
        seed_payload = f"{plan_id}::{implementation_plan}".encode("utf-8")
        seed_value = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16)
        rng = random.Random(seed_value)
        return [rng.uniform(-1.0, 1.0) for _ in range(safe_dimensions)]

    def _refresh_plan_similarity_cache(self, island_state: Dict[str, Any]) -> None:
        """Refresh pairwise plan similarity and diversity for finalized plans."""
        embedded_plans = [
            plan
            for plan in island_state["plans"].values()
            if isinstance(plan, dict) and plan.get("plan_embedding")
        ]
        for plan in island_state["plans"].values():
            if isinstance(plan, dict):
                plan["plan_similarities"] = {}
                plan["plan_distances"] = {}
                plan["plan_diversity"] = 0.0

        for i, left in enumerate(embedded_plans):
            left_similarities: Dict[str, float] = {}
            left_distances: Dict[str, float] = {}
            diversity = 0.0
            for j, right in enumerate(embedded_plans):
                if i == j:
                    continue
                similarity = cosine_similarity(
                    left.get("plan_embedding"),
                    right.get("plan_embedding"),
                )
                distance = max(0.0, min(1.0, 1.0 - float(similarity)))
                if abs(distance) < 1e-12:
                    distance = 0.0
                left_similarities[right["plan_id"]] = float(similarity)
                left_distances[right["plan_id"]] = distance
                diversity += distance
            left["plan_similarities"] = left_similarities
            left["plan_distances"] = left_distances
            left["plan_diversity"] = diversity

    def _build_plan_summary(
        self,
        plan: Dict[str, Any],
        *,
        visible_by_plan: Dict[str, List[Program]],
    ) -> Dict[str, Any]:
        """Build the planner-facing summary for one plan."""
        representative_program = self._get_alive_representative_program(plan, visible_by_plan)
        if representative_program is not None:
            summary = build_program_summary(representative_program, include_code=False)
            summary.pop("key_features", None)
            return {
                "plan_id": plan.get("plan_id"),
                "planner_mode": plan.get("planner_mode", EXPLORE_PLAN),
                "status": plan.get("status"),
                "created_iteration": plan.get("created_iteration"),
                "closed_iteration": plan.get("closed_iteration"),
                "round_count": int(plan.get("round_count", 0)),
                "bonus_rounds_granted": int(plan.get("bonus_rounds_granted", 0)),
                "plan_text": plan.get("plan_text", ""),
                "alive": True,
                "representative_source": "alive_best_program",
                "representative_program_id": summary.get("program_id"),
                "representative_implementation_plan": summary.get("implementation_plan", ""),
                "representative_combined_score": summary.get("combined_score"),
                "representative_metrics": dict(summary.get("metrics") or {}),
                "representative_validity": summary.get("validity"),
                "representative_error_info": summary.get("error_info"),
                "representative_improvement_directions": summary.get("improvement_directions", ""),
                "plan_diversity": float(plan.get("plan_diversity", 0.0)),
                "plan_similarities": dict(plan.get("plan_similarities") or {}),
                "plan_distances": dict(plan.get("plan_distances") or {}),
            }

        stored_summary = plan.get("final_representative_summary")
        if isinstance(stored_summary, dict):
            summary = dict(stored_summary)
            summary["alive"] = False
            summary["plan_diversity"] = float(plan.get("plan_diversity", 0.0))
            summary["plan_similarities"] = dict(plan.get("plan_similarities") or {})
            summary["plan_distances"] = dict(plan.get("plan_distances") or {})
            return summary

        representative = self._get_stored_representative_rollout(plan)
        return {
            "plan_id": plan.get("plan_id"),
            "planner_mode": plan.get("planner_mode", EXPLORE_PLAN),
            "status": plan.get("status"),
            "created_iteration": plan.get("created_iteration"),
            "closed_iteration": plan.get("closed_iteration"),
            "round_count": int(plan.get("round_count", 0)),
            "bonus_rounds_granted": int(plan.get("bonus_rounds_granted", 0)),
            "plan_text": plan.get("plan_text", ""),
            "alive": False,
            "representative_source": representative.get("source", "none"),
            "representative_program_id": representative.get("program_id"),
            "representative_implementation_plan": representative.get("implementation_plan", ""),
            "representative_combined_score": representative.get("combined_score"),
            "representative_metrics": dict(representative.get("metrics") or {}),
            "representative_validity": representative.get("validity"),
            "representative_error_info": representative.get("error_info"),
            "representative_improvement_directions": representative.get(
                "improvement_directions", ""
            ),
            "plan_diversity": float(plan.get("plan_diversity", 0.0)),
            "plan_similarities": dict(plan.get("plan_similarities") or {}),
            "plan_distances": dict(plan.get("plan_distances") or {}),
        }

    def _get_alive_representative_program(
        self,
        plan: Dict[str, Any],
        visible_by_plan: Dict[str, List[Program]],
    ) -> Optional[Program]:
        """Return the best currently visible program for a plan."""
        plan_id = str(plan.get("plan_id") or "").strip()
        if not plan_id:
            return None
        programs = visible_by_plan.get(plan_id) or []
        if not programs:
            return None
        return self._select_best_program(programs)

    def _get_visible_programs_by_plan(self, ctx: Context) -> Dict[str, List[Program]]:
        """Group currently visible programs by their originating plan id."""
        visible_by_plan: Dict[str, List[Program]] = {}
        for program in self._get_visible_programs(ctx):
            plan_id = str((program.meta or {}).get("plan_id") or "").strip()
            if not plan_id:
                continue
            visible_by_plan.setdefault(plan_id, []).append(program)
        return visible_by_plan

    def _get_stored_representative_rollout(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Return the best stored representative summary for a plan."""
        best_feasible = plan.get("best_feasible_rollout")
        if isinstance(best_feasible, dict):
            return {
                "source": "best_feasible_rollout",
                **best_feasible,
            }

        last_rollout = plan.get("last_rollout")
        if isinstance(last_rollout, dict):
            return {
                "source": "last_rollout",
                **last_rollout,
            }

        return {
            "source": "plan_seed_state",
            "program_id": plan.get("best_program_id"),
            "implementation_plan": "",
            "combined_score": plan.get("best_score"),
            "metrics": dict(plan.get("best_metrics") or {}),
            "validity": plan.get("best_validity"),
            "error_info": None,
            "improvement_directions": plan.get("best_improvement_directions", ""),
        }

    def _select_best_program(self, programs: Sequence[Program]) -> Program:
        """Choose the strongest program by score, validity, and recency."""
        return max(
            programs,
            key=lambda program: (
                program.combined_score if program.combined_score is not None else float("-inf"),
                program.validity if program.validity is not None else float("-inf"),
                program.iteration,
                program.generation,
            ),
        )

    def _make_decision_rng(self, ctx: Context, label: str) -> random.Random:
        """Create deterministic randomness for planner decisions and resume safety."""
        base_seed = getattr(ctx.experiment_config, "seed", 0) if ctx.experiment_config else 0
        payload = f"{base_seed}:{ctx.experiment_id}:{ctx.island_id}:{ctx.iteration}:{label}"
        seed_value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(seed_value)

    def _get_current_best_program(self, ctx: Context) -> Program:
        """Return the currently best visible program for the active island view."""
        programs = self._get_visible_programs(ctx)
        if not programs:
            raise ValueError("PlanDrivenStrategy: no visible programs available")
        return self._select_best_program(programs)

    def _get_visible_programs(self, ctx: Context) -> List[Program]:
        """List programs visible to the strategy, preferring island-local population."""
        if ctx.island_accessor:
            return list(ctx.island_accessor)
        return ctx.get_all_programs()

    def _build_plan_rollout_summary(
        self, program: Program, rollout_id: Optional[str]
    ) -> Dict[str, Any]:
        """Create the compact rollout summary stored in per-plan history."""
        summary = build_program_summary(program, include_code=False)
        summary.pop("key_features", None)
        summary["rollout_id"] = rollout_id
        return summary


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Create the registrable plan-driven strategy entrypoint."""
    strategy = PlanDrivenStrategy(evaluate_fn=evaluate_fn, params=params, evaluate_module=evaluate_module)
    return {
        "strategy": strategy,
        "description": "Planner-created text plans with explore/exploit/crossover modes",
        "tags": ["planning", "adaptive", "plan-driven"],
        "author": "Famou Framework",
    }
