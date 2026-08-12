"""Machine-learning pipeline strategy with a strategy-local ready queue and no-bubble scheduling.

Overall idea:

    1. Keep a per-island ready queue inside this strategy only.
    2. `forward()` replenishes the queue when it gets shallow.
    3. Replenishment happens by creating new plans immediately inside the strategy:
       - explore plans
       - exploit plans
       - dynamic component-enhancement plans
       - model-fusion plans
       - crossover plans
    4. Only executable rollout tasks are queued. Planning itself is done inline.
    5. One plan may have multiple sequential exec rounds, but different plans run in parallel.

Execution shape:

    [ready queue]
        |
        +--> explore exec chain
        +--> exploit exec chain
        +--> component exec chain A
        +--> component exec chain B
        +--> component exec chain C
        +--> model fusion exec
        +--> crossover exec

Phase order:

    bootstrap: 4 explore groups
        |
        v
    first component batch: 4 component groups
        |
        v
    crossover over that component batch
        |
        v
    3 exploit groups
        |
        v
    1 model fusion
        |
        v
    steady state:
        - model fusion if eligible
        - else crossover if eligible
        - else component discovery if eligible
        - else backfill by exploit:explore = 2:1

No-bubble rule:
    - if one plan is waiting for its next round, workers do not wait;
    - the strategy backfills the queue with new ready work from other plans.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from famou.config.settings import ModulesConfig
from famou.core.data import Context, Program, RolloutResult
from famou.core.protocol import Rollout, Strategy
from famou.infrastructure.llm.base import (
    get_llm_max_retries,
    get_llm_max_tokens,
    get_llm_temperature,
    get_llm_timeout,
)
from famou.modules.evaluate import EvaluateModule
from famou.modules.generate.crossover import CrossoverGenerate
from famou.modules.generate.model_fusion import ModelFusionGenerate
from famou.modules.generate.plan_driven_generate import PlanDrivenGenerate
from famou.modules.judge.component_ablation import ComponentJudge
from famou.modules.judge.llm_judge import LLMJudge
from famou.modules.planning.explore_plan import ExplorePlanModule
from famou.modules.planning.exploit_plan import ExploitPlanModule
from famou.modules.planning.crossover_plan import CrossoverPlanModule
from famou.modules.population.full_archive import FullArchivePopulation
from famou.modules.select.model_fusion_select import ModelFusionSelect
from famou.modules.select.plan_scoped_select import PlanScopedSelect
from famou.prompts import prompt_registry
from famou.utils.code_parser import extract_json
from famou.utils.id_gen import generate_short_uuid
from famou.utils.program_summary import (
    build_program_summary,
    extract_eval_wall_time,
    get_implementation_plan,
    get_improvement_directions,
    get_key_features,
)
from famou.utils.trace_utils import build_llm_trace

logger = logging.getLogger("famou.strategy.ml_pipeline")


class MLPipelineStrategy(Strategy):
    """Independent ML-oriented strategy with strategy-local ready-queue scheduling."""

    llm_client: Any = None

    def __init__(
        self,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
    ) -> None:
        """Initialize rollouts and island-local queue state."""
        params = params or ModulesConfig()
        self.evaluate_fn = evaluate_fn

        select_params = {**params.select}
        generate_params = {**params.generate}
        evaluate_params = {**params.evaluate}
        judge_params = {**params.judge}
        population_params = {**params.population}

        # Plan kind identifiers
        self.EXPLORE_PLAN = "explore"
        self.EXPLOIT_PLAN = "exploit"
        self.COMPONENT_PLAN = "component"
        self.CROSSOVER_PLAN = "crossover"
        self.MODEL_FUSION_PLAN = "model_fusion"

        # Scheduling constants
        self.BOOTSTRAP_EXPLORE_TRACKS = [
            "training_free_or_rule_based",
            "traditional_machine_learning",
            "deep_learning",
            "pretrained_model",
        ]
        self.TASK_PRIORITY = {
            self.MODEL_FUSION_PLAN: 10,
            self.CROSSOVER_PLAN: 20,
            self.COMPONENT_PLAN: 30,
            self.EXPLOIT_PLAN: 40,
            self.EXPLORE_PLAN: 50,
        }
        self.BACKFILL_PATTERN = ("exploit", "exploit", "explore")

        # Phase targets and limits
        self.bootstrap_explore_count = 4
        self.initial_exploit_count = 3
        self.initial_component_discovery_max_attempts = 3
        self.initial_model_fusion_skip_threshold = 3

        self.base_plan_budget = 3
        self.bonus_budget_on_improvement = 2
        self.model_fusion_cooldown = 15
        self.model_fusion_trigger_gap = 5
        self.model_fusion_top_k = 10

        self.component_discovery_batch_size = 4
        self.component_discovery_min_prior_plans = 2
        self.component_discovery_cooldown = 4
        self.crossover_min_component_plans = 4
        self.crossover_cooldown = 10
        self.ready_buffer_multiplier = 2

        self.population_module = FullArchivePopulation(**population_params)
        self.logger = None
        self._islands: Dict[int, Dict[str, Any]] = {}

        self.plan_rollout = Rollout(
            modules=[
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="ml_pipeline_plan_rollout",
        )

        self.component_rollout = Rollout(
            modules=[
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
                ComponentJudge(
                    bonus_budget_on_improvement=self.bonus_budget_on_improvement,
                ),
            ],
            name="ml_pipeline_component_rollout",
        )

        self.model_fusion_rollout = Rollout(
            modules=[
                ModelFusionSelect(),
                ModelFusionGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="ml_pipeline_model_fusion_rollout",
        )

        self.crossover_rollout = Rollout(
            modules=[
                PlanScopedSelect(**select_params),
                CrossoverGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="ml_pipeline_crossover_rollout",
        )

        # Planning rollouts: first round of explore/exploit/crossover plans
        self.explore_plan_rollout = Rollout(
            modules=[
                ExplorePlanModule(),
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="ml_pipeline_explore_plan_rollout",
        )
        self.exploit_plan_rollout = Rollout(
            modules=[
                ExploitPlanModule(),
                PlanScopedSelect(**select_params),
                PlanDrivenGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="ml_pipeline_exploit_plan_rollout",
        )
        self.crossover_plan_rollout = Rollout(
            modules=[
                CrossoverPlanModule(),
                PlanScopedSelect(**select_params),
                CrossoverGenerate(**generate_params),
                EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params),
                LLMJudge(**judge_params),
            ],
            name="ml_pipeline_crossover_plan_rollout",
        )

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        """Replenish the local ready queue and dispatch one executable rollout."""
        if ctx.iteration == 0 or not self._get_visible_programs(ctx):
            return self.plan_rollout

        island_state = self._ensure_island_state(ctx.island_id)
        self._reconcile_rollout_history(
            island_state=island_state,
            rollout_history=rollout_history,
        )

        ctx.metadata.pop("plan_context", None)
        ctx.metadata.pop("fusion_context", None)

        self._replenish_ready_queue(ctx, island_state)
        task = self._pop_next_task(island_state)
        if task is None:
            emergency_plan = self._create_explore_plan(
                ctx,
                island_state,
                track="emergency_open_explore",
            )
            self._enqueue_plan_task(island_state, emergency_plan)
            task = self._pop_next_task(island_state)
            if task is None:
                raise ValueError("MLPipelineStrategy failed to produce a ready task")

        plan = island_state["plans"].get(task["plan_id"])
        if not isinstance(plan, dict):
            raise ValueError(f"Unknown plan_id queued for dispatch: {task['plan_id']}")

        island_state["inflight_tasks"][plan["plan_id"]] = task
        if plan["kind"] == self.MODEL_FUSION_PLAN:
            ctx.metadata["fusion_context"] = self._build_fusion_context(
                ctx,
                island_state,
                plan,
            )
            return self.model_fusion_rollout

        ctx.metadata["plan_context"] = self._build_plan_context(ctx, island_state, plan)

        # First round: plan_text not yet generated → dispatch planning rollout
        if not plan.get("plan_text"):
            ctx.metadata["planner_inputs"] = self._build_planner_inputs_for_task(
                ctx, island_state, plan
            )
            if plan["kind"] == self.EXPLORE_PLAN:
                return self.explore_plan_rollout
            if plan["kind"] == self.EXPLOIT_PLAN:
                return self.exploit_plan_rollout
            if plan["kind"] == self.CROSSOVER_PLAN:
                return self.crossover_plan_rollout

        # Continuation rounds or non-planning plan kinds
        if plan["kind"] == self.COMPONENT_PLAN:
            return self.component_rollout
        if plan["kind"] == self.CROSSOVER_PLAN:
            return self.crossover_rollout
        return self.plan_rollout

    def dump_state(self) -> Dict[str, Any]:
        """Serialize the independent ML-pipeline strategy state."""
        return {
            "islands": self._islands,
            "base_plan_budget": self.base_plan_budget,
            "bonus_budget_on_improvement": self.bonus_budget_on_improvement,
            "model_fusion_cooldown": self.model_fusion_cooldown,
            "model_fusion_trigger_gap": self.model_fusion_trigger_gap,
            "model_fusion_top_k": self.model_fusion_top_k,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore strategy state from checkpoint."""
        islands = state.get("islands", {})
        if isinstance(islands, dict):
            self._islands = {
                int(island_id): island_state
                for island_id, island_state in islands.items()
            }

    def on_rollout_complete(self, result: RolloutResult) -> None:
        """Update plan state, budgets, and follow-up readiness after one rollout completes."""
        island_state = self._ensure_island_state(result.island_id)
        plan_id = self._extract_plan_id(result)
        if plan_id:
            island_state["inflight_tasks"].pop(plan_id, None)
        self._apply_rollout_result(island_state, result)
        island_state["processed_rollout_count"] = int(
            island_state.get("processed_rollout_count", 0)
        ) + 1

    def _ensure_island_state(self, island_id: int) -> Dict[str, Any]:
        """Create and return mutable per-island queue state."""
        if island_id not in self._islands:
            self._islands[island_id] = {
                "plans": {},
                "plan_order": [],
                "ready_queue": [],
                "inflight_tasks": {},
                "processed_rollout_count": 0,
                "dispatch_sequence": 0,
                "backfill_cycle_index": 0,
                "bootstrap_explore_index": 0,
                "initial_bootstrap_plan_ids": [],
                "bootstrap_initial_fusion_scheduled": False,
                "last_component_discovery_iteration": None,
                "component_discovery_runs": 0,
                "component_batches": [],
                "initial_component_batch_id": None,
                "initial_component_batch_crossed": False,
                "initial_component_batch_skipped": False,
                "initial_component_pending_plan_ids": [],
                "initial_component_discovery_attempts": 0,
                "initial_exploit_plan_ids": [],
                "initial_model_fusion_plan_id": None,
                "initial_model_fusion_completed": False,
                "initial_model_fusion_skipped": False,
                "initial_model_fusion_wait_attempts": 0,
                "initial_model_fusion_last_progress_count": None,
                "known_component_keys": [],
                "completed_component_plan_ids": [],
                "last_crossover_iteration": None,
                "explore_plan_count": 0,
                "last_model_fusion_iteration": None,
                "best_program_id": None,
                "best_score": None,
                "best_plan_id": None,
                "best_fused_explore_plan_count": 0,
            }
        return self._islands[island_id]

    def _reconcile_rollout_history(
        self,
        *,
        island_state: Dict[str, Any],
        rollout_history: List[RolloutResult],
    ) -> None:
        """Replay unseen rollout results into island-local state."""
        start_index = min(
            int(island_state.get("processed_rollout_count", 0)),
            len(rollout_history),
        )
        for result in rollout_history[start_index:]:
            plan_id = self._extract_plan_id(result)
            if plan_id:
                island_state["inflight_tasks"].pop(plan_id, None)
            self._apply_rollout_result(island_state, result)
        island_state["processed_rollout_count"] = len(rollout_history)

    def _apply_rollout_result(
        self,
        island_state: Dict[str, Any],
        result: RolloutResult,
    ) -> None:
        """Update one plan record from the finished rollout and queue follow-up work."""
        plan_id = self._extract_plan_id(result)
        if not plan_id:
            return
        plan = island_state["plans"].get(plan_id)
        if not isinstance(plan, dict):
            return

        if result.rollout_id not in plan["rollout_ids"]:
            plan["rollout_ids"].append(result.rollout_id)
            plan["round_count"] += 1
            plan["remaining_budget"] = max(
                0,
                int(plan.get("remaining_budget", 0)) - 1,
            )

        # Persist plan_text written by PlanningModule on the first planning rollout
        selection_extra = result.selection.extra if result.selection else {}
        plan_context = (
            selection_extra.get("plan_context")
            if isinstance(selection_extra, dict)
            else None
        )
        if isinstance(plan_context, dict):
            plan_text = plan_context.get("plan_text", "")
            if plan_text and not plan.get("plan_text"):
                plan["plan_text"] = plan_text
                planner_trace = plan_context.get("planner_trace")
                if planner_trace is not None:
                    plan["planner_trace"] = planner_trace

        child = result.generated_program
        if child is not None:
            child.meta.setdefault("plan_id", plan_id)
            child.meta.setdefault("plan_kind", plan["kind"])
            if plan["kind"] == self.COMPONENT_PLAN:
                child.meta["component_key"] = plan.get("component_key")
                child.meta["component_name"] = plan.get("component_name")
            if plan["kind"] == self.MODEL_FUSION_PLAN:
                child.meta["fused_source_plan_ids"] = list(plan.get("source_plan_ids", []))
                child.meta["fused_explore_plan_ids"] = list(
                    plan.get("fused_explore_plan_ids", [])
                )

            if child.id not in plan["child_program_ids"]:
                plan["child_program_ids"].append(child.id)
            if self._has_valid_solution(child):
                plan["has_successful_exec"] = True
                plan["successful_exec_count"] = int(
                    plan.get("successful_exec_count", 0)
                ) + 1

            if self._is_better_program(
                child,
                plan.get("best_score"),
                plan.get("best_validity"),
            ):
                plan["best_program_id"] = child.id
                plan["best_score"] = child.combined_score
                plan["best_metrics"] = dict(child.metrics or {})
                plan["best_eval_wall_time"] = extract_eval_wall_time(child)
                plan["best_validity"] = child.validity
                plan["best_error_info"] = child.error_info
                plan["best_implementation_plan"] = get_implementation_plan(child)
                plan["best_key_features"] = get_key_features(child)
                plan["best_improvement_directions"] = get_improvement_directions(child)

            if self._is_better_score(child.combined_score, island_state.get("best_score")):
                island_state["best_score"] = child.combined_score
                island_state["best_program_id"] = child.id
                island_state["best_plan_id"] = plan_id
                island_state["best_fused_explore_plan_count"] = len(
                    child.meta.get("fused_explore_plan_ids", []) or []
                )
                if plan["kind"] != self.MODEL_FUSION_PLAN:
                    plan["remaining_budget"] += self.bonus_budget_on_improvement
                    plan["bonus_rounds_granted"] += self.bonus_budget_on_improvement

        if plan["kind"] == self.MODEL_FUSION_PLAN:
            island_state["last_model_fusion_iteration"] = result.iteration

        if plan["remaining_budget"] <= 0:
            plan["status"] = "completed"
            plan["closed_iteration"] = result.iteration
            if (
                plan["kind"] == self.COMPONENT_PLAN
                and plan["plan_id"] not in island_state["completed_component_plan_ids"]
            ):
                island_state["completed_component_plan_ids"].append(plan["plan_id"])
            if plan["kind"] == self.CROSSOVER_PLAN:
                self._mark_component_batch_crossed(
                    island_state,
                    str(plan.get("component_batch_id") or ""),
                )
            if (
                plan["kind"] == self.MODEL_FUSION_PLAN
                and plan.get("schedule_role") == "initial_model_fusion"
            ):
                island_state["initial_model_fusion_completed"] = True
        else:
            self._enqueue_plan_task(
                island_state,
                plan,
                priority=self._task_priority_for_plan(plan, continuation=True),
            )

    def _replenish_ready_queue(self, ctx: Context, island_state: Dict[str, Any]) -> None:
        """Top up the queue with executable work so workers do not idle."""
        target_depth = self._target_ready_depth(ctx, island_state)
        attempts = 0
        while self._ready_depth(island_state) < target_depth and attempts < target_depth * 6:
            attempts += 1
            if self._schedule_next_ready_work(ctx, island_state):
                continue
            break

    def _target_ready_depth(self, ctx: Context, island_state: Dict[str, Any]) -> int:
        """Return the queue depth target for the current scheduler phase."""
        max_workers = max(1, int(ctx.experiment_config.max_workers))
        phase = self._get_scheduler_phase(island_state)

        if phase in {
            "bootstrap_explore",
            "bootstrap_explore_running",
            "initial_component_batch",
            "initial_component_crossover",
            "initial_exploit",
            "initial_exploit_running",
            "initial_model_fusion",
        }:
            return max_workers

        return max(1, max_workers * self.ready_buffer_multiplier)

    def _schedule_next_ready_work(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Schedule the next plan according to the phase-aware ready-queue policy."""
        phase = self._get_scheduler_phase(island_state)

        if phase == "bootstrap_explore":
            return self._schedule_bootstrap_explore(ctx, island_state)

        if phase == "bootstrap_explore_running":
            return self._schedule_backfill_plan(
                ctx,
                island_state,
                priority=90,
            )

        if phase == "initial_component_batch":
            if self._schedule_initial_component_batch(ctx, island_state):
                return True
            return self._schedule_backfill_plan(
                ctx,
                island_state,
                priority=90,
            )

        if phase == "initial_component_crossover":
            if self._schedule_crossover(ctx, island_state):
                return True
            return self._schedule_backfill_plan(
                ctx,
                island_state,
                priority=90,
            )

        if phase == "initial_exploit":
            if self._schedule_initial_exploit(ctx, island_state):
                return True
            return self._schedule_backfill_plan(
                ctx,
                island_state,
                priority=90,
            )

        if phase == "initial_exploit_running":
            return self._schedule_backfill_plan(
                ctx,
                island_state,
                priority=90,
            )

        if phase == "initial_model_fusion":
            if self._schedule_initial_model_fusion(ctx, island_state):
                return True
            return self._schedule_backfill_plan(
                ctx,
                island_state,
                priority=90,
            )

        if self._schedule_adaptive_model_fusion(ctx, island_state):
            return True
        if self._schedule_crossover(ctx, island_state):
            return True
        if self._schedule_component_discovery(ctx, island_state):
            return True
        return self._schedule_backfill_plan(ctx, island_state)

    def _get_scheduler_phase(self, island_state: Dict[str, Any]) -> str:
        """Return the coarse scheduler phase for one island."""
        bootstrap_plan_ids = [
            str(plan_id).strip()
            for plan_id in list(island_state.get("initial_bootstrap_plan_ids") or [])
            if str(plan_id).strip()
        ]
        island_state["initial_bootstrap_plan_ids"] = bootstrap_plan_ids
        if len(bootstrap_plan_ids) < self.bootstrap_explore_count:
            return "bootstrap_explore"
        if not self._are_all_plans_completed(island_state, bootstrap_plan_ids):
            return "bootstrap_explore_running"

        initial_batch_id = str(island_state.get("initial_component_batch_id") or "").strip()
        if initial_batch_id and not self._has_component_batch(island_state, initial_batch_id):
            island_state["initial_component_batch_id"] = None
            island_state["initial_component_batch_crossed"] = False
            initial_batch_id = ""
        if not initial_batch_id and not island_state.get("initial_component_batch_skipped"):
            return "initial_component_batch"

        if self._is_component_batch_crossed(island_state, initial_batch_id):
            island_state["initial_component_batch_crossed"] = True
        if initial_batch_id and not island_state.get("initial_component_batch_crossed"):
            return "initial_component_crossover"

        initial_exploit_plan_ids = [
            str(plan_id).strip()
            for plan_id in list(island_state.get("initial_exploit_plan_ids") or [])
            if str(plan_id).strip()
        ]
        island_state["initial_exploit_plan_ids"] = initial_exploit_plan_ids
        if len(initial_exploit_plan_ids) < self.initial_exploit_count:
            return "initial_exploit"
        if not self._are_all_plans_completed(island_state, initial_exploit_plan_ids):
            return "initial_exploit_running"

        initial_model_fusion_plan_id = str(
            island_state.get("initial_model_fusion_plan_id") or ""
        ).strip()
        if initial_model_fusion_plan_id:
            fusion_plan = island_state.get("plans", {}).get(initial_model_fusion_plan_id)
            if isinstance(fusion_plan, dict) and fusion_plan.get("status") == "completed":
                island_state["initial_model_fusion_completed"] = True
        if not island_state.get("initial_model_fusion_completed") and not island_state.get(
            "initial_model_fusion_skipped"
        ):
            return "initial_model_fusion"

        return "steady"

    def _schedule_bootstrap_explore(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Seed the queue with broad exploration tracks first."""
        index = int(island_state.get("bootstrap_explore_index", 0))
        if index >= len(self.BOOTSTRAP_EXPLORE_TRACKS):
            return False
        island_state["bootstrap_explore_index"] = index + 1
        plan = self._create_explore_plan(
            ctx,
            island_state,
            track=self.BOOTSTRAP_EXPLORE_TRACKS[index],
            schedule_role="initial_bootstrap_explore",
        )
        island_state.setdefault("initial_bootstrap_plan_ids", []).append(str(plan["plan_id"]))
        self._enqueue_plan_task(island_state, plan)
        return True

    def _schedule_initial_model_fusion(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Insert one early model-fusion plan after the first bootstrap wave."""
        if island_state.get("initial_model_fusion_completed"):
            return False
        if island_state.get("initial_model_fusion_skipped"):
            return False
        initial_plan_id = str(island_state.get("initial_model_fusion_plan_id") or "").strip()
        if initial_plan_id:
            plan = island_state.get("plans", {}).get(initial_plan_id)
            if not isinstance(plan, dict):
                island_state["initial_model_fusion_plan_id"] = None
                initial_plan_id = ""
            else:
                if plan.get("status") == "completed":
                    island_state["initial_model_fusion_completed"] = True
                return False
        current_progress = int(island_state.get("processed_rollout_count", 0))
        last_progress = island_state.get("initial_model_fusion_last_progress_count")
        if last_progress != current_progress:
            island_state["initial_model_fusion_wait_attempts"] = int(
                island_state.get("initial_model_fusion_wait_attempts", 0)
            ) + 1
            island_state["initial_model_fusion_last_progress_count"] = current_progress
        if not self._should_schedule_model_fusion(ctx, island_state):
            if int(island_state.get("initial_model_fusion_wait_attempts", 0)) >= int(
                self.initial_model_fusion_skip_threshold
            ):
                island_state["initial_model_fusion_skipped"] = True
            return False
        plan = self._create_model_fusion_plan(
            ctx,
            island_state,
            schedule_role="initial_model_fusion",
        )
        if plan is None:
            return False
        island_state["initial_model_fusion_plan_id"] = str(plan["plan_id"])
        self._enqueue_plan_task(island_state, plan)
        return True

    def _schedule_initial_component_batch(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Accumulate the first component batch across multiple discovery passes."""
        if island_state.get("initial_component_batch_skipped"):
            return False
        if not self._should_run_component_discovery(ctx, island_state):
            return False

        island_state["initial_component_discovery_attempts"] = int(
            island_state.get("initial_component_discovery_attempts", 0)
        ) + 1
        created = self._schedule_component_discovery(
            ctx,
            island_state,
            accumulate_into_initial_batch=True,
        )
        if island_state.get("initial_component_batch_id"):
            return created
        if int(island_state.get("initial_component_discovery_attempts", 0)) >= int(
            self.initial_component_discovery_max_attempts
        ):
            island_state["initial_component_batch_skipped"] = True
            island_state["initial_component_pending_plan_ids"] = []
        return created

    def _schedule_initial_exploit(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Create the three initial exploit groups after the first crossover."""
        existing_ids = [
            str(plan_id).strip()
            for plan_id in list(island_state.get("initial_exploit_plan_ids") or [])
            if str(plan_id).strip()
        ]
        island_state["initial_exploit_plan_ids"] = existing_ids
        if len(existing_ids) >= self.initial_exploit_count:
            return False
        plan = self._create_exploit_plan(
            ctx,
            island_state,
            schedule_role="initial_exploit",
        )
        if plan is None:
            return False
        island_state["initial_exploit_plan_ids"].append(str(plan["plan_id"]))
        self._enqueue_plan_task(island_state, plan)
        return True

    def _schedule_component_discovery(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        accumulate_into_initial_batch: bool = False,
    ) -> bool:
        """Discover dynamic component directions and enqueue their first executable rounds."""
        if not self._should_run_component_discovery(ctx, island_state):
            return False

        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            return False

        prior_plans = self._build_prior_plan_summaries(island_state)
        existing_keys = list(island_state.get("known_component_keys", []))
        components, discovery_trace = self._discover_component_specs(
            ctx,
            best_program=best_program,
            prior_plans=prior_plans,
            existing_component_keys=existing_keys,
        )
        island_state["last_component_discovery_iteration"] = ctx.iteration
        island_state["component_discovery_runs"] = int(
            island_state.get("component_discovery_runs", 0)
        ) + 1

        created = 0
        batch_plan_ids: List[str] = []
        for component in components:
            component_key = component["component_key"]
            if component_key in island_state["known_component_keys"]:
                continue
            plan = self._create_component_plan(
                ctx,
                island_state,
                component=component,
                planner_trace=discovery_trace,
            )
            island_state["known_component_keys"].append(component_key)
            self._enqueue_plan_task(island_state, plan)
            batch_plan_ids.append(str(plan["plan_id"]))
            created += 1
        if accumulate_into_initial_batch:
            pending_plan_ids = [
                str(plan_id).strip()
                for plan_id in list(island_state.get("initial_component_pending_plan_ids") or [])
                if str(plan_id).strip()
            ]
            for plan_id in batch_plan_ids:
                if plan_id not in pending_plan_ids:
                    pending_plan_ids.append(plan_id)
            island_state["initial_component_pending_plan_ids"] = pending_plan_ids
            if (
                not island_state.get("initial_component_batch_id")
                and len(pending_plan_ids) >= self.crossover_min_component_plans
            ):
                batch_id = (
                    f"component_batch_{ctx.island_id}_{ctx.iteration}_{generate_short_uuid()}"
                )
                island_state.setdefault("component_batches", []).append(
                    {
                        "batch_id": batch_id,
                        "plan_ids": list(pending_plan_ids),
                        "created_iteration": ctx.iteration,
                        "status": "running",
                        "crossover_plan_id": None,
                    }
                )
                island_state["initial_component_batch_id"] = batch_id
                island_state["initial_component_pending_plan_ids"] = []
        if (
            not accumulate_into_initial_batch
            and len(batch_plan_ids) >= self.crossover_min_component_plans
        ):
            batch_id = (
                f"component_batch_{ctx.island_id}_{ctx.iteration}_{generate_short_uuid()}"
            )
            island_state.setdefault("component_batches", []).append(
                {
                    "batch_id": batch_id,
                    "plan_ids": batch_plan_ids,
                    "created_iteration": ctx.iteration,
                    "status": "running",
                    "crossover_plan_id": None,
                }
            )
            if not island_state.get("initial_component_batch_id"):
                island_state["initial_component_batch_id"] = batch_id
        return created > 0

    def _schedule_crossover(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Queue crossover when enough completed component plans exist and cooldown passed."""
        if not self._should_schedule_crossover(ctx, island_state):
            return False
        batch = self._get_ready_component_batch(island_state)
        if not isinstance(batch, dict):
            return False
        plan = self._create_crossover_plan(ctx, island_state, batch=batch)
        if plan is None:
            return False
        island_state["last_crossover_iteration"] = ctx.iteration
        batch["status"] = "crossing"
        batch["crossover_plan_id"] = str(plan["plan_id"])
        self._enqueue_plan_task(island_state, plan)
        return True

    def _schedule_adaptive_model_fusion(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Queue adaptive model fusion when enough new explore plans accumulated."""
        if not self._should_schedule_model_fusion(ctx, island_state):
            return False
        plan = self._create_model_fusion_plan(ctx, island_state)
        if plan is None:
            return False
        self._enqueue_plan_task(island_state, plan)
        return True

    def _schedule_backfill_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        priority: Optional[int] = None,
    ) -> bool:
        """Fill bubbles with exploit/explore according to the fixed 2:1 ratio."""
        cycle_index = int(island_state.get("backfill_cycle_index", 0))
        mode = self.BACKFILL_PATTERN[cycle_index % len(self.BACKFILL_PATTERN)]
        island_state["backfill_cycle_index"] = cycle_index + 1

        if mode == "exploit":
            plan = self._create_exploit_plan(ctx, island_state)
            if plan is None:
                plan = self._create_explore_plan(ctx, island_state, track="open_explore")
        else:
            plan = self._create_explore_plan(ctx, island_state, track="open_explore")
        self._enqueue_plan_task(
            island_state,
            plan,
            priority=priority,
        )
        return True

    def _create_explore_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        track: str,
        schedule_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new explore plan over prior best plans."""
        anchor_program = self._get_current_best_program(ctx)
        if anchor_program is None:
            raise ValueError("MLPipelineStrategy requires a visible anchor program")
        return self._register_plan(
            ctx,
            island_state,
            kind=self.EXPLORE_PLAN,
            focus=track,
            plan_text="",
            seed_parent=anchor_program,
            base_budget=self.base_plan_budget,
            planner_trace=None,
            counted_as_explore=True,
            schedule_role=schedule_role,
        )

    def _create_exploit_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        schedule_role: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a focused exploit plan around the current best plan."""
        focus_candidate = self._select_exploit_focus_candidate(ctx, island_state)
        if focus_candidate is None:
            return None
        focus_plan_id = str(focus_candidate.get("plan_id") or "").strip()
        focus_plan = island_state["plans"][focus_plan_id]
        focus_program = self._get_program(
            ctx,
            str(focus_plan.get("best_program_id") or ""),
        )
        if focus_program is None:
            return None
        return self._register_plan(
            ctx,
            island_state,
            kind=self.EXPLOIT_PLAN,
            focus=str(focus_plan_id),
            plan_text="",
            seed_parent=focus_program,
            base_budget=self.base_plan_budget,
            planner_trace=None,
            schedule_role=schedule_role,
        )

    def _create_component_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        component: Dict[str, str],
        planner_trace: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create one dynamic component-enhancement plan around the current best program."""
        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            raise ValueError("component plan requires a best program")

        component_key = component["component_key"]
        component_name = component["component_name"]
        component_goal = component["component_goal"]
        component_constraints = component["component_constraints"]
        component_techniques = component["suggested_techniques"]
        plan_text = (
            "Component enhancement plan.\n"
            f"Component name: {component_name}\n"
            f"Goal: {component_goal}\n"
            f"Constraints: {component_constraints}\n"
            f"Suggested techniques: {component_techniques}\n"
            "Instruction: treat this as a controlled component-focused improvement, and avoid rewriting unrelated core logic."
        )
        return self._register_plan(
            ctx,
            island_state,
            kind=self.COMPONENT_PLAN,
            focus=component_key,
            plan_text=plan_text,
            seed_parent=best_program,
            base_budget=self.base_plan_budget,
            planner_trace=planner_trace,
            component_key=component_key,
            component_name=component_name,
            component_goal=component_goal,
            component_constraints=component_constraints,
            component_suggested_techniques=component_techniques,
        )

    def _create_crossover_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        batch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Create one algorithm-fusion crossover plan from the best completed component plans."""
        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            return None
        source_plans = [
            island_state.get("plans", {}).get(plan_id)
            for plan_id in list(batch.get("plan_ids") or [])
        ]
        source_plans = [
            plan
            for plan in source_plans
            if isinstance(plan, dict)
            and plan.get("kind") == self.COMPONENT_PLAN
            and plan.get("status") == "completed"
            and plan.get("best_program_id")
        ]
        if len(source_plans) < self.crossover_min_component_plans:
            return None
        source_plans = sorted(
            source_plans,
            key=self._plan_candidate_sort_key,
        )
        source_program_ids = [
            str(plan["best_program_id"]) for plan in source_plans[: self.model_fusion_top_k]
        ]
        source_plan_ids = [str(plan["plan_id"]) for plan in source_plans[: self.model_fusion_top_k]]
        return self._register_plan(
            ctx,
            island_state,
            kind=self.CROSSOVER_PLAN,
            focus="component_crossover",
            plan_text="",
            seed_parent=best_program,
            base_budget=self.base_plan_budget,
            planner_trace=None,
            component_batch_id=str(batch.get("batch_id") or ""),
            source_plan_ids=source_plan_ids,
            reference_plan_ids=source_plan_ids,
            reference_program_ids=source_program_ids,
        )

    def _create_model_fusion_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        schedule_role: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create a one-shot model-fusion plan from the current best plan and top alternatives."""
        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            return None
        best_plan_id = best_program.meta.get("plan_id")
        used_source_plan_ids = self._get_fused_source_plan_ids(best_program)
        excluded_plan_ids = set(used_source_plan_ids)
        if best_plan_id:
            excluded_plan_ids.add(str(best_plan_id))
        candidate_plan_summaries = [
            plan
            for plan in self._iter_alive_successful_plan_candidates(ctx, island_state)
            if str(plan.get("plan_id") or "") not in excluded_plan_ids
        ]
        if not candidate_plan_summaries:
            return None
        inspirations = candidate_plan_summaries[: self.model_fusion_top_k]
        inspiration_program_ids = [
            str(item["best_program_id"])
            for item in inspirations
            if item.get("best_program_id")
        ]
        inspiration_plan_ids = [
            str(item["plan_id"])
            for item in inspirations
            if item.get("plan_id")
        ]
        if not inspiration_program_ids:
            return None
        plan_text = (
            "Model fusion plan.\n"
            "Goal: keep the current best implementation as the main anchor and fuse it with the strongest complementary successful plans."
        )
        reference_plan_ids = [
            plan_id for plan_id in [best_plan_id, *inspiration_plan_ids] if plan_id
        ]
        source_plan_ids = list(used_source_plan_ids)
        if not source_plan_ids and best_plan_id:
            source_plan_ids.append(str(best_plan_id))
        for plan_id in inspiration_plan_ids:
            normalized_plan_id = str(plan_id).strip()
            if normalized_plan_id and normalized_plan_id not in source_plan_ids:
                source_plan_ids.append(normalized_plan_id)
        fused_explore_plan_ids = [
            plan_id
            for plan_id in source_plan_ids
            if self._plan_kind(island_state, plan_id) == self.EXPLORE_PLAN
        ]
        return self._register_plan(
            ctx,
            island_state,
            kind=self.MODEL_FUSION_PLAN,
            focus="model_fusion",
            plan_text=plan_text,
            seed_parent=best_program,
            base_budget=1,
            source_plan_ids=source_plan_ids,
            reference_plan_ids=reference_plan_ids,
            reference_program_ids=[best_program.id, *inspiration_program_ids],
            fused_explore_plan_ids=fused_explore_plan_ids,
            schedule_role=schedule_role,
        )

    def _register_plan(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        *,
        kind: str,
        focus: str,
        plan_text: str,
        seed_parent: Program,
        base_budget: int,
        planner_trace: Optional[Dict[str, Any]] = None,
        source_plan_ids: Optional[List[str]] = None,
        reference_plan_ids: Optional[List[str]] = None,
        reference_program_ids: Optional[List[str]] = None,
        fused_explore_plan_ids: Optional[List[str]] = None,
        counted_as_explore: bool = False,
        component_key: Optional[str] = None,
        component_name: Optional[str] = None,
        component_goal: Optional[str] = None,
        component_constraints: Optional[str] = None,
        component_suggested_techniques: Optional[str] = None,
        component_batch_id: Optional[str] = None,
        schedule_role: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create and store a new plan record."""
        plan_id = f"ml_plan_{ctx.island_id}_{ctx.iteration}_{generate_short_uuid()}"
        plan = {
            "plan_id": plan_id,
            "kind": kind,
            "focus": focus,
            "plan_text": plan_text,
            "status": "active",
            "created_iteration": ctx.iteration,
            "closed_iteration": None,
            "seed_parent_id": seed_parent.id,
            "best_program_id": seed_parent.id,
            "best_score": seed_parent.combined_score,
            "best_metrics": dict(seed_parent.metrics or {}),
            "best_eval_wall_time": extract_eval_wall_time(seed_parent),
            "best_validity": seed_parent.validity,
            "best_error_info": seed_parent.error_info,
            "best_implementation_plan": get_implementation_plan(seed_parent),
            "best_key_features": get_key_features(seed_parent),
            "best_improvement_directions": get_improvement_directions(seed_parent),
            "has_successful_exec": False,
            "successful_exec_count": 0,
            "base_budget": int(base_budget),
            "remaining_budget": int(base_budget),
            "bonus_rounds_granted": 0,
            "round_count": 0,
            "child_program_ids": [],
            "reference_program_ids": list(reference_program_ids or []),
            "reference_plan_ids": list(reference_plan_ids or []),
            "source_plan_ids": list(source_plan_ids or []),
            "fused_explore_plan_ids": list(fused_explore_plan_ids or []),
            "rollout_ids": [],
            "planner_trace": planner_trace,
            "component_key": component_key,
            "component_name": component_name,
            "component_goal": component_goal,
            "component_constraints": component_constraints,
            "component_suggested_techniques": component_suggested_techniques,
            "component_batch_id": component_batch_id,
            "schedule_role": schedule_role,
        }
        island_state["plans"][plan_id] = plan
        island_state["plan_order"].append(plan_id)
        if counted_as_explore:
            island_state["explore_plan_count"] = int(
                island_state.get("explore_plan_count", 0)
            ) + 1
        return plan

    def _build_planner_inputs_for_task(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build fresh planner_inputs at dispatch time for planning rollouts."""
        kind = plan["kind"]
        prior_plans = self._build_prior_plan_summaries(island_state)

        if kind == self.EXPLORE_PLAN:
            return {
                "planner_mode": "explore_plan",
                "anchor_program": self._get_current_best_program(ctx),
                "history_plan_summaries": prior_plans,
                "track_hint": str(plan.get("focus") or ""),
            }

        if kind == self.EXPLOIT_PLAN:
            focus_plan_id = str(plan.get("focus") or "")
            focus_plan = island_state["plans"].get(focus_plan_id)
            focus_program = None
            if focus_plan:
                focus_program = self._get_program(
                    ctx, str(focus_plan.get("best_program_id") or "")
                )
            if focus_program is None:
                focus_program = self._get_current_best_program(ctx)
            return {
                "planner_mode": "exploit_plan",
                "anchor_program": focus_program,
                "focus_plan_summary": self._build_plan_summary(focus_plan) if focus_plan else None,
                "history_plan_summaries": prior_plans[:64],
            }

        if kind == self.CROSSOVER_PLAN:
            source_plan_ids = list(plan.get("source_plan_ids") or [])
            source_plans = [
                island_state["plans"].get(pid)
                for pid in source_plan_ids
                if island_state["plans"].get(pid)
            ]
            best_program = self._get_current_best_program(ctx)
            best_plan_id = str(
                best_program.meta.get("plan_id") or ""
            ).strip() if best_program else ""
            focus_plan = island_state["plans"].get(best_plan_id)
            return {
                "planner_mode": "crossover_plan",
                "anchor_program": best_program,
                "focus_plan_summary": self._build_plan_summary(focus_plan) if focus_plan else None,
                "crossover_plan_summaries": [
                    self._build_plan_summary(p) for p in source_plans if isinstance(p, dict)
                ],
            }

        return {}

    def _build_plan_context(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build selection/generation metadata for plan-scoped rollouts."""
        current_parent = self._resolve_current_parent(ctx, plan)
        visible_program_ids = list(plan.get("reference_program_ids", [])) + list(
            plan.get("child_program_ids", [])
        )
        component_experience = {}
        component_key = plan.get("component_key")
        if component_key and ctx.state:
            component_experience = ctx.state.get_island(
                "ml_pipeline",
                "component_experience",
                str(component_key),
                default={},
            )
            if not isinstance(component_experience, dict):
                component_experience = {}
        component_experience = self._build_component_experience_prompt_view(
            component_experience
        )

        return {
            "plan_id": plan["plan_id"],
            "planner_mode": plan["kind"],
            "plan_text": plan["plan_text"],
            "plan_round": int(plan.get("round_count", 0)) + 1,
            "created_iteration": plan.get("created_iteration"),
            "seed_parent_id": plan.get("seed_parent_id"),
            "current_parent_id": current_parent.id,
            "best_program_id": plan.get("best_program_id") or current_parent.id,
            "best_score": plan.get("best_score"),
            "best_metrics": dict(plan.get("best_metrics") or {}),
            "best_validity": plan.get("best_validity"),
            "best_implementation_plan": plan.get("best_implementation_plan"),
            "best_key_features": plan.get("best_key_features"),
            "best_improvement_directions": plan.get("best_improvement_directions"),
            "anchor_program_id": plan.get("seed_parent_id"),
            "anchor_score": current_parent.combined_score,
            "anchor_metrics": dict(current_parent.metrics or {}),
            "anchor_validity": current_parent.validity,
            "anchor_implementation_plan": get_implementation_plan(current_parent),
            "anchor_key_features": get_key_features(current_parent),
            "anchor_improvement_directions": get_improvement_directions(current_parent),
            "plan_program_ids": list(dict.fromkeys(visible_program_ids)),
            "plan_rollout_ids": list(plan.get("rollout_ids") or []),
            "remaining_budget": plan.get("remaining_budget"),
            "planner_trace": plan.get("planner_trace"),
            "planner_history_plan_ids": list(plan.get("source_plan_ids") or []),
            "planner_crossover_plan_ids": list(plan.get("source_plan_ids") or []),
            "planner_focus_plan_id": plan.get("focus"),
            "island_best_score_at_dispatch": island_state.get("best_score"),
            "component_key": plan.get("component_key"),
            "component_name": plan.get("component_name"),
            "component_goal": plan.get("component_goal"),
            "component_constraints": plan.get("component_constraints"),
            "component_suggested_techniques": plan.get(
                "component_suggested_techniques"
            ),
            "component_experience": component_experience,
            "strategy_guardrails": self._build_generation_guardrails(
                plan_kind=str(plan.get("kind") or "")
            ),
        }

    def _build_component_experience_prompt_view(
        self,
        component_experience: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build an LLM-facing semantic summary without opaque ids."""
        if not isinstance(component_experience, dict):
            return {}

        latest_summary = component_experience.get("latest_summary")
        history = component_experience.get("history")
        semantic_history: List[Dict[str, Any]] = []
        if isinstance(history, list):
            for item in history[-3:]:
                if not isinstance(item, dict):
                    continue
                summary = item.get("summary")
                component_name = str(item.get("component_name") or "").strip()
                semantic_history.append(
                    {
                        "component_name": component_name or None,
                        "summary": summary,
                    }
                )

        return {
            "component_name": component_experience.get("component_name"),
            "latest_summary": latest_summary,
            "history": semantic_history,
        }

    def _build_fusion_context(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build one-shot fusion metadata for the selection/generation modules."""
        reference_program_ids = list(plan.get("reference_program_ids") or [])
        source_plan_ids = list(plan.get("source_plan_ids") or [])
        reference_plan_ids = list(plan.get("reference_plan_ids") or source_plan_ids)
        parent_id = (
            reference_program_ids[0]
            if reference_program_ids
            else plan.get("seed_parent_id")
        )
        inspiration_ids = reference_program_ids[1:]
        parent_program = self._get_program(ctx, str(parent_id))
        parent_plan = (
            island_state.get("plans", {}).get(reference_plan_ids[0])
            if reference_plan_ids
            else None
        )
        inspiration_program_summaries: List[Dict[str, Any]] = []
        for program_id, plan_id in zip(inspiration_ids, reference_plan_ids[1:]):
            program = self._get_program(ctx, str(program_id))
            plan_record = island_state.get("plans", {}).get(plan_id)
            if program is None:
                continue
            inspiration_program_summaries.append(
                self._build_fusion_program_summary(program, plan_record)
            )
        return {
            "plan_id": plan["plan_id"],
            "plan_kind": plan["kind"],
            "parent_id": parent_id,
            "inspiration_ids": inspiration_ids,
            "plan_text": plan["plan_text"],
            "source_plan_ids": source_plan_ids,
            "fused_explore_plan_ids": list(plan.get("fused_explore_plan_ids") or []),
            "strategy_guardrails": self._build_generation_guardrails(
                plan_kind=str(plan.get("kind") or "")
            ),
            "parent_program_summary": (
                self._build_fusion_program_summary(parent_program, parent_plan)
                if parent_program is not None
                else None
            ),
            "inspiration_program_summaries": inspiration_program_summaries,
        }

    def _build_prior_plan_summaries(
        self,
        island_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Serialize prior plan summaries for planner context."""
        return [
            self._build_plan_summary(plan)
            for plan in self._iter_completed_plans(island_state)
        ]

    def _build_plan_summary(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build a compact plan summary from stored best-program information."""
        return {
            "plan_id": plan.get("plan_id"),
            "plan_kind": plan.get("kind"),
            "plan_focus": plan.get("focus"),
            "component_name": plan.get("component_name"),
            "plan_text": plan.get("plan_text"),
            "best_program_id": plan.get("best_program_id"),
            "best_score": plan.get("best_score"),
            "best_metrics": dict(plan.get("best_metrics") or {}),
            "best_eval_wall_time": plan.get("best_eval_wall_time"),
            "best_error_info": plan.get("best_error_info"),
            "best_implementation_plan": plan.get("best_implementation_plan"),
        }

    def _build_llm_plan_summary(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Build the minimal semantic plan summary that is safe to show to the LLM."""
        return {
            "component_name": (
                str(plan.get("component_name") or "").strip() or None
            ),
            "implementation_plan": str(
                plan.get("best_implementation_plan") or ""
            ).strip(),
            "combined_score": plan.get("best_score"),
            "metrics": dict(plan.get("best_metrics") or {}),
            "eval_time": plan.get("best_eval_wall_time"),
            "error_info": plan.get("best_error_info"),
        }

    def _build_semantic_program_summary(
        self,
        program: Optional[Program],
        *,
        plan_text: str = "",
        include_code: bool = False,
    ) -> Dict[str, Any]:
        """Build the LLM-facing semantic result summary for one program."""
        if program is None:
            return {}
        summary = {
            "plan_direction": str(plan_text or "").strip(),
            "implementation_plan": get_implementation_plan(program),
            "combined_score": program.combined_score,
            "metrics": dict(program.metrics or {}),
            "eval_time": extract_eval_wall_time(program),
            "error_info": program.error_info,
        }
        if include_code:
            summary["code"] = program.code
        return summary

    def _build_fusion_program_summary(
        self,
        program: Optional[Program],
        plan: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build the semantic fusion summary shown to the LLM."""
        return self._build_semantic_program_summary(
            program,
            plan_text=str((plan or {}).get("plan_text") or "").strip(),
            include_code=True,
        )

    def _build_generation_guardrails(self, *, plan_kind: str) -> str:
        """Return strategy-level generation constraints only, without task-specific rules."""
        shared_rules = [
            "严格保持 evaluator 期望的函数签名、输入参数和输出文件约定，不能私自改接口。",
            "优先在当前代码骨架与当前 plan 的基础上做有针对性的修改，不要无故整段重写成与当前目标无关的新范式。",
            "若已有代码中存在明显有效的局部结构，应优先保留这些有效部分，再针对当前 plan 需要改变的部分做修改。",
            "输出必须是一个自洽、可运行、端到端完整的实现，不要把多个互相冲突的半成品流程拼接在一起。",
        ]

        if plan_kind == self.MODEL_FUSION_PLAN:
            shared_rules.extend(
                [
                    "当前是 model_fusion：必须以 anchor 代码为主干，只吸收真正互补的模块或思路，不要把多个完整主流程硬拼在一起。",
                    "融合后仍然只能保留一个一致的端到端实现路径，不能留下多个并行且冲突的训练/推理分支。",
                ]
            )
        elif plan_kind == self.CROSSOVER_PLAN:
            shared_rules.extend(
                [
                    "当前是 crossover：优先保留 anchor 的稳定主干，只替换确有证据更优的局部模块或关键设计。",
                    "不要同时引入多个彼此冲突的数据流、标签流、训练逻辑或后处理路径。",
                ]
            )
        elif plan_kind == self.COMPONENT_PLAN:
            shared_rules.extend(
                [
                    "当前是 component enhancement：只围绕当前组件方向做受控修改，避免同时改动多个大部件。",
                    "核心算法主干应尽量稳定，重点验证该组件方向本身是否带来正向作用。",
                ]
            )
        elif plan_kind == self.EXPLOIT_PLAN:
            shared_rules.extend(
                [
                    "当前是 exploit：围绕现有最值得继续修的方案做深入修复或强化，不要偏离到全新路线。",
                    "优先修补已暴露的弱点、失败点或瓶颈，而不是重新发明一个几乎无关的新方案。",
                ]
            )
        else:
            shared_rules.extend(
                [
                    "当前是 explore：应基于已有 plan 的历史，优先探索语义上明显不同的新方向。",
                    "给定的 exploration hint 只是启发，不是必须死守的算法标签；核心是与已有探索形成真实差异。",
                ]
            )

        return "\n".join(f"- {rule}" for rule in shared_rules)

    def _discover_component_specs(
        self,
        ctx: Context,
        *,
        best_program: Program,
        prior_plans: List[Dict[str, Any]],
        existing_component_keys: List[str],
    ) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Discover dynamic component directions using the LLM with parse retries."""
        best_program_summary = self._build_semantic_program_summary(best_program)
        llm_prior_plans = [
            self._build_llm_plan_summary(plan)
            for plan in prior_plans
            if isinstance(plan, dict)
        ]
        system_prompt = prompt_registry.get("planning/component_discovery_system.txt")
        base_prompt = prompt_registry.get(
            "planning/component_discovery.txt",
            task_description=ctx.task_description,
            best_program_json=json.dumps(
                best_program_summary,
                ensure_ascii=False,
                indent=2,
            ),
            prior_plans_json=json.dumps(
                llm_prior_plans[-10:],
                ensure_ascii=False,
                indent=2,
            ),
            existing_component_labels_json=json.dumps(
                self._build_existing_component_labels(
                    existing_component_keys=existing_component_keys,
                    prior_plans=prior_plans,
                ),
                ensure_ascii=False,
                indent=2,
            ),
            target_component_count=self.component_discovery_batch_size,
        )

        temperature = get_llm_temperature(self.llm_client)
        max_tokens = get_llm_max_tokens(self.llm_client)
        timeout = get_llm_timeout(self.llm_client)
        max_attempts = get_llm_max_retries(self.llm_client)
        last_error: Optional[Exception] = None
        last_response = None
        last_prompt = base_prompt

        for attempt in range(1, max_attempts + 1):
            prompt = base_prompt
            if attempt > 1:
                prompt = (
                    f"{base_prompt}\n\n"
                    "RETRY INSTRUCTION:\n"
                    "Return exactly one JSON object matching the requested schema. "
                    "Every component must have component_key, component_name, component_goal, "
                    "component_constraints, and suggested_techniques."
                )
            last_prompt = prompt
            response = None
            try:
                response = self.llm_client.generate(
                    prompt=prompt,
                    system=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                parsed = extract_json(response.text)
                components = self._normalize_component_specs(parsed)
                trace = build_llm_trace(
                    module_name=self.__class__.__name__,
                    system=system_prompt,
                    prompt=prompt,
                    response=response,
                    request_extra={
                        "stage": "component_discovery",
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "timeout": timeout,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                    },
                    parsed={"components": components},
                )
                return components, trace
            except Exception as exc:
                last_error = exc
                last_response = response

        fallback_components = self._fallback_component_specs(existing_component_keys)
        fallback_trace = build_llm_trace(
            module_name=self.__class__.__name__,
            system=system_prompt,
            prompt=last_prompt,
            response=last_response,
            request_extra={
                "stage": "component_discovery",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "timeout": timeout,
                "attempt": max_attempts,
                "max_attempts": max_attempts,
                "fallback": True,
            },
            parsed={"components": fallback_components},
            error=last_error,
        )
        return fallback_components, fallback_trace

    def _normalize_component_specs(self, parsed: Any) -> List[Dict[str, str]]:
        """Validate and normalize the component discovery payload."""
        if not isinstance(parsed, dict):
            raise ValueError("component discovery must return a JSON object")
        raw_components = parsed.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise ValueError("component discovery returned no components")

        components: List[Dict[str, str]] = []
        seen = set()
        for item in raw_components:
            if not isinstance(item, dict):
                continue
            component_name = str(item.get("component_name") or "").strip()
            component_goal = str(item.get("component_goal") or "").strip()
            component_constraints = str(item.get("component_constraints") or "").strip()
            suggested_techniques = str(item.get("suggested_techniques") or "").strip()
            raw_key = str(item.get("component_key") or component_name).strip()
            component_key = self._normalize_component_key(raw_key)
            if (
                not component_key
                or not component_name
                or not component_goal
                or not component_constraints
                or not suggested_techniques
                or component_key in seen
            ):
                continue
            components.append(
                {
                    "component_key": component_key,
                    "component_name": component_name,
                    "component_goal": component_goal,
                    "component_constraints": component_constraints,
                    "suggested_techniques": suggested_techniques,
                }
            )
            seen.add(component_key)

        if not components:
            raise ValueError("component discovery produced no valid components")
        return components[: self.component_discovery_batch_size]

    def _fallback_component_specs(
        self,
        existing_component_keys: List[str],
    ) -> List[Dict[str, str]]:
        """Provide a minimal fallback when component discovery output is unusable."""
        candidates = [
            {
                "component_key": "feature_pipeline",
                "component_name": "Feature Pipeline",
                "component_goal": "Improve representation quality or input preprocessing without changing the core algorithm family.",
                "component_constraints": "Keep the existing model family and training loop mostly intact; only target feature construction, preprocessing, or representation flow.",
                "suggested_techniques": "feature normalization, feature selection, richer feature extraction, representation cleanup",
            },
            {
                "component_key": "loss_and_objective",
                "component_name": "Loss and Objective Design",
                "component_goal": "Improve optimization pressure by refining the loss, objective, or scoring alignment.",
                "component_constraints": "Keep the overall architecture intact and change mainly the objective-related pieces.",
                "suggested_techniques": "auxiliary losses, reweighting, calibration-aware objectives, class-balance adjustments",
            },
            {
                "component_key": "post_processing",
                "component_name": "Post Processing",
                "component_goal": "Improve final predictions with decision-time logic that preserves the core model.",
                "component_constraints": "Do not rewrite the main training pipeline; focus on inference-time refinement only.",
                "suggested_techniques": "threshold tuning, calibration, ensembling rules, test-time refinement",
            },
            {
                "component_key": "data_filtering_and_cleaning",
                "component_name": "Data Filtering and Cleaning",
                "component_goal": "Improve signal quality by refining sample filtering, cleaning, or input sanitation without changing the core model family.",
                "component_constraints": "Keep the main model and optimization path stable; focus on data curation, cleaning rules, or conservative sample handling only.",
                "suggested_techniques": "outlier filtering, label cleanup, sample reweighting, conservative data sanitation",
            },
        ]
        components = []
        for candidate in candidates:
            if candidate["component_key"] in existing_component_keys:
                continue
            components.append(candidate)
        return components[: self.component_discovery_batch_size]

    def _normalize_component_key(self, raw_key: str) -> str:
        """Normalize a semantic component key for state storage and routing."""
        normalized = re.sub(r"[^a-z0-9]+", "_", raw_key.lower()).strip("_")
        return normalized[:64]

    def _enqueue_plan_task(
        self,
        island_state: Dict[str, Any],
        plan: Dict[str, Any],
        *,
        priority: Optional[int] = None,
    ) -> bool:
        """Queue the next executable round for a plan if it is not already queued or inflight."""
        if plan.get("status") != "active":
            return False
        if int(plan.get("remaining_budget", 0)) <= 0:
            return False
        if self._has_queued_task(island_state, str(plan["plan_id"])):
            return False
        if str(plan["plan_id"]) in island_state.get("inflight_tasks", {}):
            return False

        island_state["dispatch_sequence"] = int(island_state.get("dispatch_sequence", 0)) + 1
        task = {
            "task_id": f"task_{plan['plan_id']}_{generate_short_uuid()}",
            "plan_id": str(plan["plan_id"]),
            "task_kind": str(plan["kind"]),
            "priority": int(
                priority
                if priority is not None
                else self._task_priority_for_plan(plan, continuation=False)
            ),
            "sequence": int(island_state["dispatch_sequence"]),
        }
        ready_queue = island_state.setdefault("ready_queue", [])
        ready_queue.append(task)
        ready_queue.sort(key=lambda item: (item["priority"], item["sequence"]))
        return True

    def _pop_next_task(self, island_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Pop the next ready task from the queue."""
        ready_queue = island_state.get("ready_queue", [])
        if not ready_queue:
            return None
        return ready_queue.pop(0)

    def _has_queued_task(self, island_state: Dict[str, Any], plan_id: str) -> bool:
        """Return whether a plan already has a queued executable task."""
        return any(task.get("plan_id") == plan_id for task in island_state.get("ready_queue", []))

    def _ready_depth(self, island_state: Dict[str, Any]) -> int:
        """Return total immediately available and already-dispatched work."""
        return len(island_state.get("ready_queue", [])) + len(
            island_state.get("inflight_tasks", {})
        )

    def _task_priority_for_plan(
        self,
        plan: Dict[str, Any],
        *,
        continuation: bool,
    ) -> int:
        """Assign priority so continuing chains run before starting new low-priority work."""
        base = self.TASK_PRIORITY.get(str(plan.get("kind")), 50)
        return base - 5 if continuation else base

    def _should_run_component_discovery(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Return whether another dynamic component discovery pass should run now."""
        if self._has_open_component_batch(island_state):
            return False

        completed_non_component = len(
            [
                plan
                for plan in self._iter_completed_plans(island_state)
                if plan.get("kind") != self.COMPONENT_PLAN
            ]
        )
        if completed_non_component < self.component_discovery_min_prior_plans:
            return False

        last_iteration = island_state.get("last_component_discovery_iteration")
        if last_iteration is not None and ctx.iteration - int(last_iteration) < self.component_discovery_cooldown:
            return False

        live_component_plans = [
            plan
            for plan in island_state.get("plans", {}).values()
            if isinstance(plan, dict)
            and plan.get("kind") == self.COMPONENT_PLAN
            and plan.get("status") == "active"
        ]
        return len(live_component_plans) < self.component_discovery_batch_size

    def _should_schedule_crossover(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Return whether crossover should be inserted now."""
        ready_batch = self._get_ready_component_batch(island_state)
        if ready_batch is None:
            return False
        if self._has_live_plan_kind(island_state, self.CROSSOVER_PLAN):
            return False
        last_iteration = island_state.get("last_crossover_iteration")
        if last_iteration is not None and ctx.iteration - int(last_iteration) < self.crossover_cooldown:
            return False
        return True

    def _has_open_component_batch(self, island_state: Dict[str, Any]) -> bool:
        """Return whether there is a component batch still running or waiting to cross."""
        return any(
            isinstance(batch, dict)
            and batch.get("status") in {"running", "ready_for_crossover", "crossing"}
            for batch in island_state.get("component_batches", [])
        )

    def _get_ready_component_batch(
        self,
        island_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Return the earliest component batch whose plans have all completed."""
        for batch in island_state.get("component_batches", []):
            if not isinstance(batch, dict):
                continue
            if batch.get("status") in {"crossed", "crossing"}:
                continue
            plan_ids = [str(plan_id) for plan_id in list(batch.get("plan_ids") or []) if str(plan_id)]
            if len(plan_ids) < self.crossover_min_component_plans:
                continue
            all_completed = True
            for plan_id in plan_ids:
                plan = island_state.get("plans", {}).get(plan_id)
                if not isinstance(plan, dict) or plan.get("status") != "completed":
                    all_completed = False
                    break
            if all_completed:
                batch["status"] = "ready_for_crossover"
                return batch
        return None

    def _mark_component_batch_crossed(
        self,
        island_state: Dict[str, Any],
        batch_id: str,
    ) -> None:
        """Mark one component batch as fully completed after crossover finishes."""
        if not batch_id:
            return
        for batch in island_state.get("component_batches", []):
            if not isinstance(batch, dict):
                continue
            if str(batch.get("batch_id") or "") != batch_id:
                continue
            batch["status"] = "crossed"
            if batch_id == str(island_state.get("initial_component_batch_id") or ""):
                island_state["initial_component_batch_crossed"] = True
            return

    def _has_component_batch(
        self,
        island_state: Dict[str, Any],
        batch_id: str,
    ) -> bool:
        """Return whether the given component batch exists in island state."""
        if not batch_id:
            return False
        return any(
            isinstance(batch, dict) and str(batch.get("batch_id") or "") == batch_id
            for batch in island_state.get("component_batches", [])
        )

    def _are_all_plans_completed(
        self,
        island_state: Dict[str, Any],
        plan_ids: List[str],
    ) -> bool:
        """Return whether all named plans exist and are completed."""
        if not plan_ids:
            return False
        for plan_id in plan_ids:
            plan = island_state.get("plans", {}).get(str(plan_id))
            if not isinstance(plan, dict) or plan.get("status") != "completed":
                return False
        return True

    def _is_component_batch_crossed(
        self,
        island_state: Dict[str, Any],
        batch_id: str,
    ) -> bool:
        """Return whether the named component batch already finished crossover."""
        if not batch_id:
            return False
        for batch in island_state.get("component_batches", []):
            if not isinstance(batch, dict):
                continue
            if str(batch.get("batch_id") or "") != batch_id:
                continue
            return batch.get("status") == "crossed"
        return False

    def _has_live_plan_kind(self, island_state: Dict[str, Any], kind: str) -> bool:
        """Return whether an active/queued/inflight plan of the given kind already exists."""
        for plan in island_state.get("plans", {}).values():
            if not isinstance(plan, dict):
                continue
            if plan.get("kind") != kind:
                continue
            if plan.get("status") == "active":
                return True
        return False

    def _extract_plan_id(self, result: RolloutResult) -> Optional[str]:
        """Locate strategy-specific plan metadata from rollout selection."""
        selection_extra = result.selection.extra if result.selection else {}
        if not isinstance(selection_extra, dict):
            return None
        if isinstance(selection_extra.get("fusion_context"), dict):
            return selection_extra["fusion_context"].get("plan_id")
        if isinstance(selection_extra.get("plan_context"), dict):
            return selection_extra["plan_context"].get("plan_id")
        return None

    def _iter_completed_plans(self, island_state: Dict[str, Any]) -> Sequence[Dict[str, Any]]:
        """Return completed plans in creation order."""
        return [
            island_state["plans"][plan_id]
            for plan_id in island_state.get("plan_order", [])
            if island_state["plans"].get(plan_id, {}).get("status") == "completed"
        ]

    def _iter_alive_plan_candidates(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return plans that still have a visible best program in the current island."""
        candidates: List[Dict[str, Any]] = []
        for plan_id in island_state.get("plan_order", []):
            plan = island_state["plans"].get(plan_id)
            if not isinstance(plan, dict):
                continue
            best_program = self._get_program(ctx, str(plan.get("best_program_id") or ""))
            if best_program is None:
                continue
            summary = self._build_plan_summary(plan)
            summary["best_program_id"] = best_program.id
            summary["best_score"] = best_program.combined_score
            candidates.append(summary)
        candidates.sort(
            key=self._plan_candidate_sort_key,
        )
        return candidates

    def _iter_alive_successful_plan_candidates(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return alive plans that have produced at least one valid exec rollout."""
        candidates = [
            plan
            for plan in self._iter_alive_plan_candidates(ctx, island_state)
            if self._plan_has_successful_exec(
                island_state.get("plans", {}).get(str(plan.get("plan_id") or ""), {})
            )
        ]
        candidates.sort(key=self._plan_candidate_sort_key)
        return candidates

    def _select_exploit_focus_candidate(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Select the current most worthwhile alive plan to deepen."""
        candidates = self._iter_alive_plan_candidates(ctx, island_state)
        if not candidates:
            return None
        return candidates[0]

    def _build_existing_component_labels(
        self,
        *,
        existing_component_keys: List[str],
        prior_plans: List[Dict[str, Any]],
    ) -> List[str]:
        """Build semantic labels for component directions already explored."""
        labels: List[str] = []
        seen = set()
        plan_by_key: Dict[str, Dict[str, Any]] = {}
        for plan in prior_plans:
            if not isinstance(plan, dict):
                continue
            component_key = str(plan.get("plan_focus") or "").strip()
            if component_key:
                plan_by_key[component_key] = plan

        for component_key in existing_component_keys:
            normalized_key = str(component_key).strip()
            if not normalized_key:
                continue
            plan = plan_by_key.get(normalized_key, {})
            component_name = str(plan.get("component_name") or "").strip()
            label = component_name or normalized_key.replace("_", " ")
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
        return labels

    def _resolve_current_parent(self, ctx: Context, plan: Dict[str, Any]) -> Program:
        """Resolve the currently best visible parent for the active plan."""
        candidate_ids = [
            str(plan.get("best_program_id") or ""),
            *[str(pid) for pid in reversed(plan.get("child_program_ids", []))],
            str(plan.get("seed_parent_id") or ""),
        ]
        for program_id in candidate_ids:
            program = self._get_program(ctx, program_id)
            if program is not None:
                return program
        current_best = self._get_current_best_program(ctx)
        if current_best is None:
            raise ValueError("MLPipelineStrategy could not resolve a visible plan parent")
        return current_best

    def _get_visible_programs(self, ctx: Context) -> List[Program]:
        """Return programs visible inside the current island."""
        if ctx.island_accessor:
            return ctx.island_accessor.get_all()
        if ctx.accessor:
            return ctx.accessor.get_all()
        return []

    def _get_current_best_program(self, ctx: Context) -> Optional[Program]:
        """Return the current best visible program by combined score."""
        visible_programs = self._get_visible_programs(ctx)
        if not visible_programs:
            return None
        return max(
            visible_programs,
            key=lambda program: program.combined_score
            if program.combined_score is not None
            else float("-inf"),
        )

    def _get_program(self, ctx: Context, program_id: str) -> Optional[Program]:
        """Resolve one program id from the island or global accessor."""
        if not program_id:
            return None
        if ctx.island_accessor:
            program = ctx.island_accessor.get_by_id(program_id)
            if program is not None:
                return program
        return ctx.get_program_by_id(program_id)

    def _should_schedule_model_fusion(
        self,
        ctx: Context,
        island_state: Dict[str, Any],
    ) -> bool:
        """Return whether adaptive phase should insert a model-fusion rollout."""
        last_fusion_iteration = island_state.get("last_model_fusion_iteration")
        if last_fusion_iteration is not None:
            if ctx.iteration - int(last_fusion_iteration) < self.model_fusion_cooldown:
                return False
        best_program = self._get_current_best_program(ctx)
        if best_program is None:
            return False
        successful_plan_candidates = self._iter_alive_successful_plan_candidates(
            ctx,
            island_state,
        )
        used_source_plan_ids = self._get_fused_source_plan_ids(best_program)
        required_successful_plan_count = max(
            self.model_fusion_trigger_gap,
            len(used_source_plan_ids) + self.model_fusion_trigger_gap,
        )
        if len(successful_plan_candidates) < required_successful_plan_count:
            return False

        best_plan_id = str(best_program.meta.get("plan_id") or "").strip()
        excluded_plan_ids = set(used_source_plan_ids)
        if best_plan_id:
            excluded_plan_ids.add(best_plan_id)
        return any(
            str(item.get("plan_id") or "") not in excluded_plan_ids
            for item in successful_plan_candidates
        )

    def _plan_kind(self, island_state: Dict[str, Any], plan_id: str) -> Optional[str]:
        """Read one plan kind by id."""
        plan = island_state.get("plans", {}).get(plan_id)
        return plan.get("kind") if isinstance(plan, dict) else None

    def _get_fused_source_plan_ids(self, program: Program) -> List[str]:
        """Return the plan ids already fused into the current best program."""
        source_plan_ids = program.meta.get("source_plan_ids")
        if not isinstance(source_plan_ids, list):
            return []
        return [
            str(plan_id).strip()
            for plan_id in source_plan_ids
            if str(plan_id).strip()
        ]

    def _has_valid_solution(self, program: Optional[Program]) -> bool:
        """Return whether a program counts as a valid solution for plan success tracking."""
        if program is None:
            return False
        validity = program.validity
        return (
            isinstance(validity, (int, float))
            and not isinstance(validity, bool)
            and float(validity) > 0.0
        )

    def _plan_has_successful_exec(self, plan: Dict[str, Any]) -> bool:
        """Return whether a plan has ever produced a valid exec rollout."""
        if not isinstance(plan, dict):
            return False
        if plan.get("has_successful_exec"):
            return True
        successful_exec_count = plan.get("successful_exec_count")
        if isinstance(successful_exec_count, int) and successful_exec_count > 0:
            return True
        best_validity = plan.get("best_validity")
        round_count = plan.get("round_count")
        return (
            isinstance(round_count, int)
            and round_count > 0
            and isinstance(best_validity, (int, float))
            and not isinstance(best_validity, bool)
            and float(best_validity) > 0.0
        )

    def _plan_candidate_sort_key(self, item: Dict[str, Any]) -> tuple[float, float]:
        """Sort higher-score plans first, then prefer shorter eval time on ties."""
        score = item.get("best_score")
        eval_time = item.get("best_eval_wall_time")
        normalized_score = (
            float(score)
            if isinstance(score, (int, float)) and not isinstance(score, bool)
            else float("-inf")
        )
        normalized_eval_time = (
            float(eval_time)
            if isinstance(eval_time, (int, float)) and not isinstance(eval_time, bool)
            else float("inf")
        )
        return (-normalized_score, normalized_eval_time)

    def _make_rng(self, ctx: Context, salt: str) -> random.Random:
        """Create a deterministic RNG for scheduling choices."""
        seed_material = f"{ctx.experiment_id}:{ctx.island_id}:{ctx.iteration}:{salt}"
        digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def _is_better_score(
        self,
        candidate: Optional[float],
        incumbent: Optional[float],
    ) -> bool:
        """Return whether one score is strictly better than the incumbent."""
        if candidate is None:
            return False
        if incumbent is None:
            return True
        return candidate > incumbent

    def _is_better_program(
        self,
        program: Program,
        incumbent_score: Optional[float],
        incumbent_validity: Optional[float],
    ) -> bool:
        """Return whether program should become the plan-local best."""
        if program.validity is not None and incumbent_validity is not None:
            if program.validity != incumbent_validity:
                return program.validity > incumbent_validity
        return self._is_better_score(program.combined_score, incumbent_score)


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Create the registrable ML-pipeline strategy entrypoint."""
    strategy = MLPipelineStrategy(evaluate_fn=evaluate_fn, params=params)
    return {
        "strategy": strategy,
        "description": "Independent ML pipeline strategy with queue-based no-bubble scheduling",
        "tags": ["machine-learning", "planning", "fusion", "scheduler"],
        "author": "Famou Framework",
    }
