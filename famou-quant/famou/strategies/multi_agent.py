"""
策略说明：
每次会检查是否有空闲session（一个session相当于一个计划，每个计划有多个todo，顺序执行）
如果有空闲session，则复用这个session，继续未完成的todo
如果没有空闲session，则检查session数量是否到达上限
    若到达上限，则降级，做一个普通generate
    如果未到达上限，则创建一个新session

Multi-Agent Strategy

Description: Multi-Agent Planner with parallel Agents per island
Author: Famou Framework
Tags: multi-agent, planning, parallel

Pipeline:
1. MultiAgentSelect: Match/create Agent Session → forced parent + Rich Action
2. BlockTargetGenerate: Generate modified block based on action
3. EvaluateModule: Evaluate modified program
4. _SafeLLMJudge: LLM error analysis (non-fatal)
5. AgentFeedbackJudge: Update Agent Session state

Requirements:
- population.k must be large (e.g. 9999) to prevent program eviction
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

import yaml

from famou.config.settings import ModulesConfig
from famou.core.data import Context, Rollout, RolloutResult
from famou.core.protocol import Strategy
from famou.core.types import RolloutStatus
from famou.modules.evaluate import EvaluateModule
from famou.modules.judge.llm_judge import LLMJudge
from famou.modules.population.topk import TopKPopulation

from famou.modules.generate.block_generate import BlockTargetGenerate
from famou.modules.judge.agent_feedback import AgentFeedbackJudge
from famou.modules.planning.multi_agent_select import (
    AGENT_SESSIONS_STATE_KEY,
    AGENT_SESSION_ASSIGNMENT_KEY,
    AgentSession,
    MultiAgentSelect,
    count_active_sessions,
    find_free_session,
    mark_session_rollout_failed,
    set_session_assignment,
)

# Default module parameters live alongside this strategy.
# Per-section shallow merge: user config overrides these, unspecified keys inherit.
_DEFAULTS_PATH = Path(__file__).with_name("multi_agent.yaml")


def _load_module_defaults() -> Dict[str, Dict[str, Any]]:
    """Load default module parameters for the multi-agent strategy."""
    if not _DEFAULTS_PATH.exists():
        return {}
    with open(_DEFAULTS_PATH) as f:
        data = yaml.safe_load(f) or {}
    return data.get("modules", {})


def _normalize_module_params(
    params: Optional[ModulesConfig],
) -> Dict[str, Any]:
    """Normalize ModulesConfig or plain dict into sectioned overrides."""
    if params is None:
        return {}
    if hasattr(params, "select"):
        return {
            "select": dict(params.select) if params.select else {},
            "generate": dict(params.generate) if params.generate else {},
            "evaluate": dict(params.evaluate) if params.evaluate else {},
            "judge": dict(params.judge) if params.judge else {},
            "population": dict(params.population) if params.population else {},
        }
    return dict(params)


class _SafeLLMJudge(LLMJudge):
    """
    LLMJudge wrapper that catches errors instead of failing the pipeline.

    LLMJudge provides error analysis for AgentFeedbackJudge.
    If it fails (LLM timeout etc.), pipeline continues.
    """

    def execute(self, context: Context, result: RolloutResult, **kwargs) -> RolloutResult:
        try:
            return super().execute(context, result, **kwargs)
        except Exception as e:
            self.log_error(f"LLMJudge failed (non-fatal): {e}")
            return result


class MultiAgentStrategy(Strategy):
    """Multi-agent planner strategy with main-process session assignment."""

    def __init__(
        self,
        evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        params: Optional[ModulesConfig] = None,
    ):
        self.evaluate_fn = evaluate_fn

        user_overrides = _normalize_module_params(params)
        defaults = _load_module_defaults()

        def merge_section(section: str) -> Dict[str, Any]:
            return {
                **defaults.get(section, {}),
                **user_overrides.get(section, {}),
            }

        select_params = merge_section("select")
        generate_params = merge_section("generate")
        evaluate_params = merge_section("evaluate")
        judge_params = merge_section("judge")
        population_params = merge_section("population")

        self.max_agents_per_island = int(select_params.get("max_agents_per_island", 3))
        self.max_stagnation = int(select_params.get("max_stagnation", 3))
        self.zombie_warning_iterations = int(
            select_params.get("zombie_warning_iterations")
        )

        select_module = MultiAgentSelect(**select_params)
        generate_module = BlockTargetGenerate(**generate_params)
        evaluate_module = EvaluateModule(evaluate_fn=evaluate_fn, **evaluate_params)
        llm_judge = _SafeLLMJudge()
        feedback_module = AgentFeedbackJudge(**judge_params)

        self.rollout = Rollout(
            modules=[
                select_module,
                generate_module,
                evaluate_module,
                llm_judge,
                feedback_module,
            ],
            name="multi_agent_rollout",
        )
        self.population_module = TopKPopulation(**population_params)
        self._modules = {
            "select": select_module,
            "generate": generate_module,
            "evaluate": evaluate_module,
            "llm_judge": llm_judge,
            "feedback": feedback_module,
        }

    def _load_live_records(self, island_id: int) -> Dict[str, Any]:
        """get {session_id: session} dict of corresponding island id"""
        if not self.state_store:
            return {}
        raw = self.state_store.get_island(
            island_id,
            AGENT_SESSIONS_STATE_KEY,
            default={},
        )
        return raw if isinstance(raw, dict) else {}

    def _warn_zombie_sessions(
        self,
        records: Dict[str, Any],
        current_iteration: int,
        island_id: int,
    ) -> None:
        """Detect sessions stuck in_flight too long. Log warning only -- no auto-release."""
        for _, record in records.items():
            try:
                session = AgentSession.from_dict(record)
            except Exception:
                continue
            if not session.in_flight or session.claimed_at_iteration is None:
                continue
            age = current_iteration - session.claimed_at_iteration
            if age > self.zombie_warning_iterations:
                print(
                    f"[{island_id} | {current_iteration} | MultiAgent] Zombie session detected: "
                    f"{session.session_id}, in_flight for {age} iterations "
                    f"(claimed at iter {session.claimed_at_iteration})"
                )

    def forward(self, ctx: Context, rollout_history: List[RolloutResult]) -> Rollout:
        """create/resume session, or degrade
        
        If the number of active sessions is less than or equal to the threshold, create a new session; 
        otherwise, if there is a session whose in_flight status is false, resume it; 
        otherwise, degrade.
        """
        del rollout_history

        if ctx.iteration == 0 or not self.state_store:
            return self.rollout

        island_id = ctx.island_id
        records = self._load_live_records(island_id)

        # Zombie detection (warning only, no auto-release)
        self._warn_zombie_sessions(records, ctx.iteration, island_id)

        # Resume: find a free session (active, not in_flight, has plan)
        session = find_free_session(records)
        if session:
            claim_id = f"claim_{uuid4().hex[:8]}"
            session.claim(claim_id, iteration=ctx.iteration)
            self.state_store.set_island(
                island_id, AGENT_SESSIONS_STATE_KEY,
                session.session_id, value=session.to_dict(),
            )
            set_session_assignment(ctx, {
                "action": "resume",
                "session_id": session.session_id,
                "claim_id": claim_id,
                "session": session.to_dict(),
            }, AGENT_SESSION_ASSIGNMENT_KEY)
            return self.rollout

        # Create: if under capacity
        active_count = count_active_sessions(records)
        if active_count < self.max_agents_per_island:
            placeholder = AgentSession.create_placeholder(
                iteration=ctx.iteration,
                max_stagnation=self.max_stagnation,
            )
            claim_id = f"claim_{uuid4().hex[:8]}"
            placeholder.claim(claim_id, iteration=ctx.iteration)
            self.state_store.set_island(
                island_id, AGENT_SESSIONS_STATE_KEY,
                placeholder.session_id, value=placeholder.to_dict(),
            )
            set_session_assignment(ctx, {
                "action": "create",
                "session_id": placeholder.session_id,
                "claim_id": claim_id,
                "created_at_iteration": ctx.iteration,
            }, AGENT_SESSION_ASSIGNMENT_KEY)            
            return self.rollout

        # Degrade: all active sessions are in_flight
        set_session_assignment(ctx, {"action": "degraded"}, AGENT_SESSION_ASSIGNMENT_KEY)
        return self.rollout

    def on_rollout_complete(self, result: RolloutResult) -> None:
        """Clean up sessions for failed/discarded rollouts. Runs AFTER apply_updates.

        Success rollouts already handled by Feedback's state_updates.
        """
        if result.status == RolloutStatus.SUCCESS:
            return

        if not result.selection or not result.selection.extra:
            return
        assignment = result.selection.extra.get(AGENT_SESSION_ASSIGNMENT_KEY)
        if not isinstance(assignment, dict):
            return

        action = assignment.get("action")
        session_id = assignment.get("session_id")
        claim_id = assignment.get("claim_id")

        if not session_id or action == "degraded":
            return

        island_id = result.island_id

        record = self.state_store.get_island(
            island_id, AGENT_SESSIONS_STATE_KEY, session_id, default=None,
        )
        if not record:
            return

        # CAS: only process if we still own this session
        if record.get("flight_rollout_id") != claim_id:
            return

        try:
            session = AgentSession.from_dict(record)
        except Exception:
            self.state_store.delete_island(
                island_id, AGENT_SESSIONS_STATE_KEY, session_id,
            )
            return

        if session.plan is None:
            # Placeholder: Select never completed -> delete
            self.state_store.delete_island(
                island_id, AGENT_SESSIONS_STATE_KEY, session_id,
            )
            return

        # Full session: mark failed + release
        reason = result.error_message or f"rollout {result.status}"
        mark_session_rollout_failed(session, reason, result.iteration)
        self.state_store.set_island(
            island_id, AGENT_SESSIONS_STATE_KEY,
            session_id, value=session.to_dict(),
        )


    def dump_state(self) -> dict[str, Any]:
        """
        Serialize the strategy state for experiment checkpointing.

        Captures the backing state store's per-island state, including active
        multi-agent sessions, so the controller can persist it as the strategy
        checkpoint.

        Returns:
            dict[str, Any]: JSON-serializable strategy checkpoint containing
            the ``island_state`` mapping.
        """
        island_state = {}
        if self.state_store:
            island_state = self.state_store.model_dump(mode="json").get(
                "island_state",
                {},
            )
        return {
            "island_state": island_state,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Restore the strategy state from a previously dumped checkpoint.

        Active sessions are marked as not in flight before being written back to
        the state store

        Args:
            state: Strategy checkpoint created by ``dump_state``. Invalid or
                incomplete input is ignored.
        """
        island_state = state.get("island_state")
        if not isinstance(island_state, dict):
            return

        for raw_island_id, island_data in island_state.items():
            if not isinstance(island_data, dict):
                continue

            sessions = island_data.get(AGENT_SESSIONS_STATE_KEY)
            if isinstance(sessions, dict):
                for session in sessions.values():
                    if not isinstance(session, dict):
                        continue
                    if session.get("status") == "active":
                        session["in_flight"] = False

            if not self.state_store:
                continue

            try:
                island_id = int(raw_island_id)
            except (TypeError, ValueError):
                continue

            for key, value in island_data.items():
                self.state_store.set_island(island_id, key, value=value)


def create_strategy(
    evaluate_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
    params: Optional[ModulesConfig] = None,
    evaluate_module=None,
    *args,
    **kwargs
):
    """
    Create the multi-agent strategy.

    Configuration Example (in YAML):
        strategy: multi_agent
        modules:
          select:
            max_agents_per_island: 3
            max_stagnation: 3
            max_todos_per_plan: 5
            max_attempts_per_todo: 3
            selection_strategy: "tournament"
            default_blocks: ["A", "B", "C", "D"]
          generate:
            include_inspirations: true
            temperature: 0.8
          evaluate:
            timeout: 60
          population:
            k: 9999   # REQUIRED: prevent eviction
    """
    strategy = MultiAgentStrategy(evaluate_fn=evaluate_fn, params=params)
    return {
        "strategy": strategy,
        "population_module": strategy.population_module,
        "description": "Multi-Agent Planner with parallel Agents per island",
        "tags": ["multi-agent", "planning", "parallel"],
        "author": "Famou Framework",
        "_modules": strategy._modules,
    }
