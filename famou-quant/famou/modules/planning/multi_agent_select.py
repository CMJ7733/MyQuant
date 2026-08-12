"""
MultiAgentSelect - 多 Agent 选择模块。

核心职责:
1. 为当前 rollout 匹配或创建 Agent Session
2. Agent-guided rollout: 强制 parent + Rich Action (Plan/Todo)
3. 降级 rollout: 委托 SelectModule + stochastic action

并发模型:
- 每 island 可有多个 Agent (max_agents_per_island)
- Strategy.forward() 在主线程/主进程串行 claim/reserve session
- worker 只按 context.metadata["agent_session_assignment"] 执行分配
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from famou.core.data import Context, RolloutResult, SelectionData
from famou.core.protocol import Module, RequiresLLM
from famou.modules.planning.block_detection import detect_blocks_from_code
from famou.modules.planning.plan.data import Plan, TodoItem, TodoStatus
from famou.modules.planning.planners.base import PlannerAction
from famou.modules.select.base import SelectModule
from famou.modules.select.elite import EliteSelect
from famou.modules.select.random import RandomSelect
from famou.modules.select.tournament import TournamentSelect
from famou.prompts import prompt_registry

if TYPE_CHECKING:
    from famou.infrastructure.llm.base import LLMClient

logger = logging.getLogger("famou")


AGENT_SESSIONS_STATE_KEY = "agent_sessions"
AGENT_SESSION_ASSIGNMENT_KEY = "agent_session_assignment"


@dataclass
class AgentSession:
    """
    Agent 会话状态。

    Lifecycle:
        active -> completed (Plan 完成) / stagnated (连续无改进)

    working_program_id 更新链:
        checkout P0 -> Todo 成功 -> C1 -> Todo 成功 -> C2 -> ...
        Todo 失败 -> 保持不变，重试
    """

    session_id: str
    plan: Optional[Plan] = None

    # Program IDs (仅存 ID，轻量)
    working_program_id: str = ""
    checkout_program_id: str = ""
    best_program_id: str = ""
    best_score: float = 0.0

    # 会话状态
    status: str = "active"  # "active" | "completed" | "stagnated"

    # 并发控制 (Select claim -> Feedback release)
    in_flight: bool = False
    flight_rollout_id: Optional[str] = None
    claimed_at_iteration: Optional[int] = None

    # 双层 Stagnation - Session 级别
    session_stagnation_counter: int = 0
    max_stagnation: int = 3

    # 生命周期
    created_at_iteration: int = 0
    completed_at_iteration: Optional[int] = None

    # 跨 Agent 知识共享
    session_summary: Optional[str] = None

    # 历史记录
    history: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Whether this session is still active."""
        return self.status == "active"

    @classmethod
    def create_placeholder(
        cls,
        iteration: int,
        max_stagnation: int = 3,
    ) -> "AgentSession":
        """Create placeholder session for forward() to reserve a slot."""
        return cls(
            session_id=f"session_{uuid.uuid4().hex[:8]}",
            plan=None,
            in_flight=True,
            claimed_at_iteration=iteration,
            created_at_iteration=iteration,
            max_stagnation=max_stagnation,
        )

    @classmethod
    def create(
        cls,
        plan: Plan,
        program_id: str,
        program_score: float,
        iteration: int = 0,
        max_stagnation: int = 3,
        session_id: Optional[str] = None,
    ) -> "AgentSession":
        """
        Create a fully initialized agent session from a selected plan and program.

        Args:
            plan: Planner-generated plan that the agent session will execute.
            program_id: Identifier of the program used as the session checkout,
                working, and initial best program.
            program_score: Score of the initial best program.
            iteration: Iteration at which the session is created.
            max_stagnation: Number of consecutive non-improving attempts allowed
                before the session is considered stagnated.
            session_id: Optional stable session identifier. If omitted, a new
                short random identifier is generated.

        Returns:
            AgentSession: Active session ready to be claimed by a rollout.
        """
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        return cls(
            session_id=session_id,
            plan=plan,
            working_program_id=program_id,
            checkout_program_id=program_id,
            best_program_id=program_id,
            best_score=program_score,
            created_at_iteration=iteration,
            max_stagnation=max_stagnation,
        )

    def claim(self, rollout_id: str, iteration: Optional[int] = None) -> None:
        """标记 session 为 in_flight (被某个 rollout 占用)。"""
        self.in_flight = True
        self.flight_rollout_id = rollout_id
        self.claimed_at_iteration = iteration

    def release(self) -> None:
        """释放 in_flight 标记。"""
        self.in_flight = False
        self.flight_rollout_id = None
        self.claimed_at_iteration = None

    def add_history(self, event: str, **details: Any) -> None:
        """记录历史事件。"""
        self.history.append({
            "event": event,
            "timestamp": time.time(),
            **details,
        })

    def to_dict(self) -> Dict[str, Any]:
        """序列化为 dict (用于 StateStore 持久化)。"""
        return {
            "session_id": self.session_id,
            "plan": self.plan.to_dict() if self.plan else None,
            "working_program_id": self.working_program_id,
            "checkout_program_id": self.checkout_program_id,
            "best_program_id": self.best_program_id,
            "best_score": self.best_score,
            "status": self.status,
            "in_flight": self.in_flight,
            "flight_rollout_id": self.flight_rollout_id,
            "claimed_at_iteration": self.claimed_at_iteration,
            "session_stagnation_counter": self.session_stagnation_counter,
            "max_stagnation": self.max_stagnation,
            "created_at_iteration": self.created_at_iteration,
            "completed_at_iteration": self.completed_at_iteration,
            "session_summary": self.session_summary,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSession":
        """从 dict 反序列化。"""
        plan_data = data.get("plan")
        plan = Plan.from_dict(plan_data) if plan_data else None
        return cls(
            session_id=data["session_id"],
            plan=plan,
            working_program_id=data["working_program_id"],
            checkout_program_id=data["checkout_program_id"],
            best_program_id=data["best_program_id"],
            best_score=data.get("best_score", 0.0),
            status=data.get("status", "active"),
            in_flight=data.get("in_flight", False),
            flight_rollout_id=data.get("flight_rollout_id"),
            claimed_at_iteration=data.get("claimed_at_iteration"),
            session_stagnation_counter=data.get("session_stagnation_counter", 0),
            max_stagnation=data.get("max_stagnation", 3),
            created_at_iteration=data.get("created_at_iteration", 0),
            completed_at_iteration=data.get("completed_at_iteration"),
            session_summary=data.get("session_summary"),
            history=data.get("history", []),
        )


def deserialize_sessions(
    raw: Optional[Dict[str, Any]],
) -> Dict[str, AgentSession]:
    """Deserialize all valid AgentSession records, including placeholders."""
    if not raw:
        return {}

    sessions: Dict[str, AgentSession] = {}
    for sid, sdata in raw.items():
        try:
            sessions[sid] = AgentSession.from_dict(sdata)
        except Exception:
            continue
    return sessions


def count_active_sessions(records: Dict[str, Any]) -> int:
    """Count active sessions, including placeholders."""
    count = 0
    for session in deserialize_sessions(records).values():
        if session.status == "active":
            count += 1
    return count


def find_free_session(records: Dict[str, Any]) -> Optional[AgentSession]:
    """Find a resumable session: active, not in_flight, has plan."""
    sessions = deserialize_sessions(records)
    for session in sessions.values():
        if session.plan is not None and session.is_active and not session.in_flight:
            return session
    return None


def load_sessions(
    context: Context,
    state_key: str,
) -> Dict[str, AgentSession]:
    """Load all sessions in the current island."""
    if not context.state:
        return {}
    raw = context.state.get_island(state_key, default=None)
    return deserialize_sessions(raw)


def save_session(
    result: RolloutResult,
    session: AgentSession,
    state_key: str,
) -> None:
    """Save a single session."""
    if not result.state_updates:
        return
    result.state_updates.set_island(state_key, session.session_id, value=session.to_dict())


def get_session_assignment(context: Context, key: str) -> Dict[str, Any]:
    """Read the per-rollout session assignment from Context.metadata."""
    assignment = context.metadata.get(key, {})
    return assignment if isinstance(assignment, dict) else {}


def set_session_assignment(context: Context, assignment: Dict[str, Any], key: str) -> None:
    """Store a JSON-serializable assignment on Context.metadata."""
    context.metadata[key] = dict(assignment)


def mark_session_rollout_failed(
    session: AgentSession,
    reason: str,
    iteration: int,
) -> None:
    """Record a failed rollout attempt and release the assigned session."""
    if session.plan is None:
        session.release()
        return

    todo = session.plan.get_current_todo()
    if todo and todo.is_active:
        todo.record_attempt(result="failed", reason=reason)

    session.session_stagnation_counter += 1

    if todo and todo.attempt_count >= todo.max_attempts and todo.status != TodoStatus.COMPLETED:
        todo.status = TodoStatus.FAILED

    if session.session_stagnation_counter >= session.max_stagnation:
        session.status = "stagnated"
        session.completed_at_iteration = iteration
    elif session.plan.is_complete:
        session.status = "completed"
        session.completed_at_iteration = iteration

    session.add_history(
        "rollout_failed",
        reason=reason,
        iteration=iteration,
    )
    session.release()


def detect_blocks(
    code: str,
    language: str,
    context: Context,
    default_blocks: List[str],
    block_pattern: Optional[str] = None,
    result: Optional[RolloutResult] = None,
) -> List[str]:
    """Detect blocks from code, using cache if available."""
    blocks = detect_blocks_from_code(
        code,
        language=language,
        pattern=block_pattern,
        default_blocks=default_blocks,
    )
    return blocks


def build_rich_action(plan: Plan, todo: TodoItem) -> Dict[str, Any]:
    """Build Rich Action with full Plan/Todo context."""
    todo.status = TodoStatus.IN_PROGRESS

    return {
        "op": "modify",
        "target": todo.target_block,
        "plan_id": plan.id,
        "plan_goal": plan.goal,
        "plan_progress": plan.progress,
        "todo_id": todo.id,
        "todo_description": todo.description,
        "todo_direction": todo.direction,
        "todo_avoid": todo.avoid,
        "todo_attempt": todo.attempt_count + 1,
        "alignment_status": "normal",
        "realign_info": None,
        "previous_attempts": [
            {
                "attempt": a.attempt_number,
                "result": a.result,
                "reason": a.reason,
            }
            for a in todo.attempts
        ] if todo.attempts else None,
    }


def build_fallback_action(default_blocks: List[str]) -> Dict[str, Any]:
    """Build a simple stochastic action for degraded rollouts."""
    target = random.choice(default_blocks) if default_blocks else "A"
    return {
        "op": "modify",
        "target": target,
        "alignment_status": "normal",
        "realign_info": None,
    }


def generate_plan(
    llm_client: "LLMClient",
    parent_code: str,
    parent_id: str,
    iteration: int,
    task_description: str = "",
    available_blocks: Optional[List[str]] = None,
    max_todos: int = 5,
    max_attempts_per_todo: int = 3,
    session_summaries: Optional[List[str]] = None,
) -> Plan:
    """Generate a Plan by analyzing code with LLM."""
    arms = available_blocks or ["A", "B", "C", "D"]

    history_context = ""
    if session_summaries:
        history_context = (
            "\n\n--- 以下是之前 Agent 的经验总结，请参考避免重复错误 ---\n"
            + "\n---\n".join(session_summaries[-3:])
        )

    prompt = prompt_registry.get(
        "planning/plan_generate.txt",
        parent_code=parent_code,
        task_description=task_description + history_context,
        available_blocks=", ".join(arms),
        max_todos=max_todos,
    )

    try:
        response = llm_client.generate(
            prompt=prompt,
            system=task_description,
        )
        response_text = response.text
    except Exception as e:
        logger.warning(f"[MultiAgent] Plan generation failed: {e}, using default")
        return _create_default_plan(parent_id, iteration, arms, max_attempts_per_todo)

    try:
        plan = _parse_plan_response(
            response_text, parent_id, iteration, arms,
            max_todos, max_attempts_per_todo,
        )
        logger.info(
            f"[MultiAgent] Created Plan {plan.id}: "
            f"goal='{plan.goal}', todos={len(plan.todos)}, "
            f"anchor={parent_id}"
        )
        for i, todo in enumerate(plan.todos):
            logger.info(f"  Todo {i+1}: [{todo.target_block}] {todo.description}")
        return plan
    except Exception as e:
        logger.warning(f"[MultiAgent] Plan parsing failed: {e}, using default")
        return _create_default_plan(parent_id, iteration, arms, max_attempts_per_todo)


def _parse_plan_response(
    response: str,
    parent_id: str,
    iteration: int,
    arms: List[str],
    max_todos: int,
    max_attempts_per_todo: int,
) -> Plan:
    """Parse LLM response to extract a Plan."""
    json_data = _extract_json(response)
    if not json_data:
        raise ValueError("No valid JSON found in response")

    goal = json_data.get("goal", "代码优化")
    todos_data = json_data.get("todos", [])

    if not todos_data:
        raise ValueError("No todos in plan")

    todos = []
    for i, todo_data in enumerate(todos_data[:max_todos]):
        target = todo_data.get("target_block", "")
        target = _clean_target_block(target)
        matched_target = PlannerAction.match_target(target, arms)
        if not matched_target:
            logger.warning(
                f"[MultiAgent] Invalid target '{target}' "
                f"(original: {todo_data.get('target_block', '')}), skipping"
            )
            continue

        todo = TodoItem(
            id=f"todo_{i+1}",
            description=todo_data.get("description", f"优化 BLOCK {matched_target}"),
            target_block=matched_target,
            direction=todo_data.get("direction", ""),
            avoid=todo_data.get("avoid", []),
            max_attempts=max_attempts_per_todo,
            priority=i,
        )
        todos.append(todo)

    if not todos:
        raise ValueError("No valid todos after parsing")

    return Plan.create(
        goal=goal,
        todos=todos,
        anchor_program_id=parent_id,
        iteration=iteration,
    )


def _create_default_plan(
    parent_id: str,
    iteration: int,
    arms: List[str],
    max_attempts_per_todo: int,
) -> Plan:
    """Create a default plan when LLM fails."""
    target = arms[0] if arms else "A"
    todos = [
        TodoItem(
            id="todo_1",
            description=f"优化 BLOCK {target}",
            target_block=target,
            direction="分析并改进代码结构或算法",
            max_attempts=max_attempts_per_todo,
            priority=0,
        )
    ]
    return Plan.create(
        goal="代码优化",
        todos=todos,
        anchor_program_id=parent_id,
        iteration=iteration,
    )


def _clean_target_block(target: str) -> str:
    """Clean up target block string from various LLM output formats."""
    if not target:
        return ""
    cleaned = re.sub(
        r'^(?:block_label\s*:\s*|block\s*:\s*|block\s+|block_)',
        '',
        target.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from text, including fenced code blocks."""
    if not text:
        return None

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# Registry for creating SelectModule from string type
_SELECT_MODULE_REGISTRY = {
    "best": EliteSelect,
    "elite": EliteSelect,
    "random": RandomSelect,
    "tournament": TournamentSelect,
}


def _create_select_module(
    select_module: Optional[SelectModule] = None,
    selection_strategy: Optional[str] = None,
    num_inspirations: int = 2,
    select_module_config: Optional[Dict[str, Any]] = None,
) -> SelectModule:
    """Create a SelectModule instance from various input formats."""
    if select_module is not None:
        return select_module

    if select_module_config:
        module_type = select_module_config.pop("type", "elite")
        cls = _SELECT_MODULE_REGISTRY.get(module_type)
        if cls is None:
            raise ValueError(
                f"Unknown select module type: {module_type}. "
                f"Available: {list(_SELECT_MODULE_REGISTRY.keys())}"
            )
        return cls(**select_module_config)

    strategy = selection_strategy or "best"
    cls = _SELECT_MODULE_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown selection strategy: {strategy}. "
            f"Available: {list(_SELECT_MODULE_REGISTRY.keys())}"
        )
    return cls(num_inspirations=num_inspirations)


class MultiAgentSelect(Module, RequiresLLM):
    """
    多 Agent 选择模块。

    Pipeline Role:
    - Reads: context.accessor (population), context.state (agent sessions)
    - Writes: result.selection with parent_id and extra["planner_action"]
              extra["agent_session_id"] for AgentFeedbackJudge

    Configuration:
        max_agents_per_island: 每 island 最多 Agent 数
        max_stagnation: Session 连续无改进次数阈值
        max_todos_per_plan: 每 Plan 最多 Todo 数
        max_attempts_per_todo: 每 Todo 最多重试次数
        selection_strategy: checkout 选择策略
        default_blocks: 默认 block 列表
    """

    llm_client: "LLMClient"

    def __init__(
        self,
        max_agents_per_island: int = 3,
        max_stagnation: int = 3,
        max_todos_per_plan: int = 5,
        max_attempts_per_todo: int = 3,
        select_module: Optional[SelectModule] = None,
        selection_strategy: Optional[str] = None,
        num_inspirations: int = 2,
        select_module_config: Optional[Dict[str, Any]] = None,
        default_blocks: Optional[List[str]] = None,
        block_pattern: Optional[str] = None,
        name: Optional[str] = None,
        **config,
    ):
        super().__init__(name=name, **config)

        self.max_agents_per_island = max_agents_per_island
        self.max_stagnation = max_stagnation
        self.max_todos_per_plan = max_todos_per_plan
        self.max_attempts_per_todo = max_attempts_per_todo
        self.default_blocks = default_blocks or ["A", "B", "C", "D"]
        self.block_pattern = block_pattern

        self._select_module = _create_select_module(
            select_module=select_module,
            selection_strategy=selection_strategy,
            num_inspirations=num_inspirations,
            select_module_config=select_module_config,
        )

    # -------------------------------------------------------------------------
    # Session State Management
    # -------------------------------------------------------------------------

    def _get_completed_summaries(
        self, sessions: Dict[str, AgentSession]
    ) -> List[str]:
        """Get session summaries from completed/stagnated sessions."""
        summaries = []
        for s in sessions.values():
            if s.plan is not None and not s.is_active and s.session_summary:
                summaries.append(s.session_summary)
        return summaries

    # -------------------------------------------------------------------------
    # Agent-guided Rollout
    # -------------------------------------------------------------------------

    def _agent_rollout(
        self,
        session: AgentSession,
        context: Context,
        result: RolloutResult,
    ) -> RolloutResult:
        """
        Execute an Agent-guided rollout: forced parent + Rich Action.

        The session is already claimed (in_flight=True) before this call.
        """
        # Get working program (forced parent)
        parent = context.get_program_by_id(session.working_program_id)
        if not parent:
            if self.logger:
                self.logger.warning(
                    f"[{self.name}] Working program {session.working_program_id} "
                    f"not found for session {session.session_id}. "
                    f"Ending session."
                )
            session.status = "stagnated"
            session.release()
            save_session(result, session, AGENT_SESSIONS_STATE_KEY)
            return self._degraded_rollout(context, result)

        # Get current Todo
        todo = session.plan.get_current_todo()
        if not todo:
            if self.logger:
                self.logger.info(
                    f"[{self.name}] Session {session.session_id} has no active "
                    f"Todo (Plan complete). Marking completed."
                )
            session.status = "completed"
            session.completed_at_iteration = context.iteration
            session.release()
            save_session(result, session, AGENT_SESSIONS_STATE_KEY)
            return self._degraded_rollout(context, result)

        # Build Rich Action
        action = build_rich_action(session.plan, todo)

        # Build selection with forced parent
        parent_score = parent.combined_score or 0.0
        assignment = get_session_assignment(context, AGENT_SESSION_ASSIGNMENT_KEY)
        result.selection = SelectionData(
            parent_id=parent.id,
            inspiration_ids=[],
            extra={
                "planner_action": action,
                "detected_blocks": self.default_blocks,
                "agent_session_id": session.session_id,
                "agent_session": session.to_dict(),
                AGENT_SESSION_ASSIGNMENT_KEY: assignment,
            },
        )

        # Logging
        if self.logger:
            self.logger.info(
                f"[{self.name}] 🤖 Agent rollout: "
                f"session={session.session_id}, "
                f"parent={parent.id} (score={parent_score:.4f}), "
                f"todo=[{todo.target_block}] {todo.description}, "
                f"attempt={todo.attempt_count + 1}/{todo.max_attempts}, "
                f"plan_progress={session.plan.progress}"
            )

        return result

    # -------------------------------------------------------------------------
    # Degraded Rollout
    # -------------------------------------------------------------------------

    def _degraded_rollout(
        self, context: Context, result: RolloutResult
    ) -> RolloutResult:
        """
        Execute a degraded rollout: SelectModule + stochastic action.

        Used when all agents are busy or max_agents reached.
        Results still enter population.
        """
        # Delegate parent/inspiration selection to SelectModule
        result = self._select_module.execute(context, result)
        parent_id = result.selection.parent_id
        assignment = get_session_assignment(context, AGENT_SESSION_ASSIGNMENT_KEY)

        # Detect blocks for stochastic action
        parent = context.get_program_by_id(parent_id)
        blocks = self.default_blocks
        if parent:
            blocks = detect_blocks(
                parent.code, context.language, context,
                self.default_blocks, self.block_pattern,
                result=result,
            )

        # Build stochastic action
        action = build_fallback_action(blocks)

        # Inject into selection extra
        result.selection.extra.update({
            "planner_action": action,
            "detected_blocks": blocks,
            AGENT_SESSION_ASSIGNMENT_KEY: assignment,
        })

        if self.logger:
            self.logger.info(
                f"[{self.name}] ⬇️ Degraded rollout: "
                f"parent={parent_id}, "
                f"action=stochastic(target={action['target']})"
            )

        return result

    # -------------------------------------------------------------------------
    # Create New Session
    # -------------------------------------------------------------------------

    def _create_and_run(
        self,
        context: Context,
        result: RolloutResult,
        placeholder_session_id: str,
    ) -> RolloutResult:
        """
        Create a new Agent Session and run the first Todo.

        1. Delegate to SelectModule for parent selection (checkout)
        2. LLM generates Plan
        3. Create AgentSession (reusing placeholder's session_id), claim, and run first Todo
        """
        assignment = get_session_assignment(context, AGENT_SESSION_ASSIGNMENT_KEY)

        # 1. Select parent via SelectModule (checkout)
        result = self._select_module.execute(context, result)
        result.selection.extra[AGENT_SESSION_ASSIGNMENT_KEY] = assignment
        parent_id = result.selection.parent_id
        parent = context.get_program_by_id(parent_id)
        if not parent:
            if self.logger:
                self.logger.warning(
                    f"[{self.name}] Checkout parent {parent_id} not found"
                )
            return self._degraded_rollout(context, result)

        parent_score = parent.combined_score or 0.0

        # 2. Detect blocks
        blocks = detect_blocks(
            parent.code, context.language, context,
            self.default_blocks, self.block_pattern,
            result=result,
        )

        # 3. Get historical session summaries for cross-agent knowledge
        sessions = load_sessions(
            context,
            AGENT_SESSIONS_STATE_KEY,
        )
        summaries = self._get_completed_summaries(sessions)

        # 4. Generate Plan via LLM
        if self.logger:
            self.logger.info(
                f"[{self.name}] Creating new Agent: "
                f"checkout={parent_id} (score={parent_score:.4f}), "
                f"blocks={blocks}, "
                f"history_summaries={len(summaries)}"
            )

        plan = generate_plan(
            llm_client=self.llm_client,
            parent_code=parent.code,
            parent_id=parent_id,
            iteration=context.iteration,
            task_description=context.task_description or "",
            available_blocks=blocks,
            max_todos=self.max_todos_per_plan,
            max_attempts_per_todo=self.max_attempts_per_todo,
            session_summaries=summaries if summaries else None,
        )

        # 5. Create full session reusing placeholder's session_id
        session = AgentSession.create(
            plan=plan,
            program_id=parent_id,
            program_score=parent_score,
            iteration=context.iteration,
            max_stagnation=self.max_stagnation,
            session_id=placeholder_session_id,
        )
        claim_id = assignment.get("claim_id") or result.rollout_id
        session.claim(claim_id, iteration=context.iteration)
        session.add_history(
            "created",
            checkout_program_id=parent_id,
            checkout_score=parent_score,
            plan_goal=plan.goal,
            plan_todos=len(plan.todos),
        )

        # 6. Overwrite placeholder with full session (same key, no delete needed)
        save_session(result, session, AGENT_SESSIONS_STATE_KEY)

        if self.logger:
            self.logger.info(
                f"[{self.name}] Agent created: "
                f"session={session.session_id}, "
                f"plan_goal='{plan.goal}', "
                f"todos={len(plan.todos)}, "
                f"island={context.island_id}"
            )

        # 7. Run first todo
        return self._agent_rollout(session, context, result)

    # -------------------------------------------------------------------------
    # Module Protocol
    # -------------------------------------------------------------------------

    def validate_input(self, context: Context, result: RolloutResult) -> None:
        """Validate that population is available."""
        if not context.accessor:
            raise ValueError(
                f"{self.name}: Context has no accessor."
            )
        all_programs = context.accessor.get_all()
        if not all_programs:
            raise ValueError(
                f"{self.name}: Cannot select from empty population."
            )

    def execute(
        self, context: Context, result: RolloutResult, **kwargs
    ) -> RolloutResult:
        """
        Execute multi-agent selection.

        Flow:
        1. Read Strategy.forward() assignment from context.metadata
        2. resume -> load that session from snapshot and run it
        3. create -> replace reservation with a new AgentSession
        4. degraded -> delegate to fallback SelectModule
        """
        assignment = get_session_assignment(context, AGENT_SESSION_ASSIGNMENT_KEY)

        try:
            return self._execute_assignment(context, result, assignment)
        except Exception:
            if result.selection is None:
                result.selection = SelectionData(
                    parent_id="",
                    inspiration_ids=[],
                    extra={
                        AGENT_SESSION_ASSIGNMENT_KEY: dict(assignment),
                    },
                )
            else:
                if result.selection.extra is None:
                    result.selection.extra = {}
                result.selection.extra.setdefault(
                    AGENT_SESSION_ASSIGNMENT_KEY, dict(assignment)
                )
            raise

    def _execute_assignment(
        self,
        context: Context,
        result: RolloutResult,
        assignment: Dict[str, Any],
    ) -> RolloutResult:
        """Execute the already-reserved session assignment."""
        action = assignment.get("action", "degraded")
        if action == "resume":
            session_id = assignment.get("session_id")
            sessions = load_sessions(
                context,
                AGENT_SESSIONS_STATE_KEY,
            )
            session = sessions.get(session_id) if session_id else None

            if not session and assignment.get("session"):
                try:
                    session = AgentSession.from_dict(assignment["session"])
                    if self.logger:
                        self.logger.warning(
                            f"No corresponding session in context. "
                            f"Recovered session from assignment snapshot: "
                            f"rollout={result.rollout_id}"
                        )
                except Exception as e:
                    if self.logger:
                        self.logger.warning(
                            f"[{self.name}] Failed to recover assigned session "
                            f"{session_id}: {e}"
                        )

            if not session:
                if self.logger:
                    self.logger.warning(
                        f"[{self.name}] Assigned session {session_id} not found. "
                        "Degrading rollout."
                    )
                return self._degraded_rollout(context, result)

            claim_id = assignment.get("claim_id") or result.rollout_id
            session.claim(claim_id, iteration=context.iteration)

            if self.logger:
                self.logger.info(
                    f"[{self.name}] 🔗 Matched session: "
                    f"{session.session_id} "
                    f"(island={context.island_id}, claim={claim_id})"
                )
            return self._agent_rollout(session, context, result)

        if action == "create":
            session_id = assignment.get("session_id")
            if not session_id:
                if self.logger:
                    self.logger.warning(
                        f"[{self.name}] Create assignment missing session_id. "
                        "Degrading rollout."
                    )
                return self._degraded_rollout(context, result)

            if self.logger:
                self.logger.info(
                    f"[{self.name}] Creating new Agent from placeholder "
                    f"{session_id} on island {context.island_id}."
                )
            return self._create_and_run(context, result, session_id)

        if self.logger:
            self.logger.info(
                f"[{self.name}] ⬇️ Degraded rollout on island {context.island_id}."
            )
        return self._degraded_rollout(context, result)

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """Validate that selection was created."""
        if not result.selection:
            raise ValueError(f"{self.name}: Failed to create selection")
        if not result.selection.parent_id:
            raise ValueError(f"{self.name}: No parent selected")
        if "planner_action" not in result.selection.extra:
            raise ValueError(f"{self.name}: No planner action in selection")
