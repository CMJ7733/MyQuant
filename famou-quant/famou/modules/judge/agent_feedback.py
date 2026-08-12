"""
AgentFeedbackJudge - Agent Session 反馈更新模块。

核心职责:
1. 计算 reward (child vs parent score)
2. 更新 Todo 状态 (成功/失败/重试)
3. 更新 Session 状态 (working_program_id, stagnation, 完成)
4. 释放 in_flight
5. Session 结束时生成知识总结 (跨 Agent 共享)

Pipeline 位置: 最后一个 Module，在 Evaluate 和 LLMJudge 之后。
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any, Dict, List, Optional

from famou.core.data import Context, RolloutResult
from famou.core.protocol import Module
from famou.modules.planning.block_detection import parse_blocks_with_spans
from famou.modules.planning.multi_agent_select import (
    AGENT_SESSIONS_STATE_KEY,
    AgentSession,
    save_session,
)
from famou.modules.planning.plan.data import TodoStatus

logger = logging.getLogger("famou")


def extract_error_snippet(error_info: Any, max_len: int = 300) -> str:
    """
    从 error_info 中提取有用的错误信息。

    Handles error_info as JSON string or dict. Extracts compiler "error:" lines,
    strips long file paths to just filenames, and deduplicates.
    """
    raw = error_info
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                raw = parsed.get("details") or parsed.get("message") or raw
        except (json.JSONDecodeError, TypeError):
            pass
    elif isinstance(raw, dict):
        raw = raw.get("details") or raw.get("message") or str(raw)
    raw = str(raw)

    lines = raw.split("\n")
    error_lines = [l for l in lines if ": error:" in l or ": fatal error:" in l]
    if error_lines:
        cleaned, seen = [], set()
        for line in error_lines:
            line = re.sub(r".*/([^/]+:\d+):\d+:", r"\1:", line)
            msg = line.split("error:")[-1].strip()
            if msg not in seen:
                seen.add(msg)
                cleaned.append(line.strip())
        return "\n".join(cleaned)[:max_len]

    return raw[:max_len]


def compute_block_diff(
    parent_code: str,
    child_code: str,
    target_block: str,
    language: str = "python",
) -> str:
    """Compute unified diff between parent and child for a specific block."""
    try:
        if target_block == "EVOLVE":
            lang = language.lower().strip()
            comment = "#" if lang in ("python", "py") else "//"
            pattern = rf"({re.escape(comment)}\s*EVOLVE-BLOCK-START[\s\S]*?{re.escape(comment)}\s*EVOLVE-BLOCK-END)"

            parent_match = re.search(pattern, parent_code)
            child_match = re.search(pattern, child_code)

            if not parent_match or not child_match:
                return ""

            parent_text = parent_match.group(1)
            child_text = child_match.group(1)
        else:
            parent_blocks = parse_blocks_with_spans(parent_code, language)
            child_blocks = parse_blocks_with_spans(child_code, language)

            if target_block not in parent_blocks or target_block not in child_blocks:
                return ""

            parent_text = parent_blocks[target_block]["text"]
            child_text = child_blocks[target_block]["text"]

        if parent_text == child_text:
            return ""

        diff = difflib.unified_diff(
            parent_text.splitlines(keepends=True),
            child_text.splitlines(keepends=True),
            fromfile=f"BLOCK {target_block} (parent)",
            tofile=f"BLOCK {target_block} (child)",
            n=2,
        )
        return "".join(diff).strip()
    except Exception:
        return ""


def summarize_session(
    session_id: str,
    plan_goal: str,
    todos_results: List[Dict[str, Any]],
    checkout_score: float,
    best_score: float,
    status: str,
) -> str:
    """Generate a summary of a completed/stagnated Agent Session."""
    lines = [
        f"Agent {session_id} ({status}):",
        f"  目标: {plan_goal}",
        f"  分数变化: {checkout_score:.4f} → {best_score:.4f} "
        f"({'↑' if best_score > checkout_score else '→'}{abs(best_score - checkout_score):.4f})",
    ]

    for todo in todos_results:
        status_icon = "✅" if todo["status"] == "completed" else "❌"
        line = f"  {status_icon} [{todo['target_block']}] {todo['description']}"
        if todo.get("reason"):
            reason = todo["reason"][:100]
            line += f" — {reason}"
        lines.append(line)

    return "\n".join(lines)


class AgentFeedbackJudge(Module):
    """
    Agent Session 反馈模块。

    Pipeline Role:
    - Reads: result.selection.extra["agent_session_id"] (from MultiAgentSelect)
             result.generated_program (evaluated child)
    - Writes: session state updates via StateStore
              result.generated_program.meta["agent_feedback"]

    降级 rollout 无 agent_session_id，直接 passthrough。
    """

    def __init__(self, name: Optional[str] = None, **config):
        super().__init__(name=name, **config)

    # -------------------------------------------------------------------------
    # Reward Computation
    # -------------------------------------------------------------------------

    def _compute_reward(
        self,
        session: AgentSession,
        result: RolloutResult,
    ) -> float:
        """
        Compute reward: child_score - working_score.

        Positive reward = improvement, negative = regression.
        """
        child = result.generated_program
        if not child:
            return -1.0

        child_score = child.combined_score or 0.0
        working_score = session.best_score

        return child_score - working_score

    # -------------------------------------------------------------------------
    # Session Update
    # -------------------------------------------------------------------------

    def _update_session(
        self,
        session: AgentSession,
        result: RolloutResult,
        reward: float,
        context: Context,
    ) -> None:
        """
        Update session state based on rollout result.

        1. Update current Todo (success/failure)
        2. Update working_program_id chain
        3. Update stagnation counter
        4. Check session completion
        """
        child = result.generated_program
        todo = session.plan.get_current_todo()

        if not todo:
            session.status = "completed"
            session.completed_at_iteration = context.iteration
            return

        # --- Todo update ---
        is_crash = (
            child is None
            or result.status == "failed"
            or (child.validity is not None and child.validity <= 0)
        )

        if is_crash:
            # Crash / invalid: record attempt, keep working on same program
            error_info = ""
            if result.error_message:
                error_info = extract_error_snippet(result.error_message)
            elif child and child.meta.get("error"):
                error_info = extract_error_snippet(child.meta["error"])

            todo.record_attempt(
                result="failed",
                reason=f"Crash/invalid: {error_info}" if error_info else "Crash/invalid",
            )
            session.session_stagnation_counter += 1

            if self.logger:
                self.logger.info(
                    f"[{self.name}] ❌ Todo FAILED (crash): "
                    f"session={session.session_id}, "
                    f"todo=[{todo.target_block}] {todo.description}, "
                    f"attempt={todo.attempt_count}/{todo.max_attempts}, "
                    f"stagnation={session.session_stagnation_counter}/{session.max_stagnation}"
                )

        elif reward > 0:
            # Improvement: advance todo, update working program
            child_score = child.combined_score or 0.0
            todo.record_attempt(result="completed", reason=f"reward={reward:.4f}")
            todo.status = TodoStatus.COMPLETED

            # Update working_program_id chain
            session.working_program_id = child.id
            if child_score > session.best_score:
                session.best_program_id = child.id
                session.best_score = child_score

            # Reset stagnation on success
            session.session_stagnation_counter = 0

            # Compute diff for history
            parent = context.get_program_by_id(result.selection.parent_id)
            diff_text = ""
            if parent and child:
                diff_text = compute_block_diff(
                    parent.code, child.code,
                    todo.target_block, context.language,
                )

            session.add_history(
                "todo_completed",
                todo_id=todo.id,
                target_block=todo.target_block,
                reward=reward,
                child_id=child.id,
                child_score=child_score,
                diff_preview=diff_text[:200] if diff_text else "",
            )

            if self.logger:
                self.logger.info(
                    f"[{self.name}] ✅ Todo COMPLETED: "
                    f"session={session.session_id}, "
                    f"todo=[{todo.target_block}] {todo.description}, "
                    f"reward={reward:.4f}, "
                    f"working_program={child.id} (score={child_score:.4f}), "
                    f"plan_progress={session.plan.progress}"
                )

        else:
            # No improvement: record attempt, retry
            todo.record_attempt(result="failed", reason=f"no improvement, reward={reward:.4f}")
            session.session_stagnation_counter += 1

            if self.logger:
                self.logger.info(
                    f"[{self.name}] ↩️ Todo retry: "
                    f"session={session.session_id}, "
                    f"todo=[{todo.target_block}] {todo.description}, "
                    f"reward={reward:.4f}, "
                    f"attempt={todo.attempt_count}/{todo.max_attempts}, "
                    f"stagnation={session.session_stagnation_counter}/{session.max_stagnation}"
                )

        # --- Check Todo-level stagnation ---
        if todo.attempt_count >= todo.max_attempts and todo.status != TodoStatus.COMPLETED:
            todo.status = TodoStatus.FAILED
            if self.logger:
                self.logger.info(
                    f"[{self.name}] ⏭️ Todo max attempts reached, skipping: "
                    f"session={session.session_id}, "
                    f"todo=[{todo.target_block}] {todo.description}"
                )

        # --- Check Session-level stagnation ---
        if session.session_stagnation_counter >= session.max_stagnation:
            session.status = "stagnated"
            session.completed_at_iteration = context.iteration
            if self.logger:
                self.logger.info(
                    f"[{self.name}] 🛑 Session STAGNATED: "
                    f"session={session.session_id}, "
                    f"counter={session.session_stagnation_counter}/{session.max_stagnation}"
                )
            return

        # --- Check Plan completion ---
        if session.plan.is_complete:
            session.status = "completed"
            session.completed_at_iteration = context.iteration
            if self.logger:
                self.logger.info(
                    f"[{self.name}] 🎉 Session COMPLETED: "
                    f"session={session.session_id}, "
                    f"plan_goal='{session.plan.goal}', "
                    f"checkout_score={session.best_score:.4f}"
                )

    # -------------------------------------------------------------------------
    # Session Summary (Cross-Agent Knowledge Sharing)
    # -------------------------------------------------------------------------

    def _generate_summary(self, session: AgentSession) -> str:
        """Generate knowledge summary for a finished session."""
        todos_results = []
        for todo in session.plan.todos:
            todo_result = {
                "description": todo.description,
                "target_block": todo.target_block,
                "status": todo.status.value if hasattr(todo.status, 'value') else str(todo.status),
                "reason": "",
            }
            if todo.attempts:
                last_attempt = todo.attempts[-1]
                todo_result["reason"] = last_attempt.reason or ""
            todos_results.append(todo_result)

        # Get checkout score from history
        checkout_score = session.best_score
        for h in session.history:
            if h.get("event") == "created":
                checkout_score = h.get("checkout_score", session.best_score)
                break

        return summarize_session(
            session_id=session.session_id,
            plan_goal=session.plan.goal,
            todos_results=todos_results,
            checkout_score=checkout_score,
            best_score=session.best_score,
            status=session.status,
        )

    # -------------------------------------------------------------------------
    # Module Protocol
    # -------------------------------------------------------------------------

    def execute(
        self, context: Context, result: RolloutResult, **kwargs
    ) -> RolloutResult:
        """
        Process rollout result and update Agent Session.

        Degraded rollouts (no agent_session_id) pass through unchanged.
        """
        if not result.selection or not result.selection.extra:
            if self.logger:
                self.logger.info(
                    f"Feedback skipped: no selection extra, rollout={result.rollout_id}, "
                    f"status={result.status}"
                )
            return result

        session_data = result.selection.extra.get("agent_session")
        if not session_data:
            if self.logger:
                self.logger.info(
                    f"Feedback skipped: no agent session, rollout={result.rollout_id}"
                )
            return result

        session_id = result.selection.extra.get("agent_session_id")
        if not session_id:
            if self.logger:
                self.logger.info(
                    f"Feedback skipped: missing agent_session_id, rollout={result.rollout_id}, "
                    f"status={result.status}, parent={result.selection.parent_id}"
                )
            return result

        try:
            session = AgentSession.from_dict(session_data)
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"[{self.name}] Failed to load session {session_id}: {e}"
                )
            return result

        # Compute reward
        reward = self._compute_reward(session, result)

        # Update session state
        self._update_session(session, result, reward, context)

        # Release in_flight
        session.release()

        # Generate summary if session ended
        if not session.is_active and not session.session_summary:
            session.session_summary = self._generate_summary(session)
            session.add_history(
                "session_ended",
                status=session.status,
                best_score=session.best_score,
            )
            if self.logger:
                self.logger.info(
                    f"[{self.name}] 📝 Session summary generated: "
                    f"session={session.session_id}, "
                    f"status={session.status}"
                )

        # Save only this session. Full dict writes can lose concurrent updates.
        save_session(result, session, AGENT_SESSIONS_STATE_KEY)

        # Store feedback info in program meta for debugging
        if result.generated_program:
            result.generated_program.meta["agent_feedback"] = {
                "session_id": session_id,
                "reward": reward,
                "session_status": session.status,
                "session_stagnation": session.session_stagnation_counter,
            }

        return result

    def validate_output(self, context: Context, result: RolloutResult) -> None:
        """No output validation needed (side effects only)."""
        pass
