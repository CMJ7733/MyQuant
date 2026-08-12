"""
LLM-based Planner V2 for Planning-based evolution.

Uses LLM to decide which block to modify and how, based on:
- Current code context
- History of past actions and outcomes
- Pending (in-flight) actions
- Block-level aggregated statistics
- Parent modification track
- Task description, scores, iteration
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from famou.modules.planning.planners.base import PlannerAction
from famou.modules.planning.planners.stochastic import StochasticPlanner
from famou.prompts import prompt_registry
from famou.utils.trace_utils import build_llm_trace

if TYPE_CHECKING:
    from famou.infrastructure.llm.base import LLMClient

logger = logging.getLogger("famou.planning")


class LLMPlannerV2:
    """
    LLM-based planner that outputs:
        {"op": <op>, "target": <label>}

    Where:
        op in {"modify", "add", "remove", "adjust_params"}
        target in available blocks (arms)

    Features:
    - Maintains history of completed actions for context
    - Tracks pending (in-flight) actions to avoid conflicts
    - Block-level aggregated statistics (success rate, avg reward)
    - Parent modification track (per-parent history)
    - Optional code snippet inclusion for recent modifications
    - Falls back to random selection on LLM failure
    - Normalizes various LLM output formats to standard action

    Example:
        >>> planner = LLMPlannerV2(llm_client=client, arms=["A", "B", "C"])
        >>> action = planner.propose_action(
        ...     parent_code="...", pid="prog_123",
        ...     parent_score=89.35, best_score=91.2,
        ...     iteration=5, task_description="Optimize CVRP..."
        ... )
        >>> # action = {"op": "modify", "target": "A"}
    """

    DEFAULT_ARMS = ["A", "B", "C"]
    ALLOWED_OPS = PlannerAction.ALLOWED_OPS

    def __init__(
        self,
        llm_client: "LLMClient",
        arms: Optional[List[str]] = None,
        history_size: int = 20,
        fallback: Optional[StochasticPlanner] = None,
        allowed_ops: Optional[Sequence[str]] = None,
        track_code_snippets: bool = False,
        max_code_snippet_entries: int = 5,
    ):
        """
        Initialize LLM planner.

        Args:
            llm_client: LLM client for generating actions
            arms: Available block labels (default: ["A", "B", "C"])
            history_size: Max history entries to maintain (default: 20)
            fallback: Fallback planner for failures (default: StochasticPlanner)
            allowed_ops: Allowed operations (default: all)
            track_code_snippets: Whether to store code snippets in history
                (default: False, enable for debugging/analysis)
            max_code_snippet_entries: Max number of recent entries with code
                snippets to include in prompt (default: 5)
        """
        self.llm_client = llm_client
        self.arms = list(arms) if arms else list(self.DEFAULT_ARMS)
        self.history_size = int(history_size)

        self.allowed_ops = tuple(allowed_ops) if allowed_ops else self.ALLOWED_OPS

        # Completed outcomes (bounded history)
        self.history: List[Dict[str, Any]] = []

        # In-flight actions (bounded)
        self.pending_actions: List[Dict[str, Any]] = []

        # Fallback planner (random target)
        self.fallback = fallback or StochasticPlanner(self.arms)

        # Track decision source for debugging
        self.last_decision_source = "llm"

        # Code snippet tracking
        self.track_code_snippets = track_code_snippets
        self.max_code_snippet_entries = max_code_snippet_entries
        self.last_trace: Optional[Dict[str, Any]] = None

    # -------------------------------------------------------------------------
    # Block-level Statistics (computed from history)
    # -------------------------------------------------------------------------

    def _compute_block_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute per-block aggregated statistics from history.

        Returns:
            Dict mapping block label to stats:
            {
                "A": {
                    "attempts": 3,
                    "successes": 1,  # reward > 0
                    "avg_reward": 0.12,
                    "best_reward": 0.35,
                    "ops": {"modify": 2, "adjust_params": 1},
                    "recent_trend": "improving" | "declining" | "stable"
                },
                ...
            }
        """
        stats: Dict[str, Dict[str, Any]] = {}

        for item in self.history:
            target = item.get("target", "")
            if not target:
                continue

            if target not in stats:
                stats[target] = {
                    "attempts": 0,
                    "successes": 0,
                    "total_reward": 0.0,
                    "best_reward": float("-inf"),
                    "ops": {},
                    "rewards": [],  # for trend calculation
                }

            s = stats[target]
            reward = item.get("reward")
            if reward is None:
                reward = 0.0

            s["attempts"] += 1
            s["total_reward"] += reward
            s["rewards"].append(reward)
            if reward > 0:
                s["successes"] += 1
            if reward > s["best_reward"]:
                s["best_reward"] = reward

            op = item.get("op", "modify")
            s["ops"][op] = s["ops"].get(op, 0) + 1

        # Finalize: compute avg and trend
        for target, s in stats.items():
            s["avg_reward"] = s["total_reward"] / s["attempts"] if s["attempts"] > 0 else 0.0
            if s["best_reward"] == float("-inf"):
                s["best_reward"] = 0.0

            # Simple trend: compare last 3 vs previous
            rewards = s["rewards"]
            if len(rewards) >= 4:
                recent = sum(rewards[-3:]) / 3
                earlier = sum(rewards[:-3]) / max(len(rewards) - 3, 1)
                if recent > earlier + 0.01:
                    s["trend"] = "improving"
                elif recent < earlier - 0.01:
                    s["trend"] = "declining"
                else:
                    s["trend"] = "stable"
            else:
                s["trend"] = "insufficient_data"

            del s["total_reward"]
            del s["rewards"]

        return stats

    def _format_block_stats(self, stats: Dict[str, Dict[str, Any]], arms: List[str]) -> str:
        """Format block stats for prompt inclusion."""
        if not stats and not arms:
            return "(no data)"

        lines = []
        for arm in arms:
            if arm in stats:
                s = stats[arm]
                ops_str = ", ".join(f"{k}={v}" for k, v in s["ops"].items())
                lines.append(
                    f"- {arm}: {s['attempts']} attempts, "
                    f"{s['successes']} successes (reward>0), "
                    f"avg_reward={s['avg_reward']:+.4f}, "
                    f"best_reward={s['best_reward']:+.4f}, "
                    f"ops=[{ops_str}], trend={s['trend']}"
                )
            else:
                lines.append(f"- {arm}: 0 attempts (unexplored)")

        return "\n".join(lines) if lines else "(no data)"

    # -------------------------------------------------------------------------
    # Parent Modification Track (computed from history)
    # -------------------------------------------------------------------------

    def _compute_parent_track(self, parent_id: str) -> List[Dict[str, Any]]:
        """
        Get modification history for a specific parent.

        Args:
            parent_id: The parent program ID to look up

        Returns:
            List of history entries for this parent
        """
        if not parent_id:
            return []

        return [
            item for item in self.history
            if item.get("parent_id") == parent_id
        ]

    def _format_parent_track(self, track: List[Dict[str, Any]]) -> str:
        """Format parent track for prompt inclusion."""
        if not track:
            return "(no previous modifications on this parent)"

        lines = []
        for item in track:
            line = (
                f"- {item.get('op', '?')} {item.get('target', '?')} → "
                f"reward={item.get('reward', 'N/A')}, "
                f"score={item.get('combined_score', 'N/A')}, "
                f"validity={item.get('validity', 'N/A')}"
            )
            # Include code snippet if available
            code_snippet = item.get("code_snippet")
            if code_snippet:
                line += f"\n  code: {code_snippet}"
            lines.append(line)

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Prompt Building
    # -------------------------------------------------------------------------

    def _build_prompt(
        self,
        parent_code: str,
        history: List[Dict[str, Any]],
        arms: List[str],
        **context_kwargs,
    ) -> Dict[str, str]:
        """
        Build system and user prompts for LLM.

        Uses templates from PromptRegistry.

        Args:
            parent_code: Current code context
            history: Completed action history
            arms: Available block labels
            **context_kwargs: Additional context:
                - parent_id: Parent program ID
                - parent_score: Parent's current score
                - best_score: Best score in population
                - iteration: Current iteration number
                - task_description: Task description

        Returns:
            Dict with "system" and "user" prompts
        """
        # Build system message from template
        system_message = prompt_registry.get(
            "planning/planner_system.txt",
            allowed_ops=", ".join(self.allowed_ops),
        )

        recent_history = history[-self.history_size:]
        pending_history = self.pending_actions[-self.history_size:]

        # Completed history summary
        history_lines: List[str] = []
        for item in recent_history:
            history_lines.append(
                "- op={op}, target={target}, reward={reward}, "
                "combined_score={combined_score}, validity={validity}, crash={crash}".format(
                    op=item.get("op", ""),
                    target=item.get("target", ""),
                    reward=item.get("reward", ""),
                    combined_score=item.get("combined_score", ""),
                    validity=item.get("validity", ""),
                    crash=item.get("crash", ""),
                )
            )
        history_str = "\n".join(history_lines) if history_lines else "(empty)"

        # Pending summary
        pending_pairs = []
        pending_counts: Dict[str, int] = {}
        for a in pending_history:
            op = str(a.get("op", "")).strip()
            tgt = str(a.get("target", "")).strip()
            if not tgt:
                continue
            k = f"{op}:{tgt}" if op else tgt
            pending_pairs.append(k)
            pending_counts[k] = pending_counts.get(k, 0) + 1
        pending_targets_str = ", ".join(pending_pairs) if pending_pairs else "(empty)"
        pending_counts_str = (
            ", ".join([f"{k}={v}" for k, v in pending_counts.items()])
            if pending_counts
            else "(empty)"
        )

        # Recent completed target counts
        recent_targets = [
            str(item.get("target", "")).strip()
            for item in recent_history
            if str(item.get("target", "")).strip()
        ]
        counts: Dict[str, int] = {}
        for t in recent_targets:
            counts[t] = counts.get(t, 0) + 1
        recent_targets_str = ", ".join(recent_targets) if recent_targets else "(empty)"
        counts_str = (
            ", ".join([f"{k}={v}" for k, v in counts.items()])
            if counts
            else "(empty)"
        )

        # --- New context info ---
        parent_id = context_kwargs.get("parent_id", "")
        parent_score = context_kwargs.get("parent_score", "N/A")
        best_score = context_kwargs.get("best_score", "N/A")
        iteration = context_kwargs.get("iteration", "N/A")
        task_description = context_kwargs.get("task_description", "")

        # Block-level aggregated stats
        block_stats = self._compute_block_stats()
        block_stats_str = self._format_block_stats(block_stats, arms)

        # Parent modification track
        parent_track = self._compute_parent_track(parent_id)
        parent_track_str = self._format_parent_track(parent_track)

        # Build user message from template
        user_message = prompt_registry.get(
            "planning/planner_user.txt",
            arms=", ".join(arms),
            allowed_ops=", ".join(self.allowed_ops),
            parent_code=parent_code or "",
            history_size=self.history_size,
            pending_targets=pending_targets_str,
            pending_counts=pending_counts_str,
            recent_targets=recent_targets_str,
            counts=counts_str,
            history=history_str,
            # New fields
            task_description=task_description,
            parent_id=parent_id,
            parent_score=parent_score,
            best_score=best_score,
            iteration=iteration,
            block_stats=block_stats_str,
            parent_track=parent_track_str,
        )

        return {"system": system_message, "user": user_message}

    # -------------------------------------------------------------------------
    # LLM Call and JSON Extraction
    # -------------------------------------------------------------------------

    def _call_llm(self, prompt: Dict[str, str]):
        """
        Call LLM with prompt.

        Args:
            prompt: Dict with "system" and "user" prompts

        Returns:
            LLM response object
        """
        return self.llm_client.generate(
            prompt=prompt["user"],
            system=prompt["system"],
        )

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract JSON object from LLM response.

        Handles:
        - Clean JSON
        - JSON wrapped in markdown code blocks
        - JSON with extra text before/after

        Args:
            text: LLM response text

        Returns:
            Extracted dict or None if extraction fails
        """
        if not text:
            return None

        # Try direct parse
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding JSON object pattern
        json_match = re.search(r"\{[^{}]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    # -------------------------------------------------------------------------
    # Pending/History Bookkeeping
    # -------------------------------------------------------------------------

    def _record_pending_action(self, pid: str, action: Dict[str, Any]) -> None:
        """
        Record a pending (in-flight) action.

        Args:
            pid: Program ID
            action: Action taken
        """
        if not action:
            return

        # Remove existing entry for this pid
        if pid:
            self.pending_actions = [
                a for a in self.pending_actions if a.get("pid", "") != pid
            ]

        self.pending_actions.append({
            "pid": pid,
            "target": action.get("target", ""),
            "op": action.get("op", ""),
        })

        # Bound pending list
        if len(self.pending_actions) > self.history_size:
            self.pending_actions = self.pending_actions[-self.history_size:]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def propose_action(self, parent_code: str = "", **kwargs) -> Dict[str, Any]:
        """
        Propose next action (which block to modify and how).

        Args:
            parent_code: Current code to be modified
            **kwargs: Additional context
                - pid: Program ID for tracking
                - arms: List of available block labels (overrides default)
                - parent_id: Parent program ID
                - parent_score: Parent's current score
                - best_score: Best score in population
                - iteration: Current iteration number
                - task_description: Task description

        Returns:
            Action dict: {"op": str, "target": str}
        """
        pid = kwargs.get("pid", "")

        # Allow runtime arms override
        arms = kwargs.get("arms") or self.arms
        if arms:
            self.arms = list(arms)
            if hasattr(self.fallback, "arms"):
                self.fallback.arms = list(self.arms)

        # Log context info
        logger.info(
            f"[LLMPlannerV2] propose_action: "
            f"parent_id={kwargs.get('parent_id', 'N/A')}, "
            f"parent_score={kwargs.get('parent_score', 'N/A')}, "
            f"best_score={kwargs.get('best_score', 'N/A')}, "
            f"iteration={kwargs.get('iteration', 'N/A')}, "
            f"arms={self.arms}, history_len={len(self.history)}"
        )

        # Compute and log block stats
        block_stats = self._compute_block_stats()
        if block_stats:
            stats_summary = ", ".join(
                f"{k}({v['attempts']}att/{v['successes']}suc/avg={v['avg_reward']:+.3f})"
                for k, v in block_stats.items()
            )
            logger.info(f"[LLMPlannerV2] Block stats: {stats_summary}")

        # Log parent track
        parent_id = kwargs.get("parent_id", "")
        parent_track = self._compute_parent_track(parent_id)
        if parent_track:
            track_summary = ", ".join(
                f"{t.get('op')} {t.get('target')}→r={t.get('reward', 'N/A')}"
                for t in parent_track
            )
            logger.info(f"[LLMPlannerV2] Parent {parent_id} track: {track_summary}")
        else:
            logger.info(f"[LLMPlannerV2] Parent {parent_id}: no previous modifications")

        # Build prompt (pass all context kwargs through, but remove arms to avoid duplicate)
        build_kwargs = {k: v for k, v in kwargs.items() if k not in ("arms", "pid")}
        prompt = self._build_prompt(parent_code, self.history, self.arms, **build_kwargs)

        # Store prompt for meta saving
        self.last_prompt = prompt
        self.last_response = None
        self.last_trace = None

        # 1) Call LLM
        try:
            llm_response = self._call_llm(prompt)
            llm_resp = llm_response.text
            self.last_response = llm_resp
            self.last_trace = build_llm_trace(
                module_name="LLMPlannerV2",
                system=prompt["system"],
                prompt=prompt["user"],
                response=llm_response,
                request_extra={
                    "arms": self.arms,
                    "allowed_ops": list(self.allowed_ops),
                    "decision_source": "llm",
                },
            )
        except Exception as e:
            logger.warning(f"LLM call failed: {e}, falling back to stochastic")
            self.last_decision_source = "fallback"
            self.last_trace = build_llm_trace(
                module_name="LLMPlannerV2",
                system=prompt["system"],
                prompt=prompt["user"],
                response=None,
                request_extra={
                    "arms": self.arms,
                    "allowed_ops": list(self.allowed_ops),
                    "decision_source": "fallback",
                },
                error=e,
            )
            action = PlannerAction.normalize_action(
                self.fallback.propose_action(), self.arms
            )
            if not action.get("op"):
                action["op"] = "modify"
            self._record_pending_action(pid, action)
            return action

        # 2) Parse JSON
        try:
            data = self._extract_json(llm_resp)
        except Exception as e:
            logger.warning(f"JSON extraction failed: {e}, falling back to stochastic")
            self.last_decision_source = "fallback"
            action = PlannerAction.normalize_action(
                self.fallback.propose_action(), self.arms
            )
            if not action.get("op"):
                action["op"] = "modify"
            self._record_pending_action(pid, action)
            return action

        if not isinstance(data, dict):
            logger.warning("LLM returned non-dict, falling back to stochastic")
            self.last_decision_source = "fallback"
            action = PlannerAction.normalize_action(
                self.fallback.propose_action(), self.arms
            )
            if not action.get("op"):
                action["op"] = "modify"
            self._record_pending_action(pid, action)
            return action

        # 3) Normalize and validate
        op = PlannerAction.normalize_op(data.get("op", ""), data.get("direction", ""))
        target = PlannerAction.match_target(data.get("target", ""), self.arms)

        if not target:
            logger.warning(
                f"Invalid target '{data.get('target')}' not in {self.arms}, "
                "falling back to stochastic"
            )
            self.last_decision_source = "fallback"
            action = PlannerAction.normalize_action(
                self.fallback.propose_action(), self.arms
            )
            action["op"] = PlannerAction.normalize_op(
                action.get("op", ""), action.get("direction", "")
            )
            action["target"] = (
                PlannerAction.match_target(action.get("target", ""), self.arms)
                or action.get("target", "")
            )
            if not action.get("target"):
                action["target"] = self.arms[0] if self.arms else ""
            self._record_pending_action(pid, action)
            return action

        action = {"op": op, "target": target}
        self.last_decision_source = "llm"
        logger.info(f"[LLMPlannerV2] Decision: {action} (source=llm)")
        self._record_pending_action(pid, action)
        return action

    def update(self, batch: List[Dict[str, Any]]) -> None:
        """
        Update planner state with feedback from completed actions.

        Args:
            batch: List of feedback items, each containing:
                - pid: Program ID
                - action: {"op": str, "target": str}
                - parent_id: Parent program ID (optional)
                - parent_score: Parent's score (optional)
                - outcome: {
                    "reward": float,
                    "combined_score": float (or "child_score"),
                    "validity": float,
                    "crash": bool
                  }
        """
        for item in batch:
            pid = item.get("pid", "")

            # Remove from pending
            if pid:
                self.pending_actions = [
                    a for a in self.pending_actions if a.get("pid", "") != pid
                ]

            action = item.get("action") or {}
            outcome = item.get("outcome") or {}

            # Normalize action for consistent logging
            norm_action = PlannerAction.normalize_action(action, self.arms)

            entry = {
                "pid": pid,
                "op": norm_action.get("op", ""),
                "target": norm_action.get("target", ""),
                "reward": outcome.get("reward"),
                "combined_score": outcome.get("combined_score") or outcome.get("child_score"),
                "validity": outcome.get("validity"),
                "crash": outcome.get("crash"),
                # Enriched fields
                "parent_id": item.get("parent_id"),
                "parent_score": item.get("parent_score"),
            }

            # Optionally store code snippet (only recent N entries)
            if self.track_code_snippets:
                code_snippet = item.get("code_snippet")
                if code_snippet:
                    entry["code_snippet"] = code_snippet

            self.history.append(entry)
            logger.info(
                f"[LLMPlannerV2] History updated: "
                f"op={entry['op']}, target={entry['target']}, "
                f"reward={entry.get('reward')}, "
                f"parent_id={entry.get('parent_id')}, "
                f"parent_score={entry.get('parent_score')}, "
                f"history_len={len(self.history)}"
            )

        # Bound history
        if len(self.history) > self.history_size:
            self.history = self.history[-self.history_size:]

    def get_state(self) -> Dict[str, Any]:
        """
        Get planner state for persistence.

        Returns:
            State dict containing history and pending actions
        """
        return {
            "history": list(self.history),
            "pending_actions": list(self.pending_actions),
            "arms": list(self.arms),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore planner state from persistence.

        Args:
            state: State dict from get_state()
        """
        self.history = list(state.get("history", []))
        self.pending_actions = list(state.get("pending_actions", []))
        if "arms" in state:
            self.arms = list(state["arms"])
            if hasattr(self.fallback, "arms"):
                self.fallback.arms = list(self.arms)
