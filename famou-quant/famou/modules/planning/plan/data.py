"""
Data structures for Plan/Todo driven evolution.

Core classes:
- TodoItem: A single optimization task (independent, self-contained)
- Plan: An optimization plan with goal, todos, and Program binding

Design principles:
- Each Todo should be independent (not assuming other Todos completed)
- Plan has weak binding to Program via anchor + lineage
- Supports realignment when parent changes
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TodoStatus(str, Enum):
    """Status of a Todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class TodoAttempt:
    """Record of a single attempt at a Todo."""
    attempt_number: int
    result: str  # "success", "failed", "crash"
    reason: str = ""
    child_id: Optional[str] = None
    reward: Optional[float] = None


@dataclass
class TodoItem:
    """
    A single optimization task.

    Design: Each Todo should be independent and self-contained,
    not assuming other Todos have been completed.

    Attributes:
        id: Unique identifier
        description: What to do (e.g., "改进 local search")
        target_block: Which block to modify (e.g., "D")
        direction: How to do it (e.g., "添加 2-opt* 变体")
        avoid: What to avoid (list of strings)
        status: Current status
        attempts: List of attempts made
        max_attempts: Maximum attempts before skipping
        priority: Priority order (lower = higher priority)
    """
    id: str
    description: str
    target_block: str
    direction: str = ""
    avoid: List[str] = field(default_factory=list)
    status: TodoStatus = TodoStatus.PENDING
    attempts: List[TodoAttempt] = field(default_factory=list)
    max_attempts: int = 3
    priority: int = 0

    @property
    def attempt_count(self) -> int:
        """Number of attempts made."""
        return len(self.attempts)

    @property
    def can_retry(self) -> bool:
        """Whether more attempts are allowed."""
        return self.attempt_count < self.max_attempts

    @property
    def is_active(self) -> bool:
        """Whether this Todo is actionable."""
        return self.status in (TodoStatus.PENDING, TodoStatus.IN_PROGRESS)

    def record_attempt(
        self,
        result: str,
        reason: str = "",
        child_id: Optional[str] = None,
        reward: Optional[float] = None,
    ) -> None:
        """
        Record an attempt at this Todo.

        Args:
            result: "success", "failed", or "crash"
            reason: Reason for failure (if failed)
            child_id: ID of generated child program
            reward: Reward signal from evaluation
        """
        attempt = TodoAttempt(
            attempt_number=self.attempt_count + 1,
            result=result,
            reason=reason,
            child_id=child_id,
            reward=reward,
        )
        self.attempts.append(attempt)

        if result == "success":
            self.status = TodoStatus.COMPLETED
        elif not self.can_retry:
            self.status = TodoStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "description": self.description,
            "target_block": self.target_block,
            "direction": self.direction,
            "avoid": list(self.avoid),
            "status": self.status.value,
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "result": a.result,
                    "reason": a.reason,
                    "child_id": a.child_id,
                    "reward": a.reward,
                }
                for a in self.attempts
            ],
            "max_attempts": self.max_attempts,
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TodoItem":
        """Deserialize from dict."""
        todo = cls(
            id=data["id"],
            description=data["description"],
            target_block=data["target_block"],
            direction=data.get("direction", ""),
            avoid=data.get("avoid", []),
            status=TodoStatus(data.get("status", "pending")),
            max_attempts=data.get("max_attempts", 3),
            priority=data.get("priority", 0),
        )
        for attempt_data in data.get("attempts", []):
            todo.attempts.append(TodoAttempt(
                attempt_number=attempt_data["attempt_number"],
                result=attempt_data["result"],
                reason=attempt_data.get("reason", ""),
                child_id=attempt_data.get("child_id"),
                reward=attempt_data.get("reward"),
            ))
        return todo


@dataclass
class Plan:
    """
    An optimization plan with goal, todos, and Program binding.

    Design: Weak binding to Program via anchor + lineage.
    - anchor_program_id: Current "head" Program the Plan is working on
    - lineage: History of Programs this Plan has executed on

    Attributes:
        id: Unique identifier
        goal: Overall optimization goal
        todos: List of TodoItems
        anchor_program_id: Currently bound Program ID
        lineage: History of Program IDs [p0, p1, p3, ...]
        score: Plan quality score (based on todo completion)
        created_at_iteration: Iteration when Plan was created
    """
    id: str
    goal: str
    todos: List[TodoItem] = field(default_factory=list)
    anchor_program_id: Optional[str] = None
    lineage: List[str] = field(default_factory=list)
    score: float = 0.0
    created_at_iteration: int = 0

    @classmethod
    def create(
        cls,
        goal: str,
        todos: List[TodoItem],
        anchor_program_id: str,
        iteration: int = 0,
    ) -> "Plan":
        """
        Create a new Plan.

        Args:
            goal: Overall optimization goal
            todos: List of TodoItems
            anchor_program_id: Initial anchor Program ID
            iteration: Current iteration

        Returns:
            New Plan instance
        """
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        plan = cls(
            id=plan_id,
            goal=goal,
            todos=todos,
            anchor_program_id=anchor_program_id,
            lineage=[anchor_program_id],
            created_at_iteration=iteration,
        )
        return plan

    @property
    def completed_count(self) -> int:
        """Number of completed Todos."""
        return sum(1 for t in self.todos if t.status == TodoStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Number of failed Todos."""
        return sum(1 for t in self.todos if t.status == TodoStatus.FAILED)

    @property
    def total_count(self) -> int:
        """Total number of Todos."""
        return len(self.todos)

    @property
    def progress(self) -> str:
        """Progress string like '2/4 completed' or '1✓ 1✗ / 4'."""
        completed = self.completed_count
        failed = self.failed_count
        total = self.total_count
        if failed > 0:
            return f"{completed}✓ {failed}✗ / {total}"
        return f"{completed}/{total} completed"

    @property
    def is_complete(self) -> bool:
        """Whether all Todos are done (completed, skipped, or failed)."""
        return all(not t.is_active for t in self.todos)

    def get_current_todo(self) -> Optional[TodoItem]:
        """
        Get the current active Todo.

        Returns the first Todo that is pending or in_progress,
        ordered by priority.

        Returns:
            Current TodoItem or None if all done
        """
        active_todos = [t for t in self.todos if t.is_active]
        if not active_todos:
            return None
        # Sort by priority (lower = higher priority)
        active_todos.sort(key=lambda t: t.priority)
        return active_todos[0]

    def is_in_lineage(self, program_id: str) -> bool:
        """Check if a Program ID is in the lineage."""
        return program_id in self.lineage

    def realign(self, new_program_id: str) -> Dict[str, Any]:
        """
        Realign Plan to a new Program.

        Updates anchor and extends lineage.

        Args:
            new_program_id: New Program ID to align to

        Returns:
            Realign info dict for Rich Action
        """
        old_anchor = self.anchor_program_id
        self.anchor_program_id = new_program_id
        if new_program_id not in self.lineage:
            self.lineage.append(new_program_id)

        return {
            "original_anchor": old_anchor,
            "new_anchor": new_program_id,
            "reason": "代码已切换到新的 parent，请根据当前代码适应",
        }

    def update_anchor_on_success(self, child_program_id: str) -> None:
        """
        Update anchor after successful Todo completion.

        Args:
            child_program_id: ID of the successfully generated child
        """
        self.anchor_program_id = child_program_id
        if child_program_id not in self.lineage:
            self.lineage.append(child_program_id)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "id": self.id,
            "goal": self.goal,
            "todos": [t.to_dict() for t in self.todos],
            "anchor_program_id": self.anchor_program_id,
            "lineage": list(self.lineage),
            "score": self.score,
            "created_at_iteration": self.created_at_iteration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Plan":
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            goal=data["goal"],
            todos=[TodoItem.from_dict(t) for t in data.get("todos", [])],
            anchor_program_id=data.get("anchor_program_id"),
            lineage=data.get("lineage", []),
            score=data.get("score", 0.0),
            created_at_iteration=data.get("created_at_iteration", 0),
        )
