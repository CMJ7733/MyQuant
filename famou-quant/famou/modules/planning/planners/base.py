"""
Base Planner Protocol for Planning-based evolution.

Planners decide:
- Which code block to modify (target)
- How to modify it (op: modify/adjust_params)

All planners implement:
- propose_action(): Returns {"op": str, "target": str}
- update(batch): Updates internal state with feedback
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class BasePlanner(Protocol):
    """
    Protocol for Planning-based evolution planners.

    Planners are stateful components that decide which code block to evolve
    and how to evolve it. They maintain history of past actions and outcomes
    to make informed decisions.

    Action Format:
        {
            "op": str,      # Operation: "modify", "adjust_params"
            "target": str   # Block label: "A", "B", "C", etc.
        }

    Update Batch Format:
        [
            {
                "pid": str,                    # Program ID
                "action": {"op": str, "target": str},
                "outcome": {
                    "reward": float,           # Reward signal
                    "combined_score": float,   # Combined score
                    "validity": float,         # Validity score 0.0-1.0
                    "crash": bool,             # Whether execution crashed
                }
            },
            ...
        ]

    Allowed Operations:
        - modify: Refactor/optimize/cleanup code in the target block
        - adjust_params: Only change hyperparameters/magic numbers

    Example:
        >>> planner = LLMPlannerV2(llm_client=client, arms=["A", "B", "C"])
        >>> action = planner.propose_action(parent_code="...", pid="prog_123")
        >>> # action = {"op": "modify", "target": "A"}
        >>>
        >>> # After evaluation, update planner with feedback
        >>> planner.update([{
        ...     "pid": "prog_123",
        ...     "action": action,
        ...     "outcome": {"reward": 0.5, "validity": 1.0, "crash": False}
        ... }])
    """

    def propose_action(self, parent_code: str = "", **kwargs) -> Dict[str, Any]:
        """
        Propose next action (which block to modify and how).

        Args:
            parent_code: Current code to be modified
            **kwargs: Additional context (pid, arms, etc.)
                - pid: Program ID for tracking
                - arms: List of available block labels

        Returns:
            Action dict: {"op": str, "target": str}
        """
        ...

    def update(self, batch: List[Dict[str, Any]]) -> None:
        """
        Update planner state with feedback from completed actions.

        Args:
            batch: List of feedback items, each containing:
                - pid: Program ID
                - action: The action that was taken
                - outcome: Results (reward, validity, crash, etc.)
        """
        ...


class PlannerAction:
    """
    Utility class for action normalization and validation.

    Provides static methods for:
    - Normalizing operation names (handle aliases)
    - Validating action format
    - Matching targets to available arms
    """

    ALLOWED_OPS = ("modify", "adjust_params")

    # Operation aliases (multi-language support)
    OP_ALIASES = {
        # adjust params
        "adjust_params": "adjust_params",
        "adjust_parameters": "adjust_params",
        "adjust": "adjust_params",
        "tune": "adjust_params",
        "tune_params": "adjust_params",
        # modify / refactor (default for unknown ops)
        "modify": "modify",
        "refactor": "modify",
        "cleanup": "modify",
        "optimize": "modify",
        "mutate": "modify",
        "mutate_code": "modify",
        "add": "modify",      # map to modify
        "remove": "modify",   # map to modify
    }

    @classmethod
    def normalize_op(cls, op: Any, direction: Any = None) -> str:
        """
        Normalize operation name to one of allowed ops.

        Handles:
        - Case variations (MODIFY -> modify)
        - Aliases (refactor -> modify)
        - Legacy format (op=mutate, direction=add -> add)

        Args:
            op: Operation name (may be aliased)
            direction: Legacy direction field (optional)

        Returns:
            Normalized operation name
        """
        op_s = str(op or "").strip().lower().replace("-", "_").replace(" ", "_")
        dir_s = str(direction or "").strip().lower().replace("-", "_").replace(" ", "_")

        # Legacy pattern: op=mutate/mutate_code + direction=<kind>
        if op_s in ("mutate", "mutate_code") and dir_s:
            op_s = dir_s

        # Apply aliases
        op_s = cls.OP_ALIASES.get(op_s, op_s)

        # Fallback to modify if unknown
        if op_s not in cls.ALLOWED_OPS:
            op_s = "modify"

        return op_s

    @classmethod
    def match_target(cls, target: str, arms: List[str]) -> str:
        """
        Case-insensitive match target to one of arms.

        Args:
            target: Target block label (may have wrong case)
            arms: Available block labels

        Returns:
            Matched arm label, or empty string if no match
        """
        t = str(target or "").strip()
        if not t:
            return ""
        # Strip common prefixes like "BLOCK " or "block_"
        import re
        t_clean = re.sub(r'^(?:block\s*|block_)', '', t, flags=re.IGNORECASE).strip()
        for a in arms:
            if a.lower() == t_clean.lower():
                return a
        for a in arms:
            if a.lower() == t.lower():
                return a
        return ""

    @classmethod
    def normalize_action(cls, action: Any, arms: List[str] = None) -> Dict[str, Any]:
        """
        Normalize action dict to standard format.

        Args:
            action: Raw action dict from LLM or other source
            arms: Available arms for target matching (optional)

        Returns:
            Normalized action: {"op": str, "target": str}
        """
        if not isinstance(action, dict):
            return {"op": "modify", "target": ""}

        op = cls.normalize_op(action.get("op"), action.get("direction"))
        target = str(action.get("target", "")).strip()

        if arms:
            target = cls.match_target(target, arms) or target

        return {"op": op, "target": target}

    @classmethod
    def validate_action(cls, action: Dict[str, Any], arms: List[str]) -> bool:
        """
        Validate that action has valid op and target.

        Args:
            action: Action to validate
            arms: Available arms

        Returns:
            True if action is valid
        """
        if not isinstance(action, dict):
            return False

        op = action.get("op", "")
        target = action.get("target", "")

        return (
            op in cls.ALLOWED_OPS
            and target in arms
        )
