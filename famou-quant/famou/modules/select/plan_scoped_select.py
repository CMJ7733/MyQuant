"""Selection module that stays within the current active plan."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


def _get_visible_programs(context: Context) -> list:
    if context.island_accessor:
        return list(context.island_accessor)
    return context.get_all_programs()


def _pick_best_program(programs: list) -> Program:
    return max(
        programs,
        key=lambda p: (
            p.combined_score if p.combined_score is not None else float("-inf"),
            p.validity if p.validity is not None else float("-inf"),
            p.iteration,
            p.generation,
        ),
    )


def _resolve_plan_parent(context: Context, requested_parent_id: str, plan_context: dict):
    """Resolve the best visible parent for the active plan, with fallback."""
    visible = _get_visible_programs(context)
    visible_by_id = {p.id: p for p in visible}

    if requested_parent_id and requested_parent_id in visible_by_id:
        program = visible_by_id[requested_parent_id]
        return program, {"requested_parent_id": requested_parent_id,
                         "resolved_parent_id": program.id,
                         "resolution": "requested_parent",
                         "visible_program_ids": [p.id for p in visible]}

    candidate_ids = []
    for cid in [plan_context.get("best_program_id"), *(plan_context.get("plan_program_ids") or [])]:
        if isinstance(cid, str) and cid.strip():
            candidate_ids.append(cid.strip())
    seen = set()
    same_plan = []
    for cid in candidate_ids:
        p = visible_by_id.get(cid)
        if p and p.id not in seen:
            same_plan.append(p)
            seen.add(p.id)
    plan_id = str(plan_context.get("plan_id") or "").strip()
    if plan_id:
        for p in visible:
            if p.id not in seen and str((p.meta or {}).get("plan_id") or "").strip() == plan_id:
                same_plan.append(p)
                seen.add(p.id)

    candidates = same_plan or visible
    if not candidates:
        return None, {"requested_parent_id": requested_parent_id, "resolved_parent_id": None,
                      "resolution": "no_visible_programs", "visible_program_ids": []}
    program = _pick_best_program(candidates)
    resolution = "same_plan_survivor" if same_plan else "island_best_fallback"
    return program, {"requested_parent_id": requested_parent_id,
                     "resolved_parent_id": program.id,
                     "resolution": resolution,
                     "visible_program_ids": [p.id for p in visible]}


class PlanScopedSelect(SelectModule):
    """Select the current plan best as parent and plan-local variants as inspirations."""

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        num_inspirations: int = 3,
        **kwargs,
    ) -> None:
        """Configure plan-local parent and inspiration selection behavior."""
        super().__init__(name)
        self.num_inspirations = num_inspirations

    def validate_input(self, context: Context, result) -> None:
        """Validate that plan context and island-local accessors are available."""
        super().validate_input(context, result)
        plan_context = context.metadata.get("plan_context")
        if not isinstance(plan_context, dict):
            raise ValueError(f"{self.name}: Missing plan_context in Context.metadata")
        if not context.island_accessor:
            raise ValueError(f"{self.name}: Context has no island_accessor")

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """Return the current best program id for the active plan."""
        del population
        plan_context = self._get_plan_context(context)
        requested_parent_id = str(
            plan_context.get("current_parent_id")
            or plan_context.get("best_program_id")
            or plan_context.get("seed_parent_id")
            or ""
        ).strip()
        if not requested_parent_id:
            raise ValueError(f"{self.name}: plan_context has no selectable parent id")
        parent, _ = _resolve_plan_parent(context, requested_parent_id, plan_context)
        if parent is None:
            raise ValueError(f"{self.name}: No visible parent is available for plan-scoped selection")
        return parent.id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> List[str]:
        """Return the top-scoring prior children from the current active plan."""
        del population
        if self.num_inspirations <= 0:
            return []

        plan_context = self._get_plan_context(context)
        candidate_ids = [
            pid
            for pid in plan_context.get("plan_program_ids", [])
            if pid != parent_id
        ]
        programs = [
            context.island_accessor.get_by_id(pid)
            for pid in candidate_ids
        ]
        valid_programs = [p for p in programs if p is not None]
        valid_programs.sort(
            key=lambda prog: prog.combined_score if prog.combined_score is not None else 0.0,
            reverse=True,
        )
        return [prog.id for prog in valid_programs[: self.num_inspirations]]

    def select_state_metadata(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Attach the active plan context to selection metadata for downstream modules."""
        del population, parent_id
        plan_context = self._get_plan_context(context)
        _, parent_resolution = _resolve_plan_parent(
            context,
            str(
                plan_context.get("current_parent_id")
                or plan_context.get("best_program_id")
                or plan_context.get("seed_parent_id")
                or ""
            ).strip(),
            plan_context,
        )
        return {
            "plan_context": plan_context,
            "parent_resolution": parent_resolution,
            "selection_scope": "active_plan_only",
        }

    def _get_plan_context(self, context: Context) -> Dict[str, Any]:
        """Read and validate plan context from strategy-populated metadata."""
        plan_context = context.metadata.get("plan_context")
        if not isinstance(plan_context, dict):
            raise ValueError(f"{self.name}: Invalid plan_context")
        return plan_context
