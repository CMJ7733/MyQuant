"""Explore selection: anchor the current best, reference all branch bests.

Explore aims for architecturally divergent solutions. It is *not* UCB-based:
the parent is always the current island best (so generation starts from a
strong baseline), but the produced child opens a brand-new branch (its nominal
ancestor is the init seed, enforced downstream by the explore generate module
tagging ``meta["operation"]="explore"``).

Inspirations are the best program of every live branch, used by the explore
generate module as *negative references* ("these directions already exist —
produce something architecturally different"). The generate module renders
them with key_features only (no full code), so explore stays a divergence
signal rather than a code-reuse signal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program
from famou.modules.select.base import SelectModule


def _score(program: Program) -> float:
    s = program.combined_score
    return float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else float("-inf")


# ── branch-graph helpers (inlined; branch semantics are pipeline-specific and
# intentionally not shared via a framework util — duplicated where used). An
# explore node roots a new branch; everything else inherits its parent's. ──
def _is_explore(program: Program) -> bool:
    meta = program.meta or {}
    return str(meta.get("operation") or "").strip() == "explore"


def _branch_root_of(program: Program, accessor) -> str:
    """Branch-root id: walk up parent_id to the first explore node (or topmost)."""
    current = program
    visited: set[str] = set()
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
    bests: Dict[str, Program] = {}
    for program in accessor.get_all():
        branch_id = _branch_root_of(program, accessor)
        incumbent = bests.get(branch_id)
        if incumbent is None or _score(program) > _score(incumbent):
            bests[branch_id] = program
    return bests


class ExploreSampleSelect(SelectModule):
    """Parent = current island best; inspirations = all live branch bests."""

    OPERATION = "explore"

    def _accessor(self, context: Context):
        return context.island_accessor or context.accessor

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """Return the current best visible program as the generation anchor."""
        del population
        accessor = self._accessor(context)
        visible = accessor.get_all()
        if not visible:
            raise ValueError(f"{self.name}: no visible programs to select from")
        return max(visible, key=_score).id

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[str]]:
        """Return the best program of every live branch (negative references)."""
        del population
        accessor = self._accessor(context)
        branch_bests = _alive_branch_bests(accessor)
        return [p.id for p in branch_bests.values() if p.id != parent_id]

    def select_state_metadata(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Tag the operation so the strategy can route the result correctly.

        Explore does not participate in UCB, so no visit increment is recorded.
        """
        del population, parent_id
        return {"operation": self.OPERATION, "ucb_visit_increment": None}
