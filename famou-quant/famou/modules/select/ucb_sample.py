"""UCB-based node-selection modules for genealogy/branch strategies.

These selection modules implement the *sampling* half of a population-centric
strategy: given the current island population, pick a parent node via an
epsilon-greedy split (mostly sample the best-scored unvisited node, otherwise
UCB-exploit the visited ones), and pick inspirations by traversing the branch
graph. The *scheduling* half (deciding whether this rollout is an exploit or a
crossover) lives in the strategy and is expressed purely by which rollout —
and hence which SampleSelect subclass — is dispatched.

State contract (read-only via ``context.state`` StateAccessor snapshot):
    - ``ucb_visits``: island-scoped ``Dict[program_id, int]`` visit counts.
    - ``global_iter``: island-scoped int, current global iteration (for the
      exploration-coefficient decay schedule).

The module never writes state directly. It records the intended visit
increment in ``SelectionData.extra["ucb_visit_increment"]``; the strategy
commits the increment to the StateStore in ``on_rollout_complete``.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

from famou.core.data import Context, Program, RolloutResult
from famou.modules.select.base import SelectModule

# ── branch-graph helpers (inlined; branch semantics are pipeline-specific and
# intentionally not shared via a framework util — duplicated where used) ──
# A "branch" is rooted at an explore node: every explore-tagged program starts
# a new branch; descendants from other operations inherit their parent's branch.
# An explore node's parent_id points at the (cross-branch) program it was
# anchored on, so explore nodes act as hard branch boundaries.

_EXPLORE_OPERATION = "explore"


def _is_explore(program: Program) -> bool:
    """Whether a program is an explore node (a branch root)."""
    meta = program.meta or {}
    return str(meta.get("operation") or "").strip() == _EXPLORE_OPERATION


def _branch_root_of(program: Program, accessor) -> str:
    """Branch-root id: walk up parent_id to the first explore node (or topmost)."""
    current = program
    visited: set[str] = set()
    while True:
        if _is_explore(current):
            return current.id
        if current.id in visited:
            return current.id
        visited.add(current.id)
        parent_id = current.parent_id
        if not parent_id:
            return current.id
        parent = accessor.get_by_id(parent_id)
        if parent is None:
            return current.id
        current = parent


def _ancestors_of(program: Program, accessor, limit: int) -> List[Program]:
    """Up to ``limit`` nearest in-branch ancestors (stops at the explore root)."""
    if limit <= 0:
        return []
    if _is_explore(program):
        return []  # explore root has no same-branch ancestors (parent is cross-branch)
    out: List[Program] = []
    visited: set[str] = {program.id}
    current = program
    while len(out) < limit:
        parent_id = current.parent_id
        if not parent_id or parent_id in visited:
            break
        parent = accessor.get_by_id(parent_id)
        if parent is None:
            break
        out.append(parent)
        visited.add(parent.id)
        if _is_explore(parent):
            break  # reached the branch root; never cross into its anchor
        current = parent
    return out


def _children_of(program: Program, accessor) -> List[Program]:
    """Direct same-branch children (explore-root children start new branches)."""
    return [
        p for p in accessor.get_all()
        if p.parent_id == program.id and not _is_explore(p)
    ]


def _branch_score_key(program: Program) -> float:
    """combined_score with -inf fallback for unscored programs."""
    score = program.combined_score
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else float("-inf")


def _pick_high_low_children(program: Program, accessor, limit: int = 2) -> List[Program]:
    """Representative direct children: all if <= limit, else highest + lowest."""
    children = _children_of(program, accessor)
    if not children or limit <= 0:
        return []
    if len(children) <= limit:
        return children
    ordered = sorted(children, key=_branch_score_key)
    if limit == 1:
        return [ordered[-1]]
    picked = [ordered[-1], ordered[0]]
    if limit > 2:
        picked_ids = {p.id for p in picked}
        for p in reversed(ordered):
            if len(picked) >= limit:
                break
            if p.id not in picked_ids:
                picked.append(p)
                picked_ids.add(p.id)
    return picked


def _alive_branch_bests(accessor) -> Dict[str, Program]:
    """Best (highest combined_score) program per live branch."""
    bests: Dict[str, Program] = {}
    for program in accessor.get_all():
        branch_id = _branch_root_of(program, accessor)
        incumbent = bests.get(branch_id)
        if incumbent is None or _branch_score_key(program) > _branch_score_key(incumbent):
            bests[branch_id] = program
    return bests

# Exploration-coefficient linear-decay schedule (see design doc §3.1).
UCB_C_START = 1.0
UCB_C_END = 0.4
UCB_C_DECAY_ITERS = 100


def ucb_coefficient(iteration: int) -> float:
    """Linear decay of the UCB exploration weight from C_START to C_END."""
    if iteration >= UCB_C_DECAY_ITERS:
        return UCB_C_END
    frac = max(0.0, iteration) / UCB_C_DECAY_ITERS
    return UCB_C_START + (UCB_C_END - UCB_C_START) * frac


def _score(program: Program) -> float:
    """combined_score with -inf fallback."""
    s = program.combined_score
    return float(s) if isinstance(s, (int, float)) and not isinstance(s, bool) else float("-inf")


def _is_evaluated(program: Program) -> bool:
    """A node is eligible for UCB only after it has a combined_score."""
    return program.combined_score is not None


class UCBSampleSelect(SelectModule):
    """Base class: pick a parent over alive, evaluated island nodes.

    Selection is an epsilon-greedy split (see :meth:`select_parent`): with a
    fixed probability sample the best-scored never-visited node, otherwise
    UCB-exploit the already-visited ones. Subclasses implement
    :meth:`select_inspirations` to define how the branch graph is traversed for
    inspirations (in-branch for exploit, cross-branch for crossover).
    """

    #: Operation tag recorded in selection metadata; overridden by subclasses.
    OPERATION = "ucb"

    #: Probability of sampling from the never-selected nodes vs. UCB-exploiting
    #: the visited ones. Fixed by design — exploration over visited nodes is
    #: still annealed through :func:`ucb_coefficient`.
    UNVISITED_SAMPLE_PROB = 0.65

    def _accessor(self, context: Context):
        """Prefer the island-scoped accessor; fall back to the global one."""
        return context.island_accessor or context.accessor

    def _alive_evaluated(self, context: Context) -> List[Program]:
        """Programs visible in the current island that have been evaluated."""
        accessor = self._accessor(context)
        return [p for p in accessor.get_all() if _is_evaluated(p)]

    def _visits(self, context: Context) -> Dict[str, int]:
        """Read island-scoped UCB visit counts from the state snapshot."""
        if context.state is None:
            return {}
        visits = context.state.get_island("ucb_visits", default={})
        return visits if isinstance(visits, dict) else {}

    def _iteration(self, context: Context) -> int:
        """Read the global iteration for the decay schedule."""
        if context.state is not None:
            value = context.state.get_island("global_iter", default=None)
            if isinstance(value, int):
                return value
        return int(getattr(context, "iteration", 0) or 0)

    def select_parent(self, context: Context, population: List[Program]) -> str:
        """Select a parent via an epsilon-greedy split over the population.

        With probability :attr:`UNVISITED_SAMPLE_PROB` sample from the
        never-selected nodes — the highest ``combined_score`` among them, ties
        broken uniformly at random. Otherwise (and whenever nothing has been
        visited yet) pick the highest-UCB node among the already-visited ones.

        This replaces the classic UCB1 ``inf`` cold-start: an unvisited node is
        *preferred* most of the time but is no longer guaranteed to be swept,
        and the exploit branch only ever ranks nodes that have a real visit
        count (so the UCB term is always finite).
        """
        del population  # we use the island accessor, not the flat population
        candidates = self._alive_evaluated(context)
        if not candidates:
            # Fall back to any visible program (e.g. only the seed exists yet).
            accessor = self._accessor(context)
            visible = accessor.get_all()
            if not visible:
                raise ValueError(f"{self.name}: no visible programs to select from")
            return max(visible, key=_score).id

        visits = self._visits(context)
        unvisited = [p for p in candidates if int(visits.get(p.id, 0)) == 0]
        visited = [p for p in candidates if int(visits.get(p.id, 0)) > 0]

        # Explore branch: best-scored unvisited node, ties broken at random.
        # Taken with fixed probability, or unconditionally when nothing has
        # been visited yet (the exploit branch would have no candidates).
        if unvisited and (not visited or random.random() < self.UNVISITED_SAMPLE_PROB):
            top_score = max(_score(p) for p in unvisited)
            top_pool = [p for p in unvisited if _score(p) == top_score]
            return random.choice(top_pool).id

        # Exploit branch: highest-UCB node among the visited ones. The
        # exploration weight still decays with iteration via ucb_coefficient.
        coefficient = ucb_coefficient(self._iteration(context))
        total_visits = sum(int(visits.get(p.id, 0)) for p in candidates)
        log_total = math.log(total_visits + 1.0)

        def ucb(program: Program) -> float:
            visit_count = int(visits.get(program.id, 0))
            exploration = coefficient * math.sqrt(log_total / visit_count)
            return _score(program) + exploration

        return max(visited, key=ucb).id

    def select_state_metadata(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[Dict[str, Any]]:
        """Record operation tag, branch ids, and the pending visit increment."""
        del population
        accessor = self._accessor(context)
        parent = accessor.get_by_id(parent_id)
        parent_branch = _branch_root_of(parent, accessor) if parent else None
        return {
            "operation": self.OPERATION,
            "ucb_visit_increment": parent_id,
            "parent_branch_id": parent_branch,
        }


class ExploitSampleSelect(UCBSampleSelect):
    """Exploit: deepen one branch. Inspirations are in-branch genealogy.

    Parent is chosen by UCB; inspirations are the parent's nearest ancestors
    (<=2) plus representative direct children (<=2, highest + lowest scored),
    all within the same branch — showing the branch's evolution trajectory.
    """

    OPERATION = "exploit"

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        max_ancestors: int = 2,
        max_children: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.max_ancestors = max(0, int(max_ancestors))
        self.max_children = max(0, int(max_children))

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[str]]:
        del population
        accessor = self._accessor(context)
        parent = accessor.get_by_id(parent_id)
        if parent is None:
            return []
        ids: List[str] = []
        seen = {parent_id}
        for ancestor in _ancestors_of(parent, accessor, self.max_ancestors):
            if ancestor.id not in seen:
                ids.append(ancestor.id)
                seen.add(ancestor.id)
        for child in _pick_high_low_children(parent, accessor, self.max_children):
            if child.id not in seen:
                ids.append(child.id)
                seen.add(child.id)
        return ids


class CrossoverSampleSelect(UCBSampleSelect):
    """Crossover: fuse across branches. Inspirations come from OTHER branches.

    Parent is chosen by UCB; inspirations are ``num_inspirations`` branch-best
    programs from branches different from the parent's. When fewer than the
    required distinct branches exist, it relaxes and fills from any branch
    (still preferring higher-scored branch bests).
    """

    OPERATION = "crossover"

    def __init__(
        self,
        name: Optional[str] = None,
        *,
        num_inspirations: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(name)
        self.num_inspirations = max(0, int(num_inspirations))

    def select_inspirations(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[List[str]]:
        del population
        if self.num_inspirations <= 0:
            return []
        accessor = self._accessor(context)
        parent = accessor.get_by_id(parent_id)
        if parent is None:
            return []
        parent_branch = _branch_root_of(parent, accessor)
        branch_bests = _alive_branch_bests(accessor)

        # Rank candidate branch-bests by score, highest first.
        ranked = sorted(branch_bests.values(), key=_score, reverse=True)

        # Prefer different-branch candidates first (true cross-branch fusion).
        cross_branch = [
            p for p in ranked
            if _branch_root_of(p, accessor) != parent_branch
            and p.id != parent_id
        ]
        chosen = cross_branch[: self.num_inspirations]

        # Fallback: not enough distinct branches → fill from remaining bests.
        if len(chosen) < self.num_inspirations:
            chosen_ids = {p.id for p in chosen}
            for program in ranked:
                if len(chosen) >= self.num_inspirations:
                    break
                if program.id == parent_id or program.id in chosen_ids:
                    continue
                chosen.append(program)
                chosen_ids.add(program.id)

        return [p.id for p in chosen]

    def select_state_metadata(
        self, context: Context, population: List[Program], parent_id: str
    ) -> Optional[Dict[str, Any]]:
        meta = super().select_state_metadata(context, population, parent_id) or {}
        accessor = self._accessor(context)
        inspiration_ids = self.select_inspirations(context, population, parent_id) or []
        meta["inspiration_branch_ids"] = [
            _branch_root_of(p, accessor)
            for pid in inspiration_ids
            if (p := accessor.get_by_id(pid)) is not None
        ]
        return meta
