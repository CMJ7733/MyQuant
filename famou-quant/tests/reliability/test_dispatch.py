"""Evolver batch dispatch: one decision -> N tasks, without breaking the
one-rollout-per-decision path every other strategy relies on.

These exercise ``_plan_rollout_tasks`` directly rather than booting a whole
experiment: the goal is the slot accounting and the max_batch_size contract,
not the backend.
"""

from __future__ import annotations

import pytest

from famou.controller.evolver import Evolver, _IslandTracker
from famou.core.data import Program, Rollout, WorkBatch
from famou.reliability.strategy import (
    _FixedSelect,
    _PreGeneratedGenerate,
    _ReplayEvaluate,
)


def make_rollout(name="r") -> Rollout:
    program = Program(id=f"p_{name}", code="print(1)", generation=0, iteration=0)
    return Rollout(
        modules=[
            _FixedSelect("parent"),
            _PreGeneratedGenerate(program),
            _ReplayEvaluate(combined_score=0.0, validity=1.0, metrics={}),
        ],
        name=name,
    )


class _StubStrategy:
    """Records what max_batch_size it was offered and answers with `n`."""

    def __init__(self, n: int, *, overshoot: bool = False):
        self.n = n
        self.overshoot = overshoot
        self.offered: list[int] = []

    def forward_batch(self, ctx, history, max_batch_size=1):
        self.offered.append(max_batch_size)
        count = self.n if not self.overshoot else max_batch_size + 1
        return WorkBatch(
            rollouts=[make_rollout(f"r{i}") for i in range(count)],
            concurrency_hint=count,
            barrier=True,
        )


class _StubEvolver:
    """Minimal host for _plan_rollout_tasks (no experiment, no backend)."""

    _plan_rollout_tasks = Evolver._plan_rollout_tasks

    class _Config:
        def __init__(self, num_islands, max_workers, max_iterations=100):
            self.num_islands = num_islands
            self.max_workers = max_workers
            self.max_iterations = max_iterations

    class _Experiment:
        id = "exp_test"

    def __init__(self, strategy, *, num_islands=1, max_workers=8):
        self.current_strategy = strategy
        self.config = self._Config(num_islands, max_workers)
        self.experiment = self._Experiment()
        self.engine = None
        self.logger = None

    # collaborators _plan_rollout_tasks needs, stubbed
    def _create_context(self, iteration, island_id):
        return None

    def _get_island_rollout_history(self, island_id):
        return []

    def _prepare_rollout_for_execution(self, rollout):
        return rollout


def make_tracker(max_iterations=100, dispatched=0) -> _IslandTracker:
    tracker = _IslandTracker(
        num_islands=1, start_iteration=1, max_iterations=max_iterations
    )
    for _ in range(dispatched):
        tracker.get_next()
    return tracker


class TestBatchDispatch:
    def test_batch_produces_one_task_per_rollout(self):
        strategy = _StubStrategy(4)
        evolver = _StubEvolver(strategy)
        tracker = make_tracker()
        tracker.get_next()  # caller consumed the first slot

        tasks = evolver._plan_rollout_tasks(0, 1, tracker)
        assert len(tasks) == 4
        # each task gets its own iteration slot, none reused
        iterations = [t.iteration for t in tasks]
        assert iterations == [1, 2, 3, 4]
        assert len(set(t.task_id for t in tasks)) == 4

    def test_single_rollout_batch_matches_old_behaviour(self):
        strategy = _StubStrategy(1)
        evolver = _StubEvolver(strategy)
        tracker = make_tracker()
        tracker.get_next()

        tasks = evolver._plan_rollout_tasks(0, 1, tracker)
        assert len(tasks) == 1
        assert tasks[0].iteration == 1
        # no extra slots consumed
        assert tracker.total_iterations_dispatched == 1

    def test_offer_bounded_by_worker_pool(self):
        strategy = _StubStrategy(2)
        evolver = _StubEvolver(strategy, max_workers=3)
        tracker = make_tracker()
        tracker.get_next()

        evolver._plan_rollout_tasks(0, 1, tracker)
        assert strategy.offered == [3]

    def test_offer_bounded_by_remaining_iterations(self):
        """Near the end of the experiment there is no room for a wide batch."""
        strategy = _StubStrategy(2)
        evolver = _StubEvolver(strategy, max_workers=16)
        tracker = make_tracker(max_iterations=10, dispatched=8)
        tracker.get_next()  # 9th slot taken by the caller; 1 left

        evolver._plan_rollout_tasks(0, 9, tracker)
        assert strategy.offered == [2]  # the caller's slot + 1 remaining

    def test_multi_island_falls_back_to_single(self):
        """Slots round-robin across islands, but a batch belongs to one
        island's context — so batching is off when there are several."""
        strategy = _StubStrategy(4)
        evolver = _StubEvolver(strategy, num_islands=3)
        tracker = make_tracker()
        tracker.get_next()

        with pytest.raises(RuntimeError, match="honour max_batch_size"):
            evolver._plan_rollout_tasks(0, 1, tracker)
        assert strategy.offered == [1]

    def test_overshooting_strategy_is_an_error(self):
        """Surplus rollouts would have no slot to run in, and a strategy
        waiting for them would hang. Fail loudly instead."""
        strategy = _StubStrategy(0, overshoot=True)
        evolver = _StubEvolver(strategy, max_workers=2)
        tracker = make_tracker()
        tracker.get_next()

        with pytest.raises(RuntimeError, match="honour max_batch_size"):
            evolver._plan_rollout_tasks(0, 1, tracker)
