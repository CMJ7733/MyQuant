"""End-to-end strategy test: observation -> action -> candidate -> evidence ->
promotion -> certified archive -> transition. Uses a stub harness (no qlib)."""

from __future__ import annotations

import hashlib
import uuid

import pytest

from famou.core.accessors import IslandAccessor, PopulationAccessor
from famou.core.data import Context, Program, RolloutResult
from famou.reliability.archives import CertifiedArchive, SearchArchive
from famou.reliability.evaluator import FidelityEvaluator
from famou.reliability.promotion import BudgetedGate, SealedGateService
from famou.reliability.strategy import (
    HeuristicMetaPolicy,
    ReliabilityAwareQuantStrategy,
)
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import Fidelity, GateVerdictKind


def stub_run_fn(code: str, split_cfg: dict) -> dict:
    """Deterministic visible-split harness: better candidates get higher IC."""
    n_seeds = len(split_cfg.get("seed_list", [1]))
    base = 0.06  # above the incumbent (0.05) so promotion can trigger
    return {
        "validity": 1.0,
        "rank_ic": base,
        "per_seed_rank_ic": [base + 0.0005 * i for i in range(n_seeds)],
    }


def stub_sealed_fn(code: str, manifest, seeds) -> dict:
    return {
        "rank_ic_per_seed": [0.075, 0.076, 0.074],  # clear improvement over 0.05
        "incumbent_rank_ic": 0.05,
        "regime_improvements": {"bull": 0.02, "bear": 0.01, "range": 0.015},
    }


def make_ctx(manifest, programs=None) -> Context:
    programs = programs or []
    pop = {"population": programs}
    archive = {p.id: p for p in programs}
    visible = {p.id for p in programs}
    return Context(
        experiment_id="exp_test",
        task_description="test",
        island_id=0,
        accessor=PopulationAccessor(island_id=0, island_data=pop),
        island_accessor=IslandAccessor(island_id=0, visible_ids=visible, archive=archive),
        iteration=1,
    )


def drive_batch(strategy, ctx, *, max_batch_size=4, fail_ids=()):
    """Run one decision end-to-end the way the Evolver would.

    forward_batch() only decides and generates; the expensive evaluation lives
    in the rollout modules and the results are reported back through the
    completion hooks. Tests must do the same or nothing is ever committed.

    Returns the executed RolloutResults.
    """
    batch = strategy.forward_batch(ctx, [], max_batch_size=max_batch_size)
    results = []
    for i, rollout in enumerate(batch.rollouts):
        result = RolloutResult(rollout_id=f"r_{i}_{uuid.uuid4().hex[:6]}", iteration=1)
        # Modules run in sequence exactly as RolloutEngine would run them.
        for module in rollout.modules:
            result = module.execute(ctx, result)
        results.append(result)

    for result in results:
        if result.generated_program.id in fail_ids:
            strategy.on_rollout_failed(result)
        else:
            strategy.on_rollout_complete(result)
    return results


class ScriptedPolicy(HeuristicMetaPolicy):
    """Policy that goes straight to a promotion request with full F2 seeds."""

    policy_version = "scripted_v0"

    def act(self, obs):
        from famou.reliability.types import ExpertKind, Fidelity, StructuredAction

        parent_ids = []
        if obs.certified_candidates:
            parent_ids = [obs.certified_candidates[0].candidate_id]
        return StructuredAction(
            expert=ExpertKind.EXPLORE,
            parent_ids=parent_ids,
            model_family="gbdt",
            fidelity=Fidelity.F2_FULL,
            seed_list=[11, 29, 47],
            promotion_requested=True,
            rationale="scripted: straight to promotion",
        )


class TestReliabilityStrategy:
    def _build_strategy(self, manifest, ledger, state_store, tmp_path, policy=None):
        evaluator = FidelityEvaluator(manifest, stub_run_fn, ledger)
        gate = BudgetedGate(
            SealedGateService(manifest, stub_sealed_fn), ledger
        )
        return ReliabilityAwareQuantStrategy(
            manifest=manifest,
            fidelity_evaluator=evaluator,
            ledger=ledger,
            state_store=state_store,
            trajectory_store=TrajectoryStore(str(tmp_path / "trajectory.jsonl")),
            gate=gate,
            meta_policy=policy,
        )

    def test_forward_produces_valid_rollout(self, manifest, ledger, state_store, tmp_path):
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        ctx = make_ctx(manifest)
        batch = strategy.forward_batch(ctx, [], max_batch_size=4)

        # rollouts pass the framework's pipeline validator
        from famou.modules.evaluate import EvaluateModule
        from famou.modules.generate import GenerateModule

        rollout = batch.rollouts[0]
        assert any(isinstance(m, GenerateModule) for m in rollout.modules)
        assert any(isinstance(m, EvaluateModule) for m in rollout.modules)

        # forward_batch decides but does not evaluate: nothing committed yet
        assert strategy._barrier.state_version == 0
        assert SearchArchive(state_store).all_candidates() == {}

        # drive the rollouts the way the Evolver would
        for i, r in enumerate(batch.rollouts):
            result = RolloutResult(rollout_id=f"r{i}", iteration=1)
            for module in r.modules:
                result = module.execute(ctx, result)
            strategy.on_rollout_complete(result)

        # a decision + transition + evidence were recorded
        assert len(strategy.trajectory.decisions()) == 1
        assert len(strategy.trajectory.transitions()) == 1

        # candidate landed in the search archive with evidence
        search = SearchArchive(state_store)
        candidates = search.all_candidates()
        assert len(candidates) == 1
        cid = next(iter(candidates))
        evidence = search.get_evidence(cid)
        assert evidence and evidence[0].fidelity == Fidelity.F1_CHEAP

    def test_promotion_flow_end_to_end(self, manifest, ledger, state_store, tmp_path):
        # Seed a certified baseline so the incumbent reference exists
        certified = CertifiedArchive(state_store)
        certified.add_baseline(
            "init_0", episode_id="E1", model_family="gbdt", code_hash="h0"
        )
        search = SearchArchive(state_store)
        search.add_candidate("init_0", episode_id="E1", model_family="gbdt", code_hash="h0")
        from famou.reliability.types import EvidenceVector, MetricDistribution

        search.add_evidence(
            EvidenceVector(
                candidate_id="init_0",
                episode_id="E1",
                eval_id="ev_init",
                fidelity=Fidelity.F2_FULL,
                split_scope="visible_dev",
                data_contract_hash=manifest.compute_hash(),
                rank_ic=MetricDistribution.from_samples([0.05, 0.05, 0.05]),
                validity=1.0,
            )
        )

        strategy = self._build_strategy(
            manifest, ledger, state_store, tmp_path, policy=ScriptedPolicy()
        )
        baseline = Program(id="init_0", code="print('baseline')", generation=0, iteration=0)
        ctx = make_ctx(manifest, programs=[baseline])
        drive_batch(strategy, ctx)

        # sealed query was spent (per-episode budget)
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 2

        # candidate promoted into the certified archive
        certified_members = certified.members()
        promoted = [
            m for m in certified_members.values() if m["admission"] == "gate"
        ]
        assert len(promoted) == 1

        # transition carries the gate verdict
        transition = strategy.trajectory.transitions()[0]
        assert transition.gate_verdict is not None
        assert transition.gate_verdict.verdict == GateVerdictKind.PROMOTE

    def test_no_gate_without_policy_approval(self, manifest, ledger, state_store, tmp_path):
        # Default heuristic policy: first action is cheap explore without promotion
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        ctx = make_ctx(manifest)
        drive_batch(strategy, ctx)
        # no sealed query spent
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 3

    def test_trajectory_persisted_to_disk(self, manifest, ledger, state_store, tmp_path):
        traj_path = tmp_path / "trajectory.jsonl"
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        ctx = make_ctx(manifest)
        drive_batch(strategy, ctx)
        assert traj_path.exists()
        content = traj_path.read_text()
        assert '"kind": "decision"' in content
        assert '"kind": "transition"' in content

        # reload round-trips
        reloaded = TrajectoryStore(str(traj_path))
        assert len(reloaded.transitions()) == 1
        assert len(reloaded.decisions()) == 1

    def test_delayed_reward_survives_reload(self, manifest, ledger, state_store, tmp_path):
        """B3 regression: reward_update records were ignored on load, so every
        delayed reward was silently reset to None on resume."""
        traj_path = tmp_path / "trajectory.jsonl"
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        drive_batch(strategy, make_ctx(manifest))

        transition = strategy.trajectory.transitions()[0]
        assert strategy.trajectory.set_reward(transition.transition_id, 0.77)

        reloaded = TrajectoryStore(str(traj_path))
        assert reloaded.transitions()[0].reward == pytest.approx(0.77)

    def test_promotion_targets_existing_candidate(
        self, manifest, ledger, state_store, tmp_path
    ):
        """C regression: the heuristic used to gate a fresh mutation of the
        stable candidate rather than the stable candidate itself, and re-gated
        it every iteration until the sealed budget was gone."""
        certified = CertifiedArchive(state_store)
        certified.add_baseline(
            "init_0", episode_id="E1", model_family="gbdt", code_hash="h0"
        )
        search = SearchArchive(state_store)
        stable_code = "print('stable')"
        search.add_candidate(
            "cand_stable",
            episode_id="E1",
            model_family="gbdt",
            code_hash=hashlib.sha256(stable_code.encode()).hexdigest(),
        )
        from famou.reliability.types import EvidenceVector, MetricDistribution

        search.add_evidence(
            EvidenceVector(
                candidate_id="cand_stable",
                episode_id="E1",
                eval_id="ev_stable",
                fidelity=Fidelity.F2_FULL,
                split_scope="visible_dev",
                data_contract_hash=manifest.compute_hash(),
                rank_ic=MetricDistribution.from_samples([0.07, 0.071, 0.069]),
                validity=1.0,
            )
        )

        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        stable = Program(
            id="cand_stable", code=stable_code, generation=1, iteration=1
        )
        stable.meta["model_family"] = "gbdt"
        ctx = make_ctx(manifest, programs=[stable])

        before = set(search.all_candidates())
        drive_batch(strategy, ctx)

        # the sealed query was spent on cand_stable itself, not on a mutation
        assert set(search.all_candidates()) == before
        assert search.gate_attempts("cand_stable") == 1
        assert strategy.trajectory.transitions()[0].candidate_ids == ["cand_stable"]
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 2
        # PROMOTE verdict + intact protocol -> it really did get certified
        assert certified.is_certified("cand_stable")

        # a second iteration must NOT re-gate the same candidate
        drive_batch(strategy, ctx)
        assert search.gate_attempts("cand_stable") == 1
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 2

    def test_forward_commits_through_the_barrier(
        self, manifest, ledger, state_store, tmp_path
    ):
        """C1 at the strategy level: each forward() observes the last committed
        version, and its results land in exactly one version bump."""
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        ctx = make_ctx(manifest)

        assert strategy._barrier is None  # wired lazily
        drive_batch(strategy, ctx)
        assert strategy._barrier.state_version == 1

        t1 = strategy.trajectory.transitions()[0]
        assert t1.state_version == 0          # decided against the empty state
        assert t1.next_state_version == 1     # committed into v1
        assert t1.stale is False

        drive_batch(strategy, ctx)
        t2 = strategy.trajectory.transitions()[1]
        assert t2.state_version == 1 and t2.next_state_version == 2
        assert strategy.dump_state()["state_version"] == 2

    def test_archives_reject_writes_outside_the_barrier(
        self, manifest, ledger, state_store, tmp_path
    ):
        """The strategy's own archive handles are guarded, so no future code
        path can make a result visible without a version bump."""
        from famou.reliability.archives import ArchiveWriteOutsideBarrier

        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        drive_batch(strategy, make_ctx(manifest))

        with pytest.raises(ArchiveWriteOutsideBarrier):
            strategy._search.add_candidate(
                "sneaky", episode_id="E1", model_family="gbdt", code_hash="h"
            )

    def test_budget_exhaustion_commits_an_empty_batch(
        self, manifest, ledger, state_store, tmp_path
    ):
        """A rollout that produced nothing still has to balance its batch,
        otherwise the barrier would block forever waiting for it."""
        ledger.set_global_limit("visible_queries", 0)
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        drive_batch(strategy, make_ctx(manifest))

        search = SearchArchive(state_store)
        assert search.all_candidates() == {}       # nothing registered
        assert strategy._barrier.state_version == 1  # but the batch committed
        assert len(strategy.trajectory.transitions()) == 1


    def test_missing_promotion_target_does_not_gate_a_new_candidate(
        self, manifest, ledger, state_store, tmp_path
    ):
        """If the named target is not in the context we fall back to
        generation — but the fallback must drop promotion_requested. Gating a
        brand-new candidate on its first evaluation is the sealed-budget drain
        that promotion_target_id exists to prevent."""
        strategy = self._build_strategy(
            manifest, ledger, state_store, tmp_path, policy=ScriptedPolicy()
        )
        # ScriptedPolicy asks to promote, but points at a candidate that the
        # context knows nothing about.
        original_act = strategy.meta_policy.act

        def act(obs):
            action = original_act(obs)
            return action.model_copy(update={"promotion_target_id": "ghost"})

        strategy.meta_policy.act = act

        before = ledger.remaining(episode_id="E1")["sealed_queries"]
        drive_batch(strategy, make_ctx(manifest))

        assert ledger.remaining(episode_id="E1")["sealed_queries"] == before
        assert strategy.trajectory.transitions()[0].gate_verdict is None
        # the recorded decision reflects what actually ran, not the intent
        decision = list(strategy.trajectory.decisions().values())[0]
        assert decision.structured_action.promotion_requested is False
        assert decision.structured_action.promotion_target_id is None


class BatchPolicy(HeuristicMetaPolicy):
    """Explore with a fan-out of `n` candidates per decision."""

    policy_version = "batch_v0"

    def __init__(self, n: int):
        super().__init__()
        self._n = n

    def act(self, obs):
        from famou.reliability.types import ExpertKind, Fidelity, StructuredAction

        return StructuredAction(
            expert=ExpertKind.EXPLORE,
            model_family="gbdt",
            fidelity=Fidelity.F1_CHEAP,
            seed_list=[11],
            batch_size=self._n,
            rationale="fan out",
        )


class TestBatchedDecisions:
    """One decision -> N concurrent rollouts -> one commit (invariant C1)."""

    def _build(self, manifest, ledger, state_store, tmp_path, n):
        return ReliabilityAwareQuantStrategy(
            manifest=manifest,
            fidelity_evaluator=FidelityEvaluator(manifest, stub_run_fn, ledger),
            ledger=ledger,
            state_store=state_store,
            trajectory_store=TrajectoryStore(str(tmp_path / "t.jsonl")),
            meta_policy=BatchPolicy(n),
        )

    def test_batch_fans_out_and_commits_once(
        self, manifest, ledger, state_store, tmp_path
    ):
        strategy = self._build(manifest, ledger, state_store, tmp_path, 4)
        ctx = make_ctx(manifest)
        results = drive_batch(strategy, ctx, max_batch_size=4)

        assert len(results) == 4
        search = SearchArchive(state_store)
        assert len(search.all_candidates()) == 4
        # four rollouts, ONE decision, ONE transition, ONE version bump
        assert len(strategy.trajectory.decisions()) == 1
        assert len(strategy.trajectory.transitions()) == 1
        assert strategy._barrier.state_version == 1
        assert len(strategy.trajectory.transitions()[0].candidate_ids) == 4

    def test_partial_batch_stays_invisible(
        self, manifest, ledger, state_store, tmp_path
    ):
        """The whole point of the barrier: 3 of 4 finished changes nothing."""
        strategy = self._build(manifest, ledger, state_store, tmp_path, 4)
        ctx = make_ctx(manifest)
        batch = strategy.forward_batch(ctx, [], max_batch_size=4)

        search = SearchArchive(state_store)
        for i, rollout in enumerate(batch.rollouts[:3]):
            result = RolloutResult(rollout_id=f"r{i}", iteration=1)
            for module in rollout.modules:
                result = module.execute(ctx, result)
            strategy.on_rollout_complete(result)
            assert search.all_candidates() == {}
            assert strategy._barrier.state_version == 0

        result = RolloutResult(rollout_id="r3", iteration=1)
        for module in batch.rollouts[3].modules:
            result = module.execute(ctx, result)
        strategy.on_rollout_complete(result)
        assert len(search.all_candidates()) == 4
        assert strategy._barrier.state_version == 1

    def test_failed_member_does_not_hang_the_batch(
        self, manifest, ledger, state_store, tmp_path
    ):
        """A failed rollout must still be accounted for, or the batch never
        reaches `expected` and the barrier waits forever."""
        strategy = self._build(manifest, ledger, state_store, tmp_path, 3)
        ctx = make_ctx(manifest)
        batch = strategy.forward_batch(ctx, [], max_batch_size=3)

        results = []
        for i, rollout in enumerate(batch.rollouts):
            result = RolloutResult(rollout_id=f"r{i}", iteration=1)
            for module in rollout.modules:
                result = module.execute(ctx, result)
            results.append(result)

        strategy.on_rollout_failed(results[0])
        strategy.on_rollout_complete(results[1])
        strategy.on_rollout_complete(results[2])

        assert strategy._barrier.state_version == 1     # committed, not stuck
        search = SearchArchive(state_store)
        assert len(search.all_candidates()) == 2        # failed one not registered

    def test_batch_size_capped_by_dispatch_capacity(
        self, manifest, ledger, state_store, tmp_path
    ):
        """The strategy wants 8 but the Evolver can only run 2 right now."""
        strategy = self._build(manifest, ledger, state_store, tmp_path, 8)
        batch = strategy.forward_batch(make_ctx(manifest), [], max_batch_size=2)
        assert len(batch.rollouts) == 2

    def test_evaluation_deferred_to_worker(
        self, manifest, ledger, state_store, tmp_path
    ):
        """forward_batch must not evaluate: that is what makes the fan-out
        parallel instead of serialising the batch on the controller thread."""
        ledger.set_global_limit("visible_queries", 100)
        strategy = self._build(manifest, ledger, state_store, tmp_path, 4)
        strategy.forward_batch(make_ctx(manifest), [], max_batch_size=4)
        assert ledger.remaining()["visible_queries"] == 100  # nothing charged yet

    def test_evaluator_shared_across_worker_copies(
        self, manifest, ledger, state_store, tmp_path
    ):
        """The ThreadPool backend deep-copies each rollout. If that copied the
        evaluator it would copy the BudgetLedger with it, and every worker
        would charge a private ledger — budget limits would stop applying."""
        import copy

        ledger.set_global_limit("visible_queries", 100)
        strategy = self._build(manifest, ledger, state_store, tmp_path, 1)
        batch = strategy.forward_batch(make_ctx(manifest), [], max_batch_size=1)
        rollout = batch.rollouts[0]
        clone = copy.deepcopy(rollout)

        from famou.reliability.strategy import _ReliabilityEvaluate

        original = next(m for m in rollout.modules if isinstance(m, _ReliabilityEvaluate))
        copied = next(m for m in clone.modules if isinstance(m, _ReliabilityEvaluate))
        assert copied._evaluator is original._evaluator
        assert copied._evaluator._ledger is ledger

    def test_incomplete_batch_committed_on_finalize(
        self, manifest, ledger, state_store, tmp_path
    ):
        """A stop mid-batch must not silently discard the work already paid for."""
        strategy = self._build(manifest, ledger, state_store, tmp_path, 4)
        ctx = make_ctx(manifest)
        batch = strategy.forward_batch(ctx, [], max_batch_size=4)

        result = RolloutResult(rollout_id="r0", iteration=1)
        for module in batch.rollouts[0].modules:
            result = module.execute(ctx, result)
        strategy.on_rollout_complete(result)
        assert strategy._barrier.state_version == 0  # still waiting on 3

        strategy.finalize_experiment({})
        assert strategy._barrier.state_version == 1
        assert len(SearchArchive(state_store).all_candidates()) == 1
