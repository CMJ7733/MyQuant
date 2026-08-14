"""End-to-end strategy test: observation -> action -> candidate -> evidence ->
promotion -> certified archive -> transition. Uses a stub harness (no qlib)."""

from __future__ import annotations

import pytest

from famou.core.accessors import IslandAccessor, PopulationAccessor
from famou.core.data import Context, Program
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
        rollout = strategy.forward(ctx, rollout_history=[])

        # rollout passes the framework's pipeline validator
        from famou.modules.evaluate import EvaluateModule
        from famou.modules.generate import GenerateModule

        assert any(isinstance(m, GenerateModule) for m in rollout.modules)
        assert any(isinstance(m, EvaluateModule) for m in rollout.modules)

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
        strategy.forward(ctx, rollout_history=[])

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
        strategy.forward(ctx, rollout_history=[])
        # no sealed query spent
        assert ledger.remaining(episode_id="E1")["sealed_queries"] == 3

    def test_trajectory_persisted_to_disk(self, manifest, ledger, state_store, tmp_path):
        traj_path = tmp_path / "trajectory.jsonl"
        strategy = self._build_strategy(manifest, ledger, state_store, tmp_path)
        ctx = make_ctx(manifest)
        strategy.forward(ctx, rollout_history=[])
        assert traj_path.exists()
        content = traj_path.read_text()
        assert '"kind": "decision"' in content
        assert '"kind": "transition"' in content

        # reload round-trips
        reloaded = TrajectoryStore(str(traj_path))
        assert len(reloaded.transitions()) == 1
        assert len(reloaded.decisions()) == 1
