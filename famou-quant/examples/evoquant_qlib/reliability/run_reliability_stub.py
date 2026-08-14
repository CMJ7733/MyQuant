"""Example wiring: reliability-gated evolution on a frozen Protocol B episode.

This script shows how the pieces compose. It runs with a STUB harness by
default (no qlib needed); wire ``qlib_run_fn`` to the real Alpha158 pipeline
on the training cluster.

    conda run -n quant python run_reliability_stub.py

What it demonstrates:
1. FrozenSplitManifest loaded from the frozen splits_v2.yaml (E1, development)
2. BudgetLedger with per-episode sealed budget
3. ReliabilityAwareQuantStrategy driving explore -> F2 -> gate -> certify
4. Certified Archive + Trajectory Store contents at the end
"""

from __future__ import annotations

import sys
from pathlib import Path

# famou package root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from famou.core.accessors import IslandAccessor, PopulationAccessor
from famou.core.data import Context, Program, RolloutResult
from famou.core.state import StateStore
from famou.reliability.archives import CertifiedArchive, SearchArchive
from famou.reliability.budget import BudgetLedger
from famou.reliability.evaluator import FidelityEvaluator
from famou.reliability.promotion import BudgetedGate, SealedGateService
from famou.reliability.strategy import ReliabilityAwareQuantStrategy
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import FrozenSplitManifest

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "protocol_b" / "splits_v2.yaml"


def stub_visible_run_fn(code: str, split_cfg: dict) -> dict:
    """Stub visible harness. Replace with the qlib Alpha158 pipeline."""
    n_seeds = len(split_cfg.get("seed_list", [1]))
    return {
        "validity": 1.0,
        "rank_ic": 0.058,
        "per_seed_rank_ic": [0.058 + 0.001 * i for i in range(n_seeds)],
    }


def stub_sealed_fn(code: str, manifest, seeds) -> dict:
    """Stub sealed harness. On the cluster this is the ONLY place sealed
    data is loaded; it must live in a separate process/service."""
    return {
        "rank_ic_per_seed": [0.072, 0.073, 0.071],
        "incumbent_rank_ic": 0.05,
        "regime_improvements": {"bull": 0.02, "bear": 0.01, "range": 0.015},
    }


def main() -> None:
    manifest = FrozenSplitManifest.from_yaml(str(SPLITS), "E1")
    print(f"[setup] manifest={manifest.protocol_version}/{manifest.episode_id} "
          f"hash={manifest.compute_hash()[:12]}")

    state_store = StateStore()
    ledger = BudgetLedger(state_store)
    ledger.configure_episode(manifest.episode_id, sealed_limit=5)
    ledger.set_global_limit("visible_queries", 500)

    # Certified baseline (hand-crafted LightGBM Alpha158 reference)
    certified = CertifiedArchive(state_store)
    search = SearchArchive(state_store)
    baseline_code = "print('lightgbm alpha158 baseline')"
    import hashlib

    certified.add_baseline(
        "init_0",
        episode_id=manifest.episode_id,
        model_family="gbdt",
        code_hash=hashlib.sha256(baseline_code.encode()).hexdigest(),
    )
    search.add_candidate(
        "init_0", episode_id=manifest.episode_id, model_family="gbdt",
        code_hash=hashlib.sha256(baseline_code.encode()).hexdigest(),
    )
    # Baseline visible evidence (incumbent reference point)
    from famou.reliability.types import EvidenceVector, Fidelity, MetricDistribution

    search.add_evidence(
        EvidenceVector(
            candidate_id="init_0",
            episode_id=manifest.episode_id,
            eval_id="ev_init_0",
            fidelity=Fidelity.F2_FULL,
            split_scope="visible_dev",
            data_contract_hash=manifest.compute_hash(),
            rank_ic=MetricDistribution.from_samples([0.05, 0.05, 0.05]),
            validity=1.0,
        )
    )

    evaluator = FidelityEvaluator(manifest, stub_visible_run_fn, ledger)
    gate = BudgetedGate(SealedGateService(manifest, stub_sealed_fn), ledger)
    trajectory = TrajectoryStore(str(HERE / "reliability_trajectory.jsonl"))

    strategy = ReliabilityAwareQuantStrategy(
        manifest=manifest,
        fidelity_evaluator=evaluator,
        ledger=ledger,
        state_store=state_store,
        trajectory_store=trajectory,
        gate=gate,
    )

    baseline = Program(id="init_0", code=baseline_code, generation=0, iteration=0)
    # The Evolver rebuilds the Context each iteration from the live population,
    # so this loop does the same. It matters: a promotion decision names an
    # EXISTING candidate, and the strategy can only resolve that name through
    # the context. With a context frozen at iteration 0, promotion silently
    # never fires.
    population = [baseline]
    archive = {"init_0": baseline}

    for step in range(6):
        ctx = Context(
            experiment_id="exp_reliability_stub",
            task_description="EvoQuant reliability stub run",
            island_id=0,
            accessor=PopulationAccessor(
                island_id=0, island_data={"population": list(population)}
            ),
            island_accessor=IslandAccessor(
                island_id=0, visible_ids=set(archive), archive=dict(archive)
            ),
            iteration=step + 1,
        )

        # forward_batch decides and generates; the evaluation happens in the
        # rollout modules (that is what the Evolver runs concurrently), and the
        # batch commits only once every member has reported back.
        batch = strategy.forward_batch(ctx, [], max_batch_size=4)
        results = []
        for i, rollout in enumerate(batch.rollouts):
            result = RolloutResult(rollout_id=f"stub_{step}_{i}", iteration=step + 1)
            for module in rollout.modules:
                result = module.execute(ctx, result)
            results.append(result)
        for result in results:
            strategy.on_rollout_complete(result)
            program = result.generated_program
            if program.id not in archive:  # what _update_experiment does
                archive[program.id] = program
                population.append(program)

        transition = trajectory.transitions()[-1]
        action = transition.action
        gate_info = (
            f" gate={transition.gate_verdict.verdict.value}"
            f"/{transition.gate_verdict.reason_code.value}"
            if transition.gate_verdict
            else ""
        )
        target = action.promotion_target_id or "-"
        print(
            f"[step {step}] expert={action.expert.value} "
            f"family={action.model_family} fidelity={int(action.fidelity.value)}"
            f" rollouts={len(batch.rollouts)} v={transition.next_state_version}"
            f" target={target}{gate_info}"
        )

    print("\n=== Certified Archive ===")
    for cid, m in certified.members().items():
        print(f"  {cid}: family={m['model_family']} admission={m['admission']}")

    print("\n=== Budget ===")
    for k, v in ledger.remaining(episode_id=manifest.episode_id).items():
        print(f"  {k}: {v}")

    # The store is append-only JSONL and reloads what earlier runs wrote, so
    # these counts accumulate across invocations. That is the point — it is
    # the RL trainer's corpus — but delete reliability_trajectory.jsonl if you
    # want a clean demo.
    print(f"\n=== Trajectory (cumulative across runs): "
          f"{len(trajectory.transitions())} transitions, "
          f"{len(trajectory.decisions())} decisions ===")


if __name__ == "__main__":
    main()
