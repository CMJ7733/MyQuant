"""Reliability search on REAL qlib data (no stub harness).

Same wiring as ``run_reliability_stub.py``, but the evaluator actually trains
LightGBM/MLP models on the frozen CSI300 snapshot. Use it to sanity-check the
loop end-to-end before committing cluster time.

    /opt/conda/envs/quant/bin/python run_reliability_real.py --steps 4

Cost: each F1 candidate is ~35-60s on this machine, F2 with 3 seeds ~2-4x
that. Keep --steps small unless you mean it.

The sealed gate is wired to the SAME runner pointed at sealed_promotion, so a
promotion really does re-train the frozen candidate on data the agent has
never seen. That is the point of the exercise; it is also why this script
spends real sealed queries and should not be run casually against an episode
you intend to report.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # famou_candidate_runtime, qlib_harness
sys.path.insert(0, str(HERE.parents[2]))         # famou package root

from famou.core.accessors import IslandAccessor, PopulationAccessor
from famou.core.data import Context, Program, RolloutResult
from famou.core.state import StateStore
from famou.reliability.archives import CertifiedArchive, SearchArchive
from famou.reliability.budget import BudgetLedger
from famou.reliability.evaluator import FidelityEvaluator
from famou.reliability.promotion import BudgetedGate, SealedGateService
from famou.reliability.strategy import ReliabilityAwareQuantStrategy
from famou.reliability.trajectory import TrajectoryStore
from famou.reliability.types import (
    CandidateMode,
    EvidenceVector,
    Fidelity,
    FrozenSplitManifest,
    MetricDistribution,
    TaskSpec,
)
from qlib_harness import make_run_fn, make_sealed_eval_fn

SPLITS = HERE.parent / "protocol_b" / "splits_v2.yaml"
PROVIDER_URI = "/root/.qlib/qlib_data/cn_data_20260810"

#: Hand-written incumbent: the official Alpha158 LightGBM configuration.
#: It is the certified baseline AND the sealed gate's reference, so a
#: candidate must beat a real model rather than a constant.
BASELINE_CODE = (HERE / "baseline_lightgbm.py").read_text(encoding="utf-8") \
    if (HERE / "baseline_lightgbm.py").exists() else None

#: Formula-mode incumbent: a plain 20-day reversal score. Deliberately a
#: single well-known effect, so "the agent beat the incumbent" means it found
#: something beyond the textbook signal.
BASELINE_FORMULA = (HERE / "baseline_formula.py").read_text(encoding="utf-8") \
    if (HERE / "baseline_formula.py").exists() else "print('no formula baseline')"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", default="E1")
    ap.add_argument(
        "--mode", default="model", choices=[m.value for m in CandidateMode],
        help="formula: deterministic factor combinations (~0.06s each); "
             "model: fitted GBDT/MLP (~40-105s each); "
             "mixed: both compete in one pool",
    )
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--provider-uri", default=PROVIDER_URI)
    ap.add_argument("--sealed-limit", type=int, default=3)
    ap.add_argument("--max-batch", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args()

    mode = CandidateMode(args.mode)
    manifest = FrozenSplitManifest.from_yaml(str(SPLITS), args.episode)
    task_spec = TaskSpec.for_mode(mode, universe=manifest.market)
    print(f"[setup] {manifest.protocol_version}/{manifest.episode_id} "
          f"hash={manifest.compute_hash()[:12]}")
    print(f"[setup] train={manifest.train.start}..{manifest.train.end} "
          f"visible_dev={manifest.visible_dev.start}..{manifest.visible_dev.end}")
    print(f"[setup] mode={mode.value} families={task_spec.allowed_model_families}")
    print(f"[setup] task_spec hash={task_spec.compute_hash()[:12]} "
          f"(protected={task_spec.protected_hash()[:12]})")

    state_store = StateStore()
    ledger = BudgetLedger(state_store)
    ledger.configure_episode(manifest.episode_id,
                             sealed_limit=args.sealed_limit, final_limit=1)
    ledger.set_global_limit("visible_queries", 500)

    run_fn = make_run_fn(provider_uri=args.provider_uri,
                         default_timeout=args.timeout)
    evaluator = FidelityEvaluator(manifest, run_fn, ledger, task_spec=task_spec)

    # The gate re-trains on sealed_promotion. Without an incumbent_code it
    # cannot compute a margin, so promotion stays INCONCLUSIVE — which is the
    # honest behaviour, not a bug.
    sealed_fn = make_sealed_eval_fn(
        provider_uri=args.provider_uri, manifest=manifest,
        incumbent_code=BASELINE_CODE, default_timeout=args.timeout,
    )
    gate = BudgetedGate(SealedGateService(manifest, sealed_fn), ledger)

    trajectory = TrajectoryStore(
        str(HERE / f"real_trajectory_{args.episode}_{mode.value}.jsonl"))
    strategy = ReliabilityAwareQuantStrategy(
        manifest=manifest,
        fidelity_evaluator=evaluator,
        ledger=ledger,
        task_spec=task_spec,
        state_store=state_store,
        trajectory_store=trajectory,
        gate=gate,
        sealed_query_limit=args.sealed_limit,
    )

    # The incumbent must be drawn from the same space as the candidates. A
    # LightGBM baseline in a formula-only run is an incumbent no weighted
    # factor score will beat, so the gate would reject everything and the run
    # would measure nothing.
    if mode is CandidateMode.FORMULA:
        baseline_code = BASELINE_FORMULA
        baseline_family = "formula"
    else:
        baseline_code = BASELINE_CODE or "print('placeholder baseline')"
        baseline_family = "gbdt"

    baseline = Program(id="init_0", code=baseline_code, generation=0, iteration=0)
    baseline.meta["model_family"] = baseline_family
    strategy.seed_certified_baseline(
        "init_0", model_family=baseline_family,
        code_hash=hashlib.sha256(baseline_code.encode()).hexdigest(),
    )

    # MEASURE the incumbent rather than assuming it. A hardcoded reference IC
    # silently decides every promotion: set it too high and nothing is ever
    # promoted, too low and everything is. It also has to be measured on the
    # same visible window the candidates will be scored on.
    print("[setup] measuring incumbent on visible_dev ...", end=" ", flush=True)
    t0 = time.time()
    base_out = run_fn(baseline_code, {
        "train_start": manifest.train.start, "train_end": manifest.train.end,
        "dev_start": manifest.visible_dev.start, "dev_end": manifest.visible_dev.end,
        "embargo_days": manifest.embargo_days,
        "seed_list": [11, 29, 47] if baseline_family != "formula" else [11],
        "label_expression": manifest.label_expression,
        "universe": task_spec.universe,
    })
    base_seeds = base_out.get("per_seed_rank_ic") or [base_out.get("rank_ic", 0.0)]
    print(f"RankIC={base_out.get('rank_ic', 0.0):.4f} ({time.time()-t0:.0f}s)")

    with strategy._barrier.bootstrap():
        strategy._search.add_evidence(EvidenceVector(
            candidate_id="init_0", episode_id=manifest.episode_id,
            eval_id="ev_init_0", fidelity=Fidelity.F2_FULL,
            split_scope="visible_dev", data_contract_hash=manifest.compute_hash(),
            rank_ic=MetricDistribution.from_samples([float(v) for v in base_seeds]),
            regime_stability={
                f"sub{i}": float(v)
                for i, v in enumerate(base_out.get("subperiod_rank_ic") or [])
            },
            complexity={"deterministic": bool(base_out.get("deterministic"))},
            validity=1.0,
        ))

    population = [baseline]
    archive = {"init_0": baseline}
    started = time.time()

    for step in range(args.steps):
        ctx = Context(
            experiment_id=f"real_{args.episode}",
            task_description="EvoQuant reliability search on real qlib data",
            island_id=0,
            accessor=PopulationAccessor(
                island_id=0, island_data={"population": list(population)}),
            island_accessor=IslandAccessor(
                island_id=0, visible_ids=set(archive), archive=dict(archive)),
            iteration=step + 1,
        )
        t0 = time.time()
        batch = strategy.forward_batch(ctx, [], max_batch_size=args.max_batch)
        for i, rollout in enumerate(batch.rollouts):
            result = RolloutResult(rollout_id=f"real_{step}_{i}", iteration=step + 1)
            for module in rollout.modules:
                result = module.execute(ctx, result)
            strategy.on_rollout_complete(result)
            program = result.generated_program
            if program.id not in archive:
                archive[program.id] = program
                population.append(program)

        t = trajectory.transitions()[-1]
        a = t.action
        gate_txt = (f" gate={t.gate_verdict.verdict.value}"
                    f"/{t.gate_verdict.reason_code.value}") if t.gate_verdict else ""
        scores = [
            f"{strategy._search.best_evidence(c).rank_ic.mean:.4f}"
            if strategy._search.best_evidence(c)
            and strategy._search.best_evidence(c).rank_ic else "n/a"
            for c in t.candidate_ids
        ]
        print(f"[step {step}] {time.time()-t0:6.1f}s {a.expert.value:<10} "
              f"fid={int(a.fidelity.value)} n={len(batch.rollouts)} "
              f"v={t.next_state_version} rank_ic=[{', '.join(scores)}]"
              f" target={a.promotion_target_id or '-'}{gate_txt}")

    print(f"\n=== Certified Archive ({time.time()-started:.0f}s total) ===")
    for cid, m in CertifiedArchive(state_store).members().items():
        print(f"  {cid}: {m['model_family']} via {m['admission']}")
    print("\n=== Budget ===")
    for k, v in ledger.remaining(episode_id=manifest.episode_id).items():
        print(f"  {k}: {v}")
    search = SearchArchive(state_store)
    print(f"\n=== {len(search.all_candidates())} candidates, "
          f"{len(trajectory.transitions())} transitions, "
          f"state_version={strategy._barrier.state_version} ===")


if __name__ == "__main__":
    main()
