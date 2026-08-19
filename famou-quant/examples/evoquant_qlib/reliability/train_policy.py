"""Train the meta-policy from accumulated trajectories.

    # stage A only (needs ~50+ decisions to be worth anything)
    /opt/conda/envs/quant/bin/python train_policy.py \
        --trajectory real_trajectory_E1.jsonl --out policies/bc_v1.pt

    # stage A then B, over every round collected so far
    /opt/conda/envs/quant/bin/python train_policy.py \
        --trajectory round_0.jsonl round_1.jsonl --out policies/awr_v2.pt --awr

The resulting checkpoint plugs straight into the search::

    from famou.reliability.rl.policy import LearnedMetaPolicy
    strategy = ReliabilityAwareQuantStrategy(..., meta_policy=LearnedMetaPolicy(path))

or, from the command line, ``run_reliability_real.py --policy <path>``.

Rewards are matured at the end of each search run (``finalize_rewards``), so
``--awr`` works on any trajectory produced after that landed. Older files whose
transitions predate it carry no reward and are skipped by stage B.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from famou.reliability.rl.trainer import (  # noqa: E402
    AdvantageWeightedRegression,
    BehaviorCloning,
    Sample,
    TrainingDataError,
    build_dataset,
)
from famou.reliability.trajectory import TrajectoryStore  # noqa: E402


def load_samples(paths: List[str], gamma: float) -> Tuple[List[Sample], int]:
    """Concatenate the datasets of several trajectory files.

    Each file is turned into samples separately: return-to-go is computed
    within a run, and ``build_dataset`` already splits multi-run files on the
    state_version restart. Concatenating the per-file results keeps that
    boundary intact — pooling the raw transitions first would let the last
    decision of one file collect credit from the first decision of the next.
    """
    samples: List[Sample] = []
    n_transitions = 0
    for path in paths:
        store = TrajectoryStore(path)
        n = len(store.transitions())
        chunk = build_dataset(store, gamma=gamma)
        n_rewarded = sum(1 for s in chunk if s.ret is not None)
        print(f"[data] {Path(path).name}: {n} transitions -> {len(chunk)} samples "
              f"({n_rewarded} reward-labelled)")
        n_transitions += n
        samples.extend(chunk)
    return samples, n_transitions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectory", required=True, nargs="+",
                    help="one or more JSONL trajectory stores")
    ap.add_argument("--out", required=True, help="checkpoint path (.pt)")
    ap.add_argument("--policy-version", default=None)
    ap.add_argument("--awr", action="store_true", help="run stage B after stage A")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--beta", type=float, default=0.5,
                    help="AWR temperature; smaller trusts the advantage more")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples, n_total = load_samples(args.trajectory, args.gamma)
    n_rewarded = sum(1 for s in samples if s.ret is not None)
    print(f"[data] total: {n_total} transitions -> {len(samples)} usable samples "
          f"({n_rewarded} reward-labelled)")
    if len(samples) < n_total:
        print(f"[data] {n_total - len(samples)} skipped: missing features or "
              "written by a different encoding version")

    try:
        bc = BehaviorCloning(seed=args.seed)
        report = bc.fit(samples, epochs=args.epochs)
    except TrainingDataError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    print(f"[stage A] {report.summary()}")

    trainer = bc
    version = args.policy_version or ("awr_v1" if args.awr else "bc_v1")

    if args.awr:
        if n_rewarded == 0:
            print("[error] --awr needs matured rewards, and none of these "
                  "trajectories carry one. Search runs mature their own "
                  "rewards at the end (Strategy.finalize_rewards); files "
                  "written before that was wired have to be re-collected.",
                  file=sys.stderr)
            return 1
        try:
            awr = AdvantageWeightedRegression(seed=args.seed)
            report_b = awr.fit(samples, epochs=args.epochs, beta=args.beta,
                               init_from=bc)
        except TrainingDataError as e:
            print(f"[error] {e}", file=sys.stderr)
            return 1
        print(f"[stage B] {report_b.summary()}  mean_reward={report_b.mean_reward:.4f}")
        trainer = awr

    meta = trainer.save(args.out, policy_version=version,
                        trained_on=len(samples),
                        notes=f"gamma={args.gamma}, epochs={args.epochs}")
    print(f"[saved] {meta.policy_version} ({meta.kind}) -> {meta.artifact_uri}")
    print("        load with LearnedMetaPolicy(path) and pass as meta_policy=")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
