"""Iterated offline loop: search -> mature -> train -> search with the new policy.

This is the closed loop the architecture calls for, with the boundary between
rounds at the process level rather than inside the search:

    round 0   heuristic policy   -> round_0.jsonl -> policy_0.pt
    round 1   policy_0.pt        -> round_1.jsonl -> policy_1.pt  (trained on 0+1)
    round 2   policy_1.pt        -> round_2.jsonl -> policy_2.pt  (trained on 0+1+2)

Each round is a fresh process on purpose. A search run wants a fresh
StateStore — archives, budget ledger and state_version all restart — and the
trajectory files are what carry information forward. Trying to keep one
process alive across rounds would either leak an episode's spent sealed budget
into the next round or require tearing the whole reliability state down in
place, which is exactly the kind of half-reset that corrupts an audit trail.

Usage::

    python iterate_policy.py --rounds 3 --steps 40 --mode formula \
        --workdir policies/E1_formula

Budget note: every round spends its own per-episode sealed budget, so N rounds
means N x --sealed-limit sealed queries against the same episode. That is
defensible for development episodes and is NOT how a reported result should be
produced — freeze once, report once.

Cost: at ~60s per decision, --rounds 3 --steps 40 is roughly 2 hours. Getting
to the ~10^3 transitions where stage B starts to mean anything is closer to
--rounds 5 --steps 200.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from famou.reliability.rl.trainer import build_dataset  # noqa: E402
from famou.reliability.trajectory import TrajectoryStore  # noqa: E402


def _run(cmd: List[str], label: str) -> None:
    print(f"\n{'=' * 70}\n[{label}] {' '.join(cmd)}\n{'=' * 70}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"[{label}] failed with exit code {result.returncode}")


def _rewarded_samples(paths: List[Path], gamma: float) -> int:
    """How many reward-labelled samples exist across the collected rounds.

    Stage B is gated on this rather than on transition count: a transition
    whose reward never matured is invisible to AWR, so counting transitions
    would happily promise training data that ``AdvantageWeightedRegression``
    then refuses.
    """
    total = 0
    for path in paths:
        if not path.exists():
            continue
        samples = build_dataset(TrajectoryStore(str(path)), gamma=gamma)
        total += sum(1 for s in samples if s.ret is not None)
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40,
                    help="search decisions per round")
    ap.add_argument("--workdir", default="policies/iterated",
                    help="holds round_<n>.jsonl and policy_<n>.pt")
    ap.add_argument("--episode", default="E1")
    ap.add_argument("--mode", default="formula",
                    help="formula (~0.06s/candidate) | model | mixed")
    ap.add_argument("--sealed-limit", type=int, default=3,
                    help="sealed queries PER ROUND (see the budget note above)")
    ap.add_argument("--max-batch", type=int, default=2)
    ap.add_argument("--provider-uri", default=None)
    ap.add_argument(
        "--runner", default="real", choices=["real", "stub"],
        help="stub swaps in the no-qlib harness: use it to verify the loop "
             "end-to-end in seconds before committing an overnight job",
    )
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument(
        "--min-awr-samples", type=int, default=200,
        help="stage B only runs once this many reward-labelled samples exist; "
             "below it AWR fits noise and stage A is the honest answer",
    )
    ap.add_argument("--resume", action="store_true",
                    help="skip rounds whose trajectory and checkpoint exist")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    runner = HERE / (
        "run_reliability_real.py" if args.runner == "real"
        else "run_reliability_stub.py"
    )

    print(f"[plan] {args.rounds} rounds x {args.steps} decisions, "
          f"mode={args.mode}, episode={args.episode}, runner={args.runner}")
    print(f"[plan] workdir={workdir}")
    print(f"[plan] sealed budget: {args.sealed_limit}/round "
          f"= {args.sealed_limit * args.rounds} total against {args.episode}")

    previous_checkpoint: Path | None = None
    trajectories: List[Path] = []

    for rnd in range(args.rounds):
        trajectory = workdir / f"round_{rnd}.jsonl"
        checkpoint = workdir / f"policy_{rnd}.pt"
        trajectories.append(trajectory)

        if args.resume and trajectory.exists() and checkpoint.exists():
            print(f"\n[round {rnd}] --resume: reusing {trajectory.name} "
                  f"and {checkpoint.name}")
            previous_checkpoint = checkpoint
            continue

        # --- search ----------------------------------------------------
        search_cmd = [
            python, str(runner),
            "--episode", args.episode,
            "--mode", args.mode,
            "--steps", str(args.steps),
            "--sealed-limit", str(args.sealed_limit),
            "--max-batch", str(args.max_batch),
            "--trajectory", str(trajectory),
        ]
        if args.provider_uri:
            search_cmd += ["--provider-uri", args.provider_uri]
        if previous_checkpoint is not None:
            search_cmd += ["--policy", str(previous_checkpoint)]
        _run(search_cmd, f"round {rnd} search")

        # Rewards were matured by the search run itself; stage B is gated on
        # how many actually landed, across every round collected so far.
        n_rewarded = _rewarded_samples(trajectories, args.gamma)
        use_awr = n_rewarded >= args.min_awr_samples
        print(f"\n[round {rnd}] {n_rewarded} reward-labelled samples "
              f"-> stage {'A+B (BC then AWR)' if use_awr else 'A only (BC)'}")
        if not use_awr:
            print(f"[round {rnd}] stage B needs {args.min_awr_samples}; "
                  "collect more rounds or raise --steps")

        # --- train on everything collected so far ----------------------
        train_cmd = [
            python, str(HERE / "train_policy.py"),
            "--trajectory", *[str(p) for p in trajectories if p.exists()],
            "--out", str(checkpoint),
            "--policy-version", f"{'awr' if use_awr else 'bc'}_r{rnd}",
            "--epochs", str(args.epochs),
            "--gamma", str(args.gamma),
            "--beta", str(args.beta),
        ]
        if use_awr:
            train_cmd.append("--awr")
        _run(train_cmd, f"round {rnd} train")

        previous_checkpoint = checkpoint

    print(f"\n{'=' * 70}")
    print(f"[done] {args.rounds} rounds -> {previous_checkpoint}")
    print(f"[done] run the search with it directly:")
    print(f"       python {runner.name} --mode {args.mode} "
          f"--policy {previous_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
