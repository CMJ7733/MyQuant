"""Evaluator for LLM-evolved factor formulas on the frozen qlib snapshot.

Contract with Famou
-------------------
``evaluate(path_user_py) -> dict`` containing ``combined_score`` (required) plus
metric fields, which ``EvaluateModule`` copies into the program's metrics
(``famou/modules/evaluate/base.py``).

Trust boundary
--------------
The candidate is LLM-written code. It runs in a **subprocess** so that an
infinite loop or a segfault cannot take down the driver, and it returns only
the factor matrix — **every metric is computed here, in the parent**. That
split matters: if the subprocess reported its own score, the cheapest way for a
candidate to "win" would be to print a good number without computing anything.
Restricting it to producing factor values means the only way to score well is
to actually predict returns.

Look-ahead
----------
Handing a search process raw OHLCV means a candidate that peeks at future rows
is a matter of time, and such a factor produces a spectacular IC that looks like
a discovery. So causality is tested mechanically rather than reviewed by eye:
the candidate is re-run on panels truncated to day T, and row T of that run must
match row T of the full run. A factor using data from after T cannot satisfy
this. Failing candidates get ``validity=0`` and score 0, with the offending day
named in ``error_info`` so the repair loop has something to work with.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

#: The frozen four-segment protocol. Single source of truth for dates, universe,
#: label and embargo -- transcribing them here would let this file and the
#: protocol drift apart silently, and every score would still look fine.
SPLITS_YAML = HERE.parent / "protocol_b" / "splits_v2.yaml"

#: Which episode to score on. Override with FAMOU_EPISODE=E11.
#:
#: E1 is the DEVELOPMENT episode: its final_test was already burned under v1, so
#: it is the sandbox you may look at as often as you like while the method is
#: still moving (seed, prompt, fitness, iteration count). E2-E11 are EVALUATION
#: episodes whose value is that nobody has looked yet -- run those only once the
#: method is frozen. E11 is additionally the only post-cutoff episode.
EPISODE = os.environ.get("FAMOU_EPISODE", "E1")


def _load_split_cfg(episode: str) -> dict:
    """Read one episode's train/dev windows out of the frozen protocol file."""
    import yaml

    with open(SPLITS_YAML, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    episodes = spec.get("episodes") or {}
    if episode not in episodes:
        raise KeyError(
            f"episode {episode!r} not in {SPLITS_YAML.name}; "
            f"available: {', '.join(sorted(episodes))}"
        )
    entry = episodes[episode]
    return {
        "episode": episode,
        "role": entry.get("role"),
        "train_start": entry["train"][0],
        "train_end": entry["train"][1],
        "dev_start": entry["visible_dev"][0],
        "dev_end": entry["visible_dev"][1],
        "universe": spec["meta"]["market"],
        "embargo_days": int(spec["embargo"]["days"]),
        "label_expression": spec["label"]["expression"],
        "provider_uri": spec["meta"]["provider_uri"],
    }


SPLIT_CFG = _load_split_cfg(EPISODE)

#: Days sampled for the causality probe. Fixed, not random: a candidate that
#: passes must pass the same test its predecessors did, otherwise scores across
#: a run are not comparable. Spread across the dev window.
N_PROBE_DAYS = 6

#: Seconds for the subprocess. Panel load is ~0.1s warm, so this is all candidate.
DEFAULT_TIMEOUT = 600

#: A day needs this many valid (factor, label) pairs to contribute an IC.
MIN_CROSS_SECTION = 20

#: Long/short fraction for the Sharpe portfolio.
DECILE = 0.1

#: Trading days per year, for annualising.
ANNUALISE = 252.0


# ---------------------------------------------------------------------------
# subprocess runner
# ---------------------------------------------------------------------------

_WORKER = r'''
import json, sys
import numpy as np
sys.path.insert(0, {here!r})
from panel import load_panel
import importlib.util

spec = importlib.util.spec_from_file_location("candidate", {program!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if not hasattr(module, "compute_factor"):
    raise SystemExit("candidate defines no compute_factor(panel)")

panel = load_panel({cfg!r}).without_labels()
full = np.asarray(module.compute_factor(panel), dtype=np.float64)

# Re-run on truncated panels: row T of each must match row T of `full`.
probe_rows = {probe_rows!r}
probes = {{}}
for t in probe_rows:
    truncated = np.asarray(module.compute_factor(panel.head(t + 1)), dtype=np.float64)
    if truncated.ndim != 2 or truncated.shape[0] != t + 1:
        probes[str(t)] = None       # wrong shape; the parent reports it
    else:
        probes[str(t)] = truncated[t]

np.savez({out!r}, full=full,
         **{{("probe_" + k): (v if v is not None else np.zeros(0))
            for k, v in probes.items()}},
         probe_missing=np.asarray(
             [k for k, v in probes.items() if v is None], dtype=object))
print("OK")
'''


def _run_candidate(program_path, timeout):
    """Execute the candidate in a subprocess; return (factor, probes, probe_rows).

    Raises RuntimeError with a message intended for the repair loop.
    """
    panel = _load_panel()
    n_days, _ = panel.close.shape
    dev0 = panel.dev_start_row

    # Probe inside the dev window only: that is where scoring happens, and a
    # truncated re-run needs enough history to be meaningful anyway.
    probe_rows = [int(r) for r in np.linspace(dev0, n_days - 1, N_PROBE_DAYS).astype(int)]
    probe_rows = sorted(set(probe_rows))

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "factor.npz")
        script = os.path.join(tmp, "worker.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(_WORKER.format(here=str(HERE), program=str(program_path),
                                   cfg=SPLIT_CFG, probe_rows=probe_rows, out=out))
        try:
            proc = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"candidate exceeded {timeout}s")

        if proc.returncode != 0 or not os.path.exists(out):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise RuntimeError("candidate failed: " + " | ".join(tail[-4:]))

        with np.load(out, allow_pickle=True) as blob:
            factor = blob["full"]
            probes = {int(k): blob[f"probe_{k}"] for k in map(str, probe_rows)}
            missing = {int(k) for k in blob["probe_missing"]}
    return panel, factor, probes, probe_rows, missing


_PANEL = None


def _load_panel():
    """Load the panel once per parent process (subprocesses load from cache)."""
    global _PANEL
    if _PANEL is None:
        from panel import load_panel

        _PANEL = load_panel(SPLIT_CFG)
    return _PANEL


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def _check_shape(factor, panel):
    expected = (panel.n_days, panel.n_instruments)
    if factor.ndim != 2 or factor.shape != expected:
        return f"factor shape {getattr(factor, 'shape', None)}, expected {expected}"
    return None


def _check_causality(factor, probes, probe_rows, missing):
    """Row T recomputed on a panel truncated at T must equal row T of the full run."""
    if missing:
        return f"compute_factor returned the wrong shape on a truncated panel (day rows {sorted(missing)})"

    for t in probe_rows:
        got = probes[t]
        want = factor[t]
        if got.shape != want.shape:
            return f"row {t} changed shape when the panel was truncated at that day"
        # NaN must line up too: turning NaN into a number using later data is
        # exactly the leak being looked for.
        if not np.array_equal(np.isnan(got), np.isnan(want)):
            return (f"row {t} has a different NaN pattern when computed without "
                    f"future data -- the factor looks ahead")
        both = ~np.isnan(got)
        if both.any() and not np.allclose(got[both], want[both], rtol=1e-9, atol=1e-12):
            worst = np.max(np.abs(got[both] - want[both]))
            return (f"row {t} changes by up to {worst:.3g} when future rows are "
                    f"removed -- the factor looks ahead")
    return None


def _check_sanity(factor, panel):
    """Reject factors that cannot express a ranking, before scoring them."""
    dev = factor[panel.dev_start_row:]
    finite = np.isfinite(dev)
    if finite.mean() < 0.01:
        return f"only {finite.mean():.2%} of dev factor values are finite"

    varying = 0
    for row, mask in zip(dev, finite):
        if mask.sum() >= MIN_CROSS_SECTION and np.nanstd(row[mask]) > 0:
            varying += 1
    if varying == 0:
        return "factor is constant across every daily cross-section (no ranking possible)"
    return None


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _rank(values):
    """Average-tie ranks, matching scipy.stats.rankdata('average')."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ties so a factor with plateaus is not penalised by input order.
    sorted_values = values[order]
    i = 0
    while i < len(sorted_values):
        j = i + 1
        while j < len(sorted_values) and sorted_values[j] == sorted_values[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = ranks[order[i:j]].mean()
        i = j
    return ranks


def _daily_series(factor, panel):
    """Per-dev-day (pearson_ic, spearman_ic, long_short_return)."""
    ics, rank_ics, returns = [], [], []
    for t in range(panel.dev_start_row, panel.n_days):
        f = factor[t]
        y = panel.label_norm[t]
        r = panel.fwd_ret_raw[t]
        keep = np.isfinite(f) & np.isfinite(y)
        if keep.sum() < MIN_CROSS_SECTION:
            continue
        fv, yv = f[keep], y[keep]
        if fv.std() == 0 or yv.std() == 0:
            continue

        ics.append(float(np.corrcoef(fv, yv)[0, 1]))
        rank_ics.append(float(np.corrcoef(_rank(fv), _rank(yv))[0, 1]))

        # Sharpe uses the RAW forward return: the z-scored label has no units,
        # so a portfolio return built on it would be meaningless.
        rk = np.isfinite(f) & np.isfinite(r)
        if rk.sum() >= MIN_CROSS_SECTION:
            fr, rr = f[rk], r[rk]
            n_side = max(1, int(len(fr) * DECILE))
            order = np.argsort(fr, kind="stable")
            short_leg = rr[order[:n_side]].mean()
            long_leg = rr[order[-n_side:]].mean()
            returns.append(float(long_leg - short_leg))
    return ics, rank_ics, returns


def _subperiods(rank_ics, n=4):
    if len(rank_ics) < n * 5:
        return []
    edges = np.linspace(0, len(rank_ics), n + 1).astype(int)
    return [float(np.mean(rank_ics[edges[k]:edges[k + 1]])) for k in range(n)]


def _fail(reason, eval_time=0.0):
    return {
        "combined_score": 0.0,
        "validity": 0.0,
        "error_info": reason,
        "eval_time": round(eval_time, 2),
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def evaluate(path_user_py):
    """Score one candidate factor.

    Returns ``combined_score`` = mean daily RankIC on the dev window, with
    ``ic`` / ``icir`` / ``sharpe`` alongside it as metrics.
    """
    started = time.time()
    timeout = int(os.environ.get("FAMOU_EVAL_TIMEOUT", DEFAULT_TIMEOUT))

    try:
        panel, factor, probes, probe_rows, missing = _run_candidate(path_user_py, timeout)
    except RuntimeError as exc:
        return _fail(str(exc), time.time() - started)
    except Exception:
        return _fail("evaluator error: " + traceback.format_exc(limit=3).strip()
                     .replace("\n", " ")[:400], time.time() - started)

    for check in (
        lambda: _check_shape(factor, panel),
        lambda: _check_causality(factor, probes, probe_rows, missing),
        lambda: _check_sanity(factor, panel),
    ):
        reason = check()
        if reason:
            return _fail(reason, time.time() - started)

    ics, rank_ics, returns = _daily_series(factor, panel)
    if not rank_ics:
        return _fail(
            f"no scoreable days (need >= {MIN_CROSS_SECTION} valid stocks and a "
            f"non-constant cross-section)", time.time() - started)

    rank_ic = float(np.mean(rank_ics))
    rank_ic_std = float(np.std(rank_ics, ddof=1)) if len(rank_ics) > 1 else 0.0
    ret = np.asarray(returns, dtype=np.float64)
    ret_std = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0

    return {
        # Mean daily RankIC drives the search. Same definition the reliability
        # line uses, so numbers from the two paths are comparable.
        "combined_score": rank_ic,
        "validity": 1.0,
        "rank_ic": rank_ic,
        "rank_ic_std": rank_ic_std,
        "ic": float(np.mean(ics)) if ics else 0.0,
        "icir": (rank_ic / rank_ic_std) if rank_ic_std > 0 else 0.0,
        "sharpe": (float(ret.mean()) / ret_std * np.sqrt(ANNUALISE)) if ret_std > 0 else 0.0,
        "long_short_return": float(ret.mean() * ANNUALISE) if len(ret) else 0.0,
        "n_ic_days": len(rank_ics),
        "subperiod_rank_ic": _subperiods(rank_ics),
        "coverage": float(np.isfinite(factor[panel.dev_start_row:]).mean()),
        # Provenance: scores from different episodes are NOT comparable (different
        # years, different regime), so every record says which one produced it.
        "episode": SPLIT_CFG["episode"],
        "eval_time": round(time.time() - started, 2),
    }


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(HERE / "seed_factor.py")
    print(f"episode {SPLIT_CFG['episode']} ({SPLIT_CFG['role']})  "
          f"train {SPLIT_CFG['train_start']}..{SPLIT_CFG['train_end']}  "
          f"dev {SPLIT_CFG['dev_start']}..{SPLIT_CFG['dev_end']}", file=sys.stderr)
    print(json.dumps(evaluate(target), indent=2, ensure_ascii=False))
