"""Command-line entry point: ``cogalpha <command>``.

Five commands, each a thin shell over the library:

============  ==================================================================
search        run the working stream of Figure 1 and archive it
compose       combine a run's candidates into one prediction (the Table 1 reading)
evaluate      score one or more alpha ``.py`` files on their own
report        re-render a report, or derive an ablation, from an archived run
inspect       print the agent hierarchy, guidance modes, or a resolved config
============  ==================================================================

The point of having this at all is reproducibility: a run should be a command
someone else can re-issue, not a script that lived in ``/tmp``.  Configuration is
therefore layered rather than baked in — a tracked run config plus a git-ignored
credentials file plus command-line overrides — and every run records the resolved
configuration next to its output.

``search`` refuses to start on a window that cannot support selection.  Discovering
after six hours that the evaluation window held 12 trading days is a failure mode
worth paying a few seconds to avoid.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Importing pandas/qlib eagerly costs ~2 s, which is most of the runtime of
# ``inspect``; the heavy imports therefore live inside the command functions.


EPILOG = """\
examples:
  # offline smoke test: synthetic panel, deterministic mock backend, no network
  cogalpha search --config configs/synthetic.yaml --out runs/

  # real data, real endpoint (credentials stay in the git-ignored llm.yaml)
  cogalpha search --config configs/paper_csi300.yaml --llm-config configs/llm.yaml

  # the reading Table 1 reports: 20 candidates combined by LightGBM
  cogalpha compose --run runs/20260809-101530-csi300 --model lightgbm --split test

  # score hand-written factors without running a search
  cogalpha evaluate seeds/*.py --split valid

  # what the seven-level hierarchy contains
  cogalpha inspect hierarchy

  # watch a run in a browser, live (or replay a finished one)
  cogalpha monitor --run runs/ --port 8080
"""


# --------------------------------------------------------------------- plumbing


def _add_config_args(parser: argparse.ArgumentParser, with_llm: bool = True) -> None:
    """Arguments shared by every command that builds a config."""
    parser.add_argument(
        "--config",
        metavar="YAML",
        help="run configuration (tracked; see configs/paper_csi300.yaml)",
    )
    if with_llm:
        parser.add_argument(
            "--llm-config",
            metavar="YAML",
            help=(
                "credentials overlay, layered over --config "
                "(git-ignored; see configs/llm.yaml.example)"
            ),
        )
        parser.add_argument("--llm-provider", choices=["openai", "mock"])
        parser.add_argument("--llm-model")
        parser.add_argument("--llm-api-base")
        parser.add_argument(
            "--max-llm-calls",
            type=int,
            metavar="N",
            help="hard ceiling on LLM calls; the run stops cleanly at the limit",
        )
    parser.add_argument(
        "--provider",
        choices=["qlib", "synthetic", "noise", "csv"],
        help="data provider override",
    )
    parser.add_argument("--market", help="index / universe, e.g. csi300")
    parser.add_argument("--horizon", type=int, help="prediction horizon in trading days")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override any config field, e.g. --set evolution.generations=8",
    )


def _build_config(args: argparse.Namespace):
    """Resolve the config from files, typed flags and --set overrides."""
    from cogalpha.config import merge_configs

    overrides: Dict[str, Dict[str, Any]] = {}

    def put(section: str, key: str, value: Any) -> None:
        if value is not None:
            overrides.setdefault(section, {})[key] = value

    put("data", "provider", getattr(args, "provider", None))
    put("data", "market", getattr(args, "market", None))
    put("data", "horizon", getattr(args, "horizon", None))
    put("llm", "provider", getattr(args, "llm_provider", None))
    put("llm", "model", getattr(args, "llm_model", None))
    put("llm", "api_base", getattr(args, "llm_api_base", None))
    put("evolution", "max_llm_calls", getattr(args, "max_llm_calls", None))

    for item in getattr(args, "set", []) or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(f"--set expects SECTION.KEY=VALUE, got '{item}'")
        path, raw = item.split("=", 1)
        section, key = path.split(".", 1)
        overrides.setdefault(section, {})[key] = _parse_scalar(raw)

    return merge_configs(
        getattr(args, "config", None),
        getattr(args, "llm_config", None),
        **overrides,
    )


def _parse_scalar(raw: str) -> Any:
    """Interpret a --set value as YAML, so types survive the shell."""
    import yaml

    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _load_panel(cfg) -> Tuple[Any, Dict[str, Any]]:
    from cogalpha.data import load_panel

    started = time.time()
    panel = load_panel(cfg.data)
    description = panel.describe()
    description["load_seconds"] = round(time.time() - started, 1)
    return panel, description


def _window_for(cfg, split: str) -> Tuple[str, str]:
    windows = {"train": cfg.data.train, "valid": cfg.data.valid, "test": cfg.data.test}
    if split not in windows:
        raise SystemExit(f"unknown split '{split}'; use train, valid or test")
    return windows[split]


def _echo(message: str) -> None:
    print(message, flush=True)


def _expand_alpha_files(patterns: Sequence[str]) -> List[Path]:
    """Resolve file arguments, expanding globs the shell may have left literal.

    ``Path.glob`` rejects absolute patterns, so an absolute path containing a
    wildcard — common when pointing at an archive under ``/tmp`` — has to be split
    into a concrete anchor plus a relative pattern.  Quoted globs arrive
    unexpanded, which is the usual case when the pattern matches nothing in the
    current directory.
    """
    out: List[Path] = []
    seen: set = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if any(ch in pattern for ch in "*?["):
            if candidate.is_absolute():
                parts = candidate.parts
                idx = next(
                    (i for i, p in enumerate(parts) if any(c in p for c in "*?[")),
                    len(parts),
                )
                anchor = Path(*parts[:idx]) if idx else Path(candidate.anchor)
                matches = sorted(anchor.glob(str(Path(*parts[idx:]))))
            else:
                matches = sorted(Path().glob(pattern))
        else:
            matches = [candidate] if candidate.exists() else []

        for path in matches:
            if path.suffix == ".py" and path.is_file() and path not in seen:
                seen.add(path)
                out.append(path)
    return out


# ------------------------------------------------------------------- preflight


def preflight(cfg, panel, split: str, strict: bool = True) -> List[str]:
    """Check the search can produce meaningful numbers before it starts.

    Returns the list of problems found.  With ``strict`` the caller exits on any.

    Each check corresponds to a way a run can burn hours and return something
    worthless:

    * **too few trading days** — selection on a short window promotes noise.  On
      real CSI300 data a raw-price control (a pure size proxy, no information)
      scored RankIC 0.0825 on a 243-day window, above every genuine alpha tested.
    * **thin cross-sections** — IC on 8 names is almost surely ±1 and swamps the
      series.
    * **warm-up history** — a 120-day rolling factor evaluated on a panel that
      starts at the window's first day is NaN for six months and gets rejected for
      coverage rather than for being wrong.
    * **label observability** — the last ``horizon + 1`` days of the window have no
      forward return, so a window shorter than that scores nothing at all.
    """
    import numpy as np
    import pandas as pd

    problems: List[str] = []
    start, end = _window_for(cfg, split)
    dates = panel.dates
    in_window = dates[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    n_days = len(in_window)

    embargo = cfg.data.horizon + 1
    scored_days = max(n_days - embargo, 0)

    _echo(f"  window {split}: {start} .. {end}  ({n_days} trading days in panel)")

    if n_days == 0:
        problems.append(
            f"the {split} window {start}..{end} contains no panel dates; "
            f"the panel covers {dates.min().date() if len(dates) else '?'} .. "
            f"{dates.max().date() if len(dates) else '?'}"
        )
        return problems

    if scored_days < 250:
        severity = "too short" if scored_days < 250 else "short"
        problems.append(
            f"only {scored_days} scoreable days in the {split} window "
            f"({n_days} days minus a {embargo}-day label horizon). "
            "Selection on a window this short cannot separate a real factor from a "
            "size proxy -- use a multi-year window (train is 2189 days) or set "
            "data.fit_split accordingly."
        )

    # Warm-up: how much history precedes the window.
    before = dates[dates < pd.Timestamp(start)]
    _echo(f"  warm-up history before the window: {len(before)} days")
    if len(before) < 120 and split != "train":
        problems.append(
            f"only {len(before)} days of history precede the {split} window; "
            "rolling factors up to 120 days will be mostly NaN at the start. "
            "Load the panel from an earlier date rather than slicing it to the window."
        )

    # Cross-section width over the window, on the tradable universe.
    mask = panel.universe_mask()
    mask = mask.loc[(mask.index >= pd.Timestamp(start)) & (mask.index <= pd.Timestamp(end))]
    per_day = mask.sum(axis=1)
    median_names = int(np.median(per_day)) if len(per_day) else 0
    thin = int((per_day < cfg.fitness.min_names_per_day).sum())
    _echo(
        f"  cross-section: median {median_names} names/day, "
        f"{thin} days below the {cfg.fitness.min_names_per_day}-name minimum"
    )
    if median_names < cfg.fitness.min_names_per_day * 2:
        problems.append(
            f"median cross-section is {median_names} names, close to the "
            f"{cfg.fitness.min_names_per_day}-name floor; IC will be dominated by "
            "small-sample noise"
        )
    if len(per_day) and thin / len(per_day) > 0.2:
        problems.append(
            f"{thin / len(per_day):.0%} of days have fewer than "
            f"{cfg.fitness.min_names_per_day} tradable names"
        )

    # An LLM budget that cannot cover one generation is a silent truncation.
    ev = cfg.evolution
    if ev.max_llm_calls is not None:
        per_gen = ev.parent_pool_size * ev.children_multiplier * 3
        if ev.max_llm_calls < per_gen:
            problems.append(
                f"evolution.max_llm_calls={ev.max_llm_calls} is below the ~{per_gen} "
                "calls one generation needs; the run would stop inside its first "
                "generation"
            )

    if problems:
        _echo("")
        for problem in problems:
            _echo(f"  PROBLEM: {problem}")
        if strict:
            _echo(
                "\nRefusing to start. Fix the above, or pass --no-preflight to "
                "proceed anyway."
            )
    return problems


# ---------------------------------------------------------------------- search


def cmd_search(args: argparse.Namespace) -> int:
    """Run the search and archive it. Returns a shell exit code.

    Order of operations is deliberate: validate the credential, load the panel, run
    preflight, and only then create the archive -- so a misconfigured invocation
    fails in a second and leaves no empty run directory behind.
    """
    from cogalpha.archive import RunArchive
    from cogalpha.data.panel import slice_panel
    from cogalpha.evolution import CogAlphaSearch
    from cogalpha.fitness import FitnessEvaluator
    from cogalpha.llm import CallRecorder, build_client

    cfg = _build_config(args)
    split = args.split or cfg.data.fit_split

    _echo(f"cogalpha search  |  llm: {cfg.llm.resolve_secrets().describe()}")

    # Validate the credential before loading the panel: a missing key should fail
    # in a second, not after a minute of data loading and a created archive dir.
    build_client(cfg.llm, recorder=None, max_calls=None)

    panel, description = _load_panel(cfg)
    _echo(
        f"  panel: {description['name']}  {description['rows']:,} rows, "
        f"{description['instruments']} instruments, {description['days']} days "
        f"({description['load_seconds']}s)"
    )

    problems = preflight(cfg, panel, split, strict=args.preflight)
    if problems and args.preflight:
        return 2

    # The search sees history up to the end of its fitness window and no further.
    # Slicing here rather than inside the evaluator is what makes that structural:
    # an alpha cannot read the test period even by accident.
    _, window_end = _window_for(cfg, split)
    fit_panel = slice_panel(panel, None, window_end)

    archive = RunArchive(args.out or cfg.run.out_dir, run_name=args.name or cfg.data.market)
    _echo(f"  archive: {archive.path}")

    recorder = CallRecorder(archive.llm_log_path)
    llm = build_client(cfg.llm, recorder=recorder, max_calls=cfg.evolution.max_llm_calls)
    evaluator = FitnessEvaluator(
        fit_panel,
        cfg.fitness,
        cfg.quality,
        window=_window_for(cfg, split),
        horizon=cfg.data.horizon,
    )

    progress = _make_progress(archive, quiet=args.quiet)
    search = CogAlphaSearch(cfg, llm, evaluator, on_generation=progress)

    archive.write_config(cfg)
    archive.write_panel(description, extra={"fit_split": split})

    started = time.time()
    try:
        result = search.run()
    except KeyboardInterrupt:
        _echo("\ninterrupted; the generations written so far are in the archive")
        return 130

    info = archive.save_run(result, cfg, description)
    summary = result.summary()

    _echo("")
    _echo(f"finished in {time.time() - started:.0f}s")
    _echo(
        f"  {summary['alphas_seen']} alphas seen, "
        f"{summary['unique_structures']} unique structures, "
        f"{summary['duplicates_reused']} duplicates dropped"
    )
    _echo(f"  tiers: {summary['tiers']}")
    _echo(f"  llm: {summary['llm_calls']} calls, {summary['llm_tokens']:,} tokens")
    if summary["stopped_early"]:
        for agent, reason in summary["stopped_early"].items():
            _echo(f"  early stop [{agent}]: {reason}")
    _echo(f"  {len(result.candidates)} candidates -> {info['report']}")

    if not result.candidates:
        _echo(
            "\nNo alpha cleared the elite gate. Before loosening thresholds, read the "
            "rejection breakdown above: a run dominated by one stage has a prompt or "
            "data problem, not a threshold problem."
        )
    return 0


def _make_progress(archive, quiet: bool):
    """Per-generation callback: archive the record, and print a line unless quiet."""

    def on_generation(record) -> None:
        archive.write_generation(record)
        if quiet:
            return
        best = record.best.get("rank_ic")
        best_txt = f" best_rankic={best:+.4f}" if isinstance(best, (int, float)) else ""
        _echo(
            f"  g{record.generation:<3} c{record.cycle} {record.agent:<24} "
            f"raw={record.n_generated:<3} pass={record.n_passed_checker:<3} "
            f"qual={record.n_qualified:<3} elite={record.n_elite:<2}"
            f"{best_txt}  {record.wall_seconds:.0f}s  {record.llm_calls} calls"
        )

    return on_generation


# --------------------------------------------------------------------- compose


def cmd_compose(args: argparse.Namespace) -> int:
    """Combine candidates into one prediction and print it beside paper Table 1.

    Takes candidates either from an archived run (``--run``) or from explicit ``.py``
    paths, so a hand-written baseline can be composed without running a search.
    """
    from cogalpha.archive import RunArchive
    from cogalpha.compose import compose_from_codes

    cfg = _build_config(args)

    codes: Dict[str, str] = {}
    if args.run:
        loaded = RunArchive.load(args.run)
        codes = loaded.candidate_codes()
        if not codes:
            _echo(f"no candidate .py files in {args.run}/candidates/")
            return 2
        _echo(f"cogalpha compose  |  {len(codes)} candidates from {args.run}")
    else:
        for path in _expand_alpha_files(args.files):
            codes[path.stem] = path.read_text(encoding="utf-8")
        if not codes:
            _echo("no alpha .py files matched; pass --run DIR or one or more .py paths")
            return 2
        _echo(f"cogalpha compose  |  {len(codes)} files")

    panel, description = _load_panel(cfg)
    _echo(f"  panel: {description['days']} days, {description['instruments']} instruments")

    results = []
    for model in args.model:
        result = compose_from_codes(
            codes,
            panel,
            cfg,
            model_kind=model,
            split=args.split,
            top_n=args.top_n,
            rolling_step=args.rolling_step,
            run_backtest_flag=not args.no_backtest,
            lgbm_lr=args.lgbm_lr,
        )
        results.append(result)
        _echo(f"  {result.table_row()}  folds={result.n_folds} days={result.n_days}")
        if result.alphas_dropped:
            for name, why in result.alphas_dropped.items():
                _echo(f"    dropped {name}: {why}")
        for warning in _implausible(result):
            _echo(f"    WARNING: {warning}")

    _echo("")
    _echo(f"paper Table 1, CSI300 / {cfg.data.horizon}-day, 20-alpha combinations:")
    for line in _PAPER_TABLE1:
        _echo(f"  {line}")
    _echo(
        "\nComparability caveats: the paper trains on the 2011-2019 split of a qlib "
        "community snapshot, which is not the snapshot here; and its numbers come "
        "from LLM-generated alphas. Read the rows above as a check that the "
        "composition pipeline is on the paper's scale, not as a reproduction of it."
    )

    if args.json:
        payload = [r.to_dict() for r in results]
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _echo(f"\nwrote {args.json}")
    return 0


#: Table 1 rows worth comparing against, transcribed from the paper (CSI300, 10d).
_PAPER_TABLE1 = (
    "CogAlpha     IC=+0.0591 RankIC=+0.0814 ICIR=+0.3410 RankICIR=+0.4350 AER=+0.1639 IR=+1.8999",
    "Alpha158     IC=+0.0358 RankIC=+0.0402 ICIR=+0.2737 RankICIR=+0.2866 AER=+0.0946 IR=+0.8556",
    "LightGBM     IC=+0.0269 RankIC=+0.0412 ICIR=+0.2811 RankICIR=+0.3327 AER=+0.0878 IR=+1.0980",
    "AlphaAgent   IC=+0.0246 RankIC=+0.0289 ICIR=+0.2407 RankICIR=+0.2721 AER=+0.1072 IR=+1.2310",
    "Linear       IC=+0.0165 RankIC=+0.0211 ICIR=+0.1612 RankICIR=+0.1655 AER=-0.0076 IR=-0.0756",
)

#: |RankIC| above this is not achievable on daily cross-sectional equity data.
#: The best published daily alphas sit near 0.08; the paper's own headline is
#: 0.0814. A composition reading 0.5+ is a leak, not a result -- included here
#: because the failure is silent otherwise: a leaky column makes every metric look
#: excellent, which is the one direction a researcher will not question.
_IMPLAUSIBLE_RANK_IC = 0.20
_IMPLAUSIBLE_ICIR = 3.0


def _implausible(result) -> List[str]:
    """Flag composition scores that are too good to be real.

    Verified against a deliberately leaky control (next-10-day return) mixed into
    20 sound factors: Ridge returned RankIC +0.9809 and RankICIR +82.69. Any single
    leaky column dominates the fit, so a composition score in that range means a
    factor slipped past the leakage stage -- not that the alphas are good.
    """
    warnings: List[str] = []
    for label, value, limit in (
        ("RankIC", result.rank_ic, _IMPLAUSIBLE_RANK_IC),
        ("IC", result.ic, _IMPLAUSIBLE_RANK_IC),
        ("RankICIR", result.rank_icir, _IMPLAUSIBLE_ICIR),
        ("ICIR", result.icir, _IMPLAUSIBLE_ICIR),
    ):
        if value is None or value != value:
            continue
        if abs(float(value)) > limit:
            warnings.append(
                f"|{label}|={abs(float(value)):.4f} exceeds {limit}, which daily "
                "cross-sectional equity data does not support (the paper's headline "
                "RankIC is 0.0814). Suspect a look-ahead factor among the inputs; "
                "run `cogalpha evaluate` on each one and check its leakage verdict."
            )
            break
    return warnings



# -------------------------------------------------------------------- evaluate


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score standalone alpha files, without a search or a composition.

    Useful for two things the other commands cannot do: checking a hand-written
    baseline, and re-scoring an archived candidate on a *different* split from the
    one it was selected on — which is how alpha decay becomes visible.
    """
    from cogalpha.agents.parse import function_name
    from cogalpha.data.panel import slice_panel
    from cogalpha.fitness import FitnessEvaluator
    from cogalpha.fitness.thresholds import combined_score
    from cogalpha.types import Alpha

    cfg = _build_config(args)

    alphas: List[Alpha] = []
    for path in _expand_alpha_files(args.files):
        code = path.read_text(encoding="utf-8")
        name = function_name(code)
        if name is None:
            _echo(f"  skipped {path}: no single top-level function")
            continue
        alphas.append(Alpha(code=code, name=name, meta={"path": str(path)}))

    if not alphas:
        _echo(f"no alpha .py files matched {args.files}")
        return 2

    panel, description = _load_panel(cfg)
    window = _window_for(cfg, args.split)
    fit_panel = slice_panel(panel, None, window[1])
    _echo(
        f"cogalpha evaluate  |  {len(alphas)} alphas on {args.split} "
        f"{window[0]}..{window[1]}  ({description['instruments']} instruments)"
    )

    evaluator = FitnessEvaluator(
        fit_panel,
        cfg.fitness,
        cfg.quality,
        window=window,
        horizon=cfg.data.horizon,
        run_backtest=not args.no_backtest,
    )
    outcomes = evaluator.evaluate(alphas)

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        outcome = outcomes.get(alpha.alpha_id)
        if outcome is None or not outcome.ok:
            reason = outcome.error if outcome else "no result"
            _echo(f"  {alpha.name:<40} FAILED  {reason[:90]}")
            rows.append({"name": alpha.name, "ok": False, "error": reason})
            continue

        leak = outcome.leakage or {}
        if leak.get("leaked"):
            findings = "; ".join(leak.get("findings", []))[:110]
            _echo(f"  {alpha.name:<40} LEAKAGE  {findings}")
            rows.append({"name": alpha.name, "ok": False, "leakage": findings})
            continue

        f = outcome.fitness
        if f is None:
            issues = "; ".join((outcome.numeric or {}).get("issues", []))[:110]
            _echo(f"  {alpha.name:<40} REJECTED  {issues}")
            rows.append({"name": alpha.name, "ok": False, "numeric": issues})
            continue

        _echo(
            f"  {alpha.name:<40} IC={f.ic:+.4f} ICIR={f.icir:+.3f} "
            f"RankIC={f.rank_ic:+.4f} RankICIR={f.rank_icir:+.3f} MI={f.mi:.4f}"
            + (f" AER={f.aer:+.4f} IR={f.ir:+.3f}" if f.aer is not None else "")
        )
        row = {"name": alpha.name, "ok": True, "score": combined_score(f, cfg.fitness.use_abs_ic)}
        row.update(f.to_dict())
        rows.append(row)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
        _echo(f"\nwrote {args.json}")
    return 0


# ---------------------------------------------------------------------- report


def cmd_report(args: argparse.Namespace) -> int:
    """Summarise an archived run: totals, per-stage rejections, LLM accounting.

    Everything printed here is derived from the archive alone — no re-running and no
    model calls.  That is the point of keeping rejected alphas and the full
    transcript: the diagnosis of a bad run happens afterwards.
    """
    from cogalpha.archive import RunArchive

    loaded = RunArchive.load(args.run)
    summary = loaded.summary
    generations = loaded.generations()
    alphas = loaded.alphas()
    calls = loaded.llm_calls()

    _echo(f"cogalpha report  |  {args.run}")
    _echo("")
    _echo("== run ==")
    for key in ("candidates", "generations_run", "alphas_seen", "unique_structures",
                "duplicates_reused", "llm_calls", "llm_tokens", "wall_seconds"):
        if key in summary:
            _echo(f"  {key:<20} {summary[key]}")

    panel = loaded.panel
    if panel:
        _echo("")
        _echo("== data ==")
        for key in ("name", "provider", "market", "start", "end", "days",
                    "instruments", "fit_split"):
            if key in panel:
                _echo(f"  {key:<20} {panel[key]}")

    # --- where alphas died ---------------------------------------------------
    if alphas:
        _echo("")
        _echo("== outcome by stage ==")
        stages: Dict[str, int] = {}
        for a in alphas:
            key = a.get("tier") if not a.get("rejected_at") else f"rejected:{a['rejected_at']}"
            stages[str(key)] = stages.get(str(key), 0) + 1
        total = sum(stages.values())
        for key, count in sorted(stages.items(), key=lambda kv: -kv[1]):
            _echo(f"  {key:<32} {count:>5}  {count / total:>6.1%}")

        # A run dominated by one rejection stage is a prompt or data problem, and
        # naming it is more useful than reporting the aggregate pass rate.
        rejects = {k: v for k, v in stages.items() if k.startswith("rejected:")}
        if rejects:
            worst, worst_n = max(rejects.items(), key=lambda kv: kv[1])
            if worst_n / total > 0.35:
                _echo(
                    f"\n  NOTE: {worst_n / total:.0%} of all alphas died at "
                    f"{worst.split(':', 1)[1]}. That is a systematic failure, not "
                    "attrition -- inspect a few of those alphas before tuning "
                    "thresholds."
                )

    # --- per-operator productivity ------------------------------------------
    if generations:
        _echo("")
        _echo("== operators ==")
        ops: Dict[str, int] = {}
        for g in generations:
            for op, n in (g.get("op_counts") or {}).items():
                ops[op] = ops.get(op, 0) + int(n)
        for op, n in sorted(ops.items(), key=lambda kv: -kv[1]):
            _echo(f"  {op:<28} {n:>5}")

        _echo("")
        _echo("== elite trajectory ==")
        for g in generations:
            score = g.get("elite_mean_score")
            bar = ""
            if isinstance(score, (int, float)) and score == score:
                bar = "#" * max(int(score * 20), 0)
            _echo(
                f"  g{g['generation']:<3} c{g['cycle']} {str(g['agent'])[:22]:<22} "
                f"elite={g.get('n_elite', 0):<3} score="
                f"{'  n/a' if not isinstance(score, (int, float)) or score != score else f'{score:.4f}'} {bar}"
            )

    # --- llm accounting ------------------------------------------------------
    if calls:
        _echo("")
        _echo("== llm calls by role ==")
        by_role: Dict[str, Dict[str, int]] = {}
        for c in calls:
            role = str((c.get("tags") or {}).get("role", "?"))
            entry = by_role.setdefault(role, {"calls": 0, "tokens": 0})
            entry["calls"] += 1
            entry["tokens"] += int((c.get("usage") or {}).get("total_tokens", 0))
        for role, entry in sorted(by_role.items(), key=lambda kv: -kv[1]["calls"]):
            _echo(f"  {role:<16} {entry['calls']:>5} calls  {entry['tokens']:>10,} tokens")

    if args.regenerate:
        _echo("\n--regenerate needs the live objects; re-run `cogalpha search` instead.")
    return 0


# --------------------------------------------------------------------- inspect


def cmd_inspect(args: argparse.Namespace) -> int:
    """Print static facts: the hierarchy, the guidance modes, or a resolved config."""
    topic = args.topic

    if topic == "hierarchy":
        from cogalpha.agents.hierarchy import (
            HIERARCHY,
            LAYER_DESCRIPTIONS,
            LAYERS,
            by_level,
            select_agents,
        )

        _echo(f"Seven-Level Agent Hierarchy -- {len(HIERARCHY)} task-specific agents\n")
        for level in sorted(LAYERS):
            _echo(f"Level {level}: {LAYERS[level]}")
            _echo(f"  {LAYER_DESCRIPTIONS[level]}")
            for agent in by_level(level):
                _echo(f"  - {agent.name}")
                _echo(f"      {agent.focus}")
            _echo("")
        n = args.n or 13
        chosen = select_agents(n, seed=args.seed)
        _echo(f"golden-ratio selection of {n} (seed {args.seed}):")
        for agent in chosen:
            _echo(f"  L{agent.level} {agent.name}")
        return 0

    if topic == "guidance":
        from cogalpha.agents.guidance import DEFAULT_ORDER, MODES

        _echo("Diversified Guidance -- five paraphrasing modes\n")
        for name in DEFAULT_ORDER:
            mode = MODES[name]
            _echo(f"{mode.name}")
            _echo(f"  definition:  {mode.description}")
            _echo(f"  instruction: {mode.instruction}")
            _echo("")
        return 0

    if topic == "config":
        cfg = _build_config(args)
        cfg.llm.resolve_secrets()
        payload = cfg.to_dict()
        # Never print the credential, even when the user asked for the config.
        if payload.get("llm", {}).get("api_key"):
            payload["llm"]["api_key"] = "<redacted>"
        _echo(json.dumps(payload, indent=2, default=str))
        _echo(f"\nllm: {cfg.llm.describe()}")
        return 0

    if topic == "data":
        cfg = _build_config(args)
        panel, description = _load_panel(cfg)
        _echo(json.dumps(description, indent=2, default=str))
        for split in ("train", "valid", "test"):
            _echo(f"\n{split}:")
            preflight(cfg, panel, split, strict=False)
        return 0

    raise SystemExit(f"unknown inspect topic '{topic}'")


# --------------------------------------------------------------------- monitor


def cmd_monitor(args: argparse.Namespace) -> int:
    """Serve the live dashboard for a run directory.

    Reads only the archive's JSONL streams, so it is safe to start, stop and restart
    against a running search, and it replays a finished run identically.
    """
    from cogalpha.monitor.server import serve

    try:
        serve(
            args.run,
            host=args.host,
            port=args.port,
            poll_interval=args.interval,
            open_browser=args.open,
        )
    except KeyboardInterrupt:
        _echo("\nmonitor stopped")
    return 0


# ----------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser. Split out from :func:`main` so the help text
    can be rendered without importing pandas."""
    parser = argparse.ArgumentParser(
        prog="cogalpha",
        description=(
            "Cognitive alpha mining via LLM-driven code-based evolution "
            "(implementation of the ACL 2026 CogAlpha paper)."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- search --------------------------------------------------------------
    p_search = sub.add_parser(
        "search",
        help="run the evolutionary search and archive it",
        description="Run the working stream of Figure 1: hierarchy -> checker -> "
        "fitness -> thinking evolution, archiving every generation.",
    )
    _add_config_args(p_search)
    p_search.add_argument("--out", metavar="DIR", help="archive root (default: runs/)")
    p_search.add_argument("--name", metavar="LABEL", help="run label, prefixed with a timestamp")
    p_search.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        help="which window fitness is measured on (default: data.fit_split)",
    )
    p_search.add_argument(
        "--no-preflight",
        dest="preflight",
        action="store_false",
        help="start even if the window checks fail",
    )
    p_search.add_argument("--quiet", action="store_true", help="suppress per-generation lines")
    p_search.set_defaults(func=cmd_search, preflight=True)

    # --- compose -------------------------------------------------------------
    p_compose = sub.add_parser(
        "compose",
        help="combine candidates into one prediction (the Table 1 reading)",
        description="Table 1 scores multi-factor combinations of 20 alphas, not "
        "single alphas. This trains the downstream model on the candidates with "
        "rolling retraining and scores its prediction.",
    )
    _add_config_args(p_compose, with_llm=False)
    p_compose.add_argument("--run", metavar="DIR", help="archived run to take candidates from")
    p_compose.add_argument("files", nargs="*", help="alpha .py files, if not using --run")
    p_compose.add_argument(
        "--model",
        action="append",
        choices=["lightgbm", "ridge", "mean"],
        help="downstream model; repeatable (default: lightgbm)",
    )
    p_compose.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p_compose.add_argument("--top-n", type=int, default=20, help="how many candidates to use")
    p_compose.add_argument(
        "--rolling-step", type=int, default=126, help="retraining step in trading days (§B.4)"
    )
    p_compose.add_argument(
        "--lgbm-lr",
        type=float,
        default=0.0001,
        metavar="RATE",
        help=(
            "LightGBM learning rate (default 0.0001, the paper's §B.4 value). "
            "0.0001 x 1000 trees is a total shrinkage of 0.1, i.e. deliberately "
            "under-fitted; try 0.05 to see a converged tree model."
        ),
    )
    p_compose.add_argument("--no-backtest", action="store_true", help="skip AER/IR")
    p_compose.add_argument("--json", metavar="PATH", help="also write results as JSON")
    p_compose.set_defaults(func=cmd_compose)

    # --- evaluate ------------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate",
        help="score standalone alpha .py files",
        description="Score alphas on their own, on any split. Re-scoring an "
        "archived candidate on a later split is how alpha decay becomes visible.",
    )
    _add_config_args(p_eval, with_llm=False)
    p_eval.add_argument("files", nargs="+", help="alpha .py files or globs")
    p_eval.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    p_eval.add_argument("--no-backtest", action="store_true", help="skip AER/IR")
    p_eval.add_argument("--json", metavar="PATH", help="also write results as JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    # --- report --------------------------------------------------------------
    p_report = sub.add_parser(
        "report",
        help="summarise an archived run",
        description="Derive diagnostics from an archive: where alphas died, which "
        "operators produced anything, how the elite pool moved, what the LLM cost.",
    )
    p_report.add_argument("--run", required=True, metavar="DIR", help="archived run directory")
    p_report.add_argument("--regenerate", action="store_true", help=argparse.SUPPRESS)
    p_report.set_defaults(func=cmd_report)

    # --- monitor -------------------------------------------------------------
    p_monitor = sub.add_parser(
        "monitor",
        help="serve a live dashboard for a run",
        description="Watch a search in a browser: the 21-agent matrix, the quality "
        "checker funnel, the elite trajectory, and every prompt the run has sent. "
        "Reads the archive only -- safe against a running search.",
    )
    p_monitor.add_argument(
        "--run",
        default="runs",
        metavar="DIR",
        help="run archive, or a parent directory (the newest run inside is followed)",
    )
    p_monitor.add_argument("--port", type=int, default=8080)
    p_monitor.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "bind address. Local by default because the detail endpoints serve every "
            "prompt this run produced; pass 0.0.0.0 only on a trusted network"
        ),
    )
    p_monitor.add_argument(
        "--interval", type=float, default=1.0, metavar="SEC", help="push interval"
    )
    p_monitor.add_argument("--open", action="store_true", help="open a browser")
    p_monitor.set_defaults(func=cmd_monitor)

    # --- inspect -------------------------------------------------------------
    p_inspect = sub.add_parser(
        "inspect",
        help="print the hierarchy, guidance modes, resolved config, or data checks",
    )
    p_inspect.add_argument(
        "topic",
        choices=["hierarchy", "guidance", "config", "data"],
        help="what to print",
    )
    p_inspect.add_argument("--n", type=int, help="agents to select (hierarchy)")
    p_inspect.add_argument("--seed", type=int, default=42, help="selection seed (hierarchy)")
    _add_config_args(p_inspect)
    p_inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point (``cogalpha ...``). Returns an exit code.

    Exit codes: 0 success, 1 no command given, 2 configuration/data/credential
    problem, 130 interrupted.
    """
    from cogalpha.llm.base import LLMError

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from cogalpha import __version__

        _echo(f"cogalpha {__version__}")
        return 0

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    if args.command == "compose" and not args.model:
        args.model = ["lightgbm"]

    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        _echo("\ninterrupted")
        return 130
    except LLMError as exc:
        # Credential and endpoint problems already carry their own remediation
        # text; a traceback on top of it only obscures the instruction.
        _echo(f"error: {exc}")
        return 2
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # Configuration and data problems deserve one clear line rather than a
        # traceback the user has to read past.
        _echo(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())






