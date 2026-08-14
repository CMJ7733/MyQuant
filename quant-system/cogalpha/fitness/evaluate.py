"""Fitness evaluation: the "5-M Evaluation" stage of Figure 1.

This is where the sandbox and the metrics meet.  One pass over an alpha in the
child process produces everything the parent needs:

1. the factor values on the scored window;
2. the numerical-stability report (§A.3);
3. the five metrics (§3.4);
4. the truncation leakage probe and determinism check (§A.3);
5. optionally the top-50/drop-5 backtest (§B.2).

Doing all of it in one child visit is not an optimisation detail — recomputing the
factor per stage would triple the cost of the most expensive part of a generation,
and the leakage probe *requires* a second evaluation on truncated data anyway, so
the code is arranged to pay that price exactly once.

Warm-up
-------
The factor is computed on the *full* panel and only then restricted to the scored
window.  A 60-day rolling mean evaluated on a panel that starts at the window's
first day would be NaN for the first three months, and the alpha would be rejected
for poor coverage rather than for being wrong.  This is why
:func:`evaluate_alphas` takes ``fit_window`` separately from the panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cogalpha.config import FitnessConfig, QualityConfig
from cogalpha.data.panel import Panel, forward_return
from cogalpha.fitness import metrics as M
from cogalpha.fitness.backtest import run_backtest
from cogalpha.quality import leakage as L
from cogalpha.quality.numeric import check_numeric
from cogalpha.quality.sandbox import SandboxRunner, apply_alpha
from cogalpha.types import Alpha, Fitness


@dataclass
class EvalOutcome:
    """Everything learned about one alpha in one sandbox visit."""

    alpha_id: str
    ok: bool
    error: str = ""
    error_type: str = ""
    fitness: Optional[Fitness] = None
    numeric: Dict[str, Any] = field(default_factory=dict)
    leakage: Dict[str, Any] = field(default_factory=dict)
    backtest: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def evaluation_job(
    values: pd.DataFrame,
    fn,
    code: str,
    name: str,
    panel: Panel,
    frames: Dict[str, pd.DataFrame],
    *,
    label_wide: pd.DataFrame,
    window: Tuple[Optional[str], Optional[str]],
    fitness_cfg: FitnessConfig,
    quality_cfg: QualityConfig,
    universe_mask: Optional[pd.DataFrame] = None,
    run_leakage_probe: bool = True,
    leakage_tail_days: int = 40,
    run_backtest_flag: bool = False,
    backtest_return: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Child-side worker for :class:`~cogalpha.quality.sandbox.SandboxRunner`.

    Returns a plain dict of scalars and short lists — no frames — so the queue
    stays small regardless of panel size.
    """
    start, end = window
    scored = _restrict(values, start, end)

    numeric = check_numeric(
        scored,
        nan_ratio_limit=quality_cfg.nan_ratio_limit,
        min_coverage=quality_cfg.min_coverage,
        min_distinct_per_day=quality_cfg.min_distinct_per_day,
        max_tie_ratio=quality_cfg.max_tie_ratio,
        abs_value_limit=quality_cfg.abs_value_limit,
        universe=_restrict(universe_mask, start, end) if universe_mask is not None else None,
    )


    payload: Dict[str, Any] = {"numeric": numeric.to_dict()}

    # Leakage is checked even when the numbers are unusable: a leaky factor should
    # be recorded as leaky, not as "too many NaNs", or the rejection statistics
    # misattribute the failure.
    static_findings = L.scan_lookahead(code)
    probe = None
    determinism = None

    if run_leakage_probe and quality_cfg.leakage_shift_probe and not scored.empty:
        probe, determinism = _run_probe(
            fn=fn,
            frames=frames,
            name=name,
            values_full=values,
            tail_days=leakage_tail_days,
        )

    leak = L.build_report(static_findings, probe=probe, determinism=determinism)
    payload["leakage"] = leak.to_dict()

    if not numeric.ok or leak.leaked or not leak.deterministic:
        payload["fitness"] = None
        return payload

    label = _restrict(label_wide, start, end)
    aligned_values, aligned_label = M.align(scored, label)

    ic = M.ic_series(
        aligned_values, aligned_label, method="pearson", min_names=fitness_cfg.min_names_per_day
    )
    ric = M.ic_series(
        aligned_values,
        aligned_label,
        method=fitness_cfg.rank_ic_method,
        min_names=fitness_cfg.min_names_per_day,
    )
    mi = M.mutual_information(
        aligned_values, aligned_label, bins=fitness_cfg.mi_bins, scale=fitness_cfg.mi_scale
    )

    fitness = Fitness(
        ic=ic.mean,
        icir=ic.ir,
        rank_ic=ric.mean,
        rank_icir=ric.ir,
        mi=mi,
        nan_ratio=numeric.nan_ratio,
        coverage=numeric.coverage,
        n_days=ic.n_days,
        ic_series_std=ic.std,
    )

    if run_backtest_flag and backtest_return is not None:
        bt = run_backtest(
            aligned_values,
            _restrict(backtest_return, start, end),
            top_k=fitness_cfg.top_k,
            drop_n=fitness_cfg.drop_n,
            open_cost=fitness_cfg.open_cost,
            close_cost=fitness_cfg.close_cost,
            min_fee=fitness_cfg.min_fee,
            trading_days_per_year=fitness_cfg.trading_days_per_year,
        )
        fitness.aer = bt.aer
        fitness.ir = bt.ir
        payload["backtest"] = bt.to_dict()

    payload["fitness"] = fitness.to_dict()
    payload["ic_detail"] = {"pearson": ic.to_dict(), "rank": ric.to_dict()}
    return payload


def _restrict(
    frame: pd.DataFrame,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    idx = frame.index
    mask = np.ones(len(idx), dtype=bool)
    if start is not None:
        mask &= idx >= pd.Timestamp(start)
    if end is not None:
        mask &= idx <= pd.Timestamp(end)
    return frame[mask]


def _run_probe(
    fn,
    frames: Dict[str, pd.DataFrame],
    name: str,
    values_full: pd.DataFrame,
    tail_days: int,
) -> Tuple[Tuple[bool, float, int, str], Tuple[bool, str]]:
    """Recompute on a truncated panel and on the same panel twice.

    The truncation is **by calendar date**: a single cutoff is chosen
    ``tail_days`` trading days before the panel's last date, and every instrument
    is cut there.  Truncating a fixed number of *rows per instrument* instead is
    wrong on a real panel and produces a false positive on every alpha -- names
    delist at different times, so dropping the last 40 rows of a stock that left
    the index in 2015 deletes 2015 data that sits well before the cutoff, and the
    comparison then reports values "changing" that were simply no longer computed.
    (Observed directly: 14 of 16 clean alphas flagged, all with ``max abs diff 0``
    and cells turning NaN.)

    A causal factor is bit-identical on the remaining dates. The second identical
    run is the determinism check and is nearly free because the frames exist.
    """
    if not frames:
        return (False, 0.0, 0, "probe skipped: no instruments"), (True, "skipped")

    # Global trading calendar across the panel.
    calendar = values_full.index.sort_values()
    if len(calendar) <= tail_days + 5:
        return (
            (False, 0.0, 0, "probe skipped: panel too short to truncate"),
            (True, "skipped"),
        )
    cutoff = pd.Timestamp(calendar[-(tail_days + 1)])

    truncated = {}
    for inst, frame in frames.items():
        sub = frame.loc[frame.index <= cutoff]
        # Keep only names with enough history to compute anything; an empty frame
        # would make the alpha raise for reasons unrelated to causality.
        if len(sub) >= 2:
            truncated[inst] = sub.copy()

    if not truncated:
        return (
            (False, 0.0, 0, "probe skipped: nothing left after truncation"),
            (True, "skipped"),
        )

    try:
        values_trunc = apply_alpha(fn, truncated, column=name)
    except Exception as exc:  # noqa: BLE001
        # Failing on a shorter panel is itself disqualifying: the private-board
        # style re-run the paper's unit test imitates would hit exactly this.
        return (
            (
                True,
                float("nan"),
                0,
                f"factor raised on a truncated panel ({type(exc).__name__}: {exc}) "
                "-- it cannot be computed incrementally",
            ),
            (True, "determinism check skipped after truncation failure"),
        )

    probe = L.truncation_probe(values_full, values_trunc, cutoff=cutoff)

    try:
        values_repeat = apply_alpha(fn, frames, column=name)
        determinism = L.determinism_check(values_full, values_repeat)
    except Exception as exc:  # noqa: BLE001
        determinism = (False, f"second run raised {type(exc).__name__}: {exc}")

    return probe, determinism



class FitnessEvaluator:
    """Parent-side driver: batches alphas through the sandbox and fills ``Fitness``.

    One evaluator serves a whole run.  It owns the label frames (built once) and
    the sandbox configuration, so a caller only ever passes alphas.
    """

    def __init__(
        self,
        panel: Panel,
        fitness_cfg: FitnessConfig,
        quality_cfg: QualityConfig,
        window: Tuple[Optional[str], Optional[str]],
        horizon: int,
        label_price: str = "open",
        label_offset: int = 1,
        run_backtest: bool = False,
    ) -> None:
        self.panel = panel
        self.fitness_cfg = fitness_cfg
        self.quality_cfg = quality_cfg
        self.window = window
        self.horizon = horizon

        label = forward_return(panel.frame, horizon=horizon, price=label_price, offset=label_offset)
        self.label_wide = M.label_to_wide(label)
        #: Denominator for every coverage ratio; see :meth:`Panel.universe_mask`.
        self.universe_mask = panel.universe_mask()

        self.backtest_return: Optional[pd.DataFrame] = None
        if run_backtest:
            # The backtest payoff is the **one-period** return, not the horizon-h
            # label. §B.2's top-50/drop-5 rule rebalances every trading day, so the
            # per-day P&L is one day of return; feeding it the 10-day label counts
            # every return ten times over and compounds the overlap.  Measured on
            # real CSI300 2021-2024 the difference is not subtle: a low-volatility
            # alpha reports AER +1.2345 / IR +3.176 on the 10-day label against
            # AER +0.1075 / IR +0.861 on the daily return, and the former's
            # cumulative excess reaches +7626%.  The paper's own scale
            # (CogAlpha AER 0.1639, IR 1.8999) confirms the daily reading.
            self.backtest_return = M.label_to_wide(
                forward_return(panel.frame, horizon=1, price=label_price, offset=label_offset)
            )
        self.run_backtest = run_backtest

    def evaluate(self, alphas: Sequence[Alpha]) -> Dict[str, EvalOutcome]:
        """Evaluate ``alphas``, enriching each with fitness or a rejection reason."""
        jobs = [(a.alpha_id, a.name, a.code) for a in alphas]
        if not jobs:
            return {}

        runner = SandboxRunner(
            job_fn=evaluation_job,
            job_kwargs={
                "label_wide": self.label_wide,
                "window": self.window,
                "fitness_cfg": self.fitness_cfg,
                "quality_cfg": self.quality_cfg,
                "universe_mask": self.universe_mask,
                "run_leakage_probe": True,
                "leakage_tail_days": self.quality_cfg.leakage_tail_days,
                "run_backtest_flag": self.run_backtest,
                "backtest_return": self.backtest_return,
            },
            timeout_s=self.quality_cfg.exec_timeout_s,
            memory_mb=self.quality_cfg.memory_limit_mb,
            allowed_imports=self.quality_cfg.allowed_imports,
        )

        raw = runner.run(self.panel, jobs)
        out: Dict[str, EvalOutcome] = {}

        for alpha in alphas:
            result = raw.get(alpha.alpha_id)
            if result is None:
                out[alpha.alpha_id] = EvalOutcome(
                    alpha_id=alpha.alpha_id,
                    ok=False,
                    error="sandbox returned no result",
                    error_type="SandboxMissing",
                )
                continue
            if not result.ok:
                out[alpha.alpha_id] = EvalOutcome(
                    alpha_id=alpha.alpha_id,
                    ok=False,
                    error=result.error,
                    error_type=result.error_type,
                    seconds=result.seconds,
                )
                continue

            payload = result.payload
            fitness_dict = payload.get("fitness")
            fitness = _fitness_from_dict(fitness_dict) if fitness_dict else None
            out[alpha.alpha_id] = EvalOutcome(
                alpha_id=alpha.alpha_id,
                ok=True,
                fitness=fitness,
                numeric=payload.get("numeric", {}),
                leakage=payload.get("leakage", {}),
                backtest=payload.get("backtest", {}),
                seconds=result.seconds,
            )
        return out


def _fitness_from_dict(payload: Dict[str, Any]) -> Fitness:
    """Rebuild :class:`Fitness` from the child's JSON-safe dict."""
    def num(key: str) -> float:
        value = payload.get(key)
        return float("nan") if value is None else float(value)

    fitness = Fitness(
        ic=num("ic"),
        icir=num("icir"),
        rank_ic=num("rank_ic"),
        rank_icir=num("rank_icir"),
        mi=num("mi"),
        nan_ratio=num("nan_ratio"),
        coverage=num("coverage"),
        n_days=int(payload.get("n_days") or 0),
        ic_series_std=num("ic_series_std"),
    )
    aer = payload.get("aer")
    ir = payload.get("ir")
    fitness.aer = None if aer is None else float(aer)
    fitness.ir = None if ir is None else float(ir)
    return fitness
