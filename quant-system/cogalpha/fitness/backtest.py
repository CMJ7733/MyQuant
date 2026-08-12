"""Top-50/drop-5 backtest (Appendix B.2) producing AER and IR.

The paper reports these two alongside the five predictive metrics.  They do not
participate in tiering — they are what you quote for a finished candidate.

Rules, from §B.2 verbatim: hold the 50 highest-predicted names; on each trading
day retain previously selected high-ranking stocks and replace at most 5
positions; execute at the open; open cost 0.05%, close cost 0.15%, minimum fee
5 CNY.  AER = mean daily excess x 252; IR = (mean/std) x sqrt(252).

The "retain and replace at most 5" rule is the part worth being careful about: it
is a *turnover cap*, not a fixed rebalance. Each day the current holdings that are
still ranked highly stay; only the weakest are swapped for the strongest
non-holdings, up to five. That is what keeps the strategy's turnover low enough
for the 5 CNY minimum fee to matter at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Outcome of the ranking backtest."""

    aer: float = float("nan")
    ir: float = float("nan")
    mean_daily_excess: float = float("nan")
    std_daily_excess: float = float("nan")
    cumulative_excess: float = float("nan")
    max_drawdown: float = float("nan")
    turnover: float = float("nan")
    win_rate: float = float("nan")
    n_days: int = 0

    equity: Optional[pd.Series] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """JSON-safe metrics. The equity curve is deliberately not included."""
        return {
            "aer": _f(self.aer),
            "ir": _f(self.ir),
            "mean_daily_excess": _f(self.mean_daily_excess),
            "std_daily_excess": _f(self.std_daily_excess),
            "cumulative_excess": _f(self.cumulative_excess),
            "max_drawdown": _f(self.max_drawdown),
            "turnover": _f(self.turnover),
            "win_rate": _f(self.win_rate),
            "n_days": self.n_days,
        }


def _f(value: float) -> Optional[float]:
    v = float(value)
    return v if np.isfinite(v) else None


def run_backtest(
    values: pd.DataFrame,
    forward_return: pd.DataFrame,
    top_k: int = 50,
    drop_n: int = 5,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
    min_fee: float = 5.0,
    capital: float = 1_000_000.0,
    trading_days_per_year: int = 252,
    benchmark: Optional[pd.Series] = None,
) -> BacktestResult:
    """Simulate the top-50/drop-5 portfolio.

    Parameters
    ----------
    values:
        ``(date x instrument)`` predicted score. Higher is better.
    forward_return:
        ``(date x instrument)`` **one-period** return earned by holding from this
        day's signal to the next rebalance.  Because §B.2 rebalances daily, that is
        a one-day return -- not the ``horizon``-day label the IC is measured
        against.  Passing the h-day label counts each day's return h times and
        compounds the overlap: on real CSI300 2021-2024 the same alpha reads
        AER +1.23 / IR +3.18 with the 10-day label versus AER +0.11 / IR +0.86 with
        the daily return, the first being arithmetically impossible for a signal
        whose RankIC is 0.08.
    benchmark:
        Per-day benchmark return. Defaults to the equal-weight return of all names
        with data that day, which is the natural neutral comparison for a
        cross-sectional selection strategy over a fixed universe.
    """
    dates = values.index.intersection(forward_return.index)
    columns = values.columns.intersection(forward_return.columns)
    scores = values.loc[dates, columns]
    rets = forward_return.loc[dates, columns]

    if len(dates) < 2:
        return BacktestResult()

    if benchmark is None:
        bench = rets.mean(axis=1, skipna=True)
    else:
        bench = benchmark.reindex(dates)

    holdings: Set[str] = set()
    excess: List[float] = []
    turnovers: List[float] = []
    per_position = capital / max(top_k, 1)
    # A round trip costs both legs; the minimum fee applies per trade, so express
    # it as an equivalent rate on one position and take whichever binds.
    fee_floor_rate = min_fee / per_position if per_position > 0 else 0.0

    for date in dates:
        row = scores.loc[date].dropna()
        ret_row = rets.loc[date]

        if row.empty:
            excess.append(0.0)
            turnovers.append(0.0)
            continue

        ranked = row.sort_values(ascending=False)
        target = _next_holdings(holdings, ranked, top_k=top_k, drop_n=drop_n)

        n_sold = len(holdings - target)
        n_bought = len(target - holdings)

        if target:
            gross = float(ret_row.reindex(sorted(target)).mean(skipna=True))
            if not np.isfinite(gross):
                gross = 0.0
        else:
            gross = 0.0

        # Cost is charged on the traded fraction of the book, with the per-trade
        # minimum fee acting as a floor on each leg's rate.
        buy_rate = max(open_cost, fee_floor_rate) if n_bought else 0.0
        sell_rate = max(close_cost, fee_floor_rate) if n_sold else 0.0
        denom = max(len(target), 1)
        cost = (buy_rate * n_bought + sell_rate * n_sold) / denom

        bench_ret = float(bench.get(date, 0.0))
        if not np.isfinite(bench_ret):
            bench_ret = 0.0

        excess.append(gross - cost - bench_ret)
        turnovers.append((n_bought + n_sold) / (2.0 * denom))
        holdings = target

    series = pd.Series(excess, index=dates, dtype="float64").dropna()
    if series.empty:
        return BacktestResult()

    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    aer = mean * trading_days_per_year
    ir = (
        (mean / std) * np.sqrt(trading_days_per_year)
        if std and np.isfinite(std) and std > 0
        else float("nan")
    )

    equity = (1.0 + series).cumprod()
    peak = equity.cummax()
    drawdown = float((equity / peak - 1.0).min())

    return BacktestResult(
        aer=aer,
        ir=float(ir),
        mean_daily_excess=mean,
        std_daily_excess=std,
        cumulative_excess=float(equity.iloc[-1] - 1.0),
        max_drawdown=drawdown,
        turnover=float(np.mean(turnovers)) if turnovers else float("nan"),
        win_rate=float((series > 0).mean()),
        n_days=int(len(series)),
        equity=equity,
    )


def _next_holdings(
    current: Set[str],
    ranked: pd.Series,
    top_k: int,
    drop_n: int,
) -> Set[str]:
    """Apply the "retain, then replace at most ``drop_n``" rule of §B.2.

    Two kinds of exit have to be told apart:

    * **voluntary** -- a holding is still scoreable but has fallen out of the top
      ``top_k``.  At most ``drop_n`` of these are swapped per day; that cap is what
      keeps turnover low enough for the paper's 5 CNY minimum fee to bind.
    * **forced** -- a holding has no score today (delisted, suspended, or dropped
      out of the index).  It cannot be held, so it leaves regardless of the cap.

    The book is then refilled to ``top_k`` from the top of the ranking.  Refilling
    matters: capping *additions* at ``drop_n`` as well lets the portfolio shrink
    whenever a forced exit coincides with a voluntary swap, and a book that drifts
    between 45 and 50 names changes the per-day return for a reason that has
    nothing to do with the signal.  Observed on real CSI300 2021-2024: 14 forced
    exits across 10 days, which pulled the book to 45 names before this fix.
    """
    candidates = list(ranked.index)
    if not candidates:
        return set(current)
    if not current:
        return set(candidates[:top_k])

    top_set = set(candidates[:top_k])
    scoreable = current & set(candidates)

    # Voluntary exits: still scoreable, no longer top-ranked, worst first.
    stale = [name for name in reversed(candidates) if name in scoreable and name not in top_set]
    retained = scoreable - set(stale[:drop_n])

    room = top_k - len(retained)
    if room <= 0:
        ordered = [n for n in candidates if n in retained]
        return set(ordered[:top_k])

    additions = [n for n in candidates if n not in retained][:room]
    return retained | set(additions)

