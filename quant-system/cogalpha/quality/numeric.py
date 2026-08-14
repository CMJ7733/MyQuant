"""Numerical stability checks on a computed factor (§A.3, §B.4).

Runs in the sandbox child on the wide ``(date x instrument)`` factor frame, and
returns only scalars — the parent never sees the values.

The four checks are the paper's own list ("runtime errors, the proportion of NaN
values, overflow/underflow, and distinct values per day"), made precise:

* **NaN ratio** over the scored window, rejected above 30%;
* **coverage** per day, so a factor that is dense on average but empty for a
  stretch of days is caught — the average hides exactly the failure mode that
  breaks an IC series;
* **overflow/underflow**: non-finite values and magnitudes past a ceiling;
* **distinct values per day**: a factor that is constant across the cross-section
  carries no information even though every value is present and finite.

A fifth check, **tie mass**, is not in the paper's list and is added here because
the four above cannot see the failure it catches.  A factor built as
``max(trend, 0) * signal`` sends every stock on the inactive side of the gate to
*exactly* zero; ``(a - b).fillna(0.0)`` adds the warm-up rows to that same value.
Such a factor has a 0% NaN ratio, full coverage, and hundreds of distinct values
per day — it passes every existing gate — yet half its cross-section shares one
value.  :func:`cogalpha.fitness.metrics._rank_rows` correctly averages the ranks
within that tie group, which means half the universe receives an identical rank
and the RankIC is attenuated by an amount that depends on how big the group is.
Two factors are then not comparable on RankIC at all, and the tier percentiles
rank them against each other anyway.

Constant filling makes this worse in a second way: ``nan_ratio_limit`` exists to
measure missingness, and ``fillna(0)`` drives the measured ratio to zero without
supplying any information.  :mod:`cogalpha.quality.audit` rejects the constant
fill statically; this module catches the tie mass however it arose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class NumericReport:
    """Scalars describing a factor's numerical health."""

    ok: bool
    issues: List[str] = field(default_factory=list)
    nan_ratio: float = float("nan")
    coverage: float = float("nan")
    min_daily_coverage: float = float("nan")
    n_days: int = 0
    n_values: int = 0
    mean_distinct_per_day: float = float("nan")
    min_distinct_per_day: int = 0
    n_days_below_min_distinct: int = 0
    mean_tie_ratio: float = float("nan")
    max_daily_tie_ratio: float = float("nan")
    n_days_above_tie_limit: int = 0
    inf_count: int = 0
    over_limit_count: int = 0
    abs_max: float = float("nan")
    std_mean: float = float("nan")
    constant_days_ratio: float = float("nan")

    @property
    def detail(self) -> str:
        """Findings joined into one line, for the CheckReport detail field."""
        return "; ".join(self.issues) if self.issues else "numerically stable"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict: NaN and inf become ``None`` (invalid in JSON)."""
        payload = {
            "ok": self.ok,
            "issues": self.issues,
            "nan_ratio": _clean(self.nan_ratio),
            "coverage": _clean(self.coverage),
            "min_daily_coverage": _clean(self.min_daily_coverage),
            "n_days": self.n_days,
            "n_values": self.n_values,
            "mean_distinct_per_day": _clean(self.mean_distinct_per_day),
            "min_distinct_per_day": self.min_distinct_per_day,
            "n_days_below_min_distinct": self.n_days_below_min_distinct,
            "mean_tie_ratio": _clean(self.mean_tie_ratio),
            "max_daily_tie_ratio": _clean(self.max_daily_tie_ratio),
            "n_days_above_tie_limit": self.n_days_above_tie_limit,
            "inf_count": self.inf_count,
            "over_limit_count": self.over_limit_count,
            "abs_max": _clean(self.abs_max),
            "std_mean": _clean(self.std_mean),
            "constant_days_ratio": _clean(self.constant_days_ratio),
        }
        return payload


def _clean(value: float) -> Optional[float]:
    """JSON-safe float (``NaN``/``inf`` are not valid JSON)."""
    if value is None:
        return None
    v = float(value)
    if not np.isfinite(v):
        return None
    return v


def check_numeric(
    values: pd.DataFrame,
    nan_ratio_limit: float = 0.30,
    min_coverage: float = 0.60,
    min_distinct_per_day: int = 5,
    max_tie_ratio: float = 0.50,
    abs_value_limit: float = 1e12,
    min_days: int = 20,
    max_bad_day_ratio: float = 0.10,
    universe: Optional[pd.DataFrame] = None,
) -> NumericReport:
    """Assess a wide factor frame.

    Parameters
    ----------
    values:
        ``(date x instrument)`` float frame, already restricted to the scored
        window by the caller.
    universe:
        Boolean frame, same shape, marking the cells where the factor *could*
        have a value — i.e. where the underlying price data exists and the name is
        in the index that day.  When given, every ratio below is computed over the
        universe cells only.
    max_tie_ratio:
        Largest share of a day's cross-section allowed to sit on one value.  A day
        above this is a "bad day" in the same sense as one below
        ``min_distinct_per_day``, and the same ``max_bad_day_ratio`` decides
        whether there are too many of them.  See the module docstring for why a
        tie group is not caught by the distinct-value check.
    max_bad_day_ratio:
        Share of days allowed to fall below ``min_distinct_per_day``, or above
        ``max_tie_ratio``, before the factor is rejected.  A handful of thin days
        (holidays, halts) is normal; a tenth of the sample is not.

    Why the universe mask is not optional in practice
    ------------------------------------------------
    A panel of index constituents over many years is a *union*: CSI300 membership
    over 2011-2024 spans 748 tickers, of which ~300 are in the index on any given
    day.  The wide frame is therefore ~60% empty before an alpha does anything,
    and measuring the NaN ratio against the full rectangle rejects every alpha
    ever written -- the 30% limit of §B.4 is unreachable by construction.  The
    paper does not discuss this because it never arises on a fixed-membership
    handler.  With the mask, the same limit means what it says: of the values this
    alpha could have produced, no more than 30% may be missing.
    """
    issues: List[str] = []

    if values is None or values.empty:
        return NumericReport(ok=False, issues=["factor frame is empty"])

    frame = values.replace([np.inf, -np.inf], np.nan)

    if universe is not None:
        mask = universe.reindex(index=frame.index, columns=frame.columns).fillna(False)
        mask = mask.to_numpy(dtype=bool)
    else:
        mask = np.ones(frame.shape, dtype=bool)

    n_total = int(mask.sum())
    if n_total == 0:
        return NumericReport(ok=False, issues=["universe is empty over the scored window"])

    raw = values.to_numpy(dtype="float64", na_value=np.nan)
    inf_count = int((np.isinf(raw) & mask).sum())

    finite = frame.to_numpy(dtype="float64", na_value=np.nan)
    valid_cells = np.isfinite(finite) & mask
    n_valid = int(valid_cells.sum())
    nan_ratio = 1.0 - (n_valid / n_total)
    coverage = 1.0 - nan_ratio

    daily_valid = valid_cells.sum(axis=1)
    daily_universe = mask.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_coverage = np.where(daily_universe > 0, daily_valid / daily_universe, np.nan)
    tradable = daily_universe > 0
    min_daily_coverage = (
        float(np.nanmin(daily_coverage[tradable])) if tradable.any() else float("nan")
    )

    # Distinct values per day: the check that separates "present" from "informative".
    # The modal share ("tie ratio") comes out of the same np.unique call, because a
    # factor can be rich in distinct values and still park half the cross-section on
    # one of them -- see the module docstring.
    masked = np.where(valid_cells, finite, np.nan)
    scored_rows = daily_valid > 0
    distinct_list: List[int] = []
    tie_list: List[float] = []
    for row in masked[scored_rows]:
        row_valid = row[np.isfinite(row)]
        if row_valid.size == 0:
            distinct_list.append(0)
            tie_list.append(float("nan"))
            continue
        _, counts = np.unique(row_valid, return_counts=True)
        distinct_list.append(int(counts.size))
        tie_list.append(float(counts.max()) / float(row_valid.size))
    distinct_scored = np.asarray(distinct_list, dtype="float64")
    mean_distinct = float(distinct_scored.mean()) if distinct_scored.size else 0.0
    min_distinct = int(distinct_scored.min()) if distinct_scored.size else 0
    n_bad_days = int((distinct_scored < min_distinct_per_day).sum())
    bad_day_ratio = n_bad_days / max(distinct_scored.size, 1)
    constant_days = int((distinct_scored <= 1).sum())

    tie_scored = np.asarray(tie_list, dtype="float64")
    # The ``has_tie`` guard is what keeps nanmean/nanmax off an all-NaN slice; the
    # NaN entries can only come from a row with no valid cell, which ``scored_rows``
    # already excludes, so this is belt-and-braces rather than an expected path.
    has_tie = tie_scored.size > 0 and bool(np.isfinite(tie_scored).any())
    mean_tie_ratio = float(np.nanmean(tie_scored)) if has_tie else float("nan")
    max_daily_tie_ratio = float(np.nanmax(tie_scored)) if has_tie else float("nan")
    with np.errstate(invalid="ignore"):
        n_days_above_tie = int((tie_scored > max_tie_ratio).sum())
    tie_bad_day_ratio = n_days_above_tie / max(tie_scored.size, 1)

    abs_max = float(np.nanmax(np.abs(masked))) if n_valid else float("nan")
    over_limit = int(np.nansum(np.abs(masked) > abs_value_limit))

    with np.errstate(invalid="ignore"):
        daily_std = np.nanstd(masked, axis=1, ddof=1) if masked.shape[1] > 1 else None
    std_mean = (
        float(np.nanmean(daily_std[scored_rows])) if daily_std is not None and scored_rows.any()
        else float("nan")
    )

    n_days = int(scored_rows.sum())

    # ------------------------------------------------------------------ verdict
    if nan_ratio > nan_ratio_limit:
        issues.append(
            f"NaN ratio {nan_ratio:.1%} of the tradable universe exceeds the "
            f"{nan_ratio_limit:.0%} limit"
        )
    if coverage < min_coverage:
        issues.append(f"coverage {coverage:.1%} is below the {min_coverage:.0%} minimum")
    if n_valid == 0:
        issues.append("factor is entirely NaN")
    if inf_count:
        issues.append(f"{inf_count} infinite values (overflow)")
    if over_limit:
        issues.append(
            f"{over_limit} values exceed the magnitude limit {abs_value_limit:g}"
        )
    if n_days < min_days:
        issues.append(f"only {n_days} scored days, need at least {min_days}")
    if constant_days == distinct_scored.size and distinct_scored.size > 0:
        issues.append("factor is constant across the cross-section on every day")
    else:
        if bad_day_ratio > max_bad_day_ratio:
            issues.append(
                f"{bad_day_ratio:.1%} of days have fewer than {min_distinct_per_day} "
                f"distinct values (limit {max_bad_day_ratio:.0%})"
            )
        # Reported separately from the distinct-value check: a gated or
        # constant-filled factor clears that one comfortably and fails this one.
        if tie_bad_day_ratio > max_bad_day_ratio:
            issues.append(
                f"{tie_bad_day_ratio:.1%} of days put over {max_tie_ratio:.0%} of the "
                f"cross-section on a single value (mean tie share {mean_tie_ratio:.1%}, "
                f"limit {max_bad_day_ratio:.0%} of days); a gate or clip that sends the "
                f"inactive side to exactly zero, or a constant fillna, does this"
            )
    if np.isfinite(std_mean) and std_mean == 0.0:
        issues.append("cross-sectional dispersion is exactly zero")

    return NumericReport(
        ok=not issues,
        issues=issues,
        nan_ratio=nan_ratio,
        coverage=coverage,
        min_daily_coverage=min_daily_coverage,
        n_days=n_days,
        n_values=n_valid,
        mean_distinct_per_day=mean_distinct,
        min_distinct_per_day=min_distinct,
        n_days_below_min_distinct=n_bad_days,
        mean_tie_ratio=mean_tie_ratio,
        max_daily_tie_ratio=max_daily_tie_ratio,
        n_days_above_tie_limit=n_days_above_tie,
        inf_count=inf_count,
        over_limit_count=over_limit,
        abs_max=abs_max,
        std_mean=std_mean,
        constant_days_ratio=constant_days / max(distinct_scored.size, 1),
    )

