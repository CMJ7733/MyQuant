"""The five predictive-power metrics of §3.4, defined per Appendix B.3.

All five are computed from a wide ``(date x instrument)`` factor frame and an
aligned label frame:

============  =============================================================
IC            mean over t of the cross-sectional Pearson correlation
ICIR          mean(IC_t) / std(IC_t)
RankIC        same as IC but on cross-sectional ranks (Spearman)
RankICIR      mean(RankIC_t) / std(RankIC_t)
MI            mutual information between factor and label
============  =============================================================

Two implementation choices worth stating, because both change what gets selected:

* **Thin cross-sections are dropped.** A day with three valid names produces a
  correlation that is almost surely ±1 and would dominate the IC series. Days
  below ``min_names`` are excluded from the series entirely rather than
  down-weighted.
* **ICIR is not annualised.** §B.3 defines it as ``E[IC_t]/Std[IC_t]`` with no
  ``sqrt(T)`` factor, and the paper's reported values (0.34 at IC 0.059) are
  consistent with that reading. Multiplying by ``sqrt(252)`` would inflate every
  number by 15.9x and break comparison with Table 1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class ICSeries:
    """Per-day correlations plus their summary statistics."""

    values: pd.Series
    mean: float
    std: float
    ir: float
    n_days: int
    positive_ratio: float

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Summary statistics only; the per-day series stays in memory."""
        return {
            "mean": _json_float(self.mean),
            "std": _json_float(self.std),
            "ir": _json_float(self.ir),
            "n_days": self.n_days,
            "positive_ratio": _json_float(self.positive_ratio),
        }


def _json_float(value: float) -> Optional[float]:
    v = float(value)
    return v if np.isfinite(v) else None


def align(
    values: pd.DataFrame,
    label: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both frames to their common dates and instruments."""
    dates = values.index.intersection(label.index)
    columns = values.columns.intersection(label.columns)
    return values.loc[dates, columns], label.loc[dates, columns]


def label_to_wide(label: pd.Series) -> pd.DataFrame:
    """Turn a ``(date, instrument)`` label Series into a wide frame."""
    wide = label.unstack(level=1).sort_index()
    wide.index.name = "date"
    wide.columns.name = "instrument"
    return wide


def _cross_sectional_corr(
    values: pd.DataFrame,
    label: pd.DataFrame,
    method: str,
    min_names: int,
) -> pd.Series:
    """Per-day correlation between factor and label.

    Implemented as a vectorised Pearson correlation over rows (after ranking, for
    Spearman), which is ~50x faster than ``DataFrame.corrwith`` row-wise and
    matters because this runs for every alpha in every generation.

    Returns a Series with **one entry per usable day** -- days below ``min_names``
    or with zero variance are dropped, not zero-filled, so ``len(result)`` is the
    honest sample size behind the mean.
    """
    a = values.to_numpy(dtype="float64", na_value=np.nan)
    b = label.to_numpy(dtype="float64", na_value=np.nan)

    # Pairwise-complete: a name counts on a day only if both the factor and the
    # label are present. Masking both arrays to the same cells is what makes the
    # per-row means below the means of the *same* sample.
    valid = np.isfinite(a) & np.isfinite(b)
    counts = valid.sum(axis=1)

    a = np.where(valid, a, np.nan)
    b = np.where(valid, b, np.nan)

    if method == "spearman":
        # Spearman == Pearson on ranks, so ranking here lets one code path serve both.
        a = _rank_rows(a)
        b = _rank_rows(b)

    # Rows that are entirely NaN (no overlapping names that day) make nanmean warn;
    # they are excluded by the ``counts`` mask below, so suppress the noise rather
    # than special-case the arithmetic.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        a_mean = np.nanmean(a, axis=1, keepdims=True)
        b_mean = np.nanmean(b, axis=1, keepdims=True)
        a_dev = a - a_mean
        b_dev = b - b_mean

        # corr = cov / sqrt(var_a * var_b), all computed per row. nansum over the
        # masked arrays is equivalent to summing over the valid cells only.
        cov = np.nansum(a_dev * b_dev, axis=1)
        a_ss = np.nansum(a_dev * a_dev, axis=1)
        b_ss = np.nansum(b_dev * b_dev, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / np.sqrt(a_ss * b_ss)

    # A thin cross-section produces a correlation that is almost surely +/-1 and
    # would dominate the series mean; drop the day rather than keep the artefact.
    corr = np.where(counts >= min_names, corr, np.nan)
    # Zero variance on either side leaves the correlation undefined, not zero.
    corr = np.where((a_ss > 0) & (b_ss > 0), corr, np.nan)

    return pd.Series(corr, index=values.index, name=f"{method}_ic").dropna()


def _rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Average-tie ranks along each row, preserving NaN.

    ``scipy.stats.rankdata`` cannot ignore NaN per row without a Python loop, so
    the ranking is done by double argsort with the NaNs pushed to the end and
    ties averaged via a grouped mean.

    Ties must be averaged rather than broken arbitrarily: a factor with plateaus (a
    gate that outputs 0 for half the universe, say) would otherwise get a rank order
    determined by column position, which is an artefact of the instrument list.

    The Python-level row loop is the one deliberate inefficiency in the metric path
    (~30 ms per alpha on a 2200 x 750 panel). A fully vectorised tie-average needs a
    scatter-add over group boundaries and was not worth the opacity.
    """
    out = np.full(matrix.shape, np.nan, dtype="float64")
    for i in range(matrix.shape[0]):
        row = matrix[i]
        mask = np.isfinite(row)
        if not mask.any():
            continue
        vals = row[mask]
        # Ordinal ranks 1..n via inverse permutation of the sort order.
        order = np.argsort(vals, kind="stable")
        ranks = np.empty(len(vals), dtype="float64")
        ranks[order] = np.arange(1, len(vals) + 1, dtype="float64")
        # Average ranks within tie groups: walk the sorted values and, whenever a
        # run of equal values ends, overwrite that run with its midpoint rank.
        sorted_vals = vals[order]
        start = 0
        for end in range(1, len(sorted_vals) + 1):
            if end == len(sorted_vals) or sorted_vals[end] != sorted_vals[start]:
                if end - start > 1:
                    # Mean of the 1-based ranks start+1 .. end.
                    mean_rank = (start + end + 1) / 2.0
                    ranks[order[start:end]] = mean_rank
                start = end
        out[i, mask] = ranks
    return out


def ic_series(
    values: pd.DataFrame,
    label: pd.DataFrame,
    method: str = "pearson",
    min_names: int = 10,
) -> ICSeries:
    """Cross-sectional IC series and its summary."""
    series = _cross_sectional_corr(values, label, method=method, min_names=min_names)
    if series.empty:
        return ICSeries(
            values=series,
            mean=float("nan"),
            std=float("nan"),
            ir=float("nan"),
            n_days=0,
            positive_ratio=float("nan"),
        )
    mean = float(series.mean())
    std = float(series.std(ddof=1)) if len(series) > 1 else float("nan")
    ir = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    return ICSeries(
        values=series,
        mean=mean,
        std=std,
        ir=ir,
        n_days=int(len(series)),
        positive_ratio=float((series > 0).mean()),
    )


def mutual_information(
    values: pd.DataFrame,
    label: pd.DataFrame,
    bins: int = 10,
    min_samples: int = 200,
    cross_sectional_rank: bool = True,
    bias_correct: bool = True,
    scale: str = "corr_equivalent",
) -> float:
    """Mutual information between factor and forward return.

    Estimated with a 2-D histogram on ``bins x bins`` quantile bins over the
    pooled sample.  Three choices, each of which changes the number materially:

    * **Cross-sectional ranking first.**  The raw factor's level and scale drift
      across days, so a histogram on raw values measures that drift rather than
      the factor-label relation.  Ranking within each day removes it -- the same
      reason RankIC exists alongside IC.
    * **Quantile bins, not equal-width.**  Equal-width bins on a heavy-tailed
      factor put 99% of the mass in one cell and report MI ~ 0 for a perfectly
      informative signal.
    * **Miller-Madow bias correction.**  A plug-in histogram estimate is biased
      upward by about ``(B_xy - B_x - B_y + 1) / 2N``.  Pooled over 300 names x
      500 days that bias is small, but the correction is what makes the estimate
      track the truth rather than the bin count.

    Validated against the closed form for a bivariate Gaussian,
    ``MI = -0.5 ln(1 - rho^2)``, on the synthetic panel (300 names, 500 days,
    ``bins=10``):

    ======  ==========  =============  =============
    RankIC  MI (theory) MI (this fn)   pure noise
    ======  ==========  =============  =============
    0.076   0.00287     0.00407        0.00127
    0.181   0.01694     0.01778        --
    0.357   0.06814     0.06825        --
    ======  ==========  =============  =============

    Scale
    -----
    ``scale='nats'`` returns MI directly.  ``scale='corr_equivalent'`` (default)
    inverts the Gaussian relation to ``sqrt(1 - exp(-2 MI))``, giving a number on
    the same [0, 1) scale as |IC|.

    The default is the correlation-equivalent scale for a specific reason: the
    paper sets an absolute MI floor of 0.02 (§A.4) but never states its estimator,
    and MI in nats is not comparable across estimators -- the table above shows a
    genuinely predictive factor sits at 0.004 nats, so a 0.02-nat floor would
    reject every real alpha.  On the correlation-equivalent scale that same factor
    reads 0.090 and the floor behaves as a floor.  Set ``scale='nats'`` and lower
    ``fitness.qualified_min_bounds['mi']`` to work in nats instead.
    """
    a = values
    b = label
    if cross_sectional_rank:
        a = values.rank(axis=1, pct=True)
        b = label.rank(axis=1, pct=True)

    x = a.to_numpy(dtype="float64", na_value=np.nan).ravel()
    y = b.to_numpy(dtype="float64", na_value=np.nan).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < min_samples:
        return float("nan")

    x_edges = _quantile_edges(x, bins)
    y_edges = _quantile_edges(y, bins)
    if x_edges is None or y_edges is None:
        return 0.0

    joint, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    total = joint.sum()
    if total <= 0:
        return 0.0

    pxy = joint / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    denom = px * py

    nz = (pxy > 0) & (denom > 0)
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / denom[nz])))

    if bias_correct:
        n_x = int((px > 0).sum())
        n_y = int((py > 0).sum())
        n_xy = int((pxy > 0).sum())
        mi -= (n_xy - n_x - n_y + 1) / (2.0 * total)

    mi = max(mi, 0.0)

    if scale == "nats":
        return mi
    if scale == "corr_equivalent":
        return float(np.sqrt(max(1.0 - np.exp(-2.0 * mi), 0.0)))
    raise ValueError(f"unknown mi scale '{scale}'; use 'nats' or 'corr_equivalent'")



def _quantile_edges(sample: np.ndarray, bins: int) -> Optional[np.ndarray]:
    """Unique quantile bin edges, or ``None`` when the sample is degenerate."""
    qs = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(sample, qs))
    if len(edges) < 3:
        return None
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    return edges
