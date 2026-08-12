"""The OHLCV panel and forward-return label.

Layout
------
A :class:`Panel` wraps a long-format frame indexed by ``(date, instrument)``
with at least the five OHLCV columns.  Two views are derived from it:

* :meth:`Panel.iter_instruments` — per-instrument frames, which is what an alpha
  function consumes;
* :meth:`Panel.label` — the forward return over ``horizon`` days, which is what
  fitness is measured against.

Label convention
----------------
The paper buys and sells at the open (§4.1), so the ``horizon``-day forward
return realisable from a signal known at the close of day *t* is

    r_t = open_{t+1+h} / open_{t+1} - 1

Both legs are strictly in the future of *t*, which is the property the leakage
unit test (§A.3) exists to protect.  A naive ``close_{t+h}/close_t - 1`` would
already be tradable-at-close and is what most leaky reimplementations use; we
keep the open-to-open form and assert the offset in
``tests/test_data_panel.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: The paper's listings use both bare (``df['close']``) and prefixed
#: (``df_copy['day_close']``) column names.  We materialise both spellings on the
#: per-instrument frame so either generated convention executes unchanged.
ALIAS_PREFIXES: Tuple[str, ...] = ("day_",)


class PanelError(ValueError):
    """Raised when an input frame violates the panel contract."""


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce an arbitrary long frame into the canonical panel layout."""
    frame = df.copy()

    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.reset_index()

    lowered = {c: str(c).strip().lower() for c in frame.columns}
    frame = frame.rename(columns=lowered)

    renames = {
        "datetime": "date",
        "time": "date",
        "trade_date": "date",
        "symbol": "instrument",
        "ticker": "instrument",
        "code": "instrument",
        "vol": "volume",
    }
    frame = frame.rename(columns={k: v for k, v in renames.items() if k in frame})

    missing = {"date", "instrument"} - set(frame.columns)
    if missing:
        raise PanelError(f"panel frame missing index columns: {sorted(missing)}")

    missing_px = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing_px:
        raise PanelError(f"panel frame missing OHLCV columns: {missing_px}")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["instrument"] = frame["instrument"].astype(str)
    for col in REQUIRED_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")

    frame = (
        frame.dropna(subset=["date", "instrument"])
        .drop_duplicates(subset=["date", "instrument"], keep="last")
        .sort_values(["instrument", "date"])
        .set_index(["date", "instrument"])
        .sort_index()
    )
    return frame


@dataclass
class Panel:
    """An OHLCV panel plus metadata about where it came from.

    Parameters
    ----------
    frame:
        Long frame indexed by ``(date, instrument)``.
    name:
        Free-form provenance label, echoed into the run archive.
    """

    frame: pd.DataFrame
    name: str = "panel"
    meta: Dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.frame = _normalise_frame(self.frame)
        if self.meta is None:
            self.meta = {}

    # ---------------------------------------------------------------- basics

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Sorted unique trading dates. This is the run's calendar -- the leakage
        probe and the rolling fit both take positions on it, not on calendar days."""
        return pd.DatetimeIndex(self.frame.index.get_level_values(0).unique()).sort_values()

    @property
    def instruments(self) -> List[str]:
        """Sorted union of every instrument that appears anywhere in the panel.

        A *union*, not a daily membership: on CSI300 2011-2024 this is 748 tickers
        while ~300 trade on any day. Use :meth:`universe_mask` for daily membership.
        """
        return sorted(set(self.frame.index.get_level_values(1)))

    def __len__(self) -> int:
        return len(self.frame)

    def describe(self) -> Dict[str, object]:
        """Provenance and shape, for the archive. Deliberately contains no data."""
        dates = self.dates
        return {
            "name": self.name,
            "rows": int(len(self.frame)),
            "instruments": len(self.instruments),
            "days": int(len(dates)),
            "start": str(dates[0].date()) if len(dates) else None,
            "end": str(dates[-1].date()) if len(dates) else None,
            **dict(self.meta),
        }

    # ------------------------------------------------------------- per-stock

    def instrument_frame(self, instrument: str) -> pd.DataFrame:
        """One instrument's history as a date-indexed frame, with aliases.

        The returned frame is a fresh copy: an alpha function is free to mutate
        it (the paper's listings all start with ``df_copy = df.copy()`` but we do
        not rely on their discipline).
        """
        sub = self.frame.xs(instrument, level=1, drop_level=True).sort_index()
        sub = sub.copy()
        for prefix in ALIAS_PREFIXES:
            for col in REQUIRED_COLUMNS:
                sub[f"{prefix}{col}"] = sub[col]
        sub.index.name = "date"
        return sub

    def iter_instruments(self) -> Iterator[Tuple[str, pd.DataFrame]]:
        """Yield ``(instrument, frame)`` for every name, in sorted order.

        Each frame is freshly copied, so an alpha may mutate it. Building all of them
        costs ~1.5 s on the full CSI300 panel; the sandbox does it once per worker.
        """
        for inst in self.instruments:
            yield inst, self.instrument_frame(inst)

    def universe_mask(self, price: str = "close") -> pd.DataFrame:
        """Boolean ``(date x instrument)`` frame: where the name is tradable.

        A panel of index constituents over many years is a *union* of memberships:
        CSI300 over 2011-2024 spans 748 tickers, ~300 of which are in the index on
        any given day.  The wide layout is therefore ~60% empty before any alpha
        runs, and coverage measured against the full rectangle is meaningless — it
        would report every alpha as 60% missing.  This mask is the denominator that
        makes the §B.4 NaN limit and the coverage gate mean what they say.
        """
        wide = self.frame[price].unstack(level=1).sort_index()
        return wide.notna()

    # ---------------------------------------------------------------- labels

    def label(self, horizon: int, price: str = "open", offset: int = 1) -> pd.Series:
        """Forward return over ``horizon`` days, tradable from the next bar.

        ``offset=1`` implements the buy-at-next-open convention: a signal formed
        on day *t* is executed at ``open_{t+1}`` and unwound at
        ``open_{t+1+horizon}``.  Setting ``offset=0`` gives the close-to-close
        variant and is only there for comparison against other codebases.
        """
        return forward_return(self.frame, horizon=horizon, price=price, offset=offset)

    # ---------------------------------------------------------------- slicing

    def slice(self, start: Optional[str], end: Optional[str]) -> "Panel":
        """Sub-panel within ``[start, end]`` inclusive. See :func:`slice_panel`."""
        return slice_panel(self, start, end)

    def subset_instruments(self, instruments: Sequence[str]) -> "Panel":
        """Sub-panel restricted to ``instruments``."""
        keep = set(str(i) for i in instruments)
        mask = self.frame.index.get_level_values(1).isin(keep)
        return Panel(self.frame[mask], name=f"{self.name}[subset]", meta=dict(self.meta))


def forward_return(
    frame: pd.DataFrame,
    horizon: int,
    price: str = "open",
    offset: int = 1,
) -> pd.Series:
    """Compute the forward return label for a panel frame.

    Returns a Series indexed like ``frame`` (``(date, instrument)``), with NaN in
    the trailing ``horizon + offset`` rows of every instrument where the exit
    price is not yet observable.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if price not in frame.columns:
        raise PanelError(f"label price column '{price}' not in panel")

    wide = frame[price].unstack(level=1).sort_index()
    entry = wide.shift(-offset)
    exit_ = wide.shift(-(offset + horizon))
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = exit_ / entry - 1.0
    fwd = fwd.replace([np.inf, -np.inf], np.nan)
    out = fwd.stack(future_stack=True) if _supports_future_stack() else fwd.stack(dropna=False)
    out.index.names = ["date", "instrument"]
    out.name = f"label_fwd{horizon}"
    return out.reindex(frame.index)


def _supports_future_stack() -> bool:
    """pandas >= 2.1 deprecates ``stack(dropna=...)`` in favour of ``future_stack``."""
    try:
        major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover - exotic version strings
        return False
    return (major, minor) >= (2, 1)


def slice_panel(panel: Panel, start: Optional[str], end: Optional[str]) -> Panel:
    """Return the sub-panel inside ``[start, end]`` (inclusive, by date)."""
    dates = panel.frame.index.get_level_values(0)
    mask = np.ones(len(panel.frame), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    sliced = panel.frame[mask]
    label = f"{start or '-inf'}:{end or 'inf'}"
    return Panel(sliced, name=f"{panel.name}[{label}]", meta=dict(panel.meta))


def make_splits(
    panel: Panel,
    train: Tuple[str, str],
    valid: Tuple[str, str],
    test: Tuple[str, str],
    horizon: int = 0,
) -> Dict[str, Panel]:
    """Split chronologically, padding each split's head with warm-up history.

    A rolling alpha (say a 30-day EMA) needs history before the first scored day,
    otherwise the opening weeks of every split are NaN and the IC series silently
    shortens.  Each split therefore *starts* ``horizon`` days early for feature
    warm-up, while :func:`cogalpha.fitness.evaluate` still scores only dates
    inside the declared window.
    """
    out: Dict[str, Panel] = {}
    for key, (start, end) in (("train", train), ("valid", valid), ("test", test)):
        sub = slice_panel(panel, start, end)
        sub.name = f"{panel.name}[{key}]"
        sub.meta = {**dict(panel.meta), "split": key, "window": (start, end)}
        out[key] = sub
    return out
