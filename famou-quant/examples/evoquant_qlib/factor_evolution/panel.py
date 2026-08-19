"""Raw OHLCV panel for factor evolution, as day x instrument matrices.

Why a 2-D matrix and not qlib's long frame
------------------------------------------
The candidate code is written by an LLM, and the single most dangerous thing it
can do is look at the future. A ``(n_days, n_instruments)`` matrix makes the time
axis explicit: row ``t`` is one trading day, and "only use rows <= t" is a rule
you can state in one sentence and check mechanically (see ``evaluator.py``'s
causality probe). A MultiIndex long frame hides the same rule behind groupby
semantics that are easy to get subtly wrong.

Two labels, deliberately
------------------------
``label_norm`` is the protocol label (``Ref($close,-2)/Ref($close,-1)-1``) after
cross-sectional z-scoring, matching ``splits_v2.yaml``. RankIC is invariant to a
per-day monotone transform, so it is the right input for IC-family metrics.

``fwd_ret_raw`` is the SAME expression without normalisation. Sharpe is computed
from realised portfolio returns, and a z-scored "return" has no units — a Sharpe
built on it would be a meaningless number that still looks plausible. So the raw
series is carried alongside rather than reconstructed downstream.

Caching
-------
Every candidate runs in its own subprocess, so an in-process cache is cold every
single time. Building this panel costs ~30-60s; loading the ``.npz`` costs ~1s.
The scheme (atomic tmp-then-replace, never fail a candidate over a cache error)
follows ``reliability/famou_candidate_runtime.py``.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

#: Frozen snapshot from splits_v2.yaml. Overridable for tests only.
DEFAULT_PROVIDER_URI = os.environ.get(
    "FAMOU_QLIB_PROVIDER", "/root/.qlib/qlib_data/cn_data_20260810"
)

#: Protocol label: buy at next open, hold one day (see splits_v2.yaml `label`).
DEFAULT_LABEL = "Ref($close, -2) / Ref($close, -1) - 1"

#: Raw fields handed to the candidate. All verified present in the snapshot.
FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume", "vwap", "amount")

_CACHE_DIR = Path(
    os.environ.get("FAMOU_PANEL_CACHE", "/root/.cache/famou_panels")
)

_QLIB_URI: Optional[str] = None
_QLIB_LOCK = threading.Lock()


class PanelError(RuntimeError):
    """The panel could not be built or violates its own contract."""


@dataclass
class Panel:
    """Raw market data as aligned ``(n_days, n_instruments)`` matrices.

    Every field shares one index: ``dates[t]`` and ``instruments[j]`` address
    the same cell in every matrix. NaN means "not listed / not traded that day"
    and is expected — CSI300 membership changes over the window.
    """

    dates: List[str]
    instruments: List[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray
    amount: np.ndarray

    #: Index of the first row belonging to the dev segment. Rows before it are
    #: train history, present only so rolling windows are warm on dev day 0.
    dev_start_row: int = 0

    label_norm: Optional[np.ndarray] = None
    fwd_ret_raw: Optional[np.ndarray] = None

    @property
    def n_days(self) -> int:
        return len(self.dates)

    @property
    def n_instruments(self) -> int:
        return len(self.instruments)

    def without_labels(self) -> "Panel":
        """A copy carrying market data only -- no ``label_norm``/``fwd_ret_raw``.

        This is what a candidate is handed. The labels are the answer key: a
        candidate that returned ``panel.label_norm`` would score a perfect RankIC
        of 1.0. The causality probe does catch that (the truncated panel has no
        labels, so the two runs disagree), but relying on it means the only thing
        between the search and the answer is a side effect of :meth:`head`, and
        the failure surfaces as a confusing "wrong shape" message. Removing the
        labels from the object entirely makes the leak structurally impossible
        instead of merely detected.

        The parent process keeps its own labelled copy for scoring.
        """
        return Panel(
            dates=list(self.dates),
            instruments=list(self.instruments),
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            vwap=self.vwap,
            amount=self.amount,
            dev_start_row=self.dev_start_row,
            label_norm=None,
            fwd_ret_raw=None,
        )

    def head(self, n_rows: int) -> "Panel":
        """A copy truncated to the first ``n_rows`` days.

        This is what makes the causality probe possible: re-running a candidate
        on ``panel.head(t + 1)`` must reproduce its own row ``t``, because a
        causal factor cannot depend on data that does not exist yet.
        """
        if n_rows < 1 or n_rows > self.n_days:
            raise PanelError(f"head({n_rows}) out of range for {self.n_days} days")
        return Panel(
            dates=self.dates[:n_rows],
            instruments=list(self.instruments),
            open=self.open[:n_rows],
            high=self.high[:n_rows],
            low=self.low[:n_rows],
            close=self.close[:n_rows],
            volume=self.volume[:n_rows],
            vwap=self.vwap[:n_rows],
            amount=self.amount[:n_rows],
            dev_start_row=min(self.dev_start_row, n_rows),
            # Labels are deliberately NOT carried into the truncated copy: the
            # candidate must never see them, and the probe only compares factors.
            label_norm=None,
            fwd_ret_raw=None,
        )


# ---------------------------------------------------------------------------
# qlib init
# ---------------------------------------------------------------------------


def _ensure_qlib(provider_uri: str) -> None:
    """Initialise qlib once per process.

    qlib keeps global module state, so a second init against a different
    directory silently repoints every later query. Refuse instead — a run that
    mixed two data snapshots would produce results no hash could detect.
    """
    global _QLIB_URI
    with _QLIB_LOCK:
        if _QLIB_URI == provider_uri:
            return
        if _QLIB_URI is not None:
            raise PanelError(
                f"qlib already initialised against {_QLIB_URI}; refusing to "
                f"re-init against {provider_uri} in the same process"
            )
        import qlib

        qlib.init(provider_uri=provider_uri, region="cn")
        _QLIB_URI = provider_uri


def _purge_end(calendar, start: str, end: str, embargo: int) -> Tuple[str, str]:
    """Trim the last ``embargo`` trading days off a segment.

    The label looks 2 days forward, so without this the last samples of a
    segment carry labels drawn from the next segment — the leak the protocol's
    embargo rule exists to prevent. Same helper as the reliability runtime.
    """
    days = [d for d in calendar if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(days) <= embargo + 10:
        raise PanelError(
            f"segment {start}..{end} has {len(days)} trading days, "
            f"too few to purge {embargo}"
        )
    return days[0].strftime("%Y-%m-%d"), days[-1 - embargo].strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def _cache_key(cfg: Dict[str, Any]) -> tuple:
    return (
        "panel_v1",
        cfg.get("provider_uri"),
        cfg.get("universe"),
        cfg.get("train_start"),
        cfg.get("train_end"),
        cfg.get("dev_start"),
        cfg.get("dev_end"),
        int(cfg.get("embargo_days", 2)),
        cfg.get("label_expression"),
        FIELDS,
    )


def _cache_path(key: tuple) -> Path:
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:20]
    return _CACHE_DIR / f"panel_{digest}.npz"


def _save_cache(key: tuple, panel: Panel) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(key)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            dates=np.asarray(panel.dates, dtype=object),
            instruments=np.asarray(panel.instruments, dtype=object),
            dev_start_row=np.asarray(panel.dev_start_row),
            label_norm=panel.label_norm,
            fwd_ret_raw=panel.fwd_ret_raw,
            **{f: getattr(panel, f) for f in FIELDS},
        )
        tmp.replace(path)  # atomic: a concurrent reader never sees a partial file
    except Exception:
        pass  # cache is an optimisation; never fail a candidate over it


def _load_cache(key: tuple) -> Optional[Panel]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as blob:
            return Panel(
                dates=[str(d) for d in blob["dates"]],
                instruments=[str(s) for s in blob["instruments"]],
                dev_start_row=int(blob["dev_start_row"]),
                label_norm=blob["label_norm"],
                fwd_ret_raw=blob["fwd_ret_raw"],
                **{f: blob[f] for f in FIELDS},
            )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _pivot(df: pd.DataFrame, column: str, dates: List[str], instruments: List[str]) -> np.ndarray:
    """One long-frame column -> a (n_days, n_instruments) float64 matrix."""
    wide = df[column].unstack(level="instrument")
    wide = wide.reindex(index=pd.to_datetime(dates), columns=instruments)
    return np.ascontiguousarray(wide.to_numpy(dtype=np.float64))


def _cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    """Per-row z-score, ignoring NaN. Matches the protocol's CSZScoreNorm."""
    out = np.full_like(values, np.nan)
    for t in range(values.shape[0]):
        row = values[t]
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        mean = row[finite].mean()
        std = row[finite].std()
        if not np.isfinite(std) or std == 0:
            continue
        out[t, finite] = (row[finite] - mean) / std
    return out


def _build(cfg: Dict[str, Any]) -> Panel:
    """Query qlib for raw fields plus both label variants."""
    provider_uri = cfg.get("provider_uri") or DEFAULT_PROVIDER_URI
    _ensure_qlib(provider_uri)

    from qlib.data import D

    universe = cfg.get("universe", "csi300")
    embargo = int(cfg.get("embargo_days", 2))
    label_expr = cfg.get("label_expression") or DEFAULT_LABEL

    calendar = D.calendar(start_time=cfg["train_start"], end_time=cfg["dev_end"])
    train = _purge_end(calendar, cfg["train_start"], cfg["train_end"], embargo)
    dev = _purge_end(calendar, cfg["dev_start"], cfg["dev_end"], embargo)

    # One query spanning train..dev. The train rows exist ONLY so a rolling
    # window is already warm on the first dev day; nothing is scored on them.
    instruments = D.instruments(market=universe)
    raw_fields = [f"${f}" for f in FIELDS]
    frame = D.features(
        instruments, raw_fields + [label_expr],
        start_time=train[0], end_time=dev[1], freq="day",
    )
    if frame is None or frame.empty:
        raise PanelError(f"qlib returned no data for {universe} {train[0]}..{dev[1]}")

    frame = frame.sort_index()
    dates = sorted({d.strftime("%Y-%m-%d")
                    for d in frame.index.get_level_values("datetime")})
    tickers = sorted(set(frame.index.get_level_values("instrument")))

    # qlib yields (instrument, datetime); swap so unstack gives days x instruments.
    frame = frame.swaplevel().sort_index()

    matrices = {f: _pivot(frame, col, dates, tickers)
                for f, col in zip(FIELDS, raw_fields)}
    fwd_ret_raw = _pivot(frame, label_expr, dates, tickers)

    dev_start_row = int(np.searchsorted(np.asarray(dates), dev[0], side="left"))

    return Panel(
        dates=dates, instruments=tickers, dev_start_row=dev_start_row,
        label_norm=_cross_sectional_zscore(fwd_ret_raw),
        fwd_ret_raw=fwd_ret_raw,
        **matrices,
    )


def load_panel(cfg: Dict[str, Any]) -> Panel:
    """Build (or load from cache) the panel described by ``cfg``.

    ``cfg`` needs ``train_start``/``train_end``/``dev_start``/``dev_end`` and
    optionally ``universe``, ``embargo_days``, ``label_expression``,
    ``provider_uri``.
    """
    missing = [k for k in ("train_start", "train_end", "dev_start", "dev_end")
               if not cfg.get(k)]
    if missing:
        raise PanelError(f"panel config missing {missing}")

    key = _cache_key({**cfg, "provider_uri": cfg.get("provider_uri") or DEFAULT_PROVIDER_URI})
    cached = _load_cache(key)
    if cached is not None:
        return cached

    panel = _build(cfg)
    _save_cache(key, panel)
    return panel
