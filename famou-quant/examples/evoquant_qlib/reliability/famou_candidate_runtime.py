"""Data + training runtime injected into every EvoQuant candidate.

A generated candidate is a self-contained script whose ``train_and_predict``
calls ``run_candidate(split_cfg, params, seed, family=...)``. Everything that
must stay identical across candidates lives here, not in the generated code:

- which features (Alpha158) and which label (the frozen expression)
- that preprocessing statistics are fit on the TRAIN segment only
- that the evaluation window is whatever ``split_cfg`` names, and nothing else

That split matters. If candidates built their own datasets, a candidate could
"improve" by quietly widening its training window or fitting its scaler on the
evaluation segment. Here it can only choose a model family and its
hyperparameters — which is exactly the search space TaskSpec declares.

The same runtime serves the visible evaluator and the sealed gate. Neither the
candidate nor this module knows which split it is looking at: the caller
supplies ``dev_start``/``dev_end``, so the identical frozen code runs under
both protocols. That is what makes a sealed re-run of a promoted candidate
meaningful.

Wire it up by putting this file on the candidate's PYTHONPATH; the evaluator
harness (``qlib_harness.py``) does that automatically.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import numpy as np
import pandas as pd

#: Frozen defaults. Overridable per call through split_cfg so an episode with
#: a different data contract cannot silently reuse another one's cache.
DEFAULT_PROVIDER_URI = os.environ.get(
    "FAMOU_QLIB_PROVIDER_URI",
    "/root/.qlib/qlib_data/cn_data_20260810",
)
DEFAULT_LABEL = "Ref($close, -2) / Ref($close, -1) - 1"

_QLIB_LOCK = threading.Lock()
_QLIB_URI: Optional[str] = None

#: (cache key) -> prepared arrays. Building an Alpha158 handler costs ~30s, and
#: every candidate in a batch asks for the same window, so this is the
#: difference between a usable search loop and an unusable one.
_DATASET_CACHE: "dict[tuple, Dict[str, Any]]" = {}
_CACHE_LOCK = threading.Lock()
_MAX_CACHED = 4


class RuntimeContractError(RuntimeError):
    """The split config is missing something the protocol requires."""


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
            raise RuntimeContractError(
                f"qlib already initialised against {_QLIB_URI}; refusing to "
                f"re-init against {provider_uri} in the same process"
            )
        import qlib

        qlib.init(provider_uri=provider_uri, region="cn")
        _QLIB_URI = provider_uri


# ---------------------------------------------------------------------------
# split handling
# ---------------------------------------------------------------------------


def _purge_end(calendar, start: str, end: str, embargo: int) -> Tuple[str, str]:
    """Trim the last ``embargo`` trading days off a segment.

    The label looks 2 days forward, so without this the last samples of a
    segment carry labels drawn from the next segment — the leak the protocol's
    embargo rule exists to prevent.
    """
    days = [d for d in calendar if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if len(days) <= embargo + 10:
        raise RuntimeContractError(
            f"segment {start}..{end} has {len(days)} trading days, too few to "
            f"purge {embargo}"
        )
    return days[0].strftime("%Y-%m-%d"), days[-1 - embargo].strftime("%Y-%m-%d")


def _resolve_segments(cfg: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    required = ("train_start", "train_end", "dev_start", "dev_end")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise RuntimeContractError(f"split_config missing {missing}")

    from qlib.data import D

    embargo = int(cfg.get("embargo_days", 2))
    calendar = D.calendar(start_time=cfg["train_start"], end_time=cfg["dev_end"])
    train = _purge_end(calendar, cfg["train_start"], cfg["train_end"], embargo)
    dev = _purge_end(calendar, cfg["dev_start"], cfg["dev_end"], embargo)

    # F1 shortens the training window from the LEFT, keeping the most recent
    # data: dropping recent years instead would change the regime being learned
    # and make F1 a different question rather than a cheaper one.
    frac = float(cfg.get("train_fraction", 1.0))
    if 0.0 < frac < 1.0:
        days = [d for d in calendar
                if pd.Timestamp(train[0]) <= d <= pd.Timestamp(train[1])]
        keep = max(60, int(len(days) * frac))
        train = (days[-keep].strftime("%Y-%m-%d"), train[1])
    return {"train": train, "dev": dev}


def _subsample_universe(
    frame: pd.DataFrame, fraction: float, seed: int
) -> pd.DataFrame:
    """Keep a deterministic subset of instruments (F1 only)."""
    if not (0.0 < fraction < 1.0):
        return frame
    names = frame.index.get_level_values("instrument").unique()
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(names, size=max(20, int(len(names) * fraction)), replace=False))
    return frame[frame.index.get_level_values("instrument").isin(keep)]


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------


def _cache_key(cfg: Dict[str, Any], segments: Dict[str, Tuple[str, str]]) -> tuple:
    return (
        cfg.get("provider_uri", DEFAULT_PROVIDER_URI),
        cfg.get("universe", "csi300"),
        cfg.get("label_expression", DEFAULT_LABEL),
        segments["train"],
        segments["dev"],
        round(float(cfg.get("universe_fraction", 1.0)), 4),
    )


#: Materialised arrays are cached to DISK, not just in memory.
#:
#: Every candidate runs in its own subprocess (so a segfaulting candidate
#: cannot take down the worker), which means the in-process cache is cold every
#: single time. Building the Alpha158 handler costs ~45s, and for a formula
#: candidate — whose actual scoring takes ~0.06s — that dwarfs the work by
#: three orders of magnitude and erases the entire reason formula mode exists.
#: Loading the same arrays from an .npz takes ~1-2s.
_DISK_CACHE_DIR = Path(
    os.environ.get("FAMOU_ARRAY_CACHE", "/root/.cache/famou_arrays")
)


def _disk_path(key: tuple) -> Path:
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:20]
    return _DISK_CACHE_DIR / f"arrays_{digest}.npz"


def _save_disk(key: tuple, arrays: Dict[str, Any]) -> None:
    try:
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _disk_path(key)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            x_train=arrays["x_train"], y_train=arrays["y_train"],
            x_valid=arrays["x_valid"], y_valid=arrays["y_valid"],
            x_score=arrays["x_score"], label=arrays["label"],
            dates=np.asarray(arrays["dates"], dtype=object),
            feature_names=np.asarray(arrays["feature_names"], dtype=object),
            segments=np.asarray(
                [arrays["segments"]["train"], arrays["segments"]["dev"]], dtype=object
            ),
        )
        tmp.replace(path)   # atomic: a concurrent reader never sees a partial file
    except Exception:
        pass    # cache is an optimisation; never fail a candidate over it


def _load_disk(key: tuple) -> Optional[Dict[str, Any]]:
    path = _disk_path(key)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as blob:
            segments = blob["segments"]
            return {
                "x_train": blob["x_train"], "y_train": blob["y_train"],
                "x_valid": blob["x_valid"], "y_valid": blob["y_valid"],
                "x_score": blob["x_score"], "label": blob["label"],
                "dates": [str(d) for d in blob["dates"]],
                "feature_names": [str(n) for n in blob["feature_names"]],
                "segments": {"train": tuple(segments[0]), "dev": tuple(segments[1])},
                "n_features": int(blob["x_train"].shape[1]),
            }
    except Exception:
        return None


def _build_arrays(cfg: Dict[str, Any], segments: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
    """Materialise Alpha158 features/labels for train + dev.

    Mirrors ``LGBModel._prepare_data``: DK_L (label-processed) for fitting,
    DK_I (inference) for scoring. RankIC is invariant to the per-day monotone
    transform between them, so both give the same IC.
    """
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    market = cfg.get("universe", "csi300")
    label = cfg.get("label_expression", DEFAULT_LABEL)
    handler = Alpha158(
        instruments=market,
        start_time=segments["train"][0],
        end_time=segments["dev"][1],
        # Protocol rule: every preprocessing statistic is fit on train only.
        fit_start_time=segments["train"][0],
        fit_end_time=segments["train"][1],
        label=[label],
    )
    dataset = DatasetH(
        handler,
        segments={"train": segments["train"], "valid": segments["dev"]},
    )

    out: Dict[str, Any] = {"segments": segments}
    frac = float(cfg.get("universe_fraction", 1.0))

    # --- fitting arrays (DK_L) -----------------------------------------
    df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    df = _subsample_universe(df, frac, seed=17)
    out["x_train"] = np.ascontiguousarray(df["feature"].values, dtype=np.float64)
    out["y_train"] = np.ascontiguousarray(np.squeeze(df["label"].values), dtype=np.float64)

    df_v = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    df_v = _subsample_universe(df_v, frac, seed=17)
    out["x_valid"] = np.ascontiguousarray(df_v["feature"].values, dtype=np.float64)
    out["y_valid"] = np.ascontiguousarray(np.squeeze(df_v["label"].values), dtype=np.float64)

    # --- scoring arrays (DK_I) -----------------------------------------
    x = dataset.prepare("valid", col_set="feature", data_key=DataHandlerLP.DK_I)
    y = dataset.prepare("valid", col_set="label", data_key=DataHandlerLP.DK_I).iloc[:, 0]
    x = _subsample_universe(x, frac, seed=17)
    y = y.loc[x.index]
    # Only the label is dropna'd. LightGBM handles NaN features natively, and a
    # row-wise dropna would empty the frame on any snapshot with an all-NaN
    # column (v1 had exactly that with VWAP0).
    keep = y.notna().to_numpy()
    x, y = x[keep], y[keep]
    order = np.argsort(x.index.get_level_values("datetime").to_numpy(), kind="stable")
    x, y = x.iloc[order], y.iloc[order]

    out["x_score"] = np.ascontiguousarray(x.values, dtype=np.float64)
    out["label"] = np.ascontiguousarray(y.values, dtype=np.float64)
    days = x.index.get_level_values("datetime")
    out["dates"] = [d.strftime("%Y-%m-%d") for d in days]
    out["feature_names"] = [str(c) for c in x.columns]
    out["n_features"] = out["x_train"].shape[1]
    return out


def _get_arrays(cfg: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_qlib(cfg.get("provider_uri", DEFAULT_PROVIDER_URI))
    segments = _resolve_segments(cfg)
    key = _cache_key(cfg, segments)
    with _CACHE_LOCK:
        hit = _DATASET_CACHE.get(key)
    if hit is not None:
        return hit

    arrays = _load_disk(key)
    if arrays is None:
        arrays = _build_arrays(cfg, segments)
        _save_disk(key, arrays)

    with _CACHE_LOCK:
        if len(_DATASET_CACHE) >= _MAX_CACHED:
            _DATASET_CACHE.pop(next(iter(_DATASET_CACHE)))
        _DATASET_CACHE[key] = arrays
    return arrays


# ---------------------------------------------------------------------------
# model families
# ---------------------------------------------------------------------------


def _train_gbdt(arrays: Dict[str, Any], params: Dict[str, Any], seed: int,
                cfg: Dict[str, Any]) -> np.ndarray:
    import lightgbm as lgb

    hp = {
        "objective": params.get("objective", "mse"),
        "learning_rate": float(params.get("learning_rate", 0.08)),
        "num_leaves": int(params.get("num_leaves", 128)),
        "max_depth": int(params.get("max_depth", 8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "subsample": float(params.get("subsample", 0.8)),
        "lambda_l1": float(params.get("lambda_l1", 100.0)),
        "lambda_l2": float(params.get("lambda_l2", 500.0)),
        "min_data_in_leaf": int(params.get("min_data_in_leaf", 50)),
        # The official Alpha158 yaml omits bagging_freq, which makes its
        # `subsample` a no-op. Set it so the parameter actually does something.
        "bagging_freq": int(params.get("bagging_freq", 1)),
        "seed": int(seed),
        "verbose": -1,
        "num_threads": int(cfg.get("num_threads", 4)),
        # Kept off for ALL candidates: when on, LightGBM pre-filters features
        # using each candidate's own min_data_in_leaf, which makes binning
        # candidate-specific and the pool no longer comparable.
        "feature_pre_filter": False,
    }
    rounds = int(cfg.get("max_boost_rounds") or params.get("num_boost_round", 1000))

    train_set = lgb.Dataset(arrays["x_train"], label=arrays["y_train"], params=hp)
    valid_set = lgb.Dataset(arrays["x_valid"], label=arrays["y_valid"], params=hp,
                            reference=train_set)
    booster = lgb.train(
        hp,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return booster.predict(arrays["x_score"])


def _train_mlp(arrays: Dict[str, Any], params: Dict[str, Any], seed: int,
               cfg: Dict[str, Any]) -> np.ndarray:
    import torch
    import torch.nn as nn

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def clean(a: np.ndarray) -> np.ndarray:
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    x_tr = torch.tensor(clean(arrays["x_train"]), dtype=torch.float32)
    y_tr = torch.tensor(clean(arrays["y_train"]), dtype=torch.float32).view(-1, 1)
    x_va = torch.tensor(clean(arrays["x_valid"]), dtype=torch.float32)
    y_va = torch.tensor(clean(arrays["y_valid"]), dtype=torch.float32).view(-1, 1)
    x_sc = torch.tensor(clean(arrays["x_score"]), dtype=torch.float32)

    dims: List[int] = [int(d) for d in params.get("hidden_dims", [256, 128])]
    dropout = float(params.get("dropout", 0.1))
    use_ln = bool(params.get("layer_norm", True))

    layers: List[nn.Module] = []
    prev = x_tr.shape[1]
    for d in dims:
        layers.append(nn.Linear(prev, d))
        if use_ln:
            layers.append(nn.LayerNorm(d))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = d
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get("learning_rate", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-4)),
    )
    loss_fn = nn.MSELoss()
    batch = int(params.get("batch_size", 512))
    epochs = int(params.get("epochs", 20))
    patience = int(params.get("early_stopping", 5))

    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_va, y_va = x_va.to(device), y_va.to(device)
    best, best_state, stale = float("inf"), None, 0
    n = x_tr.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(x_va), y_va).item()
        if val < best - 1e-6:
            best, stale = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, x_sc.shape[0], 8192):
            preds.append(model(x_sc[i:i + 8192].to(device)).cpu().numpy().ravel())
    return np.concatenate(preds)


_FAMILIES = {
    "gbdt": _train_gbdt,
    "linear": _train_gbdt,   # degenerate GBDT config; kept for TaskSpec parity
    "mlp": _train_mlp,
    "temporal_transformer": _train_mlp,
}


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def run_candidate(
    split_cfg: Dict[str, Any],
    params: Dict[str, Any],
    seed: int,
    *,
    family: str = "gbdt",
) -> Dict[str, List[Any]]:
    """Train on the train segment, predict on the dev segment.

    Returns ``{"date": [...], "pred": [...], "label": [...]}`` — the candidate
    computes its own daily RankIC from that, so the metric definition stays in
    the (auditable, generated) candidate rather than hidden in here.
    """
    arrays = _get_arrays(split_cfg)
    trainer = _FAMILIES.get(family)
    if trainer is None:
        raise RuntimeContractError(
            f"unknown model family {family!r}; have {sorted(_FAMILIES)}"
        )
    pred = np.asarray(trainer(arrays, params, seed, split_cfg), dtype=np.float64)
    if pred.shape[0] != arrays["x_score"].shape[0]:
        raise RuntimeContractError(
            f"model returned {pred.shape[0]} predictions for "
            f"{arrays['x_score'].shape[0]} scoring rows"
        )
    return {
        "date": list(arrays["dates"]),
        "pred": pred.tolist(),
        "label": arrays["label"].tolist(),
    }


def dataset_info(split_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Shapes and resolved segments — for harness preflight, not for candidates."""
    arrays = _get_arrays(split_cfg)
    return {
        "segments": arrays["segments"],
        "n_features": arrays["n_features"],
        "n_train_rows": int(arrays["x_train"].shape[0]),
        "n_score_rows": int(arrays["x_score"].shape[0]),
        "n_score_days": len(set(arrays["dates"])),
    }


# ---------------------------------------------------------------------------
# formula candidates (no training)
# ---------------------------------------------------------------------------


def load_features(split_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Factor matrix for the dev segment, with no model fitted.

    This is the entry point for FORMULA candidates: they combine factors
    directly instead of training something on them. The features come from the
    same frozen Alpha158 pipeline and the same train-only preprocessing as the
    model path, so a formula and a GBDT are scored on identical inputs and can
    compete in one pool.
    """
    arrays = _get_arrays(split_cfg)
    return {
        "names": list(arrays["feature_names"]),
        "X": arrays["x_score"],
        "date": list(arrays["dates"]),
        "label": arrays["label"],
    }


def _cross_sectional(values: np.ndarray, bounds: np.ndarray, how: str) -> np.ndarray:
    """Normalise a factor WITHIN each day.

    Cross-sectional, not time-series, and that distinction is the whole game:
    raw Alpha158 columns live on wildly different scales (KMID ~1e-2 against
    VOLUME-derived terms orders of magnitude larger), so a weighted sum of raw
    values is decided entirely by whichever factor happens to be biggest. The
    prediction target is a cross-sectional ranking, so the normalisation has to
    be cross-sectional too.

    Uses only same-day data across names — no lookahead.
    """
    out = np.zeros_like(values)
    for i in range(len(bounds) - 1):
        s = slice(bounds[i], bounds[i + 1])
        block = values[s]
        finite = np.isfinite(block)
        if finite.sum() < 3:
            continue
        clean = np.where(finite, block, 0.0)
        if how == "rank":
            order = np.argsort(np.argsort(np.where(finite, block, -np.inf)))
            n = max(1, finite.sum() - 1)
            out[s] = np.where(finite, order / n - 0.5, 0.0)
        elif how == "sign":
            out[s] = np.sign(clean)
        else:  # zscore
            mu = clean[finite].mean()
            sd = clean[finite].std()
            out[s] = np.where(finite, (clean - mu) / sd, 0.0) if sd > 0 else 0.0
    return out


def _day_bounds(dates: List[str]) -> np.ndarray:
    """Start index of each day; dates are already sorted by construction."""
    codes = np.asarray(pd.factorize(pd.Series(dates), sort=False)[0])
    return np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])


def combine_factors(
    split_cfg: Dict[str, Any],
    terms: List[Dict[str, Any]],
    *,
    n_subperiods: int = 4,
) -> Dict[str, Any]:
    """Score the universe with a weighted factor combination.

    ``terms`` is a list of ``{"feature": name, "weight": float,
    "transform": "zscore"|"rank"|"sign"}``.

    Returns the usual date/pred/label plus ``subperiod_rank_ic``. That last
    field exists because a formula is DETERMINISTIC: re-running it with a
    different seed gives byte-identical output, so multi-seed dispersion — the
    stability evidence the promotion policy relies on for trained models — is
    identically zero and tells you nothing. Splitting the dev window into
    contiguous subperiods gives the real uncertainty for this kind of
    candidate: does the formula hold up across time, or only in one regime?
    """
    data = load_features(split_cfg)
    names, X = data["names"], data["X"]
    dates, label = data["date"], data["label"]
    index = {n: i for i, n in enumerate(names)}

    bounds = _day_bounds(dates)
    score = np.zeros(X.shape[0], dtype=np.float64)
    used: List[str] = []
    for term in terms:
        feature = str(term.get("feature", ""))
        col = index.get(feature)
        if col is None:
            continue  # unknown factor: skip rather than fail the candidate
        weight = float(term.get("weight", 0.0))
        if weight == 0.0:
            continue
        transformed = _cross_sectional(X[:, col], bounds, str(term.get("transform", "zscore")))
        score += weight * transformed
        used.append(feature)

    if not used:
        raise ValueError("formula referenced no known factors")

    # Per-subperiod IC on contiguous day blocks.
    n_days = len(bounds) - 1
    per_sub: List[float] = []
    if n_subperiods > 1 and n_days >= n_subperiods * 5:
        edges = np.linspace(0, n_days, n_subperiods + 1).astype(int)
        for k in range(n_subperiods):
            lo, hi = bounds[edges[k]], bounds[edges[k + 1]]
            ics = _daily_ic(score[lo:hi], label[lo:hi], _day_bounds(dates[lo:hi]))
            if ics:
                per_sub.append(float(np.mean(ics)))

    return {
        "date": list(dates),
        "pred": score.tolist(),
        "label": label.tolist(),
        "subperiod_rank_ic": per_sub,
        "factors_used": used,
    }


def _daily_ic(pred: np.ndarray, label: np.ndarray, bounds: np.ndarray) -> List[float]:
    from scipy.stats import rankdata

    ics: List[float] = []
    for i in range(len(bounds) - 1):
        s = slice(bounds[i], bounds[i + 1])
        p, y = pred[s], label[s]
        keep = np.isfinite(p) & np.isfinite(y)
        if keep.sum() < 3:
            continue
        p, y = p[keep], y[keep]
        if p.std() == 0 or y.std() == 0:
            continue
        ics.append(float(np.corrcoef(rankdata(p), rankdata(y))[0, 1]))
    return ics

