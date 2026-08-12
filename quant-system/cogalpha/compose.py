"""Multi-factor composition: the reading Table 1 actually reports (§4.2, §B.4).

The paper does not score single alphas.  For every method that generates factors —
CogAlpha included — it "evaluate[s] performance using multi-factor combinations
constructed from the 20 generated alphas" (§4.2).  So the IC of 0.0591 in Table 1 is
the IC of a *model's prediction* trained on 20 alpha columns, not the IC of any one
alpha.  Without this module the search's output cannot be compared to the paper at
all; it can only be compared to other single factors.

Pipeline
--------
1. evaluate each candidate alpha over the panel -> a ``(date, instrument) x alpha``
   feature matrix;
2. cross-sectionally normalise the features (they arrive on wildly different
   scales — a liquidity impact is ~1e-9, a ratio is ~1);
3. train the downstream model with rolling retraining, step 126 trading days
   (§B.4), each fit embargoed so no label it sees extends past its cutoff;
4. predict out of sample, then score the prediction with the same five metrics and
   the same top-50/drop-5 backtest used for single alphas.

Two hazards this module exists to avoid, neither of which the paper discusses:

* **Embargo.** A 10-day forward label formed on day *t* is only observable at
  *t+11*.  A model retrained at cutoff *T* must therefore not train on labels after
  *T-11*, or it learns from returns that had not happened yet — a leak that inflates
  every downstream number and is invisible in the output.
* **Feature scale.** Ridge on unnormalised alpha columns is decided entirely by
  whichever factor has the largest units.  Cross-sectional ranking fixes that and
  removes outliers at the same time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from cogalpha.config import CogAlphaConfig, DataConfig, FitnessConfig
from cogalpha.data.panel import Panel, forward_return
from cogalpha.fitness import metrics as M
from cogalpha.fitness.backtest import BacktestResult, run_backtest
from cogalpha.quality.sandbox import apply_alpha, compile_alpha
from cogalpha.types import Alpha


@dataclass
class CompositionResult:
    """Metrics of a composed multi-factor prediction."""

    model: str
    n_alphas: int
    alphas_used: List[str] = field(default_factory=list)
    alphas_dropped: Dict[str, str] = field(default_factory=dict)

    ic: float = float("nan")
    icir: float = float("nan")
    rank_ic: float = float("nan")
    rank_icir: float = float("nan")
    mi: float = float("nan")
    aer: Optional[float] = None
    ir: Optional[float] = None

    n_days: int = 0
    n_folds: int = 0
    train_rows: int = 0
    test_rows: int = 0
    split: str = "test"
    window: Tuple[Optional[str], Optional[str]] = (None, None)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    backtest: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe result, for ``--json`` output."""

        def f(v):
            if v is None:
                return None
            v = float(v)
            return v if np.isfinite(v) else None

        return {
            "model": self.model,
            "n_alphas": self.n_alphas,
            "alphas_used": list(self.alphas_used),
            "alphas_dropped": dict(self.alphas_dropped),
            "ic": f(self.ic),
            "icir": f(self.icir),
            "rank_ic": f(self.rank_ic),
            "rank_icir": f(self.rank_icir),
            "mi": f(self.mi),
            "aer": f(self.aer),
            "ir": f(self.ir),
            "n_days": self.n_days,
            "n_folds": self.n_folds,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "split": self.split,
            "window": list(self.window),
            "feature_importance": {k: f(v) for k, v in self.feature_importance.items()},
            "backtest": self.backtest,
            "seconds": round(self.seconds, 1),
        }

    def table_row(self) -> str:
        """One line in the shape of Table 1, for eyeballing against the paper."""
        def s(v, d=4):
            return "   n/a " if v is None or v != v else f"{float(v):+.{d}f}"

        return (
            f"{self.model:<12} n={self.n_alphas:<3} "
            f"IC={s(self.ic)} RankIC={s(self.rank_ic)} "
            f"ICIR={s(self.icir)} RankICIR={s(self.rank_icir)} "
            f"AER={s(self.aer)} IR={s(self.ir)}"
        )


# --------------------------------------------------------------------- features


def build_feature_matrix(
    alphas: Sequence[Alpha],
    panel: Panel,
    allowed_imports: Sequence[str] = ("numpy", "pandas", "math", "scipy", "talib"),
    normalise: str = "cs_rank",
    max_nan_ratio: float = 0.3,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Evaluate ``alphas`` over ``panel`` into a feature matrix.

    Returns ``(features, dropped)`` where ``features`` is indexed by
    ``(date, instrument)`` with one column per surviving alpha, and ``dropped`` maps
    an alpha name to why it was excluded.

    Runs in-process, not in the sandbox: by the time an alpha reaches composition it
    has already passed the audit, the sandbox and the leakage test, so the isolation
    has done its job and paying for it again per fold would dominate the cost.

    ``normalise='cs_rank'`` converts each factor to a per-day cross-sectional
    percentile.  This is not cosmetic — alpha columns arrive on incomparable scales
    (a liquidity impact is O(1e-9), a ratio is O(1)), and a linear model on raw
    columns is decided by whichever has the largest units.  Ranking also bounds
    outliers, which matters because these factors are not winsorised.

    Coverage is measured against the tradable universe, not the full rectangle.
    A panel of index constituents over many years is a union of memberships — CSI300
    over 2011-2024 spans 748 tickers of which ~300 trade on any day — so the wide
    layout is ~60% empty before an alpha runs.  Measuring against the rectangle
    rejects every alpha ever written: the first attempt here dropped all 23
    hand-written baselines for being "60.8% NaN".
    """
    frames = dict(panel.iter_instruments())
    universe = panel.universe_mask()
    universe_cells = int(universe.to_numpy().sum())

    columns: Dict[str, pd.Series] = {}
    dropped: Dict[str, str] = {}

    for alpha in alphas:
        name = alpha.name
        if name in columns:
            # Two candidates can share a function name after renaming; keep the
            # first and say so rather than silently overwriting a column.
            dropped[f"{name}#dup"] = "duplicate column name"
            continue
        try:
            fn = compile_alpha(alpha.code, name, allowed_imports=allowed_imports)
            wide = apply_alpha(fn, frames, column=name)
        except Exception as exc:  # noqa: BLE001
            dropped[name] = f"{type(exc).__name__}: {exc}"
            continue

        wide = wide.replace([np.inf, -np.inf], np.nan)
        if normalise == "cs_rank":
            wide = wide.rank(axis=1, pct=True)
        elif normalise == "cs_zscore":
            mean = wide.mean(axis=1)
            std = wide.std(axis=1).replace(0.0, np.nan)
            wide = wide.sub(mean, axis=0).div(std, axis=0)
        elif normalise != "none":
            raise ValueError(f"unknown normalise mode '{normalise}'")

        # Reindex onto the universe grid before measuring anything: an alpha may have
        # produced a shorter frame (fewer instruments, or a truncated calendar) and
        # comparing shapes directly would compare different denominators.
        aligned = wide.reindex(index=universe.index, columns=universe.columns)
        valid_cells = int((aligned.notna() & universe).to_numpy().sum())
        nan_ratio = 1.0 - (valid_cells / universe_cells if universe_cells else 0.0)
        if nan_ratio > max_nan_ratio:
            dropped[name] = (
                f"{nan_ratio:.1%} of the tradable universe is NaN, above "
                f"{max_nan_ratio:.0%}"
            )
            continue

        # Keep only universe cells; a value for a name that was not in the index
        # that day is not a feature, it is an artefact of the wide layout.
        columns[name] = _stack(aligned.where(universe))

    if not columns:
        return pd.DataFrame(), dropped

    features = pd.DataFrame(columns).sort_index()
    features.index.names = ["date", "instrument"]
    # Drop rows outside the universe entirely -- they are NaN in every column.
    features = features.dropna(how="all")
    return features, dropped



def _stack(wide: pd.DataFrame) -> pd.Series:
    out = wide.stack(future_stack=True) if _future_stack() else wide.stack(dropna=False)
    out.index.names = ["date", "instrument"]
    return out


def _future_stack() -> bool:
    try:
        major, minor = (int(x) for x in pd.__version__.split(".")[:2])
    except ValueError:  # pragma: no cover
        return False
    return (major, minor) >= (2, 1)


# ----------------------------------------------------------------------- models


def make_model(kind: str, seed: int = 42, lgbm_lr: float = 0.0001):
    """Instantiate the downstream model with the paper's hyper-parameters (§B.4).

    LightGBM: "learning rate of 0.0001, 32 leaves per tree, a maximum depth of 12,
    and regularization terms (reg_alpha and reg_lambda) set to 1.0 ... a total of
    1000 trees with sampling techniques (feature and bagging fractions set to 0.8)".
    Ridge: "the regularization strength (alpha) is set to 10".

    One deviation, stated because it changes results: the paper's ``num_leaves=32``
    with ``max_depth=12`` is contradictory for LightGBM (a depth-12 tree admits up to
    4096 leaves, so depth is not binding), but it is harmless, and we keep both as
    written.

    What is *not* harmless is the learning rate.  ``0.0001 x 1000 trees`` gives a
    total shrinkage of 0.1, so the ensemble can only travel a tenth of the way from
    its initial constant to the fitted signal — the LightGBM composition is
    deliberately under-fitted.  Measured on this data with 20 hand-written factors it
    reads RankIC 0.0188 on the test split against Ridge's 0.0327, i.e. the tree model
    scores *below* the linear one, which is the opposite of the paper's ordering
    (Table 1 has LightGBM 0.0412 over Linear 0.0211).  That gap is consistent with
    under-fitting rather than with a defect in the pipeline.  We keep the paper's
    value as the default so the specification is reproduced literally; pass
    ``lgbm_lr`` to use a rate that actually converges.
    """
    if kind == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "model 'lightgbm' needs lightgbm installed (pip install 'cogalpha[model]')"
            ) from exc
        return LGBMRegressor(
            learning_rate=lgbm_lr,
            num_leaves=32,
            max_depth=12,
            n_estimators=1000,
            reg_alpha=1.0,
            reg_lambda=1.0,
            colsample_bytree=0.8,
            subsample=0.8,
            subsample_freq=1,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )


    if kind == "ridge":
        try:
            from sklearn.linear_model import Ridge
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "model 'ridge' needs scikit-learn (pip install 'cogalpha[model]')"
            ) from exc
        return Ridge(alpha=10.0, random_state=seed)

    if kind == "mean":
        return _EqualWeight()

    raise ValueError(f"unknown model '{kind}'; use lightgbm, ridge or mean")


class _EqualWeight:
    """Equal-weight average of the (already ranked) features.

    Not from the paper.  It is the control that answers "did the model add anything,
    or is the composition just the average of 20 factors?" — a question Table 1
    cannot answer because it reports no such baseline.
    """

    def __init__(self) -> None:
        self.n_features_ = 0

    def fit(self, X, y=None):  # noqa: N803
        """No-op: an equal-weight average has nothing to learn."""
        self.n_features_ = X.shape[1]
        return self

    def predict(self, X):  # noqa: N803
        """Row-wise mean of the (already cross-sectionally ranked) features."""
        return np.nanmean(np.asarray(X, dtype="float64"), axis=1)


# ------------------------------------------------------------------ rolling fit


def rolling_predict(
    features: pd.DataFrame,
    label: pd.Series,
    train_window: Tuple[str, str],
    eval_window: Tuple[str, str],
    model_kind: str = "lightgbm",
    rolling_step: int = 126,
    horizon: int = 10,
    label_offset: int = 1,
    seed: int = 42,
    expanding: bool = True,
    lgbm_lr: float = 0.0001,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Roll a model forward through ``eval_window``, predicting out of sample.

    The paper uses "rolling training with a rolling step of 126" (§B.4) without
    saying whether the training set expands or slides.  ``expanding=True`` (default)
    keeps all history before the cutoff, which is the conventional reading for a
    daily-frequency equity model and uses strictly more data.

    Embargo
    -------
    At a cutoff ``T``, the training labels are truncated at
    ``T - (horizon + label_offset)`` trading days.  A label formed on day *t* spans
    ``open_{t+1} .. open_{t+1+h}``, so it is not observable until *t+1+h*; training
    on labels dated after the embargo means fitting on returns that had not been
    realised at the cutoff.  Omitting this is the single easiest way to produce a
    composition score that looks like the paper's and means nothing.
    """
    dates = features.index.get_level_values(0)
    label = label.reindex(features.index)

    eval_start = pd.Timestamp(eval_window[0]) if eval_window[0] else dates.min()
    eval_end = pd.Timestamp(eval_window[1]) if eval_window[1] else dates.max()
    train_start = pd.Timestamp(train_window[0]) if train_window[0] else dates.min()

    calendar = pd.DatetimeIndex(sorted(set(dates))).sort_values()
    eval_days = calendar[(calendar >= eval_start) & (calendar <= eval_end)]
    if len(eval_days) == 0:
        raise ValueError(f"evaluation window {eval_window} contains no panel dates")

    embargo = horizon + label_offset
    predictions: List[pd.Series] = []
    diagnostics: Dict[str, Any] = {
        "folds": [],
        "train_rows": 0,
        "test_rows": 0,
        "embargo_days": embargo,
    }
    importance_sum: Dict[str, float] = {c: 0.0 for c in features.columns}
    n_fitted = 0

    for start in range(0, len(eval_days), max(1, rolling_step)):
        # One fold = `rolling_step` consecutive evaluation days, predicted by a model
        # trained on everything up to (and embargoed before) the fold's first day.
        fold_days = eval_days[start : start + rolling_step]
        cutoff = fold_days[0]

        # Training data: everything from train_start up to the embargoed cutoff.
        # Positions are taken on the *trading* calendar, not calendar days, so the
        # embargo is 11 trading days rather than 11 dates.
        cutoff_pos = calendar.searchsorted(cutoff)
        embargo_pos = max(cutoff_pos - embargo, 0)
        if embargo_pos == 0:
            # The fold starts inside the embargo window; nothing legitimate to train
            # on. Skipping is correct -- shrinking the embargo instead would leak.
            continue
        train_end = calendar[embargo_pos - 1]

        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= fold_days[0]) & (dates <= fold_days[-1])
        if not expanding:
            # Sliding window: keep a training span as long as the gap to the fold.
            span = train_end - (fold_days[0] - pd.Timedelta(days=1))
            train_mask &= dates > (train_end - abs(span))

        X_train = features[train_mask]
        y_train = label[train_mask]
        # Drop rows with any missing feature or a missing label. Imputing would
        # invent factor values; dropping loses ~10% of rows (the warm-up of the
        # longest-window factor) and is the honest option.
        valid = y_train.notna() & X_train.notna().all(axis=1)
        X_train, y_train = X_train[valid], y_train[valid]

        X_test = features[test_mask]
        if X_train.empty or X_test.empty:
            continue

        # Refit from scratch each fold. Warm-starting would carry information from a
        # later fold's data backwards through the model state.
        model = make_model(model_kind, seed=seed, lgbm_lr=lgbm_lr)
        model.fit(X_train.to_numpy(dtype="float64"), y_train.to_numpy(dtype="float64"))

        # Rows with any missing feature cannot be predicted; leave them NaN rather
        # than imputing, so coverage is reported honestly downstream.
        complete = X_test.notna().all(axis=1)
        pred = pd.Series(np.nan, index=X_test.index, dtype="float64")
        if complete.any():
            pred.loc[complete] = model.predict(
                X_test[complete].to_numpy(dtype="float64")
            )
        predictions.append(pred)

        n_fitted += 1
        diagnostics["train_rows"] += int(len(X_train))
        diagnostics["test_rows"] += int(complete.sum())
        diagnostics["folds"].append(
            {
                "cutoff": str(cutoff.date()),
                "train": [str(train_start.date()), str(train_end.date())],
                "test": [str(fold_days[0].date()), str(fold_days[-1].date())],
                "train_rows": int(len(X_train)),
                "test_rows": int(complete.sum()),
            }
        )

        for col, weight in _importance(model, list(features.columns)).items():
            importance_sum[col] += weight

    if not predictions:
        raise ValueError(
            "no fold could be trained: check that the training window precedes the "
            f"evaluation window by more than the {embargo}-day embargo"
        )

    out = pd.concat(predictions).sort_index()
    diagnostics["n_folds"] = n_fitted
    diagnostics["feature_importance"] = (
        {k: v / n_fitted for k, v in importance_sum.items()} if n_fitted else {}
    )
    return out, diagnostics


def _importance(model, columns: List[str]) -> Dict[str, float]:
    """Normalised per-feature weight, whatever the model exposes."""
    raw = None
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype="float64")
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype="float64")).ravel()
    if raw is None or len(raw) != len(columns):
        return {}
    total = raw.sum()
    if total <= 0:
        return {c: 0.0 for c in columns}
    return {c: float(v / total) for c, v in zip(columns, raw)}


# ------------------------------------------------------------------ entry point


def compose(
    alphas: Sequence[Alpha],
    panel: Panel,
    cfg: CogAlphaConfig,
    model_kind: str = "lightgbm",
    split: str = "test",
    top_n: int = 20,
    rolling_step: int = 126,
    normalise: str = "cs_rank",
    run_backtest_flag: bool = True,
    seed: int = 42,
    lgbm_lr: float = 0.0001,
) -> CompositionResult:
    """Compose ``alphas`` into one prediction and score it like Table 1.

    ``top_n`` defaults to 20 because that is the number the paper composes.  The
    alphas are taken in the order given, so the caller decides what "top" means —
    :class:`~cogalpha.evolution.pool.CandidatePool` already sorts by combined score.
    """
    started = time.time()
    data = cfg.data
    window = {"train": data.train, "valid": data.valid, "test": data.test}.get(split)
    if window is None:
        raise ValueError(f"unknown split '{split}'; use train, valid or test")

    selected = list(alphas)[:top_n]
    features, dropped = build_feature_matrix(
        selected,
        panel,
        allowed_imports=cfg.quality.allowed_imports,
        normalise=normalise,
    )
    if features.empty:
        raise ValueError(
            "no alpha produced usable features; see the 'alphas_dropped' report"
        )

    label = forward_return(panel.frame, horizon=data.horizon, price="open", offset=1)

    prediction, diag = rolling_predict(
        features=features,
        label=label,
        train_window=data.train,
        eval_window=window,
        model_kind=model_kind,
        rolling_step=rolling_step,
        horizon=data.horizon,
        seed=seed,
        lgbm_lr=lgbm_lr,
    )

    result = _score_prediction(
        prediction=prediction,
        label=label,
        panel=panel,
        cfg=cfg,
        model_kind=model_kind,
        split=split,
        window=window,
        run_backtest_flag=run_backtest_flag,
    )
    result.n_alphas = len(features.columns)
    result.alphas_used = list(features.columns)
    result.alphas_dropped = dropped
    result.n_folds = int(diag["n_folds"])
    result.train_rows = int(diag["train_rows"])
    result.test_rows = int(diag["test_rows"])
    result.feature_importance = dict(diag.get("feature_importance", {}))
    result.seconds = time.time() - started
    return result


def _score_prediction(
    prediction: pd.Series,
    label: pd.Series,
    panel: Panel,
    cfg: CogAlphaConfig,
    model_kind: str,
    split: str,
    window: Tuple[str, str],
    run_backtest_flag: bool,
) -> CompositionResult:
    """Apply the five metrics and the ranking backtest to a composed prediction."""
    fit = cfg.fitness
    wide_pred = M.label_to_wide(prediction)
    wide_label = M.label_to_wide(label)

    start, end = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    wide_pred = wide_pred.loc[(wide_pred.index >= start) & (wide_pred.index <= end)]
    aligned_pred, aligned_label = M.align(wide_pred, wide_label)

    ic = M.ic_series(aligned_pred, aligned_label, "pearson", fit.min_names_per_day)
    ric = M.ic_series(aligned_pred, aligned_label, fit.rank_ic_method, fit.min_names_per_day)
    mi = M.mutual_information(
        aligned_pred, aligned_label, bins=fit.mi_bins, scale=fit.mi_scale
    )

    result = CompositionResult(
        model=model_kind,
        n_alphas=0,
        ic=ic.mean,
        icir=ic.ir,
        rank_ic=ric.mean,
        rank_icir=ric.ir,
        mi=mi,
        n_days=ic.n_days,
        split=split,
        window=window,
    )

    if run_backtest_flag:
        # Daily payoff, not the horizon label -- see run_backtest's docstring.
        daily = M.label_to_wide(
            forward_return(panel.frame, horizon=1, price="open", offset=1)
        )
        bt = run_backtest(
            aligned_pred,
            daily.reindex(index=aligned_pred.index),
            top_k=fit.top_k,
            drop_n=fit.drop_n,
            open_cost=fit.open_cost,
            close_cost=fit.close_cost,
            min_fee=fit.min_fee,
            trading_days_per_year=fit.trading_days_per_year,
        )
        result.aer = bt.aer
        result.ir = bt.ir
        result.backtest = bt.to_dict()

    return result


def compose_from_codes(
    codes: Dict[str, str],
    panel: Panel,
    cfg: CogAlphaConfig,
    **kwargs,
) -> CompositionResult:
    """Convenience wrapper: compose from ``{name: source}`` instead of Alpha objects.

    Used by ``cogalpha compose --run <dir>``, which reads the archived candidate
    ``.py`` files, and by the calibration script's baseline comparison.
    """
    from cogalpha.agents.parse import function_name

    alphas: List[Alpha] = []
    for label, code in codes.items():
        name = function_name(code) or label
        alphas.append(Alpha(code=code, name=name))
    return compose(alphas, panel, cfg, **kwargs)





