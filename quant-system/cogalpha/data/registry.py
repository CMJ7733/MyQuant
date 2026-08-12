"""Provider registry: synthetic / qlib / csv -> :class:`~cogalpha.data.panel.Panel`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from cogalpha.config import DataConfig
from cogalpha.data.panel import Panel, PanelError
from cogalpha.data.synthetic import make_noise_panel, make_synthetic_panel

ProviderFn = Callable[[DataConfig], Panel]
_PROVIDERS: Dict[str, ProviderFn] = {}

#: Set once ``qlib.init`` has run in this process, keyed by provider_uri.  qlib
#: holds global module state, so a second init against a different directory
#: silently repoints every later query -- we make that an explicit error instead.
_QLIB_INITIALISED: Dict[str, bool] = {}



def register_provider(name: str) -> Callable[[ProviderFn], ProviderFn]:
    """Decorator registering a ``DataConfig -> Panel`` function under ``name``."""

    def deco(fn: ProviderFn) -> ProviderFn:
        _PROVIDERS[name] = fn
        return fn

    return deco


def get_provider(name: str) -> ProviderFn:
    """Look up a provider; raises KeyError listing the registered names."""
    if name not in _PROVIDERS:
        raise KeyError(f"unknown data provider '{name}'; have {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]


def load_panel(cfg: DataConfig) -> Panel:
    """Load the full panel described by ``cfg`` (all splits, one frame)."""
    return get_provider(cfg.provider)(cfg)


@register_provider("synthetic")
def _synthetic(cfg: DataConfig) -> Panel:
    return make_synthetic_panel(
        n_instruments=cfg.synth_n_instruments,
        n_days=cfg.synth_n_days,
        seed=cfg.synth_seed,
        signal_strength=cfg.synth_signal_strength,
        horizon=cfg.horizon,
    )


@register_provider("noise")
def _noise(cfg: DataConfig) -> Panel:
    """Negative control: same generator with the planted signal switched off.

    Any alpha that reaches ``elite`` here is fitting noise, which is the
    multiple-testing blind spot ``reflection.md`` flags in the quality checker.
    """
    panel = make_noise_panel(
        n_instruments=cfg.synth_n_instruments,
        n_days=cfg.synth_n_days,
        seed=cfg.synth_seed + 1000,
    )
    panel.name = "noise"
    panel.meta = {**dict(panel.meta), "provider": "noise", "negative_control": True}
    return panel



@register_provider("csv")
def _csv(cfg: DataConfig) -> Panel:
    """Read a long-format CSV with date/instrument/OHLCV columns."""
    if not cfg.csv_path:
        raise PanelError("provider 'csv' requires data.csv_path")
    path = Path(cfg.csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"csv panel not found: {path}")
    frame = pd.read_csv(path)
    return Panel(frame, name=path.stem, meta={"provider": "csv", "path": str(path)})


@register_provider("qlib")
def _qlib(cfg: DataConfig) -> Panel:
    """Load OHLCV for an index's constituents from a local qlib data directory.

    Only raw ``$open/$high/$low/$close/$volume`` fields are requested — no
    Alpha158-style expression handler.  Using a feature handler here would import
    hundreds of engineered features and quietly hand the alphas a head start the
    paper does not grant them.
    """
    try:
        import qlib
        from qlib.data import D
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "provider 'qlib' needs pyqlib installed (pip install 'cogalpha[qlib]')"
        ) from exc

    uri = str(Path(cfg.qlib_provider_uri).expanduser())
    if not Path(uri).exists():
        raise FileNotFoundError(
            f"qlib data directory not found: {uri}\n"
            "Download a snapshot, e.g.\n"
            "  curl -L -o qlib_bin.tar.gz https://gh-proxy.com/https://github.com/"
            "chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz\n"
            f"  mkdir -p {uri} && tar xzf qlib_bin.tar.gz --strip-components=1 -C {uri}"
        )

    if _QLIB_INITIALISED and uri not in _QLIB_INITIALISED:
        raise RuntimeError(
            f"qlib was already initialised against {sorted(_QLIB_INITIALISED)}; "
            f"re-initialising against {uri} in the same process would silently "
            "repoint earlier handles. Run one provider_uri per process."
        )
    if uri not in _QLIB_INITIALISED:
        qlib.init(provider_uri=uri, region=cfg.qlib_region)
        _QLIB_INITIALISED[uri] = True

    instruments = D.instruments(market=cfg.market)
    fields = ["$open", "$high", "$low", "$close", "$volume"]
    start = cfg.train[0]
    end = cfg.test[1]

    raw = D.features(
        instruments,
        fields,
        start_time=start,
        end_time=end,
        freq="day",
        disk_cache=0,
    )
    if raw is None or raw.empty:
        raise PanelError(
            f"qlib returned no data for market={cfg.market} {start}..{end}; "
            f"check provider_uri={uri} and the calendar range"
        )

    frame = raw.rename(columns={f: f.lstrip("$") for f in fields})
    frame = frame.reset_index().rename(columns={"datetime": "date"})
    return Panel(
        frame,
        name=f"qlib-{cfg.market}",
        meta={
            "provider": "qlib",
            "provider_uri": uri,
            "market": cfg.market,
            "region": cfg.qlib_region,
            "price_adjustment": "backward-adjusted (qlib $close = raw * $factor)",
        },
    )



def available_providers() -> List[str]:
    """Registered provider names, for CLI help and error messages."""
    return sorted(_PROVIDERS)
