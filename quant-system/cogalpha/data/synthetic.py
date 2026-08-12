"""Synthetic OHLCV generator for offline development and tests.

The generator is not trying to be a realistic market simulator.  It has one job:
produce a panel where a *known* alpha is recoverable, so the whole pipeline
(quality checker -> fitness -> tiering -> evolution) can be exercised end to end
without an LLM, without qlib data, and deterministically.

Planted signal
--------------
Next-period returns are built as

    r_{t+1} = beta * s_t + noise

where ``s_t`` is a standardised "illiquidity impact" state closely related to the
paper's seed alpha ``(high - close) / volume`` (Listing 1).  Consequently that
alpha, and its evolved dollar-volume variant, score positive IC here while an
arbitrary expression does not — which is what makes the tier assertions in
``tests/test_fitness.py`` meaningful rather than tautological.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from cogalpha.data.panel import Panel


def make_synthetic_panel(
    n_instruments: int = 60,
    n_days: int = 700,
    seed: int = 7,
    signal_strength: float = 0.0012,
    start: str = "2015-01-05",
    horizon: int = 10,
) -> Panel:
    """Build a synthetic OHLCV panel with a planted, recoverable signal.

    Parameters
    ----------
    n_instruments, n_days:
        Panel dimensions.  Business-day calendar starting at ``start``.
    seed:
        Numpy seed; identical inputs give a byte-identical panel.
    signal_strength:
        Coefficient on the standardised state.  Calibrated so the default lands in
        the paper's own regime: on a 300 x 500 panel the seed alpha scores
        RankIC 0.076 against the paper's 0.0814, and MI 0.090.  Raising it to 0.01
        gives RankIC 0.36 -- convenient for a fast test but a signal-to-noise ratio
        no equity market has, which would make every threshold behave wrongly.
    horizon:
        Horizon the signal is planted at; must match the fitness horizon for the
        planted alpha to be detectable.
    """

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    instruments = [f"SYN{i:04d}" for i in range(n_instruments)]

    # Market factor shared across names, plus a per-name volatility level.
    market = rng.normal(0.0002, 0.011, size=n_days)
    vol_level = rng.uniform(0.010, 0.028, size=n_instruments)
    beta = rng.normal(1.0, 0.25, size=n_instruments)

    records = []
    for j, inst in enumerate(instruments):
        idio = rng.normal(0.0, vol_level[j], size=n_days)

        # A slow-moving liquidity state drives both the observable volume and the
        # future return, which is what creates a learnable relationship.
        state = np.zeros(n_days)
        eps = rng.normal(0.0, 1.0, size=n_days)
        for t in range(1, n_days):
            state[t] = 0.94 * state[t - 1] + 0.34 * eps[t]

        ret = beta[j] * market + idio
        # Plant the signal: the state at t lifts the cumulative return over
        # [t+1, t+1+horizon], matching the label's open-to-open window.
        planted = signal_strength * state / max(horizon, 1)
        for t in range(n_days):
            lo = t + 1
            hi = min(n_days, t + 1 + horizon)
            if lo < hi:
                ret[lo:hi] += planted[t]

        close = 20.0 * np.exp(np.cumsum(ret))
        open_ = close * np.exp(rng.normal(0.0, 0.004, size=n_days))
        intraday = np.abs(rng.normal(0.0, vol_level[j], size=n_days)) + 0.002
        high = np.maximum(open_, close) * (1.0 + intraday)
        low = np.minimum(open_, close) * (1.0 - intraday)

        # Volume is inversely related to the state: high state == thin trading,
        # so (high - close) / volume picks the state up with a positive sign.
        base_volume = np.exp(rng.normal(15.0, 0.4))
        volume = base_volume * np.exp(-0.55 * state + rng.normal(0.0, 0.30, size=n_days))
        volume = np.maximum(volume, 1.0)

        records.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "instrument": inst,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )

    frame = pd.concat(records, ignore_index=True)
    meta: Dict[str, object] = {
        "provider": "synthetic",
        "seed": seed,
        "signal_strength": signal_strength,
        "planted_horizon": horizon,
        "planted_alpha": "(high - close) / volume",
    }
    return Panel(frame, name="synthetic", meta=meta)


def make_noise_panel(
    n_instruments: int = 40,
    n_days: int = 400,
    seed: int = 11,
    start: str = "2015-01-05",
) -> Panel:
    """A panel with no planted signal at all.

    Used as a negative control: any alpha that scores well here is fitting noise,
    which is the multiple-testing failure mode ``reflection.md`` flags as the
    checker's blind spot.
    """
    return make_synthetic_panel(
        n_instruments=n_instruments,
        n_days=n_days,
        seed=seed,
        signal_strength=0.0,
        start=start,
    )
