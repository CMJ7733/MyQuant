"""Measure the five metrics on real CSI300 data, so thresholds are set from
observed distributions rather than from a synthetic generator's assumptions.

Runs a battery of hand-written alphas (including the paper's Listing 1/2/3
liquidity-impact lineage) over the paper's own splits and horizon, and prints the
distribution of each metric.  What we need out of this:

1. the magnitude of MI for an alpha whose IC is in the paper's reported range --
   this decides whether the 0.02 MI floor of Appendix A.4 is in nats or on some
   other scale;
2. whether the IC/ICIR floors admit real alphas and exclude noise;
3. a realistic spread for the 65th/80th percentile gates.
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from cogalpha.config import load_config
from cogalpha.data import load_panel
from cogalpha.fitness.metrics import ic_series, label_to_wide, mutual_information
from cogalpha.quality.sandbox import apply_alpha, compile_alpha

ALPHAS = {}

def alpha(name):
    def deco(code):
        ALPHAS[name] = code
        return code
    return deco

# ---- the paper's evolution example, all three versions (Listing 1/2/3) --------
alpha("L1_impact_high_close")('''
def f(df):
    """Listing 1: (high - close) / volume."""
    out = df.copy(); eps = 1e-9
    return (out['high'] - out['close']) / (out['volume'] + eps)
''')
alpha("L2_impact_high_low")('''
def f(df):
    """Listing 2 (mutated, discarded in the paper): (high - low) / volume."""
    out = df.copy(); eps = 1e-9
    return (out['high'] - out['low']) / (out['volume'] + eps)
''')
alpha("L3_impact_tanh_dollar")('''
def f(df):
    """Listing 3 (evolved): tanh(|close-open| / dollar volume)."""
    import numpy as np
    out = df.copy(); eps = 1e-9
    absmove = (out['close'] - out['open']).abs()
    dollar = out['volume'] * out['close']
    return np.tanh(absmove / (dollar + eps))
''')

# ---- one representative alpha per hierarchy level ----------------------------
alpha("L1_cycle_position")('''
def f(df):
    """Level I: position in the 120-day range, sign-flipped."""
    out = df.copy(); eps = 1e-12
    hi = out['high'].rolling(120, min_periods=60).max()
    lo = out['low'].rolling(120, min_periods=60).min()
    return -((out['close'] - lo) / (hi - lo + eps) - 0.5)
''')
alpha("L1_vol_regime")('''
def f(df):
    """Level I: short vs long realised volatility ratio."""
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    return -(r.rolling(20, min_periods=10).std() / (r.rolling(120, min_periods=60).std() + eps))
''')
alpha("L2_tail_worst")('''
def f(df):
    """Level II: worst 60-day return scaled by volatility."""
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    return r.rolling(60, min_periods=30).min() / (r.rolling(60, min_periods=30).std() + eps)
''')
alpha("L3_illiq_amihud")('''
def f(df):
    """Level III: Amihud illiquidity, |return| per dollar volume, 20d mean."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change().abs()
    dollar = out['volume'] * out['close']
    return (r / (dollar + eps)).rolling(20, min_periods=10).mean()
''')
alpha("L3_close_location")('''
def f(df):
    """Level III: where the close sits in the day's range, 5-day mean."""
    out = df.copy(); eps = 1e-12
    loc = (out['close'] - out['low']) / (out['high'] - out['low'] + eps)
    return -loc.rolling(5, min_periods=3).mean()
''')
alpha("L3_pv_corr")('''
def f(df):
    """Level III: 20-day correlation of return and volume change, flipped."""
    out = df.copy()
    r = out['close'].pct_change(); dv = out['volume'].pct_change()
    return -r.rolling(20, min_periods=10).corr(dv)
''')
alpha("L3_volume_surprise")('''
def f(df):
    """Level III: log volume vs its 20-day mean, flipped."""
    import numpy as np
    out = df.copy()
    lv = np.log(out['volume'].clip(lower=1.0))
    return -(lv - lv.rolling(20, min_periods=10).mean())
''')
alpha("L4_reversal_5")('''
def f(df):
    """Level IV: 5-day reversal."""
    out = df.copy(); eps = 1e-12
    return -(out['close'] / (out['close'].shift(5) + eps) - 1.0)
''')
alpha("L4_reversal_20")('''
def f(df):
    """Level IV: 20-day reversal."""
    out = df.copy(); eps = 1e-12
    return -(out['close'] / (out['close'].shift(20) + eps) - 1.0)
''')
alpha("L4_momentum_120")('''
def f(df):
    """Level IV: 120-day momentum."""
    out = df.copy(); eps = 1e-12
    return out['close'] / (out['close'].shift(120) + eps) - 1.0
''')
alpha("L4_range_compression")('''
def f(df):
    """Level IV: today's range vs its 20-day average, flipped."""
    out = df.copy(); eps = 1e-12
    rng = (out['high'] - out['low']) / (out['close'] + eps)
    return -(rng / (rng.rolling(20, min_periods=10).mean() + eps))
''')
alpha("L4_vol_asymmetry")('''
def f(df):
    """Level IV: downside minus upside volatility, 60d."""
    out = df.copy()
    r = out['close'].pct_change()
    dn = r.where(r < 0, 0.0); up = r.where(r > 0, 0.0)
    return dn.rolling(60, min_periods=30).std() - up.rolling(60, min_periods=30).std()
''')
alpha("L4_idio_vol")('''
def f(df):
    """Level IV: 60-day realised volatility, flipped (low-vol anomaly)."""
    out = df.copy()
    return -out['close'].pct_change().rolling(60, min_periods=30).std()
''')
alpha("L5_drawdown")('''
def f(df):
    """Level V: drawdown against the 120-day peak."""
    out = df.copy(); eps = 1e-12
    peak = out['close'].rolling(120, min_periods=60).max()
    return out['close'] / (peak + eps) - 1.0
''')
alpha("L5_roughness")('''
def f(df):
    """Level V: 20d vol scaled to 60d vol -- multi-scale roughness."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    s20 = r.rolling(20, min_periods=10).std(); s60 = r.rolling(60, min_periods=30).std()
    return -(s20 * np.sqrt(3.0)) / (s60 + eps)
''')
alpha("L6_gated_reversal")('''
def f(df):
    """Level VI: 5-day reversal weighted by the volatility and volume regime.

    Reversal corrects overreaction, so it needs a move large for this stock (vol
    above its own expanding median) made by a crowd (volume above its own 20-day
    average). The two gates are near-orthogonal in the cross-section (-0.028), so
    stacking them beats either alone. Weights are smooth rather than 0/1: a hard
    gate put 47.9% of the cross-section on exactly zero and inverted the measured
    RankIC. Kept in sync with seeds/L6_gated_reversal.py, which carries the
    measurements and the out-of-sample decay.
    """
    out = df.copy(); eps = 1e-12
    rev = -(out['close'] / (out['close'].shift(5) + eps) - 1.0)
    v = out['close'].pct_change().rolling(20, min_periods=10).std()
    vol_r = (v + eps) / (v.expanding(min_periods=60).median().shift(1) + eps)
    vlm_r = (out['volume'] + eps) / (
        out['volume'].shift(1).rolling(20, min_periods=10).mean() + eps)
    w_vol = vol_r ** 4 / (1.0 + vol_r ** 4)
    w_vlm = vlm_r ** 4 / (1.0 + vlm_r ** 4)
    return rev * w_vol * w_vlm
''')
alpha("L6_stability_autocorr")('''
def f(df):
    """Level VI: 60-day lag-1 autocorrelation of returns."""
    out = df.copy()
    r = out['close'].pct_change()
    return r.rolling(60, min_periods=30).corr(r.shift(1))
''')
alpha("L7_upper_shadow")('''
def f(df):
    """Level VII: upper-shadow share of the range, 5d mean."""
    out = df.copy(); eps = 1e-12
    span = (out['high'] - out['low']) + eps
    upper = out['high'] - out[['open', 'close']].max(axis=1)
    return -(upper / span).rolling(5, min_periods=3).mean()
''')
alpha("L7_composite")('''
def f(df):
    """Level VII: z-scored illiquidity plus z-scored reversal."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    illiq = (r.abs() / (out['volume'] * out['close'] + eps)).rolling(20, min_periods=10).mean()
    rev = -(out['close'] / (out['close'].shift(5) + eps) - 1.0)
    z = lambda s: (s - s.rolling(250, min_periods=120).mean()) / (s.rolling(250, min_periods=120).std() + eps)
    return z(illiq) + z(rev)
''')
alpha("L7_herding")('''
def f(df):
    """Level VII: 10-day signed-return streak weighted by volume growth."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    streak = np.sign(r).rolling(10, min_periods=5).mean()
    vg = out['volume'] / (out['volume'].rolling(20, min_periods=10).mean() + eps)
    return -(streak * vg)
''')

# ---- negative controls -------------------------------------------------------
alpha("NOISE_random")('''
def f(df):
    """Negative control: deterministic pseudo-noise, no information."""
    import numpy as np
    import pandas as pd
    out = df.copy()
    idx = np.arange(len(out))
    return pd.Series(np.sin(idx * 12.9898) * 43758.5453 % 1.0, index=out.index)
''')
alpha("NOISE_price_level")('''
def f(df):
    """Negative control: raw price level -- a pure size/level proxy."""
    out = df.copy()
    return out['close']
''')
# ---- leakage control (must be caught, reported for reference only) ----------
alpha("LEAK_future_return")('''
def f(df):
    """LEAKY control: next-10-day return. Reported to show what leakage looks like."""
    out = df.copy(); eps = 1e-12
    return out['close'].shift(-10) / (out['close'] + eps) - 1.0
''')


def main() -> None:
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_config(None, data={"provider": "qlib", "market": "csi300", "horizon": horizon})

    t0 = time.time()
    panel = load_panel(cfg.data)
    print(f"panel loaded in {time.time()-t0:.1f}s: {json.dumps(panel.describe(), default=str)}", flush=True)

    t0 = time.time()
    frames = dict(panel.iter_instruments())
    print(f"{len(frames)} instrument frames in {time.time()-t0:.1f}s", flush=True)

    label = label_to_wide(panel.label(horizon, price="open", offset=1))
    print(f"label: {label.shape}, non-null {label.notna().mean().mean():.1%}", flush=True)

    splits = {
        "train(2011-2019)": cfg.data.train,
        "valid(2020)": cfg.data.valid,
        "test(2021-2024)": cfg.data.test,
    }

    rows = []
    for name, code in ALPHAS.items():
        t0 = time.time()
        try:
            fn = compile_alpha(code, "f")
            values = apply_alpha(fn, frames, "f")
        except Exception as exc:
            print(f"{name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue

        row = {"alpha": name}
        for split_name, (start, end) in splits.items():
            v = values.loc[(values.index >= start) & (values.index <= end)]
            l = label.loc[(label.index >= start) & (label.index <= end)]
            d = v.index.intersection(l.index)
            c = v.columns.intersection(l.columns)
            V, L = v.loc[d, c], l.loc[d, c]
            ic = ic_series(V, L, "pearson", 10)
            ric = ic_series(V, L, "spearman", 10)
            mi_n = mutual_information(V, L, bins=10, scale="nats")
            mi_c = mutual_information(V, L, bins=10, scale="corr_equivalent")
            row[split_name] = {
                "ic": ic.mean, "icir": ic.ir, "rank_ic": ric.mean, "rank_icir": ric.ir,
                "mi_nats": mi_n, "mi_corr": mi_c, "n_days": ic.n_days,
                "nan_ratio": float(V.isna().mean().mean()),
            }
        row["seconds"] = time.time() - t0
        rows.append(row)
        s = row["valid(2020)"]
        print(
            f"{name:<24} valid: IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} "
            f"RankIC={s['rank_ic']:+.4f} RankICIR={s['rank_icir']:+.3f} "
            f"MI={s['mi_nats']:.5f}nats/{s['mi_corr']:.4f}corr "
            f"nan={s['nan_ratio']:.1%} ({row['seconds']:.1f}s)",
            flush=True,
        )

    out = Path(__file__).parent / f"calibration_csi300_h{horizon}.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nwrote {out}")

    # ---------------------------------------------------------------- summary
    for split_name in splits:
        real = [r for r in rows if not r["alpha"].startswith(("NOISE", "LEAK"))]
        noise = [r for r in rows if r["alpha"].startswith("NOISE")]
        print(f"\n===== {split_name} =====")
        for label_, group in (("real alphas", real), ("noise controls", noise)):
            if not group:
                continue
            print(f"  {label_} (n={len(group)}):")
            for metric in ("ic", "icir", "rank_ic", "rank_icir", "mi_nats", "mi_corr"):
                vals = np.array([r[split_name][metric] for r in group], dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                a = np.abs(vals)
                print(
                    f"    |{metric:<8}|  min={a.min():.4f}  p25={np.percentile(a,25):.4f} "
                    f" p50={np.percentile(a,50):.4f}  p65={np.percentile(a,65):.4f} "
                    f" p80={np.percentile(a,80):.4f}  max={a.max():.4f}"
                )


if __name__ == "__main__":
    main()
