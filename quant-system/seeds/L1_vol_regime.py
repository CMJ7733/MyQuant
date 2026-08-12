# baseline alpha: L1_vol_regime
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l1_vol_regime(df):
    """Level I: short vs long realised volatility ratio."""
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    return -(r.rolling(20, min_periods=10).std() / (r.rolling(120, min_periods=60).std() + eps))
