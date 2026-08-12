# baseline alpha: L6_gated_reversal
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l6_gated_reversal(df):
    """Level VI: 5-day reversal gated on the calm volatility regime."""
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    vol = r.rolling(20, min_periods=10).std()
    gate = (vol <= vol.expanding(min_periods=60).median().shift(1)).astype(float)
    return -(out['close'] / (out['close'].shift(5) + eps) - 1.0) * gate
