# baseline alpha: L4_idio_vol
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l4_idio_vol(df):
    """Level IV: 60-day realised volatility, flipped (low-vol anomaly)."""
    out = df.copy()
    return -out['close'].pct_change().rolling(60, min_periods=30).std()
