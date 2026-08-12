# baseline alpha: L2_tail_worst
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l2_tail_worst(df):
    """Level II: worst 60-day return scaled by volatility."""
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    return r.rolling(60, min_periods=30).min() / (r.rolling(60, min_periods=30).std() + eps)
