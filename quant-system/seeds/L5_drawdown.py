# baseline alpha: L5_drawdown
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l5_drawdown(df):
    """Level V: drawdown against the 120-day peak."""
    out = df.copy(); eps = 1e-12
    peak = out['close'].rolling(120, min_periods=60).max()
    return out['close'] / (peak + eps) - 1.0
