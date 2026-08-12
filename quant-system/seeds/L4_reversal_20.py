# baseline alpha: L4_reversal_20
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l4_reversal_20(df):
    """Level IV: 20-day reversal."""
    out = df.copy(); eps = 1e-12
    return -(out['close'] / (out['close'].shift(20) + eps) - 1.0)
