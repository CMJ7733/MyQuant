# baseline alpha: L4_reversal_5
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l4_reversal_5(df):
    """Level IV: 5-day reversal."""
    out = df.copy(); eps = 1e-12
    return -(out['close'] / (out['close'].shift(5) + eps) - 1.0)
