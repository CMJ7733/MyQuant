# baseline alpha: L4_momentum_120
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l4_momentum_120(df):
    """Level IV: 120-day momentum."""
    out = df.copy(); eps = 1e-12
    return out['close'] / (out['close'].shift(120) + eps) - 1.0
