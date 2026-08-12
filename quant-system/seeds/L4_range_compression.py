# baseline alpha: L4_range_compression
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l4_range_compression(df):
    """Level IV: today's range vs its 20-day average, flipped."""
    out = df.copy(); eps = 1e-12
    rng = (out['high'] - out['low']) / (out['close'] + eps)
    return -(rng / (rng.rolling(20, min_periods=10).mean() + eps))
