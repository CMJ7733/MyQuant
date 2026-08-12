# baseline alpha: L1_cycle_position
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l1_cycle_position(df):
    """Level I: position in the 120-day range, sign-flipped."""
    out = df.copy(); eps = 1e-12
    hi = out['high'].rolling(120, min_periods=60).max()
    lo = out['low'].rolling(120, min_periods=60).min()
    return -((out['close'] - lo) / (hi - lo + eps) - 0.5)
