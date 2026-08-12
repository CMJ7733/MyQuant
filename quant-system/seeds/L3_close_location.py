# baseline alpha: L3_close_location
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l3_close_location(df):
    """Level III: where the close sits in the day's range, 5-day mean."""
    out = df.copy(); eps = 1e-12
    loc = (out['close'] - out['low']) / (out['high'] - out['low'] + eps)
    return -loc.rolling(5, min_periods=3).mean()
