# baseline alpha: L3_volume_surprise
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l3_volume_surprise(df):
    """Level III: log volume vs its 20-day mean, flipped."""
    import numpy as np
    out = df.copy()
    lv = np.log(out['volume'].clip(lower=1.0))
    return -(lv - lv.rolling(20, min_periods=10).mean())
