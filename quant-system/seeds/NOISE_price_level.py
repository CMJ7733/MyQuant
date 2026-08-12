# NEGATIVE CONTROL: NOISE_price_level
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_noise_price_level(df):
    """Negative control: raw price level -- a pure size/level proxy."""
    out = df.copy()
    return out['close']
