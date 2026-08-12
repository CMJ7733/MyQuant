# baseline alpha: L7_upper_shadow
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l7_upper_shadow(df):
    """Level VII: upper-shadow share of the range, 5d mean."""
    out = df.copy(); eps = 1e-12
    span = (out['high'] - out['low']) + eps
    upper = out['high'] - out[['open', 'close']].max(axis=1)
    return -(upper / span).rolling(5, min_periods=3).mean()
