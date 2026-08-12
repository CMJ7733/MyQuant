# baseline alpha: L3_pv_corr
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l3_pv_corr(df):
    """Level III: 20-day correlation of return and volume change, flipped."""
    out = df.copy()
    r = out['close'].pct_change(); dv = out['volume'].pct_change()
    return -r.rolling(20, min_periods=10).corr(dv)
