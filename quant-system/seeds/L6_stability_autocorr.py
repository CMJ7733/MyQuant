# baseline alpha: L6_stability_autocorr
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l6_stability_autocorr(df):
    """Level VI: 60-day lag-1 autocorrelation of returns."""
    out = df.copy()
    r = out['close'].pct_change()
    return r.rolling(60, min_periods=30).corr(r.shift(1))
