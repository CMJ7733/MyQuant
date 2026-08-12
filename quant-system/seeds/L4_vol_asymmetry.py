# baseline alpha: L4_vol_asymmetry
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l4_vol_asymmetry(df):
    """Level IV: downside minus upside volatility, 60d."""
    out = df.copy()
    r = out['close'].pct_change()
    dn = r.where(r < 0, 0.0); up = r.where(r > 0, 0.0)
    return dn.rolling(60, min_periods=30).std() - up.rolling(60, min_periods=30).std()
