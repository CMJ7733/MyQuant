# baseline alpha: L3_illiq_amihud
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l3_illiq_amihud(df):
    """Level III: Amihud illiquidity, |return| per dollar volume, 20d mean."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change().abs()
    dollar = out['volume'] * out['close']
    return (r / (dollar + eps)).rolling(20, min_periods=10).mean()
