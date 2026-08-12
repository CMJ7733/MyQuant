# baseline alpha: L7_herding
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l7_herding(df):
    """Level VII: 10-day signed-return streak weighted by volume growth."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    streak = np.sign(r).rolling(10, min_periods=5).mean()
    vg = out['volume'] / (out['volume'].rolling(20, min_periods=10).mean() + eps)
    return -(streak * vg)
