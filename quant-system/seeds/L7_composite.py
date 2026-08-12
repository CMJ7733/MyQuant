# baseline alpha: L7_composite
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l7_composite(df):
    """Level VII: z-scored illiquidity plus z-scored reversal."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    illiq = (r.abs() / (out['volume'] * out['close'] + eps)).rolling(20, min_periods=10).mean()
    rev = -(out['close'] / (out['close'].shift(5) + eps) - 1.0)
    z = lambda s: (s - s.rolling(250, min_periods=120).mean()) / (s.rolling(250, min_periods=120).std() + eps)
    return z(illiq) + z(rev)
