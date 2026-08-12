# baseline alpha: L5_roughness
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l5_roughness(df):
    """Level V: 20d vol scaled to 60d vol -- multi-scale roughness."""
    import numpy as np
    out = df.copy(); eps = 1e-12
    r = out['close'].pct_change()
    s20 = r.rolling(20, min_periods=10).std(); s60 = r.rolling(60, min_periods=30).std()
    return -(s20 * np.sqrt(3.0)) / (s60 + eps)
