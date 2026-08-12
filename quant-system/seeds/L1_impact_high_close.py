# baseline alpha: L1_impact_high_close
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l1_impact_high_close(df):
    """Listing 1: (high - close) / volume."""
    out = df.copy(); eps = 1e-9
    return (out['high'] - out['close']) / (out['volume'] + eps)
