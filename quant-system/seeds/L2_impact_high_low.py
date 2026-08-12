# baseline alpha: L2_impact_high_low
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l2_impact_high_low(df):
    """Listing 2 (mutated, discarded in the paper): (high - low) / volume."""
    out = df.copy(); eps = 1e-9
    return (out['high'] - out['low']) / (out['volume'] + eps)
