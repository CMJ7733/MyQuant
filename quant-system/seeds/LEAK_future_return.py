# LEAKY CONTROL: LEAK_future_return
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_leak_future_return(df):
    """LEAKY control: next-10-day return. Reported to show what leakage looks like."""
    out = df.copy(); eps = 1e-12
    return out['close'].shift(-10) / (out['close'] + eps) - 1.0
