# baseline alpha: L3_impact_tanh_dollar
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_l3_impact_tanh_dollar(df):
    """Listing 3 (evolved): tanh(|close-open| / dollar volume)."""
    import numpy as np
    out = df.copy(); eps = 1e-9
    absmove = (out['close'] - out['open']).abs()
    dollar = out['volume'] * out['close']
    return np.tanh(absmove / (dollar + eps))
