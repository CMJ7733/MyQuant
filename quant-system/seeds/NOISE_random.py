# NEGATIVE CONTROL: NOISE_random
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.

def factor_noise_random(df):
    """Negative control: deterministic pseudo-noise, no information."""
    import numpy as np
    import pandas as pd
    out = df.copy()
    idx = np.arange(len(out))
    return pd.Series(np.sin(idx * 12.9898) * 43758.5453 % 1.0, index=out.index)
