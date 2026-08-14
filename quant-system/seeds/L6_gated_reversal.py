# baseline alpha: L6_gated_reversal
# Hand-written, from scripts/calibrate_real.py. Used to calibrate thresholds
# and to validate the composition pipeline against paper Table 1 without an LLM.
#
# Two things about this seed differ from the obvious way to write it, both measured
# on CSI300 2011-2024 rather than assumed:
#
# 1. The gate points at the TURBULENT side, not the calm one. The original version
#    kept `vol <= expanding median` on the story that mean reversion works in stable
#    regimes. On this data that is backwards. Sorting stock-days by vol / own
#    expanding median and measuring 5-day reversal inside each quintile gives
#    -0.0135, -0.0019, +0.0034, +0.0237, +0.0395 -- monotone, and the old gate cut
#    it exactly where the sign changes, keeping the two negative buckets and
#    discarding the two strongest positive ones. The mechanism is that reversal is
#    an overreaction correction, so it needs a reaction large enough to be worth
#    correcting; the discarded half had ~1.9x the |ret5| of the half kept.
#
# 2. The weights are smooth, not 0/1. A hard gate sends the whole inactive side to
#    exactly zero: 47.9% of the average cross-section on one value, 953 of 2184 days
#    over the tie limit in quality/numeric.py. That tie block takes a single shared
#    rank and inverted the measured RankIC (-0.0075 with ties against +0.0028 on the
#    same days and the same selection). It also defeats `min_names_per_day`, which
#    counts the zeros as valid names -- 124 days scored on a cross-section that was
#    94% zeros.
#
# Measured RankIC (Newey-West t at lag 9, for the 10-day overlap):
#
#     split              no gate         this seed
#     train 2011-2019    +0.0181 (2.51)  +0.0294 (4.55)
#     valid 2020         -0.0016 (0.08)  +0.0144 (0.85)
#     test  2021-2024    +0.0024 (0.20)  +0.0058 (0.53)
#
# Read the last two rows before using this number for anything. The gated version
# beats the ungated one in all three splits and the ordering of every variant tested
# was stable across them, but the level decays ~80% out of sample and is not
# significant there -- CSI300 short-term reversal largely stopped paying after 2020.
# This is a calibration baseline, not a tradable alpha.


def factor_l6_gated_reversal(df):
    """Level VI: 5-day reversal weighted by the volatility and volume regime.

    What it measures: the negative 5-day return, weighted up when the stock is both
    more volatile than its own history and trading above its own recent volume.

    Why it predicts returns: reversal corrects overreaction, which needs two things
    to be present. Volatility above the stock's own norm says the move was large for
    this name rather than large in absolute terms. Volume above its own average says
    a crowd was involved rather than the price drifting on thin trade. When both
    hold, the move is more likely a crowd overreaction that reverts; when either is
    absent it is more likely ordinary noise, and the weight product suppresses it.

    The two gates are near-orthogonal in the cross-section (rank correlation -0.028),
    which is why stacking them beats either alone: +0.0261 and +0.0239 separately
    against +0.0294 together, on train.

    Formula:
        rev    = -(close / close.shift(5) - 1)
        vol_r  = std20(ret) / expanding_median(std20(ret)).shift(1)
        vlm_r  = volume / SMA20(volume.shift(1))
        w(x)   = x**4 / (1 + x**4)
        factor = rev * w(vol_r) * w(vlm_r)

    w is 0.5 at x = 1 -- the point the old hard gate flipped at -- and strictly
    monotone, so no two stocks share a value unless their inputs do. The exponent 4
    puts the transition between 0.61x and 1.65x, which is about six standard errors
    of a 20-day volatility estimate; a sharper one would track the noise in `vol`
    rather than the regime. Multiplying the two weights rather than averaging them
    is deliberate: it is an AND, so one missing condition vetoes the position.
    """
    out = df.copy(); eps = 1e-12

    rev = -(out['close'] / (out['close'].shift(5) + eps) - 1.0)

    v = out['close'].pct_change().rolling(20, min_periods=10).std()
    vol_r = (v + eps) / (v.expanding(min_periods=60).median().shift(1) + eps)

    vlm_r = (out['volume'] + eps) / (
        out['volume'].shift(1).rolling(20, min_periods=10).mean() + eps)

    w_vol = vol_r ** 4 / (1.0 + vol_r ** 4)
    w_vlm = vlm_r ** 4 / (1.0 + vlm_r ** 4)

    return rev * w_vol * w_vlm
