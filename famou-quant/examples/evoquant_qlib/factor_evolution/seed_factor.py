"""Seed factor: 20-day cross-sectional reversal, computed from raw prices.

This is the starting point the evolutionary search mutates. It is deliberately
simple, correct, and boring — a single well-known effect — so that "the search
improved on the seed" means it found something beyond the textbook signal.

THE CONTRACT (this docstring is also the spec the LLM is given)
---------------------------------------------------------------
Define ``compute_factor(panel) -> np.ndarray``.

``panel`` exposes these aligned ``(n_days, n_instruments)`` float64 matrices::

    panel.open  panel.high  panel.low  panel.close
    panel.volume  panel.vwap  panel.amount

plus ``panel.dates`` (list[str], ascending) and ``panel.instruments`` (list[str]).
NaN means the stock was not in the universe / not trading that day.

Return a ``(n_days, n_instruments)`` array. NaN entries are simply excluded from
that day's cross-section.

**The one hard rule: row ``t`` of your output may only depend on rows ``0..t`` of
the input.** Using row ``t+1`` or later is look-ahead — it reads the future. The
evaluator tests this mechanically by re-running your function on a truncated
panel and checking that row ``t`` comes out identical; a factor that fails is
scored zero. There is no way to "get away with it", so do not try.

Higher factor value should predict higher forward return.
"""

import numpy as np


def compute_factor(panel):
    """Negated 20-day return: past losers are expected to bounce back.

    Reversal rather than momentum because at the ~1-day horizon this label
    measures (buy next open, hold one day), short-horizon reversal is the
    better-documented effect in the A-share cross-section.
    """
    close = panel.close
    window = 20

    factor = np.full_like(close, np.nan)
    # Row t uses rows t-20..t only — never anything later. This is the
    # causality rule the evaluator enforces.
    factor[window:] = -(close[window:] / close[:-window] - 1.0)
    return factor
