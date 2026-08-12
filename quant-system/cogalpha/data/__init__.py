"""Data access: OHLCV panels and forward-return labels.

The alpha function contract (see :mod:`cogalpha.quality.sandbox`) is
*per-instrument*: an alpha receives one stock's OHLCV history as a
:class:`pandas.DataFrame` sorted by date and returns a float series aligned to
it.  That is exactly the shape the paper's listings assume — ``talib.EMA`` over
``df['day_close']`` only makes sense on a single time series — and it keeps
cross-sectional work (ranking, IC) inside the evaluator where it belongs.
"""

from cogalpha.data.panel import (  # noqa: F401
    Panel,
    forward_return,
    make_splits,
    slice_panel,
)
from cogalpha.data.registry import get_provider, load_panel  # noqa: F401

__all__ = [
    "Panel",
    "forward_return",
    "make_splits",
    "slice_panel",
    "get_provider",
    "load_panel",
]
