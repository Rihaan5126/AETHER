"""Supervised prediction targets aligned with execution timing.

For a decision made after the close of ``t`` the simulator fills at
``open[t+1]``. Holding for ``h`` sessions and exiting at ``open[t+1+h]`` earns

    r(t, h) = open[t+1+h] / open[t+1] - 1

before costs. That is exactly the label used for training, so the model is asked
to predict the return the strategy can actually capture. Labels whose window runs
past the end of the provided bars are ``NaN`` and must be dropped -- this is what
purges training rows near a split boundary when the bars are truncated there.
"""

from __future__ import annotations

import pandas as pd

from trading_engine.timing import EXECUTION_DELAY_BARS, label_span


def forward_open_return(bars: pd.DataFrame, horizon: int = 1) -> pd.Series:
    label_span(horizon)  # validates horizon
    entry = bars["open"].shift(-EXECUTION_DELAY_BARS)
    exit_ = bars["open"].shift(-(EXECUTION_DELAY_BARS + horizon))
    return (exit_ / entry - 1).rename(f"fwd_open_ret_{horizon}")


def make_direction_labels(bars: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """1.0 if the executable forward return is positive, 0.0 otherwise, NaN if unknown."""
    fwd = forward_open_return(bars, horizon)
    labels = (fwd > 0).astype("float64")
    return labels.where(fwd.notna()).rename(f"label_up_{horizon}")
