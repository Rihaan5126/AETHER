"""Single source of truth for decision, execution and label timing.

Convention (daily bars, see docs/methodology.md):

* Features for day ``t`` use only bars ``<= t`` and become available after the
  close of ``t`` (``FEATURE_AVAILABILITY = "close"``).
* An agent's decision for day ``t`` is submitted after that close and before the
  next session; it executes at the **open of t+1**.
* A position opened at ``open[t+1]`` and held for ``h`` sessions is marked out at
  ``open[t+1+h]``. The supervised label for horizon ``h`` is therefore
  ``open[t+1+h] / open[t+1] - 1``.

Because a sample at ``t`` "sees" prices up to index ``t + 1 + h``, training
samples within ``label_span(h)`` bars of a validation/test boundary must be purged.
"""

from __future__ import annotations

from typing import Final, Literal

FEATURE_AVAILABILITY: Final[Literal["close"]] = "close"
EXECUTION_DELAY_BARS: Final[int] = 1  # decision at close t -> fill at open t+1
EXECUTION_PRICE_FIELD: Final[Literal["open"]] = "open"
TRADING_DAYS_PER_YEAR: Final[int] = 252


def label_span(horizon: int) -> int:
    """Number of bars after the decision bar that a horizon-``h`` label touches."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    return EXECUTION_DELAY_BARS + horizon
