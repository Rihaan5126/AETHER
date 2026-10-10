"""Non-learned agents: random null baseline, buy-and-hold benchmark, momentum rules."""

from __future__ import annotations

import numpy as np
import pandas as pd

from trading_engine.agents.base import BaseTradingAgent, Observations
from trading_engine.agents.genomes import BuyAndHoldGenome, MomentumGenome, RandomGenome


class RandomAgent(BaseTradingAgent):
    """Seeded random long/flat decisions with persistence (a two-state Markov chain).

    It has no information about the market. Any apparent skill it shows in a
    tournament is pure luck, which makes it the reference for selection bias.
    The random stream is keyed on (seed, first decision date), so re-running the
    same window reproduces the same decisions.
    """

    family = "random"
    genome: RandomGenome

    def forecast(self, observations: Observations) -> pd.Series:
        idx = observations.features.index
        if len(idx) == 0:
            return pd.Series(dtype="float64", index=idx)
        rng = np.random.default_rng([self.seed, int(idx[0].toordinal())])
        g = self.genome
        draws = rng.random(len(idx))
        switch = rng.random(len(idx))
        state = np.empty(len(idx))
        current = 1.0 if draws[0] < g.p_long else 0.0
        for i in range(len(idx)):
            if switch[i] < g.switch_prob:
                current = 1.0 if draws[i] < g.p_long else 0.0
            state[i] = current
        return pd.Series(state, index=idx)

    def position_policy(self, forecast: pd.Series) -> pd.Series:
        return forecast * self.genome.max_allocation


class BuyAndHoldAgent(BaseTradingAgent):
    """Passive benchmark: hold a constant allocation. Not a learned model."""

    family = "buy_and_hold"
    genome: BuyAndHoldGenome

    def forecast(self, observations: Observations) -> pd.Series:
        return pd.Series(self.genome.allocation, index=observations.features.index)


class MomentumAgent(BaseTradingAgent):
    """Parameterised trend following (traditional rule, not machine learning).

    * ``ma_cross``: long when SMA(fast) > SMA(slow)
    * ``time_series_momentum``: long when the trailing ``trend_lookback`` return > 0
    * ``both``: long only when both agree
    """

    family = "momentum"
    genome: MomentumGenome

    def forecast(self, observations: Observations) -> pd.Series:
        g = self.genome
        close = observations.history["close"]
        fast = close.rolling(g.fast_window).mean()
        slow = close.rolling(g.slow_window).mean()
        cross = (fast > slow).astype(float).where(slow.notna())
        tsmom = (close.pct_change(g.trend_lookback, fill_method=None) > 0).astype(float)
        tsmom = tsmom.where(close.shift(g.trend_lookback).notna())
        signal = {"ma_cross": cross, "time_series_momentum": tsmom, "both": cross * tsmom}[g.rule]
        return signal.reindex(observations.features.index)

    def position_policy(self, forecast: pd.Series) -> pd.Series:
        return forecast * self.genome.max_allocation
