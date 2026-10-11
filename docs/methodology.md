# Methodology

AETHER is built around one rule: **nothing that chooses, tunes or evolves an
agent may see information that would not have been available at decision time,
and nothing may see the final test period at all.** This page explains how each
stage enforces that.

## 1. Data

- Bars are daily OHLCV in a canonical schema (`timestamp, symbol, open, high,
  low, close, volume, adjustment, source`).
- Cleaning (`data/validation.py`) drops duplicates, non-positive prices,
  impossible OHLC relationships and missing values, and records each issue with
  a severity and count. Fatal problems (too few rows, unsorted after repair)
  raise `DataQualityError`.
- A dataset's identity is the SHA-256 of its canonical bytes. It is stored once,
  immutably, with a manifest (source, parameters, retrieval time, symbols,
  range). Loading re-checks the hash.
- **Synthetic data** is a seeded regime-switching process (calm and turbulent
  regimes with Markov transitions, small return autocorrelation, a shared market
  factor across symbols). It exists to exercise the pipeline reproducibly. It is
  not calibrated to any real market, and an edge found on it is an edge on the
  generator.
- **Yahoo Finance** import is optional, disabled by default, and subject to
  Yahoo's terms. AETHER makes no attempt to work around any provider's limits.
- **Adjusted prices**: uploads can use `adj_close`, in which case open, high and
  low are rescaled by the same factor so returns stay consistent.

## 2. Timing convention

Defined once in `trading_engine.timing` and imported everywhere:

| Event | Time |
| --- | --- |
| Feature for day *t* | uses bars up to and including close(*t*) |
| Decision | made after close(*t*) |
| Execution | open(*t+1*) |
| Label for horizon *h* | `open[t+1+h] / open[t+1] − 1 > 0` |

The label is exactly the pre-cost return the simulated position earns, so the
model is trained on the quantity it is judged on. A label spans `1 + h` bars
after *t*.

## 3. Features

25 features in five groups (returns, trend, volatility, momentum, volume) plus a
regime group computed from a benchmark (the cross-sectional mean of the
dataset's symbols, or the asset itself when single-symbol). Every feature is a
rolling or exponentially weighted statistic ending at *t*; rows inside a
feature's lookback are `NaN` and dropped. Tests verify causality directly:
changing future prices never changes past feature values.

Scalers and models are fitted per fold on that fold's training rows only
(scikit-learn `Pipeline`, and the MLP's own scaler).

## 4. Splits

```
|<------------------- development ------------------->|<-- final test -->|
| warm-up | train₁ | gap | val₁ |                      |                  |
| warm-up | train₂ (expanding)  | gap | val₂ |         |                  |
| warm-up | train₃ (expanding)              | gap | val₃|                 |
```

- **Final test** is cut first: the most recent `test_fraction` of the data
  (default 20%) or everything from `test_start`. The development object
  physically excludes those bars. Tournament, evolution and optimisation
  functions raise `TypeError` if given anything but `DevelopmentData`.
- **Walk-forward folds** (default 3, expanding; rolling optional) start after a
  minimum training share (default 45%) and validate on consecutive windows.
- **Purging**: training labels are computed on bars truncated at the validation
  start, so a label whose holding window would touch validation prices is
  undefined and dropped.
- **Embargo**: an extra gap of `embargo_bars` (default 5) between the last
  training label and validation. Total gap = `h + 1 + embargo`.
- Each fold's validation simulation starts from cash. Folds are chained only
  for display.

## 5. Agents

| Family | Decision rule | Genome highlights |
| --- | --- | --- |
| Random | Seeded two-state Markov chain (in or out) | `p_long`, `switch_prob`, `max_allocation` |
| Buy & Hold | Fully (or `allocation`) invested | `allocation` |
| Momentum | MA cross, time-series momentum, or both | windows, lookback, allocation |
| Random Forest | Probability that the next holding-period return is positive | trees, depth, leaf size, features, threshold |
| XGBoost | Same target, gradient boosting | estimators, depth, learning rate, subsampling, regularisation |
| MLP (PyTorch) | Same target, small feed-forward net | layers, dropout, learning rate, weight decay, epochs |

Learned agents share a **position policy**: forecast → target weight via a
threshold (binary or proportional sizing) capped at `max_allocation`. Each
learned genome also selects a feature subset and an optional rolling training
window. The MLP uses early stopping on the last 20% of its training rows, with
a gap, never on validation data.

All genes have declared bounds, mutation ranges and log scaling where
appropriate. Invalid combinations are rejected by genome validators.

## 6. Simulation

Event order per session *i*: fill the pending order at open(*i*) → mark to
market at close(*i*) → read the target weight for day *i*, pass it through the
risk manager, and if the change exceeds `min_rebalance_weight` (or is a full
exit) submit an order for open(*i+1*).

- **Costs**: buys fill at `open × (1 + slippage)`, sells at
  `open × (1 − slippage)`; fee = `fee_bps × notional + fee_fixed`. Defaults: 1 bp
  fee, 5 bp slippage.
- **Constraints**: long-only, no leverage (both enforced by validators that
  reject `allow_short`/`allow_leverage`), cash-constrained sizing, fractional
  shares by default.
- **Risk manager**: maximum position, minimum cash, maximum daily turnover
  (partial execution), volatility targeting with the point-in-time 20-day
  realised volatility, and a drawdown circuit breaker with cooldown. Every
  intervention is recorded as a risk event; every refused order as a rejected
  order with a reason.

## 7. Evaluation and ranking

See [evaluation.md](evaluation.md). In short: every agent in a tournament is
evaluated on identical folds, costs and risk limits; the score rewards mean
fold Sharpe and penalises instability, drawdown and turnover; agents breaching
the drawdown limit are excluded; active strategies with too few round trips are
marked as insufficient evidence; Random and Buy & Hold are always present as
benchmarks and are never selected.

## 8. Evolution

Per generation:

1. Evaluate the population (plus benchmarks) on the validation folds.
2. Rank. Only `eligible` agents can be parents.
3. Copy the top `elite_count` unchanged (their cached score is reused, so an
   elite's score cannot drift by re-evaluation luck).
4. Fill the rest by tournament selection (size 3): mutate the winner (bounded
   Gaussian or log-scale steps, feature-set edits, policy changes, layer
   changes) or, with probability `crossover_rate`, cross it with a
   same-family mate.
5. Add `immigrant_count` fresh random genomes to keep diversity.
6. Record lineage: parents, operation, human-readable mutations, genome before
   and after, training seed.

**Caveat, stated in the UI as well**: every generation is selected on the same
validation folds, so the validation score of later generations is optimistic.
That is precisely why the final test exists.

## 9. Hyperparameter optimisation

Optuna (TPE sampler seeded from the experiment, median pruner) searches one
family's genes. The objective is the mean validation-fold Sharpe; each fold
reports an intermediate value so poor trials are pruned early. Pruned and
failed trials are recorded. The test period is not available to the study.

## 10. Final test

- Uses only final-generation agents whose models were refitted on the full
  development period and stored as verified artifacts; by default the top three
  eligible agents plus the Random baseline, always alongside Buy & Hold.
- Features for the test period are computed with development history as
  warm-up, so the first test day has a full lookback without peeking forward.
- The first run is the honest estimate. A second run requires
  `acknowledge_contamination=true` and permanently sets
  `test_contaminated` on the experiment; the dashboard explains why before
  asking.

## 11. Reproducibility

The manifest records the full config, dataset hash, code version (git commit,
`-dirty` if the tree had changes, or `AETHER_GIT_COMMIT` in containers),
dependency versions and platform. Seeds are explicit. Rerunning the same
config on the same dataset with the same seed reproduces the same agents,
scores and trades (tested).
