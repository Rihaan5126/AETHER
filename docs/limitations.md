# Limitations

AETHER is a research and teaching platform for historical simulation. These
limits are deliberate or known; read them before trusting any number.

## Scope

- **No live or paper-broker trading.** There is no order routing, broker SDK,
  account credential or real-time data feed, and none should be added.
- **Single asset per experiment, daily bars, long-only, no leverage.** No
  portfolio construction across symbols, intraday data, shorting, options or
  futures.
- **No reinforcement-learning agents.** The agent interface supports them, but
  none ship.

## Market realism

- Execution is at the next open with a fixed basis-point slippage. Real slippage
  depends on liquidity, volatility, order size and time of day. No market
  impact, partial fills from liquidity, halts or auctions are modelled.
- Fees are a simple proportional plus fixed charge. No taxes, borrow costs,
  financing or currency effects.
- Volume is used only as a feature; positions are not capped by traded volume.
- Corporate actions are handled only through adjusted prices supplied by the
  data; dividends are not modelled separately.
- Survivorship bias is not addressed: a dataset contains whatever symbols you
  give it.

## Statistics

- Walk-forward validation with a handful of folds yields few independent
  observations. Fold Sharpe ratios are noisy, and the ranking score inherits
  that noise.
- Evolution reuses the same validation folds every generation, so later
  generations overfit them. Validation scores of evolved agents are
  optimistically biased; only the final test is an unbiased estimate, and only
  the first time it is run.
- The multiple-comparisons threshold is an approximation that assumes
  independent strategies. Related agents (parents and children) are not
  independent. No Deflated Sharpe Ratio, White's Reality Check or SPA test is
  implemented.
- Regime and parameter-sensitivity analytics are descriptive, on small
  samples, and not significance tests.
- One locked test period is one draw of history. Good test performance can
  still be luck.

## Data

- The synthetic generator is not calibrated to any real market. Findings on it
  are findings about the generator.
- Yahoo Finance data is unofficial, may change or disappear, and is subject to
  Yahoo's terms. The integration is optional and off by default.
- Uploaded files are validated for structure, not for correctness of the
  prices themselves.

## Engineering

- The local artifact store assumes a shared filesystem (the Compose volume)
  between API and workers. Multi-host deployments need an object-store
  implementation of the artifact interface.
- Only final-generation models are stored as artifacts (earlier generations
  keep metrics, trades and equity curves, not fitted models), to bound disk use.
- Rate limiting is in-memory per API process. Behind several API replicas, use
  a shared limiter at the proxy.
- The API key protects mutating endpoints only; read endpoints are open. Put
  the service behind authentication before exposing it beyond a trusted
  network. `NEXT_PUBLIC_API_KEY` is visible to anyone using the dashboard, so it
  is only suitable for local use.
- Job cancellation is cooperative: a single long model fit finishes (bounded by
  the per-fit time budget) before the job stops.
- Run-time estimates are not shown; the experiment builder reports relative
  workload (fits and simulations) instead, because timings vary widely by
  machine.

## Not a recommendation

Nothing in AETHER is investment advice. A high rank, a positive final test, or
a strategy that beats buy-and-hold in simulation is not evidence that it will
make money in the future.

## Known implementation limitations (Phase 2 audit)

Recorded with evidence in `docs/PHASE2_AUDIT.md`:

* **Selection-free holdout is not automatic.** Experiments rank every
  generation on the same validation folds, so the best validation score can
  only rise. The research harness (`scripts/research/phase2_study.py`)
  measures generalisation on an inner holdout; the product does not yet do
  this per experiment.
* **Random baseline and window length.** The random agent draws its random
  numbers in two blocks sized to the window, so the same day can get a
  different decision in a longer window. This is not look-ahead (it never
  reads prices), but it is not prefix-stable.
* **Regime features are self-referential.** In the single-asset setting each
  asset is its own "benchmark", so `bench_*` features repeat asset features.
* **Rate limiting is per process, keyed on the client address.** Behind a
  proxy, every client shares one address unless proxy headers are trusted.
* **Job timeouts are cooperative.** A single long model fit cannot be
  interrupted mid-fit; a worker-level hard limit is needed in production.
* **Cross-platform bit-identity is not promised.** Results are bit-identical
  on one machine (measured); across operating systems, CPUs or library
  versions, expect last-bit floating-point differences
  (`docs/REPRODUCIBILITY.md`).
