# Performance benchmarks

Measured on 2026-10-11 with `scripts/benchmarks/run_benchmarks.py` (raw
output: `docs/benchmarks/results.json`). Every number is a measurement on
this machine, mostly the median of 5 repeats. Nothing is estimated.

**Machine**: AMD Ryzen (Family 23 Model 104), 6 cores / 12 threads, 15.3 GB RAM,
Windows 11, Python 3.13.0, scikit-learn 1.9.1, XGBoost 3.4.1, PyTorch 2.14.1.
Data: SPY from the real Yahoo dataset (5,477 daily bars, 2005-2026).

## Data preparation

| Operation | Median |
| --- | ---: |
| Load the 5-symbol dataset (27,385 rows) **with** SHA-256 verification | 585 ms |
| Load without verification | 19.5 ms |
| Compute all 25 features for SPY (5,477 bars) | 44 ms |
| Load cached features (Parquet) | 9.6 ms |

Integrity verification costs about 0.57 s per load (hashing the canonical
bytes). It is kept on, because datasets are loaded once per experiment.
Feature caching cuts feature preparation from 44 ms to 9.6 ms (4.6x).

## Model training and prediction (one walk-forward fold)

3,521 training rows, 794 validation rows, default genomes.

| Family | Fit | Predict (794 rows) |
| --- | ---: | ---: |
| Random Forest (300 trees) | 3.67 s | 41 ms |
| XGBoost | 1.03 s | 7.9 ms |
| PyTorch MLP (with early stopping) | 12.5 s | 4.6 ms |
| Momentum (rule) | 0 s | 4.4 ms |

## Full agent evaluation (3 walk-forward folds, fit + simulate + metrics + training-window backtest)

| Family | Median wall time |
| --- | ---: |
| Momentum | 0.24 s |
| XGBoost | 2.94 s |
| MLP | 2.76 s |
| Random Forest | 9.20 s |

The MLP fold fit above (12.5 s on the largest fold) is longer than its
full three-fold evaluation (2.76 s). Early stopping ends most fits quickly;
the largest fold happened to train longer. Profiling (cProfile) showed
model fitting dominates evaluation: for Random Forest, 10.4 s of a 12.0 s
evaluation was inside scikit-learn tree building.

## Simulator throughput and the Phase 2 optimisation

Full SPY history (5,477 bars), same weights. "Before" is commit `a3660eb`;
"after" is the current code. The two produce identical outputs (verified on
40 cases / 2,535 trades, and on a full real-data research rerun).

| Workload | Before | After | Speed-up |
| --- | ---: | ---: | ---: |
| `simulate()`, trade-heavy random agent (284 trades) | 82.2 ms | 60.4 ms | 1.36x |
| `simulate()`, momentum (55 trades) | 59.3 ms | 52.2 ms | 1.14x |
| `compute_metrics()`, random agent | 59.3 ms | 18.5 ms | **3.2x** |
| `compute_metrics()`, momentum | 67.4 ms | 18.1 ms | **3.7x** |
| 100-strategy random null (1,095-bar window) | 3.51 s | 2.14 s | **1.64x** |

Throughput after the change: about 72,000 bars/s for the trade-heavy agent
and about 94,000 bars/s for momentum, in pure Python with an event loop and
risk checks on every bar.

## Tournament scaling (one generation, mixed families)

Equal numbers of Momentum, RF, XGBoost and MLP agents (the evolution
initial-population generator), SPY, 3 folds. Peak RSS includes worker
processes.

| Agents | Workers | Wall time | Seconds per agent | Peak RSS |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 32.2 s | 4.02 | 492 MB |
| 8 | 4 | 47.6 s | 5.94 | 1,678 MB |
| 16 | 1 | 62.8 s | 3.93 | 498 MB |
| 16 | 4 | 53.2 s | 3.32 | 1,913 MB |
| 32 | 4 | 101.9 s | 3.18 | 2,176 MB |
| 64 | 4 | 151.4 s | 2.37 | 2,193 MB |

Findings:

* **Parallelism has a fixed start-up cost.** Each spawned worker imports
  PyTorch, XGBoost and scikit-learn (about 0.3-0.4 GB each), so 4 workers
  are *slower* than serial at 8 agents and only 1.18x faster at 16. The
  benefit grows with the population: per-agent time falls from 4.0 s
  (serial) to 2.37 s at 64 agents. The laptop default (`max_workers=1`) is
  therefore the right choice for small experiments.
* **Memory is bounded by the worker count, not the population.** Peak RSS
  stays around 2.2 GB from 32 to 64 agents with 4 workers.
* Wall time is dominated by Random Forest and MLP fits (see above).

## API latency

In-process (FastAPI TestClient) against the local SQLite database with 13
experiments; 30 requests each.

| Endpoint | Median | p95 | Response |
| --- | ---: | ---: | ---: |
| `GET /health` | 3.2 ms | 4.3 ms | <0.1 KB |
| `GET /experiments` | 18.8 ms | 21.2 ms | 5.2 KB |
| `GET /experiments/{id}` | 33.6 ms | 39.0 ms | 9.9 KB |
| `GET /tournaments/{id}/leaderboard` | 10.0 ms | 11.9 ms | 10.3 KB |
| `GET /tournaments/{id}/trades?limit=50` | 12.7 ms | 15.6 ms | 18.1 KB |
| `GET /agents?limit=200` | 65.1 ms | 68.7 ms | 171 KB |
| `GET /agents/{id}` | 9.0 ms | 10.3 ms | 1.3 KB |
| `GET /tournaments/{id}/equity-curves` (600 pts) | 119.6 ms | 130.3 ms | 515 KB |
| `GET /experiments/{id}/manifest` (full export) | 37.1 ms | 45.8 ms | 190 KB |
| `GET /experiments/{id}/analytics` (with bootstrap) | 2,734 ms -> **1,435 ms** | 3,014 ms | 21.8 KB |

### Optimisation found by this benchmark: analytics

The analytics endpoint was 20-100x slower than every other endpoint.
Profiling showed it was the new bootstrap: for each of the 9 final-generation
agents it regenerated identical resampling indices (1,000 × ~1,400; 152 ms
each), plus the matching Sharpe computations.

* A first attempt (vectorising the index generator) gave **no speed-up**
  (82 ms vs 82 ms, measured) and was reverted. The loop was not the cost;
  the repetition was.
* The fix: generate the indices once per request and share them across
  agents (`bootstrap_performance(..., indices=...)`). The output is
  **identical** (checked against the previously exported analytics of all
  13 experiments), and the median latency fell from 2,734 ms to 1,435 ms
  (1.9x). Sharing also gives common random numbers across agents.
* What remains is mostly Sharpe ratios over the 1,000 × 1,400 resampled
  matrices. Caching analytics for completed experiments would remove it
  entirely. Not done: it adds invalidation logic when a final test is added.

## Storage

| Item | Size |
| --- | ---: |
| SQLite database (13 experiments, 390 agents) | 15.5 MB |
| Dataset store (2 real + synthetic datasets, Parquet + manifests + feature caches) | 11.4 MB in 18 files |
| Model artifacts, all experiments (final generations only) | 30.1 MB |
| Median artifact: XGBoost | 667 KB (native JSON) |
| Median artifact: Random Forest | 388 KB (joblib) |
| Median artifact: MLP | 7.6 KB (state dict + scaler JSON) |
| Median artifact: momentum / baselines | 0.5-0.6 KB (genome only) |
| Static dashboard snapshot (13 experiments) | 106 MB raw; each file is served Brotli-compressed (a 1.28 MB curve file transfers about 150 KB) and a page loads only the files it needs |

Equity curves are stored as Parquet artifacts referenced from the database
(one file per tournament), not as rows. That keeps the database at 15.5 MB
for 390 agents.

## Reproducing

```bash
uv run python scripts/benchmarks/run_benchmarks.py                  # full suite (~10 min)
uv run python scripts/benchmarks/run_benchmarks.py --only sim       # simulator only
PYTHONPATH=<old checkout>/packages/trading_engine/src \
  uv run python scripts/benchmarks/run_benchmarks.py --only sim     # "before" numbers
```

Timings depend on hardware and background load. The CPU was at 23.5%
utilisation when the run started (OS background activity), so treat small
differences (<10%) as noise.
