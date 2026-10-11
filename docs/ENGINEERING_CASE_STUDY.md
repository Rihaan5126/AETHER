# Engineering case study: AETHER

A full-stack ML research platform that trains, evolves and evaluates
populations of trading agents, and is designed above all to tell when a
result is not real. Live read-only dashboard:
https://aetherbyrihaan.vercel.app · Code: https://github.com/Rihaan5126/AETHER

Every number in this document was measured or counted from the repository;
sources are linked.

## 1. The engineering problem

"Train some models on stock prices and show a backtest" is easy, and almost
always misleading. The hard problems are:

1. **Leakage and look-ahead.** A feature, label, scaler or fill that sees one
   day into the future makes any model look brilliant.
2. **Selection bias.** Evaluating many configurations and reporting the best
   guarantees an impressive number even when nothing works.
3. **Reproducibility.** Results that cannot be regenerated bit for bit cannot
   be audited.
4. **Systems constraints.** Training is CPU-heavy and slow; the web layer
   must never block on it or let the public trigger unbounded compute.

The goal was a platform where a positive result has to survive all of
these, and where a negative result ("nothing beats buy-and-hold") is
reported as readily as a positive one.

## 2. Architecture

```
Next.js dashboard  --(Zod-validated REST)-->  FastAPI service  --> SQL (Postgres/SQLite)
   |  snapshot mode: static JSON on Vercel            |  jobs table (durable state)
                                                      v
                                    in-process pool / Celery + Redis worker
                                                      v
                          trading_engine (pure Python: no web or DB imports)
     data -> features -> agents -> simulator -> metrics -> tournaments -> evolution
```

| Layer | Size | Responsibility |
| --- | ---: | --- |
| `packages/trading_engine` | 4,851 lines | Data validation and versioning, 25 point-in-time features, 6 agent families, event-driven simulator, risk layer, metrics, walk-forward splits, tournaments, evolution, Optuna, robustness statistics |
| `services/api` | 4,389 lines | FastAPI, SQLAlchemy + Alembic, durable jobs, persistence, analytics, export, security |
| `apps/web` | 5,032 lines | Next.js 16 / React / TypeScript dashboard (9 pages), snapshot mode for static hosting |
| Tests | 1,741 lines (Python) | 128 pytest tests, 33 Vitest tests, a Playwright end-to-end suite in CI |

The engine has no web or database dependencies, so every research property
is testable in isolation. The API owns persistence and long-running jobs
(requests return a job id; work runs in a bounded pool or a separate Celery
worker).

## 3. Key design decisions

| Decision | Why | Trade-off |
| --- | --- | --- |
| Custom event-driven simulator instead of VectorBT | Needed explicit signal, submission and fill timestamps; a risk layer that can clip or reject orders with reasons; fees with fixed and proportional parts; analytic testability | Pure Python: ~72-94k bars/s, more than enough for daily data |
| One timing module (`timing.py`) | Labels (`open[t+1+h]/open[t+1]-1`), purge gaps and the simulator (decide at close *t*, fill at open *t+1*) import the same constants, so they cannot drift apart | None |
| Final test locked **by type** | Tournaments and evolution only accept `DevelopmentData`, which cannot contain test bars; a second final test run requires explicit acknowledgement and permanently flags the experiment | Slightly more ceremony |
| Typed, bounded genomes (Pydantic discriminated union) | Mutation can never create an invalid model; configurations serialise exactly | One schema per family |
| Content-hashed, immutable datasets | Every experiment points at exact bytes; re-downloads produce new versions | 0.57 s SHA-256 verification per load |
| Parquet artifacts referenced from SQL | Equity curves and trades do not bloat the database (15.5 MB for 390 agents) | Two storage systems |
| Static snapshot mode for the public site | Vercel hosts the dashboard with zero backend; no public endpoint can start training | Data updates require an export and push |

## 4. Difficult technical challenges

**Proving there is no look-ahead.** Unit tests on features are not enough;
leakage can enter through agents, labels or fills. `test_lookahead.py`
corrupts every bar after day *t* (prices x1.7 with a random walk, volume x5),
recomputes features and checks that all 6 agent families make
**bit-identical** decisions up to *t* and that the simulated portfolio is
unchanged. No leakage was found.

**Measuring whether evolution actually works.** Elitism makes the best
validation score rise by construction, so the obvious chart is meaningless.
The pre-registered study (`docs/EVOLUTION_RESEARCH.md`) added an inner
holdout and an equal-budget random-search control across 5 symbols × 3
seeds. Selection score rose in 13 of 15 runs; the evolved champion did
**not** beat the generation-0 champion on unseen data (6 wins / 7 losses)
and lost to random search (4 / 9). On the locked 2022-2026 test, **0 of 15**
champions beat buy-and-hold. The platform reports exactly that.

**Uncertainty that is actually calibrated.** Daily returns cluster in
volatility, so i.i.d. bootstrap intervals are too narrow. I implemented a
stationary block bootstrap and tested its assumptions: 90% intervals covered
a known true Sharpe in 80-97% of 150 simulated trials, and on AR(1) data the
intervals widened 1.3-2.1x versus i.i.d. resampling (theory: 1.73x).

**Bit-identical reproducibility across processes.** With seeded generators,
single-threaded XGBoost/RF and deterministic PyTorch, an identical
configuration reproduces identical results through the full API, and
parallel evaluation equals serial. A full real-data research run re-executed
from scratch matched on **797 of 797** values. An earlier attempt matched
796 of 797. The one difference (2×10⁻¹⁶) came from averaging over a Python
`set`, whose order changes with per-process hash randomisation. It is now
sorted.

## 5. Bugs found by auditing my own system

| Bug | How it was found | Impact | Fix |
| --- | --- | --- | --- |
| Evolution produced **behavioural clones**: mutations changed genes the agent ignores (e.g. `trend_lookback` under the `ma_cross` rule) | Two "different" agents with identical scores in stored results | Duplicates took both elite slots, faking diversity | `behaviour_key()` identity; mutation must change behaviour; no duplicates per generation |
| Simulator **negative-cash false positive**: fixed tolerance 1e-9 | A new in-sample backtest compounded equity past ~1e7; float residue exceeded the tolerance | Correct trades raised errors; 6 of 150 showcase results were wrongly excluded | Tolerance relative to cash; regression test; affected study attempt archived and re-run |
| Host paths in stored tracebacks reached the public snapshot | The exporter's own safety check | User name exposed in one commit | Paths stripped at the source and scrubbed in the exporter |
| API could not start on Windows (Unix-only `resource`, `os.getloadavg`) | Running the suite on Windows | Total outage on Windows | `psutil` |
| Unbounded job queue | Security audit | ~43,000 CPU-heavy jobs/day possible despite rate limiting | `AETHER_MAX_PENDING_JOBS` -> 429 |

## 6. Quantitative performance

From `docs/PERFORMANCE_BENCHMARKS.md` (6-core / 12-thread AMD laptop CPU):

* **Profiling first.** cProfile showed 87% of a Random Forest evaluation is
  scikit-learn tree building. Within AETHER's own code, timestamp formatting
  and round-trip detection dominated simulation-heavy paths. Rewriting them
  (output identity verified on 40 cases and 2,535 trades) sped up metrics
  **3.2-3.7x** and the 100-strategy null distribution **1.64x** (3.51 s ->
  2.14 s).
* **Benchmark-driven fix.** The analytics endpoint took 2.7 s. A vectorised
  rewrite of the bootstrap loop gave no gain and was reverted. Profiling
  showed identical resamples regenerated per agent; sharing them halved
  latency (2,734 -> 1,435 ms) with identical output.
* **Scaling.** 64 agents × 3 folds in 151 s with 4 workers (2.37 s/agent
  vs 4.0 s serial). Process start-up (each worker imports PyTorch) makes 4
  workers slower than serial at 8 agents. Peak memory stays around 2.2 GB
  from 32 to 64 agents.
* **API.** Most endpoints answer in 3-65 ms (median, in-process); a full
  reproducibility export takes 37 ms.

## 7. Testing strategy

| Layer | What is tested |
| --- | --- |
| Simulator | Analytic accounting (fees, slippage, cash conservation, partial and integer sizing, round trips, circuit breaker), no look-ahead fills, large-equity regression |
| Data and features | Validation, calendar gaps, immutability and hash verification, poisoned-future feature tests, rolling windows vs manual |
| Training | Labels match the executable holding period; purged folds; training labels never see validation prices |
| Agents | All 6 families: valid weights, real fitting, determinism, save/load, tamper detection |
| Evolution | Bounds, behaviour-changing mutations, no duplicates, elite preservation, reproducible selection, final-test isolation, parallel = serial |
| Statistics | Bootstrap coverage and dependence, paired excess, null determinism, exact cost sensitivity, warning thresholds |
| API | Validation, status codes, idempotency, API-key enforcement, rate limit, queue cap, interrupted-job recovery, full lifecycle, export contents, exact rerun reproducibility |
| Frontend | Formatting, series maths, genome diff, builder validation, API client, snapshot resolver, components; Playwright smoke across all pages in CI |
| Deployment | `scripts/verify_deployment.py`: HTTPS, HSTS, CSP, CORS, auth, read-only snapshot (9/9 API checks pass in production mode) |

CI (GitHub Actions) runs ruff, mypy, an Alembic migration check, pytest,
ESLint, tsc, Vitest, `next build` and Playwright on every push. The suite also
passes on Windows 11.

## 8. Limitations

* Single-asset, long-only, daily bars, one-day horizon; regime features are
  self-referential in the single-asset setting.
* The product does not yet hold out a selection-free window per experiment
  (the research harness does).
* Rate limiting is per process and keyed on the client address; job
  timeouts are cooperative.
* Bit-identity is measured on one machine; across OS, CPU or library
  versions, expect last-bit differences.
* Yahoo data is unofficial; no survivorship-free universe or point-in-time
  fundamentals.

## 9. Future improvements

1. Per-experiment selection-free holdout (`evolution_holdout_fraction`).
2. Deflated Sharpe Ratio / White's Reality Check for formal
   multiple-comparison correction.
3. Multi-asset portfolios with real benchmark regime features.
4. Cache analytics for completed experiments; prefork workers to remove
   per-tournament import cost.
5. Gateway rate limiting and hard worker time limits for a public API.
