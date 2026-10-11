# Architecture

```
            browser
               │  HTTP (zod-validated JSON, polling while jobs run)
      ┌────────▼────────┐
      │  apps/web       │  Next.js 16 (App Router), TanStack Query, Recharts
      └────────┬────────┘
               │  /api/v1
      ┌────────▼────────┐        enqueue job id        ┌──────────────────┐
      │  services/api   │ ───────────────────────────▶ │ Redis (queue)    │
      │  FastAPI        │                              └────────┬─────────┘
      │  aether_api     │                                       │
      └───┬─────────┬───┘                              ┌────────▼─────────┐
          │         │  same job function (run_job)     │ Celery worker    │
          │         └────────────────────────────────▶ │ aether_api.worker│
          │                                            └───┬──────────┬───┘
   ┌──────▼──────┐   metadata, jobs, results              │          │
   │ PostgreSQL  │◀───────────────────────────────────────┘          │
   └─────────────┘                                                   │
   ┌─────────────────────────────┐   parquet datasets, equity curves,│
   │ shared volume (/data)       │◀──model artifacts (SHA-256) ──────┘
   └─────────────────────────────┘
                 ▲
                 │ imports
   ┌─────────────┴──────────────┐
   │ packages/trading_engine    │  pure Python research engine
   └────────────────────────────┘
```

## Components

### `packages/trading_engine` (research engine)

No web, database or queue dependency, so every research rule is testable in
isolation.

| Module | Responsibility |
| --- | --- |
| `timing` | The single timing convention: features at close *t*, execution at open *t+1*, label span `1 + h`. |
| `data` | Canonical OHLCV schema; providers (synthetic, file, optional Yahoo); validation and cleaning with an issue report; content-hashed immutable `DatasetStore`. |
| `features` | 25 declared `FeatureSpec`s with lookbacks; `compute_features` is causal by construction (rolling windows ending at *t*). |
| `training` | Direction labels, `DevelopmentData` / `TestData` split, walk-forward folds with purge and embargo, `training_matrix`. |
| `agents` | Typed genomes (Pydantic discriminated union), `BaseTradingAgent` (fit, forecast, position policy, save/load), baselines and ML agents. |
| `risk`, `simulation` | Hard risk limits and the event-driven simulator (trades, rejected orders, risk events, equity). |
| `evaluation` | Metrics with explicit undefined values and statistical warnings. |
| `tournaments` | Fold evaluation of one agent, multi-agent tournaments in a process pool, ranking policy. |
| `evolution` | Mutation/crossover operators and the generational loop with lineage. |
| `optimization` | Optuna studies on validation folds. |
| `experiments` | `ExperimentConfig`, workload estimate, manifest, data preparation, final test. |
| `storage` | Local artifact store with SHA-256 verification. |

### `services/api` (aether_api)

- **Routes** (`/api/v1`): `system` (health, status), `datasets` (list, import job,
  upload, prices), `experiments` (create, estimate, start, cancel, final test,
  optimise, lineage, analytics), `tournaments` (create single tournament,
  leaderboard, equity curves, trades), `agents` (list, detail, metrics,
  lineage), `jobs`.
- **Errors**: one envelope, `{"error": {"code", "message", "details"}}`.
- **Persistence**: SQLAlchemy 2 models (`database/models.py`), Alembic migration
  `0001_initial_schema`. JSON columns use JSONB on Postgres. Large time series
  (equity curves) are Parquet files referenced by path and hash.
- **Jobs** (`services/jobs.py`): the `jobs` table is the source of truth for
  state (`queued → running → completed | failed | cancelled`), progress,
  heartbeat and errors. Two interchangeable backends run the same `run_job`
  function: a bounded in-process thread pool (`inline`, for tests and laptops)
  and Celery on Redis (`celery`, Docker and deployment). Jobs are enqueued only
  after the creating transaction commits. Cancellation is cooperative; a job
  exceeding `max_job_seconds` fails with a timeout; on start-up, running jobs
  with a stale heartbeat are marked failed ("interrupted"), never silently
  resumed. Each job type registers a failure hook that marks its related
  records (experiment, final-test run, study).
- **Security**: optional `X-API-Key` (constant-time comparison) and a
  per-client rate limit on every mutating endpoint; server-side caps on
  population, generations, Optuna trials, upload size, job duration, worker
  processes and concurrent jobs. The service refuses to start with
  `AETHER_ENVIRONMENT=production` and no API key. CORS allows GET/POST from
  configured origins only.

### `apps/web`

Client-rendered pages over the API. The API client validates every response
with zod (`lib/api/schemas.ts`), so a contract drift fails visibly instead of
rendering wrong numbers. Experiment, generation and agent selections live in
the URL so every view is linkable. Queries poll every 2 s only while a job is
active. Charts are Recharts; the lineage graph is hand-written SVG with
keyboard support.

## Data model (main tables)

`datasets` → `experiments` (config, manifest, periods, status, contamination
flag) → `tournaments` (one per generation, config fingerprint, folds) →
`tournament_results` (rank, score, components, metrics, fold metrics,
warnings, risk events) → `trades`; `agents` (genome, seed, origin, artifact
path and hash) with `model_lineage` edges (operation, mutations, genome before
and after); `training_runs`; `equity_curves`; `final_test_runs`;
`optimization_studies` and `optimization_trials`; `jobs`.

Completed experiment records are immutable: starting a completed experiment is
rejected, and changes require a new experiment.

## Execution flow of an evolution experiment

1. `POST /experiments` validates the request against server limits, builds an
   `ExperimentConfig`, stores it (idempotent with `Idempotency-Key`).
2. `POST /experiments/{id}/start` creates a `run_experiment` job and enqueues it.
3. The worker writes the manifest, loads the dataset by hash (integrity
   checked), computes or loads cached features, and splits into
   `DevelopmentData` and `TestData`. Only `DevelopmentData` is passed on.
4. `run_evolution` evaluates each generation with `run_tournament` (process pool,
   per-fit time budget), ranks it, and after each generation the API persists
   agents, lineage, results, trades and equity curves, then reports progress
   and checks for cancellation.
5. Final-generation agents are refitted on the full development period and
   saved as verified artifacts; those are the only models the final test may
   use.
6. `POST /experiments/{id}/final-test` (separate job) loads those artifacts
   after hash verification and simulates the test period exactly once per run.

## Determinism

Seeds flow from the experiment seed into the evolution RNG, per-agent training
seeds and benchmark seeds. scikit-learn and XGBoost run with `n_jobs=1`;
PyTorch uses one thread, `use_deterministic_algorithms(True)` and seeds set
before layer construction. Tests assert identical results for identical
(config, data, seed).

## Why a custom simulator

See [PLAN.md](PLAN.md#key-architecture-decisions): explicit signal and
execution timestamps, a risk layer that records rejected and clipped orders,
cash-constrained fractional sizing and a circuit breaker are simpler to get
exactly right, and to test analytically, in a few hundred lines than through a
vectorised backtesting library.

## Extension points

- New agent family: add a genome model to `agents/genomes.py` and an agent
  class to `AGENT_CLASSES`; mutation bounds come from the field metadata.
- New data source: implement `MarketDataProvider.fetch_ohlcv`.
- Remote artifact storage: implement the `LocalArtifactStore` interface
  (`commit`, `open_verified`).
- Reinforcement-learning agents are deliberately not shipped; they would plug
  in as another agent family that consumes the same `Observations`.
