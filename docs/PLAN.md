# AETHER implementation plan

This is the working plan recorded before implementation (spec §32 step 2). It is
kept short on purpose; the detailed design lives in `architecture.md`,
`methodology.md` and `evaluation.md`.

## Environment findings

- Empty repository; nothing to preserve.
- Python 3.13, Node 22, uv, pnpm, PostgreSQL and Redis binaries available; 4 CPU, no GPU.
- Docker CLI present without a running daemon at first; development verified
  the services directly (Postgres + Redis + uvicorn + Celery + Next.js), and the
  Compose stack was built and exercised once a daemon was started.

## Key architecture decisions

1. **Custom event-driven simulator instead of VectorBT.** The MVP needs explicit
   signal/submission/execution timestamps, a risk layer that can reject or clip
   orders with recorded reasons, cash-constrained fractional sizing, fees with a
   fixed + proportional component, and a drawdown circuit breaker. Those are easy
   to get exactly right (and to test analytically) in ~300 lines of plain Python;
   wrapping VectorBT would hide the accounting we most need to audit. Daily bars
   x a few thousand days x tens of agents is cheap enough without vectorisation.
2. **Timing convention (single source of truth in `trading_engine.timing`).**
   Features use bars up to and including the close of day *t*. The decision is
   submitted after that close and executes at the open of *t+1*. The label for
   horizon *h* is `open[t+1+h] / open[t+1] - 1 > 0` — the exact return the
   simulated position earns before costs. Labels, splits (purge/embargo of
   `h + 1` bars) and the simulator all import the same constants.
3. **Pure-Python engine package (`packages/trading_engine`)** with no web or
   database dependency, so research code is testable in isolation. The API
   (`services/api`, package `aether_api`) owns persistence, jobs and HTTP.
4. **Jobs:** durable `jobs` table is the source of truth for status/progress.
   Execution backend is Celery+Redis in Docker/production, with an in-process
   bounded thread executor (`JOB_BACKEND=inline`) for tests and laptop dev.
   Both call the same task function. Cancellation is cooperative (checked
   between agents/generations).
5. **Storage:** relational metadata in PostgreSQL (SQLite works for tests);
   datasets as content-hashed Parquet with JSON manifests; equity curves and
   predictions as Parquet artifacts; model artifacts on the filesystem behind an
   `ArtifactStore` interface with SHA-256 verification before every load.
6. **Typed genomes:** one Pydantic model per family (discriminated union on
   `family`), bounds declared on fields and enforced by mutation operators.
7. **Final test lock:** evolution and ranking receive a `DevelopmentData`
   object that physically excludes the test period. The final evaluation is a
   separate, explicit operation recorded once; repeating it marks the experiment
   as test-contaminated.
8. **Frontend:** Next.js 16 App Router, Tailwind v4, shadcn-style components
   (vendored source), TanStack Query polling, Zod-validated API client,
   Recharts for charts, a hand-written SVG lineage graph.

## First vertical slice

Synthetic data -> dataset version -> features -> Random/Buy&Hold/Momentum ->
simulator -> metrics -> tournament ranking -> persisted via API -> leaderboard
page. Then ML agents, evolution, the remaining dashboard pages, Optuna.

## Main risks

- Look-ahead leakage through labels/scalers -> single timing module, purge gaps,
  pipelines fitted per fold, leakage tests with "poisoned future" data.
- Torch CPU wheel size / install time -> MLP kept small; seeded and single-threaded.
- Non-determinism (threads, XGBoost, torch) -> fixed seeds, `n_jobs` pinned,
  `torch.use_deterministic_algorithms(True)`; determinism tests.
- Scope: RL module is left as a documented extension point, not shipped.
