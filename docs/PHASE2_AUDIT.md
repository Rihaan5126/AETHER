# Phase 2 audit

Scope: the whole repository as of commit `86fee1a` (end of Phase 1), audited
by reading the code, profiling it, and adding tests that try to break the
properties the platform claims. Each finding has a severity, its evidence,
and its resolution. "Fixed" means the change is committed and covered by a
test, unless a test is impossible.

Severity scale: **Critical**: results are wrong or a public endpoint can
cause harm. **High**: research conclusions can be misleading, or evolution
does not do what it claims. **Medium**: a real gap in safety, robustness or
statistics. **Low**: a correctness nuance, inefficiency or documentation gap.

## Summary

| Severity | Found | Fixed | Documented / open |
| --- | --- | --- | --- |
| Critical | 0 | 0 | 0 |
| High | 2 | 1 | 1 (mitigated by the inner-holdout study) |
| Medium | 10 | 8 | 2 |
| Low | 8 | 4 | 4 |

No critical issue was found. In particular, no look-ahead or leakage was
found at the feature, label, agent or portfolio level. The new tests
corrupt every bar after day *t* and show that decisions and the portfolio up
to *t* are bit-identical for all six agent families.

## Findings

### High

**H1. Evolution produced behavioural clones. Fixed.**
`mutate()` accepted any child whose genome differed from the parent, even
when the only change was a gene the agent never reads (e.g. `trend_lookback`
under the momentum `ma_cross` rule). In the stored SPY experiment, momentum
`G2·00` (lookback 60) and `G2·01` (lookback 37) were the same `ma_cross
20/100` strategy. They had identical scores (0.2686) and took both elite
slots. That wasted population slots and overstated diversity.
*Fix:* `behaviour_key()` (inactive genes dropped; seed included only for
learned families). `mutate()` must change it, and `_next_population()` never
admits two agents with the same key. Tests: key semantics; 300 chained
mutations per family always change behaviour; no duplicates over 3
momentum-only generations. The dashboard labels inactive genes "(unused)".

**H2. "Evolution improves the best score" is measured on the data used for
selection. Mitigated; open in the product.**
Elitism makes the best validation score non-decreasing *by construction*,
because every generation is ranked on the same walk-forward folds. A rising
curve in the Evolution Lab is therefore not evidence that evolution finds
better strategies. The dashboard already said "validation folds are reused
every generation".
*Mitigation:* `scripts/research/phase2_study.py` adds an **inner holdout**
(the last 20% of development data, never seen by selection), an
**equal-budget random-search control**, and paired comparisons across 5
symbols × 3 seeds. Results are in `docs/EVOLUTION_RESEARCH.md`. *Open:* the
API's experiment flow does not yet hold out a selection-free window
automatically. That is the main recommended product change (see
"Recommended next steps").

### Medium

| ID | Finding | Evidence | Resolution |
| --- | --- | --- | --- |
| M1 | API failed to start on Windows | `import resource` (Unix-only) in `services/runner.py`; `os.getloadavg()` in `routes/system.py` | **Fixed** (psutil); suite runs on Windows and Linux CI |
| M2 | Unbounded job queue | Rate limit capped requests per minute, not total queued CPU work (~43k jobs/day possible) | **Fixed**: `AETHER_MAX_PENDING_JOBS` (default 5) -> 429 `queue_full`; test |
| M3 | No Content-Security-Policy on the live site | `verify_deployment.py` against Vercel | **Fixed**: production CSP, `connect-src` restricted to the API origin; no violations on any page |
| M4 | Public `/system/status` revealed absolute storage paths | Read the route; snapshot export caught it | **Fixed**: paths hidden in production; test; the snapshot exporter refuses files containing local paths |
| M5 | No route error boundary | `apps/web/app` had no `error.tsx` | **Fixed**: `error.tsx` (retry) and `not-found.tsx` |
| M6 | No uncertainty quantification | Point estimates only | **Fixed**: stationary block bootstrap (validated coverage and dependence handling) in the analytics API, dashboard and research harness |
| M7 | No train -> validation gap | Training metrics were classification-only | **Fixed**: per-fold in-sample backtest (`train_sharpe_ratio`, `train_sharpe_mean`), shown next to validation and test |
| M8 | Missing warnings for implausible results | No flag for a high Sharpe on thin evidence or for overfitting | **Fixed**: `suspiciously_strong`, `overfit_gap`; unit tests |
| M9 | Rate limiter keys on `request.client.host` | Behind a proxy every client shares one address | **Open** (documented): run uvicorn with `--proxy-headers` or limit at the gateway |
| M10 | Job timeout is cooperative | `max_job_seconds` is checked between units of work; one long fit cannot be interrupted | **Open** (documented): Celery `time_limit` on the worker; MLP epochs are already bounded by `max_fit_seconds` |

### Low

| ID | Finding | Resolution |
| --- | --- | --- |
| L1 | Random baseline decisions depend on window length (two RNG blocks sized to the window). Not look-ahead (no market data is read), but the same day can get a different decision in a longer window | **Open, documented**. Changing it would change every stored baseline |
| L2 | Regime features use the asset as its own benchmark (single-asset MVP), so `bench_*` duplicate asset features | Documented in `limitations.md`; multi-asset regime features are future work |
| L3 | Ranking requires 5 round trips, the warning fires below 10 | Intentional (exclusion vs caution); documented |
| L4 | MLP in-sample metric includes its early-stopping tail | Reporting nuance only; documented |
| L5 | Multiple-comparison threshold `sqrt(2 ln N)/sqrt(years)` assumes independent strategies | Conservative for correlated strategies; documented on the Analytics page |
| L6 | SQLite runtime files committed (`aether.db-shm/-wal`) | **Fixed** (untracked, `.gitignore`) |
| L7 | Overview said "No eligible agent… not an error" while the API was unreachable | **Fixed** |
| L8 | Pages scrolled horizontally at phone width | **Fixed** (`min-w-0` on grid cards) |

## Audit coverage

| Area | Verdict | Evidence |
| --- | --- | --- |
| Architecture, module boundaries | Sound | Engine package has no web or DB imports; API owns persistence and jobs; one timing module |
| ML implementations | Sound | Scaler fit on training rows only; MLP early stopping on a purged tail of the training window; XGB/torch saved without pickle |
| Training pipelines | Sound | Labels computed on bars truncated at the validation start; purge gap `horizon + 1 + embargo` |
| Backtesting accuracy | Sound | Analytic tests (fees, slippage, cash conservation, integer sizing, circuit breaker) + new portfolio look-ahead test |
| Data leakage | None found | Feature and agent poisoned-future tests for all 6 families; final test locked by type (`DevelopmentData`) |
| Evolutionary selection | **H1 fixed, H2 mitigated** | See above |
| Tournament evaluation | Sound + improved | Same folds, costs, risk; score components exposed; new warnings |
| Database and migrations | Sound | Normalised schema, FKs, indexes; CI runs `alembic check` |
| API behaviour | Sound + M2/M4 fixed | Typed schemas, structured errors, 202 + job ids, idempotent creation |
| Worker orchestration | Sound, M10 open | Durable job table, interrupted-job recovery test, bounded pool |
| Frontend/backend integration | Sound | Zod-validated client; snapshot mode reuses the same schemas |
| Error handling | Improved (M5, L7) | Error boundary, explicit error states |
| Logging | Adequate | structlog, JSON in Compose; no aggregation configured |
| Security | M2-M4 fixed, M9 open | Every job-creating route requires `X-API-Key`; production refuses to start without it; constant-time key check; only a constant `SELECT 1` raw SQL; uploads size-capped before parsing; pickle loaded only after SHA-256 verification; no secrets in git (pattern scan) |
| Test coverage | Extended | Python 93 -> 130 tests and Vitest 24 -> 33 during Phase 2 (counts in `docs/ENGINEERING_CASE_STUDY.md`) |
| Dependency compatibility | Fixed (M1) | Windows (this machine) and Linux (CI) both green |

## Things that were checked and are correct

* Decision at close *t* fills at open *t+1*; labels are `open[t+1+h]/open[t+1]-1`.
* Training rows end `horizon + 1 + embargo` bars before each validation window.
* The final test is fitted on development data only and simulated once; a
  second run must be acknowledged and marks the experiment contaminated.
* Process-pool evaluation is bit-identical to serial evaluation.
* An identical configuration reproduces identical results through the full
  API path (exact comparison).
* The simulator optimisation in Phase 2 is output-identical (40-case
  reference with exact equality).

## Recommended next steps (not done)

1. **Selection-free holdout in the product** (H2): add
   `evolution_holdout_fraction` to `ExperimentConfig`, so every experiment
   reports its champions on a window evolution never saw.
2. Rate limiting at the gateway, or with proxy headers (M9); worker
   `time_limit` (M10).
3. Multi-asset regime features from a real benchmark series (L2).
4. Make the random baseline prefix-stable (L1) in a versioned change that
   records the break in stored baselines.
