# Deployment verification

This records what was **verified by running something** and what **could not be
verified**. Re-run the black-box checks at any time:

```bash
python scripts/verify_deployment.py --web https://aetherbyrihaan.vercel.app
python scripts/verify_deployment.py --api https://<api-host> --api-key <key> \
    --allowed-origin https://aetherbyrihaan.vercel.app
```

The script uses only the standard library. Its API checks never start training:
the only write requests are unauthenticated or deliberately invalid, so they
are rejected before a job is created.

## Deployed architecture

| Component | Where | Status |
| --- | --- | --- |
| Dashboard (Next.js 16) | Vercel, `https://aetherbyrihaan.vercel.app` | **Live**, verified |
| Data shown on the dashboard | Static snapshot in `apps/web/public/snapshot/`, exported from the real API (`scripts/export_snapshot.py`) | **Live**, read-only |
| API (FastAPI) | Not deployed publicly | Verified locally in production mode |
| Training worker (Celery) | Not deployed | Not verifiable here |
| PostgreSQL, Redis | Not deployed | Not verifiable here |

The public site is deliberately **read-only**. No public endpoint exists that
could start CPU-heavy training, so cost and abuse risk is zero. Training runs
on a developer machine (or a persistent worker host), never inside Vercel
functions. Snapshot mode switches on automatically when `VERCEL=1`
(`apps/web/next.config.ts`).

## 1. Live dashboard (verified 2026-10-11)

`python scripts/verify_deployment.py --web https://aetherbyrihaan.vercel.app`

| Check | Result | Observed |
| --- | --- | --- |
| HTTPS page loads | PASS | 200 |
| HTTP redirects to HTTPS | PASS | 308, `Location: https://...` |
| HSTS | PASS | `max-age=63072000; includeSubDomains; preload` (Vercel) |
| X-Frame-Options | PASS | `DENY` |
| X-Content-Type-Options | PASS | `nosniff` |
| Referrer-Policy | PASS | `strict-origin-when-cross-origin` |
| Content-Security-Policy | **FAIL, fixed** | Missing on the live site. Added in this phase (see below); live after the next deploy |
| No `X-Powered-By` | PASS | `poweredByHeader: false` |
| Snapshot metadata | PASS | 3 experiments |
| Snapshot is read-only | PASS | `write_protected=true`, `environment=snapshot` |
| JSON served compressed | PASS | `content-encoding: br` (1.28 MB curve file transfers about 150 KB) |

Browser checks on the live site (built-in browser, 2026-10-11): every page
(Overview, Agent Arena, agent detail, Tournaments, Agent Comparison,
Evolution Lab, Research Analytics, Datasets, experiment monitor, System)
loaded with **zero failed queries** in the TanStack Query cache and **zero
requests** to any API host. The builder shows the read-only message instead
of submitting.

### Content-Security-Policy (added)

Production builds now send:

```
default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';
img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' [API origin];
object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'
```

`connect-src` contains only `'self'` in snapshot mode, and the configured
`NEXT_PUBLIC_API_URL` origin otherwise. `'unsafe-inline'` is needed for
Next.js's inline bootstrap scripts and Recharts' inline styles; there are no
third-party origins. A local production build in snapshot mode loaded every
page with the header present and **no CSP violations** in the console. It is
not applied in `next dev`, which needs `eval`.

## 2. API in production mode (verified locally, 2026-10-11)

Started with a temporary SQLite database, a generated key,
`AETHER_ENVIRONMENT=production` and
`AETHER_CORS_ORIGINS=["https://aetherbyrihaan.vercel.app"]`.

| Check | Result | Observed |
| --- | --- | --- |
| Refuses to start without `AETHER_API_KEY` in production | PASS | `RuntimeError: AETHER_API_KEY must be set in production` |
| Health check | PASS | `/api/v1/health` 200 |
| System status | PASS | `environment=production`, `database.ok=true` |
| Write protection on | PASS | `write_protected=true` |
| Job queue bounded | PASS | `max_pending_jobs=5` (new in Phase 2) |
| CORS allows dashboard origin | PASS | preflight 200, `allow-origin` echoes it |
| CORS rejects other origins | PASS | preflight 400, no `allow-origin` |
| Write without key refused | PASS | 401 `unauthorized` |
| Wrong key refused | PASS | 401 (constant-time comparison, `hmac.compare_digest`) |
| Authenticated invalid write | PASS | 422, no job created |
| Storage paths hidden in production | PASS (unit test) | `/system/status` storage shows `configured` (fixed in Phase 2) |

## 3. Static review (not executable here)

Docker is not installed on the verification machine, so the Compose stack was
**not run** in Phase 2. Reviewed by reading `docker-compose.yml`:

* Postgres and Redis have health checks; the API waits for both (`service_healthy`).
* The worker runs Celery with `--concurrency=1` and container limits
  (`WORKER_CPUS`, default 2; `WORKER_MEMORY`, default 4g). Training never runs
  in the API process when `AETHER_JOB_BACKEND=celery`.
* The API image runs `alembic upgrade head` on start; CI checks that
  migrations match the models (`alembic check`).
* `AETHER_API_KEY` defaults to empty in Compose. That is fine for local use,
  but `AETHER_ENVIRONMENT=production` refuses to start without it.

CI (`.github/workflows/ci.yml`) runs on every push: ruff, mypy, migrations,
pytest, ESLint, tsc, Vitest, `next build`, and a Playwright end-to-end suite
against a live API and dashboard. Every pushed commit before Phase 2 passed
(checked through the GitHub API on 2026-10-11).

## 4. Not verified, and what a live deployment would still need

| Item | Why it could not be verified | What to do |
| --- | --- | --- |
| Managed Postgres/Redis connectivity, TLS to the DB | No hosted services or credentials | Point `AETHER_DATABASE_URL` / `AETHER_REDIS_URL` at managed instances; run `verify_deployment.py --api` |
| Worker connectivity and job persistence across restarts | No hosted worker | Covered locally by tests (`test_interrupted_jobs_are_marked_failed`); re-check on the host |
| Rate limiting behind a proxy | The limiter keys on `request.client.host`; behind a load balancer every client shares one address | Run uvicorn with `--proxy-headers --forwarded-allow-ips=<proxy>`, or rate-limit at the proxy or gateway |
| Multi-replica rate limiting | The limiter is in-process | Use a shared limiter (Redis) if the API runs more than one replica |
| Hard job timeouts | `max_job_seconds` is checked cooperatively between units of work; a single long model fit cannot be interrupted mid-fit | Celery `time_limit` on the worker; per-fit `max_fit_seconds` already bounds MLP epochs |
| Log aggregation | `AETHER_LOG_JSON=true` emits structured JSON; no aggregator configured | Ship stdout to the host's log service |

## 5. Environment variables

| Variable | Where | Notes |
| --- | --- | --- |
| `NEXT_PUBLIC_AETHER_SNAPSHOT` | Vercel / web build | Defaults to `true` on Vercel, `false` elsewhere |
| `NEXT_PUBLIC_API_URL` | web build | Baked in at build time; also added to the CSP `connect-src` |
| `NEXT_PUBLIC_REPO_URL` | web build | Defaults to the GitHub repository Vercel builds from |
| `AETHER_ENVIRONMENT` | API | `production` requires `AETHER_API_KEY` and hides storage paths |
| `AETHER_API_KEY` | API (secret) | Required header `X-API-Key` for every write |
| `AETHER_CORS_ORIGINS` | API | JSON list; set it to the dashboard origin only |
| `AETHER_MAX_PENDING_JOBS` | API | Total queued + running jobs (default 5) |
| `AETHER_MAX_*` | API | Population, generations, Optuna trials, upload size, job duration |
| `AETHER_ENABLE_YAHOO` | API | Off by default; allows outbound data downloads |

No secrets are committed: `.env` is git-ignored, and `.env.example` contains
placeholders only.
