# AETHER developer commands. Run `make help` for the list.
SHELL := /bin/bash
UV ?= uv
WEB := apps/web
API_PORT ?= 8000

.PHONY: help install init-db demo api worker web dev test test-py test-web e2e lint typecheck format check up down logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install Python (uv) and web (npm) dependencies
	$(UV) sync
	cd $(WEB) && npm ci

init-db: ## Create/upgrade the database schema (SQLite by default)
	$(UV) run alembic -c services/api/alembic.ini upgrade head

demo: init-db ## Import synthetic data and run a small reproducible evolution experiment
	$(UV) run aether demo --generations 3 --population 8 --seed 42

api: ## Run the API with the in-process job runner (http://localhost:8000)
	$(UV) run uvicorn aether_api.main:app --reload --port $(API_PORT)

worker: ## Run a Celery training worker (requires AETHER_JOB_BACKEND=celery and Redis)
	$(UV) run celery -A aether_api.worker worker -Q training --concurrency=1 --loglevel=INFO

web: ## Run the dashboard dev server (http://localhost:3000)
	cd $(WEB) && npm run dev

dev: init-db ## Run API and dashboard together (Ctrl-C stops both)
	@trap 'kill 0' EXIT; \
	$(UV) run uvicorn aether_api.main:app --port $(API_PORT) & \
	(cd $(WEB) && npm run dev) & \
	wait

test: test-py test-web ## Run all unit and integration tests

test-py: ## Python tests (engine + API integration)
	$(UV) run pytest

test-web: ## Dashboard unit tests (Vitest)
	cd $(WEB) && npm test

e2e: ## Playwright smoke test (needs `make api` with demo data and `make web` running)
	cd $(WEB) && npx playwright test

lint: ## Ruff + ESLint
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	cd $(WEB) && npm run lint

typecheck: ## mypy + tsc
	$(UV) run mypy
	cd $(WEB) && npm run typecheck

format: ## Auto-format Python
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

check: lint typecheck test ## Everything CI runs (except e2e)

up: ## Start the full Docker Compose stack
	docker compose up --build -d

down: ## Stop the Docker Compose stack
	docker compose down

logs: ## Tail Docker Compose logs
	docker compose logs -f api worker

clean: ## Remove local runtime data (SQLite DB, datasets, artifacts)
	rm -rf aether.db datasets/store artifacts/experiments
