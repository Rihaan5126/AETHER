# Reproducibility

AETHER aims for **bit-identical** results when the same configuration runs on
the same data, code and dependency versions on the same platform. This file
lists what is recorded, how determinism is enforced, what was measured, and
where exact reproduction is not promised.

## What every experiment records

| Item | Where it is stored | Export field (`GET /experiments/{id}/manifest`) |
| --- | --- | --- |
| Dataset version and hash | `datasets.hash`, the SHA-256 of canonical bytes, re-verified on every load | `dataset.hash`, `manifest.dataset_hash` |
| Dataset provenance | source, symbols, requested range, adjustment method, retrieval time, validation issues | `dataset.*` |
| Agent configurations | typed genome per agent (`agents.genome`), lineage before/after | `agents[].genome`, `lineage[]` |
| Feature configuration | 25 named features + cache version | `features` |
| Training parameters | genome + walk-forward config (folds, embargo, horizon) | `config`, `execution_assumptions.walk_forward` |
| Random seeds | experiment seed; per-agent training seed | `experiment.seed`, `agents[].seed`, `lineage[].training_seed` |
| Code revision | git SHA, `-dirty` if uncommitted changes | `manifest.code_version` |
| Dependency versions | Python, numpy, pandas, pyarrow, scikit-learn, xgboost, torch, pydantic, optuna | `manifest.dependencies` |
| Platform | `platform.platform()` | `manifest.platform` |
| Execution assumptions | fill timing (close *t* -> open *t+1*), fees, slippage, risk limits, ranking policy, warning thresholds | `execution_assumptions` |
| Training metrics | per fold: classification metrics, early stopping, feature importance, in-sample backtest | `training_runs[]`, `tournaments[].results[].fold_metrics` |
| Evaluation metrics | every metric, score component and warning | `tournaments[].results[]` |
| Model artifacts | path, SHA-256 and per-file hashes; verified before every load | `agents[].artifact` |

The manifest (`experiments.manifest`) is written once when the experiment
starts and never modified. The export is a single JSON document; the dashboard
offers it as **Download JSON** on every experiment page (live site included).

## How determinism is enforced

| Source of randomness | Control |
| --- | --- |
| Population, mutation, selection | one `numpy.random.Generator` seeded from the experiment seed; agent seeds are drawn from it |
| Random Forest | `random_state=seed`, `n_jobs=1` |
| XGBoost | `random_state=seed`, `n_jobs=1`, `tree_method="hist"` |
| PyTorch MLP | `torch.manual_seed(seed)` before initialisation, a seeded `Generator` for shuffling, `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)` |
| Random baseline | stream keyed on (seed, first decision date) |
| Synthetic data | seeded generator; identical inputs give byte-identical data and the same hash |
| Parallel evaluation | each agent is evaluated independently in a spawned process; results are collected in submission order |
| Bootstrap / random null | seeded generators (`seed` argument) |

## Measured

| Test | What is compared | Result |
| --- | --- | --- |
| `services/api/tests/test_integration.py::test_identical_config_reproduces_identical_results` | Two experiments from one configuration through the full API and job path: every genome, rank, status, score and metric of every tournament | **Exact match** |
| `packages/trading_engine/tests/test_evolution.py::test_parallel_tournament_matches_serial` | `max_workers=2` vs serial for momentum, XGBoost and MLP: score, metrics, fold metrics | **Exact match** |
| `test_selection_is_reproducible` | Lineage (ids, operations, parents, genomes) of two evolution runs | Exact match |
| `test_training_is_deterministic` (per learned family) | Predictions of two fits | Exact match |
| `test_simulation_is_deterministic`, `test_features_are_deterministic`, `test_synthetic_is_deterministic_and_seed_sensitive`, `test_optuna_study_is_bounded_and_reproducible`, `test_random_null_is_deterministic_and_percentile_monotone` | Simulator, features, data, Optuna, null | Exact match |
| **Real-data rerun** (`scripts/research/rerun_check.py "SPY|8x3|seed1"`) | A complete research run on real SPY data (stored run with 6 worker processes, rerun with 4; random search, holdout fits of every agent, bootstrap CIs, 300-strategy null, cost sensitivity), re-run from scratch: **797 values** | **797 / 797 bit-identical** (final code). An earlier attempt had 796 / 797; the 1-ulp difference is root-caused below and fixed |

The rerun record is `docs/research/rerun_SPY_8x3_seed1.json`. An earlier rerun (in
`docs/research/archive/attempt1_negative_cash_bug/`) also crossed a code change: the stored run
used the original simulator and the rerun used the optimised one. Every Sharpe, return, interval
bound, null percentile and cost row still matched exactly, which independently confirms the
simulator optimisation is output-identical.

### The 1-ulp difference in the earlier attempt, and the fix

`holdout_population.last_generation.mean_sharpe` was averaged over a Python
`set` of agent ids. Set iteration order depends on per-process string-hash
randomisation (`PYTHONHASHSEED`), and floating-point addition is not
associative, so the sum came out one unit in the last place different. The
ids are now sorted before averaging. Lesson: **never aggregate floats in set
or dict-of-set order when bit-identity matters.** The rest of the engine
iterates lists and ordered dicts.

## Tolerances and what is not promised

| Situation | Expectation |
| --- | --- |
| Same machine, same code, same dependency versions | Bit-identical (measured above) |
| Same code on another OS or CPU | Not measured across machines. Expect last-bit floating-point differences (BLAS kernels, SIMD paths, libm). Compare metrics with `rel_tol=1e-9`; rankings can only change where two scores tie to within that tolerance |
| Different library versions (e.g. xgboost or torch upgrade) | Not guaranteed. Tree-building and kernels change between releases. Re-run and record a new manifest |
| Re-downloaded Yahoo data | New bars or revised history give a **new dataset hash**; old experiments still point at the stored bytes they used |
| `wall_seconds`, `resource_usage`, timestamps | Never reproducible; excluded from comparisons |

CI (`.github/workflows/ci.yml`) runs the full deterministic test suite on
Linux, and this phase ran it on Windows 11. Both pass, so the determinism
tests hold on both platforms. They compare runs *within* a platform, not
results across platforms.

## Reproducing an experiment

1. Download its export: `GET /api/v1/experiments/{id}/manifest`.
2. Check out `manifest.code_version` and install the locked dependencies
   (`uv sync`; the versions match `manifest.dependencies`).
3. Make sure the dataset with `dataset.hash` is in the store. Re-import from
   the same source and check the hash. If Yahoo revised history, the hash
   differs and you are running a different experiment.
4. `POST /api/v1/experiments` with `manifest.config`, mapped back to the
   request fields, or run the engine directly with
   `ExperimentConfig.model_validate(manifest["config"])`.
