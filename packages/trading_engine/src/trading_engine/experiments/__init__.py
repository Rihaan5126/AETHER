"""Experiment configuration, manifests and orchestration."""

from trading_engine.experiments.config import ExperimentConfig, estimate_workload
from trading_engine.experiments.final_test import FinalTestResult, run_final_test
from trading_engine.experiments.manifest import build_manifest, code_version, dependency_versions
from trading_engine.experiments.runner import fit_on_development, prepare_market_data

__all__ = [
    "ExperimentConfig",
    "FinalTestResult",
    "build_manifest",
    "code_version",
    "dependency_versions",
    "estimate_workload",
    "fit_on_development",
    "prepare_market_data",
    "run_final_test",
]
