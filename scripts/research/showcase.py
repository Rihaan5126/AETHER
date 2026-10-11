"""Run one dashboard showcase experiment per symbol through the real API service layer,
then its locked final test once.

    uv run python scripts/research/showcase.py --seed 11

Results land in the AETHER database (and therefore the dashboard and snapshot).
"""

from __future__ import annotations

import argparse
import json
import logging

from aether_api.cli import final_test, run_experiment
from aether_api.schemas.experiments import ExperimentCreateRequest

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ds_aa0bb65d9164")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    logging.disable(logging.INFO)
    summary = []
    for symbol in SYMBOLS:
        req = ExperimentCreateRequest(
            name=f"Phase 2 - {symbol} (seed {args.seed})",
            dataset_id=args.dataset,
            symbol=symbol,
            seed=args.seed,
        )
        exp = run_experiment(req, f"showcase-{args.dataset}-{symbol}-{args.seed}")
        ft = final_test(str(exp["experiment_id"])) if exp["status"] == "completed" else None
        summary.append({"symbol": symbol, **exp, "final_test": ft})
        print(json.dumps(summary[-1], default=str), flush=True)


if __name__ == "__main__":
    main()
