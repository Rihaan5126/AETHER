"""Re-run one completed phase-2 selection run and compare it field by field.

    uv run python scripts/research/rerun_check.py "SPY|8x3|seed1"

Everything except wall-clock time must match the stored result exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import phase2_study as study


def diff(a: object, b: object, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k == "wall_seconds":
                continue
            out += diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [
            d for i, (x, y) in enumerate(zip(a, b, strict=True)) for d in diff(x, y, f"{path}[{i}]")
        ]
    if a == b:
        return []
    if isinstance(a, float) and isinstance(b, float):
        rel = abs(a - b) / max(abs(a), abs(b), 1e-300)
        return [f"{path}: {a!r} != {b!r} (relative difference {rel:.1e})"]
    return [f"{path}: {a!r} != {b!r}"]


def count_leaves(x: object) -> int:
    if isinstance(x, dict):
        return sum(count_leaves(v) for k, v in x.items() if k != "wall_seconds")
    if isinstance(x, list):
        return sum(count_leaves(v) for v in x)
    return 1


def main() -> None:
    key = sys.argv[1]
    stored = json.loads((study.OUT / "phase2_results.json").read_text())["selection"][key]
    symbol, size, seed = key.split("|")
    pop, gens = (int(x) for x in size.split("x"))
    dataset = study.resolve_dataset("ds_aa0bb65d9164")
    fresh = study.run_selection(
        dataset, symbol, pop, gens, int(seed.removeprefix("seed")), workers=4
    )
    fresh = json.loads(json.dumps(fresh, default=float))  # same serialisation as the stored file
    problems = diff(stored, fresh)
    print(
        json.dumps(
            {
                "key": key,
                "identical": not problems,
                "values_compared": count_leaves(stored),
                "differences": problems[:50],
                "stored_wall_seconds": stored["wall_seconds"],
                "rerun_wall_seconds": fresh["wall_seconds"],
            },
            indent=2,
        )
    )
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
