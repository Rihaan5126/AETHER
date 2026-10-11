"""Turn docs/research/phase2_results.json into markdown tables and paired statistics.

    uv run python scripts/research/report.py   # writes docs/research/PHASE2_TABLES.md

Every number in the output is computed here from the stored results; nothing is
typed by hand. Paired comparisons across runs use a two-sided exact sign test
(ties dropped), which assumes only that runs are independent. Runs on the
same symbol share data, so treat p-values as descriptive, not confirmatory.
"""

# Markdown table templates are kept on one line each for readability.
# ruff: noqa: E501, RUF005
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[2] / "docs" / "research"


def f(v: Any, d: int = 2, pct: bool = False, signed: bool = False) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    x = v * 100 if pct else v
    s = f"{x:+.{d}f}" if signed else f"{x:.{d}f}"
    return s + ("%" if pct else "")


def sign_test(diffs: list[float]) -> dict[str, Any]:
    wins = sum(1 for d in diffs if d > 1e-12)
    losses = sum(1 for d in diffs if d < -1e-12)
    ties = len(diffs) - wins - losses
    n = wins + losses
    if n == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
    return {
        "n": len(diffs),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "mean_diff": statistics.fmean(diffs) if diffs else None,
        "median_diff": statistics.median(diffs) if diffs else None,
        "p_two_sided": p,
    }


def champ(run: dict[str, Any], role: str, window: str = "holdout") -> dict[str, Any] | None:
    aid = run["champion_holdout_ids"].get(role)
    return run[window]["agents"].get(aid) if aid else None


def selection_delta(run: dict[str, Any]) -> float | None:
    g = run["generations_summary"]
    first = (g[0]["best_eligible"] or {}).get("score")
    last = (g[-1]["best_eligible"] or {}).get("score")
    return None if first is None or last is None else last - first


def run_table(runs: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "| Run | Unique evals | Best val score gen0 -> last | Champions (initial / evolved / search) | Holdout Sharpe: initial / evolved / search / B&H | Evolved: excess vs B&H [95% CI] | Evolved: null pct |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for key, r in runs.items():
        g = r["generations_summary"]
        s0 = (g[0]["best_eligible"] or {}).get("score")
        s1 = (g[-1]["best_eligible"] or {}).get("score")
        fams = " / ".join(
            (r["champion_specs"][k] or {}).get("family", "none")
            for k in ("initial", "evolved", "search")
        )
        same = len({v for v in r["champion_holdout_ids"].values() if v}) == 1
        sh = [champ(r, k) for k in ("initial", "evolved", "search")]
        ev = sh[1] or {}
        ci = ((ev.get("bootstrap") or {}).get("annual_excess_return")) or {}
        lines.append(
            f"| {key} | {r['unique_evaluations']} | {f(s0, 3)} -> {f(s1, 3)} | {fams}{' (all same agent)' if same else ''} | "
            + " / ".join(f((x or {}).get("sharpe_ratio")) for x in sh)
            + f" / {f(r['holdout']['buy_and_hold']['sharpe_ratio'])} | {f(ci.get('estimate'), 1, True, True)} [{f(ci.get('low'), 1, True, True)}, {f(ci.get('high'), 1, True, True)}] | {f(ev.get('null_percentile'), 2)} |"
        )
    return lines


def comparisons(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def diffs(a: str, b: str) -> list[float]:
        out = []
        for r in runs.values():
            x, y = champ(r, a), champ(r, b)
            if x and y and x["sharpe_ratio"] is not None and y["sharpe_ratio"] is not None:
                out.append(x["sharpe_ratio"] - y["sharpe_ratio"])
        return out

    pop = []
    for r in runs.values():
        a = r["holdout_population"]["last_generation"]["mean_sharpe"]
        b = r["holdout_population"]["generation_0"]["mean_sharpe"]
        if a is not None and b is not None:
            pop.append(a - b)
    sel = [d for d in (selection_delta(r) for r in runs.values()) if d is not None]
    vs_bh = [
        champ(r, "evolved")["sharpe_ratio"] - r["holdout"]["buy_and_hold"]["sharpe_ratio"]
        for r in runs.values()
        if champ(r, "evolved") and champ(r, "evolved")["sharpe_ratio"] is not None
    ]
    excess_sig = [
        (champ(r, "evolved")["bootstrap"]["annual_excess_return"] or {}).get("low")
        for r in runs.values()
        if champ(r, "evolved")
    ]
    nulls = [champ(r, "evolved")["null_percentile"] for r in runs.values() if champ(r, "evolved")]
    unchanged = sum(
        1
        for r in runs.values()
        if r["champion_holdout_ids"]["initial"] == r["champion_holdout_ids"]["evolved"]
    )
    return {
        "runs": len(runs),
        "selection_score_gain": sign_test(sel),
        "evolved_champion_unchanged_from_gen0": unchanged,
        "holdout_sharpe_evolved_minus_initial": sign_test(diffs("evolved", "initial")),
        "holdout_sharpe_evolved_minus_search": sign_test(diffs("evolved", "search")),
        "holdout_population_mean_sharpe_last_minus_gen0": sign_test(pop),
        "holdout_sharpe_evolved_minus_buy_and_hold": sign_test(vs_bh),
        "evolved_excess_ci_lower_bound_above_zero": sum(
            1 for x in excess_sig if x is not None and x > 0
        ),
        "evolved_null_percentile_median": statistics.median(nulls) if nulls else None,
        "evolved_null_percentile_at_least_95": sum(1 for x in nulls if x is not None and x >= 0.95),
    }


def cost_table(runs: dict[str, dict[str, Any]]) -> list[str]:
    by_mult: dict[float, list[float]] = {}
    for r in runs.values():
        c = champ(r, "evolved")
        for row in (c or {}).get("cost_sensitivity", []):
            if row["sharpe_ratio"] is not None:
                by_mult.setdefault(row["cost_multiplier"], []).append(row["sharpe_ratio"])
    lines = [
        "| Cost multiplier (base: 1 bp fee + 5 bp slippage) | Runs | Mean holdout Sharpe of evolved champion |",
        "| ---: | ---: | ---: |",
    ]
    for m in sorted(by_mult):
        v = by_mult[m]
        lines.append(f"| {m:g}x | {len(v)} | {f(statistics.fmean(v))} |")
    return lines


def comparison_lines(c: dict[str, Any]) -> list[str]:
    def row(name: str, s: dict[str, Any], pct: bool = False) -> str:
        return (
            f"| {name} | {s['n']} | {s['wins']} / {s['losses']} / {s['ties']} | "
            f"{f(s['mean_diff'], 3)} | {f(s['median_diff'], 3)} | {f(s['p_two_sided'], 3)} |"
        )

    return [
        "| Paired comparison (per run) | Runs | Wins / losses / ties | Mean diff | Median diff | Sign test p |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
        row(
            "Best validation score, last gen minus gen 0 (selection data)",
            c["selection_score_gain"],
        ),
        row(
            "Holdout Sharpe, evolved champion minus initial champion",
            c["holdout_sharpe_evolved_minus_initial"],
        ),
        row(
            "Holdout Sharpe, evolved champion minus random-search champion",
            c["holdout_sharpe_evolved_minus_search"],
        ),
        row(
            "Holdout mean Sharpe of population, last gen minus gen 0",
            c["holdout_population_mean_sharpe_last_minus_gen0"],
        ),
        row(
            "Holdout Sharpe, evolved champion minus buy-and-hold",
            c["holdout_sharpe_evolved_minus_buy_and_hold"],
        ),
    ]


def final_lines(final: dict[str, Any]) -> list[str]:
    lines = [
        "| Run | Test period | Agent (roles) | Test Sharpe [95% CI] | B&H Sharpe | Excess vs B&H [95% CI] | Null pct | Round trips |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for key, r in final["runs"].items():
        for aid, a in r["agents"].items():
            b = a["bootstrap"]
            s, e = b["sharpe"], b.get("annual_excess_return") or {}
            lines.append(
                f"| {key} | {r['start']} to {r['end']} | {a['family']} ({', '.join(r['roles'][aid])}) | "
                f"{f(s['estimate'])} [{f(s['low'])}, {f(s['high'])}] | {f(r['buy_and_hold']['sharpe_ratio'])} | "
                f"{f(e.get('estimate'), 1, True, True)} [{f(e.get('low'), 1, True, True)}, {f(e.get('high'), 1, True, True)}] | "
                f"{f(a['null_percentile'])} | {a['n_round_trips']} |"
            )
    return lines


def final_summary(final: dict[str, Any]) -> dict[str, Any]:
    rows = [(k, a) for k, r in final["runs"].items() for a in r["agents"].values()]
    evolved = [
        (k, a, final["runs"][k])
        for k, r in final["runs"].items()
        for aid, a in r["agents"].items()
        if "evolved" in r["roles"][aid]
    ]
    return {
        "agents_tested": len(rows),
        "evolved_champions": len(evolved),
        "evolved_beat_bh_sharpe": sum(
            1
            for _, a, r in evolved
            if (a["sharpe_ratio"] or -9) > r["buy_and_hold"]["sharpe_ratio"]
        ),
        "evolved_positive_excess": sum(1 for _, a, _ in evolved if (a["excess_return"] or -9) > 0),
        "evolved_excess_ci_above_zero": sum(
            1
            for _, a, _ in evolved
            if ((a["bootstrap"].get("annual_excess_return") or {}).get("low") or -9) > 0
        ),
        "evolved_sharpe_ci_above_zero": sum(
            1 for _, a, _ in evolved if (a["bootstrap"]["sharpe"].get("low") or -9) > 0
        ),
        "evolved_null_pct_at_least_95": sum(
            1 for _, a, _ in evolved if (a["null_percentile"] or 0) >= 0.95
        ),
    }


def main() -> None:
    res = json.loads((OUT / "phase2_results.json").read_text())
    sel = res.get("selection", {})
    main_runs = {k: v for k, v in sel.items() if "|8x3|" in k}
    scaling = {k: v for k, v in sel.items() if k.startswith("SPY|")}
    md = [
        "# Phase 2 research tables (generated)",
        "",
        "Generated by `scripts/research/report.py` from `phase2_results.json`. Do not edit by hand.",
        "",
    ]
    first = next(iter(sel.values()), None)
    if first:
        md += [
            f"Periods (SPY; similar for all symbols): selection {first['selection_period'][0]} to {first['selection_period'][1]}, "
            f"inner holdout {first['holdout_period'][0]} to {first['holdout_period'][1]}, "
            f"locked final test {first['outer_test_period'][0]} to {first['outer_test_period'][1]}.",
            "",
        ]
    md += (
        ["## Main grid: 8 agents x 3 generations, 5 symbols x 3 seeds", ""]
        + run_table(main_runs)
        + [""]
    )
    comp = comparisons(main_runs)
    md += comparison_lines(comp) + [""]
    md += (
        [
            f"* Evolved champion identical to the generation-0 champion in **{comp['evolved_champion_unchanged_from_gen0']} of {comp['runs']}** runs.",
            f"* Evolved champion's holdout excess-return 95% CI entirely above zero in **{comp['evolved_excess_ci_lower_bound_above_zero']} of {comp['runs']}** runs.",
            f"* Evolved champion at or above the 95th percentile of 300 random strategies in **{comp['evolved_null_percentile_at_least_95']} of {comp['runs']}** runs (median percentile {f(comp['evolved_null_percentile_median'])}).",
            "",
            "### Transaction-cost sensitivity (inner holdout)",
            "",
        ]
        + cost_table(main_runs)
        + [""]
    )
    md += ["## Scaling on SPY (population x generations)", ""] + run_table(scaling) + [""]
    if "final" in res:
        fs = final_summary(res["final"])
        md += (
            [
                "## Locked final test (run once)",
                "",
                f"Run at {res['final']['run_at']} on code `{res['final']['code_version']}`. Champions were refitted on the full development period and simulated once.",
                "",
            ]
            + final_lines(res["final"])
            + [
                "",
                f"* Evolved champions with a higher test Sharpe than buy-and-hold: **{fs['evolved_beat_bh_sharpe']} of {fs['evolved_champions']}**.",
                f"* Positive test excess return: **{fs['evolved_positive_excess']} of {fs['evolved_champions']}**; excess CI entirely above zero: **{fs['evolved_excess_ci_above_zero']}**.",
                f"* Test Sharpe CI entirely above zero: **{fs['evolved_sharpe_ci_above_zero']}**; at or above the 95th null percentile: **{fs['evolved_null_pct_at_least_95']}**.",
                "",
            ]
        )
        comp["final"] = fs
    (OUT / "PHASE2_TABLES.md").write_text("\n".join(md), encoding="utf-8")
    (OUT / "phase2_summary.json").write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
