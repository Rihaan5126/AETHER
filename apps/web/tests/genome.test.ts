import { describe, expect, it } from "vitest";

import { flattenGenome, genomeDiff, genomeSummary, inactiveGenes } from "@/lib/genome";

describe("genome", () => {
  const parent = { family: "xgboost", max_depth: 3, learning_rate: 0.05, policy: { threshold: 0.55, sizing: "binary" } };
  const child = { ...parent, max_depth: 4, policy: { threshold: 0.6, sizing: "binary" } };

  it("flattens nested genes", () => {
    expect(flattenGenome(parent)).toMatchObject({ "policy.threshold": "0.550", max_depth: "3" });
  });

  it("diffs only changed genes", () => {
    expect(genomeDiff(parent, child)).toEqual([
      { key: "max_depth", before: "3", after: "4" },
      { key: "policy.threshold", before: "0.550", after: "0.600" },
    ]);
    expect(genomeDiff(null, { family: "random" })).toEqual([{ key: "family", before: null, after: "random" }]);
  });

  it("summarises key genes per family", () => {
    const rows = Object.fromEntries(genomeSummary(child));
    expect(rows.max_depth).toBe("4");
    expect(rows.threshold).toBe("0.600");
  });
});

describe("inactiveGenes", () => {
  it("mirrors the engine's behaviour key for momentum rules", () => {
    expect([...inactiveGenes({ family: "momentum", rule: "ma_cross" })]).toEqual(["trend_lookback"]);
    expect([...inactiveGenes({ family: "momentum", rule: "time_series_momentum" })].sort()).toEqual(["fast_window", "slow_window"]);
    expect(inactiveGenes({ family: "momentum", rule: "both" }).size).toBe(0);
    expect(inactiveGenes({ family: "xgboost" }).size).toBe(0);
  });
});
