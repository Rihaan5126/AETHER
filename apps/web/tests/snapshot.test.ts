import { describe, expect, it } from "vitest";

import { resolveSnapshot, snapshotUrl } from "@/lib/api/snapshot";

const agents = {
  items: [
    { id: "a1", experiment_id: "e1", generation: 0, algorithm: "mlp", is_benchmark: false },
    { id: "a2", experiment_id: "e1", generation: 1, algorithm: "xgboost", is_benchmark: false },
    { id: "a3", experiment_id: "e2", generation: 1, algorithm: "buy_and_hold", is_benchmark: true },
    { id: "a4", experiment_id: "e1", generation: 1, algorithm: "mlp", is_benchmark: false },
  ],
  total: 4,
  limit: 4,
  offset: 0,
};

describe("resolveSnapshot", () => {
  it("maps detail routes straight to files", () => {
    expect(resolveSnapshot("/experiments/exp_1")).toEqual({ file: "experiments/exp_1" });
    expect(resolveSnapshot("/tournaments/t1/leaderboard").file).toBe("tournaments/t1/leaderboard");
    expect(resolveSnapshot("/system/status").file).toBe("system/status");
  });

  it("filters and paginates list endpoints like the API", () => {
    const r = resolveSnapshot("/agents", { experiment_id: "e1", generation: 1, limit: 1 });
    expect(r.file).toBe("agents");
    expect(r.transform?.(agents)).toEqual({ items: [agents.items[1]], total: 2, limit: 1, offset: 0 });

    const page2 = resolveSnapshot("/agents", { experiment_id: "e1", limit: 2, offset: 2 }).transform?.(agents);
    expect(page2).toMatchObject({ items: [agents.items[3]], total: 3, offset: 2 });

    const noBench = resolveSnapshot("/agents", { include_benchmarks: false }).transform?.(agents) as { total: number };
    expect(noBench.total).toBe(3);
  });

  it("uses the API's default page size when no limit is given", () => {
    const many = { items: Array.from({ length: 70 }, (_, i) => ({ id: i })), total: 70, limit: 70, offset: 0 };
    const out = resolveSnapshot("/experiments").transform?.(many) as { items: unknown[]; total: number; limit: number };
    expect(out.items).toHaveLength(50);
    expect(out.total).toBe(70);
    expect(out.limit).toBe(50);
  });

  it("selects equity curves by resolution and filters agents but keeps the benchmark", () => {
    const r = resolveSnapshot("/tournaments/t1/equity-curves", { agent_ids: ["a1"], max_points: 1500 });
    expect(r.file).toBe("tournaments/t1/equity-curves.mp1500");
    const data = { period: "validation", downsampled: true, series: [{ agent_id: "a1" }, { agent_id: "a2" }], benchmark: { agent_id: "__benchmark__" } };
    expect(r.transform?.(data)).toEqual({ ...data, series: [{ agent_id: "a1" }] });
    expect(resolveSnapshot("/tournaments/t1/equity-curves").transform).toBeUndefined();
  });

  it("keys parameterised files by their parameters", () => {
    expect(resolveSnapshot("/datasets/d1/prices", { symbol: "VOO", max_points: 600 }).file).toBe("datasets/d1/prices.VOO.mp600");
    expect(resolveSnapshot("/experiments/e1/final-test/1/equity-curves", { max_points: 800 }).file).toBe(
      "experiments/e1/final-test/1/equity-curves.mp800",
    );
  });

  it("filters trades by agent", () => {
    const trades = { items: [{ agent_id: "a1" }, { agent_id: "a2" }, { agent_id: "a1" }], total: 3, limit: 3, offset: 0 };
    const out = resolveSnapshot("/tournaments/t1/trades", { agent_id: "a1", limit: 25 }).transform?.(trades) as { total: number };
    expect(out.total).toBe(2);
  });

  it("builds URLs under /snapshot", () => {
    expect(snapshotUrl("tournaments/t1/equity-curves.mp600")).toBe("/snapshot/tournaments/t1/equity-curves.mp600.json");
  });
});
