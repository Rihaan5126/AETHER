import { describe, expect, it } from "vitest";

import type { CurvePoint } from "@/lib/api/schemas";
import { histogram, mean, mergeSeries, periodReturns, rollingReturn, rollingVolatility } from "@/lib/series";

const pt = (t: string, equity: number, fold = 0): CurvePoint => ({ t, equity, drawdown: 0, weight: 1, fold });

describe("series", () => {
  it("merges curves by date and normalises to growth of 1", () => {
    const rows = mergeSeries(
      [
        { key: "a", points: [pt("2020-01-02", 100), pt("2020-01-03", 110)] },
        { key: "b", points: [pt("2020-01-03", 50)] },
      ],
      "equity",
      true,
    );
    expect(rows).toEqual([
      { t: "2020-01-02", a: 1 },
      { t: "2020-01-03", a: 1.1, b: 1 },
    ]);
  });

  it("does not compute returns across fold boundaries", () => {
    const r = periodReturns([pt("d1", 100, 0), pt("d2", 110, 0), pt("d3", 50, 1), pt("d4", 55, 1)]);
    expect(r).toHaveLength(2);
    expect(r[0]).toBeCloseTo(0.1);
    expect(r[1]).toBeCloseTo(0.1);
  });

  it("builds histograms that count every value", () => {
    const values = [0, 0.01, 0.02, -0.01, 0.05];
    const h = histogram(values, 5);
    expect(h).toHaveLength(5);
    expect(h.reduce((a, b) => a + b.count, 0)).toBe(values.length);
    expect(histogram([])).toEqual([]);
    expect(histogram([0.1, 0.1], 3).reduce((a, b) => a + b.count, 0)).toBe(2);
  });

  it("returns null volatility until the window is mostly filled", () => {
    const points = Array.from({ length: 30 }, (_, i) => pt(`d${i}`, 100 * (1 + (i % 2 ? 0.01 : -0.01))));
    const vol = rollingVolatility(points, 20);
    expect(vol[5]!.vol).toBeNull();
    expect(vol[29]!.vol).toBeGreaterThan(0);
  });

  it("averages", () => {
    expect(mean([])).toBeNull();
    expect(mean([1, 2, 3])).toBe(2);
  });
});

describe("rollingReturn", () => {
  it("is the trailing return over the window and null before it", () => {
    const pts = [100, 110, 121, 133.1].map((equity, i) => pt(`2020-01-0${i + 1}`, equity));
    const out = rollingReturn(pts, 2);
    expect(out.map((p) => p.ret)).toEqual([null, null, expect.closeTo(0.21, 10), expect.closeTo(0.21, 10)]);
  });
});
