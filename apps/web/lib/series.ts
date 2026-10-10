/** Pure time-series helpers used by charts (unit-tested in tests/series.test.ts). */
import type { CurvePoint } from "@/lib/api/schemas";

export type Row = { t: string } & Record<string, number | string | null>;

/** Merge several curves into one row per date: { t, [key]: value }. Missing values stay null. */
export function mergeSeries(
  series: { key: string; points: CurvePoint[] }[],
  field: "equity" | "drawdown" | "weight" = "equity",
  normalise = false,
): Row[] {
  const rows = new Map<string, Row>();
  for (const { key, points } of series) {
    const base = normalise && points.length ? points[0]![field] : 1;
    for (const p of points) {
      const row = rows.get(p.t) ?? ({ t: p.t } as Row);
      const value = normalise && base ? p[field] / base : p[field];
      row[key] = Number.isFinite(value) ? value : null;
      rows.set(p.t, row);
    }
  }
  return [...rows.values()].sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
}

/** Simple returns between consecutive points of the same fold. */
export function periodReturns(points: CurvePoint[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1]!;
    const cur = points[i]!;
    if (prev.fold !== cur.fold || prev.equity <= 0) continue;
    out.push(cur.equity / prev.equity - 1);
  }
  return out;
}

export function histogram(values: number[], bins = 30): { x: number; label: string; count: number }[] {
  if (!values.length) return [];
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (lo === hi) {
    lo -= 1e-4;
    hi += 1e-4;
  }
  const width = (hi - lo) / bins;
  const counts = new Array<number>(bins).fill(0);
  for (const v of values) {
    const i = Math.min(bins - 1, Math.floor((v - lo) / width));
    counts[i]! += 1;
  }
  return counts.map((count, i) => {
    const x = lo + (i + 0.5) * width;
    return { x, label: `${(x * 100).toFixed(2)}%`, count };
  });
}

/** Annualised rolling volatility of period returns (window in periods). */
export function rollingVolatility(points: CurvePoint[], window = 20, periodsPerYear = 252): { t: string; vol: number | null }[] {
  const rets: (number | null)[] = points.map((p, i) => {
    if (i === 0) return null;
    const prev = points[i - 1]!;
    return prev.fold === p.fold && prev.equity > 0 ? p.equity / prev.equity - 1 : null;
  });
  return points.map((p, i) => {
    const slice = rets.slice(Math.max(0, i - window + 1), i + 1).filter((r): r is number => r !== null);
    if (slice.length < window * 0.8) return { t: p.t, vol: null };
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(slice.length - 1, 1);
    return { t: p.t, vol: Math.sqrt(variance) * Math.sqrt(periodsPerYear) };
  });
}

export function mean(values: number[]): number | null {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
}
