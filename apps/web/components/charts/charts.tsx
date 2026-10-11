"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { EmptyState } from "@/components/ui/misc";
import type { EquitySeries } from "@/lib/api/schemas";
import { pct } from "@/lib/format";
import { histogram, mergeSeries, periodReturns, rollingReturn, rollingVolatility } from "@/lib/series";

import { AXIS, BENCHMARK_STYLE, GRID, TOOLTIP } from "./theme";

export type NamedSeries = { key: string; label: string; color: string; series: EquitySeries };

const yearTick = (t: string) => t.slice(0, 7);

function NoData({ height, message = "No data for this selection yet." }: { height: number; message?: string }) {
  return (
    <div style={{ height }} className="flex items-center justify-center">
      <EmptyState title={message} />
    </div>
  );
}

export function EquityChart({
  series,
  benchmark,
  height = 320,
  normalise = true,
}: {
  series: NamedSeries[];
  benchmark?: EquitySeries | null;
  height?: number;
  normalise?: boolean;
}) {
  if (!series.length && !benchmark?.points.length) return <NoData height={height} />;
  const input = series.map((s) => ({ key: s.key, points: s.series.points }));
  if (benchmark) input.push({ key: "__benchmark__", points: benchmark.points });
  const rows = mergeSeries(input, "equity", normalise);
  const labels = Object.fromEntries(series.map((s) => [s.key, s.label]));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="t" {...AXIS} tickFormatter={yearTick} minTickGap={48} />
        <YAxis
          {...AXIS}
          width={56}
          domain={["auto", "auto"]}
          tickFormatter={(v: number) => (normalise ? `${((v - 1) * 100).toFixed(0)}%` : v.toFixed(0))}
        />
        <Tooltip
          {...TOOLTIP}
          formatter={(v, name) => [
            normalise ? pct(Number(v) - 1, 2, true) : Number(v).toFixed(0),
            name === "__benchmark__" ? "Buy & hold" : (labels[String(name)] ?? String(name)),
          ]}
        />
        <ReferenceLine y={normalise ? 1 : undefined} stroke="var(--color-line-strong)" />
        {benchmark ? (
          <Line dataKey="__benchmark__" dot={false} strokeWidth={1.25} isAnimationActive={false} connectNulls {...BENCHMARK_STYLE} />
        ) : null}
        {series.map((s) => (
          <Line key={s.key} dataKey={s.key} stroke={s.color} dot={false} strokeWidth={1.6} isAnimationActive={false} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function DrawdownChart({ series, height = 200 }: { series: NamedSeries[]; height?: number }) {
  if (!series.length) return <NoData height={height} />;
  const rows = mergeSeries(series.map((s) => ({ key: s.key, points: s.series.points })), "drawdown");
  const labels = Object.fromEntries(series.map((s) => [s.key, s.label]));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="t" {...AXIS} tickFormatter={yearTick} minTickGap={48} />
        <YAxis {...AXIS} width={56} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip {...TOOLTIP} formatter={(v, name) => [pct(Number(v), 1), labels[String(name)] ?? String(name)]} />
        {series.map((s) => (
          <Area
            key={s.key}
            dataKey={s.key}
            stroke={s.color}
            fill={s.color}
            fillOpacity={0.08}
            strokeWidth={1.2}
            isAnimationActive={false}
            connectNulls
          />
        ))}
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function ReturnDistribution({ series, height = 220 }: { series: NamedSeries[]; height?: number }) {
  const all = series.flatMap((s) => periodReturns(s.series.points));
  if (all.length < 10) return <NoData height={height} message="Not enough return observations." />;
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const bins = 32;
  const byAgent = series.map((s) => {
    const values = periodReturns(s.series.points);
    // Fix the bin edges across agents so the bars line up.
    return histogram([...values, lo, hi], bins).map((b, i, arr) => ({
      ...b,
      count: b.count - (i === 0 ? 1 : 0) - (i === arr.length - 1 ? 1 : 0),
    }));
  });
  const rows = byAgent[0]!.map((b, i) => {
    const row: Record<string, number | string> = { label: b.label, x: b.x };
    series.forEach((s, j) => {
      row[s.key] = byAgent[j]![i]!.count;
    });
    return row;
  });
  const labels = Object.fromEntries(series.map((s) => [s.key, s.label]));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }} barGap={0} barCategoryGap={1}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="label" {...AXIS} minTickGap={32} />
        <YAxis {...AXIS} width={40} allowDecimals={false} />
        <Tooltip {...TOOLTIP} formatter={(v, name) => [String(v), labels[String(name)] ?? String(name)]} />
        <ReferenceLine x={rows.reduce((best, r) => (Math.abs(Number(r.x)) < Math.abs(Number(best.x)) ? r : best), rows[0]!).label} stroke="var(--color-line-strong)" />
        {series.map((s) => (
          <Bar key={s.key} dataKey={s.key} fill={s.color} fillOpacity={0.75} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function RollingLines({
  series,
  height,
  values,
  pctDigits,
  zeroLine,
}: {
  series: NamedSeries[];
  height: number;
  values: (s: NamedSeries) => { t: string; v: number | null }[];
  pctDigits: number;
  zeroLine?: boolean;
}) {
  if (!series.length) return <NoData height={height} />;
  const merged = new Map<string, Record<string, number | string | null>>();
  for (const s of series) {
    for (const p of values(s)) {
      const row = merged.get(p.t) ?? { t: p.t };
      row[s.key] = p.v;
      merged.set(p.t, row);
    }
  }
  const rows = [...merged.values()].sort((a, b) => String(a.t).localeCompare(String(b.t)));
  const labels = Object.fromEntries(series.map((s) => [s.key, s.label]));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="t" {...AXIS} tickFormatter={yearTick} minTickGap={48} />
        <YAxis {...AXIS} width={48} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
        {zeroLine ? <ReferenceLine y={0} stroke="var(--color-line-strong)" /> : null}
        <Tooltip {...TOOLTIP} formatter={(v, name) => [pct(Number(v), pctDigits), labels[String(name)] ?? String(name)]} />
        {series.map((s) => (
          <Line key={s.key} dataKey={s.key} stroke={s.color} dot={false} strokeWidth={1.3} isAnimationActive={false} connectNulls={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function RollingVolChart({ series, height = 200, window = 20 }: { series: NamedSeries[]; height?: number; window?: number }) {
  return (
    <RollingLines
      series={series}
      height={height}
      pctDigits={1}
      values={(s) => rollingVolatility(s.series.points, window).map((p) => ({ t: p.t, v: p.vol }))}
    />
  );
}

export function RollingReturnChart({ series, height = 200, window = 60 }: { series: NamedSeries[]; height?: number; window?: number }) {
  return (
    <RollingLines
      series={series}
      height={height}
      pctDigits={1}
      zeroLine
      values={(s) => rollingReturn(s.series.points, window).map((p) => ({ t: p.t, v: p.ret }))}
    />
  );
}

export function FoldBars({
  rows,
  keys,
  height = 220,
  format = (v: number) => v.toFixed(2),
}: {
  rows: Record<string, number | string | null>[];
  keys: { key: string; label: string; color: string }[];
  height?: number;
  format?: (v: number) => string;
}) {
  if (!rows.length) return <NoData height={height} />;
  const labels = Object.fromEntries(keys.map((k) => [k.key, k.label]));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="name" {...AXIS} />
        <YAxis {...AXIS} width={48} tickFormatter={(v: number) => format(v)} />
        <Tooltip {...TOOLTIP} formatter={(v, name) => [v === null ? "—" : format(Number(v)), labels[String(name)] ?? String(name)]} />
        <ReferenceLine y={0} stroke="var(--color-line-strong)" />
        <Legend wrapperStyle={{ fontSize: 11, color: "var(--color-fg-muted)" }} formatter={(v) => labels[String(v)] ?? String(v)} />
        {keys.map((k) => (
          <Bar key={k.key} dataKey={k.key} fill={k.color} fillOpacity={0.8} isAnimationActive={false} maxBarSize={28} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

export function GenerationScatter({
  points,
  height = 260,
  onSelect,
}: {
  points: { generation: number; score: number; algorithm: string; color: string; id: string; benchmark: boolean }[];
  height?: number;
  onSelect?: (id: string) => void;
}) {
  if (!points.length) return <NoData height={height} />;
  const jittered = points.map((p, i) => ({ ...p, x: p.generation + (((i * 37) % 11) - 5) * 0.025 }));
  const gens = Math.max(...points.map((p) => p.generation));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis
          type="number"
          dataKey="x"
          domain={[-0.5, gens + 0.5]}
          ticks={Array.from({ length: gens + 1 }, (_, i) => i)}
          tickFormatter={(v: number) => `G${v}`}
          {...AXIS}
        />
        <YAxis type="number" dataKey="score" {...AXIS} width={48} tickFormatter={(v: number) => v.toFixed(1)} />
        <ZAxis range={[48, 48]} />
        <Tooltip
          {...TOOLTIP}
          formatter={(v, name) => [typeof v === "number" ? v.toFixed(3) : String(v), String(name)]}
          labelFormatter={() => ""}
        />
        <Scatter data={jittered} isAnimationActive={false} onClick={(d) => onSelect?.((d as unknown as { id: string }).id)}>
          {jittered.map((p) => (
            <Cell
              key={p.id}
              fill={p.color}
              fillOpacity={p.benchmark ? 0.25 : 0.85}
              stroke={p.benchmark ? p.color : "none"}
              strokeDasharray={p.benchmark ? "2 2" : undefined}
              cursor={onSelect ? "pointer" : undefined}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}

export function PriceChart({ points, height = 220 }: { points: { timestamp: string; close: number }[]; height?: number }) {
  if (!points.length) return <NoData height={height} />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={points} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="price-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--color-accent)" stopOpacity={0.18} />
            <stop offset="100%" stopColor="var(--color-accent)" stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="timestamp" {...AXIS} tickFormatter={yearTick} minTickGap={48} />
        <YAxis {...AXIS} width={48} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(0)} />
        <Tooltip {...TOOLTIP} formatter={(v) => [Number(v).toFixed(2), "close"]} />
        <Area dataKey="close" stroke="var(--color-accent)" fill="url(#price-fill)" strokeWidth={1.4} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function GenerationProgress({
  rows,
  height = 240,
}: {
  rows: { generation: number; best: number | null; median: number | null; random: number | null }[];
  height?: number;
}) {
  if (!rows.length) return <NoData height={height} />;
  const labels: Record<string, string> = { best: "Best", median: "Median", random: "Random baseline" };
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
        <CartesianGrid {...GRID} />
        <XAxis dataKey="generation" {...AXIS} tickFormatter={(v: number) => `G${v}`} />
        <YAxis {...AXIS} width={48} tickFormatter={(v: number) => v.toFixed(1)} />
        <Tooltip {...TOOLTIP} labelFormatter={(v) => `Generation ${String(v)}`}
          formatter={(v, name) => [v === null ? "—" : Number(v).toFixed(3), labels[String(name)] ?? String(name)]} />
        <Legend wrapperStyle={{ fontSize: 11, color: "var(--color-fg-muted)" }} formatter={(v) => labels[String(v)] ?? String(v)} />
        <Line dataKey="best" stroke="var(--color-accent)" strokeWidth={1.8} isAnimationActive={false} connectNulls />
        <Line dataKey="median" stroke="var(--color-pos)" strokeWidth={1.4} isAnimationActive={false} connectNulls />
        <Line dataKey="random" dot={false} strokeWidth={1.2} isAnimationActive={false} connectNulls {...BENCHMARK_STYLE} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function ValidationVsTest({
  points,
  height = 260,
}: {
  points: { id: string; label: string; x: number; y: number; color: string }[];
  height?: number;
}) {
  if (!points.length) return <NoData height={height} message="Run the final test to compare validation with test." />;
  const all = points.flatMap((p) => [p.x, p.y]);
  const lo = Math.min(0, ...all);
  const hi = Math.max(0, ...all);
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 12, bottom: 16, left: 0 }}>
        <CartesianGrid {...GRID} vertical />
        <XAxis type="number" dataKey="x" name="Validation Sharpe" domain={[lo, hi]} {...AXIS} tickFormatter={(v: number) => v.toFixed(1)}
          label={{ value: "validation Sharpe", position: "insideBottom", offset: -8, fill: "var(--color-fg-subtle)", fontSize: 11 }} />
        <YAxis type="number" dataKey="y" name="Test Sharpe" domain={[lo, hi]} {...AXIS} width={48} tickFormatter={(v: number) => v.toFixed(1)} />
        <ZAxis range={[60, 60]} />
        <ReferenceLine segment={[{ x: lo, y: lo }, { x: hi, y: hi }]} stroke="var(--color-line-strong)" strokeDasharray="4 3" />
        <Tooltip {...TOOLTIP} formatter={(v, name) => [Number(v).toFixed(2), String(name)]} labelFormatter={() => ""} />
        <Scatter data={points} isAnimationActive={false}>
          {points.map((p) => <Cell key={p.id} fill={p.color} />)}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
