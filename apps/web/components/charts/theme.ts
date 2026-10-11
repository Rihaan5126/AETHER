export const AXIS = {
  stroke: "var(--color-line-strong)",
  tick: { fill: "var(--color-fg-subtle)", fontSize: 11, fontFamily: "var(--font-mono)" },
  tickLine: false,
} as const;

export const GRID = { stroke: "var(--color-line)", strokeDasharray: "2 4", vertical: false } as const;

export const TOOLTIP = {
  contentStyle: {
    background: "var(--color-panel-2)",
    border: "1px solid var(--color-line-strong)",
    borderRadius: 6,
    fontSize: 12,
    fontFamily: "var(--font-mono)",
    color: "var(--color-fg)",
    boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
  },
  labelStyle: { color: "var(--color-fg-muted)", marginBottom: 4 },
  itemStyle: { padding: 0 },
  cursor: { stroke: "var(--color-line-strong)", strokeWidth: 1 },
} as const;

/** Distinct colours for comparing several agents (independent of algorithm colour). */
export const SERIES_COLORS = ["#8fb3ff", "#5cc497", "#e0b064", "#c49bf0", "#ec7b82", "#6fd3d6", "#f2a6c8", "#b8c27a"];

export const BENCHMARK_STYLE = { stroke: "var(--color-fg-subtle)", strokeDasharray: "4 3" } as const;
