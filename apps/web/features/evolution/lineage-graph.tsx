"use client";

import * as React from "react";

import type { LineageGraph as Graph } from "@/lib/api/schemas";
import { algoColor } from "@/lib/format";

const COL_W = 150;
const ROW_H = 34;
const PAD_X = 40;
const PAD_Y = 34;
const R = 9;

export type LayoutNode = Graph["nodes"][number] & { x: number; y: number };

/** Columns are generations; within a column, agents are ordered by validation rank. */
export function layoutLineage(graph: Graph): { nodes: LayoutNode[]; width: number; height: number } {
  const byGen = new Map<number, Graph["nodes"]>();
  for (const n of graph.nodes) byGen.set(n.generation, [...(byGen.get(n.generation) ?? []), n]);
  const gens = [...byGen.keys()].sort((a, b) => a - b);
  let maxRows = 0;
  const nodes: LayoutNode[] = [];
  gens.forEach((g, ci) => {
    const col = [...(byGen.get(g) ?? [])].sort(
      (a, b) => (a.rank ?? 1e9) - (b.rank ?? 1e9) || a.agent_id.localeCompare(b.agent_id),
    );
    maxRows = Math.max(maxRows, col.length);
    col.forEach((n, ri) => nodes.push({ ...n, x: PAD_X + ci * COL_W, y: PAD_Y + ri * ROW_H }));
  });
  return { nodes, width: PAD_X * 2 + Math.max(0, gens.length - 1) * COL_W, height: PAD_Y * 2 + Math.max(0, maxRows - 1) * ROW_H };
}

export function LineageGraph({
  graph,
  selected,
  onSelect,
}: {
  graph: Graph;
  selected?: string;
  onSelect: (id: string) => void;
}) {
  const { nodes, width, height } = React.useMemo(() => layoutLineage(graph), [graph]);
  const pos = new Map(nodes.map((n) => [n.agent_id, n]));
  const related = new Set<string>();
  if (selected) {
    related.add(selected);
    for (const e of graph.edges) {
      if (e.child_id === selected) related.add(e.parent_id);
      if (e.parent_id === selected) related.add(e.child_id);
    }
  }
  const gens = [...new Set(nodes.map((n) => n.generation))].sort((a, b) => a - b);

  return (
    <div className="scroll-thin overflow-auto">
      <svg width={Math.max(width, 320)} height={height} role="img" aria-label="Agent lineage graph" className="block">
        {gens.map((g, i) => (
          <text key={g} x={PAD_X + i * COL_W} y={14} textAnchor="middle" className="fill-fg-subtle text-[10px] uppercase tracking-wider">
            Gen {g}
          </text>
        ))}
        {graph.edges.map((e) => {
          const a = pos.get(e.parent_id);
          const b = pos.get(e.child_id);
          if (!a || !b) return null;
          const on = selected && (e.parent_id === selected || e.child_id === selected);
          const mx = (a.x + b.x) / 2;
          return (
            <path
              key={`${e.parent_id}-${e.child_id}`}
              d={`M${a.x + R},${a.y} C${mx},${a.y} ${mx},${b.y} ${b.x - R},${b.y}`}
              fill="none"
              stroke={on ? "var(--color-accent)" : "var(--color-line-strong)"}
              strokeWidth={on ? 1.6 : 1}
              strokeDasharray={e.operation === "elite" ? undefined : e.operation === "crossover" ? "4 3" : undefined}
              opacity={selected && !on ? 0.35 : 0.9}
            >
              <title>{`${e.operation}${e.mutations.length ? `: ${e.mutations.join(", ")}` : ""}`}</title>
            </path>
          );
        })}
        {nodes.map((n) => {
          const isSel = n.agent_id === selected;
          const dim = selected && !related.has(n.agent_id);
          const excluded = n.status === "excluded" || n.status === "insufficient_evidence";
          return (
            <g
              key={n.agent_id}
              transform={`translate(${n.x},${n.y})`}
              role="button"
              tabIndex={0}
              aria-label={`${n.algorithm} agent generation ${n.generation}${n.rank ? ` rank ${n.rank}` : ""}`}
              aria-pressed={isSel}
              onClick={() => onSelect(n.agent_id)}
              onKeyDown={(ev) => (ev.key === "Enter" || ev.key === " ") && (ev.preventDefault(), onSelect(n.agent_id))}
              className="cursor-pointer outline-none"
              opacity={dim ? 0.3 : 1}
            >
              {isSel ? <circle r={R + 4} fill="none" stroke="var(--color-accent)" strokeWidth={1.5} /> : null}
              <circle
                r={R}
                fill={n.is_benchmark || excluded ? "var(--color-panel)" : algoColor(n.algorithm)}
                fillOpacity={n.is_benchmark || excluded ? 1 : 0.85}
                stroke={algoColor(n.algorithm)}
                strokeWidth={1.5}
                strokeDasharray={n.is_benchmark ? "2 2" : undefined}
              />
              {n.origin === "immigrant" ? <circle r={2.5} fill="var(--color-bg)" /> : null}
              <text x={R + 5} y={3.5} className="fill-fg-muted font-mono text-[10px]">
                {n.rank ? `#${n.rank}` : n.is_benchmark ? "bench" : "–"}
              </text>
              <title>{`${n.algorithm} · ${n.origin}${n.score !== null ? ` · score ${n.score.toFixed(3)}` : ""}`}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
