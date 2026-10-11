"use client";

import { X } from "lucide-react";
import * as React from "react";

import { SERIES_COLORS } from "@/components/charts/theme";
import { DrawdownChart, EquityChart, FoldBars, ReturnDistribution, RollingReturnChart, RollingVolChart } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote } from "@/components/domain";
import { ExperimentSelect, useExperimentParam } from "@/components/experiment-select";
import { Badge } from "@/components/ui/badge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Select } from "@/components/ui/input";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import type { LeaderboardRow } from "@/lib/api/schemas";
import { dateTime, int, money, num, pct, shortAgentId, signClass } from "@/lib/format";
import { flattenGenome } from "@/lib/genome";
import { useAgentMetrics, useAgents, useEquityCurves, useLeaderboard, useTournaments, useTrades } from "@/lib/queries";

const MAX_COMPARE = 4;

const METRICS: { label: string; get: (r: LeaderboardRow) => string; better?: "high" | "low"; raw?: (r: LeaderboardRow) => number | null }[] = [
  { label: "Validation score", get: (r) => num(r.score, 3), better: "high", raw: (r) => r.score },
  { label: "Total return", get: (r) => pct(r.total_return, 1, true), better: "high", raw: (r) => r.total_return },
  { label: "Annualised return", get: (r) => pct(r.annualized_return, 1, true), better: "high", raw: (r) => r.annualized_return },
  { label: "Excess vs buy & hold", get: (r) => pct(r.excess_return, 1, true), better: "high", raw: (r) => r.excess_return },
  { label: "Sharpe", get: (r) => num(r.sharpe_ratio), better: "high", raw: (r) => r.sharpe_ratio },
  { label: "Sortino", get: (r) => num(r.sortino_ratio), better: "high", raw: (r) => r.sortino_ratio },
  { label: "Max drawdown", get: (r) => pct(r.max_drawdown), better: "low", raw: (r) => r.max_drawdown },
  { label: "Fold stability (std Sharpe)", get: (r) => num(r.stability), better: "low", raw: (r) => r.stability },
  { label: "Annual turnover", get: (r) => num(r.turnover, 1), better: "low", raw: (r) => r.turnover },
  { label: "Exposure", get: (r) => pct(r.exposure, 0) },
  { label: "Trades", get: (r) => int(r.n_trades) },
  { label: "Round trips", get: (r) => int(r.n_round_trips) },
  { label: "Win rate", get: (r) => pct(r.win_rate, 0) },
  { label: "Status", get: (r) => r.status.replace("_", " ") },
];

function FoldComparison({ ids, names }: { ids: string[]; names: Record<string, LeaderboardRow> }) {
  // Hooks are called a fixed number of times (MAX_COMPARE) to keep hook order stable.
  const q0 = useAgentMetrics(ids[0]);
  const q1 = useAgentMetrics(ids[1]);
  const q2 = useAgentMetrics(ids[2]);
  const q3 = useAgentMetrics(ids[3]);
  const queries = [q0, q1, q2, q3].slice(0, ids.length);
  const folds = new Map<number, Record<string, number | string | null>>();
  queries.forEach((q, i) => {
    const id = ids[i]!;
    for (const f of q.data?.results.at(-1)?.fold_metrics ?? []) {
      const row = folds.get(f.fold) ?? { name: `Fold ${f.fold + 1}` };
      row[id] = f.sharpe_ratio;
      folds.set(f.fold, row);
    }
  });
  return (
    <FoldBars
      rows={[...folds.entries()].sort((a, b) => a[0] - b[0]).map(([, r]) => r)}
      keys={ids.map((id, i) => ({ key: id, label: names[id]?.agent_name ?? id, color: SERIES_COLORS[i]! }))}
    />
  );
}

function ConfigurationCard({
  ids,
  names,
  experimentId,
  generation,
}: {
  ids: string[];
  names: Record<string, { agent_name: string }>;
  experimentId: string | undefined;
  generation: number;
}) {
  const agents = useAgents({ experiment_id: experimentId, generation });
  const byId = Object.fromEntries((agents.data?.items ?? []).map((a) => [a.id, a]));
  const flat = ids.map((id) => {
    const a = byId[id];
    if (!a) return {} as Record<string, string>;
    const genome = { ...(a.genome as Record<string, unknown>) };
    const features = genome.features;
    delete genome.features;
    return {
      ...flattenGenome(genome),
      ...(Array.isArray(features) ? { features: `${features.length} selected` } : {}),
      seed: String(a.seed),
      origin: a.origin,
    };
  });
  const keys = [...new Set(flat.flatMap((f) => Object.keys(f)))].filter((k) => k !== "family").sort();
  return (
    <Card className="xl:col-span-2">
      <CardHeader title="Model configuration" description="Genome of each agent; rows where the agents differ are highlighted" />
      <CardBody className="p-0">
        {agents.isLoading ? (
          <Skeleton className="m-4 h-40" />
        ) : (
          <Table>
            <thead><tr><Th>Gene</Th>{ids.map((id) => <Th key={id} className="text-right">{names[id]?.agent_name ?? shortAgentId(id)}</Th>)}</tr></thead>
            <tbody>
              {keys.map((k) => {
                const vals = flat.map((f) => f[k] ?? "—");
                const differs = new Set(vals).size > 1;
                return (
                  <Tr key={k}>
                    <Td className={differs ? "text-fg" : "text-fg-subtle"}>{k}</Td>
                    {vals.map((v, i) => <Td key={ids[i]} className={`text-right font-mono ${differs ? "text-accent" : "text-fg-muted"}`}>{v}</Td>)}
                  </Tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </CardBody>
    </Card>
  );
}

export function ComparePage() {
  const { experimentId, setExperiment, params } = useExperimentParam();
  const tournaments = useTournaments({ experiment_id: experimentId });
  const items = [...(tournaments.data?.items ?? [])].sort((a, b) => b.generation - a.generation);
  const selected = items.find((t) => String(t.generation) === params.get("generation")) ?? items[0];
  const board = useLeaderboard(selected?.id);
  const rows = board.data?.rows ?? [];
  const names = Object.fromEntries(rows.map((r) => [r.agent_id, r]));

  const requested = (params.get("agents") ?? "").split(",").filter((id) => id && names[id]);
  const ids = requested.length
    ? requested.slice(0, MAX_COMPARE)
    : rows.filter((r) => r.status === "eligible").slice(0, 2).map((r) => r.agent_id)
        .concat(rows.filter((r) => r.algorithm === "random").slice(0, 1).map((r) => r.agent_id));
  const setIds = (next: string[]) => experimentId && setExperiment(experimentId, { agents: next.join(",") || undefined });

  const curves = useEquityCurves(selected?.id, ids.length ? ids : undefined, 1500);
  const [tradeAgent, setTradeAgent] = React.useState<string>();
  const tradeFor = tradeAgent && ids.includes(tradeAgent) ? tradeAgent : ids[0];
  const trades = useTrades(selected?.id, tradeFor, 25);

  // Per-slot colours: several compared agents often share an algorithm.
  const colorFor = (_id: string, i: number) => SERIES_COLORS[i % SERIES_COLORS.length]!;
  const named = (curves.data?.series ?? [])
    .filter((s) => ids.includes(s.agent_id))
    .map((s) => ({ key: s.agent_id, label: names[s.agent_id]?.agent_name ?? s.agent_id, color: colorFor(s.agent_id, ids.indexOf(s.agent_id)), series: s }));
  const bench = curves.data?.benchmark;
  const withBench = bench ? [...named, { key: "__benchmark__", label: "Buy & hold", color: "var(--color-fg-subtle)", series: bench }] : named;

  const best = (m: (typeof METRICS)[number]) => {
    if (!m.better || !m.raw) return undefined;
    const vals = ids.map((id) => (names[id] ? m.raw!(names[id]!) : null)).filter((v): v is number => v !== null);
    if (vals.length < 2) return undefined;
    return m.better === "high" ? Math.max(...vals) : Math.min(...vals);
  };

  return (
    <>
      <PageHeader
        title="Agent Comparison"
        description="Up to four agents from the same tournament, on identical validation folds, costs and risk rules."
        actions={
          <>
            <ExperimentSelect value={experimentId} onChange={(id) => setExperiment(id, { generation: undefined, agents: undefined })} />
            <Select aria-label="Generation" className="w-36" value={selected ? String(selected.generation) : ""}
              onChange={(e) => experimentId && setExperiment(experimentId, { generation: e.target.value, agents: undefined })}>
              {items.map((t) => <option key={t.id} value={t.generation}>Generation {t.generation}</option>)}
            </Select>
          </>
        }
      />
      {board.isError ? <ErrorState error={board.error} /> : null}
      {!tournaments.isLoading && !items.length ? <EmptyState title="Nothing to compare yet">Run an experiment first.</EmptyState> : null}
      {selected ? (
        <div className="space-y-6">
          <Card>
            <CardBody className="flex flex-wrap items-center gap-2">
              {ids.map((id, i) => (
                <Badge key={id} className="gap-2 py-1 pl-2 pr-1 text-xs">
                  <span className="size-2 rounded-full" style={{ background: colorFor(id, i) }} />
                  {names[id]?.agent_name ?? shortAgentId(id)}
                  <button type="button" aria-label={`Remove ${names[id]?.agent_name ?? id}`} className="rounded p-0.5 hover:bg-line" onClick={() => setIds(ids.filter((x) => x !== id))}>
                    <X className="size-3" />
                  </button>
                </Badge>
              ))}
              <Select
                aria-label="Add agent"
                className="w-64"
                value=""
                disabled={ids.length >= MAX_COMPARE}
                onChange={(e) => e.target.value && setIds([...ids, e.target.value])}
              >
                <option value="">{ids.length >= MAX_COMPARE ? "Maximum of four agents" : "Add an agent…"}</option>
                {rows.filter((r) => !ids.includes(r.agent_id)).map((r) => (
                  <option key={r.agent_id} value={r.agent_id}>{r.rank ? `#${r.rank} ` : ""}{r.agent_name} ({r.status})</option>
                ))}
              </Select>
              <span className="ml-auto"><PeriodBadge period="validation" /></span>
            </CardBody>
          </Card>

          {board.isLoading ? <Skeleton className="h-80" /> : (
            <>
              <div className="grid gap-6 xl:grid-cols-2">
                <Card><CardHeader title="Equity" description="Growth of 1, folds chained" /><CardBody><EquityChart series={named} benchmark={bench} /></CardBody></Card>
                <Card><CardHeader title="Drawdown" /><CardBody><DrawdownChart series={withBench} height={320} /></CardBody></Card>
                <Card><CardHeader title="Daily return distribution" /><CardBody><ReturnDistribution series={withBench} /></CardBody></Card>
                <Card><CardHeader title="Rolling 20-day volatility" description="Annualised" /><CardBody><RollingVolChart series={withBench} /></CardBody></Card>
                <Card><CardHeader title="Rolling 60-day return" description="Trailing return of the chained validation curve" /><CardBody><RollingReturnChart series={withBench} /></CardBody></Card>
                <ConfigurationCard ids={ids} names={names} experimentId={experimentId} generation={selected.generation} />
              </div>

              <div className="grid gap-6 xl:grid-cols-3">
                <Card className="xl:col-span-2">
                  <CardHeader title="Metrics" description="Best value per row highlighted where higher/lower is unambiguously better" />
                  <CardBody className="p-0">
                    <Table>
                      <thead><tr><Th>Metric</Th>{ids.map((id) => <Th key={id} className="text-right">{names[id]?.agent_name ?? id}</Th>)}</tr></thead>
                      <tbody>
                        {METRICS.map((m) => {
                          const b = best(m);
                          return (
                            <Tr key={m.label}>
                              <Td className="text-fg-muted">{m.label}</Td>
                              {ids.map((id) => {
                                const r = names[id];
                                const isBest = r && b !== undefined && m.raw?.(r) === b;
                                return <Td key={id} className={`text-right font-mono tabular ${isBest ? "text-accent" : ""}`}>{r ? m.get(r) : "—"}</Td>;
                              })}
                            </Tr>
                          );
                        })}
                        <Tr>
                          <Td className="text-fg-muted">Algorithm</Td>
                          {ids.map((id) => <Td key={id} className="text-right"><AlgoTag algorithm={names[id]?.algorithm ?? ""} className="justify-end" /></Td>)}
                        </Tr>
                      </tbody>
                    </Table>
                  </CardBody>
                </Card>
                <Card>
                  <CardHeader title="Sharpe by fold" description="Consistency matters more than one good fold" />
                  <CardBody><FoldComparison ids={ids} names={names} /></CardBody>
                </Card>
              </div>

              <Card>
                <CardHeader
                  title="Trades"
                  description="Executed at the next bar's open after the signal; slippage and fees included"
                  actions={
                    <Select aria-label="Trades for agent" className="w-56" value={tradeFor ?? ""} onChange={(e) => setTradeAgent(e.target.value)}>
                      {ids.map((id) => <option key={id} value={id}>{names[id]?.agent_name ?? id}</option>)}
                    </Select>
                  }
                />
                <CardBody className="p-0">
                  {trades.data?.items.length ? (
                    <Table>
                      <thead><tr><Th>Signal</Th><Th>Executed</Th><Th>Side</Th><Th className="text-right">Qty</Th><Th className="text-right">Price</Th><Th className="text-right">Notional</Th><Th className="text-right">Fees</Th><Th className="text-right">Slippage</Th><Th className="text-right">Target</Th></tr></thead>
                      <tbody>
                        {trades.data.items.map((t) => (
                          <Tr key={t.id}>
                            <Td className="text-fg-subtle">{t.signal_time.slice(0, 10)}</Td>
                            <Td>{dateTime(t.timestamp).split(",")[0]}</Td>
                            <Td className={t.side === "buy" ? "text-pos" : "text-neg"}>{t.side}</Td>
                            <Td className="text-right font-mono">{t.quantity.toFixed(3)}</Td>
                            <Td className="text-right font-mono">{t.execution_price.toFixed(2)}</Td>
                            <Td className="text-right font-mono">{money(t.notional)}</Td>
                            <Td className="text-right font-mono">{money(t.fees)}</Td>
                            <Td className="text-right font-mono">{money(t.slippage)}</Td>
                            <Td className={`text-right font-mono ${signClass(t.target_weight)}`}>{pct(t.target_weight, 0)}</Td>
                          </Tr>
                        ))}
                      </tbody>
                    </Table>
                  ) : <EmptyState title="No trades recorded" className="py-8" />}
                  {trades.data && trades.data.total > trades.data.items.length ? (
                    <p className="border-t border-line px-4 py-2 text-xs text-fg-subtle">Showing the first {trades.data.items.length} of {trades.data.total} trades.</p>
                  ) : null}
                </CardBody>
              </Card>
              <ResearchNote>Return distributions and volatility are computed from the chained validation equity curves shown above; fold boundaries reset to cash, so the first day of each fold has zero exposure.</ResearchNote>
            </>
          )}
        </div>
      ) : null}
    </>
  );
}
