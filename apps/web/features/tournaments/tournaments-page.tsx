"use client";

import { ArrowDown, ArrowUp } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { EquityChart } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote, StatusBadge, WarningBadges } from "@/components/domain";
import { ExperimentSelect, useExperimentParam } from "@/components/experiment-select";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Select } from "@/components/ui/input";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import type { LeaderboardRow } from "@/lib/api/schemas";
import { SERIES_COLORS } from "@/components/charts/theme";
import { algoLabel, int, num, pct, signClass } from "@/lib/format";
import { useEquityCurves, useLeaderboard, useTournaments } from "@/lib/queries";

type SortKey = "rank" | "score" | "total_return" | "sharpe_ratio" | "max_drawdown" | "turnover" | "n_trades" | "win_rate" | "excess_return";

const COLUMNS: { key: SortKey; label: string; fmt: (r: LeaderboardRow) => string; signed?: boolean; title?: string }[] = [
  { key: "score", label: "Score", fmt: (r) => num(r.score, 3), title: "Validation score per the ranking policy" },
  { key: "total_return", label: "Return", fmt: (r) => pct(r.total_return, 1, true), signed: true },
  { key: "excess_return", label: "vs B&H", fmt: (r) => pct(r.excess_return, 1, true), signed: true, title: "Return minus buy-and-hold over the same folds" },
  { key: "sharpe_ratio", label: "Sharpe", fmt: (r) => num(r.sharpe_ratio) },
  { key: "max_drawdown", label: "Max DD", fmt: (r) => pct(r.max_drawdown) },
  { key: "turnover", label: "Turnover", fmt: (r) => num(r.turnover, 1), title: "Annualised one-way turnover" },
  { key: "n_trades", label: "Trades", fmt: (r) => int(r.n_trades) },
  { key: "win_rate", label: "Win rate", fmt: (r) => pct(r.win_rate, 0) },
];

function sortRows(rows: LeaderboardRow[], key: SortKey, desc: boolean): LeaderboardRow[] {
  const val = (r: LeaderboardRow): number | null => (key === "rank" ? r.rank : (r[key] as number | null));
  return [...rows].sort((a, b) => {
    const va = val(a);
    const vb = val(b);
    if (va === null && vb === null) return 0;
    if (va === null) return 1; // undefined metrics always sort last
    if (vb === null) return -1;
    return desc ? vb - va : va - vb;
  });
}

function RankingPolicy({ policy }: { policy: Record<string, unknown> }) {
  return (
    <ResearchNote>
      <p className="mb-1 font-medium text-fg">How agents are ranked</p>
      <p>
        Score = mean fold Sharpe − {String(policy.stability_weight ?? 0.5)} × std(fold Sharpe) − {String(policy.drawdown_weight ?? 1)} × max drawdown − {String(policy.turnover_weight ?? 0.01)} × annual turnover.
        Agents with max drawdown above {pct(Number(policy.max_drawdown_limit ?? 0.5), 0)} are <em>excluded</em>; fewer than {String(policy.min_round_trips ?? 5)} completed round trips is <em>insufficient evidence</em>.
        Benchmarks (random, buy-and-hold) are scored and ranked alongside for reference, but are never selected as parents. Rankings use validation folds only; the final test period is never visible here.
      </p>
    </ResearchNote>
  );
}

export function TournamentsPage() {
  const { experimentId, setExperiment, params } = useExperimentParam();
  const tournaments = useTournaments({ experiment_id: experimentId });
  const items = tournaments.data?.items ?? [];
  const genParam = params.get("generation");
  const selected =
    items.find((t) => String(t.generation) === genParam) ??
    [...items].sort((a, b) => b.generation - a.generation)[0];
  const board = useLeaderboard(selected?.id);
  const [sort, setSort] = React.useState<{ key: SortKey; desc: boolean }>({ key: "rank", desc: false });
  const [algo, setAlgo] = React.useState("");
  const [showBench, setShowBench] = React.useState(true);

  const rows = React.useMemo(() => {
    const all = board.data?.rows ?? [];
    return sortRows(
      all.filter((r) => (!algo || r.algorithm === algo) && (showBench || !r.is_benchmark)),
      sort.key,
      sort.desc,
    );
  }, [board.data, algo, showBench, sort]);
  const top = rows.filter((r) => r.status === "eligible").slice(0, 5).map((r) => r.agent_id);
  const curves = useEquityCurves(selected?.id, top.length ? top : undefined);
  const names = Object.fromEntries((board.data?.rows ?? []).map((r) => [r.agent_id, r]));
  const algorithms = [...new Set((board.data?.rows ?? []).map((r) => r.algorithm))];

  const header = (key: SortKey, label: string, title?: string) => (
    <Th className={key === "rank" ? "" : "text-right"} title={title}>
      <button
        type="button"
        className="inline-flex items-center gap-1 hover:text-fg"
        onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : key !== "max_drawdown" && key !== "rank" && key !== "turnover" }))}
      >
        {label}
        {sort.key === key ? (sort.desc ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />) : null}
      </button>
    </Th>
  );

  return (
    <>
      <PageHeader
        title="Tournaments"
        description="Every agent is evaluated on the same walk-forward validation folds, with the same costs and risk rules."
        actions={
          <>
            <ExperimentSelect value={experimentId} onChange={(id) => setExperiment(id, { generation: undefined })} />
            <Select aria-label="Generation" className="w-36" value={selected ? String(selected.generation) : ""}
              onChange={(e) => experimentId && setExperiment(experimentId, { generation: e.target.value })}>
              {[...items].sort((a, b) => a.generation - b.generation).map((t) => (
                <option key={t.id} value={t.generation}>Generation {t.generation}</option>
              ))}
            </Select>
          </>
        }
      />
      {tournaments.isError ? <ErrorState error={tournaments.error} /> : null}
      {!tournaments.isLoading && !items.length ? (
        <EmptyState title="No tournaments yet" action={<Link className="text-sm text-accent hover:underline" href="/experiments/new">Create an experiment</Link>}>
          Run an experiment to produce validation tournaments.
        </EmptyState>
      ) : null}
      {selected ? (
        <div className="space-y-6">
          <Card>
            <CardHeader
              title={`Leaderboard · generation ${selected.generation}`}
              description={`${selected.n_agents} agents · ${selected.experiment_name} · config ${selected.config_fingerprint.slice(0, 10)}`}
              actions={
                <div className="flex items-center gap-2">
                  <PeriodBadge period="validation" />
                  <Select aria-label="Algorithm filter" className="w-40" value={algo} onChange={(e) => setAlgo(e.target.value)}>
                    <option value="">All algorithms</option>
                    {algorithms.map((a) => <option key={a} value={a}>{algoLabel(a)}</option>)}
                  </Select>
                  <label className="flex items-center gap-1.5 text-xs text-fg-muted">
                    <input type="checkbox" checked={showBench} onChange={(e) => setShowBench(e.target.checked)} className="accent-[var(--color-accent)]" />
                    Benchmarks
                  </label>
                </div>
              }
            />
            <CardBody className="p-0">
              {board.isError ? <ErrorState error={board.error} className="m-4" /> : null}
              {board.isLoading ? <Skeleton className="m-4 h-64" /> : (
                <Table aria-label="Leaderboard">
                  <thead>
                    <tr>
                      {header("rank", "Rank")}
                      <Th>Agent</Th>
                      <Th>Status</Th>
                      {COLUMNS.map((c) => <React.Fragment key={c.key}>{header(c.key, c.label, c.title)}</React.Fragment>)}
                      <Th>Warnings</Th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r) => (
                      <Tr key={r.agent_id} className={r.is_benchmark ? "bg-panel-2/60" : undefined}>
                        <Td className="font-mono">{r.rank ? `#${r.rank}` : "—"}</Td>
                        <Td>
                          <Link href={`/agents/${r.agent_id}`} className="block hover:text-accent">
                            <span className="font-medium">{r.agent_name}</span>
                            <span className="mt-0.5 flex items-center gap-2 text-[11px] text-fg-subtle"><AlgoTag algorithm={r.algorithm} /> gen {r.generation}</span>
                          </Link>
                        </Td>
                        <Td title={r.status_reason}><StatusBadge status={r.status} /></Td>
                        {COLUMNS.map((c) => (
                          <Td key={c.key} className={`text-right font-mono tabular ${c.signed ? signClass(r[c.key] as number | null) : ""}`}>{c.fmt(r)}</Td>
                        ))}
                        <Td><WarningBadges warnings={r.warnings} max={2} /></Td>
                      </Tr>
                    ))}
                  </tbody>
                </Table>
              )}
            </CardBody>
          </Card>
          <div className="grid gap-6 xl:grid-cols-3">
            <Card className="xl:col-span-2">
              <CardHeader title="Top eligible agents vs buy-and-hold" description="Validation folds chained; each fold starts from cash" />
              <CardBody>
                <EquityChart
                  series={(curves.data?.series ?? []).map((s, i) => ({
                    key: s.agent_id,
                    label: names[s.agent_id]?.agent_name ?? s.agent_id,
                    color: SERIES_COLORS[i % SERIES_COLORS.length]!,
                    series: s,
                  }))}
                  benchmark={curves.data?.benchmark}
                />
              </CardBody>
            </Card>
            <div className="space-y-4">
              {board.data ? <RankingPolicy policy={board.data.ranking_policy} /> : null}
              <ResearchNote>
                A high rank on validation is evidence, not proof. With many agents compared, the best score is inflated by selection; see Research Analytics for the multiple-comparisons estimate.
              </ResearchNote>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
