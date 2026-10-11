"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Beaker, Play } from "lucide-react";
import Link from "next/link";
import { z } from "zod";

import { EquityChart } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote, StatusBadge } from "@/components/domain";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState, Progress, Skeleton, Stat } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import { api } from "@/lib/api/client";
import * as S from "@/lib/api/schemas";
import { algoColor, dateTime, num, pct, relative, signClass } from "@/lib/format";
import { useEquityCurves, useExperiment, useExperiments, useLeaderboard } from "@/lib/queries";

function useCount(path: string, query: Record<string, string> = {}) {
  return useQuery({
    queryKey: ["count", path, query],
    queryFn: () => api.get(S.page(z.unknown()), path, { ...query, limit: 1 }),
    select: (d) => d.total,
  });
}

export function OverviewPage() {
  const experiments = useExperiments({ limit: 20 });
  const agents = useCount("/agents");
  const completed = useCount("/experiments", { status: "completed" });
  const tournaments = useCount("/tournaments");
  const latest = experiments.data?.items[0];
  const latestCompleted = experiments.data?.items.find((e) => e.status === "completed");
  const detail = useExperiment(latestCompleted?.id);
  const finalGen = detail.data?.generations.at(-1);
  const board = useLeaderboard(finalGen?.tournament_id);
  const top = board.data?.rows.find((r) => r.status === "eligible");
  const bh = board.data?.rows.find((r) => r.algorithm === "buy_and_hold");
  const curves = useEquityCurves(finalGen?.tournament_id, top ? [top.agent_id] : undefined);

  if (experiments.isError) return <ErrorState error={experiments.error} />;
  const empty = experiments.data && experiments.data.total === 0;

  return (
    <>
      <PageHeader
        title="Overview"
        description="Populations of trading agents are trained, evaluated on walk-forward validation folds and evolved. Every number below comes from a stored simulation run."
        actions={
          <Button asChild variant="primary">
            <Link href="/experiments/new">
              <Play /> New experiment
            </Link>
          </Button>
        }
      />

      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-4">
        {[
          { label: "Agents", value: agents.data, sub: "across all experiments" },
          { label: "Completed experiments", value: completed.data, sub: `${experiments.data?.total ?? "—"} total` },
          { label: "Tournaments", value: tournaments.data, sub: "one per generation" },
          {
            label: "Generations completed",
            value: latest ? (latest.current_generation !== null ? latest.current_generation + 1 : 0) : undefined,
            sub: latest ? `of ${latest.planned_generations} · ${latest.name}` : "no experiment yet",
          },
        ].map((s) => (
          <div key={s.label} className="bg-panel p-4">
            {s.value === undefined && !empty ? <Skeleton className="h-12" /> : <Stat label={s.label} value={s.value ?? 0} sub={s.sub} />}
          </div>
        ))}
      </div>

      {empty ? (
        <Card className="mt-6">
          <EmptyState
            title="No experiments yet"
            action={
              <Button asChild variant="primary">
                <Link href="/experiments/new">
                  <Beaker /> Create the first experiment
                </Link>
              </Button>
            }
          >
            Import or generate a dataset, then create an experiment. For a fully offline demo run{" "}
            <code className="font-mono text-fg-muted">make demo</code>, which trains a small population on deterministic synthetic data.
          </EmptyState>
        </Card>
      ) : (
        <div className="mt-6 grid gap-6 xl:grid-cols-3">
          <Card className="xl:col-span-2">
            <CardHeader
              title="Highest validation-ranked agent vs buy-and-hold"
              description={latestCompleted ? `${latestCompleted.name} · final generation` : undefined}
              actions={<PeriodBadge period="validation" />}
            />
            <CardBody>
              {experiments.isPending || board.isLoading || detail.isLoading ? (
                <Skeleton className="h-72" />
              ) : detail.isError || board.isError ? (
                <ErrorState error={detail.error ?? board.error} />
              ) : !latestCompleted ? (
                <EmptyState title="No completed experiment yet">
                  The leaderboard appears here once an experiment finishes its last generation.
                </EmptyState>
              ) : top ? (
                <>
                  <div className="mb-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <Stat label="Agent" value={<Link className="hover:text-accent" href={`/agents/${top.agent_id}`}>{top.agent_name}</Link>} sub={<AlgoTag algorithm={top.algorithm} />} valueClassName="text-base font-sans" />
                    <Stat label="Validation return" value={pct(top.total_return, 1, true)} valueClassName={signClass(top.total_return)} sub={`buy & hold ${pct(bh?.total_return, 1, true)}`} />
                    <Stat label="Sharpe" value={num(top.sharpe_ratio)} sub={`buy & hold ${num(bh?.sharpe_ratio)}`} />
                    <Stat label="Excess vs passive" value={pct(top.excess_return, 1, true)} valueClassName={signClass(top.excess_return)} sub={`max DD ${pct(top.max_drawdown)}`} />
                  </div>
                  <EquityChart
                    height={260}
                    series={curves.data?.series.filter((s) => s.agent_id === top.agent_id).map((s) => ({ key: s.agent_id, label: top.agent_name, color: algoColor(top.algorithm), series: s })) ?? []}
                    benchmark={curves.data?.benchmark}
                  />
                </>
              ) : (
                <EmptyState title="No eligible agent in the latest completed experiment">
                  Every agent failed the risk or evidence checks. That is a legitimate research outcome, not an error.
                </EmptyState>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Latest experiment" />
            <CardBody className="space-y-4">
              {latest ? (
                <>
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <Link href={`/experiments/${latest.id}`} className="block truncate text-sm font-medium hover:text-accent">
                        {latest.name}
                      </Link>
                      <p className="mt-0.5 text-xs text-fg-subtle">
                        {latest.symbol} · seed {latest.seed} · created {relative(latest.created_at)}
                      </p>
                    </div>
                    <StatusBadge status={latest.status} />
                  </div>
                  <Progress value={latest.progress ?? 0} />
                  <p className="text-xs text-fg-subtle">
                    {latest.current_generation !== null ? latest.current_generation + 1 : 0} of {latest.planned_generations} generations · population {latest.population_size}
                  </p>
                  <Button asChild size="sm" className="w-full">
                    <Link href={`/experiments/${latest.id}`}>
                      Open monitor <ArrowRight />
                    </Link>
                  </Button>
                </>
              ) : (
                <Skeleton className="h-28" />
              )}
              <ResearchNote>
                A tournament win on historical data is not evidence of a real edge. Compare every result with the random and
                buy-and-hold baselines, and use the locked final test only once.
              </ResearchNote>
            </CardBody>
          </Card>

          <Card className="xl:col-span-3">
            <CardHeader title="Recent experiments" />
            <Table>
              <thead>
                <tr>
                  <Th>Experiment</Th>
                  <Th>Status</Th>
                  <Th>Kind</Th>
                  <Th>Symbol</Th>
                  <Th className="text-right">Generations</Th>
                  <Th className="text-right">Population</Th>
                  <Th>Created</Th>
                </tr>
              </thead>
              <tbody>
                {experiments.data?.items.slice(0, 8).map((e) => (
                  <Tr key={e.id}>
                    <Td>
                      <Link href={`/experiments/${e.id}`} className="font-sans hover:text-accent">
                        {e.name}
                      </Link>
                      {e.test_contaminated ? <span className="ml-2 text-[11px] text-warn">test contaminated</span> : null}
                    </Td>
                    <Td><StatusBadge status={e.status} /></Td>
                    <Td className="text-fg-muted">{e.kind}</Td>
                    <Td className="font-mono">{e.symbol}</Td>
                    <Td className="text-right">{e.current_generation !== null ? e.current_generation + 1 : 0} / {e.planned_generations}</Td>
                    <Td className="text-right">{e.population_size}</Td>
                    <Td className="text-fg-muted">{dateTime(e.created_at)}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          </Card>
        </div>
      )}
    </>
  );
}
