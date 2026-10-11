"use client";

import { ArrowLeft, GitBranch, ShieldCheck } from "lucide-react";
import Link from "next/link";

import { EquityChart, FoldBars } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote, StatusBadge, WarningList } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState, KeyValue, Skeleton, Stat } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import { algoColor, int, isNum, num, pct, shortAgentId, signClass } from "@/lib/format";
import { flattenGenome, genomeDiff } from "@/lib/genome";
import { useAgent, useAgentMetrics, useEquityCurves } from "@/lib/queries";

function ScoreBreakdown({ components }: { components: Record<string, number> }) {
  const rows: [string, string, number | undefined][] = [
    ["Mean fold Sharpe", "base", components.mean_fold_sharpe],
    ["Stability penalty", "0.5 × std(fold Sharpe)", components.stability_penalty],
    ["Drawdown penalty", "1.0 × max drawdown", components.drawdown_penalty],
    ["Turnover penalty", "0.01 × annual turnover", components.turnover_penalty],
  ];
  return (
    <div className="space-y-1.5 text-xs">
      {rows.map(([label, hint, v]) => (
        <div key={label} className="flex items-center justify-between gap-3">
          <span className="text-fg-muted">
            {label} <span className="text-fg-subtle">({hint})</span>
          </span>
          <span className={`font-mono tabular ${signClass(v)}`}>{num(v, 3, true)}</span>
        </div>
      ))}
      <div className="flex items-center justify-between border-t border-line pt-1.5 font-medium">
        <span>Validation score</span>
        <span className="font-mono tabular">{num(components.score, 3)}</span>
      </div>
    </div>
  );
}

function FeatureImportance({ importance }: { importance: Record<string, number> }) {
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 10);
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);
  return (
    <div className="space-y-1.5">
      {entries.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[9rem_1fr_3.5rem] items-center gap-2 text-xs">
          <span className="truncate font-mono text-fg-muted">{k}</span>
          <div className="h-1.5 rounded-full bg-raised">
            <div className="h-full rounded-full bg-accent/70" style={{ width: `${(v / max) * 100}%` }} />
          </div>
          <span className="text-right font-mono tabular text-fg-subtle">{v.toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

export function AgentDetailPage({ id }: { id: string }) {
  const agent = useAgent(id);
  const metrics = useAgentMetrics(id);
  const latest = metrics.data?.results.at(-1);
  const curves = useEquityCurves(latest?.tournament_id, [id]);

  if (agent.isError) return <ErrorState error={agent.error} />;
  if (agent.isLoading || !agent.data) return <Skeleton className="h-96" />;
  const a = agent.data;
  const m = (latest?.metrics ?? {}) as Record<string, number | null>;
  const lineage = a.lineage[0];
  const diff = lineage ? genomeDiff(lineage.genome_before, lineage.genome_after) : [];
  const lastRun = a.training_runs.at(-1);
  const foldTraining = ((lastRun?.metrics?.folds as Record<string, unknown>[] | undefined) ?? []).filter((f) => "fold" in f);
  const importance = foldTraining.at(-1)?.feature_importance as Record<string, number> | undefined;

  return (
    <>
      <Button asChild variant="ghost" size="sm" className="mb-3 -ml-2">
        <Link href={`/agents?experiment=${a.experiment_id}`}>
          <ArrowLeft /> Agent Arena
        </Link>
      </Button>
      <PageHeader
        title={a.name}
        description={
          <span className="flex flex-wrap items-center gap-3">
            <AlgoTag algorithm={a.algorithm} />
            <span>generation {a.generation}</span>
            <span>origin: {a.origin}</span>
            <span className="font-mono text-xs text-fg-subtle">{a.id}</span>
          </span>
        }
        actions={
          <>
            <StatusBadge status={latest?.status} />
            <PeriodBadge period="validation" />
            <Button asChild size="sm">
              <Link href={`/evolution?experiment=${a.experiment_id}&agent=${a.id}`}>
                <GitBranch /> Lineage
              </Link>
            </Button>
          </>
        }
      />

      <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
        {[
          ["Rank", latest?.rank ? `#${latest.rank}` : "—", latest?.status_reason],
          ["Score", num(latest?.score, 3), "validation"],
          ["Total return", pct(m.total_return, 1, true), `buy & hold ${pct(m.benchmark_return, 1, true)}`],
          ["Sharpe", num(m.sharpe_ratio), `Sortino ${num(m.sortino_ratio)}`],
          ["Max drawdown", pct(m.max_drawdown), `Calmar ${num(m.calmar_ratio)}`],
          ["Trades", int(m.n_trades), `${int(m.n_round_trips)} round trips`],
        ].map(([label, value, sub]) => (
          <div key={label} className="bg-panel p-4">
            <Stat label={String(label)} value={value} sub={sub} />
          </div>
        ))}
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader title="Validation equity vs buy-and-hold" description="Folds are simulated independently from cash and chained; costs included." />
          <CardBody>
            <EquityChart
              series={curves.data?.series.map((s) => ({ key: s.agent_id, label: a.name, color: algoColor(a.algorithm), series: s })) ?? []}
              benchmark={curves.data?.benchmark}
            />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Why this rank" description="Documented ranking policy, not total return" />
          <CardBody className="space-y-4">
            {latest && Object.keys(latest.score_components).length ? (
              <ScoreBreakdown components={latest.score_components} />
            ) : (
              <p className="text-xs text-fg-subtle">No score: {latest?.status_reason ?? "not evaluated"}</p>
            )}
            <div>
              <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-fg-subtle">Statistical warnings</p>
              <WarningList warnings={latest?.warnings ?? []} />
            </div>
          </CardBody>
        </Card>

        <Card className="xl:col-span-2">
          <CardHeader title="Performance by validation fold" description="Each fold: train on the past (purged), simulate the next unseen window" />
          <CardBody className="space-y-4">
            <FoldBars
              rows={(latest?.fold_metrics ?? []).map((f) => ({
                name: `Fold ${f.fold + 1}`,
                agent: f.total_return,
                benchmark: f.benchmark_return,
              }))}
              keys={[
                { key: "agent", label: "Agent return", color: algoColor(a.algorithm) },
                { key: "benchmark", label: "Buy & hold", color: "var(--color-fg-subtle)" },
              ]}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Table>
              <thead>
                <tr>
                  <Th>Fold</Th><Th>Train</Th><Th>Validation</Th><Th className="text-right">Return</Th>
                  <Th className="text-right">Sharpe</Th><Th className="text-right">Max DD</Th><Th className="text-right">Trades</Th>
                </tr>
              </thead>
              <tbody>
                {(latest?.fold_metrics ?? []).map((f) => (
                  <Tr key={f.fold}>
                    <Td>{f.fold + 1}</Td>
                    <Td className="text-fg-muted">{f.train_start} → {f.train_end}</Td>
                    <Td className="text-fg-muted">{f.val_start} → {f.val_end}</Td>
                    <Td className={`text-right ${signClass(f.total_return)}`}>{pct(f.total_return, 1, true)}</Td>
                    <Td className="text-right">{num(f.sharpe_ratio)}</Td>
                    <Td className="text-right">{pct(f.max_drawdown)}</Td>
                    <Td className="text-right">{f.n_trades}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Lineage" />
          <CardBody className="space-y-3 text-xs">
            {lineage?.parent_id ? (
              <p>
                <span className="text-fg-subtle">Parent: </span>
                <Link className="font-mono text-accent hover:underline" href={`/agents/${lineage.parent_id}`}>
                  {shortAgentId(lineage.parent_id)}
                </Link>
              </p>
            ) : (
              <p className="text-fg-subtle">No parent ({a.origin}).</p>
            )}
            {lineage?.mutations.length ? (
              <ul className="space-y-1">
                {lineage.mutations.map((mu) => (
                  <li key={mu} className="rounded border border-line bg-panel-2 px-2 py-1 font-mono text-fg-muted">{mu}</li>
                ))}
              </ul>
            ) : null}
            {diff.length && lineage?.genome_before ? (
              <Table>
                <thead><tr><Th>Gene</Th><Th>Before</Th><Th>After</Th></tr></thead>
                <tbody>
                  {diff.map((d) => (
                    <Tr key={d.key}>
                      <Td className="text-fg-muted">{d.key}</Td>
                      <Td className="text-fg-subtle">{d.before ?? "—"}</Td>
                      <Td>{d.after}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            ) : null}
            <div>
              <p className="mb-1 text-fg-subtle">Children ({a.children.length})</p>
              <div className="flex flex-wrap gap-1">
                {a.children.map((c) => (
                  <Link key={c} href={`/agents/${c}`}><Badge className="hover:border-accent">{shortAgentId(c)}</Badge></Link>
                ))}
                {!a.children.length ? <span className="text-fg-subtle">none</span> : null}
              </div>
            </div>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Genome" description={`training seed ${a.seed}`} />
          <CardBody>
            <KeyValue rows={Object.entries(flattenGenome(a.genome)).map(([k, v]) => [k, v])} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Training" description="Most recent evaluation, per fold" />
          <CardBody className="space-y-4">
            {lastRun?.error ? <ErrorState error={new Error(lastRun.error)} /> : null}
            {foldTraining.length ? (
              <Table>
                <thead><tr><Th>Fold</Th><Th className="text-right">Rows</Th><Th className="text-right">Train AUC</Th><Th className="text-right">Fit</Th></tr></thead>
                <tbody>
                  {foldTraining.map((f) => (
                    <Tr key={String(f.fold)}>
                      <Td>{Number(f.fold) + 1}</Td>
                      <Td className="text-right">{int(f.n_samples as number)}</Td>
                      <Td className="text-right">{num(f.train_auc as number, 3)}</Td>
                      <Td className="text-right">{isNum(f.fit_seconds) ? `${f.fit_seconds.toFixed(2)}s` : "—"}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            ) : (
              <p className="text-xs text-fg-subtle">Rule-based agent: nothing is fitted.</p>
            )}
            {lastRun?.metrics?.cached_from ? (
              <p className="text-xs text-fg-subtle">Evaluation reused from identical genome and seed ({shortAgentId(String(lastRun.metrics.cached_from))}).</p>
            ) : null}
            <ResearchNote>In-sample AUC is reported to expose overfitting: a high training AUC with weak validation results means the model memorised noise.</ResearchNote>
          </CardBody>
        </Card>

        <Card>
          <CardHeader title="Feature importance" description="Last fold model" />
          <CardBody>
            {importance && Object.keys(importance).length ? (
              <FeatureImportance importance={importance} />
            ) : (
              <EmptyState title="Not available for this algorithm" />
            )}
          </CardBody>
        </Card>

        <Card className="xl:col-span-3">
          <CardHeader title="Model artifact and final test" actions={<PeriodBadge period="test" />} />
          <CardBody className="grid gap-6 lg:grid-cols-2">
            <KeyValue
              rows={[
                ["Artifact", a.artifact_path ?? "not retained (only final-generation models are stored)"],
                ["SHA-256", a.artifact_sha256 ? `${a.artifact_sha256.slice(0, 24)}…` : "—"],
                ["Integrity", a.artifact_path ? <span className="inline-flex items-center gap-1 text-pos"><ShieldCheck className="size-3.5" /> verified on every load</span> : "—"],
              ]}
            />
            <div>
              {metrics.data?.final_test.length ? (
                <Table>
                  <thead><tr><Th>Run</Th><Th className="text-right">Test return</Th><Th className="text-right">Sharpe</Th><Th className="text-right">Buy & hold</Th></tr></thead>
                  <tbody>
                    {metrics.data.final_test.map((r, i) => {
                      const tm = r.metrics as Record<string, number | null>;
                      const bm = r.benchmark_metrics as Record<string, number | null> | null;
                      return (
                        <Tr key={i}>
                          <Td>#{String(r.run_number)}{r.acknowledged_contamination ? <Badge tone="warn" className="ml-2">contaminated</Badge> : null}</Td>
                          <Td className={`text-right ${signClass(tm.total_return)}`}>{pct(tm.total_return, 1, true)}</Td>
                          <Td className="text-right">{num(tm.sharpe_ratio)}</Td>
                          <Td className="text-right">{pct(bm?.total_return, 1, true)}</Td>
                        </Tr>
                      );
                    })}
                  </tbody>
                </Table>
              ) : (
                <p className="text-xs text-fg-subtle">This agent has not been evaluated on the locked final test period.</p>
              )}
            </div>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
