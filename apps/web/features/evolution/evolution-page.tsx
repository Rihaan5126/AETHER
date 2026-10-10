"use client";

import Link from "next/link";

import { GenerationScatter } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote, StatusBadge, WarningList } from "@/components/domain";
import { ExperimentSelect, useExperimentParam } from "@/components/experiment-select";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState, KeyValue, Skeleton, Stat } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import { ALGORITHM_LABEL, algoColor, num, pct, shortAgentId, signClass } from "@/lib/format";
import { flattenGenome, genomeDiff } from "@/lib/genome";
import { useAgent, useAgentMetrics, useAnalytics, useExperiment, useLineage } from "@/lib/queries";

import { LineageGraph } from "./lineage-graph";

function NodePanel({ id, onSelect }: { id: string; onSelect: (id: string) => void }) {
  const agent = useAgent(id);
  const metrics = useAgentMetrics(id);
  if (agent.isError) return <ErrorState error={agent.error} />;
  if (!agent.data) return <Skeleton className="h-80" />;
  const a = agent.data;
  const latest = metrics.data?.results.at(-1);
  const m = (latest?.metrics ?? {}) as Record<string, number | null>;
  const lin = a.lineage[0];
  const diff = lin ? genomeDiff(lin.genome_before, lin.genome_after) : [];
  return (
    <div className="space-y-4 text-xs">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-sm font-medium text-fg">{a.name}</p>
          <div className="mt-1 flex items-center gap-2"><AlgoTag algorithm={a.algorithm} /><span className="text-fg-subtle">gen {a.generation} · {lin?.operation ?? a.origin}</span></div>
        </div>
        <StatusBadge status={latest?.status} />
      </div>
      <div className="grid grid-cols-3 gap-3">
        <Stat label="Score" value={num(latest?.score, 3)} />
        <Stat label="Return" value={pct(m.total_return, 1, true)} valueClassName={signClass(m.total_return)} />
        <Stat label="Sharpe" value={num(m.sharpe_ratio)} />
      </div>
      <div>
        <p className="mb-1 text-fg-subtle">Parent</p>
        {lin?.parent_id ? (
          <button type="button" className="font-mono text-accent hover:underline" onClick={() => onSelect(lin.parent_id!)}>{shortAgentId(lin.parent_id)}</button>
        ) : <span className="text-fg-muted">none ({a.origin})</span>}
      </div>
      {lin?.mutations.length ? (
        <div>
          <p className="mb-1 text-fg-subtle">Mutations</p>
          <ul className="space-y-1">{lin.mutations.map((mu) => <li key={mu} className="rounded border border-line bg-panel-2 px-2 py-1 font-mono text-fg-muted">{mu}</li>)}</ul>
        </div>
      ) : null}
      {diff.length && lin?.genome_before ? (
        <Table>
          <thead><tr><Th>Gene</Th><Th>Parent</Th><Th>Child</Th></tr></thead>
          <tbody>{diff.map((d) => <Tr key={d.key}><Td className="text-fg-muted">{d.key}</Td><Td className="text-fg-subtle">{d.before ?? "—"}</Td><Td>{d.after}</Td></Tr>)}</tbody>
        </Table>
      ) : null}
      <div>
        <p className="mb-1 text-fg-subtle">Children ({a.children.length})</p>
        <div className="flex flex-wrap gap-1">
          {a.children.map((c) => <button key={c} type="button" onClick={() => onSelect(c)}><Badge className="hover:border-accent">{shortAgentId(c)}</Badge></button>)}
          {!a.children.length ? <span className="text-fg-muted">none</span> : null}
        </div>
      </div>
      <div>
        <p className="mb-1 text-fg-subtle">Genome</p>
        <KeyValue rows={Object.entries(flattenGenome(a.genome)).slice(0, 14).map(([k, v]) => [k, v])} />
      </div>
      <WarningList warnings={latest?.warnings ?? []} />
      <Button asChild size="sm" variant="secondary" className="w-full"><Link href={`/agents/${a.id}`}>Open agent</Link></Button>
    </div>
  );
}

export function EvolutionPage() {
  const { experimentId, setExperiment, params } = useExperimentParam();
  const lineage = useLineage(experimentId);
  const experiment = useExperiment(experimentId);
  const analytics = useAnalytics(experimentId);
  const selected = params.get("agent") ?? undefined;
  const select = (id: string) => experimentId && setExperiment(experimentId, { agent: id });
  const gens = experiment.data?.generations ?? [];
  const first = gens[0];
  const last = gens.at(-1);
  const points = (analytics.data?.generations ?? []).flatMap((g) =>
    g.agents.filter((a) => a.score !== null).map((a) => ({
      generation: g.generation, score: a.score!, algorithm: a.algorithm, color: algoColor(a.algorithm),
      id: a.agent_id, benchmark: a.algorithm === "random" || a.algorithm === "buy_and_hold",
    })),
  );

  return (
    <>
      <PageHeader
        title="Evolution Lab"
        description="How the population changed: who survived, who was mutated, and whether validation scores actually improved."
        actions={<><ExperimentSelect value={experimentId} onChange={(id) => setExperiment(id, { agent: undefined })} /><PeriodBadge period="validation" /></>}
      />
      {lineage.isError ? <ErrorState error={lineage.error} /> : null}
      {experiment.data && experiment.data.kind !== "evolution" ? (
        <EmptyState title="This experiment is a single tournament">Select an evolution experiment to see lineage.</EmptyState>
      ) : null}
      <div className="mb-6 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line lg:grid-cols-4">
        <div className="bg-panel p-4"><Stat label="Generations" value={gens.length || "—"} sub={`of ${experiment.data?.planned_generations ?? "—"} planned`} /></div>
        <div className="bg-panel p-4"><Stat label="Best score, first gen" value={num(first?.best_score, 3)} sub={`random baseline ${num(first?.random_baseline_score, 3)}`} /></div>
        <div className="bg-panel p-4"><Stat label="Best score, last gen" value={num(last?.best_score, 3)} sub={`median ${num(last?.median_score, 3)}`} /></div>
        <div className="bg-panel p-4">
          <Stat
            label="Change in best"
            value={first?.best_score != null && last?.best_score != null ? num(last.best_score - first.best_score, 3, true) : "—"}
            sub="validation folds are reused every generation"
          />
        </div>
      </div>
      <div className="grid gap-6 xl:grid-cols-[1fr_22rem]">
        <div className="min-w-0 space-y-6">
          <Card>
            <CardHeader
              title="Lineage"
              description="Columns are generations, rows are ordered by rank. Solid edges: elite or mutation; dashed: crossover. Hollow nodes: benchmarks or unranked."
            />
            <CardBody>
              {lineage.isLoading ? <Skeleton className="h-80" /> : lineage.data?.nodes.length ? (
                <LineageGraph graph={lineage.data} selected={selected} onSelect={select} />
              ) : <EmptyState title="No lineage yet" />}
              <div className="mt-3 flex flex-wrap gap-3 text-[11px] text-fg-subtle">
                {Object.entries(ALGORITHM_LABEL).map(([k, v]) => (
                  <span key={k} className="flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: algoColor(k) }} />{v}</span>
                ))}
              </div>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Validation score by generation" description="Each dot is one agent; benchmarks are faint. Click a dot to inspect it." />
            <CardBody><GenerationScatter points={points} onSelect={select} /></CardBody>
          </Card>
          <Card>
            <CardHeader title="Generations" />
            <CardBody className="p-0">
              <Table>
                <thead><tr><Th>Gen</Th><Th className="text-right">Agents</Th><Th className="text-right">Best</Th><Th className="text-right">Median</Th><Th className="text-right">Random</Th><Th className="text-right">Buy & hold return</Th><Th>Best agent</Th></tr></thead>
                <tbody>
                  {gens.map((g) => (
                    <Tr key={g.generation}>
                      <Td>{g.generation}</Td>
                      <Td className="text-right">{g.n_agents}</Td>
                      <Td className="text-right font-mono">{num(g.best_score, 3)}</Td>
                      <Td className="text-right font-mono">{num(g.median_score, 3)}</Td>
                      <Td className="text-right font-mono text-fg-subtle">{num(g.random_baseline_score, 3)}</Td>
                      <Td className="text-right font-mono">{pct(g.buy_and_hold_return, 1, true)}</Td>
                      <Td>{g.best_agent_id ? <button type="button" className="font-mono text-accent hover:underline" onClick={() => select(g.best_agent_id!)}>{shortAgentId(g.best_agent_id)}</button> : "—"}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            </CardBody>
          </Card>
          <ResearchNote>
            Selection uses the same validation folds in every generation, so later generations are partially fitted to those folds. Improvement in the best validation score is expected even without real skill; only the locked final test can show whether it generalises.
          </ResearchNote>
        </div>
        <Card className="h-fit xl:sticky xl:top-4">
          <CardHeader title="Selected agent" />
          <CardBody>{selected ? <NodePanel id={selected} onSelect={select} /> : <p className="text-xs text-fg-subtle">Click a node in the lineage graph.</p>}</CardBody>
        </Card>
      </div>
    </>
  );
}
