"use client";

import Link from "next/link";

import { ExperimentSelect, useExperimentParam } from "@/components/experiment-select";
import { AlgoTag, PageHeader, PeriodBadge, StatusBadge, WarningBadges } from "@/components/domain";
import { Card } from "@/components/ui/card";
import { Select } from "@/components/ui/input";
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/misc";
import type { AgentSummary } from "@/lib/api/schemas";
import { algoColor, algoLabel, num, pct, signClass } from "@/lib/format";
import { genomeSummary } from "@/lib/genome";
import { useAgents, useExperiment } from "@/lib/queries";

function AgentCard({ agent }: { agent: AgentSummary }) {
  return (
    <Link
      href={`/agents/${agent.id}`}
      className="group flex flex-col rounded-lg border border-line bg-panel transition-colors hover:border-line-strong hover:bg-panel-2"
    >
      <div className="h-0.5 rounded-t-lg" style={{ background: algoColor(agent.algorithm), opacity: agent.is_benchmark ? 0.35 : 0.8 }} />
      <div className="flex items-start justify-between gap-2 px-4 pt-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-fg group-hover:text-accent">{agent.name}</p>
          <div className="mt-1 flex items-center gap-2">
            <AlgoTag algorithm={agent.algorithm} />
            <span className="text-[11px] text-fg-subtle">· {agent.origin}</span>
          </div>
        </div>
        <div className="flex flex-col items-end gap-1">
          <StatusBadge status={agent.status} />
          {agent.rank ? <span className="font-mono text-[11px] text-fg-subtle">rank #{agent.rank}</span> : null}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 px-4 py-3">
        <div>
          <p className="text-[10px] uppercase tracking-wider text-fg-subtle">Score</p>
          <p className="font-mono text-sm tabular">{num(agent.score, 3)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-fg-subtle">Return</p>
          <p className={`font-mono text-sm tabular ${signClass(agent.total_return)}`}>{pct(agent.total_return, 1, true)}</p>
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-wider text-fg-subtle">Max DD</p>
          <p className="font-mono text-sm tabular">{pct(agent.max_drawdown)}</p>
        </div>
      </div>
      <dl className="mx-4 grid grid-cols-2 gap-x-3 gap-y-1 border-t border-line py-3 text-[11px]">
        {genomeSummary(agent.genome).slice(0, 6).map(([k, v]) => (
          <div key={k} className="flex min-w-0 justify-between gap-2">
            <dt className="truncate text-fg-subtle">{k}</dt>
            <dd className="truncate font-mono text-fg-muted">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-auto border-t border-line px-4 py-2.5">
        <WarningBadges warnings={agent.warnings} max={2} />
      </div>
    </Link>
  );
}

export function ArenaPage() {
  const { experimentId, setExperiment, params } = useExperimentParam();
  const exp = useExperiment(experimentId);
  const generationParam = params.get("generation");
  const algorithm = params.get("algorithm") ?? undefined;
  const generation =
    generationParam !== null ? Number(generationParam) : (exp.data?.current_generation ?? undefined);
  const agents = useAgents({ experiment_id: experimentId, generation, algorithm });
  const gens = exp.data?.generations.map((g) => g.generation) ?? [];

  return (
    <>
      <PageHeader
        title="Agent Arena"
        description="Every agent in the selected generation. Scores and returns are aggregated over walk-forward validation folds; benchmarks are shown for reference and never reproduce."
        actions={<PeriodBadge period="validation" />}
      />
      <div className="mb-5 flex flex-wrap items-center gap-2">
        <ExperimentSelect value={experimentId} onChange={(id) => setExperiment(id, { generation: undefined })} />
        <Select
          aria-label="Generation"
          className="w-36"
          value={generation ?? ""}
          onChange={(e) => experimentId && setExperiment(experimentId, { generation: e.target.value })}
        >
          {gens.map((g) => (
            <option key={g} value={g}>Generation {g}</option>
          ))}
        </Select>
        <Select
          aria-label="Algorithm"
          className="w-44"
          value={algorithm ?? ""}
          onChange={(e) => experimentId && setExperiment(experimentId, { algorithm: e.target.value || undefined })}
        >
          <option value="">All algorithms</option>
          {["momentum", "random_forest", "xgboost", "mlp", "random", "buy_and_hold"].map((a) => (
            <option key={a} value={a}>{algoLabel(a)}</option>
          ))}
        </Select>
      </div>
      {agents.isError ? (
        <ErrorState error={agents.error} />
      ) : agents.isLoading || !experimentId ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-60" />)}
        </div>
      ) : agents.data?.items.length ? (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {[...agents.data.items]
            .sort((a, b) => (a.rank ?? 999) - (b.rank ?? 999))
            .map((a) => <AgentCard key={a.id} agent={a} />)}
        </div>
      ) : (
        <Card>
          <EmptyState title="No agents match this selection">
            Agents appear once a generation has been evaluated. Start an experiment from the Experiments page.
          </EmptyState>
        </Card>
      )}
    </>
  );
}
