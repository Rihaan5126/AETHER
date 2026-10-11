"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Download, FlaskConical, Lock, Sparkles, Square } from "lucide-react";
import Link from "next/link";
import * as React from "react";

import { EquityChart } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote, StatusBadge } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Field, Input, Select } from "@/components/ui/input";
import { EmptyState, ErrorState, KeyValue, Progress, Skeleton, Stat } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import { api, downloadUrl } from "@/lib/api/client";
import * as S from "@/lib/api/schemas";
import { algoColor, ALGORITHM_LABEL, dateTime, duration, num, pct, relative, shortAgentId, signClass } from "@/lib/format";
import { keys, useExperiment, useFinalTestCurves } from "@/lib/queries";

type Detail = S.ExperimentDetail;

function FinalTestPanel({ exp }: { exp: Detail }) {
  const qc = useQueryClient();
  const runs = exp.final_test_runs;
  const latest = runs.at(-1);
  const [confirming, setConfirming] = React.useState(false);
  const [ack, setAck] = React.useState(false);
  const curves = useFinalTestCurves(exp.id, latest?.status === "completed" ? latest.run_number : undefined);
  const run = useMutation({
    mutationFn: () => api.post(S.JobRef, `/experiments/${exp.id}/final-test`, { acknowledge_contamination: runs.length > 0 && ack }),
    onSuccess: () => {
      setConfirming(false);
      setAck(false);
      void qc.invalidateQueries({ queryKey: keys.experiment(exp.id) });
    },
  });
  const busy = exp.jobs.some((j) => j.job_type === "final_test" && (j.status === "queued" || j.status === "running"));
  const results = (latest?.results ?? []) as { agent_id?: string; family?: string; metrics?: Record<string, number | null>; error?: string }[];
  const bench = latest?.benchmark_metrics as Record<string, number | null> | null | undefined;

  return (
    <Card className={runs.length ? "" : "border-dashed"}>
      <CardHeader
        title={<span className="flex items-center gap-2"><Lock className="size-4 text-warn" /> Locked final test</span>}
        description={
          exp.test_period
            ? `Held out: ${String(exp.test_period.start)} → ${String(exp.test_period.end)}. Never used for training, selection, evolution or tuning.`
            : "The most recent part of the data, held out from all development."
        }
        actions={<><PeriodBadge period="test" />{exp.test_contaminated ? <Badge tone="warn">contaminated</Badge> : null}</>}
      />
      <CardBody className="space-y-4">
        {exp.status !== "completed" ? (
          <p className="text-xs text-fg-subtle">Available once the experiment completes.</p>
        ) : !confirming ? (
          <div className="flex flex-wrap items-center gap-3">
            <Button variant={runs.length ? "secondary" : "primary"} disabled={busy} onClick={() => setConfirming(true)}>
              <FlaskConical /> {runs.length ? "Run the final test again" : "Run the final test"}
            </Button>
            <span className="text-xs text-fg-subtle">Evaluates the top 3 eligible final-generation agents and a random baseline, once.</span>
          </div>
        ) : (
          <div className="space-y-3 rounded-md border border-warn/40 bg-warn/5 p-4 text-sm">
            {runs.length ? (
              <>
                <p className="flex items-start gap-2 text-warn"><AlertTriangle className="mt-0.5 size-4 shrink-0" /> The test period has already been viewed ({runs.length} run{runs.length > 1 ? "s" : ""}).</p>
                <p className="text-xs text-fg-muted">
                  Running it again after looking at the results turns the test period into another validation set: any decision informed by it is no longer an unbiased out-of-sample estimate. This experiment will be permanently marked as contaminated.
                </p>
                <label className="flex items-center gap-2 text-xs">
                  <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} /> I understand the test result will no longer be unbiased
                </label>
              </>
            ) : (
              <p className="text-xs text-fg-muted">
                Run this once, when you are done developing. The result is the honest estimate of out-of-sample performance; there is no going back to adjust agents afterwards without contaminating it.
              </p>
            )}
            <div className="flex gap-2">
              <Button size="sm" disabled={run.isPending || (runs.length > 0 && !ack)} onClick={() => run.mutate()}>Confirm and run</Button>
              <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
            </div>
          </div>
        )}
        {run.isError ? <ErrorState error={run.error} /> : null}
        {latest?.status === "failed" ? <ErrorState error={new Error(results[0]?.error ?? "final test failed")} /> : null}
        {latest?.status === "completed" ? (
          <div className="grid gap-6 xl:grid-cols-2">
            <Table>
              <thead><tr><Th>Agent</Th><Th className="text-right">Return</Th><Th className="text-right">Sharpe</Th><Th className="text-right">Max DD</Th><Th className="text-right">vs B&H</Th></tr></thead>
              <tbody>
                {results.map((r) => (
                  <Tr key={r.agent_id}>
                    <Td><Link href={`/agents/${r.agent_id}`} className="flex items-center gap-2 hover:text-accent"><AlgoTag algorithm={r.family ?? ""} /><span className="font-mono text-fg-subtle">{shortAgentId(r.agent_id ?? "")}</span></Link></Td>
                    <Td className={`text-right font-mono ${signClass(r.metrics?.total_return)}`}>{pct(r.metrics?.total_return, 1, true)}</Td>
                    <Td className="text-right font-mono">{num(r.metrics?.sharpe_ratio)}</Td>
                    <Td className="text-right font-mono">{pct(r.metrics?.max_drawdown)}</Td>
                    <Td className={`text-right font-mono ${signClass(r.metrics?.excess_return)}`}>{pct(r.metrics?.excess_return, 1, true)}</Td>
                  </Tr>
                ))}
                <Tr className="bg-panel-2/60">
                  <Td className="text-fg-muted">Buy & hold</Td>
                  <Td className={`text-right font-mono ${signClass(bench?.total_return)}`}>{pct(bench?.total_return, 1, true)}</Td>
                  <Td className="text-right font-mono">{num(bench?.sharpe_ratio)}</Td>
                  <Td className="text-right font-mono">{pct(bench?.max_drawdown)}</Td>
                  <Td />
                </Tr>
              </tbody>
            </Table>
            <EquityChart
              height={260}
              series={(curves.data?.series ?? []).map((s) => {
                const r = results.find((x) => x.agent_id === s.agent_id);
                return { key: s.agent_id, label: `${ALGORITHM_LABEL[r?.family ?? ""] ?? ""} ${shortAgentId(s.agent_id)}`, color: algoColor(r?.family ?? ""), series: s };
              })}
              benchmark={curves.data?.benchmark}
            />
          </div>
        ) : null}
      </CardBody>
    </Card>
  );
}

function OptimizePanel({ exp }: { exp: Detail }) {
  const qc = useQueryClient();
  const [family, setFamily] = React.useState("xgboost");
  const [trials, setTrials] = React.useState(10);
  const run = useMutation({
    mutationFn: () => api.post(S.JobRef, `/experiments/${exp.id}/optimize`, { family, n_trials: trials }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.experiment(exp.id) }),
  });
  const studies = exp.optimization_studies as { id: string; family: string; status: string; best_value: number | null; n_trials?: number }[];
  return (
    <Card>
      <CardHeader title="Hyperparameter search (Optuna)" description="TPE sampler with median pruning; objective is mean validation-fold Sharpe." actions={<PeriodBadge period="validation" />} />
      <CardBody className="space-y-4">
        <form className="flex flex-wrap items-end gap-3" onSubmit={(e) => { e.preventDefault(); run.mutate(); }}>
          <Field label="Family" htmlFor="opt-family">
            <Select id="opt-family" value={family} onChange={(e) => setFamily(e.target.value)} className="w-40">
              {["momentum", "random_forest", "xgboost", "mlp"].map((f) => <option key={f} value={f}>{ALGORITHM_LABEL[f]}</option>)}
            </Select>
          </Field>
          <Field label="Trials" htmlFor="opt-trials">
            <Input id="opt-trials" type="number" min={1} max={100} value={trials} onChange={(e) => setTrials(Number(e.target.value))} className="w-24" />
          </Field>
          <Button type="submit" variant="secondary" disabled={exp.status !== "completed" || run.isPending}><Sparkles /> Start study</Button>
        </form>
        {run.isError ? <ErrorState error={run.error} /> : null}
        {studies.length ? (
          <Table>
            <thead><tr><Th>Study</Th><Th>Family</Th><Th>Status</Th><Th className="text-right">Best objective</Th></tr></thead>
            <tbody>
              {studies.map((s) => (
                <Tr key={s.id}>
                  <Td className="font-mono text-fg-subtle">{s.id}</Td>
                  <Td><AlgoTag algorithm={s.family} /></Td>
                  <Td><StatusBadge status={s.status} /></Td>
                  <Td className="text-right font-mono">{num(s.best_value, 3)}</Td>
                </Tr>
              ))}
            </tbody>
          </Table>
        ) : null}
        <p className="text-xs text-fg-subtle">Trial details are in <Link className="text-accent hover:underline" href={`/analytics?experiment=${exp.id}`}>Research Analytics</Link>.</p>
      </CardBody>
    </Card>
  );
}

export function ExperimentMonitor({ id }: { id: string }) {
  const qc = useQueryClient();
  const q = useExperiment(id);
  const cancel = useMutation({
    mutationFn: () => api.post(S.Experiment, `/experiments/${id}/cancel`),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.experiment(id) }),
  });
  const start = useMutation({
    mutationFn: () => api.post(S.JobRef, `/experiments/${id}/start`),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.experiment(id) }),
  });
  if (q.isError) return <ErrorState error={q.error} />;
  if (!q.data) return <Skeleton className="h-96" />;
  const exp = q.data;
  const runJob = exp.jobs.find((j) => j.job_type === "run_experiment");
  const active = exp.status === "queued" || exp.status === "running";
  const config = exp.config as Record<string, Record<string, unknown> | unknown>;
  const evo = (config.evolution ?? {}) as Record<string, unknown>;
  const ev = (config.evaluation ?? {}) as Record<string, Record<string, unknown>>;
  const manifest = (exp.manifest ?? {}) as Record<string, unknown>;

  return (
    <>
      <PageHeader
        title={exp.name}
        description={<span className="font-mono text-xs">{exp.id} · {exp.kind} · {exp.symbol} · seed {exp.seed}</span>}
        actions={
          <>
            <StatusBadge status={exp.status} />
            {exp.status === "draft" ? <Button size="sm" onClick={() => start.mutate()} disabled={start.isPending}>Start</Button> : null}
            {active ? <Button size="sm" variant="danger" onClick={() => cancel.mutate()} disabled={cancel.isPending || runJob?.cancel_requested}><Square /> {runJob?.cancel_requested ? "Cancelling…" : "Cancel"}</Button> : null}
            {exp.status === "completed" ? <Button asChild size="sm" variant="secondary"><Link href={`/tournaments?experiment=${exp.id}`}>Leaderboard</Link></Button> : null}
          </>
        }
      />
      {cancel.isError ? <ErrorState error={cancel.error} className="mb-4" /> : null}
      {start.isError ? <ErrorState error={start.error} className="mb-4" /> : null}
      {exp.error_message ? <ErrorState error={new Error(exp.error_message)} className="mb-4" /> : null}

      <Card className="mb-6">
        <CardBody className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span className="text-fg-muted">{runJob?.message ?? exp.status}</span>
            <span className="font-mono text-xs text-fg-subtle">
              {exp.current_generation !== null ? `${exp.current_generation + 1} of ${exp.planned_generations} generations done` : `${exp.planned_generations} generation(s) planned`}
            </span>
          </div>
          <Progress value={exp.status === "completed" ? 1 : (exp.progress ?? 0)} />
          <div className="grid grid-cols-2 gap-4 pt-2 md:grid-cols-4">
            <Stat label="Started" value={<span className="text-sm">{relative(exp.started_at)}</span>} sub={dateTime(exp.started_at)} />
            <Stat label="Wall time" value={<span className="text-sm">{duration(exp.wall_seconds)}</span>} />
            <Stat label="Population" value={<span className="text-sm">{exp.population_size}</span>} sub={`${String(evo.elite_count ?? "—")} elites · ${String(evo.immigrant_count ?? "—")} immigrants`} />
            <Stat label="Process peak memory" value={<span className="text-sm">{exp.resource_usage?.max_rss_mb ? `${Number(exp.resource_usage.max_rss_mb).toFixed(0)} MB` : "—"}</span>} sub={exp.resource_usage?.cpu_user_seconds ? `${Number(exp.resource_usage.cpu_user_seconds).toFixed(0)}s CPU (worker process)` : undefined} />
          </div>
        </CardBody>
      </Card>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardHeader title="Generations" description="Validation results as each generation completes" actions={<PeriodBadge period="validation" />} />
          <CardBody className="p-0">
            {exp.generations.length ? (
              <Table>
                <thead><tr><Th>Gen</Th><Th className="text-right">Agents</Th><Th className="text-right">Best score</Th><Th className="text-right">Median</Th><Th className="text-right">Random</Th><Th className="text-right">Buy & hold</Th><Th>Best agent</Th></tr></thead>
                <tbody>
                  {exp.generations.map((g) => (
                    <Tr key={g.generation}>
                      <Td><Link className="hover:text-accent" href={`/tournaments?experiment=${exp.id}&generation=${g.generation}`}>{g.generation}</Link></Td>
                      <Td className="text-right">{g.n_agents}</Td>
                      <Td className="text-right font-mono">{num(g.best_score, 3)}</Td>
                      <Td className="text-right font-mono">{num(g.median_score, 3)}</Td>
                      <Td className="text-right font-mono text-fg-subtle">{num(g.random_baseline_score, 3)}</Td>
                      <Td className={`text-right font-mono ${signClass(g.buy_and_hold_return)}`}>{pct(g.buy_and_hold_return, 1, true)}</Td>
                      <Td>{g.best_agent_id ? <Link className="font-mono text-accent hover:underline" href={`/agents/${g.best_agent_id}`}>{shortAgentId(g.best_agent_id)}</Link> : "—"}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            ) : <EmptyState title={active ? "Waiting for the first generation…" : "No generations"} className="py-10" />}
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Jobs" />
          <CardBody className="space-y-3">
            {exp.jobs.map((j) => (
              <div key={j.id} className="space-y-1 rounded-md border border-line p-2.5 text-xs">
                <div className="flex items-center justify-between"><span className="font-medium">{j.job_type.replace("_", " ")}</span><StatusBadge status={j.status} /></div>
                {j.status === "running" ? <Progress value={j.progress} /> : null}
                <p className="text-fg-subtle">{j.error_message ?? j.message} · {relative(j.created_at)}</p>
              </div>
            ))}
          </CardBody>
        </Card>
      </div>

      <div className="mt-6 space-y-6">
        <FinalTestPanel exp={exp} />
        <OptimizePanel exp={exp} />
        <div className="grid gap-6 xl:grid-cols-2">
          <Card>
            <CardHeader title="Periods" />
            <CardBody>
              <KeyValue rows={[
                ["Development", exp.development_period ? `${String(exp.development_period.start)} → ${String(exp.development_period.end)}` : "—"],
                ["Final test", exp.test_period ? `${String(exp.test_period.start)} → ${String(exp.test_period.end)}` : "—"],
                ["Folds", String(ev.walk_forward?.n_folds ?? "—")],
                ["Embargo", `${String(ev.walk_forward?.embargo_bars ?? "—")} bars`],
                ["Costs", `${String(ev.execution?.fee_bps ?? "—")} bps fee + ${String(ev.execution?.slippage_bps ?? "—")} bps slippage`],
                ["Max position", String(ev.risk?.max_position ?? "—")],
              ]} />
            </CardBody>
          </Card>
          <Card>
            <CardHeader
              title="Reproducibility manifest"
              actions={
                <Button asChild size="sm">
                  <a href={downloadUrl(`/experiments/${exp.id}/manifest`)} download={`aether-${exp.id}.json`}>
                    <Download /> Download JSON
                  </a>
                </Button>
              }
            />
            <CardBody>
              <KeyValue rows={[
                ["Dataset hash", String(manifest.dataset_hash ?? (config.dataset_hash as string) ?? "—")],
                ["Manifest version", String(manifest.manifest_version ?? "—")],
                ["Code version", exp.code_version ?? "—"],
                ["Platform", String(manifest.platform ?? "—")],
                ["Dependencies", Object.entries((manifest.dependencies ?? {}) as Record<string, string>).map(([k, v]) => `${k} ${v}`).join(", ") || "—"],
                ["Seed", String(exp.seed)],
              ]} />
              <details className="mt-3 text-xs">
                <summary className="cursor-pointer text-fg-subtle hover:text-fg">Full config</summary>
                <pre className="scroll-thin mt-2 max-h-80 overflow-auto rounded-md bg-panel-2 p-3 font-mono text-[11px] text-fg-muted">{JSON.stringify(exp.config, null, 2)}</pre>
              </details>
            </CardBody>
          </Card>
        </div>
        <ResearchNote>Completed experiment records are immutable. To change anything, create a new experiment; the seed and dataset hash make the original reproducible.</ResearchNote>
      </div>
    </>
  );
}
