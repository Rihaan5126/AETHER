"use client";

import { AlertTriangle } from "lucide-react";

import { GenerationProgress, ValidationVsTest } from "@/components/charts/charts";
import { AlgoTag, PageHeader, PeriodBadge, ResearchNote } from "@/components/domain";
import { ExperimentSelect, useExperimentParam } from "@/components/experiment-select";
import { Badge } from "@/components/ui/badge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { EmptyState, ErrorState, Skeleton, Stat } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import type { Analytics } from "@/lib/api/schemas";
import { algoColor, algoLabel, int, num, pct, shortAgentId, signClass } from "@/lib/format";
import { useAnalytics } from "@/lib/queries";

function heat(v: number | null): string {
  if (v === null) return "transparent";
  const a = Math.min(1, Math.abs(v) / 2) * 0.55;
  return v >= 0 ? `rgba(92,196,151,${a})` : `rgba(236,123,130,${a})`;
}

function Interval({ i, format }: { i: NonNullable<Analytics["uncertainty"]>[number]["sharpe"]; format: (v: number | null) => string }) {
  if (i.estimate === null || i.low === null || i.high === null) return <span className="text-fg-subtle">undefined</span>;
  const straddles = i.low < 0 && i.high > 0;
  return (
    <span className="font-mono">
      {format(i.estimate)} <span className={straddles ? "text-warn" : "text-fg-subtle"}>[{format(i.low)}, {format(i.high)}]</span>
    </span>
  );
}

function UncertaintyCard({ rows }: { rows: NonNullable<Analytics["uncertainty"]> }) {
  return (
    <Card>
      <CardHeader
        title="Uncertainty (final generation)"
        description="95% stationary block-bootstrap intervals (mean block 20 sessions) on stitched validation returns. Agent and buy-and-hold days are resampled together. An interval that spans zero is shown in amber: the data cannot tell the result apart from no effect."
        actions={<PeriodBadge period="validation" />}
      />
      <CardBody className="p-0">
        <Table>
          <thead>
            <tr>
              <Th>Agent</Th>
              <Th className="text-right">Train Sharpe</Th>
              <Th className="text-right">Validation Sharpe [95% CI]</Th>
              <Th className="text-right">P(Sharpe &gt; 0)</Th>
              <Th className="text-right">Excess vs B&amp;H, annual [95% CI]</Th>
              <Th className="text-right">P(excess &gt; 0)</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((u) => (
              <Tr key={u.agent_id}>
                <Td><span className="flex items-center gap-2"><AlgoTag algorithm={u.algorithm} /><span className="font-mono text-fg-subtle">{shortAgentId(u.agent_id)}</span></span></Td>
                <Td className="text-right font-mono text-fg-subtle">{u.algorithm === "buy_and_hold" ? "n/a" : num(u.train_sharpe ?? null)}</Td>
                <Td className="text-right"><Interval i={u.sharpe} format={(v) => num(v)} /></Td>
                <Td className="text-right font-mono">{pct(u.sharpe.prob_positive, 0)}</Td>
                <Td className="text-right">{u.algorithm === "buy_and_hold" ? <span className="text-fg-subtle">benchmark</span> : <Interval i={u.excess_return} format={(v) => pct(v, 1, true)} />}</Td>
                <Td className="text-right font-mono">{u.algorithm === "buy_and_hold" ? "" : pct(u.excess_return.prob_positive, 0)}</Td>
              </Tr>
            ))}
          </tbody>
        </Table>
      </CardBody>
    </Card>
  );
}

function FoldHeatmap({ rows }: { rows: NonNullable<Analytics["fold_performance"]> }) {
  const nFolds = Math.max(0, ...rows.map((r) => r.folds.length));
  return (
    <Table>
      <thead>
        <tr><Th>Agent</Th>{Array.from({ length: nFolds }, (_, i) => <Th key={i} className="text-center">Fold {i + 1}</Th>)}<Th className="text-right">Spread</Th></tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const vals = r.folds.map((f) => f.sharpe_ratio).filter((v): v is number => v !== null);
          const spread = vals.length ? Math.max(...vals) - Math.min(...vals) : null;
          return (
            <Tr key={r.agent_id}>
              <Td><span className="flex items-center gap-2"><AlgoTag algorithm={r.algorithm} /><span className="font-mono text-fg-subtle">{r.is_benchmark ? "benchmark" : shortAgentId(r.agent_id)}</span></span></Td>
              {r.folds.map((f, i) => (
                <Td key={i} className="text-center font-mono" style={{ background: heat(f.sharpe_ratio) }} title={`${f.val_start} → ${f.val_end}, return ${pct(f.total_return, 1, true)}`}>
                  {num(f.sharpe_ratio)}
                </Td>
              ))}
              <Td className="text-right font-mono text-fg-muted">{num(spread)}</Td>
            </Tr>
          );
        })}
      </tbody>
    </Table>
  );
}

const REGIMES = [
  ["uptrend", "Uptrend"],
  ["downtrend", "Downtrend"],
  ["high_volatility", "High vol"],
  ["low_volatility", "Low vol"],
] as const;

export function AnalyticsPage() {
  const { experimentId, setExperiment } = useExperimentParam();
  const q = useAnalytics(experimentId);
  const d = q.data;
  const mc = d?.multiple_comparisons;
  const gap = d?.generalisation ?? [];

  return (
    <>
      <PageHeader
        title="Research Analytics"
        description="Is any of this real? Stability across folds, sensitivity to parameters, selection bias, and validation-to-test decay."
        actions={<ExperimentSelect value={experimentId} onChange={(id) => setExperiment(id)} />}
      />
      {q.isError ? <ErrorState error={q.error} /> : null}
      {q.isLoading ? <Skeleton className="h-96" /> : null}
      {d && !d.available ? <EmptyState title="No results for this experiment yet" /> : null}
      {d?.available ? (
        <div className="space-y-6">
          {mc ? (
            <Card>
              <CardHeader title="Selection bias" description="How good would the best agent look by luck alone?" actions={<PeriodBadge period="validation" />} />
              <CardBody className="space-y-4">
                <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
                  <Stat label="Agents evaluated" value={int(mc.agents_evaluated)} />
                  <Stat label="Unique training runs" value={int(mc.unique_training_runs)} />
                  <Stat label="Validation years" value={num(mc.validation_years, 1)} />
                  <Stat label="Best observed Sharpe" value={num(mc.best_observed_sharpe)} />
                  <Stat
                    label="Expected max Sharpe if no skill"
                    value={num(mc.expected_max_sharpe_under_null)}
                    valueClassName={mc.best_observed_sharpe !== null && mc.expected_max_sharpe_under_null !== null && mc.best_observed_sharpe <= mc.expected_max_sharpe_under_null ? "text-warn" : undefined}
                  />
                </div>
                {mc.best_observed_sharpe !== null && mc.expected_max_sharpe_under_null !== null && mc.best_observed_sharpe <= mc.expected_max_sharpe_under_null ? (
                  <p className="flex items-center gap-2 text-xs text-warn"><AlertTriangle className="size-3.5" /> The best validation Sharpe is within what the luckiest of {mc.agents_evaluated} skill-less agents would be expected to reach.</p>
                ) : null}
                <ResearchNote>{mc.explanation}</ResearchNote>
              </CardBody>
            </Card>
          ) : null}

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Score across generations" description="Validation folds are reused by selection, so improvement here is optimistic." />
              <CardBody>
                <GenerationProgress rows={(d.generations ?? []).map((g) => ({ generation: g.generation, best: g.best_score, median: g.median_score, random: g.random_score }))} />
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Validation vs final test" description="Points below the diagonal lost performance out of sample." actions={<PeriodBadge period="test" />} />
              <CardBody>
                <ValidationVsTest
                  points={gap.filter((g) => g.validation_sharpe !== null && g.test_sharpe !== null).map((g) => ({
                    id: `${g.run_number}-${g.agent_id}`, label: g.agent_id, x: g.validation_sharpe!, y: g.test_sharpe!, color: algoColor(g.algorithm),
                  }))}
                />
              </CardBody>
            </Card>
          </div>

          {gap.length ? (
            <Card>
              <CardHeader title="Generalisation gap" description="Validation (folds) against the locked test period, per final-test run" actions={<PeriodBadge period="test" />} />
              <CardBody className="p-0">
                <Table>
                  <thead><tr><Th>Run</Th><Th>Agent</Th><Th className="text-right">Train Sharpe</Th><Th className="text-right">Val Sharpe</Th><Th className="text-right">Test Sharpe</Th><Th className="text-right">Val excess</Th><Th className="text-right">Test excess</Th></tr></thead>
                  <tbody>
                    {gap.map((g) => (
                      <Tr key={`${g.run_number}-${g.agent_id}`}>
                        <Td>#{g.run_number}{g.contaminated ? <Badge tone="warn" className="ml-2">contaminated</Badge> : null}</Td>
                        <Td><span className="flex items-center gap-2"><AlgoTag algorithm={g.algorithm} /><span className="font-mono text-fg-subtle">{shortAgentId(g.agent_id)}</span></span></Td>
                        <Td className="text-right font-mono text-fg-subtle">{num(g.train_sharpe)}</Td>
                        <Td className="text-right font-mono">{num(g.validation_sharpe)}</Td>
                        <Td className="text-right font-mono">{num(g.test_sharpe)}</Td>
                        <Td className={`text-right font-mono ${signClass(g.validation_excess)}`}>{pct(g.validation_excess, 1, true)}</Td>
                        <Td className={`text-right font-mono ${signClass(g.test_excess)}`}>{pct(g.test_excess, 1, true)}</Td>
                      </Tr>
                    ))}
                  </tbody>
                </Table>
              </CardBody>
            </Card>
          ) : null}

          {d.uncertainty?.length ? <UncertaintyCard rows={d.uncertainty} /> : null}

          <Card>
            <CardHeader title="Sharpe by fold (final generation)" description="Large spread between folds suggests results depend on the period, not the strategy." />
            <CardBody className="p-0">{d.fold_performance?.length ? <FoldHeatmap rows={d.fold_performance} /> : <EmptyState title="No fold data" />}</CardBody>
          </Card>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader title="Variance within each algorithm" description="Validation score across every evaluated agent of the family" />
              <CardBody className="p-0">
                <Table>
                  <thead><tr><Th>Algorithm</Th><Th className="text-right">n</Th><Th className="text-right">Mean</Th><Th className="text-right">Std</Th><Th className="text-right">Min</Th><Th className="text-right">Max</Th></tr></thead>
                  <tbody>
                    {(d.family_variance ?? []).map((f) => (
                      <Tr key={f.algorithm}>
                        <Td><AlgoTag algorithm={f.algorithm} /></Td>
                        <Td className="text-right">{f.n}</Td>
                        <Td className="text-right font-mono">{num(f.mean, 3)}</Td>
                        <Td className="text-right font-mono">{num(f.std, 3)}</Td>
                        <Td className="text-right font-mono">{num(f.min, 3)}</Td>
                        <Td className="text-right font-mono">{num(f.max, 3)}</Td>
                      </Tr>
                    ))}
                  </tbody>
                </Table>
              </CardBody>
            </Card>
            <Card>
              <CardHeader title="Parameter sensitivity" description="Spearman rank correlation between a gene and validation score (descriptive, small samples)" />
              <CardBody className="p-0">
                {d.parameter_sensitivity?.length ? (
                  <Table>
                    <thead><tr><Th>Algorithm</Th><Th>Gene</Th><Th className="text-right">n</Th><Th className="text-right">ρ</Th><Th className="w-32" /></tr></thead>
                    <tbody>
                      {d.parameter_sensitivity.map((p) => (
                        <Tr key={`${p.algorithm}-${p.gene}`}>
                          <Td>{algoLabel(p.algorithm)}</Td>
                          <Td className="font-mono text-fg-muted">{p.gene}</Td>
                          <Td className="text-right">{p.n}</Td>
                          <Td className={`text-right font-mono ${signClass(p.spearman)}`}>{num(p.spearman, 2, true)}</Td>
                          <Td>
                            <div className="relative h-1.5 rounded-full bg-raised">
                              {p.spearman !== null ? (
                                <div className={`absolute top-0 h-full rounded-full ${p.spearman >= 0 ? "left-1/2 bg-pos/70" : "right-1/2 bg-neg/70"}`} style={{ width: `${Math.abs(p.spearman) * 50}%` }} />
                              ) : null}
                            </div>
                          </Td>
                        </Tr>
                      ))}
                    </tbody>
                  </Table>
                ) : <EmptyState title="Not enough agents per family" className="py-8" />}
              </CardBody>
            </Card>
          </div>

          <Card>
            <CardHeader title="Market regimes" description="Annualised mean daily return of final-generation agents, split by trailing 60-day buy-and-hold trend and volatility (validation period). Cells with fewer than 20 sessions are blank." />
            <CardBody className="p-0">
              {d.regimes?.length ? (
                <Table>
                  <thead><tr><Th>Agent</Th>{REGIMES.map(([k, l]) => <Th key={k} className="text-right">{l}</Th>)}</tr></thead>
                  <tbody>
                    {d.regimes.map((r) => {
                      const id = String(r.agent_id);
                      return (
                        <Tr key={id}>
                          <Td className="font-mono text-fg-muted">{id === "__benchmark__" ? "Buy & hold" : shortAgentId(id)}</Td>
                          {REGIMES.map(([k]) => {
                            const v = r[k] as number | null;
                            return <Td key={k} className={`text-right font-mono ${signClass(v)}`} title={`${String(r[`${k}_sessions`])} sessions`}>{pct(v, 1, true)}</Td>;
                          })}
                        </Tr>
                      );
                    })}
                  </tbody>
                </Table>
              ) : <EmptyState title="No regime breakdown available" className="py-8" />}
            </CardBody>
          </Card>

          {d.optimization?.length ? (
            <Card>
              <CardHeader title="Hyperparameter studies (Optuna)" description="Objective: mean fold Sharpe on validation folds; the test period is not used." />
              <CardBody className="space-y-4">
                {d.optimization.map((s) => (
                  <div key={s.id} className="space-y-2">
                    <p className="text-xs text-fg-muted">
                      <AlgoTag algorithm={s.family} /> · {s.status} · best {num(s.best_value, 3)} · {s.trials.length} trials ({s.trials.filter((t) => t.state === "pruned").length} pruned)
                    </p>
                    <div className="flex h-16 items-end gap-0.5">
                      {s.trials.map((t) => (
                        <div
                          key={t.number}
                          title={`trial ${t.number}: ${t.state} ${num(t.value, 3)}`}
                          className={`flex-1 rounded-t ${t.state === "complete" ? "bg-accent/70" : t.state === "pruned" ? "bg-fg-subtle/40" : "bg-neg/60"}`}
                          style={{ height: `${t.value === null ? 8 : Math.max(8, Math.min(100, 50 + t.value * 25))}%` }}
                        />
                      ))}
                    </div>
                  </div>
                ))}
              </CardBody>
            </Card>
          ) : null}
        </div>
      ) : null}
    </>
  );
}
