"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { Play } from "lucide-react";
import { useRouter } from "next/navigation";
import * as React from "react";

import { PageHeader, ResearchNote } from "@/components/domain";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Field, Input, Select } from "@/components/ui/input";
import { ErrorState, KeyValue, Spinner } from "@/components/ui/misc";
import { api } from "@/lib/api/client";
import * as S from "@/lib/api/schemas";
import { ALGORITHM_LABEL, int } from "@/lib/format";
import { useDatasets } from "@/lib/queries";

export const FAMILIES = ["momentum", "random_forest", "xgboost", "mlp"] as const;

export type BuilderForm = {
  name: string;
  dataset_id: string;
  symbol: string;
  families: string[];
  population_size: number;
  generations: number;
  elite_count: number;
  immigrant_count: number;
  initial_capital: number;
  fee_bps: number;
  slippage_bps: number;
  max_position: number;
  drawdown_circuit_breaker: number | null;
  n_folds: number;
  test_fraction: number;
  seed: number;
  max_fit_seconds: number;
};

export const DEFAULT_FORM: BuilderForm = {
  name: "Evolution run",
  dataset_id: "",
  symbol: "",
  families: [...FAMILIES],
  population_size: 8,
  generations: 3,
  elite_count: 2,
  immigrant_count: 1,
  initial_capital: 10000,
  fee_bps: 1,
  slippage_bps: 5,
  max_position: 1,
  drawdown_circuit_breaker: null,
  n_folds: 3,
  test_fraction: 0.2,
  seed: 42,
  max_fit_seconds: 120,
};

/** Client-side mirror of the API's request validation, for immediate feedback. */
export function validateForm(f: BuilderForm): Partial<Record<keyof BuilderForm, string>> {
  const e: Partial<Record<keyof BuilderForm, string>> = {};
  if (!f.name.trim()) e.name = "Name is required";
  if (!f.dataset_id) e.dataset_id = "Choose a dataset";
  if (!f.symbol) e.symbol = "Choose a symbol";
  if (!f.families.length) e.families = "Select at least one algorithm";
  if (!(f.population_size >= 2 && f.population_size <= 64)) e.population_size = "2 to 64";
  if (!(f.generations >= 1 && f.generations <= 20)) e.generations = "1 to 20";
  if (f.elite_count < 0 || f.elite_count >= f.population_size) e.elite_count = "Must be smaller than the population";
  if (f.immigrant_count < 0 || f.elite_count + f.immigrant_count > f.population_size) e.immigrant_count = "Elites + immigrants cannot exceed the population";
  if (!(f.initial_capital > 0)) e.initial_capital = "Must be positive";
  if (f.fee_bps < 0 || f.fee_bps > 500) e.fee_bps = "0 to 500 bps";
  if (f.slippage_bps < 0 || f.slippage_bps > 500) e.slippage_bps = "0 to 500 bps";
  if (!(f.max_position > 0 && f.max_position <= 1)) e.max_position = "Between 0 and 1 (no leverage)";
  if (f.drawdown_circuit_breaker !== null && !(f.drawdown_circuit_breaker > 0 && f.drawdown_circuit_breaker < 1)) e.drawdown_circuit_breaker = "Between 0 and 1";
  if (!(f.n_folds >= 1 && f.n_folds <= 8)) e.n_folds = "1 to 8";
  if (!(f.test_fraction > 0.05 && f.test_fraction < 0.5)) e.test_fraction = "Between 0.05 and 0.5";
  if (!(Number.isInteger(f.seed) && f.seed >= 0)) e.seed = "Non-negative integer";
  if (!(f.max_fit_seconds > 0 && f.max_fit_seconds <= 3600)) e.max_fit_seconds = "1 to 3600 seconds";
  return e;
}

/** crypto.randomUUID needs a secure context; fall back for plain-http LAN access. */
function newKey(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function useDebounced<T>(value: T, ms = 400): T {
  const [v, setV] = React.useState(value);
  React.useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

function NumberField({
  id, label, value, onChange, error, hint, step = 1, min, max,
}: {
  id: string; label: string; value: number | null; onChange: (v: number | null) => void; error?: string; hint?: string; step?: number; min?: number; max?: number;
}) {
  return (
    <Field label={label} htmlFor={id} error={error} hint={hint}>
      <Input id={id} type="number" inputMode="decimal" step={step} min={min} max={max} value={value ?? ""} aria-invalid={!!error}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))} />
    </Field>
  );
}

export function ExperimentBuilder() {
  const router = useRouter();
  const datasets = useDatasets();
  const [draft, setForm] = React.useState<BuilderForm>(DEFAULT_FORM);
  const [touched, setTouched] = React.useState(false);
  // One key per form instance: retrying a failed submit cannot create duplicates.
  const [idempotencyKey] = React.useState(newKey);
  // Until the user picks one, default to the newest dataset and its first symbol.
  const dataset = datasets.data?.items.find((d) => d.id === draft.dataset_id) ?? datasets.data?.items[0];
  const form: BuilderForm = {
    ...draft,
    dataset_id: dataset?.id ?? "",
    symbol: draft.symbol && dataset?.symbols.includes(draft.symbol) ? draft.symbol : (dataset?.symbols[0] ?? ""),
  };

  const errors = validateForm(form);
  const valid = Object.keys(errors).length === 0;
  const set = <K extends keyof BuilderForm>(k: K) => (v: BuilderForm[K]) => setForm((f) => ({ ...f, [k]: v }));
  const num = <K extends keyof BuilderForm>(k: K) => (v: number | null) => set(k)((v ?? (k === "drawdown_circuit_breaker" ? null : NaN)) as BuilderForm[K]);
  const shown = (k: keyof BuilderForm) => (touched ? errors[k] : undefined);

  const debounced = useDebounced(form);
  const estimate = useQuery({
    queryKey: ["estimate", debounced],
    queryFn: () => api.post(S.Workload, "/experiments/estimate", debounced),
    enabled: Object.keys(validateForm(debounced)).length === 0,
    retry: false,
  });

  const create = useMutation({
    mutationFn: async () => {
      const exp = await api.post(S.Experiment, "/experiments", form, { "Idempotency-Key": idempotencyKey });
      await api.post(S.JobRef, `/experiments/${exp.id}/start`);
      return exp;
    },
    onSuccess: (exp) => router.push(`/experiments/${exp.id}`),
  });

  return (
    <>
      <PageHeader title="New experiment" description="Configure an evolutionary search. Everything here is recorded in the experiment manifest." />
      <form
        className="grid gap-6 xl:grid-cols-[1fr_22rem]"
        noValidate
        onSubmit={(e) => {
          e.preventDefault();
          setTouched(true);
          if (valid) create.mutate();
        }}
      >
        <div className="space-y-6">
          <Card>
            <CardHeader title="Data" />
            <CardBody className="grid gap-4 md:grid-cols-3">
              <Field label="Name" htmlFor="name" error={shown("name")}>
                <Input id="name" value={form.name} maxLength={120} onChange={(e) => set("name")(e.target.value)} />
              </Field>
              <Field label="Dataset" htmlFor="dataset" error={shown("dataset_id")}>
                <Select id="dataset" value={form.dataset_id} onChange={(e) => {
                  const d = datasets.data?.items.find((x) => x.id === e.target.value);
                  setForm((f) => ({ ...f, dataset_id: e.target.value, symbol: d?.symbols[0] ?? "" }));
                }}>
                  {!datasets.data?.items.length ? <option value="">No datasets: import one first</option> : null}
                  {datasets.data?.items.map((d) => <option key={d.id} value={d.id}>{d.name} ({d.start_date.slice(0, 4)}–{d.end_date.slice(0, 4)})</option>)}
                </Select>
              </Field>
              <Field label="Symbol" htmlFor="symbol" error={shown("symbol")}>
                <Select id="symbol" value={form.symbol} onChange={(e) => set("symbol")(e.target.value)}>
                  {dataset?.symbols.map((s) => <option key={s}>{s}</option>)}
                </Select>
              </Field>
              <NumberField id="test_fraction" label="Locked test fraction" step={0.05} value={form.test_fraction} onChange={num("test_fraction")} error={shown("test_fraction")}
                hint="Most recent share of the data, held out until you run the final test" />
              <NumberField id="n_folds" label="Walk-forward folds" value={form.n_folds} onChange={num("n_folds")} error={shown("n_folds")} hint="Purged and embargoed" />
              <NumberField id="seed" label="Seed" value={form.seed} onChange={num("seed")} error={shown("seed")} hint="Same seed + config + data = same result" />
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Population" description="Random and buy-and-hold benchmarks are always included and never selected." />
            <CardBody className="space-y-4">
              <fieldset>
                <legend className="mb-2 text-xs font-medium text-fg-muted">Algorithm families</legend>
                <div className="flex flex-wrap gap-2">
                  {FAMILIES.map((fam) => {
                    const on = form.families.includes(fam);
                    return (
                      <label key={fam} className={`flex cursor-pointer items-center gap-2 rounded-md border px-3 py-1.5 text-sm ${on ? "border-accent/50 bg-accent/10 text-fg" : "border-line text-fg-muted"}`}>
                        <input type="checkbox" className="sr-only" checked={on}
                          onChange={() => set("families")(on ? form.families.filter((x) => x !== fam) : [...form.families, fam])} />
                        {ALGORITHM_LABEL[fam]}
                      </label>
                    );
                  })}
                </div>
                {shown("families") ? <p role="alert" className="mt-1 text-xs text-neg">{errors.families}</p> : null}
              </fieldset>
              <div className="grid gap-4 md:grid-cols-4">
                <NumberField id="population_size" label="Population" value={form.population_size} onChange={num("population_size")} error={shown("population_size")} />
                <NumberField id="generations" label="Generations" value={form.generations} onChange={num("generations")} error={shown("generations")} />
                <NumberField id="elite_count" label="Elites" value={form.elite_count} onChange={num("elite_count")} error={shown("elite_count")} hint="Survive unchanged" />
                <NumberField id="immigrant_count" label="Immigrants" value={form.immigrant_count} onChange={num("immigrant_count")} error={shown("immigrant_count")} hint="Fresh random genomes" />
              </div>
            </CardBody>
          </Card>

          <Card>
            <CardHeader title="Simulation and risk" description="Orders fill at the next bar's open. Long-only, no leverage." />
            <CardBody className="grid gap-4 md:grid-cols-3">
              <NumberField id="initial_capital" label="Initial capital (paper)" value={form.initial_capital} onChange={num("initial_capital")} error={shown("initial_capital")} />
              <NumberField id="fee_bps" label="Fee (bps)" step={0.5} value={form.fee_bps} onChange={num("fee_bps")} error={shown("fee_bps")} />
              <NumberField id="slippage_bps" label="Slippage (bps)" step={0.5} value={form.slippage_bps} onChange={num("slippage_bps")} error={shown("slippage_bps")} />
              <NumberField id="max_position" label="Max position (fraction of equity)" step={0.05} value={form.max_position} onChange={num("max_position")} error={shown("max_position")} />
              <NumberField id="drawdown_circuit_breaker" label="Drawdown circuit breaker" step={0.05} value={form.drawdown_circuit_breaker} onChange={num("drawdown_circuit_breaker")} error={shown("drawdown_circuit_breaker")} hint="Optional, e.g. 0.25 flattens after a 25% drawdown" />
              <NumberField id="max_fit_seconds" label="Training budget per fit (s)" value={form.max_fit_seconds} onChange={num("max_fit_seconds")} error={shown("max_fit_seconds")} />
            </CardBody>
          </Card>
        </div>

        <div className="space-y-4 xl:sticky xl:top-4 xl:h-fit">
          <Card>
            <CardHeader title="Workload" description="Relative cost, not a time estimate" />
            <CardBody className="space-y-3">
              {!valid ? <p className="text-xs text-fg-subtle">Fix the highlighted fields to see the workload.</p> : null}
              {estimate.isFetching ? <Spinner label="Estimating" /> : null}
              {estimate.isError ? <ErrorState error={estimate.error} /> : null}
              {estimate.data && valid ? (
                <>
                  <KeyValue rows={[
                    ["Agent evaluations", int(estimate.data.agent_evaluations)],
                    ["Model fits (max)", int(estimate.data.model_fits_upper_bound)],
                    ["Simulations (max)", int(estimate.data.simulations_upper_bound)],
                    ["Relative cost", `${estimate.data.relative_cost_index.toFixed(1)}×`],
                  ]} />
                  <p className="text-[11px] leading-relaxed text-fg-subtle">{estimate.data.note}</p>
                </>
              ) : null}
            </CardBody>
          </Card>
          {create.isError ? <ErrorState error={create.error} /> : null}
          <Button type="submit" variant="primary" className="w-full" disabled={create.isPending || (touched && !valid)}>
            {create.isPending ? <Spinner label="Starting" /> : <><Play /> Create and start</>}
          </Button>
          <ResearchNote>
            Historical simulation only. Nothing here places orders or connects to a brokerage, and validation results are not investment advice.
          </ResearchNote>
        </div>
      </form>
    </>
  );
}
