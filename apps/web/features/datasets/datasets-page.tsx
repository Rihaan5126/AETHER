"use client";

import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Database, Upload } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import * as React from "react";

import { PriceChart } from "@/components/charts/charts";
import { PageHeader, ResearchNote } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Field, Input, Select } from "@/components/ui/input";
import { EmptyState, ErrorState, KeyValue, Progress, Skeleton, Tabs } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import { api } from "@/lib/api/client";
import * as S from "@/lib/api/schemas";
import { dateTime, int } from "@/lib/format";
import { useDataset, useDatasetPrices, useDatasets, useJob } from "@/lib/queries";

function ImportForm({ onDone }: { onDone: (datasetId?: string) => void }) {
  const qc = useQueryClient();
  const [form, setForm] = React.useState({
    name: "Synthetic market", source: "synthetic", symbols: "SYN, ALT", start_date: "2012-01-02", end_date: "2023-12-29", seed: "7",
  });
  const [jobId, setJobId] = React.useState<string>();
  const job = useJob(jobId);
  const set = (k: keyof typeof form) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setForm({ ...form, [k]: e.target.value });
  const mutation = useMutation({
    mutationFn: () =>
      api.post(S.JobRef, "/datasets/import", {
        name: form.name,
        source: form.source,
        symbols: form.symbols.split(/[\s,]+/).filter(Boolean),
        start_date: form.start_date,
        end_date: form.end_date,
        synthetic: { seed: Number(form.seed) },
      }),
    onSuccess: (r) => setJobId(r.job_id),
  });
  const finished = job.data?.status === "completed" ? job.data : undefined;
  const busy = !!jobId && !finished && job.data?.status !== "failed";
  React.useEffect(() => {
    if (finished) {
      void qc.invalidateQueries({ queryKey: ["datasets"] });
      onDone(finished.result?.dataset_id as string | undefined);
    }
  }, [finished, qc, onDone]);

  return (
    <form
      className="space-y-3"
      onSubmit={(e) => {
        e.preventDefault();
        mutation.mutate();
      }}
    >
      <Field label="Name" htmlFor="ds-name"><Input id="ds-name" required maxLength={120} value={form.name} onChange={set("name")} /></Field>
      <Field label="Source" htmlFor="ds-source" hint={form.source === "yahoo" ? "Optional; only works if the API has AETHER_ENABLE_YAHOO=true and network access. Subject to Yahoo's terms." : "Seeded regime-switching generator; fully reproducible offline."}>
        <Select id="ds-source" value={form.source} onChange={set("source")}>
          <option value="synthetic">Synthetic (offline)</option>
          <option value="yahoo">Yahoo Finance (optional)</option>
        </Select>
      </Field>
      <Field label="Symbols" htmlFor="ds-symbols" hint="Up to 10, comma separated"><Input id="ds-symbols" required value={form.symbols} onChange={set("symbols")} /></Field>
      <div className="grid grid-cols-2 gap-3">
        <Field label="Start" htmlFor="ds-start"><Input id="ds-start" type="date" required value={form.start_date} onChange={set("start_date")} /></Field>
        <Field label="End" htmlFor="ds-end"><Input id="ds-end" type="date" required value={form.end_date} onChange={set("end_date")} /></Field>
      </div>
      {form.source === "synthetic" ? (
        <Field label="Generator seed" htmlFor="ds-seed"><Input id="ds-seed" type="number" min={0} value={form.seed} onChange={set("seed")} /></Field>
      ) : null}
      {mutation.isError ? <ErrorState error={mutation.error} /> : null}
      {job.data?.status === "failed" ? <ErrorState error={new Error(job.data.error_message ?? "import failed")} /> : null}
      {busy && job.data ? (
        <div className="space-y-1"><Progress value={job.data.progress} /><p className="text-xs text-fg-subtle">{job.data.message}</p></div>
      ) : null}
      <Button type="submit" variant="primary" disabled={mutation.isPending || busy}><Database /> Import dataset</Button>
    </form>
  );
}

function UploadForm({ onDone }: { onDone: (datasetId?: string) => void }) {
  const qc = useQueryClient();
  const [file, setFile] = React.useState<File | null>(null);
  const [name, setName] = React.useState("");
  const [symbol, setSymbol] = React.useState("");
  const [adjusted, setAdjusted] = React.useState(false);
  const mutation = useMutation({
    mutationFn: () => {
      const fd = new FormData();
      fd.set("file", file!);
      fd.set("name", name || file!.name);
      if (symbol) fd.set("symbol", symbol);
      fd.set("use_adjusted_close", String(adjusted));
      return api.post(S.Dataset, "/datasets/upload", fd);
    },
    onSuccess: (d) => {
      void qc.invalidateQueries({ queryKey: ["datasets"] });
      onDone(d.id);
    },
  });
  return (
    <form className="space-y-3" onSubmit={(e) => { e.preventDefault(); if (file) mutation.mutate(); }}>
      <Field label="CSV or Parquet file" htmlFor="up-file" hint="Columns: date/timestamp, open, high, low, close, volume, optional symbol and adj_close. Validated and cleaned on import.">
        <Input id="up-file" type="file" accept=".csv,.parquet" required onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
      </Field>
      <Field label="Name" htmlFor="up-name"><Input id="up-name" value={name} placeholder={file?.name ?? "My dataset"} onChange={(e) => setName(e.target.value)} /></Field>
      <Field label="Symbol" htmlFor="up-symbol" hint="Required when the file has no symbol column"><Input id="up-symbol" value={symbol} onChange={(e) => setSymbol(e.target.value)} /></Field>
      <label className="flex items-center gap-2 text-xs text-fg-muted">
        <input type="checkbox" checked={adjusted} onChange={(e) => setAdjusted(e.target.checked)} /> Use adjusted close (rescales OHLC)
      </label>
      {mutation.isError ? <ErrorState error={mutation.error} /> : null}
      <Button type="submit" variant="primary" disabled={!file || mutation.isPending}><Upload /> Upload</Button>
    </form>
  );
}

function DatasetDetailPanel({ id }: { id: string }) {
  const ds = useDataset(id);
  const [symbol, setSymbol] = React.useState<string>();
  const sym = symbol && ds.data?.symbols.includes(symbol) ? symbol : ds.data?.symbols[0];
  const prices = useDatasetPrices(id, sym);
  if (ds.isError) return <ErrorState error={ds.error} />;
  if (!ds.data) return <Skeleton className="h-96" />;
  const d = ds.data;
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader
          title={d.name}
          description={`${d.source} · ${d.symbols.length} symbol(s) · ${int(d.rows)} rows`}
          actions={
            <Select aria-label="Symbol" className="w-32" value={sym ?? ""} onChange={(e) => setSymbol(e.target.value)}>
              {d.symbols.map((s) => <option key={s}>{s}</option>)}
            </Select>
          }
        />
        <CardBody>
          {prices.isLoading ? <Skeleton className="h-56" /> : <PriceChart points={prices.data?.points ?? []} />}
          {prices.data?.downsampled ? <p className="mt-1 text-[11px] text-fg-subtle">Downsampled for display.</p> : null}
        </CardBody>
      </Card>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Manifest" description="Content-addressed and immutable" />
          <CardBody>
            <KeyValue rows={[
              ["Dataset hash", d.hash],
              ["Version", d.version],
              ["Range", `${d.start_date} → ${d.end_date}`],
              ["Symbols", d.symbols.join(", ")],
              ["Retrieved", dateTime(d.retrieved_at)],
              ["Used by", `${d.used_by_experiments} experiment(s)`],
              ...Object.entries(d.params).slice(0, 8).map(([k, v]) => [k, typeof v === "object" ? JSON.stringify(v) : String(v)] as [string, string]),
            ]} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Validation" description={`${int(d.validation.rows_in)} rows in, ${int(d.validation.rows_out)} rows kept`} />
          <CardBody className="p-0">
            {d.validation.issues.length ? (
              <Table>
                <thead><tr><Th>Severity</Th><Th>Issue</Th><Th className="text-right">Count</Th></tr></thead>
                <tbody>
                  {d.validation.issues.map((i, k) => (
                    <Tr key={k}>
                      <Td><Badge tone={i.severity === "error" ? "neg" : i.severity === "warning" ? "warn" : "neutral"}>{i.severity}</Badge></Td>
                      <Td><p className="font-mono text-[11px] text-fg-subtle">{i.code}{i.symbol ? ` · ${i.symbol}` : ""}</p><p className="text-fg-muted">{i.message}</p></Td>
                      <Td className="text-right">{i.count}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            ) : <EmptyState title="No issues found" className="py-8" />}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

export function DatasetsPage() {
  const datasets = useDatasets();
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const selected = params.get("dataset") ?? datasets.data?.items[0]?.id;
  const select = React.useCallback((id?: string) => id && router.replace(`${pathname}?dataset=${id}`, { scroll: false }), [pathname, router]);
  const [tab, setTab] = React.useState<"import" | "upload">("import");

  return (
    <>
      <PageHeader title="Datasets" description="Market data used by experiments. Every dataset is validated, versioned by content hash and stored immutably." />
      <div className="grid gap-6 xl:grid-cols-[20rem_1fr]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="Library" />
            <CardBody className="p-0">
              {datasets.isError ? <ErrorState error={datasets.error} className="m-4" /> : null}
              {datasets.isLoading ? <Skeleton className="m-4 h-32" /> : null}
              {datasets.data && !datasets.data.items.length ? <EmptyState title="No datasets yet">Import a synthetic dataset to get started.</EmptyState> : null}
              <ul className="divide-y divide-line">
                {datasets.data?.items.map((d) => (
                  <li key={d.id}>
                    <button type="button" onClick={() => select(d.id)}
                      className={`w-full px-4 py-3 text-left hover:bg-panel-2 ${d.id === selected ? "bg-panel-2" : ""}`}>
                      <p className={`text-sm font-medium ${d.id === selected ? "text-accent" : "text-fg"}`}>{d.name}</p>
                      <p className="mt-0.5 flex items-center gap-2 text-[11px] text-fg-subtle">
                        <Badge>{d.source}</Badge> {d.symbols.join(", ")} · {d.start_date.slice(0, 4)}–{d.end_date.slice(0, 4)}
                      </p>
                    </button>
                  </li>
                ))}
              </ul>
            </CardBody>
          </Card>
          <Card>
            <CardHeader title="Add data" actions={<Tabs value={tab} onChange={setTab} options={[{ value: "import", label: "Import" }, { value: "upload", label: "Upload" }]} />} />
            <CardBody>{tab === "import" ? <ImportForm onDone={select} /> : <UploadForm onDone={select} />}</CardBody>
          </Card>
          <ResearchNote>Synthetic data has no real-world meaning. Results on it test the pipeline and the methodology, not any market.</ResearchNote>
        </div>
        <div className="min-w-0">{selected ? <DatasetDetailPanel id={selected} /> : null}</div>
      </div>
    </>
  );
}
