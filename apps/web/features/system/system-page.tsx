"use client";

import { ShieldAlert } from "lucide-react";

import { PageHeader } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { ErrorState, KeyValue, Skeleton, Stat } from "@/components/ui/misc";
import { API_URL } from "@/lib/api/client";
import { int } from "@/lib/format";
import { useSystemStatus } from "@/lib/queries";

const LIMIT_LABEL: Record<string, string> = {
  max_concurrent_jobs: "Concurrent jobs",
  max_workers_per_job: "Worker processes per job",
  max_job_seconds: "Max job duration (s)",
  max_population: "Max population",
  max_generations: "Max generations",
  max_optuna_trials: "Max Optuna trials",
  max_upload_mb: "Max upload (MB)",
};

export function SystemPage() {
  const q = useSystemStatus();
  if (q.isError) {
    return (
      <>
        <PageHeader title="System" />
        <ErrorState error={q.error} />
      </>
    );
  }
  if (!q.data) return <Skeleton className="h-96" />;
  const s = q.data;
  const r = s.resources as Record<string, unknown>;
  return (
    <>
      <PageHeader
        title="System"
        description={`API ${API_URL} · version ${s.version} · ${s.environment}`}
        actions={<Badge tone={s.status === "ok" ? "pos" : "warn"}>{s.status}</Badge>}
      />
      <div className="mb-6 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line lg:grid-cols-5">
        {Object.entries(s.jobs).map(([k, v]) => (
          <div key={k} className="bg-panel p-4"><Stat label={`Jobs ${k}`} value={int(v)} /></div>
        ))}
      </div>
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Services" />
          <CardBody>
            <KeyValue rows={[
              ["Database", <span key="db" className={s.database.ok ? "text-pos" : "text-neg"}>{s.database.ok ? `ok (${String(s.database.dialect)})` : String(s.database.error)}</span>],
              ["Job backend", String(s.queue.backend)],
              ...(s.queue.backend === "celery"
                ? ([
                    ["Redis", <span key="redis" className={s.queue.redis_ok ? "text-pos" : "text-neg"}>{s.queue.redis_ok ? "ok" : String(s.queue.error ?? "unreachable")}</span>],
                    ["Queued tasks", String(s.queue.pending ?? "—")],
                  ] as [string, React.ReactNode][])
                : []),
              ["Code version", s.code_version],
              ["Write protection", s.write_protected ? "API key required" : "open (local development)"],
            ]} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Host resources" />
          <CardBody>
            <KeyValue rows={[
              ["CPUs", String(r.cpu_count ?? "—")],
              ["CPU usage", `${String(r.cpu_percent ?? "—")}%`],
              ["Load average", Array.isArray(r.load_average) ? r.load_average.join(" / ") : "—"],
              ["Memory", `${String(r.memory_total_mb ?? "—")} MB, ${String(r.memory_used_percent ?? "—")}% used`],
              ["Disk free (artifacts)", `${String(r.disk_free_gb ?? "—")} GB`],
            ]} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Resource limits" description="Server-side caps on what a request can trigger" />
          <CardBody>
            <KeyValue rows={Object.entries(s.limits).map(([k, v]) => [LIMIT_LABEL[k] ?? k, int(v)])} />
          </CardBody>
        </Card>
        <Card>
          <CardHeader title="Dependencies" />
          <CardBody>
            <KeyValue rows={Object.entries(s.dependencies).map(([k, v]) => [k, v])} />
          </CardBody>
        </Card>
        <Card className="lg:col-span-2">
          <CardBody className="flex items-start gap-3 text-sm text-fg-muted">
            <ShieldAlert className="mt-0.5 size-5 shrink-0 text-warn" />
            <p>{s.disclaimer}</p>
          </CardBody>
        </Card>
      </div>
    </>
  );
}
