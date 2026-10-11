"use client";

import { Info, Lock } from "lucide-react";
import * as React from "react";

import { Badge } from "@/components/ui/badge";
import type { Warning } from "@/lib/api/schemas";
import { algoColor, algoLabel } from "@/lib/format";
import { cn } from "@/lib/utils";

const STATUS_TONE: Record<string, "neutral" | "accent" | "pos" | "neg" | "warn"> = {
  draft: "neutral",
  queued: "neutral",
  running: "accent",
  completed: "pos",
  failed: "neg",
  cancelled: "warn",
  eligible: "pos",
  benchmark: "neutral",
  insufficient_evidence: "warn",
  excluded: "neg",
};

const STATUS_LABEL: Record<string, string> = {
  insufficient_evidence: "insufficient evidence",
};

export function StatusBadge({ status, className }: { status: string | null | undefined; className?: string }) {
  if (!status) return <Badge className={className}>not evaluated</Badge>;
  return (
    <Badge tone={STATUS_TONE[status] ?? "neutral"} className={className}>
      {status === "running" ? <span className="size-1.5 animate-pulse rounded-full bg-accent" aria-hidden /> : null}
      {STATUS_LABEL[status] ?? status}
    </Badge>
  );
}

export function AlgoTag({ algorithm, className }: { algorithm: string; className?: string }) {
  return (
    <span className={cn("inline-flex items-center gap-1.5 text-xs text-fg-muted", className)}>
      <span className="size-2 rounded-full" style={{ background: algoColor(algorithm) }} aria-hidden />
      {algoLabel(algorithm)}
    </span>
  );
}

export function PeriodBadge({ period }: { period: "validation" | "test" | string }) {
  if (period === "test") {
    return (
      <Badge tone="warn" title="Locked final test period: never used for selection or evolution">
        <Lock className="size-3" aria-hidden /> Final test
      </Badge>
    );
  }
  return (
    <Badge tone="accent" title="Walk-forward validation folds inside the development period">
      Validation
    </Badge>
  );
}

const SEVERITY_TONE: Record<string, "neg" | "warn" | "neutral"> = { high: "neg", medium: "warn", info: "neutral" };

export function WarningBadges({ warnings, max = 3 }: { warnings: Warning[]; max?: number }) {
  if (!warnings.length) return <span className="text-xs text-fg-subtle">none</span>;
  const shown = warnings.slice(0, max);
  return (
    <span className="inline-flex flex-wrap gap-1">
      {shown.map((w) => (
        <Badge key={w.code} tone={SEVERITY_TONE[w.severity] ?? "neutral"} title={w.message}>
          {w.code.replaceAll("_", " ")}
        </Badge>
      ))}
      {warnings.length > max ? <Badge title={warnings.slice(max).map((w) => w.message).join("\n")}>+{warnings.length - max}</Badge> : null}
    </span>
  );
}

export function WarningList({ warnings }: { warnings: Warning[] }) {
  if (!warnings.length) {
    return <p className="text-xs text-fg-subtle">No statistical warnings were raised for this result.</p>;
  }
  return (
    <ul className="space-y-1.5">
      {warnings.map((w) => (
        <li key={w.code} className="flex items-start gap-2 text-xs">
          <Badge tone={SEVERITY_TONE[w.severity] ?? "neutral"}>{w.severity}</Badge>
          <span className="text-fg-muted">{w.message}</span>
        </li>
      ))}
    </ul>
  );
}

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-fg">{title}</h1>
        {description ? <p className="mt-1 max-w-3xl text-sm text-fg-muted">{description}</p> : null}
      </div>
      {actions ? <div className="flex flex-wrap items-center gap-2">{actions}</div> : null}
    </div>
  );
}

export function ResearchNote({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-start gap-2.5 rounded-md border border-line bg-panel-2 px-3 py-2.5 text-xs leading-relaxed text-fg-muted", className)}>
      <Info className="mt-0.5 size-3.5 shrink-0 text-accent" aria-hidden />
      <div>{children}</div>
    </div>
  );
}
