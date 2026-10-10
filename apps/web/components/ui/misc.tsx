import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import * as React from "react";

import { ApiError } from "@/lib/api/client";
import { cn } from "@/lib/utils";

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("animate-pulse rounded-md bg-raised", className)} aria-hidden />;
}

export function Spinner({ className, label = "Loading" }: { className?: string; label?: string }) {
  return (
    <span role="status" className={cn("inline-flex items-center gap-2 text-xs text-fg-subtle", className)}>
      <Loader2 className="size-4 animate-spin" aria-hidden />
      {label}
    </span>
  );
}

export function Progress({ value, className }: { value: number; className?: string }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-raised", className)}
    >
      <div className="h-full rounded-full bg-accent transition-[width] duration-500" style={{ width: `${pct}%` }} />
    </div>
  );
}

export function EmptyState({
  title,
  children,
  action,
  className,
}: {
  title: string;
  children?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-2 px-6 py-12 text-center", className)}>
      <Inbox className="size-6 text-fg-subtle" aria-hidden />
      <p className="text-sm font-medium text-fg">{title}</p>
      {children ? <div className="max-w-md text-xs leading-relaxed text-fg-subtle">{children}</div> : null}
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export function ErrorState({ error, className }: { error: unknown; className?: string }) {
  const message =
    error instanceof ApiError ? error.message : error instanceof Error ? error.message : "Something went wrong";
  const code = error instanceof ApiError ? error.code : undefined;
  return (
    <div role="alert" className={cn("flex items-start gap-3 rounded-md border border-neg/30 bg-neg/5 p-4 text-sm", className)}>
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-neg" aria-hidden />
      <div>
        <p className="font-medium text-fg">Could not load data</p>
        <p className="mt-0.5 text-xs text-fg-muted">
          {message}
          {code ? <span className="ml-1 font-mono text-fg-subtle">({code})</span> : null}
        </p>
      </div>
    </div>
  );
}

export function Stat({
  label,
  value,
  sub,
  className,
  valueClassName,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  className?: string;
  valueClassName?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <p className="text-[11px] font-medium uppercase tracking-wider text-fg-subtle">{label}</p>
      <p className={cn("mt-1 truncate font-mono text-xl font-medium tabular text-fg", valueClassName)}>{value}</p>
      {sub ? <p className="mt-0.5 truncate text-xs text-fg-subtle">{sub}</p> : null}
    </div>
  );
}

export function KeyValue({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[minmax(0,10rem)_1fr] gap-x-4 gap-y-1.5 text-sm">
      {rows.map(([k, v]) => (
        <React.Fragment key={k}>
          <dt className="truncate text-fg-subtle">{k}</dt>
          <dd className="min-w-0 truncate font-mono text-xs leading-5 tabular text-fg">{v}</dd>
        </React.Fragment>
      ))}
    </dl>
  );
}

export function Tabs<T extends string>({
  value,
  onChange,
  options,
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string }[];
  className?: string;
}) {
  return (
    <div role="tablist" className={cn("inline-flex rounded-md border border-line bg-panel-2 p-0.5", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          type="button"
          aria-selected={value === o.value}
          onClick={() => onChange(o.value)}
          className={cn(
            "rounded px-3 py-1 text-xs font-medium transition-colors",
            value === o.value ? "bg-raised text-fg shadow-sm" : "text-fg-subtle hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
