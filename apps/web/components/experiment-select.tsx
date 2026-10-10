"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import * as React from "react";

import { Select } from "@/components/ui/input";
import { useExperiments } from "@/lib/queries";

/**
 * The experiment in focus is part of the URL (?experiment=...), so views are
 * linkable. Without one, the most recent completed experiment is used.
 */
export function useExperimentParam() {
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const experiments = useExperiments({ limit: 100 });
  const explicit = params.get("experiment") ?? undefined;
  const fallback =
    experiments.data?.items.find((e) => e.status === "completed")?.id ?? experiments.data?.items[0]?.id;
  const experimentId = explicit ?? fallback;
  const setExperiment = React.useCallback(
    (id: string, extra: Record<string, string | undefined> = {}) => {
      const next = new URLSearchParams(params.toString());
      next.set("experiment", id);
      for (const [k, v] of Object.entries(extra)) {
        if (v === undefined) next.delete(k);
        else next.set(k, v);
      }
      router.replace(`${pathname}?${next.toString()}`, { scroll: false });
    },
    [params, pathname, router],
  );
  return { experimentId, setExperiment, experiments, params };
}

export function ExperimentSelect({
  value,
  onChange,
  onlyCompleted = false,
}: {
  value: string | undefined;
  onChange: (id: string) => void;
  onlyCompleted?: boolean;
}) {
  const experiments = useExperiments({ limit: 100 });
  const items = (experiments.data?.items ?? []).filter((e) => !onlyCompleted || e.status === "completed");
  return (
    <Select
      aria-label="Experiment"
      className="w-72 max-w-full"
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value)}
      disabled={!items.length}
    >
      {!items.length ? <option value="">No experiments yet</option> : null}
      {items.map((e) => (
        <option key={e.id} value={e.id}>
          {e.name} · {e.kind === "tournament" ? "tournament" : `${e.planned_generations} gen`} · {e.status}
        </option>
      ))}
    </Select>
  );
}
