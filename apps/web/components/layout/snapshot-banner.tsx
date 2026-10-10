"use client";

import { useQuery } from "@tanstack/react-query";
import { Archive } from "lucide-react";
import { z } from "zod";

import { SNAPSHOT, snapshotRequest } from "@/lib/api/client";
import { dateTime } from "@/lib/format";

const REPO_URL = process.env.NEXT_PUBLIC_REPO_URL;
const Meta = z.object({ generated_at: z.string(), code_version: z.string() });

/** Shown only in snapshot mode: says the page is a static export and when it was made. */
export function SnapshotBanner() {
  const meta = useQuery({
    queryKey: ["snapshot-meta"],
    queryFn: () => snapshotRequest(Meta, "/meta", "GET"),
    enabled: SNAPSHOT,
    staleTime: Infinity,
  });
  if (!SNAPSHOT) return null;
  return (
    <div className="mb-6 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-line bg-panel px-3 py-2 text-xs text-fg-muted">
      <Archive className="size-3.5 shrink-0 text-accent" aria-hidden />
      <span className="font-medium text-fg">Read-only showcase.</span>
      <span>
        Results computed locally and exported
        {meta.data ? ` ${dateTime(meta.data.generated_at)}` : ""}. Training and new experiments are disabled here.
      </span>
      {REPO_URL ? (
        <a className="text-accent hover:underline" href={REPO_URL} target="_blank" rel="noreferrer">
          Source on GitHub
        </a>
      ) : null}
    </div>
  );
}
