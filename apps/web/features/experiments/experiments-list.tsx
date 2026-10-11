"use client";

import { Plus } from "lucide-react";
import Link from "next/link";

import { PageHeader, StatusBadge } from "@/components/domain";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody } from "@/components/ui/card";
import { EmptyState, ErrorState, Progress, Skeleton } from "@/components/ui/misc";
import { Table, Td, Th, Tr } from "@/components/ui/table";
import { relative } from "@/lib/format";
import { useExperiments } from "@/lib/queries";

export function ExperimentsListPage() {
  const q = useExperiments({ limit: 100 });
  return (
    <>
      <PageHeader
        title="Experiments"
        description="Each experiment is an immutable, reproducible record: config, dataset hash, code version, seeds and every result."
        actions={<Button asChild><Link href="/experiments/new"><Plus /> New experiment</Link></Button>}
      />
      <Card>
        <CardBody className="p-0">
          {q.isError ? <ErrorState error={q.error} className="m-4" /> : null}
          {q.isLoading ? <Skeleton className="m-4 h-48" /> : null}
          {q.data && !q.data.items.length ? (
            <EmptyState title="No experiments yet" action={<Button asChild size="sm"><Link href="/experiments/new">Create one</Link></Button>}>
              Or run <code className="font-mono">make demo</code> to produce a small reproducible run.
            </EmptyState>
          ) : null}
          {q.data?.items.length ? (
            <Table aria-label="Experiments">
              <thead>
                <tr><Th>Name</Th><Th>Kind</Th><Th>Status</Th><Th>Symbol</Th><Th className="text-right">Generations</Th><Th className="text-right">Population</Th><Th className="text-right">Seed</Th><Th>Created</Th></tr>
              </thead>
              <tbody>
                {q.data.items.map((e) => (
                  <Tr key={e.id}>
                    <Td>
                      <Link href={`/experiments/${e.id}`} className="font-medium hover:text-accent">{e.name}</Link>
                      {e.test_contaminated ? <Badge tone="warn" className="ml-2">test contaminated</Badge> : null}
                      <p className="font-mono text-[11px] text-fg-subtle">{e.id}</p>
                    </Td>
                    <Td className="text-fg-muted">{e.kind}</Td>
                    <Td>
                      <div className="w-28 space-y-1">
                        <StatusBadge status={e.status} />
                        {e.status === "running" && e.progress !== null ? <Progress value={e.progress} /> : null}
                      </div>
                    </Td>
                    <Td className="font-mono">{e.symbol}</Td>
                    <Td className="text-right">{e.current_generation !== null ? e.current_generation + 1 : 0} / {e.planned_generations}</Td>
                    <Td className="text-right">{e.population_size}</Td>
                    <Td className="text-right font-mono">{e.seed}</Td>
                    <Td className="text-fg-subtle">{relative(e.created_at)}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          ) : null}
        </CardBody>
      </Card>
    </>
  );
}
