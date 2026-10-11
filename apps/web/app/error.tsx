"use client";

import { RotateCcw } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { ErrorState } from "@/components/ui/misc";

/** Route-level error boundary: a rendering bug on one page keeps the navigation usable. */
export default function RouteError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  React.useEffect(() => {
    console.error(error);
  }, [error]);
  return (
    <Card className="p-6">
      <ErrorState error={new Error(`This page failed to render: ${error.message || "unknown error"}`)} />
      <div className="mt-4 flex items-center gap-3">
        <Button onClick={reset}>
          <RotateCcw /> Try again
        </Button>
        {error.digest ? <span className="font-mono text-xs text-fg-subtle">ref {error.digest}</span> : null}
      </div>
    </Card>
  );
}
