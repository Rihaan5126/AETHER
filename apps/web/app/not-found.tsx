import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { EmptyState } from "@/components/ui/misc";

export default function NotFound() {
  return (
    <Card>
      <EmptyState
        title="Page not found"
        action={
          <Button asChild variant="primary">
            <Link href="/">Back to overview</Link>
          </Button>
        }
      >
        The page or record does not exist. Agents and experiments are addressed by their stored IDs.
      </EmptyState>
    </Card>
  );
}
