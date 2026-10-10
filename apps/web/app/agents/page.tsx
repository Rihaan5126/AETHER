import { Suspense } from "react";

import { ArenaPage } from "@/features/agents/arena-page";

export const metadata = { title: "Agent Arena" };

export default function Page() {
  return (
    <Suspense>
      <ArenaPage />
    </Suspense>
  );
}
