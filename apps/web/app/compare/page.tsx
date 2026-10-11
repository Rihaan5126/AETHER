import { Suspense } from "react";

import { ComparePage } from "@/features/compare/compare-page";

export const metadata = { title: "Agent Comparison" };

export default function Page() {
  return (
    <Suspense>
      <ComparePage />
    </Suspense>
  );
}
