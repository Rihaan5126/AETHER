import { Suspense } from "react";

import { OverviewPage } from "@/features/overview/overview-page";

export default function Page() {
  return (
    <Suspense>
      <OverviewPage />
    </Suspense>
  );
}
