import { Suspense } from "react";

import { AnalyticsPage } from "@/features/analytics/analytics-page";

export const metadata = { title: "Research Analytics" };

export default function Page() {
  return (
    <Suspense>
      <AnalyticsPage />
    </Suspense>
  );
}
