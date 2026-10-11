import { Suspense } from "react";

import { DatasetsPage } from "@/features/datasets/datasets-page";

export const metadata = { title: "Datasets" };

export default function Page() {
  return (
    <Suspense>
      <DatasetsPage />
    </Suspense>
  );
}
