import { Suspense } from "react";

import { EvolutionPage } from "@/features/evolution/evolution-page";

export const metadata = { title: "Evolution Lab" };

export default function Page() {
  return (
    <Suspense>
      <EvolutionPage />
    </Suspense>
  );
}
