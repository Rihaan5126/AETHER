import { Suspense } from "react";

import { TournamentsPage } from "@/features/tournaments/tournaments-page";

export const metadata = { title: "Tournaments" };

export default function Page() {
  return (
    <Suspense>
      <TournamentsPage />
    </Suspense>
  );
}
