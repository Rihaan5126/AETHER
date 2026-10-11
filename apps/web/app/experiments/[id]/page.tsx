import { ExperimentMonitor } from "@/features/experiments/experiment-monitor";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ExperimentMonitor id={decodeURIComponent(id)} />;
}
