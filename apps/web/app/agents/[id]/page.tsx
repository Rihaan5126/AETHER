import { AgentDetailPage } from "@/features/agents/agent-detail";

export default async function Page({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <AgentDetailPage id={decodeURIComponent(id)} />;
}
