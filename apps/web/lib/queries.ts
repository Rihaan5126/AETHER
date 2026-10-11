"use client";

import { useQuery } from "@tanstack/react-query";
import { z } from "zod";

import { api } from "@/lib/api/client";
import * as S from "@/lib/api/schemas";

const ACTIVE = new Set(["queued", "running"]);
/** Poll every 2s while something is running, otherwise rely on manual refresh. */
const pollWhile = (active: boolean) => (active ? 2000 : false);

export const keys = {
  experiments: ["experiments"] as const,
  experiment: (id: string) => ["experiment", id] as const,
};

export function useExperiments(params: { limit?: number; status?: string } = {}) {
  return useQuery({
    queryKey: [...keys.experiments, params],
    queryFn: () => api.get(S.page(S.Experiment), "/experiments", { limit: params.limit ?? 50, status: params.status }),
    refetchInterval: (q) => pollWhile(!!q.state.data?.items.some((e) => ACTIVE.has(e.status))),
  });
}

export function useExperiment(id: string | undefined) {
  return useQuery({
    queryKey: keys.experiment(id ?? ""),
    queryFn: () => api.get(S.ExperimentDetail, `/experiments/${id}`),
    enabled: !!id,
    refetchInterval: (q) =>
      pollWhile(
        !!q.state.data &&
          (ACTIVE.has(q.state.data.status) || q.state.data.jobs.some((j) => ACTIVE.has(j.status))),
      ),
  });
}

export function useJob(id: string | undefined) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => api.get(S.Job, `/jobs/${id}`),
    enabled: !!id,
    refetchInterval: (q) => pollWhile(!q.state.data || ACTIVE.has(q.state.data.status)),
  });
}

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: () => api.get(S.page(S.Dataset), "/datasets", { limit: 100 }) });
}

export function useDataset(id: string | undefined) {
  return useQuery({ queryKey: ["dataset", id], queryFn: () => api.get(S.DatasetDetail, `/datasets/${id}`), enabled: !!id });
}

export function useDatasetPrices(id: string | undefined, symbol: string | undefined) {
  return useQuery({
    queryKey: ["dataset-prices", id, symbol],
    queryFn: () => api.get(S.DatasetPrices, `/datasets/${id}/prices`, { symbol, max_points: 600 }),
    enabled: !!id && !!symbol,
  });
}

export function useTournaments(params: { experiment_id?: string; generation?: number } = {}) {
  return useQuery({
    queryKey: ["tournaments", params],
    queryFn: () => api.get(S.page(S.Tournament), "/tournaments", { ...params, limit: 200 }),
  });
}

export function useLeaderboard(id: string | undefined) {
  return useQuery({
    queryKey: ["leaderboard", id],
    queryFn: () => api.get(S.Leaderboard, `/tournaments/${id}/leaderboard`),
    enabled: !!id,
  });
}

export function useEquityCurves(id: string | undefined, agentIds?: string[], maxPoints = 600) {
  return useQuery({
    queryKey: ["equity", id, agentIds, maxPoints],
    queryFn: () =>
      api.get(S.EquityCurves.extend({ tournament_id: z.string() }), `/tournaments/${id}/equity-curves`, {
        agent_ids: agentIds, max_points: maxPoints,
      }),
    enabled: !!id,
  });
}

export function useTrades(id: string | undefined, agentId?: string, limit = 50, offset = 0) {
  return useQuery({
    queryKey: ["trades", id, agentId, limit, offset],
    queryFn: () => api.get(S.page(S.TradeRow), `/tournaments/${id}/trades`, { agent_id: agentId, limit, offset }),
    enabled: !!id,
  });
}

export function useAgents(params: { experiment_id?: string; generation?: number; algorithm?: string; limit?: number }) {
  return useQuery({
    queryKey: ["agents", params],
    queryFn: () => api.get(S.page(S.AgentSummary), "/agents", { ...params, limit: params.limit ?? 200 }),
  });
}

export function useAgent(id: string | undefined) {
  return useQuery({ queryKey: ["agent", id], queryFn: () => api.get(S.AgentDetail, `/agents/${id}`), enabled: !!id });
}

export function useAgentMetrics(id: string | undefined) {
  return useQuery({
    queryKey: ["agent-metrics", id],
    queryFn: () => api.get(S.AgentMetrics, `/agents/${id}/metrics`),
    enabled: !!id,
  });
}

export function useLineage(experimentId: string | undefined) {
  return useQuery({
    queryKey: ["lineage", experimentId],
    queryFn: () => api.get(S.LineageGraph, `/experiments/${experimentId}/lineage`),
    enabled: !!experimentId,
  });
}

export function useAnalytics(experimentId: string | undefined) {
  return useQuery({
    queryKey: ["analytics", experimentId],
    queryFn: () => api.get(S.Analytics, `/experiments/${experimentId}/analytics`),
    enabled: !!experimentId,
  });
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ["system"],
    queryFn: () => api.get(S.SystemStatus, "/system/status"),
    refetchInterval: 5000,
  });
}

export function useFinalTestCurves(experimentId: string | undefined, runNumber: number | undefined) {
  return useQuery({
    queryKey: ["final-test-curves", experimentId, runNumber],
    queryFn: () =>
      api.get(S.EquityCurves.extend({ run_number: z.number() }), `/experiments/${experimentId}/final-test/${runNumber}/equity-curves`, {
        max_points: 800,
      }),
    enabled: !!experimentId && runNumber !== undefined,
  });
}
