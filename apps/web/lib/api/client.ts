import type { z } from "zod";

import { type Query, resolveSnapshot, snapshotUrl } from "@/lib/api/snapshot";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;
/** Read-only showcase: serve exported results from /snapshot instead of the API (see lib/api/snapshot.ts). */
export const SNAPSHOT = process.env.NEXT_PUBLIC_AETHER_SNAPSHOT === "true";
export const READ_ONLY_MESSAGE =
  "This is a read-only showcase of stored results. Clone the repository and run AETHER locally to train agents and start experiments.";

/** Structured error from the API (`{"error": {code, message, details}}`) or the network. */
export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export function buildUrl(path: string, query?: Query): string {
  const url = new URL(`${API_URL}/api/v1${path}`);
  for (const [k, v] of Object.entries(query ?? {})) {
    if (v === undefined || v === null || v === "") continue;
    if (Array.isArray(v)) v.forEach((item) => url.searchParams.append(k, item));
    else url.searchParams.set(k, String(v));
  }
  return url.toString();
}

async function request<T extends z.ZodTypeAny>(
  schema: T,
  path: string,
  init: RequestInit & { query?: Query } = {},
): Promise<z.infer<T>> {
  const { query, ...rest } = init;
  if (SNAPSHOT) return snapshotRequest(schema, path, rest.method ?? "GET", query);
  const headers = new Headers(rest.headers);
  if (rest.body && !(rest.body instanceof FormData)) headers.set("Content-Type", "application/json");
  if (API_KEY && rest.method && rest.method !== "GET") headers.set("X-API-Key", API_KEY);
  let res: Response;
  try {
    res = await fetch(buildUrl(path, query), { ...rest, headers });
  } catch (err) {
    throw new ApiError(0, "network_error", `Cannot reach the AETHER API at ${API_URL}. Is it running?`, err);
  }
  const text = await res.text();
  const body: unknown = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const e = (body as { error?: { code?: string; message?: string; details?: unknown } } | null)?.error;
    throw new ApiError(res.status, e?.code ?? "http_error", e?.message ?? res.statusText, e?.details);
  }
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError(res.status, "contract_mismatch", `Unexpected response shape from ${path}`, parsed.error.issues);
  }
  return parsed.data;
}

export async function snapshotRequest<T extends z.ZodTypeAny>(
  schema: T,
  path: string,
  method: string,
  query?: Query,
): Promise<z.infer<T>> {
  if (method !== "GET") throw new ApiError(403, "read_only", READ_ONLY_MESSAGE);
  const { file, transform } = resolveSnapshot(path, query);
  let res: Response;
  try {
    res = await fetch(snapshotUrl(file));
  } catch (err) {
    throw new ApiError(0, "network_error", "Could not load the results snapshot.", err);
  }
  if (!res.ok) {
    throw new ApiError(res.status === 404 ? 404 : res.status, "not_in_snapshot", `${path} is not part of this snapshot`);
  }
  const body: unknown = await res.json();
  const parsed = schema.safeParse(transform ? transform(body) : body);
  if (!parsed.success) {
    throw new ApiError(200, "contract_mismatch", `Unexpected snapshot shape for ${path}`, parsed.error.issues);
  }
  return parsed.data;
}

export const api = {
  get: <T extends z.ZodTypeAny>(schema: T, path: string, query?: Query) =>
    request(schema, path, { method: "GET", query }),
  post: <T extends z.ZodTypeAny>(schema: T, path: string, body?: unknown, headers?: HeadersInit) =>
    request(schema, path, {
      method: "POST",
      body: body instanceof FormData ? body : JSON.stringify(body ?? {}),
      headers,
    }),
};
