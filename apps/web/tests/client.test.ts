import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";

import { api, ApiError, buildUrl } from "@/lib/api/client";

afterEach(() => vi.restoreAllMocks());

describe("api client", () => {
  it("builds versioned URLs and drops empty params", () => {
    const url = new URL(buildUrl("/agents", { experiment_id: "e1", generation: 0, algorithm: "", agent_ids: ["a", "b"] }));
    expect(url.pathname).toBe("/api/v1/agents");
    expect(url.searchParams.get("generation")).toBe("0");
    expect(url.searchParams.has("algorithm")).toBe(false);
    expect(url.searchParams.getAll("agent_ids")).toEqual(["a", "b"]);
  });

  it("surfaces structured API errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ error: { code: "final_test_locked", message: "locked" } }), { status: 409 }),
    );
    const err = await api.post(z.object({}), "/experiments/x/final-test").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect((err as ApiError).code).toBe("final_test_locked");
    expect((err as ApiError).status).toBe(409);
  });

  it("fails loudly on contract drift", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(JSON.stringify({ status: 1 }), { status: 200 }));
    await expect(api.get(z.object({ status: z.string() }), "/health")).rejects.toMatchObject({ code: "contract_mismatch" });
  });

  it("reports an unreachable API", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new TypeError("fetch failed"));
    await expect(api.get(z.object({}), "/health")).rejects.toMatchObject({ code: "network_error" });
  });
});
