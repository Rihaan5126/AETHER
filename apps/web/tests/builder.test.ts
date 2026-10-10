import { describe, expect, it } from "vitest";

import { DEFAULT_FORM, validateForm } from "@/features/experiments/experiment-builder";

const valid = { ...DEFAULT_FORM, dataset_id: "ds_1", symbol: "SYN" };

describe("experiment builder validation", () => {
  it("accepts the defaults once a dataset is chosen", () => {
    expect(validateForm(valid)).toEqual({});
    expect(validateForm(DEFAULT_FORM)).toHaveProperty("dataset_id");
  });

  it("mirrors the API's population constraints", () => {
    expect(validateForm({ ...valid, elite_count: 8 })).toHaveProperty("elite_count");
    expect(validateForm({ ...valid, elite_count: 4, immigrant_count: 5 })).toHaveProperty("immigrant_count");
    expect(validateForm({ ...valid, families: [] })).toHaveProperty("families");
  });

  it("rejects leverage and invalid costs", () => {
    expect(validateForm({ ...valid, max_position: 1.5 })).toHaveProperty("max_position");
    expect(validateForm({ ...valid, fee_bps: -1 })).toHaveProperty("fee_bps");
    expect(validateForm({ ...valid, test_fraction: 0.6 })).toHaveProperty("test_fraction");
    expect(validateForm({ ...valid, seed: Number.NaN })).toHaveProperty("seed");
  });
});
