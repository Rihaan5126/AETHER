import { describe, expect, it } from "vitest";

import { duration, int, num, pct, relative, shortAgentId, signClass } from "@/lib/format";

describe("format", () => {
  it("renders undefined metrics as a dash, never 0 or Infinity", () => {
    for (const v of [null, undefined, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(pct(v)).toBe("—");
      expect(num(v)).toBe("—");
      expect(int(v)).toBe("—");
    }
  });

  it("formats signed percentages", () => {
    expect(pct(0.1234, 1, true)).toBe("+12.3%");
    expect(pct(-0.05, 1, true)).toBe("-5.0%");
    expect(pct(0, 1, true)).toBe("0.0%");
  });

  it("formats durations", () => {
    expect(duration(12.34)).toBe("12.3s");
    expect(duration(125)).toBe("2m 5s");
    expect(duration(3 * 3600 + 120)).toBe("3h 2m");
  });

  it("formats relative times", () => {
    const now = Date.parse("2026-01-01T12:00:00Z");
    expect(relative("2026-01-01T11:59:30Z", now)).toBe("just now");
    expect(relative("2026-01-01T11:00:00Z", now)).toBe("1h ago");
    expect(relative(null, now)).toBe("—");
  });

  it("colours by sign", () => {
    expect(signClass(1)).toBe("text-pos");
    expect(signClass(-1)).toBe("text-neg");
    expect(signClass(null)).toBe("text-fg");
  });

  it("shortens agent ids", () => {
    expect(shortAgentId("exp_6c6e37da5d0f-g2-07-mlp")).toBe("G2·07");
    expect(shortAgentId("abcdefghijklmnop")).toBe("ghijklmnop");
  });
});
