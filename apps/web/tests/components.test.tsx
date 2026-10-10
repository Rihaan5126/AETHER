import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { StatusBadge, WarningList } from "@/components/domain";
import { ErrorState } from "@/components/ui/misc";
import { ApiError } from "@/lib/api/client";

describe("components", () => {
  it("labels unevaluated agents", () => {
    render(<StatusBadge status={null} />);
    expect(screen.getByText("not evaluated")).toBeInTheDocument();
  });

  it("lists statistical warnings", () => {
    render(<WarningList warnings={[{ code: "small_sample", severity: "medium", message: "only 3 round trips" }]} />);
    expect(screen.getByText("only 3 round trips")).toBeInTheDocument();
  });

  it("shows API error messages", () => {
    render(<ErrorState error={new ApiError(0, "network_error", "Cannot reach the AETHER API")} />);
    expect(screen.getByText(/Cannot reach the AETHER API/)).toBeInTheDocument();
  });
});
