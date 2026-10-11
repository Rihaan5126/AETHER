import { expect, test } from "@playwright/test";

/**
 * Smoke test against a running API with at least one completed experiment
 * (`make demo`). Every page must render without uncaught errors.
 */
const PAGES: [string, RegExp][] = [
  ["/", /Overview/],
  ["/agents", /Agent Arena/],
  ["/tournaments", /Tournaments/],
  ["/compare", /Agent Comparison/],
  ["/evolution", /Evolution Lab/],
  ["/analytics", /Research Analytics/],
  ["/datasets", /Datasets/],
  ["/experiments", /Experiments/],
  ["/experiments/new", /New experiment/],
  ["/system", /System/],
];

test.describe("dashboard", () => {
  for (const [path, heading] of PAGES) {
    test(`renders ${path}`, async ({ page }) => {
      const errors: string[] = [];
      page.on("pageerror", (e) => errors.push(e.message));
      await page.goto(path);
      await expect(page.getByRole("heading", { level: 1, name: heading })).toBeVisible();
      await expect(page.getByText(/Cannot reach the AETHER API/)).toHaveCount(0);
      expect(errors).toEqual([]);
    });
  }

  test("leaderboard links to agent detail", async ({ page }) => {
    await page.goto("/tournaments");
    const table = page.getByRole("table", { name: "Leaderboard" });
    await expect(table.locator("tbody tr").first()).toBeVisible();
    await table.locator("tbody tr").first().getByRole("link").click();
    await expect(page.getByText("Why this rank")).toBeVisible();
    await expect(page.getByText("Performance by validation fold")).toBeVisible();
  });

  test("builder validates input before submitting", async ({ page }) => {
    await page.goto("/experiments/new");
    await page.getByLabel("Population").fill("4");
    await page.getByLabel("Elites").fill("4");
    await page.getByRole("button", { name: /Create and start/ }).click();
    await expect(page.getByText("Must be smaller than the population")).toBeVisible();
    await expect(page).toHaveURL(/\/experiments\/new$/);
  });

  test("system page shows the simulation-only disclaimer", async ({ page }) => {
    await page.goto("/system");
    await expect(page.getByText(/No live trading|simulation/i).first()).toBeVisible();
  });

  test("create and run a small experiment end to end", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/experiments/new");
    await page.getByLabel("Name").fill(`e2e ${Date.now()}`);
    for (const fam of ["Random Forest", "XGBoost", "MLP"]) await page.getByText(fam, { exact: true }).click();
    await page.getByLabel("Population").fill("4");
    await page.getByLabel("Generations").fill("1");
    await page.getByLabel("Elites").fill("1");
    await page.getByLabel("Immigrants").fill("0");
    await page.getByLabel("Walk-forward folds").fill("2");
    await expect(page.getByText("Agent evaluations")).toBeVisible();
    await page.getByRole("button", { name: /Create and start/ }).click();
    await expect(page).toHaveURL(/\/experiments\/exp_/);
    await expect(page.getByText("Locked final test")).toBeVisible();
    await expect(page.getByRole("button", { name: "Run the final test" })).toBeVisible({ timeout: 200_000 });
    await page.getByRole("button", { name: "Run the final test" }).click();
    await expect(page.getByText(/Run this once, when you are done developing/)).toBeVisible();
  });
});
