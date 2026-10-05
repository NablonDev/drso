import { expect, test } from "@playwright/test";
import { appErrors, watchConsole } from "./helpers";

test("home lists runs", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "My routes" })).toBeVisible();
  const rows = page.getByTestId("run-row");
  await expect(rows.first()).toBeVisible();
  expect(await rows.count()).toBeGreaterThan(0);
  await expect(page.getByTestId("runs-table")).toContainText("BHU-SHI-0003");
  // the global GPU indicator is always present (busy chip or idle text)
  await expect(page.locator(".topbar")).toContainText(/GPU idle|Extracting|Solving/);
  expect(appErrors(errors)).toEqual([]);
});

test("deep link to an unknown run shows the server's message", async ({ page }) => {
  await page.goto("/runs/NOPE-0000/map");
  await expect(page.getByRole("alert")).toContainText("run NOPE-0000 not found");
});
