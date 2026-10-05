import { expect, test } from "@playwright/test";
import { createDraft, deleteRun } from "./helpers";

// One real solve on our own run. Layers are reused from BHU-SHI-0003, so no extraction runs.
let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("solve a reused run end to end", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const active = await (await request.get("/api/jobs/active")).json();
  test.skip(active.job !== null, "a GPU job is running — not starting a solve");
  runId = await createDraft(request, { reuse_from: "BHU-SHI-0003", routes_wanted: 2 });
  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("layer-table")).toBeVisible();
  await page.getByRole("button", { name: "Solve the route", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/solve$`));
  await expect(page.getByTestId("busy-chip")).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/map$`), { timeout: 5 * 60_000 });
  await expect(page.getByTestId("compare-table").locator("tbody tr").first()).toBeVisible();
});
