import { expect, test } from "@playwright/test";
import { createDraft, deleteRun, reuseSource } from "./helpers";

// One real solve on our own run. Layers are reused from the newest solve-ready BHU-SHI run, so no extraction runs.
let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("solve a reused run end to end", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const active = await (await request.get("/api/jobs/active")).json();
  test.skip(active.job !== null, "a GPU job is running — not starting a solve");
  runId = await createDraft(request, { reuse_from: await reuseSource(request), routes_wanted: 2 });
  // The delivered Kachchh Desert Sanctuary polygon closes the ±5 km band at km 35 (No-Go), so this
  // end-to-end check relaxes that one rule to exercise a solve that returns routes.
  const pa = await request.put(`/api/runs/${runId}/rules/protected_areas`, {
    data: { enabled: true, params: { tower_setback_m: 0, conductor_buffer_m: 0, treatment: "Allowed" }, reason: "playwright: sanctuary closes the test band" },
  });
  expect(pa.status(), await pa.text()).toBe(200);
  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("layer-table")).toBeVisible();
  await page.getByRole("button", { name: "Solve the route", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/solve$`));
  await expect(page.getByTestId("busy-chip")).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/map$`), { timeout: 5 * 60_000 });
  await expect(page.getByTestId("compare-table").locator("tbody tr").first()).toBeVisible();
});
