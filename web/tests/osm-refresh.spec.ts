import { expect, test } from "@playwright/test";
import { createDraft, deleteRun } from "./helpers";

// "Refresh from OSM" on the power layer: Overpass for the run's band -> store -> re-extract power.
let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("refresh power from OSM re-extracts the layer with voltages", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const active = await (await request.get("/api/jobs/active")).json();
  test.skip(active.job !== null, "a GPU job is running — not starting a refresh");
  runId = await createDraft(request, { reuse_from: "BHU-SHI-0003" });
  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("layer-table")).toBeVisible();
  await page.getByTestId("refresh-power").click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/extracting$`));
  await expect(page.getByTestId("busy-chip")).toBeVisible();
  // the refresh is an extraction job: a solve is refused while it runs
  const busy = await request.post(`/api/runs/${runId}/solve`);
  expect(busy.status()).toBe(409);
  await page.goto(`/runs/${runId}/layers`);
  await expect.poll(async () => (await (await request.get("/api/jobs/active")).json()).job, { timeout: 5 * 60_000 }).toBeNull();
  await page.reload();
  const layers = await (await request.get(`/api/runs/${runId}/layers`)).json();
  const power = layers.layers.find((l: { key: string }) => l.key === "power");
  expect(power.image_source).toBe("OpenStreetMap");
  expect(power.source).toContain("refreshed for this area");
  expect(Object.keys(power.stats.by_voltage).length).toBeGreaterThan(0);
  await expect(page.getByTestId("voltage-legend")).toBeVisible();
});
