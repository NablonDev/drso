import { expect, test } from "@playwright/test";
import { createDraft, deleteRun, reuseSource } from "./helpers";

// "Refresh from OSM" on the Highways layer (key roads): Overpass for the run's band -> store -> re-extract.
let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("refresh highways from OSM re-extracts NH / SH with a class legend", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const active = await (await request.get("/api/jobs/active")).json();
  test.skip(active.job !== null, "a GPU job is running — not starting a refresh");
  runId = await createDraft(request, { reuse_from: await reuseSource(request) });
  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("layer-table")).toBeVisible();
  const btn = page.getByTestId("refresh-roads");
  await expect(btn).toHaveText("Refresh from OSM");
  await btn.click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/extracting$`));
  const busy = await request.post(`/api/runs/${runId}/solve`);
  expect(busy.status()).toBe(409);
  await expect.poll(async () => (await (await request.get("/api/jobs/active")).json()).job, { timeout: 5 * 60_000 }).toBeNull();
  const layers = await (await request.get(`/api/runs/${runId}/layers`)).json();
  const roads = layers.layers.find((l: { key: string }) => l.key === "roads");
  expect(roads.name).toContain("Highways");
  expect(roads.image_source).toBe("OpenStreetMap");
  expect(roads.source).toContain("refreshed for this area");
  expect(roads.stats.by_class.nh).toBeGreaterThan(0);
  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("highway-legend")).toContainText("NH");
});
