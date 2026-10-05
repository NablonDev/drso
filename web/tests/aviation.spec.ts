import { expect, test } from "@playwright/test";
import { createDraft, deleteRun } from "./helpers";

// Airports refresh from Esri Living Atlas + OSM; the approach path (runways) refreshes from OSM.
let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("airports refresh from Living Atlas + OSM; runways refresh from OSM", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const active = await (await request.get("/api/jobs/active")).json();
  test.skip(active.job !== null, "a GPU job is running — not starting an extraction");
  runId = await createDraft(request, { reuse_from: "BHU-SHI-0003" });
  const ex = await request.post(`/api/runs/${runId}/extract`, { data: { layers: ["aviation", "approach"] } });
  expect(ex.status(), await ex.text()).toBe(202);
  await expect.poll(async () => (await (await request.get("/api/jobs/active")).json()).job, { timeout: 5 * 60_000 }).toBeNull();

  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("layer-table")).toBeVisible();
  const btn = page.getByTestId("refresh-aviation");
  await expect(btn).toHaveText("Refresh from Living Atlas + OSM");
  await expect(btn).toBeEnabled();
  await btn.click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/extracting$`));
  await expect.poll(async () => (await (await request.get("/api/jobs/active")).json()).job, { timeout: 5 * 60_000 }).toBeNull();

  // runways: their own refresh, from OSM
  await page.goto(`/runs/${runId}/layers`);
  const rbtn = page.getByTestId("refresh-approach");
  await expect(rbtn).toHaveText("Refresh from OSM");
  await rbtn.click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/extracting$`));
  await expect.poll(async () => (await (await request.get("/api/jobs/active")).json()).job, { timeout: 5 * 60_000 }).toBeNull();

  const layers = await (await request.get(`/api/runs/${runId}/layers`)).json();
  const ap = layers.layers.find((l: { key: string }) => l.key === "aviation");
  const rw = layers.layers.find((l: { key: string }) => l.key === "approach");
  expect(ap.name).toBe("Airports");
  expect(ap.source).toContain("refreshed for this area");
  expect(ap.feature_count).toBeGreaterThan(0);
  expect(rw.name).toBe("Airport approach path");
  expect(rw.status).toBe("done");
  expect(rw.source).toContain("refreshed for this area");
  expect(ap.stats.refresh.osm_fetched).toBeGreaterThanOrEqual(0);
});
