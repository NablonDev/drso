import { expect, test } from "@playwright/test";
import { createDraft, deleteRun } from "./helpers";

// Cell towers (Tarang Sanchar): extracted from the India store, shown on the review screen with
// structure counts; "Refresh from Tarang Sanchar" is visible but disabled until a source exists.
let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("cell towers layer extracts and shows a disabled Tarang Sanchar refresh", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const active = await (await request.get("/api/jobs/active")).json();
  test.skip(active.job !== null, "a GPU job is running — not starting an extraction");
  runId = await createDraft(request, { reuse_from: "BHU-SHI-0003" });
  const ex = await request.post(`/api/runs/${runId}/extract`, { data: { layers: ["telecom"] } });
  expect(ex.status(), await ex.text()).toBe(202);
  await expect.poll(async () => (await (await request.get("/api/jobs/active")).json()).job, { timeout: 5 * 60_000 }).toBeNull();

  const layers = await (await request.get(`/api/runs/${runId}/layers`)).json();
  const t = layers.layers.find((l: { key: string }) => l.key === "telecom");
  expect(t.status).toBe("done");
  expect(t.image_source).toBe("Tarang Sanchar");
  expect(t.feature_count).toBeGreaterThan(0);

  await page.goto(`/runs/${runId}/layers`);
  await expect(page.getByTestId("layer-table")).toBeVisible();
  await expect(page.getByTestId("structure-counts")).toContainText("ground");
  const btn = page.getByTestId("refresh-telecom");
  await expect(btn).toHaveText("Refresh from Tarang Sanchar");
  await expect(btn).toBeDisabled();
  await expect(btn).toHaveAttribute("title", /OTP/);
  // the power layer's button is unchanged
  await expect(page.getByTestId("refresh-power")).toHaveText("Refresh from OSM");
  expect((await request.post(`/api/runs/${runId}/layers/telecom/refresh`)).status()).toBe(503);
});
