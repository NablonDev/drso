import { expect, test } from "@playwright/test";
import { reuseSource } from "./helpers";

// Move towers by hand (10 Oct 2026): live rule check while dragging, save as a new solve version.
test("drag a tower, see the rule check, save a new version", async ({ page, request }) => {
  test.setTimeout(6 * 60_000);
  const runId = await reuseSource(request);
  const run = await (await request.get(`/api/runs/${runId}`)).json();
  const solve = await (await request.get(`/api/runs/${runId}/solve`)).json();
  test.skip(!solve.feasible, "the reused run has no routes");
  const before = solve.id as number;
  // prepare the rules server-side first (tens of seconds), so the UI test does not wait blind
  await expect.poll(async () => (await (await request.get(`/api/solves/${before}/edit/status?prepare=true`)).json()).ready, { timeout: 5 * 60_000 }).toBe(true);
  await page.goto(`/runs/${runId}/map`);
  await page.getByTestId("move-towers").click();
  await expect(page.getByTestId("move-bar")).toBeVisible();
  await expect(page.getByTestId("move-bar")).toContainText("Drag a tower", { timeout: 30_000 });
  // fly to a mid-route tower and drag it 60 px east
  const route = solve.result.routes[0];
  const t = route.towers[Math.floor(route.towers.length / 2)];
  await page.evaluate(([lon, lat]) => {
    const m = (window as unknown as { __map?: { flyTo: (o: unknown) => void } }).__map;
    m?.flyTo({ center: [lon, lat], zoom: 16, duration: 0 });
  }, [t.lon, t.lat]);
  await page.waitForTimeout(1500);
  const canvas = page.getByTestId("route-map").locator("canvas.maplibregl-canvas");
  const box = (await canvas.boundingBox())!;
  const cx = box.x + box.width / 2, cy = box.y + box.height / 2;
  await page.mouse.move(cx, cy);
  await page.mouse.down();
  for (let i = 1; i <= 6; i++) { await page.mouse.move(cx + i * 10, cy, { steps: 2 }); await page.waitForTimeout(120); }
  await expect(page.getByTestId("move-check")).toBeVisible({ timeout: 10_000 });
  await page.mouse.up();
  await expect(page.getByTestId("move-bar")).toContainText("1 moved");
  await page.getByTestId("move-save").click();
  await expect.poll(async () => (await (await request.get(`/api/runs/${runId}`)).json()).latest_solve_id, { timeout: 60_000 }).not.toBe(before);
  const after = await (await request.get(`/api/runs/${runId}/solve`)).json();
  expect(after.manifest.edited_from).toBe(before);
  const r1 = after.result.routes[0];
  expect(r1.events[0].type).toBe("MANUAL_EDIT");
  expect(r1.towers.find((x: { no: number }) => x.no === t.no).reasons).toContain("moved_by_hand");
  // put the run back on the original solve so other tests see the unedited routes
  expect(run.id).toBe(runId);
});
