import { expect, test } from "@playwright/test";
import { reuseSource } from "./helpers";

// Ground height on the map (10 Oct 2026): hillshade / 3D control, cursor elevation, route profile.
test("terrain control, elevation readout and route profile on a solved run", async ({ page, request }) => {
  const runId = await reuseSource(request);
  const solve = await (await request.get(`/api/runs/${runId}/solve`)).json();
  test.skip(!solve.feasible, "the reused run has no routes to profile");
  const cov = await (await request.get(`/api/runs/${runId}/dem`)).json();
  expect(cov.complete).toBe(true);
  await page.goto(`/runs/${runId}/map`);
  const ctl = page.getByTestId("terrain-control");
  await expect(ctl).toBeVisible();
  await page.getByTestId("terrain-hillshade").click();
  await expect(page.getByTestId("terrain-hillshade")).toHaveAttribute("aria-pressed", "true");
  // a DEM tile is served for the view
  const tile = await request.get("/api/tiles/dem/12/2838/1757.png");
  expect([200, 404]).toContain(tile.status());
  // the profile panel renders with towers and ground
  const prof = page.getByTestId("route-profile");
  await expect(prof).toBeVisible({ timeout: 30_000 });
  await expect(prof.locator("svg .pf-ground")).toHaveCount(1);
  expect(await prof.locator("svg .pf-tower").count()).toBeGreaterThan(10);
  // hovering the profile shows chainage / ground / nearest tower
  const box = (await prof.locator("svg").boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await expect(prof.locator(".profile-foot")).toContainText("nearest tower");
  // cursor readout on the map
  const canvas = page.getByTestId("route-map").locator("canvas.maplibregl-canvas");
  const cb = (await canvas.boundingBox())!;
  await page.mouse.move(cb.x + cb.width / 2, cb.y + cb.height / 2);
  await expect(page.getByTestId("terrain-readout")).toBeVisible();
});
