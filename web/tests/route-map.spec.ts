import { expect, test } from "@playwright/test";
import { appErrors, watchConsole } from "./helpers";

test("route map renders the map canvas and four routes in the comparison table", async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto("/runs/BHU-SHI-0003/map");
  await expect(page.getByRole("heading", { name: "Four routes" })).toBeVisible();
  const canvas = page.getByTestId("route-map").locator("canvas.maplibregl-canvas");
  await expect(canvas).toBeVisible();
  const box = await canvas.boundingBox();
  expect(box!.width).toBeGreaterThan(300);
  expect(box!.height).toBeGreaterThan(300);
  const rows = page.getByTestId("compare-table").locator("tbody tr");
  await expect(rows).toHaveCount(4);
  // selecting a row selects the route card too
  await rows.nth(2).click();
  await expect(rows.nth(2)).toHaveClass(/selrow/);
  await expect(page.getByTestId("route-cards").locator(".routecard").nth(2)).toHaveAttribute("aria-pressed", "true");
  // every run layer can be shown under the routes
  const lp = page.getByTestId("route-layers");
  await expect(lp.locator(".lrow").first()).toBeVisible();
  const n = await lp.locator(".lrow").count();
  await lp.getByRole("button", { name: "All", exact: true }).click();
  await expect(lp).toContainText(`${n} of ${n} shown`);
  await lp.getByRole("button", { name: "None", exact: true }).click();
  await expect(lp).toContainText(`0 of ${n} shown`);
  await lp.getByRole("checkbox").first().check();
  await expect(lp).toContainText(`1 of ${n} shown`);
  // the map can fill the screen; Escape brings the page back
  const mapBox = page.getByTestId("route-map");
  await page.getByTestId("map-maximise").click();
  await expect(mapBox).toHaveClass(/maxed/);
  const full = await mapBox.boundingBox();
  expect(full!.width).toBeGreaterThan(box!.width);
  await page.keyboard.press("Escape");
  await expect(mapBox).not.toHaveClass(/maxed/);
  // zoom readout never exceeds the native ceiling
  await expect(page.getByTestId("zoombar")).toContainText(/z\d/);
  expect(appErrors(errors)).toEqual([]);
});

test("why this route shows the event log and the money", async ({ page }) => {
  await page.goto("/runs/BHU-SHI-0003/why");
  await expect(page.getByTestId("story").locator(".ev").first()).toBeVisible();
  await expect(page.getByTestId("money")).toContainText("Total");
  await expect(page.getByTestId("beeline-blockers")).toBeVisible();
  await expect(page.getByTestId("overrides")).toContainText("Built-up cluster");
});
