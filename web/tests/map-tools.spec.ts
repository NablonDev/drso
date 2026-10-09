import { expect, test } from "@playwright/test";
import { appErrors, createDraft, deleteRun, reuseSource, watchConsole } from "./helpers";

// Map tools (Vivek, 8 Oct 2026): route show / hide, right-click copy coordinates, measure, draw + add shapes.

test("route map: routes can be hidden, right-click shows coordinates, the ruler measures in metres", async ({ page, context }) => {
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  const errors = watchConsole(page);
  await page.goto("/runs/BHU-SHI-0003/map");
  await expect(page.getByRole("heading", { name: "Four routes" })).toBeVisible();
  const canvas = page.getByTestId("route-map").locator("canvas.maplibregl-canvas");
  await expect(canvas).toBeVisible();
  await canvas.scrollIntoViewIfNeeded();

  const rt = page.getByTestId("route-toggles");
  await expect(rt).toContainText("4 of 4 shown");
  await page.getByTestId("show-route-2").uncheck();
  await expect(rt).toContainText("3 of 4 shown");
  await rt.getByRole("button", { name: "None", exact: true }).click();
  await expect(rt).toContainText("0 of 4 shown");
  await rt.getByRole("button", { name: "All", exact: true }).click();
  await expect(rt).toContainText("4 of 4 shown");

  const box = (await canvas.boundingBox())!;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  await page.mouse.click(cx, cy, { button: "right" });
  const menu = page.getByTestId("map-context-menu");
  await expect(menu).toBeVisible();
  const coords = (await menu.locator(".mt-coords").textContent()) ?? "";
  expect(coords).toMatch(/^2\d\.\d{6}, [67]\d\.\d{6}$/);
  await menu.getByRole("menuitem", { name: "Copy coordinates (lat, lon)" }).click();
  await expect(menu).toBeHidden();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toBe(coords);

  await page.getByTestId("tool-measure").click();
  await expect(page.getByTestId("tool-hint")).toContainText("Click to start");
  await page.mouse.click(cx - 150, cy);
  await page.mouse.move(cx + 50, cy, { steps: 4 });
  const label = page.getByTestId("measure-label");
  await expect(label).toBeVisible();
  const live = (await label.textContent()) ?? "";
  expect(live).toMatch(/^[\d,]+ m$/);
  await page.mouse.move(cx + 150, cy, { steps: 4 });
  await expect(label).not.toHaveText(live);            // the length follows the pointer
  await page.mouse.dblclick(cx + 150, cy);
  await expect(page.getByText(/^Measured [\d,]+ m\./)).toBeVisible();
  // Esc clears the measurement without leaving the page or the map
  await page.keyboard.press("Escape");
  await expect(page.getByText(/^Measured/)).toBeHidden();
  expect(appErrors(errors)).toEqual([]);
});

let runId: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("layer map: a drawn polygon is added to a layer for this run, listed, and deleted", async ({ page, request }) => {
  runId = await createDraft(request, { reuse_from: await reuseSource(request) });
  await page.goto(`/runs/${runId}/layers`);
  const canvas = page.getByTestId("layer-map").locator("canvas.maplibregl-canvas");
  await expect(canvas).toBeVisible();
  await canvas.scrollIntoViewIfNeeded();
  await page.mouse.wheel(0, 300);
  const box = (await canvas.boundingBox())!;
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  await page.getByTestId("tool-polygon").click();
  await page.mouse.click(cx - 40, cy - 40);
  await page.mouse.click(cx + 40, cy - 40);
  await page.mouse.click(cx + 40, cy + 40);
  await page.keyboard.press("Enter");

  const dlg = page.getByTestId("shape-dialog");
  await expect(dlg).toBeVisible();
  await expect(dlg).toContainText("Polygon");
  // only layers that take polygons are offered
  const opts = await page.getByTestId("shape-layer").locator("option").allTextContents();
  expect(opts).toContain("Buildings");
  await expect(page.getByTestId("shape-save")).toBeDisabled();
  await page.getByTestId("shape-layer").selectOption({ label: "Buildings" });
  await page.getByTestId("shape-name").fill("Playwright shed");
  await page.getByTestId("scope-temporary").check();
  await page.getByTestId("shape-save").click();

  const rerun = page.getByTestId("rerun-dialog");
  await expect(rerun).toBeVisible();
  await expect(rerun).toContainText("for this run only");
  await page.getByTestId("rerun-later").click();
  await expect(rerun).toBeHidden();

  await page.getByTestId("tool-shapes").click();
  const list = page.getByTestId("shapes-list");
  await expect(list).toContainText("Playwright shed");
  await expect(list).toContainText("temporary");
  // permanent shapes from earlier runs (e.g. the traced switchyards) are listed too; count only ours
  const shapes = (await (await request.get(`/api/runs/${runId}/shapes`)).json()).shapes as { name: string; scope: string }[];
  expect(shapes.filter((x) => x.name === "Playwright shed" && x.scope === "temporary").length).toBe(1);

  const row = list.locator(".mt-item").filter({ hasText: "Playwright shed" });
  await row.getByRole("button", { name: "Delete" }).click();
  await row.getByRole("button", { name: "Confirm delete" }).click();
  await expect(page.getByTestId("rerun-dialog")).toBeVisible();
  await page.getByTestId("rerun-later").click();
  await expect(list).not.toContainText("Playwright shed");
});
