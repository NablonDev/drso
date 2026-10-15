import { expect, test } from "@playwright/test";

// Design check on the route map (15 Oct 2026): the limits in force, crossings / towers / sections
// tables of the selected route, and the Angles / Spans / Crossings map toggles.
test("design check lists crossings, towers and sections against the limits", async ({ page }) => {
  await page.goto("/runs/BHU-SHI-0003/map");
  const dc = page.getByTestId("design-check");
  await expect(dc).toBeVisible();
  await expect(page.getByTestId("dc-limits")).toContainText("Design span");
  await expect(page.getByTestId("dc-limits")).toContainText("Section");

  // crossings first: every row carries an angle and a status
  const xrows = page.getByTestId("dc-crossings").locator("tbody tr");
  await expect(xrows.first()).toBeVisible();
  await expect(xrows.first()).toContainText("°");
  await expect(xrows.first()).toContainText(/Preferred|Relaxed|Violation|crosses nothing/);

  await page.getByTestId("dc-tab-towers").click();
  const trows = page.getByTestId("dc-towers").locator("tbody tr");
  expect(await trows.count()).toBeGreaterThan(2);
  await expect(trows.first()).toContainText("Gantry");
  await page.getByRole("button", { name: "Angle towers" }).click();
  await expect(trows.nth(1)).toContainText("°");

  await page.getByTestId("dc-tab-rules").click();
  const rrows = page.getByTestId("dc-rules").locator("tbody tr");
  expect(await rrows.count()).toBeGreaterThan(0);
  await expect(page.getByTestId("dc-rules")).toContainText("Crossing");

  await page.getByTestId("dc-tab-sections").click();
  const srows = page.getByTestId("dc-sections").locator("tbody tr");
  expect(await srows.count()).toBeGreaterThan(0);
  await expect(srows.first()).toContainText("km");

  // map toggles
  await expect(page.getByTestId("toggle-crossings")).toHaveAttribute("aria-pressed", "true");
  for (const id of ["toggle-angles", "toggle-spans", "toggle-bends"]) await expect(page.getByTestId(id)).toHaveAttribute("aria-pressed", "false");
  await page.getByTestId("toggle-spans").click();
  await expect(page.getByTestId("toggle-spans")).toHaveAttribute("aria-pressed", "true");
  await page.getByTestId("toggle-crossings").click();
  await expect(page.getByTestId("toggle-crossings")).toHaveAttribute("aria-pressed", "false");
});
