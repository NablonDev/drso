import { expect, test } from "@playwright/test";
import { createDraft, deleteRun } from "./helpers";

let runId: string | null = null;

test.beforeAll(async ({ request }) => {
  runId = await createDraft(request);
});

test.afterAll(async ({ request }) => {
  await deleteRun(request, runId);
});

test("rule dialog requires a reason and nothing is saved without one", async ({ page, request }) => {
  await page.goto(`/runs/${runId}/rules`);
  await expect(page.getByRole("heading", { name: "Rules", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Change Buildings / manmade structure" }).click();
  const dlg = page.locator("#ruleDlg");
  await expect(dlg).toBeVisible();
  await expect(dlg.getByRole("heading", { name: "Buildings / manmade structure" })).toBeVisible();
  const setback = dlg.getByLabel("Tower setback", { exact: true });
  await setback.fill("90");
  await expect(dlg.getByRole("button", { name: "Save 1 change" })).toBeVisible();
  await dlg.getByRole("button", { name: "Save 1 change" }).click();
  await expect(dlg.getByTestId("rule-error")).toContainText("A reason is required");
  await dlg.getByRole("button", { name: "Cancel" }).click();
  await expect(dlg).toBeHidden();
  const rules = await (await request.get(`/api/runs/${runId}/rules`)).json();
  expect(rules.n_changed).toBe(0);
});

test("costs: a tower price saves and resets", async ({ page }) => {
  await page.goto(`/runs/${runId}/costs`);
  await expect(page.getByTestId("tower-price-table")).toBeVisible();
  const rows = page.getByTestId("tower-price-table").locator("tbody tr");
  await expect(rows).toHaveCount(5);
  const price = page.locator("#price-D60");
  const dflt = await price.inputValue();
  await price.fill("1.5");
  await price.press("Enter");
  await expect.poll(async () => (await (await page.request.get(`/api/runs/${runId}/costs`)).json()).towers[4].price_cr).toBe(1.5);
  // reset to the default so the draft is clean
  await price.fill(dflt);
  await price.press("Enter");
  await expect.poll(async () => (await (await page.request.get(`/api/runs/${runId}/costs`)).json()).towers[4].changed).toBe(false);
});

test("a GPU-busy 409 is shown prominently and keeps the user on the screen", async ({ page }) => {
  await page.route(`**/api/runs/${runId}/extract`, (route) =>
    route.fulfill({
      status: 409,
      contentType: "application/json",
      body: JSON.stringify({
        detail: "Route solve is running for BHU-SHI-0003 (Solving route 2). Nothing is queued — start a LULC extraction once it finishes.",
        busy: true,
        job: { id: 99, run_id: "BHU-SHI-0003", kind: "solve", status: "running", progress: 0.4, stage: "Solving route 2" },
      }),
    }),
  );
  await page.goto(`/runs/${runId}/review`);
  await page.getByTestId("start-extract").click();
  await expect(page.getByTestId("toast")).toContainText("Nothing is queued");
  await expect(page.getByTestId("toast")).toContainText("The GPU is busy");
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/review$`));
  await expect(page.getByRole("alert").filter({ hasText: "Route solve is running" }).first()).toBeVisible();
});
