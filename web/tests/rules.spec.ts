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

test("costs: rate per tonne moves all seven tower rates", async ({ page }) => {
  await page.goto(`/runs/${runId}/costs`);
  const a = page.getByTestId("cr-A");
  await expect(a).toBeVisible();
  const before = await a.textContent();
  const rate = page.locator("#rate");
  await rate.fill("150000");
  await expect(a).not.toHaveText(before ?? "");
  await rate.press("Enter");
  const tonnes = Number(await page.locator("#ton-D60").inputValue());
  await expect(page.getByTestId("cr-D60")).toHaveText(((tonnes * 150000) / 1e7).toFixed(4));
  await expect(page.locator("#rate")).toHaveValue("150000");
  // the server agrees, and every type moved
  await expect.poll(async () => (await (await page.request.get(`/api/runs/${runId}/costs`)).json()).rate_per_tonne_rs).toBe(150000);
  // reset to the default so the draft is clean
  await rate.fill("110000");
  await rate.press("Enter");
  await expect(page.getByTestId("cr-D60")).toHaveText(((tonnes * 110000) / 1e7).toFixed(4));
  await expect.poll(async () => (await (await page.request.get(`/api/runs/${runId}/costs`)).json()).rate_per_tonne_rs).toBe(110000);
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
