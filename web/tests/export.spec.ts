import { expect, test } from "@playwright/test";

test("export lists the solve files and a download link returns 200", async ({ page, request }) => {
  await page.goto("/runs/BHU-SHI-0003/export");
  const rows = page.getByTestId("files-table").locator("tbody tr");
  await expect(rows.first()).toBeVisible();
  expect(await rows.count()).toBeGreaterThan(5);
  await expect(page.getByTestId("download-all")).toBeVisible();
  const href = await page.getByTestId("file-link").first().getAttribute("href");
  expect(href).toMatch(/^\/api\/solves\/\d+\/files\//);
  const r = await request.get(href!);
  expect(r.status()).toBe(200);
  expect((await r.body()).length).toBeGreaterThan(0);
  const all = await page.getByTestId("download-all").getAttribute("href");
  const z = await request.head(all!).catch(() => null);
  if (z && z.status() !== 405) expect(z.status()).toBe(200);
});
