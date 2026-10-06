import { expect, test } from "@playwright/test";
import { deleteRun } from "./helpers";

let created: string | null = null;
test.afterAll(async ({ request }) => {
  await deleteRun(request, created);
});

test("new route: DMS paste converts, geo-validation runs, the run is created", async ({ page }) => {
  await page.goto("/new");
  await page.locator("#lat1").focus();
  await page.evaluate(() => {
    const dt = new DataTransfer();
    dt.setData("text", `23°27'9.0"N 69°35'32.6"E`);
    document.querySelector("#lat1")!.dispatchEvent(new ClipboardEvent("paste", { clipboardData: dt, bubbles: true, cancelable: true }));
  });
  await expect(page.locator("#lat1")).toHaveValue("23.45250");
  await expect(page.locator("#lon1")).toHaveValue(/69\.592/);
  await page.locator("#lat2").fill("23.3932");
  await page.locator("#lon2").fill("70.5955");
  await expect(page.getByTestId("geo-checks")).toContainText("Both points fall inside India");
  // name is mandatory, circuit is not asked, the band starts at ± 25 km
  await expect(page.getByTestId("corridor-next")).toBeDisabled();
  await expect(page.locator("#circuit")).toHaveCount(0);
  await expect(page.locator("#bandLbl")).toHaveText("± 25 km");
  await page.getByTestId("corridor-name-input").fill("PW corridor Bhuj Shivlaka");
  await expect(page.getByTestId("reuse-table")).toBeVisible();
  // do not reuse — this keeps the run a plain draft
  for (const cb of await page.getByTestId("reuse-table").locator("input[type=checkbox]:checked").all()) await cb.uncheck();
  await page.getByTestId("corridor-next").click();
  await expect(page).toHaveURL(/\/runs\/PWX-COR-\d+\/images$/);       // id from the corridor name
  created = page.url().match(/runs\/([^/]+)\//)![1];
  await expect(page.getByTestId("corridor-name")).toHaveText("PW corridor Bhuj Shivlaka");
  // imagery check starts by itself and shows progress
  await expect(page.getByTestId("imagery-progress").or(page.getByTestId("imagery-table"))).toBeVisible();
});
