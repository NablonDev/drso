import { expect, type APIRequestContext, type Page } from "@playwright/test";

export const TEST_CORRIDOR = {
  name: "PW test corridor",
  a: { lat: 23.4525, lon: 69.5924, name: "PW test A" },
  b: { lat: 23.3932, lon: 70.5955, name: "PW test B" },
  kv: 400,
  circuit: "dc",
  band_km: 5,
};

/** Create a throwaway draft run (no layers) through the API. */
export async function createDraft(request: APIRequestContext, extra: Record<string, unknown> = {}): Promise<string> {
  const r = await request.post("/api/runs", { data: { ...TEST_CORRIDOR, note: "playwright", ...extra } });
  expect(r.status(), await r.text()).toBe(201);
  return (await r.json()).id as string;
}

export async function deleteRun(request: APIRequestContext, id: string | null | undefined) {
  if (!id) return;
  await request.delete(`/api/runs/${id}`);
}

/** Collect console errors and page errors; network failures from known-broken endpoints are filtered by the caller. */
export function watchConsole(page: Page) {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  return errors;
}

/** Errors other than HTTP resource failures (tile 500s are a known backend bug, reported separately). */
export function appErrors(errors: string[]) {
  return errors.filter((e) => !/Failed to load resource/.test(e));
}
