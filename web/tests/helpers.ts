import { expect, type APIRequestContext, type Page } from "@playwright/test";

export const TEST_CORRIDOR = {
  name: "PW test corridor",
  a: { lat: 23.456596, lon: 69.591253, name: "PW test A" },   // gantry points moved 8–9 Oct 2026
  b: { lat: 23.394667, lon: 70.594966, name: "PW test B" },
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

/** Newest Bhujpool–Shivlaka run with the test corridor's end points whose layers are complete under the current layer set — the run the
 *  tests reuse layers from (a run made before a layer was added is not solve-ready). */
export async function reuseSource(request: APIRequestContext): Promise<string> {
  const runs = (await (await request.get("/api/runs?limit=200")).json()).runs as { id: string; family: string }[];
  for (const r of runs.filter((x) => x.family === "BHU-SHI")) {
    const run = await (await request.get(`/api/runs/${r.id}`)).json();
    // same end points as the test corridor (runs with a moved gantry do not cover its band)
    const same = (p: { lat: number; lon: number }, q: { lat: number; lon: number }) =>
      Math.abs(p.lat - q.lat) < 1e-6 && Math.abs(p.lon - q.lon) < 1e-6;
    if (run.layers_ready && same(run.a, TEST_CORRIDOR.a) && same(run.b, TEST_CORRIDOR.b)) return r.id;
  }
  throw new Error("no solve-ready BHU-SHI run: create 'Bhujpool Shivlaka' (±5 km) and extract its layers");
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
