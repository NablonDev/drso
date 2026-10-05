import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutationState, useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useConfig, useLayers } from "../api/hooks";
import type { Imagery, ProcurementResponse, RunDetail } from "../api/types";
import ImageryCatalogPanel from "../components/ImageryCatalog";
import { Btn, ErrorNote, Head, Note, Spinner } from "../components/ui";
import { date, int, num, when } from "../lib/format";
import { useTicker } from "../lib/useDebounced";

const PRODUCTS = [
  { k: "Pléiades Neo, new tasking", res: "30 cm" },
  { k: "SkySat archive", res: "50 cm" },
  { k: "AESL drone survey request", res: "10 cm" },
];

function usedFor(imagery: string, layers: { imagery?: string; name: string }[]) {
  return layers.filter((l) => l.imagery === imagery).map((l) => l.name.split(" — ")[0]);
}

export default function Images() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const cfg = useConfig().data;
  const area = useLayers(run.id).data?.band_km2;
  const mkey = useMemo(() => ["imagery", run.id], [run.id]);

  const check = useApiMutation<Imagery, void>({
    mutationKey: mkey,
    mutationFn: () => apiCalls.imageryCheck(run.id),
    errorTitle: "Imagery check failed",
    onSuccess: (img) => {
      qc.setQueryData<RunDetail>(qk.run(run.id), (r) => (r ? { ...r, imagery: img } : r));
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
    },
  });
  // a check started earlier (then left) is still visible when coming back
  const pendingElsewhere = useMutationState({
    filters: { mutationKey: mkey, status: "pending" },
    select: (m) => m.state.submittedAt,
  });
  const running = check.isPending || pendingElsewhere.length > 0;
  const startedAt = pendingElsewhere[0] ?? (check.isPending ? check.submittedAt : 0);
  useTicker(running, 1000);
  const elapsed = running && startedAt ? Math.round((Date.now() - startedAt) / 1000) : 0;

  const auto = useRef(false);
  useEffect(() => {
    if (!run.imagery && !running && !auto.current) {
      auto.current = true;
      check.mutate();
    }
  }, [run.imagery, running, check]);

  const [product, setProduct] = useState(PRODUCTS[0].k);
  const [chargeTo, setChargeTo] = useState("");
  const [proc, setProc] = useState<ProcurementResponse | null>(null);
  const procure = useApiMutation<ProcurementResponse, void>({
    mutationFn: () => apiCalls.procure(run.id, product, chargeTo.trim() || null, null),
    errorTitle: "Procurement request not recorded",
    onSuccess: setProc,
  });

  const [accept, setAccept] = useState(Boolean(run.accept_partial));
  const partial = useApiMutation<RunDetail, void>({
    mutationFn: () => apiCalls.patchRun(run.id, { accept_partial: true }),
    errorTitle: "Could not record the choice",
    onSuccess: (r) => {
      qc.setQueryData(qk.run(r.id), r);
      nav(`/runs/${run.id}/sources`);
    },
  });

  const img = run.imagery;
  const esri = img?.esri;
  const s2 = img?.sentinel2;
  const dem = img?.dem;
  const covered = Boolean(img?.covered);
  const layers = cfg?.layers ?? [];
  const esriLayers = usedFor("esri", layers);
  const s2Layers = usedFor("sentinel2", layers);
  const bandArea = area ?? null;

  const productsTable = img && !img.error ? (
    <div className="card">
      <div className="ch">
        <h3>Imagery over this run</h3>
        <span className="hint" style={{ marginLeft: "auto" }}>checked {when(img.checked_at)}</span>
      </div>
      <div className="tw">
        <table data-testid="imagery-table">
          <thead>
            <tr>
              <th>Product</th>
              <th>Resolution</th>
              <th>Captured</th>
              <th>Cloud</th>
              <th>Coverage</th>
              <th style={{ textAlign: "right" }}>Used for</th>
            </tr>
          </thead>
          <tbody>
            {esri ? (
              <tr>
                <td>
                  <b>{esri.product}</b>
                  <div className="hint">{esri.sources.join(", ")} · {esri.samples} samples · rule {esri.rule}</div>
                </td>
                <td className="mono">{esri.resolutions_m.length ? `${esri.resolutions_m.map((r) => num(r, 2)).join(" / ")} m` : "—"}</td>
                <td className="mono">
                  {date(esri.newest)}
                  {esri.oldest && esri.oldest !== esri.newest ? <div className="hint">oldest {date(esri.oldest)}</div> : null}
                </td>
                <td className="mono">—</td>
                <td>
                  <span className={`pill ${esri.covered_pct >= 99.5 ? "p-go" : "p-cross"}`}>{num(esri.covered_pct, 1)}%</span>
                </td>
                <td style={{ textAlign: "right" }}>{esriLayers.join(", ") || "—"}</td>
              </tr>
            ) : null}
            {s2 ? (
              <tr>
                <td>
                  <b>{s2.product}</b>
                  <div className="hint">
                    {s2.scenes.length
                      ? s2.scenes.map((s) => `${date(s.date)} · ${num(s.cloud, 0)}% cloud`).join(" · ")
                      : "no scenes found"}
                  </div>
                </td>
                <td className="mono">{s2.resolution_m} m</td>
                <td className="mono">{date(s2.newest)}</td>
                <td className="mono">{s2.cloud === null ? "—" : `${num(s2.cloud, 0)}%`}</td>
                <td>
                  <span className={`pill ${s2.covered_pct >= 99.5 ? "p-go" : "p-cross"}`}>{num(s2.covered_pct, 1)}%</span>
                </td>
                <td style={{ textAlign: "right" }}>{s2Layers.join(", ") || "—"}</td>
              </tr>
            ) : null}
            {dem ? (
              <tr>
                <td>
                  <b>{dem.product}</b>
                  <div className="hint">{dem.tiles.length} tile{dem.tiles.length === 1 ? "" : "s"} · {dem.note ?? ""}</div>
                </td>
                <td className="mono">{dem.resolution_m} m</td>
                <td className="mono">—</td>
                <td className="mono">—</td>
                <td>
                  <span className={`pill ${dem.covered_pct >= 99.5 ? "p-go" : "p-cross"}`}>{num(dem.covered_pct, 1)}%</span>
                </td>
                <td style={{ textAlign: "right" }}>Terrain class, ground clearance</td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  ) : null;

  return (
    <section>
      <Head
        title="Imagery over the band"
        right={
          <Btn ghost pending={running} onClick={() => check.mutate()}>
            {img ? "Check again" : "Check imagery"}
          </Btn>
        }
      >
        Every vision model reads imagery. Below: what exists online over the band, and what is already stored on this machine —
        the models use only the stored imagery, so fetch (under our licences) or register what they need before Fetch LULCs.
      </Head>

      {running ? (
        <div className="card" style={{ marginBottom: 14, borderColor: "var(--accent-line)" }} data-testid="imagery-progress">
          <div className="cb row" style={{ gap: 12 }}>
            <Spinner />
            <div style={{ flex: 1 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Checking Esri, Sentinel-2 and the DEM over the band…</div>
              <div className="hint">
                {elapsed}s · this usually takes 10–40 seconds. You can leave this screen — the result is saved to the run.
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <ErrorNote error={check.error} retry={() => check.mutate()} />

      <ImageryCatalogPanel runId={run.id} busy={Boolean(run.job)} />
      {img?.error ? <Note kind="stop" style={{ marginBottom: 14 }}>The last check failed: {img.error}</Note> : null}

      {!img && !running && !check.error ? <p className="hint">Not checked yet.</p> : null}

      {img && !img.error && covered ? (
        <div>
          <Note kind="ok" style={{ marginBottom: 14 }}>
            <b>The whole {bandArea ? `${int(bandArea)} km² ` : ""}band is covered.</b> Nothing to do — the imagery below is attached to
            the run and the next screen is already populated.
          </Note>
          {productsTable}
          <div className="actions">
            <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/corridor`)}>Back</button>
            <button className="btn" onClick={() => nav(`/runs/${run.id}/sources`)}>Next: data sources</button>
          </div>
        </div>
      ) : null}

      {img && !img.error && !covered ? (
        <div>
          <Note kind="stop" style={{ marginBottom: 14 }}>
            <b>
              {num(esri?.covered_pct ?? 0, 1)}% of the band is covered. {num(100 - (esri?.covered_pct ?? 0), 1)}% is not.
            </b>{" "}
            {esri?.gap_ranges_km.length
              ? `The gap runs ${esri.gap_ranges_km.map(([a, b]) => `from km ${num(a, 1)} to km ${num(b, 1)}`).join(", ")} — no usable high-resolution capture (${esri.rule}).`
              : `No usable high-resolution capture over part of the band (${esri?.rule ?? "rule"}).`}{" "}
            The {esriLayers.join(", ").toLowerCase() || "high-resolution"} models cannot run there, so the route would be solved half blind.
          </Note>
          {accept || run.accept_partial ? (
            run.accept_partial ? (
              <Note kind="ok" style={{ marginBottom: 14 }}>
                <b>Partial coverage accepted.</b> The gap is treated as unclassified ground and every affected kilometre is listed on
                Why this route.
              </Note>
            ) : null
          ) : null}

          <div className="grid2">
            <div className="card">
              <div className="ch"><h3>What is missing</h3></div>
              <div className="cb">
                <div className="tw">
                  <table className="kv">
                    <tbody>
                      <tr><td>Gap area</td><td className="mono">{num(esri?.gap_km2 ?? 0, 0)} km²{bandArea ? ` of ${int(bandArea)}` : ""}</td></tr>
                      <tr><td>Chainage</td><td className="mono">{esri?.gap_ranges_km.map(([a, b]) => `km ${num(a, 1)} – ${num(b, 1)}`).join(", ") || "—"}</td></tr>
                      <tr><td>Needed at</td><td className="mono">{esri?.rule ?? "—"}</td></tr>
                      <tr><td>Blocks</td><td>{esriLayers.join(" · ") || "—"}<div className="hint">{esriLayers.length} layers read this imagery</div></td></tr>
                      <tr><td>Covered by</td><td>{s2 ? `${s2.product} at ${s2.resolution_m} m only` : "nothing"}<div className="hint">Enough for LULC and water, not for footprints</div></td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="card" style={{ borderColor: "var(--accent-line)" }}>
              <div className="ch">
                <h3>Procure the missing tiles</h3>
                <span className="pill p-lock" style={{ marginLeft: "auto" }}>stub in this release</span>
              </div>
              <div className="cb stack">
                <div className="tw">
                  <table>
                    <thead><tr><th>Product</th><th>Res.</th></tr></thead>
                    <tbody>
                      {PRODUCTS.map((p) => (
                        <tr key={p.k}>
                          <td>
                            <label className="row" style={{ gap: 8, cursor: "pointer" }}>
                              <input type="radio" name="proc" checked={product === p.k} onChange={() => setProduct(p.k)} style={{ width: "auto" }} />
                              {product === p.k ? <b>{p.k}</b> : p.k}
                            </label>
                          </td>
                          <td className="mono">{p.res}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <label className="fld">
                  <span>Charge to</span>
                  <input type="text" value={chargeTo} onChange={(e) => setChargeTo(e.target.value)} placeholder="AESL / project / WBS" />
                </label>
                <Btn pending={procure.isPending} onClick={() => procure.mutate()}>Raise the procurement request</Btn>
                {proc ? (
                  <Note kind="ok">
                    <b>Request {proc.id} recorded.</b> {proc.message}
                  </Note>
                ) : null}
              </div>
            </div>
          </div>

          <div style={{ marginTop: 14 }}>{productsTable}</div>

          <div className="grid2" style={{ marginTop: 14 }}>
            <div className="card">
              <div className="ch"><h3>How procurement works</h3></div>
              <div className="cb stack">
                <div className="tw">
                  <table>
                    <tbody>
                      <tr><td><span className="pill p-acc">1</span></td><td>Request raised here, with the gap polygon attached</td></tr>
                      <tr><td><span className="pill p-acc">2</span></td><td>GIS cell confirms the product and the budget line</td></tr>
                      <tr><td><span className="pill p-acc">3</span></td><td>Vendor tasks the satellite and delivers to the bucket</td></tr>
                      <tr><td><span className="pill p-acc">4</span></td><td>Tiles attach to the run; check the imagery again here</td></tr>
                    </tbody>
                  </table>
                </div>
                <div className="hint">This release records the request only — nothing is sent to a vendor.</div>
              </div>
            </div>
            <div className="card">
              <div className="ch"><h3>Or go ahead without it</h3></div>
              <div className="cb stack">
                <Note kind="warn">
                  <b>Allowed, and recorded.</b> The route can be solved over the {num(esri?.covered_pct ?? 0, 1)}% that is covered, with
                  the gap treated as unclassified ground at 10 m. The {esriLayers.length} high-resolution layers will be blank there,
                  their rules will not fire, and every affected kilometre is listed on <b>Why this route</b>.
                </Note>
                <label className="row" style={{ gap: 9, cursor: "pointer" }}>
                  <input type="checkbox" style={{ width: "auto" }} checked={accept || Boolean(run.accept_partial)}
                    disabled={Boolean(run.accept_partial)} onChange={(e) => setAccept(e.target.checked)} />
                  <span style={{ fontSize: 13 }}>I accept a route solved over partial imagery</span>
                </label>
                {run.accept_partial ? (
                  <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/sources`)}>Continue to data sources</button>
                ) : (
                  <Btn ghost pending={partial.isPending} disabled={!accept} onClick={() => partial.mutate()}>
                    Continue with {num(esri?.covered_pct ?? 0, 0)}% coverage
                  </Btn>
                )}
                <ErrorNote error={partial.error} />
              </div>
            </div>
          </div>
          <div className="actions">
            <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/corridor`)}>Back</button>
            <button className="btn ghost" onClick={() => nav("/")}>Park the run and come back</button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
