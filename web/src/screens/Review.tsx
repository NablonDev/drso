import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useCosts, useLayers, useRules } from "../api/hooks";
import type { Job, RunDetail } from "../api/types";
import { Btn, ErrorNote, Head, Loading, Note } from "../components/ui";
import { clock, corridorName, date, fmtLatLon, hm, int, lineSpec, num, seconds } from "../lib/format";
import { useUi } from "../store/ui";

export default function Review() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const lq = useLayers(run.id);
  const rules = useRules(run.id).data;
  const costs = useCosts(run.id).data;
  const names = useUi((s) => s.uploadNames);

  const routes = useApiMutation<RunDetail, number>({
    mutationFn: (n) => apiCalls.patchRun(run.id, { routes_wanted: n }),
    errorTitle: "Routes wanted not changed",
    onSuccess: (r) => qc.setQueryData(qk.run(r.id), r),
  });
  const [forceModels, setForceModels] = useState(false);
  const extract = useApiMutation<Job, void>({
    mutationFn: () => apiCalls.extract(run.id, undefined, forceModels),
    errorTitle: "Extraction not started",
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.active });
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
      void qc.invalidateQueries({ queryKey: qk.runs });
      nav(`/runs/${run.id}/extracting`);
    },
  });

  if (lq.isPending) return <Loading />;
  if (lq.isError) return <ErrorNote error={lq.error} retry={() => void lq.refetch()} />;
  const L = lq.data;
  const anyExtracted = L.layers.some((l) => l.extracted);
  const todo = L.layers.filter((l) => !l.extracted || l.stale || l.status === "failed");
  const minutes = todo.reduce((a, l) => a + (l.source_setting.mode === "replace" ? 0 : (l.minutes_per_km2 ?? 0) * L.band_km2), 0);
  const custom = L.layers.filter((l) => l.source_setting.mode !== "auto");
  const ovr = (rules?.groups ?? []).flatMap((g) => g.rules.filter((r) => r.n_changed > 0 || !r.enabled).map((r) => ({ r, statutory: g.statutory })));
  const tcc = costs?.categories.filter((c) => c.status === "tcc") ?? [];
  const costChanged = costs?.categories.reduce((a, c) => a + c.items.filter((i) => i.changed).length, 0) ?? 0;
  const img = run.imagery;
  const finish = new Date(Date.now() + minutes * 60000).toISOString();
  const lastSolveS =
    run.last_job?.kind === "solve" && run.last_job.started_at && run.last_job.finished_at
      ? (new Date(run.last_job.finished_at).getTime() - new Date(run.last_job.started_at).getTime()) / 1000
      : null;
  const nothingToDo = anyExtracted && todo.length === 0;

  return (
    <section>
      <Head title="Review & extract">
        This starts step 1 only — the layers. The route is not solved until you have looked at what came back.
      </Head>

      <div className="grid2">
        <div className="card">
          <div className="ch">
            <h3>Corridor</h3>
            <button className="mini" style={{ marginLeft: "auto" }} onClick={() => nav(`/runs/${run.id}/corridor`)}>Edit</button>
          </div>
          <div className="cb">
            <div className="tw">
              <table className="kv">
                <tbody>
                  <tr><td>Corridor</td><td><b>{corridorName(run)}</b></td></tr>
                  <tr><td>From</td><td>{run.a_name || "Point A"}<div className="hint mono">{fmtLatLon(run.a_lat, run.a_lon)}</div></td></tr>
                  <tr><td>To</td><td>{run.b_name || "Point B"}<div className="hint mono">{fmtLatLon(run.b_lat, run.b_lon)}</div></td></tr>
                  <tr><td>Line</td><td>{lineSpec(run.kv, run.circuit)}</td></tr>
                  <tr>
                    <td>Band</td>
                    <td className="mono">
                      ± {num(run.band_km, 0)} km · {int(L.band_km2)} km²
                      {run.reuse_from ? <div className="hint">layers reused from {run.reuse_from}</div> : null}
                      {run.parent_id ? <div className="hint">layers carried from {run.parent_id}</div> : null}
                    </td>
                  </tr>
                  <tr>
                    <td>Imagery</td>
                    <td>
                      {!img ? (
                        <span>Not checked yet <button className="mini" onClick={() => nav(`/runs/${run.id}/images`)}>Check</button></span>
                      ) : img.error ? (
                        <span style={{ color: "var(--nogo)" }}>Check failed</span>
                      ) : img.covered ? (
                        "Covered"
                      ) : (
                        <span>
                          {num(img.esri?.covered_pct ?? 0, 1)}% covered{run.accept_partial ? " · partial accepted" : ""}
                        </span>
                      )}
                      {img && !img.error ? (
                        <div className="hint">
                          Esri {date(img.esri?.newest)} · Sentinel-2 {date(img.sentinel2?.newest)} · {img.dem?.product ?? "DEM"}
                        </div>
                      ) : null}
                    </td>
                  </tr>
                  <tr>
                    <td>Routes wanted</td>
                    <td>
                      <div className="seg" role="group" aria-label="Routes wanted">
                        {[1, 2, 3, 4].map((n) => (
                          <button key={n} aria-pressed={(routes.isPending ? routes.variables : run.routes_wanted) === n}
                            disabled={routes.isPending || Boolean(run.job)} onClick={() => routes.mutate(n)}>
                            {n}
                          </button>
                        ))}
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="ch"><h3>Departures from default</h3></div>
          <div className="cb">
            <div className="tw">
              <table data-testid="departures">
                <tbody>
                  {custom.map((l) => (
                    <tr key={l.key}>
                      <td><span className={`pill ${l.source_setting.mode === "replace" ? "p-chg" : "p-acc"}`}>Source</span></td>
                      <td>
                        {l.name} — <b>{l.source_setting.mode === "replace" ? "replaced" : "supplemented"}</b>
                        <div className="hint mono">
                          {l.source_setting.upload_id ? names[l.source_setting.upload_id] ?? l.file_name ?? `upload #${l.source_setting.upload_id}` : "no file attached"}
                        </div>
                      </td>
                    </tr>
                  ))}
                  {ovr.length ? (
                    ovr.map(({ r, statutory }) => (
                      <tr key={r.id}>
                        <td><span className={`pill ${statutory ? "p-lock" : "p-chg"}`}>Rule</span></td>
                        <td>
                          <b>{r.name}</b> — {r.enabled ? `${r.n_changed} field${r.n_changed > 1 ? "s" : ""} changed` : "switched off"}
                          <div className="hint">{r.reason || "no reason recorded"}</div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td><span className="pill p-chg">Rule</span></td>
                      <td>
                        No rule has been changed — the set is at defaults
                        <div className="hint">Change a field on the Rules screen and it appears here with the reason you gave.</div>
                      </td>
                    </tr>
                  )}
                  <tr>
                    <td><span className="pill p-cross">Cost</span></td>
                    <td>
                      {tcc.length} of {costs?.categories.length ?? 8} cost categories are still on draft rates
                      {costChanged ? ` · ${costChanged} rate${costChanged > 1 ? "s" : ""} changed from draft` : ""}
                      <div className="hint">Awaiting TCC sign-off — the run proceeds, the report carries the notice</div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch"><h3>Two steps, with you in between</h3></div>
        <div className="cb">
          <div className="tw">
            <table>
              <tbody>
                <tr>
                  <td style={{ width: 30 }}><span className="pill p-acc">1</span></td>
                  <td>
                    <b>
                      Extract {todo.length} layer{todo.length === 1 ? "" : "s"}
                    </b>{" "}
                    over {int(L.band_km2)} km² — {todo.filter((l) => l.source_setting.mode === "auto").length} auto-fetched,{" "}
                    {todo.filter((l) => l.source_setting.mode === "replace").length} from your files,{" "}
                    {todo.filter((l) => l.source_setting.mode === "add").length} supplemented
                    {anyExtracted ? <div className="hint">{L.layers.length - todo.length} layers are already extracted and are kept.</div> : null}
                    <div className="hint">Unattended. You can close the tab.</div>
                  </td>
                  <td className="mono" style={{ textAlign: "right" }}>{todo.length ? hm(minutes) : "—"}</td>
                </tr>
                <tr style={{ background: "var(--surface-2)" }}>
                  <td><span className="pill p-cross">⏸</span></td>
                  <td>
                    <b>You review the layers</b> — check what the models found, add what is missing, then release step 2
                    <div className="hint">The route does not solve on its own.</div>
                  </td>
                  <td className="mono" style={{ textAlign: "right" }}>—</td>
                </tr>
                <tr>
                  <td><span className="pill p-acc">2</span></td>
                  <td>
                    <b>Solve the route</b> — cost surface, {rules ? rules.n_rules - rules.n_off : "all"} rules, {run.routes_wanted} alignment
                    {run.routes_wanted > 1 ? "s" : ""}, towers, decision log
                    <div className="hint">Re-runnable on the same layers as often as you like</div>
                  </td>
                  <td className="mono" style={{ textAlign: "right" }}>{lastSolveS ? seconds(lastSolveS) : "about 1m"}</td>
                </tr>
              </tbody>
            </table>
          </div>
          {img && !img.error && !img.covered && !run.accept_partial ? (
            <Note kind="warn" style={{ marginTop: 12 }}>
              The imagery check found a gap and partial coverage has not been accepted on the <b>Images</b> screen. Extraction can still
              run; the high-resolution layers will be blank over the gap.
            </Note>
          ) : null}
          {nothingToDo ? (
            <Note kind="ok" style={{ marginTop: 12 }}>
              Every layer is already extracted and current. Go straight to the layer review, or change a source to re-extract it.
            </Note>
          ) : null}
          <ErrorNote error={extract.error} />
          <div className="row" style={{ justifyContent: "space-between", marginTop: 14, gap: 12 }}>
            <div className="hint" style={{ maxWidth: "46ch", margin: 0 }}>
              One GPU job runs at a time. If a solve or another extraction is running, this is refused and nothing is queued.
            </div>
            {nothingToDo ? (
              <button className="btn" onClick={() => nav(`/runs/${run.id}/layers`)}>Review the layers →</button>
            ) : (
              <>
                <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
                  <input type="checkbox" checked={forceModels} onChange={(e) => setForceModels(e.target.checked)}
                    data-testid="force-models" />
                  Re-run the LULC models even where a stored output covers the band
                </label>
                <Btn pending={extract.isPending} disabled={Boolean(run.job)} onClick={() => extract.mutate()} data-testid="start-extract">
                  Fetch LULCs · layers ready about {clock(finish)}
                </Btn>
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
