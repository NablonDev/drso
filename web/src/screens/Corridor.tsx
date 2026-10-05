import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { apiCalls, qk, useApiMutation, useConfig, useGeoBand, useGeoValidate, useHealth, useRun, type RunCreateBody } from "../api/hooks";
import type { Circuit, Pt, RunDetail } from "../api/types";
import { hasLayers } from "../components/Shell";
import { Btn, ErrorNote, Head, Kpi, Loading, Note, Spinner } from "../components/ui";
import { fmtDec, parseCoord, parsePair } from "../lib/dms";
import { CIRCUIT_LONG, date, fold, hm, int, num } from "../lib/format";
import { useDebounced } from "../lib/useDebounced";

interface Form {
  lat1: string;
  lon1: string;
  nm1: string;
  lat2: string;
  lon2: string;
  nm2: string;
  kv: number;
  circuit: Circuit;
  band: number;
}

const EMPTY: Form = { lat1: "", lon1: "", nm1: "", lat2: "", lon2: "", nm2: "", kv: 765, circuit: "dc", band: 5 };

function fromRun(r: RunDetail): Form {
  return {
    lat1: fmtDec(r.a_lat),
    lon1: fmtDec(r.a_lon),
    nm1: r.a_name ?? "",
    lat2: fmtDec(r.b_lat),
    lon2: fmtDec(r.b_lon),
    nm2: r.b_name ?? "",
    kv: r.kv,
    circuit: r.circuit,
    band: Math.min(15, Math.max(1, Math.round(r.band_km))),
  };
}

function coordErr(v: string, max: number): string | null {
  if (!v.trim()) return null;
  const n = parseCoord(v);
  if (n === null) return "Not a coordinate — use decimal degrees or 22°18'25.9\"N";
  if (Math.abs(n) > max) return `Must be between −${max} and ${max}`;
  return null;
}

function toPt(lat: string, lon: string, name: string): Pt | null {
  const la = parseCoord(lat);
  const lo = parseCoord(lon);
  if (la === null || lo === null || Math.abs(la) > 90 || Math.abs(lo) > 180) return null;
  return { lat: la, lon: lo, name: name.trim() || null };
}

function CoordInput({
  id,
  label,
  value,
  onChange,
  onPair,
  max,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  onPair: (lat: number, lon: number) => void;
  max: number;
}) {
  const err = coordErr(value, max);
  return (
    <label className="fld" style={{ flex: 1, minWidth: 0 }}>
      <span>{label}</span>
      <input
        id={id}
        type="text"
        className="mono"
        inputMode="decimal"
        value={value}
        aria-invalid={Boolean(err)}
        onChange={(e) => {
          const pair = parsePair(e.target.value);
          if (pair && /[°'"NSEW,]/i.test(e.target.value)) onPair(pair[0], pair[1]);
          else onChange(e.target.value);
        }}
        onPaste={(e) => {
          const t = e.clipboardData.getData("text");
          const pair = parsePair(t);
          if (pair) {
            e.preventDefault();
            onPair(pair[0], pair[1]);
          }
        }}
        onBlur={() => {
          const n = parseCoord(value);
          if (n !== null && !/^-?\d+(\.\d+)?$/.test(value.trim())) onChange(fmtDec(n));
        }}
      />
      {err ? <div className="err">{err}</div> : null}
    </label>
  );
}

export default function Corridor() {
  const { runId } = useParams();
  const runQ = useRun(runId);
  const run = runQ.data;
  const cfg = useConfig().data;
  const health = useHealth().data;
  const nav = useNavigate();
  const qc = useQueryClient();
  const [f, setF] = useState<Form>(EMPTY);
  const [reuse, setReuse] = useState<string | null>(null);
  const reuseTouched = useRef(false);
  const loadedFor = useRef<string | null>(null);

  useEffect(() => {
    if (run && loadedFor.current !== run.id) {
      loadedFor.current = run.id;
      setF(fromRun(run));
    }
    if (!runId && loadedFor.current !== "new") {
      loadedFor.current = "new";
      setF(EMPTY);
    }
  }, [run, runId]);

  const set = <K extends keyof Form>(k: K, v: Form[K]) => setF((s) => ({ ...s, [k]: v }));
  const a = useMemo(() => toPt(f.lat1, f.lon1, f.nm1), [f.lat1, f.lon1, f.nm1]);
  const b = useMemo(() => toPt(f.lat2, f.lon2, f.nm2), [f.lat2, f.lon2, f.nm2]);
  const da = useDebounced(a, 450);
  const db = useDebounced(b, 450);
  const dband = useDebounced(f.band, 250);
  const geo = useGeoValidate(da, db, dband);
  const bandQ = useGeoBand(geo.data?.ok ? da : null, geo.data?.ok ? db : null, dband, runId);
  const locked = run ? hasLayers(run) : false;

  const firstFull = (bandQ.data?.reuse ?? []).find((r) => r.full_cover)?.run_id ?? null;
  // pre-select the best full-cover reuse when creating, until the user chooses
  useEffect(() => {
    if (runId || reuseTouched.current) return;
    setReuse(firstFull);
  }, [runId, firstFull]);

  const dirty = useMemo(() => {
    if (!run) return true;
    const o = fromRun(run);
    return (
      (["nm1", "nm2", "kv", "circuit", "band"] as const).some((k) => o[k] !== f[k]) ||
      parseCoord(o.lat1) !== parseCoord(f.lat1) ||
      parseCoord(o.lon1) !== parseCoord(f.lon1) ||
      parseCoord(o.lat2) !== parseCoord(f.lat2) ||
      parseCoord(o.lon2) !== parseCoord(f.lon2)
    );
  }, [run, f]);

  const create = useApiMutation<RunDetail, RunCreateBody>({
    mutationFn: (body) => apiCalls.createRun(body),
    errorTitle: "Could not create the run",
    onSuccess: (r) => {
      // POST /api/runs returns the bare run row (no layers_ready / last_extract…), so fetch the full record
      void qc.invalidateQueries({ queryKey: qk.runs });
      void qc.invalidateQueries({ queryKey: qk.run(r.id) });
      nav(`/runs/${r.id}/images`);
    },
  });
  const patch = useApiMutation<RunDetail, Partial<RunCreateBody>>({
    mutationFn: (body) => apiCalls.patchRun(runId!, body),
    errorTitle: "Corridor not saved",
    onSuccess: (r) => {
      qc.setQueryData(qk.run(r.id), r);
      void qc.invalidateQueries({ queryKey: qk.runs });
      void qc.invalidateQueries({ queryKey: qk.layers(r.id) });
      nav(`/runs/${r.id}/images`);
    },
  });

  if (runId && runQ.isPending) return <Loading />;

  const canSubmit = Boolean(a && b && geo.data?.ok && !geo.isFetching);
  const submit = () => {
    if (!a || !b) return;
    const body: RunCreateBody = { a, b, kv: f.kv, circuit: f.circuit, band_km: f.band };
    if (!runId) {
      create.mutate({ ...body, routes_wanted: 4, reuse_from: reuse });
    } else if (dirty) {
      patch.mutate(body);
    } else {
      nav(`/runs/${runId}/images`);
    }
  };

  const regionOk =
    geo.data &&
    (geo.data.region_supported ||
      (health?.region && fold(geo.data.a.state) === fold(health.region) && fold(geo.data.b.state) === fold(health.region)));
  const area = bandQ.data?.area_km2;
  const minutes = bandQ.data?.extraction_minutes_full;
  const reuseRow = (bandQ.data?.reuse ?? []).find((r) => r.run_id === (run?.reuse_from ?? reuse));

  return (
    <section>
      <Head title="Where does the line run?">
        Two coordinates, the line spec, and how wide a band to examine. Extraction takes time on the GPU, so the band is the one
        input worth thinking about.
      </Head>

      {locked ? (
        <Note kind="warn" style={{ marginBottom: 14 }}>
          <b>The corridor is locked.</b> This run already holds extracted layers, so its end points, line and band are fixed. In this
          version a copy keeps the layers and stays locked too — widening the band means a new extraction, so start a{" "}
          <b>New route</b> with the same end points. You can still edit here; the server will refuse the change and say why.
        </Note>
      ) : null}

      <div className="grid2">
        <div className="stack">
          <div className="card">
            <div className="ch">
              <h3>Start point</h3>
              <span className="pill p-acc">A</span>
            </div>
            <div className="cb stack">
              <div className="row" style={{ gap: 10, flexWrap: "nowrap", alignItems: "flex-start" }}>
                <CoordInput id="lat1" label="Latitude" value={f.lat1} max={90} onChange={(v) => set("lat1", v)}
                  onPair={(la, lo) => setF((s) => ({ ...s, lat1: fmtDec(la), lon1: fmtDec(lo) }))} />
                <CoordInput id="lon1" label="Longitude" value={f.lon1} max={180} onChange={(v) => set("lon1", v)}
                  onPair={(la, lo) => setF((s) => ({ ...s, lat1: fmtDec(la), lon1: fmtDec(lo) }))} />
              </div>
              <label className="fld">
                <span>Name (optional)</span>
                <input id="nm1" type="text" value={f.nm1} onChange={(e) => set("nm1", e.target.value)} placeholder="Vadodara 400/220 kV S/S" />
              </label>
              <div className="hint">
                Decimal degrees, WGS 84. Paste <span className="mono">22°18'25.9"N 73°10'52.3"E</span> and it converts.
              </div>
            </div>
          </div>
          <div className="card">
            <div className="ch">
              <h3>End point</h3>
              <span className="pill p-acc">B</span>
            </div>
            <div className="cb stack">
              <div className="row" style={{ gap: 10, flexWrap: "nowrap", alignItems: "flex-start" }}>
                <CoordInput id="lat2" label="Latitude" value={f.lat2} max={90} onChange={(v) => set("lat2", v)}
                  onPair={(la, lo) => setF((s) => ({ ...s, lat2: fmtDec(la), lon2: fmtDec(lo) }))} />
                <CoordInput id="lon2" label="Longitude" value={f.lon2} max={180} onChange={(v) => set("lon2", v)}
                  onPair={(la, lo) => setF((s) => ({ ...s, lat2: fmtDec(la), lon2: fmtDec(lo) }))} />
              </div>
              <label className="fld">
                <span>Name (optional)</span>
                <input id="nm2" type="text" value={f.nm2} onChange={(e) => set("nm2", e.target.value)} placeholder="Halol 765 kV S/S" />
              </label>
            </div>
          </div>
          <div className="card">
            <div className="ch">
              <h3>Line</h3>
            </div>
            <div className="cb">
              <div className="row" style={{ gap: 10, flexWrap: "nowrap" }}>
                <label className="fld" style={{ flex: 1 }}>
                  <span>Voltage</span>
                  <select id="kv" value={f.kv} onChange={(e) => set("kv", Number(e.target.value))}>
                    {(cfg?.voltages ?? [765, 400, 220, 132]).map((v) => (
                      <option key={v} value={v}>
                        {v} kV
                      </option>
                    ))}
                  </select>
                </label>
                <label className="fld" style={{ flex: 1 }}>
                  <span>Circuits</span>
                  <select id="circuit" value={f.circuit} onChange={(e) => set("circuit", e.target.value as Circuit)}>
                    {(["dc", "sc", "qc"] as Circuit[]).map((c) => (
                      <option key={c} value={c}>
                        {CIRCUIT_LONG[c]}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="hint" style={{ marginTop: 6 }}>Rules and unit costs resolve per voltage and circuit on the server.</div>
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <div className="ch">
              <h3>Geo-validation</h3>
              {geo.isFetching ? <span style={{ marginLeft: "auto" }}><Spinner label="Checking" /></span> : null}
            </div>
            <div className="cb" style={{ padding: 0 }}>
              {!a || !b ? (
                <p className="hint" style={{ padding: "12px 14px" }}>
                  Enter both points to check them against the state boundaries.
                </p>
              ) : geo.isError ? (
                <div style={{ padding: "0 14px" }}>
                  <ErrorNote error={geo.error} retry={() => void geo.refetch()} />
                </div>
              ) : !geo.data ? (
                <div style={{ padding: "0 14px" }}>
                  <Loading text="Checking the coordinates…" />
                </div>
              ) : (
                <div className="tw">
                  <table data-testid="geo-checks">
                    <tbody>
                      {geo.data.checks.map((c, i) => (
                        <tr key={i}>
                          <td style={{ width: 26 }}>
                            <span className={`pill ${c.ok ? "p-go" : "p-nogo"}`}>{c.ok ? "✓" : "✗"}</span>
                          </td>
                          <td>
                            {c.text}
                            {c.sub ? <div className="hint">{c.sub}</div> : null}
                            {i === geo.data!.checks.length - 1 && c.ok ? (
                              <div className="hint">From the state boundary layer — the only thing known before any imagery is read</div>
                            ) : null}
                          </td>
                        </tr>
                      ))}
                      {geo.data.beeline_km ? (
                        <tr>
                          <td><span className="pill p-acc">↗</span></td>
                          <td>
                            Bee-line <span className="mono">{num(geo.data.beeline_km, 1)} km</span>
                          </td>
                        </tr>
                      ) : null}
                    </tbody>
                  </table>
                </div>
              )}
              {geo.data && geo.data.ok && !regionOk ? (
                <div style={{ padding: "0 14px 12px" }}>
                  <Note kind="warn">
                    Reference layers are held for <b>{health?.region ?? "the supported region"}</b> only. Outside it, cached layers
                    may come back empty and their rules will not fire.
                  </Note>
                </div>
              ) : null}
              <div className="hint" style={{ padding: "0 14px 12px" }}>Imagery availability is checked on the next screen.</div>
            </div>
          </div>

          <div className="card">
            <div className="ch">
              <h3>How wide a band to examine</h3>
            </div>
            <div className="cb stack">
              <div className="row" style={{ justifyContent: "space-between" }}>
                <label htmlFor="band" style={{ fontSize: 13 }}>Search corridor</label>
                <span className="mono" style={{ fontWeight: 500 }} id="bandLbl">± {f.band} km</span>
              </div>
              <input type="range" id="band" min={1} max={15} step={1} value={f.band} style={{ width: "100%" }}
                aria-valuetext={`plus or minus ${f.band} kilometres`} onChange={(e) => set("band", Number(e.target.value))} />
              <div className="row" style={{ justifyContent: "space-between" }}>
                <span className="hint">Tight — less GPU time, less room to deviate</span>
                <span className="hint">Wide — more GPU time, more options</span>
              </div>
              <div className="kpis">
                <Kpi k="Area to process" v={area !== undefined ? int(area) : "—"} unit="km²" />
                <Kpi k="Extraction" v={minutes !== undefined ? (reuseRow?.full_cover ? "reused" : hm(minutes)) : "—"} />
              </div>
              {bandQ.isError ? <ErrorNote error={bandQ.error} /> : null}
              <Note kind="warn">
                <b>The expensive decision on the page.</b> Widening the band later means running the vision models again. Too
                tight, though, and the route is trapped against an obstacle with no room to go round. ± 5 km suits most corridors.
              </Note>
            </div>
          </div>

          {(bandQ.data?.reuse.length ?? 0) > 0 || run?.reuse_from ? (
            <div className="card" style={{ borderColor: "var(--accent-line)" }}>
              <div className="ch">
                <h3>Reuse an earlier extraction</h3>
                {reuseRow?.full_cover && minutes ? (
                  <span className="pill p-go" style={{ marginLeft: "auto" }}>saves {hm(minutes)}</span>
                ) : null}
              </div>
              <div className="cb stack">
                {run?.reuse_from ? (
                  <Note kind="ok">
                    Layers reused from <b>{run.reuse_from}</b>. Only layers whose source you change are extracted again.
                  </Note>
                ) : null}
                <div className="tw">
                  <table data-testid="reuse-table">
                    <thead>
                      <tr>
                        <th>Overlaps</th>
                        <th style={{ textAlign: "right" }}>Shared</th>
                        <th style={{ textAlign: "right" }}>New ground</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bandQ.data?.reuse ?? []).map((r) => (
                        <tr key={r.run_id}>
                          <td>
                            {runId ? (
                              <b>{r.run_id}</b>
                            ) : (
                              <label className="row" style={{ gap: 8, cursor: r.full_cover ? "pointer" : "not-allowed" }}>
                                <input type="checkbox" style={{ width: "auto" }} disabled={!r.full_cover} checked={reuse === r.run_id}
                                  onChange={(e) => {
                                    reuseTouched.current = true;
                                    setReuse(e.target.checked ? r.run_id : null);
                                  }} />
                                <b>{r.run_id}</b>
                              </label>
                            )}
                            <div className="hint">
                              {date(r.created_at)} · ± {num(r.band_km, 0)} km band{r.full_cover ? "" : " · partial — cannot be reused"}
                            </div>
                          </td>
                          <td className="mono" style={{ textAlign: "right" }}>
                            {int(r.shared_km2)} km² · {num(r.shared_pct, 0)}%
                          </td>
                          <td className="mono" style={{ textAlign: "right" }}>{int(Math.max(0, r.new_km2))} km²</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="hint">
                  {runId
                    ? "Reuse is chosen when a run is created. Only a band fully inside an earlier one can be reused in this version."
                    : "Tick a run to reuse its layers. Only a band fully inside an earlier one can be reused in this version."}
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>

      <ErrorNote error={create.error ?? patch.error} />
      <div className="actions">
        <button className="btn ghost" onClick={() => nav("/")}>
          Back
        </button>
        <Btn pending={create.isPending || patch.isPending} disabled={!canSubmit} onClick={submit} data-testid="corridor-next">
          {!runId ? "Next: check imagery" : dirty ? "Save and check imagery" : "Next: check imagery"}
        </Btn>
      </div>
    </section>
  );
}
