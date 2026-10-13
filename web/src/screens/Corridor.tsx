import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { apiCalls, qk, useApiMutation, useConfig, useGeoBand, useGeoValidate, useHealth, useRun, type RunCreateBody } from "../api/hooks";
import type { Pt, RunDetail } from "../api/types";
import { hasLayers } from "../components/Shell";
import { Btn, ErrorNote, Head, Kpi, Loading, Note, Spinner } from "../components/ui";
import { fmtDec, parseCoord, parsePair } from "../lib/dms";
import { date, fold, hm, int, num } from "../lib/format";
import { useDebounced } from "../lib/useDebounced";

interface Form {
  name: string;
  lat1: string;
  lon1: string;
  lat2: string;
  lon2: string;
  kv: number;
  band: number;
}

/** Every line is double circuit (Vivek, 6 Oct 2026); the band defaults to ± 25 km. */
const CIRCUIT = "dc";
const BAND_MAX = 25;
const EMPTY: Form = { name: "", lat1: "", lon1: "", lat2: "", lon2: "", kv: 765, band: BAND_MAX };

function fromRun(r: RunDetail): Form {
  return {
    name: r.name ?? "",
    lat1: fmtDec(r.a_lat),
    lon1: fmtDec(r.a_lon),
    lat2: fmtDec(r.b_lat),
    lon2: fmtDec(r.b_lon),
    kv: r.kv,
    band: Math.min(BAND_MAX, Math.max(1, Math.round(r.band_km))),
  };
}

function coordErr(v: string, max: number): string | null {
  if (!v.trim()) return null;
  const n = parseCoord(v);
  if (n === null) return "Not a coordinate — use decimal degrees or 22°18'25.9\"N";
  if (Math.abs(n) > max) return `Must be between −${max} and ${max}`;
  return null;
}

function toPt(lat: string, lon: string, name: string | null | undefined): Pt | null {
  const la = parseCoord(lat);
  const lo = parseCoord(lon);
  if (la === null || lo === null || Math.abs(la) > 90 || Math.abs(lo) > 180) return null;
  return { lat: la, lon: lo, name: name?.trim() || null };
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
  const [nameTouched, setNameTouched] = useState(false);

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
  // point names are no longer asked for; an older run keeps the ones it has
  const a = useMemo(() => toPt(f.lat1, f.lon1, run?.a_name), [f.lat1, f.lon1, run?.a_name]);
  const b = useMemo(() => toPt(f.lat2, f.lon2, run?.b_name), [f.lat2, f.lon2, run?.b_name]);
  const name = f.name.trim().replace(/\s+/g, " ");
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

  // the corridor proper (fixed once layers exist) vs. the name (always editable)
  const corridorDirty = useMemo(() => {
    if (!run) return true;
    const o = fromRun(run);
    return (
      (["kv", "band"] as const).some((k) => o[k] !== f[k]) ||
      parseCoord(o.lat1) !== parseCoord(f.lat1) ||
      parseCoord(o.lon1) !== parseCoord(f.lon1) ||
      parseCoord(o.lat2) !== parseCoord(f.lat2) ||
      parseCoord(o.lon2) !== parseCoord(f.lon2)
    );
  }, [run, f]);
  const nameDirty = Boolean(run) && name !== (run?.name ?? "");
  const dirty = corridorDirty || nameDirty;

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

  const canSubmit = Boolean(name && a && b && geo.data?.ok && !geo.isFetching);
  const submit = () => {
    setNameTouched(true);
    if (!a || !b || !name) return;
    if (!runId) {
      const body: RunCreateBody = { name, a, b, kv: f.kv, circuit: CIRCUIT, band_km: f.band };
      create.mutate({ ...body, routes_selected: [1], reuse_from: reuse });
    } else if (corridorDirty) {
      patch.mutate({ ...(nameDirty ? { name } : {}), a, b, kv: f.kv, band_km: f.band });
    } else if (nameDirty) {
      patch.mutate({ name });                       // a rename is allowed even on a locked run
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
        A name for the corridor, two coordinates, the voltage, and how wide a band to examine. Every line is planned as double
        circuit. Extraction takes time on the GPU, so the band is the one input worth thinking about.
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
              <h3>Corridor</h3>
            </div>
            <div className="cb stack">
              <label className="fld">
                <span>Corridor name</span>
                <input id="corridor-name" data-testid="corridor-name-input" type="text" required maxLength={80} value={f.name}
                  aria-invalid={nameTouched && !name} onChange={(e) => set("name", e.target.value)}
                  onBlur={() => setNameTouched(true)} placeholder="Bhuj – Lakadia 765 kV" />
                {nameTouched && !name ? <div className="err">A corridor name is required</div> : null}
              </label>
              <div className="hint">Shown at the top of every page as this run's reference. It can be changed later.</div>
            </div>
          </div>
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
              </div>
              <div className="hint" style={{ marginTop: 6 }}>
                Double circuit. Rules and unit costs resolve per voltage on the server.
              </div>
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
              <input type="range" id="band" min={1} max={BAND_MAX} step={1} value={f.band} style={{ width: "100%" }}
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
                tight, though, and the route is trapped against an obstacle with no room to go round. The default is ± 25 km.
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
