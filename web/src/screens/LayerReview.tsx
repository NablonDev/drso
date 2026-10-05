import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useLayers } from "../api/hooks";
import type { Job, LayerFlag, LayersResponse, RunLayer } from "../api/types";
import FeatureInfo from "../components/map/FeatureInfo";
import {
  bboxOf,
  DEFAULT_ON,
  geomCoords,
  LAYER_COLOURS,
  LAYER_ORDER,
  layerIds,
  LULC_CLASSES,
  setLulcClasses,
  setRunLayerOpacity,
  setRunLayerVisibility,
} from "../components/map/style";
import { useFitBounds, useMapLibre, useRunTiles, type Bounds } from "../components/map/useMap";
import { CeilingNote, ScaleBar, useCeiling, ZoomBar } from "../components/map/ZoomBar";
import { Btn, ConfPill, ErrorNote, Head, Loading, Note, SourcePill } from "../components/ui";
import { date, elapsedMin, hm, int, num, when } from "../lib/format";
import { useStartSolve } from "./Solve";

/** Imagery-only map drawn over the left part of the view so Compare reveals bare pixels. */
function BareMap({ main, bounds, clipRight }: { main: maplibregl.Map | null; bounds: Bounds | null; clipRight: number }) {
  const ref = useRef<HTMLDivElement>(null);
  const bare = useMapLibre(ref, { bounds, interactive: false });
  useEffect(() => {
    if (!main || !bare) return;
    const sync = () => bare.jumpTo({ center: main.getCenter(), zoom: main.getZoom(), bearing: main.getBearing(), pitch: main.getPitch() });
    sync();
    main.on("move", sync);
    return () => {
      main.off("move", sync);
    };
  }, [main, bare]);
  return <div ref={ref} className="mlmap under" style={{ clipPath: `inset(0 ${clipRight}% 0 0)`, zIndex: 3 }} aria-hidden />;
}

function flagFix(f: LayerFlag): "retry" | "sources" | null {
  if (f.kind === "failed" || f.kind === "stale") return "retry";
  if (f.kind === "empty" || f.kind === "low_conf" || f.kind === "coverage") return "sources";
  return null;
}

const FLAG_PILL: Record<LayerFlag["kind"], [string, string]> = {
  failed: ["p-nogo", "failed"],
  empty: ["p-lock", "nothing found"],
  low_conf: ["p-nogo", "low confidence"],
  coverage: ["p-cross", "partial coverage"],
  stale: ["p-cross", "source changed"],
};

export default function LayerReview() {
  const run = useRunCtx();
  const lq = useLayers(run.id);
  if (lq.isPending) return <Loading text="Loading layers…" />;
  if (lq.isError) return <ErrorNote error={lq.error} retry={() => void lq.refetch()} />;
  return <Body L={lq.data} />;
}

function Body({ L }: { L: LayersResponse }) {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const startSolve = useStartSolve(run.id);

  const [on, setOn] = useState<Record<string, boolean>>(() => Object.fromEntries(LAYER_ORDER.map((k) => [k, DEFAULT_ON.has(k)])));
  const [solo, setSolo] = useState<string | null>(null);
  const [cls, setCls] = useState<Record<string, boolean>>(() => Object.fromEntries(LULC_CLASSES.map((c) => [c.key, true])));
  const [op, setOp] = useState(55);
  const [swipe, setSwipe] = useState(100);
  const [imagery, setImagery] = useState(true);
  const [sel, setSel] = useState<{ fid: number; key: string; lngLat: [number, number] } | null>(null);
  const [popupEl, setPopupEl] = useState<HTMLDivElement | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);

  const bounds = useMemo<Bounds | null>(() => bboxOf(geomCoords(L.band)), [L.band]);
  const mapRef = useRef<HTMLDivElement>(null);
  const map = useMapLibre(mapRef, { bounds });
  useFitBounds(map, bounds);
  const ceiling = useCeiling(map);
  const gap = run.imagery?.esri?.gap_geojson ?? null;

  const vis = useCallback((k: string) => (solo ? solo === k : Boolean(on[k])), [solo, on]);
  const state = useRef({ vis, op, cls });
  state.current = { vis, op, cls };
  const applyAll = useCallback((m: maplibregl.Map) => {
    const s = state.current;
    setRunLayerVisibility(m, s.vis);
    setRunLayerOpacity(m, s.op / 100);
    const enabled = Object.entries(s.cls).filter(([, v]) => v).map(([k]) => k);
    setLulcClasses(m, enabled.length === LULC_CLASSES.length ? null : enabled);
  }, []);
  const tiles = useRunTiles(map, run.id, true, applyAll);

  // band outline + imagery gap
  useEffect(() => {
    if (!map || !L) return;
    if (!map.getSource("band")) {
      map.addSource("band", { type: "geojson", data: { type: "Feature", properties: {}, geometry: L.band } });
      map.addLayer({ id: "band-line", type: "line", source: "band", paint: { "line-color": "#ffffff", "line-width": 1.6, "line-dasharray": [4, 3], "line-opacity": 0.85 } });
    } else (map.getSource("band") as GeoJSONSource).setData({ type: "Feature", properties: {}, geometry: L.band });
    if (gap && !map.getSource("gap")) {
      map.addSource("gap", { type: "geojson", data: { type: "Feature", properties: {}, geometry: gap } });
      map.addLayer({ id: "gap-fill", type: "fill", source: "gap", paint: { "fill-color": "#A33A31", "fill-opacity": 0.12 } });
      map.addLayer({ id: "gap-line", type: "line", source: "gap", paint: { "line-color": "#E28B80", "line-width": 1.4, "line-dasharray": [3, 2] } });
    }
  }, [map, L, gap]);

  useEffect(() => {
    if (map) applyAll(map);
  }, [map, vis, op, cls, applyAll]);

  useEffect(() => {
    if (map?.getLayer("esri")) map.setLayoutProperty("esri", "visibility", imagery ? "visible" : "none");
  }, [map, imagery]);

  // click → feature provenance
  useEffect(() => {
    if (!map) return;
    const onClick = (e: maplibregl.MapMouseEvent) => {
      if (!map.getSource("run")) return;
      const ids = LAYER_ORDER.flatMap(layerIds).filter((id) => map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== "none");
      const fs = map.queryRenderedFeatures(e.point, { layers: ids });
      // prefer points, then lines, then the smallest polygons (lulc last)
      const f = fs.sort((a, b) => (a.sourceLayer === "lulc" ? 1 : 0) - (b.sourceLayer === "lulc" ? 1 : 0))[0];
      const fid = f ? Number(f.id ?? f.properties?.id) : NaN;
      if (!f || !Number.isFinite(fid)) {
        setSel(null);
        return;
      }
      setSel({ fid, key: f.sourceLayer ?? "", lngLat: [e.lngLat.lng, e.lngLat.lat] });
    };
    const onMove = (e: maplibregl.MapMouseEvent) => {
      if (!map.getSource("run")) return;
      const ids = LAYER_ORDER.flatMap(layerIds).filter((id) => map.getLayer(id));
      map.getCanvas().style.cursor = map.queryRenderedFeatures(e.point, { layers: ids }).length ? "pointer" : "";
    };
    map.on("click", onClick);
    map.on("mousemove", onMove);
    return () => {
      map.off("click", onClick);
      map.off("mousemove", onMove);
    };
  }, [map]);

  // highlight + popup for the selected feature
  useEffect(() => {
    if (!map || !sel) {
      popupRef.current?.remove();
      return;
    }
    try {
      map.setFeatureState({ source: "run", sourceLayer: sel.key, id: sel.fid }, { sel: true });
    } catch {
      /* tiles not loaded */
    }
    const el = document.createElement("div");
    const p = new maplibregl.Popup({ maxWidth: "340px", offset: 8 }).setLngLat(sel.lngLat).setDOMContent(el).addTo(map);
    p.on("close", () => setSel((s) => (s && s.fid === sel.fid ? null : s)));
    popupRef.current = p;
    setPopupEl(el);
    return () => {
      p.remove();
      try {
        if (map.getSource("run")) map.setFeatureState({ source: "run", sourceLayer: sel.key, id: sel.fid }, { sel: false });
      } catch {
        /* gone */
      }
    };
  }, [map, sel]);

  const extract = useApiMutation<Job, string[]>({
    mutationFn: (keys) => apiCalls.extract(run.id, keys),
    errorTitle: "Extraction not started",
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.active });
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
      nav(`/runs/${run.id}/extracting`);
    },
  });

  // swipe handle dragging
  const canvasRef = useRef<HTMLDivElement>(null);
  const dragSwipe = (e: React.PointerEvent) => {
    const box = canvasRef.current?.getBoundingClientRect();
    if (!box) return;
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
    const move = (ev: PointerEvent) => {
      const x = Math.min(1, Math.max(0, (ev.clientX - box.left) / box.width));
      setSwipe(Math.round((1 - x) * 100));
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const byKey: Record<string, RunLayer> = Object.fromEntries(L.layers.map((l) => [l.key, l]));
  const extracted = L.layers.filter((l) => l.extracted);
  const lowConf = extracted.filter((l) => (l.confidence ?? 1) < 0.8 && (l.feature_count ?? 0) > 0);
  const empty = extracted.filter((l) => l.feature_count === 0);
  const carried = extracted.filter((l) => l.carried);
  const ex = run.last_extract;
  const exMin = ex?.started_at && ex.finished_at ? elapsedMin(ex.started_at, ex.finished_at) : null;
  const flags = L.flags;
  const lulcCounts = byKey.lulc?.stats?.classes ?? {};
  const allDone = extracted.length === L.layers.length;
  const ordered = [...L.layers].sort((a, b) => LAYER_ORDER.indexOf(a.key) - LAYER_ORDER.indexOf(b.key));
  const selLayer = sel ? byKey[sel.key] : null;

  return (
    <section>
      <Head title="Review the layers" right={allDone && !run.layers_stale.length ? <span className="pill p-go">extraction complete</span> : <span className="pill p-cross">{run.layers_stale.length ? "re-extract needed" : "partly extracted"}</span>}>
        {extracted.length} of {L.layers.length} layers extracted{exMin ? ` in ${hm(exMin)}` : ""}. {run.latest_solve_id ? "A route has been solved on these layers." : "Nothing has been routed."} Check what the models found, fix what they missed, then release the solve.
      </Head>

      {flags.length ? (
        <div className="card gapcard" style={{ marginBottom: 14 }}>
          <div className="ch">
            <h3>{flags.length === 1 ? "One thing" : flags.length <= 10 ? `${["", "", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"][flags.length]} things` : `${flags.length} things`} worth a look</h3>
            <span className="hint" style={{ marginLeft: "auto" }}>flagged automatically</span>
          </div>
          <div className="tw">
            <table data-testid="flags">
              <thead><tr><th>Layer</th><th>What was flagged</th><th style={{ textAlign: "right" }}>Fix</th></tr></thead>
              <tbody>
                {flags.map((f, i) => {
                  const l = byKey[f.layer];
                  const fix = flagFix(f);
                  const [pc, pt] = FLAG_PILL[f.kind] ?? ["p-lock", f.kind];
                  return (
                    <tr key={`${f.layer}-${f.kind}-${i}`}>
                      <td><b>{l?.name ?? f.layer}</b> <span className={`pill ${pc}`}>{pt}</span></td>
                      <td>
                        {f.text}
                        {l?.extracted ? <div className="hint">Confidence {l.confidence?.toFixed(2) ?? "n/a"} · {l.source ?? ""}</div> : null}
                      </td>
                      <td style={{ textAlign: "right" }}>
                        {fix === "retry" ? (
                          <Btn ghost className="mini" style={{ padding: "3px 9px" }} pending={extract.isPending && extract.variables?.includes(f.layer)}
                            disabled={Boolean(run.job) || extract.isPending} onClick={() => extract.mutate([f.layer])}>
                            {f.kind === "failed" ? "Retry this layer" : "Re-extract"}
                          </Btn>
                        ) : fix === "sources" ? (
                          <button className="mini" onClick={() => nav(`/runs/${run.id}/sources`)}>Supply a file</button>
                        ) : (
                          <span className="hint">nothing to do</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <ErrorNote error={extract.error} />
        </div>
      ) : null}

      <div className="m8grid">
        <div>
          <div className="m8bar">
            <span className="pill p-acc">Corridor · {int(L.band_km2)} km²</span>
            <ZoomBar map={map} ceiling={ceiling} />
            <button className="mini" aria-pressed={imagery} onClick={() => setImagery((x) => !x)}>Imagery</button>
            {gap ? (
              <button className="mini" onClick={() => {
                const b = bboxOf(geomCoords(gap));
                if (map && b) map.fitBounds(b as maplibregl.LngLatBoundsLike, { padding: 40 });
              }}>Jump to the imagery gap</button>
            ) : null}
            <div className="spacer" />
            <button className="mini" onClick={() => {
              setOn(Object.fromEntries(LAYER_ORDER.map((k) => [k, DEFAULT_ON.has(k)])));
              setCls(Object.fromEntries(LULC_CLASSES.map((c) => [c.key, true])));
              setSolo(null);
              setSel(null);
              setOp(55);
              setSwipe(100);
              setImagery(true);
              if (map && bounds) map.fitBounds(bounds as maplibregl.LngLatBoundsLike, { padding: 30 });
            }}>Reset the view</button>
          </div>
          <div className="mapcanvas" ref={canvasRef} data-testid="layer-map">
            <div ref={mapRef} className="mlmap" />
            {swipe < 100 ? <BareMap main={map} bounds={bounds} clipRight={swipe} /> : null}
            {swipe < 100 ? (
              <div className="swipe" style={{ left: `${100 - swipe}%` }}>
                <i />
                <button aria-label="Drag to compare layers with bare imagery" onPointerDown={dragSwipe}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowLeft") setSwipe((s) => Math.min(100, s + 5));
                    if (e.key === "ArrowRight") setSwipe((s) => Math.max(0, s - 5));
                  }}>⇆</button>
              </div>
            ) : null}
            {!map ? <div className="mapnote"><Loading text="Loading imagery…" /></div> : null}
          </div>
          <ScaleBar map={map} />
          <CeilingNote ceiling={ceiling} />
          {tiles.error ? (
            <div className="note n-stop" style={{ marginTop: 8 }} role="alert" data-testid="tile-error">
              <div style={{ flex: 1 }}>
                <b>{tiles.error}</b> The imagery is shown without the layers. The table below still reflects what was extracted.
              </div>
              <button className="mini" onClick={tiles.retry}>Try again</button>
            </div>
          ) : null}
          <div className="card" style={{ marginTop: 10 }}>
            <div className="cb stack">
              <div className="slrow">
                <span className="k">Overlay</span>
                <input type="range" min={0} max={100} step={5} value={op} onChange={(e) => setOp(Number(e.target.value))} aria-label="Overlay opacity" />
                <b>{op}%</b>
              </div>
              <div className="slrow">
                <span className="k">Compare</span>
                <input type="range" min={0} max={100} step={1} value={swipe} onChange={(e) => setSwipe(Number(e.target.value))} aria-label="Compare layers with bare imagery" />
                <b style={{ width: "auto", fontFamily: "var(--sans)", color: "var(--muted)" }}>
                  {swipe >= 100 ? "layers over the whole view" : swipe <= 0 ? "imagery only" : "drag to compare"}
                </b>
              </div>
              <div className="hint">
                Drag <b>Compare</b> (or the handle on the map) to wipe the layers off and see the bare imagery underneath — the fastest way
                to tell whether a mask actually follows what is on the ground.
              </div>
            </div>
          </div>
          <div className="grid2" style={{ marginTop: 12 }}>
            <div className="card">
              <div className="ch"><h3>This revision</h3></div>
              <div className="cb">
                <div className="tw">
                  <table>
                    <tbody>
                      <tr><td>Layers extracted</td><td className="mono" style={{ textAlign: "right" }}>{extracted.length}</td></tr>
                      <tr><td>Features found</td><td className="mono" style={{ textAlign: "right" }}>{int(L.features_total)}</td></tr>
                      <tr><td>Below 0.80 confidence</td><td className="mono" style={{ textAlign: "right", color: lowConf.length ? "var(--nogo)" : undefined }}>{lowConf.length}</td></tr>
                      <tr><td>Empty in the band</td><td className="mono" style={{ textAlign: "right" }}>{empty.length}</td></tr>
                      <tr><td>Carried from {run.parent_id ?? run.reuse_from ?? "an earlier run"}</td><td className="mono" style={{ textAlign: "right" }}>{carried.length}</td></tr>
                      <tr><td>Extraction time</td><td className="mono" style={{ textAlign: "right" }}>{exMin !== null ? hm(exMin) : "—"}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
            <div className="card">
              <div className="ch"><h3>How the map is served</h3><span className="pill p-acc" style={{ marginLeft: "auto" }}>proxied</span></div>
              <div className="cb">
                <div className="tw">
                  <table className="kv">
                    <tbody>
                      <tr><td>Basemap</td><td>Esri World Imagery through the Beeline proxy<div className="hint">the same imagery the high-resolution models read</div></td></tr>
                      <tr><td>Native zoom</td><td className="mono">z18 · about 0.5 m/px<div className="hint">the viewer refuses to go finer</div></td></tr>
                      <tr><td>Layers</td><td className="mono">Vector tiles z6–z16<div className="hint">buildings from z12 · classes restyle in the browser</div></td></tr>
                      <tr><td>Imagery dates</td><td className="mono">{run.imagery?.esri ? `${date(run.imagery.esri.oldest)} – ${date(run.imagery.esri.newest)}` : "not checked"}</td></tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div className="stack">
          <div className="card">
            <div className="ch">
              <h3 style={{ fontSize: 13 }}>Layers</h3>
              <span className="hint" style={{ marginLeft: "auto" }}>{L.layers.length} · click <b>only</b> to isolate</span>
            </div>
            <div data-testid="layer-list">
              {ordered.map((l) => {
                const v = vis(l.key);
                const isSolo = solo === l.key;
                const emptyL = l.extracted && l.feature_count === 0;
                return (
                  <div key={l.key} className={`lrow ${v ? "" : "loff"} ${isSolo ? "lsolo" : ""}`}>
                    <label className="ln">
                      <input type="checkbox" checked={v} disabled={Boolean(solo)} onChange={() => setOn((o) => ({ ...o, [l.key]: !o[l.key] }))}
                        aria-label={`Show ${l.name}`} />
                      <i className="sw" style={{ background: LAYER_COLOURS[l.key] ?? "#888" }} />
                      <span>
                        <b>{l.name}</b>
                        <em>
                          {l.extracted ? `${int(l.feature_count ?? 0)} features` : "not extracted"}
                          {l.extracted && l.confidence !== null && l.confidence !== undefined ? ` · ${l.confidence.toFixed(2)}` : ""}
                        </em>
                      </span>
                    </label>
                    {emptyL || !l.extracted ? (
                      <span className="pill p-lock">{emptyL ? "empty" : "—"}</span>
                    ) : (
                      <button className={`only ${isSolo ? "onx" : ""}`} onClick={() => {
                        setSolo((s) => (s === l.key ? null : l.key));
                        setSel(null);
                      }} title="Show only this layer" aria-pressed={isSolo}>
                        {isSolo ? "exit" : "only"}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <div className="ch"><h3 style={{ fontSize: 13 }}>LULC classes</h3></div>
            <div className="cb">
              {LULC_CLASSES.map((c) => (
                <label key={c.key} className={`clsrow ${cls[c.key] ? "" : "loff"}`}>
                  <input type="checkbox" checked={cls[c.key]} onChange={() => setCls((s) => ({ ...s, [c.key]: !s[c.key] }))} />
                  <i className="sw" style={{ background: c.colour }} />
                  <span>{c.label}</span>
                  <b>{lulcCounts[c.key] ? `${int(lulcCounts[c.key])}` : "none"}</b>
                </label>
              ))}
              <div className="hint" style={{ marginTop: 6 }}>
                Polygon counts per class. Toggling a class restyles the vector tiles in the browser — no server round trip.
              </div>
            </div>
          </div>

          <div className="card">
            <div className="cb">
              {sel ? (
                <>
                  <FeatureInfo runId={run.id} fid={sel.fid} layerName={selLayer?.name} />
                  <div className="row" style={{ gap: 6, marginTop: 8 }}>
                    <button className="mini" onClick={() => map?.flyTo({ center: sel.lngLat, zoom: 17 })}>Inspect at 0.5 m</button>
                    <button className="mini" onClick={() => nav(`/runs/${run.id}/sources`)}>Correct this layer</button>
                  </div>
                </>
              ) : (
                <>
                  <div className="eyebrow">Selected feature</div>
                  <p className="hint" style={{ marginTop: 6 }}>
                    Click any polygon or line on the map. Every feature carries its class, extent, source, confidence and the model version
                    that drew it.
                  </p>
                </>
              )}
            </div>
          </div>

          <div className="card" style={{ borderColor: "var(--accent-line)" }}>
            <div className="cb stack">
              <div className="eyebrow">Release step 2</div>
              <p style={{ fontSize: 12.5, color: "var(--ink-2)" }}>
                The solve re-uses these layers. Run it as often as you like — changing a rule and re-solving does not re-extract anything.
              </p>
              {!run.layers_ready ? (
                <div className="hint" style={{ margin: 0 }}>
                  {run.layers_stale.length ? `Re-extract first: ${run.layers_stale.join(", ")}` : "Every layer must be extracted before solving."}
                </div>
              ) : null}
              <Btn pending={startSolve.isPending} disabled={!run.layers_ready || Boolean(run.job)} onClick={() => startSolve.mutate()}>
                Solve the route
              </Btn>
              <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/sources`)}>Correct a layer first</button>
              <ErrorNote error={startSolve.error} />
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch">
          <h3>What every layer returned</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>{L.layers.length} layers · source and confidence per layer</span>
        </div>
        <div className="tw">
          <table data-testid="layer-table">
            <thead>
              <tr>
                <th>Layer</th>
                <th>Found</th>
                <th>Source</th>
                <th>Mode</th>
                <th>Conf.</th>
                <th>Model</th>
                <th>Image date</th>
                <th>Model run</th>
                <th style={{ textAlign: "right" }}>Coverage</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {ordered.map((l) => {
                const fl = flags.filter((f) => f.layer === l.key);
                return (
                  <tr key={l.key}>
                    <td>
                      <b>{l.name}</b>{" "}
                      {fl.map((f, i) => {
                        const [pc, pt] = FLAG_PILL[f.kind] ?? ["p-lock", f.kind];
                        return <span key={i} className={`pill ${pc}`} style={{ marginRight: 4 }}>{pt}</span>;
                      })}
                    </td>
                    <td className="mono" style={{ fontSize: 12 }}>
                      {!l.extracted ? "—" : l.status === "failed" ? <span style={{ color: "var(--nogo)" }}>failed</span> : int(l.feature_count ?? 0)}
                    </td>
                    <td style={{ fontSize: 12, maxWidth: 260 }}>{l.source ?? l.desc ?? "—"}{l.file_name ? <div className="hint mono">{l.file_name}</div> : null}</td>
                    <td><SourcePill mode={l.mode ?? l.source_setting.mode} producer={l.producer} /></td>
                    <td>{l.extracted ? <ConfPill value={l.confidence} /> : <span className="hint">—</span>}</td>
                    <td className="mono" style={{ fontSize: 11.5 }}>{l.model ? `${l.model}${l.model_version ? ` · ${l.model_version}` : ""}` : l.model_version ?? "—"}</td>
                    <td className="mono" style={{ fontSize: 12 }}>{date(l.image_date)}</td>
                    <td className="mono" style={{ fontSize: 12 }}>{when(l.model_run_at)}</td>
                    <td className="mono" style={{ textAlign: "right" }}>{l.coverage_pct === null || l.coverage_pct === undefined ? "—" : `${num(l.coverage_pct, 0)}%`}</td>
                    <td style={{ textAlign: "right" }}>
                      {l.status === "failed" || l.stale ? (
                        <button className="mini" disabled={Boolean(run.job) || extract.isPending} onClick={() => extract.mutate([l.key])}>Retry</button>
                      ) : (
                        <button className="mini" onClick={() => nav(`/runs/${run.id}/sources`)}>Add or correct</button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {lowConf.length ? (
        <Note kind="warn" style={{ marginTop: 14 }}>
          <b>{lowConf.length} layer{lowConf.length > 1 ? "s" : ""} below 0.80 confidence</b> —{" "}
          {lowConf.map((l) => `${l.name.split(" — ")[0].toLowerCase()} ${l.confidence?.toFixed(2)}`).join(", ")}. The route treats their
          features as certain; the report does not.
        </Note>
      ) : null}

      <div className="actions">
        <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/${run.last_extract ? "extracting" : "review"}`)}>Back</button>
        <Btn pending={startSolve.isPending} disabled={!run.layers_ready || Boolean(run.job)} onClick={() => startSolve.mutate()}>
          Solve the route →
        </Btn>
      </div>

      {sel && popupEl ? createPortal(<FeatureInfo runId={run.id} fid={sel.fid} layerName={selLayer?.name} />, popupEl) : null}
    </section>
  );
}
