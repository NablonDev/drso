import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import maplibregl, { type GeoJSONSource } from "maplibre-gl";
import { useNavigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { apiCalls, qk, useConfig } from "../../api/hooks";
import type { LayerConfig } from "../../api/types";
import { useUi } from "../../store/ui";
import { quickSolveOn } from "../../lib/quickSolve";
import { Btn, Dialog } from "../ui";

/**
 * Map tools shared by the route map and the layer map (Vivek, 8 Oct 2026):
 * right-click → copy coordinates · measure distance · draw a point / line / polygon ·
 * load a KML / KMZ / GeoJSON / GPKG / zipped shapefile · add the shapes to a layer
 * (temporary for this run, or permanent) · list and delete hand-added shapes.
 */

type Mode = "none" | "measure" | "point" | "line" | "polygon";
type LngLat = [number, number];
type FC = GeoJSON.FeatureCollection;

/** True while a measuring or drawing tool owns map clicks; other click / hover handlers stand aside. */
export const mapToolState = { busy: false };

export interface Shape {
  id: number;
  scope: "temporary" | "permanent";
  layer_key: string;
  layer_name: string;
  class: string;
  name: string;
  added_in_run: string;
  created_at: string;
  gtype: string;
  geometry: GeoJSON.Geometry;
}

const EMPTY: FC = { type: "FeatureCollection", features: [] };

function haversine(a: LngLat, b: LngLat): number {
  const R = 6371008.8;
  const toR = Math.PI / 180;
  const dLat = (b[1] - a[1]) * toR;
  const dLon = (b[0] - a[0]) * toR;
  const s = Math.sin(dLat / 2) ** 2 + Math.cos(a[1] * toR) * Math.cos(b[1] * toR) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(s));
}

function pathLength(pts: LngLat[]): number {
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += haversine(pts[i - 1], pts[i]);
  return s;
}

/** drop consecutive near-duplicate vertices (a double click adds the same point twice) */
function dedupe(pts: LngLat[]): LngLat[] {
  return pts.filter((p, i) => i === 0 || haversine(pts[i - 1], p) > 0.5);
}

const metres = (m: number) => `${Math.round(m).toLocaleString("en-IN")} m`;
const coordText = (ll: LngLat) => `${ll[1].toFixed(6)}, ${ll[0].toFixed(6)}`;

async function copyText(t: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(t);
    return true;
  } catch {
    const ta = document.createElement("textarea");
    ta.value = t;
    ta.style.cssText = "position:fixed;left:-9999px";
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = document.execCommand("copy");
    } catch {
      ok = false;
    }
    ta.remove();
    return ok;
  }
}

function accepts(layer: LayerConfig, gtype: string): boolean {
  if (layer.kind === "any") return true;
  const base = gtype.replace("Multi", "");
  return (layer.kind === "polygon" && base === "Polygon") || (layer.kind === "line" && base === "LineString") || (layer.kind === "point" && base === "Point");
}

function bboxOfFC(fc: FC): [number, number, number, number] | null {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const walk = (c: unknown): void => {
    if (Array.isArray(c) && typeof c[0] === "number") {
      const [x, y] = c as number[];
      x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
    } else if (Array.isArray(c)) c.forEach(walk);
  };
  for (const f of fc.features) if (f.geometry && "coordinates" in f.geometry) walk(f.geometry.coordinates);
  return Number.isFinite(x0) ? [x0, y0, x1, y1] : null;
}

function setData(map: maplibregl.Map, id: string, data: FC) {
  const s = map.getSource(id) as GeoJSONSource | undefined;
  if (s) s.setData(data);
}

export default function MapTools({ map, runId, onLayersChanged }: {
  map: maplibregl.Map | null;
  runId: string;
  /** called after shapes change a run layer, so the screen reloads its tiles */
  onLayersChanged?: () => void;
}) {
  const [mode, setMode] = useState<Mode>("none");
  const [pts, setPts] = useState<LngLat[]>([]);
  const [hover, setHover] = useState<LngLat | null>(null);
  const [measured, setMeasured] = useState<LngLat[] | null>(null);
  const [menu, setMenu] = useState<{ x: number; y: number; ll: LngLat; z?: number | null } | null>(null);
  const [pending, setPending] = useState<FC | null>(null);
  const [pendingName, setPendingName] = useState("");
  const [askRerun, setAskRerun] = useState<string | null>(null);
  const [showList, setShowList] = useState(false);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const push = useUi((s) => s.push);
  const qc = useQueryClient();
  const nav = useNavigate();
  const cfg = useConfig().data;
  const shapesQ = useQuery({
    queryKey: ["shapes", runId],
    queryFn: () => api<{ shapes: Shape[] }>(`/api/runs/${encodeURIComponent(runId)}/shapes`),
  });
  const shapes = useMemo(() => shapesQ.data?.shapes ?? [], [shapesQ.data]);

  mapToolState.busy = mode !== "none";

  // ---- sources and layers ------------------------------------------------------------------
  useEffect(() => {
    if (!map) return;
    const ensure = () => {
      for (const id of ["mt-shapes", "mt-pending", "mt-draft"]) if (!map.getSource(id)) map.addSource(id, { type: "geojson", data: EMPTY });
      const add = (l: maplibregl.LayerSpecification) => {
        if (!map.getLayer(l.id)) map.addLayer(l);
      };
      add({ id: "mt-shapes-fill", type: "fill", source: "mt-shapes", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#ff4fd8", "fill-opacity": 0.12 } });
      add({ id: "mt-shapes-line", type: "line", source: "mt-shapes", filter: ["!=", ["geometry-type"], "Point"], paint: { "line-color": "#ff4fd8", "line-width": 2 } });
      add({ id: "mt-shapes-pt", type: "circle", source: "mt-shapes", filter: ["==", ["geometry-type"], "Point"], paint: { "circle-radius": 5, "circle-color": "#ff4fd8", "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 } });
      add({ id: "mt-pending-fill", type: "fill", source: "mt-pending", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#ffd23f", "fill-opacity": 0.25 } });
      add({ id: "mt-pending-line", type: "line", source: "mt-pending", filter: ["!=", ["geometry-type"], "Point"], paint: { "line-color": "#ffd23f", "line-width": 2.5 } });
      add({ id: "mt-pending-pt", type: "circle", source: "mt-pending", filter: ["==", ["geometry-type"], "Point"], paint: { "circle-radius": 6, "circle-color": "#ffd23f", "circle-stroke-color": "#14202A", "circle-stroke-width": 1.5 } });
      add({ id: "mt-draft-fill", type: "fill", source: "mt-draft", filter: ["==", ["geometry-type"], "Polygon"], paint: { "fill-color": "#ffffff", "fill-opacity": 0.15 } });
      add({ id: "mt-draft-line", type: "line", source: "mt-draft", filter: ["!=", ["geometry-type"], "Point"], paint: { "line-color": "#ffffff", "line-width": 2, "line-dasharray": [2, 2] } });
      add({ id: "mt-draft-pt", type: "circle", source: "mt-draft", filter: ["==", ["geometry-type"], "Point"], paint: { "circle-radius": 4, "circle-color": "#ffffff", "circle-stroke-color": "#14202A", "circle-stroke-width": 1.5 } });
    };
    ensure();
    map.on("styledata", ensure);
    return () => {
      map.off("styledata", ensure);
    };
  }, [map]);

  // hand-added shapes on the map
  useEffect(() => {
    if (!map || !map.getSource("mt-shapes")) return;
    setData(map, "mt-shapes", {
      type: "FeatureCollection",
      features: shapes.map((s) => ({ type: "Feature", properties: { id: s.id, name: s.name }, geometry: s.geometry })),
    });
  }, [map, shapes]);

  useEffect(() => {
    if (!map || !map.getSource("mt-pending")) return;
    setData(map, "mt-pending", pending ?? EMPTY);
  }, [map, pending]);

  // the draft (measure / draw) follows the pointer
  const draft = useMemo<FC>(() => {
    const line = measured ?? (hover && pts.length ? [...pts, hover] : pts);
    const feats: GeoJSON.Feature[] = line.map((p) => ({ type: "Feature", properties: {}, geometry: { type: "Point", coordinates: p } }));
    if (mode === "polygon" && line.length >= 3)
      feats.push({ type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[...line, line[0]]] } });
    else if (line.length >= 2) feats.push({ type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: line } });
    return { type: "FeatureCollection", features: feats };
  }, [pts, hover, mode, measured]);
  useEffect(() => {
    if (map && map.getSource("mt-draft")) setData(map, "mt-draft", draft);
  }, [map, draft]);

  // live length label next to the pointer (measure)
  const label = useRef<maplibregl.Marker | null>(null);
  useEffect(() => {
    if (!map) return;
    const line = measured ?? (hover && pts.length ? [...pts, hover] : null);
    if (mode !== "measure" && !measured) {
      label.current?.remove();
      label.current = null;
      return;
    }
    if (!line || line.length < 2) {
      label.current?.remove();
      label.current = null;
      return;
    }
    const total = pathLength(line);
    const seg = haversine(line[line.length - 2], line[line.length - 1]);
    const html = line.length > 2 ? `${metres(total)}<span>last segment ${metres(seg)}</span>` : metres(total);
    if (!label.current) {
      const el = document.createElement("div");
      el.className = "mt-label";
      el.setAttribute("data-testid", "measure-label");
      label.current = new maplibregl.Marker({ element: el, anchor: "bottom-left", offset: [10, -8] });
    }
    label.current.getElement().innerHTML = html;
    label.current.setLngLat(line[line.length - 1]).addTo(map);
  }, [map, pts, hover, mode, measured]);
  useEffect(() => () => {
    label.current?.remove();
  }, []);

  const reset = useCallback(() => {
    setPts([]);
    setHover(null);
  }, []);

  const finish = useCallback(
    (raw: LngLat[]) => {
      const final = dedupe(raw);
      if (mode === "measure") {
        setMeasured(final.length >= 2 ? final : null);
        setMode("none");
        reset();
        return;
      }
      let geom: GeoJSON.Geometry | null = null;
      if (mode === "point" && final.length >= 1) geom = { type: "Point", coordinates: final[0] };
      if (mode === "line" && final.length >= 2) geom = { type: "LineString", coordinates: final };
      if (mode === "polygon" && final.length >= 3) geom = { type: "Polygon", coordinates: [[...final, final[0]]] };
      setMode("none");
      reset();
      if (geom) {
        setPending({ type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: geom }] });
        setPendingName("");
      }
    },
    [mode, reset],
  );

  // ---- map events ---------------------------------------------------------------------------
  useEffect(() => {
    if (!map) return;
    const active = mode !== "none";
    if (active) {
      map.doubleClickZoom.disable();
      map.getCanvas().style.cursor = "crosshair";
    }
    const onClick = (e: maplibregl.MapMouseEvent) => {
      if (!active) return;
      e.preventDefault();
      const ll: LngLat = [e.lngLat.lng, e.lngLat.lat];
      if (mode === "point") return finish([ll]);
      if (mode === "polygon" && pts.length >= 3) {
        const p0 = map.project(pts[0]);
        if (Math.hypot(p0.x - e.point.x, p0.y - e.point.y) < 10) return finish(pts);
      }
      if (mode === "measure" && measured) setMeasured(null);
      setPts((p) => [...p, ll]);
    };
    const onDbl = (e: maplibregl.MapMouseEvent) => {
      if (!active) return;
      e.preventDefault();
      // the double click's two clicks were already added as (near-)duplicate vertices; finish() drops them
      finish(pts);
    };
    const onMove = (e: maplibregl.MapMouseEvent) => {
      if (active && pts.length) setHover([e.lngLat.lng, e.lngLat.lat]);
    };
    const onCtx = (e: maplibregl.MapMouseEvent) => {
      e.preventDefault();
      const ll: LngLat = [e.lngLat.lng, e.lngLat.lat];
      setMenu({ x: e.point.x, y: e.point.y, ll, z: undefined });
      // ground height at the point (10 Oct 2026), from the DEM the solver used
      fetch(`/api/dem/elevation?lon=${ll[0]}&lat=${ll[1]}`).then((r) => (r.ok ? r.json() : null)).then((d) => {
        setMenu((m) => (m && m.ll[0] === ll[0] && m.ll[1] === ll[1] ? { ...m, z: d ? (d as { elevation_m: number | null }).elevation_m : null } : m));
      }).catch(() => setMenu((m) => (m ? { ...m, z: null } : m)));
    };
    map.on("click", onClick);
    map.on("dblclick", onDbl);
    map.on("mousemove", onMove);
    map.on("contextmenu", onCtx);
    return () => {
      map.off("click", onClick);
      map.off("dblclick", onDbl);
      map.off("mousemove", onMove);
      map.off("contextmenu", onCtx);
      if (active) {
        map.doubleClickZoom.enable();
        map.getCanvas().style.cursor = "";
      }
    };
  }, [map, mode, pts, finish, measured]);

  // Esc cancels a tool before anything else on the page sees it (e.g. leaving full screen); Enter finishes
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && (mode !== "none" || measured || menu)) {
        e.stopPropagation();
        e.preventDefault();
        setMode("none");
        reset();
        setMeasured(null);
        setMenu(null);
      } else if (e.key === "Enter" && mode !== "none" && mode !== "point") {
        e.preventDefault();
        finish(pts);
      } else if (e.key === "Backspace" && mode !== "none" && pts.length) {
        e.preventDefault();
        setPts((p) => p.slice(0, -1));
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [mode, measured, menu, pts, finish, reset]);

  const pick = (m: Mode) => {
    reset();
    setMeasured(null);
    setMenu(null);
    setMode((cur) => (cur === m ? "none" : m));
  };

  // ---- file loading -------------------------------------------------------------------------
  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setLoadErr(null);
    setLoading(true);
    try {
      const fd = new FormData();
      fd.append("file", f);
      const fc = await api<FC & { count: number }>("/api/geo/parse", { method: "POST", body: fd });
      if (!fc.features.length) throw new Error("The file has no features.");
      setPending({ type: "FeatureCollection", features: fc.features });
      setPendingName(f.name.replace(/\.[^.]+$/, ""));
      const bb = bboxOfFC(fc);
      if (map && bb) map.fitBounds(bb as maplibregl.LngLatBoundsLike, { padding: 60, maxZoom: 16, duration: 400 });
    } catch (e) {
      setLoadErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const changed = (msg: string) => {
    void qc.invalidateQueries({ queryKey: ["shapes", runId] });
    void qc.invalidateQueries({ queryKey: qk.layers(runId) });
    void qc.invalidateQueries({ queryKey: qk.run(runId) });
    onLayersChanged?.();
    setAskRerun(msg);
  };

  const del = async (s: Shape) => {
    try {
      await api(`/api/runs/${encodeURIComponent(runId)}/shapes/${s.id}`, { method: "DELETE" });
      changed(`“${s.name}” was removed from ${s.layer_name}${s.scope === "permanent" ? " for this run and all future runs" : ""}.`);
    } catch (e) {
      push({ kind: "stop", title: "Shape not deleted", text: e instanceof Error ? e.message : String(e) });
    }
  };

  const rerun = async () => {
    setAskRerun(null);
    try {
      await apiCalls.solve(runId, !quickSolveOn());
      void qc.invalidateQueries({ queryKey: qk.active });
      void qc.invalidateQueries({ queryKey: qk.run(runId) });
      nav(`/runs/${runId}/solve`);
    } catch (e) {
      push({ kind: "stop", title: "Solve not started", text: e instanceof Error ? e.message : String(e) });
    }
  };

  const hint =
    mode === "measure" ? "Click to start, click to add points. Double-click or Enter ends, Esc clears."
      : mode === "point" ? "Click where the point goes. Esc cancels."
        : mode === "line" ? "Click each vertex. Double-click or Enter finishes, Backspace undoes, Esc cancels."
          : mode === "polygon" ? "Click each corner. Double-click, Enter or the first corner closes it. Esc cancels."
            : null;

  return (
    <>
      <div className="mt-bar" role="toolbar" aria-label="Map tools" onContextMenu={(e) => e.preventDefault()}>
        <button aria-pressed={mode === "measure"} onClick={() => pick("measure")} title="Measure distance" data-testid="tool-measure">📏<span>Measure</span></button>
        <i />
        <button aria-pressed={mode === "point"} onClick={() => pick("point")} title="Draw a point" data-testid="tool-point">•<span>Point</span></button>
        <button aria-pressed={mode === "line"} onClick={() => pick("line")} title="Draw a line" data-testid="tool-line">╱<span>Line</span></button>
        <button aria-pressed={mode === "polygon"} onClick={() => pick("polygon")} title="Draw a polygon" data-testid="tool-polygon">⬠<span>Polygon</span></button>
        <button onClick={() => fileRef.current?.click()} title="Load a KML, KMZ, GeoJSON, GeoPackage or zipped shapefile" data-testid="tool-load" disabled={loading}>
          {loading ? "…" : "⇪"}<span>Load file</span>
        </button>
        <i />
        <button aria-pressed={showList} onClick={() => setShowList((x) => !x)} title="Shapes added by hand" data-testid="tool-shapes">
          ☰<span>Shapes{shapes.length ? ` (${shapes.length})` : ""}</span>
        </button>
        <input ref={fileRef} type="file" hidden accept=".kml,.kmz,.geojson,.json,.gpkg,.zip" onChange={(e) => void onFile(e.target.files?.[0])} data-testid="tool-file" />
      </div>
      {hint ? <div className="mt-hint" data-testid="tool-hint">{hint}</div> : null}
      {loadErr ? (
        <div className="mt-hint mt-err" role="alert">{loadErr} <button className="link" onClick={() => setLoadErr(null)}>Dismiss</button></div>
      ) : null}
      {measured ? (
        <div className="mt-hint">Measured {metres(pathLength(measured))}. <button className="link" onClick={() => setMeasured(null)}>Clear</button></div>
      ) : null}

      {menu ? (
        <div className="mt-menu" style={{ left: menu.x, top: menu.y }} role="menu" data-testid="map-context-menu" onMouseLeave={() => setMenu(null)}>
          <div className="mt-coords">{coordText(menu.ll)}{menu.z === undefined ? "" : ` · ${menu.z === null ? "no DEM" : `${menu.z.toFixed(1)} m`}`}</div>
          <button role="menuitem" onClick={async () => {
            const ok = await copyText(coordText(menu.ll));
            push({ kind: ok ? "ok" : "warn", text: ok ? `Copied ${coordText(menu.ll)}` : `Copy failed — select and copy: ${coordText(menu.ll)}` });
            setMenu(null);
          }}>Copy coordinates (lat, lon)</button>
          <button role="menuitem" onClick={async () => {
            const t = `${menu.ll[0].toFixed(6)}, ${menu.ll[1].toFixed(6)}`;
            const ok = await copyText(t);
            push({ kind: ok ? "ok" : "warn", text: ok ? `Copied ${t}` : `Copy failed — select and copy: ${t}` });
            setMenu(null);
          }}>Copy as lon, lat</button>
          {typeof menu.z === "number" ? (
            <button role="menuitem" onClick={async () => {
              const t = `${coordText(menu.ll)}, ${menu.z!.toFixed(1)} m`;
              const ok = await copyText(t);
              push({ kind: ok ? "ok" : "warn", text: ok ? `Copied ${t}` : `Copy failed — select and copy: ${t}` });
              setMenu(null);
            }}>Copy with elevation (lat, lon, m)</button>
          ) : null}
        </div>
      ) : null}

      {showList ? (
        <div className="mt-list" data-testid="shapes-list">
          <div className="mt-list-h">
            <b>Shapes added by hand</b>
            <button className="link" onClick={() => setShowList(false)}>Close</button>
          </div>
          {!shapes.length ? <p className="hint">None yet. Draw a shape or load a file, then add it to a layer.</p> : null}
          {shapes.map((s) => (
            <div key={s.id} className="mt-item">
              <div>
                <b>{s.name}</b>
                <em>{s.layer_name} · {s.class} · {s.gtype}</em>
                <em>
                  <span className={`pill ${s.scope === "permanent" ? "p-acc" : "p-chg"}`}>{s.scope}</span>{" "}
                  added in {s.added_in_run} · {new Date(s.created_at).toLocaleString("en-IN", { dateStyle: "medium", timeStyle: "short" })}
                </em>
              </div>
              <div className="mt-actions">
                <button className="mini" onClick={() => {
                  const bb = bboxOfFC({ type: "FeatureCollection", features: [{ type: "Feature", properties: {}, geometry: s.geometry }] });
                  if (map && bb) map.fitBounds(bb as maplibregl.LngLatBoundsLike, { padding: 80, maxZoom: 17, duration: 400 });
                }}>Zoom</button>
                <DeleteButton onConfirm={() => void del(s)} />
              </div>
            </div>
          ))}
        </div>
      ) : null}

      <AddShapeDialog
        open={Boolean(pending)}
        fc={pending}
        defaultName={pendingName}
        layers={cfg?.layers ?? []}
        runId={runId}
        onCancel={() => setPending(null)}
        onSaved={(msg) => {
          setPending(null);
          changed(msg);
        }}
      />

      <Dialog open={Boolean(askRerun)} onClose={() => setAskRerun(null)} id="rerunDlg" labelledBy="rerunDlgT">
        <div className="dlg" data-testid="rerun-dialog">
          <h3 id="rerunDlgT">Re-run the route?</h3>
          <p>{askRerun} The routes on screen were solved before this change.</p>
          <div className="row" style={{ gap: 8, justifyContent: "flex-end" }}>
            <Btn ghost onClick={() => setAskRerun(null)} data-testid="rerun-later">Later</Btn>
            <Btn onClick={() => void rerun()} data-testid="rerun-now">Re-run now</Btn>
          </div>
        </div>
      </Dialog>
    </>
  );
}

function DeleteButton({ onConfirm }: { onConfirm: () => void }) {
  const [sure, setSure] = useState(false);
  useEffect(() => {
    if (!sure) return;
    const t = window.setTimeout(() => setSure(false), 4000);
    return () => window.clearTimeout(t);
  }, [sure]);
  return sure ? (
    <button className="mini danger" onClick={onConfirm}>Confirm delete</button>
  ) : (
    <button className="mini" onClick={() => setSure(true)}>Delete</button>
  );
}

function AddShapeDialog({ open, fc, defaultName, layers, runId, onCancel, onSaved }: {
  open: boolean;
  fc: FC | null;
  defaultName: string;
  layers: LayerConfig[];
  runId: string;
  onCancel: () => void;
  onSaved: (msg: string) => void;
}) {
  const gtypes = useMemo(() => [...new Set((fc?.features ?? []).map((f) => f.geometry.type))], [fc]);
  const options = useMemo(
    () => layers.filter((l) => l.producer !== "derived" || l.key === "clusters").filter((l) => gtypes.every((g) => accepts(l, g))),
    [layers, gtypes],
  );
  const [key, setKey] = useState("");
  const [cls, setCls] = useState("");
  const [name, setName] = useState("");
  const [scope, setScope] = useState<"temporary" | "permanent" | "">("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    if (!open) return;
    setKey("");
    setCls("");
    setName(defaultName);
    setScope("");
    setErr(null);
  }, [open, defaultName]);
  const layer = options.find((l) => l.key === key);
  useEffect(() => {
    setCls(layer?.classes[0] ?? "");
  }, [layer]);
  const n = fc?.features.length ?? 0;
  const ready = Boolean(layer && cls && name.trim() && scope);

  const save = async () => {
    if (!fc || !layer || !scope) return;
    setBusy(true);
    setErr(null);
    try {
      await api(`/api/runs/${encodeURIComponent(runId)}/shapes`, {
        method: "POST",
        json: { layer_key: layer.key, class: cls, name: name.trim(), scope, features: fc.features },
      });
      onSaved(
        `${n === 1 ? `“${name.trim()}”` : `${n} shapes`} added to ${layer.name} ${scope === "permanent" ? "permanently: this run and every future run" : "for this run only"}.`,
      );
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog open={open} onClose={onCancel} id="shapeDlg" labelledBy="shapeDlgT">
      <div className="dlg" data-testid="shape-dialog">
        <h3 id="shapeDlgT">Add {n === 1 ? "this shape" : `these ${n} shapes`} to a layer</h3>
        <p>
          {gtypes.join(", ")}. The shape becomes part of the chosen layer, and the rules treat it like any other feature of that layer.
        </p>
        <div style={{ display: "grid", gap: 10 }}>
          <label className="fld">
            <span>Layer (LULC)</span>
            <select id="shape-layer" value={key} onChange={(e) => setKey(e.target.value)} data-testid="shape-layer">
              <option value="">Choose a layer…</option>
              {options.map((l) => (
                <option key={l.key} value={l.key}>{l.name}</option>
              ))}
            </select>
            {!options.length ? <em className="hint">No layer takes {gtypes.join(" and ")} shapes.</em> : null}
          </label>
          {layer && layer.classes.length > 1 ? (
            <label className="fld">
              <span>Class</span>
              <select id="shape-class" value={cls} onChange={(e) => setCls(e.target.value)} data-testid="shape-class">
                {layer.classes.map((c) => (
                  <option key={c} value={c}>{c.replace(/_/g, " ")}</option>
                ))}
              </select>
            </label>
          ) : null}
          <label className="fld">
            <span>Name</span>
            <input id="shape-name" type="text" value={name} maxLength={120} onChange={(e) => setName(e.target.value)} placeholder="e.g. New poultry farm, Ratnal" data-testid="shape-name" />
            {n > 1 ? <em className="hint">Each shape is named “{name || "name"} — its own name or number”.</em> : null}
          </label>
          <fieldset className="mt-scope">
            <legend>Keep it</legend>
            <label className="mt-radio">
              <input type="radio" name="shape-scope" checked={scope === "temporary"} onChange={() => setScope("temporary")} data-testid="scope-temporary" />
              <span><b>Temporary</b> — used in this run only.</span>
            </label>
            <label className="mt-radio">
              <input type="radio" name="shape-scope" checked={scope === "permanent"} onChange={() => setScope("permanent")} data-testid="scope-permanent" />
              <span><b>Permanent</b> — becomes part of this layer's data, for this run and every run extracted from now on.</span>
            </label>
          </fieldset>
          {err ? <div className="note n-stop" role="alert">{err}</div> : null}
        </div>
        <div className="row" style={{ gap: 8, justifyContent: "flex-end", marginTop: 14 }}>
          <Btn ghost onClick={onCancel}>Cancel</Btn>
          <Btn onClick={() => void save()} disabled={!ready} pending={busy} data-testid="shape-save">Add to layer</Btn>
        </div>
      </div>
    </Dialog>
  );
}
