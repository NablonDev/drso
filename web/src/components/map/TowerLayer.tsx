import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { mapToolState } from "./MapTools";
import { moveHandlers } from "./TowerMoveTool";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { towersToGeoJSON, useTowers, type StoredTower } from "../../store/towers";
import { num } from "../../lib/format";
import TowerWhyPanel, { type WhyCtx } from "./TowerWhy";

const SRC = "towers";
const CIRCLE = "towers-circle";
const LABEL_MIN_ZOOM = 13.5;
const ANGLE_MIN_ZOOM = 14;      // deflection arcs at the angle towers (15 Oct 2026)
const ARC_PX = 30;              // arc radius on screen, shrunk on short spans
const ARC_MIN_PX = 11;          // below this the arc is unreadable: the angle goes on the tower label instead
const SPAN_MIN_ZOOM = 14.5;     // span length at the midpoint of each span
const MAX_LABELS = 220;

export function TowerPopup({ t, onWhy }: { t: StoredTower; onWhy?: () => void }) {
  const isG = t.type === "G";
  return (
    <div>
      <div className="eyebrow">{isG ? "Gantry" : t.angle_point ? "Angle tower" : "Tower"}</div>
      <div style={{ fontSize: 13, fontWeight: 500, margin: "3px 0 4px" }}>
        {isG ? `Gantry · route ${t.route}` : `Tower ${t.no} · Type ${t.type} · route ${t.route}`}
      </div>
      <table className="kv">
        <tbody>
          <tr><td>Chainage</td><td className="mono">km {num(t.chainage_km, 3)}</td></tr>
          {!isG ? <tr><td>Deflection</td><td className="mono">{num(t.deflection_deg, 1)}° {t.turn && t.turn !== "-" ? t.turn : ""}</td></tr> : null}
          <tr><td>Span ahead</td><td className="mono">{t.span_ahead_m ? `${num(t.span_ahead_m, 0)} m` : "—"}</td></tr>
          {t.extension_m ? <tr><td>Extension</td><td className="mono">+{num(t.extension_m, 0)} m</td></tr> : null}
          <tr><td>Ground</td><td className="mono">{t.ground_z === null || t.ground_z === undefined ? "—" : `${num(t.ground_z, 1)} m`}</td></tr>
          <tr><td>Position</td><td className="mono">{t.lat.toFixed(5)}, {t.lon.toFixed(5)}</td></tr>
          {t.reasons.length ? <tr><td>Why here</td><td>{t.reasons.map((r) => r.replace(/[_:]/g, " ")).join(", ")}</td></tr> : null}
          {t.warnings.length ? <tr><td>Warnings</td><td style={{ color: "var(--cross)" }}>{t.warnings.join("; ")}</td></tr> : null}
          {t.violations.length ? <tr><td>Violations</td><td style={{ color: "var(--nogo)" }}>{t.violations.join("; ")}</td></tr> : null}
        </tbody>
      </table>
      {onWhy && !isG ? (
        <button className="mini" style={{ marginTop: 8 }} onClick={onWhy} data-testid="tower-why-open">
          Why this tower{t.angle_point ? ", and why not straight?" : "?"}
        </button>
      ) : null}
    </div>
  );
}

/**
 * Spotted towers as a GeoJSON source with stable ids (`${route}-${no}`) read from the tower
 * store. `editable` turns on drag-to-move: the store's moveTower() updates one tower and this
 * layer re-renders from the store — no route data or map layers are rebuilt.
 */
export default function TowerLayer({
  map,
  routes,
  editable = false,
  before,
  angles = false,
  spans = false,
  spanNotes,
  why,
}: {
  map: maplibregl.Map | null;
  routes: number[];
  editable?: boolean;
  before?: string;
  /** deflection on the angle-tower labels, e.g. "12 · C1 · 29° R" */
  angles?: boolean;
  /** span length at the midpoint of every span */
  spans?: boolean;
  /** why a span is allowed its length (crossing / hilly / approach), by the tower it leaves */
  spanNotes?: Map<number, string>;
  /** the solve and route for the "why this tower" section of the popup */
  why?: WhyCtx;
}) {
  const byRoute = useTowers((s) => s.byRoute);
  const moveTower = useTowers((s) => s.moveTower);
  const towers = useMemo(() => routes.flatMap((r) => byRoute[r] ?? []), [byRoute, routes]);
  const data = useMemo(() => towersToGeoJSON(towers), [towers]);
  const index = useMemo(() => new Map(towers.map((t) => [t.fid, t])), [towers]);
  const [popup, setPopup] = useState<{ t: StoredTower; el: HTMLDivElement } | null>(null);
  // the "why this tower" panel, docked inside the map so the straight alternative stays visible beside it
  const [whyT, setWhyT] = useState<StoredTower | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const [labels, setLabels] = useState<{ fid: string; x: number; y: number; text: string; cls: string }[]>([]);
  // deflection arcs in screen space (SVG over the map): extension of the incoming line, the arc to the
  // outgoing span, the angle at the arc's bisector
  const [arcs, setArcs] = useState<{ fid: string; ext: string; arc: string }[]>([]);

  // source + layers, once per map
  useEffect(() => {
    if (!map) return;
    if (!map.getSource(SRC)) {
      map.addSource(SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "fid" });
      map.addLayer(
        {
          id: CIRCLE,
          type: "circle",
          source: SRC,
          paint: {
            "circle-color": ["case", ["get", "gantry"], "#14202A", ["get", "angle"], "#D69A4E", "#FFFFFF"],
            "circle-radius": [
              "interpolate", ["linear"], ["zoom"],
              8, ["case", ["get", "gantry"], 3.5, ["get", "angle"], 1.8, 0],
              10.5, ["case", ["get", "gantry"], 4.5, ["get", "angle"], 3.2, 1.3],
              13, ["case", ["get", "gantry"], 6, ["get", "angle"], 5, 3.2],
              17, ["case", ["get", "gantry"], 9, ["get", "angle"], 8, 5.5],
            ],
            "circle-stroke-color": ["case", ["get", "angle"], "#5C3508", "#14202A"],
            "circle-stroke-width": ["case", ["boolean", ["feature-state", "hover"], false], 3, 1],
          },
        },
        before && map.getLayer(before) ? before : undefined,
      );
    }
    return () => {
      try {
        if (map.getStyle()) {
          if (map.getLayer(CIRCLE)) map.removeLayer(CIRCLE);
          if (map.getSource(SRC)) map.removeSource(SRC);
        }
      } catch {
        /* map gone */
      }
    };
  }, [map, before]);

  // data from the store
  useEffect(() => {
    if (!map) return;
    (map.getSource(SRC) as GeoJSONSource | undefined)?.setData(data);
    // keep towers above route lines added by the parent after this layer
    if (map.getLayer(CIRCLE)) map.moveLayer(CIRCLE);
  }, [map, data]);

  // click → popup; hover cursor
  useEffect(() => {
    if (!map) return;
    let hovered: string | null = null;
    const onClick = (e: maplibregl.MapLayerMouseEvent) => {
      if (mapToolState.busy) return;
      const f = e.features?.[0];
      const t = f && index.get(String(f.properties?.fid));
      if (!t) return;
      e.preventDefault();
      popupRef.current?.remove();
      const el = document.createElement("div");
      const p = new maplibregl.Popup({ closeButton: true, maxWidth: "340px", offset: 10 })
        .setLngLat([t.lon, t.lat])
        .setDOMContent(el)
        .addTo(map);
      p.on("close", () => setPopup((cur) => (cur?.el === el ? null : cur)));
      popupRef.current = p;
      setPopup({ t, el });
    };
    const onMove = (e: maplibregl.MapLayerMouseEvent) => {
      if (mapToolState.busy) return;
      map.getCanvas().style.cursor = editable ? "grab" : "pointer";
      const id = e.features?.[0]?.id;
      if (id === undefined || String(id) === hovered) return;
      if (hovered) map.setFeatureState({ source: SRC, id: hovered }, { hover: false });
      hovered = String(id);
      map.setFeatureState({ source: SRC, id: hovered }, { hover: true });
    };
    const onLeave = () => {
      map.getCanvas().style.cursor = "";
      if (hovered) map.setFeatureState({ source: SRC, id: hovered }, { hover: false });
      hovered = null;
    };
    map.on("click", CIRCLE, onClick);
    map.on("mousemove", CIRCLE, onMove);
    map.on("mouseleave", CIRCLE, onLeave);
    return () => {
      map.off("click", CIRCLE, onClick);
      map.off("mousemove", CIRCLE, onMove);
      map.off("mouseleave", CIRCLE, onLeave);
    };
  }, [map, index, editable, why]);

  // drag to move (future editing) — only wired when editable
  useEffect(() => {
    if (!map || !editable) return;
    let dragging: string | null = null;
    let last: [number, number] | null = null;
    const onDown = (e: maplibregl.MapLayerMouseEvent) => {
      const fid = e.features?.[0]?.properties?.fid;
      if (!fid || e.features?.[0]?.properties?.gantry) return;
      e.preventDefault();
      dragging = String(fid);
      map.dragPan.disable();
      map.getCanvas().style.cursor = "grabbing";
    };
    const onMove = (e: maplibregl.MapMouseEvent) => {
      if (!dragging) return;
      last = [e.lngLat.lng, e.lngLat.lat];
      moveTower(dragging, e.lngLat.lng, e.lngLat.lat);
      moveHandlers.drag?.(dragging, e.lngLat.lng, e.lngLat.lat);
    };
    const onUp = () => {
      if (!dragging) return;
      if (last) moveHandlers.drop?.(dragging, last[0], last[1]);
      dragging = null;
      last = null;
      map.dragPan.enable();
      map.getCanvas().style.cursor = "";
    };
    map.on("mousedown", CIRCLE, onDown);
    map.on("mousemove", onMove);
    map.on("mouseup", onUp);
    return () => {
      map.off("mousedown", CIRCLE, onDown);
      map.off("mousemove", onMove);
      map.off("mouseup", onUp);
    };
  }, [map, editable, moveTower]);

  // type labels at high zoom (HTML, so no glyph server is needed)
  useEffect(() => {
    if (!map) return;
    let raf = 0;
    const compute = () => {
      raf = 0;
      if (map.getZoom() < LABEL_MIN_ZOOM || !towers.length) {
        setLabels((l) => (l.length ? [] : l));
        setArcs((l) => (l.length ? [] : l));
        return;
      }
      const b = map.getBounds();
      const z = map.getZoom();
      const withAngle = angles && z >= ANGLE_MIN_ZOOM;
      const withSpan = spans && z >= SPAN_MIN_ZOOM;
      const out: { fid: string; x: number; y: number; text: string; cls: string }[] = [];
      const arcOut: { fid: string; ext: string; arc: string }[] = [];
      for (let i = 0; i < towers.length; i++) {
        const t = towers[i];
        const inView = b.contains([t.lon, t.lat]);
        if (inView) {
          const p = map.project([t.lon, t.lat]);
          const degTxt = `${t.deflection_deg.toFixed(t.deflection_deg < 10 ? 1 : 0)}°${t.turn && t.turn !== "-" ? ` ${t.turn}` : ""}`;
          let ang = "";
          if (withAngle && t.type !== "G" && (t.angle_point || t.deflection_deg >= 2)) {
            const prev = towers[i - 1], next = towers[i + 1];
            const arc = prev && next && prev.route === t.route && next.route === t.route && prev.no === t.no - 1 && next.no === t.no + 1
              ? arcAt(p, map.project([prev.lon, prev.lat]), map.project([next.lon, next.lat]))
              : null;
            if (arc) {
              arcOut.push({ fid: t.fid, ext: arc.ext, arc: arc.arc });
              out.push({ fid: `${t.fid}-arc`, x: arc.lx, y: arc.ly, text: degTxt, cls: "arc" });
            } else ang = ` · ${degTxt}`;
          }
          out.push({
            fid: t.fid,
            x: p.x,
            y: p.y,
            text: t.type === "G" ? "Gantry" : `${t.no} · ${t.type}${ang}`,
            cls: t.type === "G" ? "g" : t.angle_point ? "ang" : "",
          });
        }
        // the span ahead, labelled at its midpoint when either end is in view
        const n = towers[i + 1];
        if (withSpan && n && n.route === t.route && n.no === t.no + 1 && (inView || b.contains([n.lon, n.lat]))) {
          const len = t.span_ahead_m;
          if (len) {
            const m = map.project([(t.lon + n.lon) / 2, (t.lat + n.lat) / 2]);
            const note = spanNotes?.get(t.no);
            out.push({ fid: `${t.fid}-sp`, x: m.x, y: m.y, text: `${len.toFixed(0)} m${note ? ` · ${note}` : ""}`, cls: "sp" });
          }
        }
        if (out.length >= MAX_LABELS) break;
      }
      setLabels(out);
      setArcs(arcOut);
    };
    const schedule = () => {
      if (!raf) raf = requestAnimationFrame(compute);
    };
    compute();
    map.on("move", schedule);
    map.on("resize", schedule);
    return () => {
      if (raf) cancelAnimationFrame(raf);
      map.off("move", schedule);
      map.off("resize", schedule);
    };
  }, [map, towers, angles, spans, spanNotes]);

  useEffect(() => () => {
    popupRef.current?.remove();
  }, []);

  const container = map?.getContainer();
  return (
    <>
      {container && arcs.length
        ? createPortal(
            <svg className="maparcs" aria-hidden>
              {arcs.map((a) => (
                <g key={a.fid}>
                  <path className="case" d={a.ext} />
                  <path className="case" d={a.arc} />
                  <path className="ext" d={a.ext} />
                  <path className="arc" d={a.arc} />
                </g>
              ))}
            </svg>,
            container,
          )
        : null}
      {container && labels.length
        ? createPortal(
            <div className="maplabels" aria-hidden>
              {labels.map((l) => (
                <span key={l.fid} className={l.cls} style={{ left: l.x, top: l.y }}>
                  {l.text}
                </span>
              ))}
            </div>,
            container,
          )
        : null}
      {popup ? createPortal(<TowerPopup t={popup.t} onWhy={why ? () => { setWhyT(popup.t); popupRef.current?.remove(); } : undefined} />, popup.el) : null}
      {container && why && whyT ? createPortal(
        <div className="whypanel" data-testid="why-panel">
          <div className="whypanel-h">
            <div>
              <div className="eyebrow">{whyT.angle_point ? "Angle tower" : "Tower"} · route {whyT.route}</div>
              <b>Tower {whyT.no} · {whyT.type} · {num(whyT.deflection_deg, 1)}°{whyT.turn && whyT.turn !== "-" ? ` ${whyT.turn}` : ""}</b>
              <span className="hint" style={{ margin: 0 }}> · km {num(whyT.chainage_km, 2)}</span>
            </div>
            <button className="mini" onClick={() => setWhyT(null)} aria-label="Close" data-testid="why-close">✕</button>
          </div>
          <TowerWhyPanel t={whyT} ctx={why} map={map} />
        </div>,
        container,
      ) : null}
    </>
  );
}

/**
 * The deflection at a tower drawn in screen space: the incoming span's line carried on past the tower
 * (dashed) and an arc from that line round to the outgoing span, the angle label on the arc's bisector.
 * Returns null when the spans are too short on screen for the arc to read.
 */
function arcAt(p: { x: number; y: number }, pIn: { x: number; y: number }, pOut: { x: number; y: number }) {
  const dIn = { x: p.x - pIn.x, y: p.y - pIn.y }, dOut = { x: pOut.x - p.x, y: pOut.y - p.y };
  const lIn = Math.hypot(dIn.x, dIn.y), lOut = Math.hypot(dOut.x, dOut.y);
  if (lIn < 1 || lOut < 1) return null;
  const r = Math.min(ARC_PX, 0.4 * Math.min(lIn, lOut));
  if (r < ARC_MIN_PX) return null;
  const uIn = { x: dIn.x / lIn, y: dIn.y / lIn }, uOut = { x: dOut.x / lOut, y: dOut.y / lOut };
  const a0 = { x: p.x + uIn.x * r, y: p.y + uIn.y * r };
  const a1 = { x: p.x + uOut.x * r, y: p.y + uOut.y * r };
  const cross = uIn.x * uOut.y - uIn.y * uOut.x;        // screen y points down: >0 = clockwise on screen
  const sweep = cross > 0 ? 1 : 0;
  const ext = `M${p.x.toFixed(1)},${p.y.toFixed(1)} L${(p.x + uIn.x * r * 1.6).toFixed(1)},${(p.y + uIn.y * r * 1.6).toFixed(1)}`;
  const arc = `M${a0.x.toFixed(1)},${a0.y.toFixed(1)} A${r.toFixed(1)},${r.toFixed(1)} 0 0 ${sweep} ${a1.x.toFixed(1)},${a1.y.toFixed(1)}`;
  let bx = uIn.x + uOut.x, by = uIn.y + uOut.y;
  const bl = Math.hypot(bx, by) || 1;
  bx /= bl; by /= bl;
  return { ext, arc, lx: p.x + bx * (r + 16), ly: p.y + by * (r + 16) };
}
