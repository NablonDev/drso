import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { mapToolState } from "./MapTools";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { towersToGeoJSON, useTowers, type StoredTower } from "../../store/towers";
import { num } from "../../lib/format";

const SRC = "towers";
const CIRCLE = "towers-circle";
const LABEL_MIN_ZOOM = 13.5;
const MAX_LABELS = 220;

export function TowerPopup({ t }: { t: StoredTower }) {
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
}: {
  map: maplibregl.Map | null;
  routes: number[];
  editable?: boolean;
  before?: string;
}) {
  const byRoute = useTowers((s) => s.byRoute);
  const moveTower = useTowers((s) => s.moveTower);
  const towers = useMemo(() => routes.flatMap((r) => byRoute[r] ?? []), [byRoute, routes]);
  const data = useMemo(() => towersToGeoJSON(towers), [towers]);
  const index = useMemo(() => new Map(towers.map((t) => [t.fid, t])), [towers]);
  const [popup, setPopup] = useState<{ t: StoredTower; el: HTMLDivElement } | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const [labels, setLabels] = useState<{ fid: string; x: number; y: number; text: string; cls: string }[]>([]);

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
  }, [map, index, editable]);

  // drag to move (future editing) — only wired when editable
  useEffect(() => {
    if (!map || !editable) return;
    let dragging: string | null = null;
    const onDown = (e: maplibregl.MapLayerMouseEvent) => {
      const fid = e.features?.[0]?.properties?.fid;
      if (!fid) return;
      e.preventDefault();
      dragging = String(fid);
      map.dragPan.disable();
      map.getCanvas().style.cursor = "grabbing";
    };
    const onMove = (e: maplibregl.MapMouseEvent) => {
      if (dragging) moveTower(dragging, e.lngLat.lng, e.lngLat.lat);
    };
    const onUp = () => {
      if (!dragging) return;
      dragging = null;
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
        return;
      }
      const b = map.getBounds();
      const out: { fid: string; x: number; y: number; text: string; cls: string }[] = [];
      for (const t of towers) {
        if (!b.contains([t.lon, t.lat])) continue;
        const p = map.project([t.lon, t.lat]);
        out.push({
          fid: t.fid,
          x: p.x,
          y: p.y,
          text: t.type === "G" ? "Gantry" : `${t.no} · ${t.type}`,
          cls: t.type === "G" ? "g" : t.angle_point ? "ang" : "",
        });
        if (out.length >= MAX_LABELS) break;
      }
      setLabels(out);
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
  }, [map, towers]);

  useEffect(() => () => {
    popupRef.current?.remove();
  }, []);

  const container = map?.getContainer();
  return (
    <>
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
      {popup ? createPortal(<TowerPopup t={popup.t} />, popup.el) : null}
    </>
  );
}
