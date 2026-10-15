import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { mapToolState } from "./MapTools";
import { num } from "../../lib/format";
import { STATUS_COLOUR, STATUS_LABEL, type CrossingRow } from "../../lib/design";

const SRC = "crossings";
const LAYER = "crossings-circle";
const LABEL_MIN_ZOOM = 13.5;

/** One crossing's metrics against its rule's limits, in the map popup. */
export function CrossingPopup({ r }: { r: CrossingRow }) {
  const c = r.c;
  const rng = (v: [number, number] | null, u = "") => (v ? `${v[0]}–${v[1]}${u}` : "any");
  return (
    <div data-testid="crossing-popup">
      <div className="eyebrow" style={{ color: STATUS_COLOUR[r.status] }}>{STATUS_LABEL[r.status]} crossing</div>
      <div style={{ fontSize: 13, fontWeight: 500, margin: "3px 0 4px" }}>{c.name} · km {num(r.km, 2)}</div>
      <table className="kv">
        <tbody>
          <tr><td>Towers</td><td className="mono">{r.a.no} {r.a.type} → {r.b.no} {r.b.type}{c.min_type ? <span className="lim"> min {c.min_type}</span> : null}</td></tr>
          <tr><td>Angle</td><td className="mono">{num(c.angle_deg, 1)}°<span className="lim"> allowed {rng(r.angleAllowed, "°")}{r.anglePref ? `, preferred ${rng(r.anglePref, "°")}` : ""}</span></td></tr>
          <tr><td>Span</td><td className="mono">{num(c.span_m, 0)} m{r.spanMax !== null ? <span className="lim"> max {r.spanMax} m</span> : null}</td></tr>
          <tr><td>Tower distance</td><td className="mono">{num(c.side_a_m, 0)} / {num(c.side_b_m, 0)} m{r.sideWindow ? <span className="lim"> window {rng(r.sideWindow, " m")}</span> : null}</td></tr>
          {typeof c.null_offset_pct === "number" ? (
            <tr><td>Null point</td><td className="mono">{num(c.null_offset_pct, 1)}% off<span className="lim"> preferred ±{r.nullPref ?? "—"}%, allowed ±{r.nullAllowed ?? "—"}%</span></td></tr>
          ) : null}
          <tr><td>Deviation</td><td className="mono">{num(r.a.deflection_deg, 1)}° / {num(r.b.deflection_deg, 1)}°{r.maxDev !== null ? <span className="lim"> max {r.maxDev}°</span> : null}</td></tr>
          {r.span.min_clearance_m !== null ? <tr><td>Clearance</td><td className="mono">{num(r.span.min_clearance_m, 1)} m</td></tr> : null}
          {c.width_m ? <tr><td>Width</td><td className="mono">{num(c.width_m, 0)} m{c.major ? " · major" : ""}</td></tr> : null}
          {r.confidence !== null ? <tr><td>Source</td><td className="mono"><span className={r.confidence < 0.8 ? "warn" : ""}>confidence {r.confidence.toFixed(2)}</span>{r.confidence < 0.8 ? <span className="lim">verify on site</span> : null}</td></tr> : null}
          {r.binding.length ? <tr><td>At limit</td><td className="bind">{r.binding.join(", ")}</td></tr> : null}
          {r.issues.length ? (
            <tr><td>Notes</td><td>{r.issues.map((f, i) => <span key={i} className={`fl ${f.kind}`}>{f.text}</span>)}</td></tr>
          ) : null}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Crossing points of the selected route (15 Oct 2026): a dot where each span crosses a feature,
 * coloured by how the crossing stands against its rule (preferred / relaxed / violation), a short
 * label (angle · span) from z13.5, the full metrics in a popup on click.
 */
export default function CrossingLayer({ map, rows, visible }: { map: maplibregl.Map | null; rows: CrossingRow[]; visible: boolean }) {
  const [popup, setPopup] = useState<{ r: CrossingRow; el: HTMLDivElement } | null>(null);
  const popupRef = useRef<maplibregl.Popup | null>(null);
  const [labels, setLabels] = useState<{ key: string; x: number; y: number; text: string; cls: string }[]>([]);
  const index = useMemo(() => new Map(rows.map((r) => [r.key, r])), [rows]);
  const data = useMemo<GeoJSON.FeatureCollection<GeoJSON.Point, { key: string; status: string; lowconf: boolean }>>(
    () => ({
      type: "FeatureCollection",
      features: rows.map((r) => ({ type: "Feature", geometry: { type: "Point", coordinates: r.lonlat }, properties: { key: r.key, status: r.status, lowconf: r.confidence !== null && r.confidence < 0.8 } })),
    }),
    [rows],
  );

  useEffect(() => {
    if (!map) return;
    if (!map.getSource(SRC)) {
      map.addSource(SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] }, promoteId: "key" });
      map.addLayer({
        id: LAYER,
        type: "circle",
        source: SRC,
        paint: {
          "circle-color": ["match", ["get", "status"], "violation", STATUS_COLOUR.violation, "relaxed", STATUS_COLOUR.relaxed, STATUS_COLOUR.ok],
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 9, 2.5, 12, 4, 15, 6, 18, 8],
          // a low-confidence source gets an amber ring: the crossed feature may not be where the map says
          "circle-stroke-color": ["case", ["get", "lowconf"], "#D69A4E", "#ffffff"],
          "circle-stroke-width": ["case", ["boolean", ["feature-state", "hover"], false], 2.5, ["case", ["get", "lowconf"], 2.2, 1.2]],
          "circle-opacity": 0.95,
        },
      });
    }
    return () => {
      try {
        if (map.getStyle()) {
          if (map.getLayer(LAYER)) map.removeLayer(LAYER);
          if (map.getSource(SRC)) map.removeSource(SRC);
        }
      } catch {
        /* map gone */
      }
    };
  }, [map]);

  useEffect(() => {
    if (!map) return;
    (map.getSource(SRC) as GeoJSONSource | undefined)?.setData(data);
    // above the route lines, under the towers
    if (map.getLayer(LAYER)) map.moveLayer(LAYER, map.getLayer("towers-circle") ? "towers-circle" : undefined);
  }, [map, data]);

  useEffect(() => {
    if (!map || !map.getLayer(LAYER)) return;
    map.setLayoutProperty(LAYER, "visibility", visible ? "visible" : "none");
    if (!visible) {
      popupRef.current?.remove();
      setLabels([]);
    }
  }, [map, visible, data]);

  useEffect(() => {
    if (!map) return;
    let hovered: string | null = null;
    const onClick = (e: maplibregl.MapLayerMouseEvent) => {
      if (mapToolState.busy) return;
      const r = index.get(String(e.features?.[0]?.properties?.key));
      if (!r) return;
      e.preventDefault();
      popupRef.current?.remove();
      const el = document.createElement("div");
      const p = new maplibregl.Popup({ closeButton: true, maxWidth: "360px", offset: 10 }).setLngLat(r.lonlat).setDOMContent(el).addTo(map);
      p.on("close", () => setPopup((cur) => (cur?.el === el ? null : cur)));
      popupRef.current = p;
      setPopup({ r, el });
    };
    const onMove = (e: maplibregl.MapLayerMouseEvent) => {
      if (mapToolState.busy) return;
      map.getCanvas().style.cursor = "pointer";
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
    map.on("click", LAYER, onClick);
    map.on("mousemove", LAYER, onMove);
    map.on("mouseleave", LAYER, onLeave);
    return () => {
      map.off("click", LAYER, onClick);
      map.off("mousemove", LAYER, onMove);
      map.off("mouseleave", LAYER, onLeave);
    };
  }, [map, index]);

  // angle · span labels under the dots at high zoom
  useEffect(() => {
    if (!map) return;
    let raf = 0;
    const compute = () => {
      raf = 0;
      if (!visible || map.getZoom() < LABEL_MIN_ZOOM || !rows.length) {
        setLabels((l) => (l.length ? [] : l));
        return;
      }
      const b = map.getBounds();
      const out: typeof labels = [];
      for (const r of rows) {
        if (!b.contains(r.lonlat)) continue;
        const p = map.project(r.lonlat);
        out.push({ key: r.key, x: p.x, y: p.y, text: `${r.c.angle_deg.toFixed(0)}° · ${r.c.span_m.toFixed(0)} m`, cls: r.status === "violation" ? "x bad" : r.status === "relaxed" ? "x rel" : "x" });
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
  }, [map, rows, visible]);

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
                <span key={l.key} className={l.cls} style={{ left: l.x, top: l.y }}>{l.text}</span>
              ))}
            </div>,
            container,
          )
        : null}
      {popup ? createPortal(<CrossingPopup r={popup.r} />, popup.el) : null}
    </>
  );
}
