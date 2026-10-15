import { useEffect, useMemo } from "react";
import type maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useBuffers, useNearest } from "../../api/hooks";
import type { BufferItem, LonLat, NearestPoint, Route, RulesResponse } from "../../api/types";
import { bendsOf, BEND_COLOUR } from "./BendLayer";

const SRC = "buffers";
const FILL = "buffers-fill";
const LINE = "buffers-line";
export const BUFFER_COLOUR = { parallel: "#D69A4E", nocross: "#D7263D" };

function pnum(rules: RulesResponse | undefined, id: string, key: string): number | null {
  for (const g of rules?.groups ?? []) for (const r of g.rules) if (r.id === id) {
    const p = r.params.find((x) => x.key === key);
    return p && typeof p.value === "number" ? p.value : null;
  }
  return null;
}

/** every crossing rule's parallel-run band, plus the no-crossing zones (15 Oct 2026) */
export function bufferItems(rules: RulesResponse | undefined): BufferItem[] {
  const out: BufferItem[] = [];
  for (const g of rules?.groups ?? []) {
    if (g.key !== "cross" && g.key !== "lock") continue;
    for (const r of g.rules) {
      if (!r.enabled || !r.layer) continue;
      if (r.id === "platform_crossing_zone") out.push({ kind: "nocross", layer: r.layer, buffer_m: 0, classes: r.classes ?? null, near_m: 1500 });
      const par = pnum(rules, r.id, "parallel_setback_m");
      if (par !== null && par > 0) out.push({ kind: "parallel", layer: r.layer, buffer_m: par, classes: r.classes ?? null, near_m: Math.max(400, par * 2) });
    }
  }
  return out;
}

/** reduce a route to a few hundred points for the API (every tower is already a vertex) */
function thin(geom: LonLat[]): LonLat[] {
  const step = Math.max(1, Math.ceil(geom.length / 400));
  const out = geom.filter((_, i) => i % step === 0);
  if (out[out.length - 1] !== geom[geom.length - 1]) out.push(geom[geom.length - 1]);
  return out;
}

/**
 * Buffers the route had to respect, drawn under the route (15 Oct 2026):
 *  - "setback" (with Why bends): the tower setback round the feature each bend reacts to, in the bend's colour
 *  - "parallel" (Buffers toggle): the parallel-run band along roads, canals, rail, rivers and power lines
 *  - "nocross" (Buffers toggle): railway platform crossing zones
 */
export default function BufferLayer({ map, runId, sid, route, rules, bends, buffers }: {
  map: maplibregl.Map | null; runId: string; sid: number; route: Route; rules: RulesResponse | undefined; bends: boolean; buffers: boolean;
}) {
  const key = `${sid}-${route.id}`;
  const line = useMemo(() => thin(route.geometry), [route.geometry]);
  // the bends' features: same request as BendLayer, served from the cache
  const bendList = useMemo(() => bendsOf(route, rules), [route, rules]);
  const points = useMemo<NearestPoint[]>(() => bendList.map((b) => ({ layer: b.layer, lon: b.tower.lon, lat: b.tower.lat, within_m: b.kind === "cross" ? 3000 : 10000 })), [bendList]);
  const near = useNearest(runId, key, points, bends);
  const setbackItems = useMemo<BufferItem[]>(() => {
    const by = new Map<string, BufferItem>();
    const res = near.data?.results ?? [];
    bendList.forEach((b, i) => {
      const n = res[i];
      if (!n || b.setback === null || b.setback <= 0) return;
      const k = `${b.layer}:${b.setback}:${b.kind}`;
      const it = by.get(k) ?? { kind: "setback", layer: b.layer, buffer_m: b.setback, ids: [], near_m: 5000, limit: 200 };
      if (!it.ids!.includes(n.id)) it.ids!.push(n.id);
      by.set(k, it);
    });
    return [...by.values()];
  }, [bendList, near.data]);
  const bandItems = useMemo(() => bufferItems(rules), [rules]);
  const qSet = useBuffers(runId, `${key}-setback`, setbackItems.length ? { route: line, items: setbackItems } : null, bends);
  const qBand = useBuffers(runId, `${key}-bands`, bandItems.length ? { route: line, items: bandItems } : null, buffers);
  const bendKind = useMemo(() => {
    const m = new Map<string, string>();
    for (const b of bendList) m.set(`${b.layer}:${b.setback}`, b.kind);
    return m;
  }, [bendList]);

  const data = useMemo<GeoJSON.FeatureCollection>(() => {
    const feats: GeoJSON.Feature[] = [];
    if (bends) for (const f of qSet.data?.features ?? []) {
      const p = f.properties ?? {};
      feats.push({ ...f, properties: { ...p, colour: BEND_COLOUR[(bendKind.get(`${p.layer}:${p.buffer_m}`) as keyof typeof BEND_COLOUR) ?? "avoid"] } });
    }
    if (buffers) for (const f of qBand.data?.features ?? []) {
      const p = f.properties ?? {};
      feats.push({ ...f, properties: { ...p, colour: p.kind === "nocross" ? BUFFER_COLOUR.nocross : BUFFER_COLOUR.parallel } });
    }
    return { type: "FeatureCollection", features: feats };
  }, [bends, buffers, qSet.data, qBand.data, bendKind]);

  useEffect(() => {
    if (!map) return;
    if (!map.getSource(SRC)) {
      map.addSource(SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      const before = map.getLayer("routes-casing") ? "routes-casing" : undefined;
      map.addLayer({ id: FILL, type: "fill", source: SRC, paint: { "fill-color": ["get", "colour"], "fill-opacity": ["case", ["==", ["get", "kind"], "nocross"], 0.18, 0.1] } }, before);
      map.addLayer({ id: LINE, type: "line", source: SRC, paint: { "line-color": ["get", "colour"], "line-width": 1, "line-dasharray": [3, 2], "line-opacity": 0.8 } }, before);
    }
    return () => {
      try {
        if (map.getStyle()) {
          for (const id of [FILL, LINE]) if (map.getLayer(id)) map.removeLayer(id);
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
  }, [map, data]);

  return null;
}
