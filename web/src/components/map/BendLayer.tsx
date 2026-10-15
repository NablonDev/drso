import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import type maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useNearest } from "../../api/hooks";
import type { NearestPoint, Route, RouteEvent, RulesResponse } from "../../api/types";

const SRC = "bends";
const LINE = "bends-line";
const LABEL_MIN_ZOOM = 13;
const MAX_LABELS = 120;

type Kind = "avoid" | "cross" | "clear";
const KIND: Record<string, Kind> = { TURN_TO_AVOID: "avoid", TURN_TO_CROSS: "cross", TURN_TO_KEEP_CLEAR: "clear" };
export const BEND_COLOUR: Record<Kind, string> = { avoid: "#E05A4E", cross: "#1F9E8E", clear: "#D69A4E" };

interface Bend {
  key: string;
  kind: Kind;
  tower: Route["towers"][number];
  feature: string;
  ruleId: string;
  layer: string;
  clearance: number;
  setback: number | null;
}

/** The rule's tower setback for a bend's feature, when the rule has one. */
function setbackOf(rules: RulesResponse | undefined, ruleId: string): number | null {
  for (const g of rules?.groups ?? [])
    for (const r of g.rules)
      if (r.id === ruleId) {
        const p = r.params.find((x) => x.key === "tower_setback_m" || x.key === "setback_m" || x.key === "conductor_tower_setback_m");
        return p && typeof p.value === "number" ? p.value : null;
      }
  return null;
}

export function bendsOf(route: Route, rules: RulesResponse | undefined): Bend[] {
  const byNo = new Map(route.towers.map((t) => [t.no, t]));
  const out: Bend[] = [];
  for (const e of route.events as RouteEvent[]) {
    const kind = KIND[e.type];
    if (!kind) continue;
    const d = (e.data ?? {}) as { tower?: number; feature?: string; rule_id?: string; clearance_m?: number };
    const t = d.tower !== undefined ? byNo.get(d.tower) : undefined;
    const layer = e.source?.layer;
    if (!t || !layer) continue;
    out.push({ key: `${route.id}-${t.no}-${d.rule_id ?? layer}`, kind, tower: t, feature: d.feature ?? e.source?.feature ?? layer, ruleId: d.rule_id ?? layer, layer, clearance: d.clearance_m ?? 0, setback: setbackOf(rules, d.rule_id ?? layer) });
  }
  return out;
}

/**
 * Why this bend (15 Oct 2026): a tie line from each angle tower to the feature its turn reacts to, with
 * the distance and the rule's setback on it. The feature's closest point comes from the API, one request
 * per route. Red = turned to avoid, teal = turned to cross, amber = turned to keep clear.
 */
export default function BendLayer({ map, runId, sid, route, rules, visible }: {
  map: maplibregl.Map | null; runId: string; sid: number; route: Route; rules: RulesResponse | undefined; visible: boolean;
}) {
  const bends = useMemo(() => bendsOf(route, rules), [route, rules]);
  const points = useMemo<NearestPoint[]>(() => bends.map((b) => ({ layer: b.layer, lon: b.tower.lon, lat: b.tower.lat, within_m: 3000 })), [bends]);
  const q = useNearest(runId, `${sid}-${route.id}`, points, visible);
  const ties = useMemo(() => {
    const res = q.data?.results ?? [];
    return bends.map((b, i) => ({ b, n: res[i] ?? null })).filter((x) => x.n);
  }, [bends, q.data]);
  const [labels, setLabels] = useState<{ key: string; x: number; y: number; text: string; cls: string }[]>([]);

  const data = useMemo<GeoJSON.FeatureCollection<GeoJSON.LineString, { kind: Kind }>>(
    () => ({
      type: "FeatureCollection",
      features: ties.map(({ b, n }) => ({ type: "Feature", properties: { kind: b.kind }, geometry: { type: "LineString", coordinates: [[b.tower.lon, b.tower.lat], [n!.lon, n!.lat]] } })),
    }),
    [ties],
  );

  useEffect(() => {
    if (!map) return;
    if (!map.getSource(SRC)) {
      map.addSource(SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({
        id: LINE,
        type: "line",
        source: SRC,
        layout: { "line-cap": "round" },
        paint: {
          "line-color": ["match", ["get", "kind"], "avoid", BEND_COLOUR.avoid, "cross", BEND_COLOUR.cross, BEND_COLOUR.clear],
          "line-width": ["interpolate", ["linear"], ["zoom"], 11, 1, 15, 2],
          "line-dasharray": [1.5, 2],
          "line-opacity": 0.95,
        },
      });
    }
    return () => {
      try {
        if (map.getStyle()) {
          if (map.getLayer(LINE)) map.removeLayer(LINE);
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
    // under the crossing dots and the towers
    if (map.getLayer(LINE)) map.moveLayer(LINE, map.getLayer("crossings-circle") ? "crossings-circle" : map.getLayer("towers-circle") ? "towers-circle" : undefined);
  }, [map, data]);

  useEffect(() => {
    if (!map || !map.getLayer(LINE)) return;
    map.setLayoutProperty(LINE, "visibility", visible ? "visible" : "none");
  }, [map, visible, data]);

  useEffect(() => {
    if (!map) return;
    let raf = 0;
    const compute = () => {
      raf = 0;
      if (!visible || map.getZoom() < LABEL_MIN_ZOOM || !ties.length) {
        setLabels((l) => (l.length ? [] : l));
        return;
      }
      const b = map.getBounds();
      const out: typeof labels = [];
      for (const { b: bend, n } of ties) {
        const mid: [number, number] = [(bend.tower.lon + n!.lon) / 2, (bend.tower.lat + n!.lat) / 2];
        if (!b.contains(mid) && !b.contains([bend.tower.lon, bend.tower.lat])) continue;
        const p = map.project(mid);
        // measured: tower to the nearest point of the feature; solver: the clearance the engine logged for the
        // turn (from the conductor, after the setback), shown when it differs from the measured distance
        const sb = bend.setback !== null ? ` (≥ ${bend.setback} m)` : "";
        const solver = bend.clearance > 0 && Math.abs(bend.clearance - n!.distance_m) > 2 ? ` · solver ${bend.clearance.toFixed(0)} m` : "";
        out.push({ key: bend.key, x: p.x, y: p.y, text: `${n!.distance_m.toFixed(0)} m · ${bend.feature}${sb}${solver}`, cls: `bend ${bend.kind}` });
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
  }, [map, ties, visible]);

  const container = map?.getContainer();
  return container && labels.length
    ? createPortal(
        <div className="maplabels" aria-hidden>
          {labels.map((l) => (
            <span key={l.key} className={l.cls} style={{ left: l.x, top: l.y }}>{l.text}</span>
          ))}
        </div>,
        container,
      )
    : null;
}
