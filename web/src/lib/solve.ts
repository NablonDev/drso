// Derived views of a solve. Pure functions — callers memoise them on the solve id.
import type { LonLat, Route, RouteCost, RouteEvent } from "../api/types";

export function routesGeoJSON(routes: Route[]): GeoJSON.FeatureCollection<GeoJSON.LineString, { id: number }> {
  return {
    type: "FeatureCollection",
    features: routes.map((r) => ({
      type: "Feature",
      properties: { id: r.id },
      geometry: { type: "LineString", coordinates: r.geometry },
    })),
  };
}

export function linesGeoJSON(lines: { id: number; geometry: LonLat[] }[]): GeoJSON.FeatureCollection<GeoJSON.LineString, { id: number }> {
  return {
    type: "FeatureCollection",
    features: lines.map((r) => ({ type: "Feature", properties: { id: r.id }, geometry: { type: "LineString", coordinates: r.geometry } })),
  };
}

/** A and B as [lon, lat]. (The manifest's corridor.a/b are [lat, lon] — use the run's columns.) */
export function endpoints(run: { a_lon: number; a_lat: number; b_lon: number; b_lat: number }): { a: LonLat; b: LonLat } {
  return { a: [run.a_lon, run.a_lat], b: [run.b_lon, run.b_lat] };
}

export interface BlockerGroup {
  key: string;
  rule: string;
  ruleId: string;
  feature: string;
  km: number;
  count: number;
  firstKm: number;
  layerName: string | null;
  mode: string | null;
  confidence: number | null;
  sourceText: string | null;
}

/** The bee-line blockers grouped by rule and feature, longest first. */
export function beelineBlockers(events: RouteEvent[]): BlockerGroup[] {
  const m = new Map<string, BlockerGroup>();
  for (const e of events) {
    if (e.type !== "BEELINE_BLOCKER") continue;
    const d = (e.data ?? {}) as { rule?: string; rule_id?: string; feature?: string; length_km?: number; from_km?: number };
    const key = `${d.rule_id ?? d.rule}|${d.feature}`;
    const g = m.get(key) ?? {
      key,
      rule: d.rule ?? d.rule_id ?? "rule",
      ruleId: d.rule_id ?? "",
      feature: d.feature ?? e.source?.feature ?? "feature",
      km: 0,
      count: 0,
      firstKm: d.from_km ?? e.chainage_km ?? 0,
      layerName: e.source?.layer_name ?? null,
      mode: e.source?.mode ?? null,
      confidence: e.source?.confidence ?? null,
      sourceText: e.source?.source ?? null,
    };
    g.km += d.length_km ?? 0;
    g.count += 1;
    g.firstKm = Math.min(g.firstKm, d.from_km ?? e.chainage_km ?? g.firstKm);
    m.set(key, g);
  }
  return [...m.values()].sort((a, b) => b.km - a.km);
}

export interface CostGroup {
  key: string;
  label: string;
  rs: number;
  pill: string;
  colour: string;
}

export function costGroups(c: RouteCost): CostGroup[] {
  const g: CostGroup[] = [
    { key: "towers", label: "Towers", rs: (c.towers ?? 0) + (c.foundation ?? 0), pill: "p-acc", colour: "var(--accent)" },
    { key: "conductor", label: "Line", rs: c.conductor ?? 0, pill: "p-lock", colour: "var(--locked)" },
    { key: "land", label: "Land & RoW", rs: (c.land_row ?? 0) + (c.crops_trees ?? 0) + (c.forest ?? 0), pill: "p-go", colour: "var(--go)" },
  ];
  if (c.crossings) g.push({ key: "crossings", label: "Crossings", rs: c.crossings, pill: "p-cross", colour: "var(--cross)" });
  if (c.environment) g.push({ key: "env", label: "Environment", rs: c.environment, pill: "p-chg", colour: "var(--changed)" });
  return g;
}

export function eventClass(t: string): string {
  if (t.startsWith("TURN")) return "turn";
  if (t === "CROSSING" || t === "SECTION_POINT" || t === "NOC_REQUIRED") return "cross";
  if (t === "VIOLATION" || t === "CLEARANCE_SHORT" || t === "FOREST_ENTRY" || t === "RULE_WARNING") return "avoid";
  return "";
}

export const MODE_LABEL: Record<string, string> = { auto: "auto-fetch", replace: "your file", add: "supplemented" };

/** Nearest tower position to a chainage — for "show me this on the map". */
export function pointAtKm(r: Route, km: number): LonLat | null {
  let best: LonLat | null = null;
  let d = Infinity;
  for (const t of r.towers) {
    const x = Math.abs(t.chainage_km - km);
    if (x < d) {
      d = x;
      best = [t.lon, t.lat];
    }
  }
  return best;
}

export function tradeOff(routes: Route[]): string | null {
  if (routes.length < 2) return null;
  const cheapest = [...routes].sort((a, b) => a.cost.total - b.cost.total)[0];
  const gentlest = [...routes].sort((a, b) => a.stats.forest_ha + a.stats.agri_ha - (b.stats.forest_ha + b.stats.agri_ha))[0];
  if (cheapest.id === gentlest.id) return `Route ${cheapest.id} is both the cheapest and the lightest on forest and farmland here.`;
  const diff = (gentlest.cost.total - cheapest.cost.total) / 1e7;
  return `Route ${cheapest.id} is cheapest; Route ${gentlest.id} takes the least forest and farmland and costs ₹${diff.toFixed(1)} Cr more.`;
}
