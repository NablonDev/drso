// Design-check views of one route (15 Oct 2026): what a route designer verifies tower by tower and
// crossing by crossing — deflection against the type band, spans against the design and wind span,
// section runs against the interval, each crossing against its rule's angle, span, side-distance and
// null-point limits. Pure functions; callers memoise on the route and the rules.
import type { LonLat, Route, RuleParam, RulesResponse, Span, SpanCrossing, Tower } from "../api/types";

export type Status = "ok" | "relaxed" | "violation";

interface EngineCfg {
  tsd?: Record<string, { wind_span_m?: Record<string, number> }>;
  spotting?: { ground_clearance_m?: Record<string, number> };
  crossings?: { single_angle_tolerance_deg?: number };
}

export interface Limits {
  designSpanM: number | null;
  minSpanM: number | null;
  windBaseM: number | null;
  wind: Record<string, number>;
  sectionKm: number | null;
  sectionSpans: number | null;
  hillyKm: number | null;
  hillySpans: number | null;
  clearanceM: number | null;
  maxDevDeg: number | null;
  deadEndDevDeg: number | null;
  gantryMinSpanM: number | null;
  /** tower type by deflection: [type, up to °] */
  bands: [string, number][];
  singleAngleTolDeg: number;
}

const TENSION = /^(B|C|D)/;
const TYPE_ORDER = ["A", "B", "B1", "C", "C1", "D", "D45", "D60"];
const rank = (t: string) => Math.max(0, TYPE_ORDER.indexOf(t));
/** a tower the line can be sectioned at: gantry or tension type */
export const isAnchor = (t: Tower) => t.type === "G" || TENSION.test(t.type);

function ruleParams(rules: RulesResponse | undefined, id: string): RuleParam[] {
  for (const g of rules?.groups ?? []) for (const r of g.rules) if (r.id === id) return r.params;
  return [];
}
function pval(ps: RuleParam[], key: string): RuleParam["value"] | null {
  return ps.find((x) => x.key === key)?.value ?? null;
}
function pnum(ps: RuleParam[], key: string): number | null {
  const v = pval(ps, key);
  return typeof v === "number" ? v : null;
}
function prange(ps: RuleParam[], key: string): [number, number] | null {
  const v = pval(ps, key);
  return Array.isArray(v) ? [v[0], v[1]] : null;
}

export function limitsOf(rules: RulesResponse | undefined, manifest: unknown, kv: number): Limits {
  const eng = ((manifest as { engine?: EngineCfg } | undefined)?.engine ?? {}) as EngineCfg;
  const tsd = eng.tsd?.[String(kv)] ?? {};
  const wind = tsd.wind_span_m ?? {};
  const bands = ruleParams(rules, "tower_type_bands")
    .filter((p) => typeof p.value === "number")
    .map((p) => [p.key, p.value as number] as [string, number])
    .sort((a, b) => a[1] - b[1]);
  const sec = ruleParams(rules, "section_interval");
  const sub = ruleParams(rules, "substations");
  return {
    designSpanM: pnum(ruleParams(rules, "design_span"), "plain_m"),
    minSpanM: pnum(ruleParams(rules, "crossing_span_limits"), "min_span_m"),
    windBaseM: wind.base ?? null,
    wind,
    sectionKm: pnum(sec, "plain_km"),
    sectionSpans: pnum(sec, "plain_spans"),
    hillyKm: pnum(sec, "hilly_km"),
    hillySpans: pnum(sec, "hilly_spans"),
    clearanceM: eng.spotting?.ground_clearance_m?.[String(kv)] ?? null,
    maxDevDeg: pnum(ruleParams(rules, "max_deviation"), "limit_deg"),
    deadEndDevDeg: pnum(sub, "dead_end_max_dev_deg"),
    gantryMinSpanM: pnum(sub, "gantry_min_span_m"),
    bands,
    singleAngleTolDeg: eng.crossings?.single_angle_tolerance_deg ?? 15,
  };
}

/** The wind-span limit the spotting applied to this tower (TSD: reduced for extended D towers). */
export function windLimit(t: Tower, lim: Limits): number | null {
  const w = lim.wind;
  if (!("base" in w) && lim.windBaseM === null) return null;
  if ((t.type === "D45" || t.type === "D60") && t.extension_m >= 45) return w.d45 ?? w.base ?? null;
  if ((t.type === "D45" || t.type === "D60") && t.extension_m >= 33) return w.d33 ?? w.base ?? null;
  if ((t.type === "D45" || t.type === "D60") && t.extension_m >= 18) return w.d18 ?? w.base ?? null;
  return w.base ?? lim.windBaseM;
}

export interface Flag {
  kind: "warn" | "bad";
  text: string;
}

/** a value within this much of its limit counts as binding: the route has no room there */
const NEAR = { deg: 2, m: 10, sideM: 3, pct: 2, windM: 5, frac: 0.95 };

export interface TowerRow {
  t: Tower;
  back: number | null;
  ahead: number | null;
  wind: number | null;
  windLimit: number | null;
  /** chainage and spans since the previous anchor (for an anchor: the section it closes) */
  runKm: number;
  runSpans: number;
  anchor: boolean;
  flags: Flag[];
  crossing: boolean;
  /** which values sit at their limit: wind, band, section */
  binding: string[];
}

const nice = (s: string) => s.replace(/_/g, " ");

/** warnings / violations as the engine writes them (`code:value`) → readable */
export function flagText(code: string): string {
  const [k, ...rest] = code.split(":");
  const v = rest.join(":");
  switch (k) {
    case "wind_span": return `wind span ${v}`;
    case "weight_span": return `weight span ${v}`;
    case "span_sum": return `adjacent spans ${v.replace("/", " of ")}`;
    case "span_sum_single": return `single span ${v.replace("/", " of ")}`;
    case "crossing_angle": return `crossing angle (${nice(v)})`;
    case "crossing_deviation": { const [rule, deg] = v.split(":"); return `deviation ${deg ?? ""} at ${nice(rule)} crossing`; }
    case "dead_end_deviation": return `dead-end deviation ${v}`;
    case "tower_buffer": return "tower inside a setback";
    case "conductor_buffer": return "conductor inside a setback";
    default: return nice(code);
  }
}

export function towerRows(route: Route, lim: Limits): TowerRow[] {
  const spanBy = new Map<number, Span>();
  for (const s of route.spans) spanBy.set(s.a, s);
  const T = route.towers;
  const out: TowerRow[] = [];
  let last = 0;
  for (let i = 0; i < T.length; i++) {
    const t = T[i];
    const back = i > 0 ? (spanBy.get(T[i - 1].no)?.length_m ?? T[i - 1].span_ahead_m ?? null) : null;
    const ahead = spanBy.get(t.no)?.length_m ?? t.span_ahead_m ?? null;
    const wind = back !== null && ahead !== null ? (back + ahead) / 2 : null;
    const wl = windLimit(t, lim);
    const anchor = isAnchor(t);
    const flags: Flag[] = [
      ...t.violations.map((v) => ({ kind: "bad" as const, text: flagText(v) })),
      ...t.warnings.map((v) => ({ kind: "warn" as const, text: flagText(v) })),
    ];
    if (t.type !== "G" && wind !== null && wl !== null && wind > wl + 1e-3 && !t.warnings.some((w) => w.startsWith("wind_span")))
      flags.push({ kind: "warn", text: `wind span ${wind.toFixed(0)} m over ${wl} m` });
    if (t.type !== "G" && lim.maxDevDeg !== null && t.deflection_deg > lim.maxDevDeg + 1e-6)
      flags.push({ kind: "bad", text: `deflection ${t.deflection_deg.toFixed(0)}° over ${lim.maxDevDeg}°` });
    const binding: string[] = [];
    if (t.type !== "G" && wind !== null && wl !== null && wind >= wl - NEAR.windM) binding.push("wind");
    // deflection just under the top of its type band: a nudge would need the next tower type
    const band = lim.bands.find(([k]) => k === t.type);
    if (band && t.deflection_deg >= band[1] - NEAR.deg && t.deflection_deg > 0) binding.push("band");
    const runKm = t.chainage_km - T[last].chainage_km, runSpans = i - last;
    if (anchor && i > 0 && ((lim.sectionKm !== null && runKm >= lim.sectionKm * NEAR.frac) || (lim.sectionSpans !== null && runSpans >= lim.sectionSpans))) binding.push("section");
    out.push({
      t, back, ahead, wind, windLimit: wl,
      runKm, runSpans, anchor, flags,
      crossing: Boolean(spanBy.get(t.no)?.crossings.length) || (i > 0 && Boolean(spanBy.get(T[i - 1].no)?.crossings.length)),
      binding,
    });
    if (anchor) last = i;
  }
  return out;
}

export interface SectionRow {
  n: number;
  from: Tower;
  to: Tower;
  km: number;
  spans: number;
  hilly: boolean;
  limitKm: number | null;
  limitSpans: number | null;
  ok: boolean;
  binding: boolean;
}

export function sectionRows(route: Route, lim: Limits): SectionRow[] {
  const T = route.towers;
  const out: SectionRow[] = [];
  let last = 0;
  for (let i = 1; i < T.length; i++) {
    if (!isAnchor(T[i])) continue;
    const hilly = T.slice(last + 1, i + 1).some((t) => t.reasons.some((r) => r === "hilly" || r === "section_hilly"));
    const limitKm = hilly ? lim.hillyKm : lim.sectionKm;
    const limitSpans = hilly ? lim.hillySpans : lim.sectionSpans;
    const km = T[i].chainage_km - T[last].chainage_km;
    const spans = i - last;
    out.push({
      n: out.length + 1, from: T[last], to: T[i], km, spans, hilly, limitKm, limitSpans,
      ok: (limitKm === null || km <= limitKm + 1e-6) && (limitSpans === null || spans <= limitSpans),
      binding: (limitKm !== null && km >= limitKm * NEAR.frac) || (limitSpans !== null && spans >= limitSpans),
    });
    last = i;
  }
  return out;
}

export interface CrossingRow {
  key: string;
  span: Span;
  c: SpanCrossing;
  a: Tower;
  b: Tower;
  km: number;
  lonlat: LonLat;
  angleAllowed: [number, number] | null;
  anglePref: [number, number] | null;
  spanMax: number | null;
  sideWindow: [number, number] | null;
  nullPref: number | null;
  nullAllowed: number | null;
  maxDev: number | null;
  status: Status;
  issues: Flag[];
  /** which values sit at their limit: angle, span, side, null, deviation */
  binding: string[];
}

export function crossingRows(route: Route, rules: RulesResponse | undefined, lim: Limits): CrossingRow[] {
  const byNo = new Map(route.towers.map((t) => [t.no, t]));
  const out: CrossingRow[] = [];
  for (const span of route.spans) {
    const a = byNo.get(span.a), b = byNo.get(span.b);
    if (!a || !b) continue;
    span.crossings.forEach((c, j) => {
      const ps = ruleParams(rules, c.rule);
      const L = c.span_m || span.length_m;
      const along = typeof c.along_a_m === "number" ? c.along_a_m : L / 2;
      const f = Math.max(0, Math.min(1, along / Math.max(1, L)));
      const lonlat: LonLat = [a.lon + f * (b.lon - a.lon), a.lat + f * (b.lat - a.lat)];
      const av = pval(ps, "angle_deg");
      const angleAllowed: [number, number] | null = Array.isArray(av) ? [av[0], av[1]]
        : typeof av === "number" ? [av - lim.singleAngleTolDeg, av + lim.singleAngleTolDeg] : null;
      const pa = pnum(ps, "preferred_angle_deg"), pt = pnum(ps, "preferred_angle_tol_deg");
      const anglePref: [number, number] | null = pa !== null && pt !== null && angleAllowed ? [pa - pt, pa + pt] : null;
      const spanMax = pnum(ps, "max_span_m");
      const sideWindow = c.side_window_m ?? prange(ps, "crossing_side_m");
      const hasNull = typeof c.null_offset_pct === "number";
      const nullPref = hasNull ? pnum(ps, "crossing_point_pref_pct") : null;
      const nullAllowed = hasNull ? pnum(ps, "crossing_point_allowed_pct") : null;
      const maxDev = c.max_dev_deg ?? pnum(ps, "max_dev_adjacent_deg");
      const issues: Flag[] = [];
      const deg = c.angle_deg;
      if (angleAllowed && (deg < angleAllowed[0] - 1e-6 || deg > angleAllowed[1] + 1e-6))
        issues.push({ kind: "bad", text: `angle ${deg.toFixed(0)}° outside ${angleAllowed[0]}–${angleAllowed[1]}°` });
      else if (anglePref && (deg < anglePref[0] || deg > anglePref[1]))
        issues.push({ kind: "warn", text: `angle ${deg.toFixed(0)}°, preferred ${anglePref[0]}–${anglePref[1]}°` });
      if (spanMax !== null && L > spanMax + 0.05) issues.push({ kind: "bad", text: `span ${L.toFixed(0)} m over ${spanMax} m` });
      if (hasNull) {
        const off = c.null_offset_pct as number;
        if (nullAllowed !== null && off > nullAllowed + 1e-6) issues.push({ kind: "bad", text: `null point ${off}% off, allowed ±${nullAllowed}%` });
        else if (nullPref !== null && off > nullPref + 1e-6) issues.push({ kind: "warn", text: `null point ${off}% off, preferred ±${nullPref}%` });
      }
      if (sideWindow) {
        for (const [side, d] of [["A", c.side_a_m], ["B", c.side_b_m]] as const)
          if (d < sideWindow[0] - 0.5 || d > sideWindow[1] + 0.5)
            issues.push({ kind: "warn", text: `tower ${side === "A" ? a.no : b.no} at ${d.toFixed(0)} m, window ${sideWindow[0]}–${sideWindow[1]} m` });
      }
      for (const t of [a, b]) {
        if (maxDev !== null && t.type !== "G" && t.deflection_deg > maxDev + 1e-6)
          issues.push({ kind: "bad", text: `tower ${t.no} deflects ${t.deflection_deg.toFixed(0)}°, max ${maxDev}°` });
        if (t.type !== "G" && c.min_type && rank(t.type) < rank(c.min_type))
          issues.push({ kind: "bad", text: `tower ${t.no} is ${t.type}, needs ${c.min_type}` });
        for (const v of t.violations)
          if (v.startsWith("crossing_") && v.includes(`:${c.rule}`) && !issues.some((x) => x.kind === "bad" && x.text.startsWith("angle")))
            issues.push({ kind: "bad", text: flagText(v) });
      }
      // the DEM clearance at mid-span sag: the spotting raises towers for it where it can, so a shortfall
      // left here is a check for the designer (profile), not a rule violation
      if (lim.clearanceM !== null && span.min_clearance_m !== null && span.min_clearance_m < lim.clearanceM - 1e-6)
        issues.push({ kind: "warn", text: `clearance ${span.min_clearance_m} m under ${lim.clearanceM} m, verify on the profile` });
      if (c.violation && !issues.some((x) => x.kind === "bad")) issues.push({ kind: "bad", text: nice(c.violation) });
      const status: Status = issues.some((x) => x.kind === "bad") ? "violation" : issues.length ? "relaxed" : "ok";
      const binding: string[] = [];
      if (angleAllowed && (deg <= angleAllowed[0] + NEAR.deg || deg >= angleAllowed[1] - NEAR.deg)) binding.push("angle");
      if (spanMax !== null && L >= spanMax - NEAR.m) binding.push("span");
      if (sideWindow && [c.side_a_m, c.side_b_m].some((d) => d <= sideWindow[0] + NEAR.sideM || d >= sideWindow[1] - NEAR.sideM)) binding.push("side");
      if (hasNull && nullAllowed !== null && (c.null_offset_pct as number) >= nullAllowed - NEAR.pct) binding.push("null");
      if (maxDev !== null && [a, b].some((t) => t.type !== "G" && t.deflection_deg >= maxDev - NEAR.deg)) binding.push("deviation");
      out.push({ key: `${route.id}-${span.a}-${j}`, span, c, a, b, km: a.chainage_km + along / 1000, lonlat, angleAllowed, anglePref, spanMax, sideWindow, nullPref, nullAllowed, maxDev, status, issues, binding });
    });
  }
  return out;
}

export interface Design {
  limits: Limits;
  towers: TowerRow[];
  sections: SectionRow[];
  crossings: CrossingRow[];
}

export function designOf(route: Route, rules: RulesResponse | undefined, manifest: unknown, kv: number): Design {
  const limits = limitsOf(rules, manifest, kv);
  return { limits, towers: towerRows(route, limits), sections: sectionRows(route, limits), crossings: crossingRows(route, rules, limits) };
}

export const STATUS_COLOUR: Record<Status, string> = { ok: "#1F9E8E", relaxed: "#7A5AD8", violation: "#D7263D" };
export const STATUS_LABEL: Record<Status, string> = { ok: "Preferred", relaxed: "Relaxed", violation: "Violation" };
