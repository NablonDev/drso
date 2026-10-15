import { useMemo, useState } from "react";
import type { Route, RulesResponse } from "../api/types";
import { num } from "../lib/format";
import { ConfPill } from "./ui";
import { ruleRows, STATUS_COLOUR, STATUS_LABEL, type Design, type Flag } from "../lib/design";

type Tab = "towers" | "crossings" | "sections" | "rules";
type Filter = "all" | "angle" | "flagged" | "crossing" | "statutory";

/**
 * What a route designer checks before accepting an alignment (15 Oct 2026): the limits in force,
 * every tower with its deflection, spans, wind span and section run, every crossing against its
 * rule's angle / span / tower-distance / null-point limits, and the section intervals.
 * Click a row to fly there on the map.
 */
export default function DesignCheck({ route, d, onFly, rules, fired, statutory }: {
  route: Route; d: Design; onFly: (lon: number, lat: number) => void; rules?: RulesResponse; fired?: Record<string, number>;
  /** statutory notes by tower number (airport NOC, pipeline footing, building setback) */
  statutory?: Map<number, string[]>;
}) {
  const [tab, setTab] = useState<Tab>("crossings");
  const [filter, setFilter] = useState<Filter>("all");
  const [allRules, setAllRules] = useState(false);
  const ruleList = useMemo(() => ruleRows(route, rules, fired), [route, rules, fired]);
  const nTouched = ruleList.filter((r) => r.touched).length;
  const L = d.limits;
  const nFlag = d.towers.filter((r) => r.flags.length).length;
  const nViol = d.crossings.filter((r) => r.status === "violation").length;
  const nRel = d.crossings.filter((r) => r.status === "relaxed").length;
  const nSecBad = d.sections.filter((s) => !s.ok).length;
  const longest = d.sections.length ? Math.max(...d.sections.map((s) => s.km)) : 0;
  const nBind = d.crossings.reduce((n, r) => n + r.binding.length, 0) + d.towers.reduce((n, r) => n + r.binding.length, 0);
  const rows = useMemo(() => {
    switch (filter) {
      case "angle": return d.towers.filter((r) => r.t.angle_point || r.t.type === "G");
      case "flagged": return d.towers.filter((r) => r.flags.length);
      case "crossing": return d.towers.filter((r) => r.crossing);
      case "statutory": return d.towers.filter((r) => statutory?.get(r.t.no)?.length);
      default: return d.towers;
    }
  }, [d.towers, filter, statutory]);
  const nStat = d.towers.filter((r) => statutory?.get(r.t.no)?.length).length;
  const bands = L.bands.map(([k, v]) => `${k} ≤${v}°`).join(", ");

  return (
    <div className="card dc" style={{ marginTop: 14 }} data-testid="design-check">
      <div className="ch">
        <h3>Design check — route {route.id}</h3>
        <span className="hint" style={{ margin: 0 }}>
          {nFlag ? <span className="warn">{nFlag} tower{nFlag > 1 ? "s" : ""} flagged</span> : "no tower flagged"} ·{" "}
          {d.crossings.length} crossings{nViol ? <>, <span className="bad">{nViol} in violation</span></> : null}{nRel ? `, ${nRel} relaxed` : ""} ·{" "}
          {d.sections.length} sections{nSecBad ? <>, <span className="bad">{nSecBad} over the interval</span></> : `, longest ${num(longest, 1)} km`}
          {nBind ? <> · <span className="bind" title="values with no slack against their limit">{nBind} at a limit</span></> : null}
        </span>
        <div className="dctabs" role="tablist">
          {(["crossings", "towers", "sections", "rules"] as Tab[]).map((k) => (
            <button key={k} role="tab" aria-pressed={tab === k} onClick={() => setTab(k)} data-testid={`dc-tab-${k}`}>
              {k === "towers" ? `Towers (${d.towers.length})` : k === "crossings" ? `Crossings (${d.crossings.length})` : k === "sections" ? `Sections (${d.sections.length})` : `Rules (${nTouched} of ${ruleList.length})`}
            </button>
          ))}
        </div>
      </div>
      <div className="chips" data-testid="dc-limits">
        <span className="chip">Design span <b>{L.designSpanM ?? "—"} m</b></span>
        <span className="chip">Min span <b>{L.minSpanM ?? "—"} m</b></span>
        <span className="chip">Wind span <b>≤ {L.windBaseM ?? "—"} m</b>{L.wind.d18 ? <span className="lim">D +18/33/45 m: {L.wind.d18}/{L.wind.d33}/{L.wind.d45}</span> : null}</span>
        <span className="chip">Section <b>≤ {L.sectionKm ?? "—"} km / {L.sectionSpans ?? "—"} spans</b><span className="lim">hilly {L.hillyKm ?? "—"} km / {L.hillySpans ?? "—"}</span></span>
        <span className="chip">Ground clearance <b>{L.clearanceM ?? "—"} m</b></span>
        <span className="chip">Max deviation <b>{L.maxDevDeg ?? "—"}°</b><span className="lim">dead-end {L.deadEndDevDeg ?? "—"}°</span></span>
        {bands ? <span className="chip">Type bands <b>{bands}</b></span> : null}
      </div>

      {tab === "towers" ? (
        <>
          <div className="row" style={{ gap: 4, padding: "8px 15px 0" }}>
            {([["all", "All"], ["angle", "Angle towers"], ["crossing", "Crossing towers"], ["flagged", "Flagged"], ["statutory", `Statutory (${nStat})`]] as [Filter, string][]).map(([k, lbl]) => (
              <button key={k} className="mini" aria-pressed={filter === k} style={filter === k ? { borderColor: "var(--accent-line)", color: "var(--accent-ink)" } : undefined} onClick={() => setFilter(k)}>{lbl}</button>
            ))}
          </div>
          <div className="scroll tw" style={{ marginTop: 8 }}>
            <table data-testid="dc-towers">
              <thead>
                <tr>
                  <th className="r">No</th><th>Type</th><th className="r">Deflection</th><th className="r">Span back</th><th className="r">Span ahead</th>
                  <th className="r">Wind span</th><th className="r">Ext.</th><th className="r">Since anchor</th><th>Statutory</th><th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
                  const t = r.t;
                  const isG = t.type === "G";
                  return (
                    <tr key={t.no} className="pick" onClick={() => onFly(t.lon, t.lat)} tabIndex={0} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onFly(t.lon, t.lat)}>
                      <td className="r mono">{t.no}</td>
                      <td className="mono">{isG ? "Gantry" : t.type}{r.anchor && !isG ? <span className="lim">anchor</span> : null}</td>
                      <td className="r mono">{isG ? "—" : <span className={bind(r.binding, "band")} title={r.binding.includes("band") ? "at the top of its type band" : undefined}>{num(t.deflection_deg, 1)}°{t.turn && t.turn !== "-" ? ` ${t.turn}` : ""}</span>}</td>
                      <td className="r mono">{r.back === null ? "—" : num(r.back, 0)}</td>
                      <td className="r mono">{r.ahead === null ? "—" : <>{num(r.ahead, 0)}{r.aheadNote ? <span className={`lim ${r.aheadNote === "over design" ? "warn" : ""}`}>{r.aheadNote}</span> : null}</>}</td>
                      <td className="r mono">{isG || r.wind === null ? "—" : <span className={`${r.windLimit !== null && r.wind > r.windLimit + 1e-3 ? "warn" : ""} ${bind(r.binding, "wind")}`}>{num(r.wind, 0)}{r.windLimit !== null ? <span className="lim">/ {r.windLimit}</span> : null}</span>}</td>
                      <td className="r mono">{t.extension_m ? `+${num(t.extension_m, 0)}` : "—"}</td>
                      <td className="r mono">{t.no === 0 ? "—" : <span className={bind(r.binding, "section")}>{num(r.runKm, 2)} km<span className="lim">{r.runSpans} sp</span></span>}</td>
                      <td>{(statutory?.get(t.no) ?? []).map((x) => <span key={x} className="fl" style={{ color: "var(--locked, var(--ink-2))" }}>{x}</span>)}</td>
                      <td>
                        {t.reasons.length ? <span className="fl" style={{ color: "var(--muted)" }}>{t.reasons.map((x) => x.replace(/[_:]/g, " ")).join(", ")}</span> : null}
                        <Flags flags={r.flags} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="foot">Wind span = mean of the two adjacent spans against the TSD limit for the type and extension. Weight-span and adjacent-span-sum checks are the engine's (notes). Span ahead notes: crossing / hilly = allowed over the design span for that reason; approach = inside the approach section at a gantry. Since anchor = chainage and spans since the last gantry or tension tower. Statutory = airport NOC radius, pipeline footing and building setback, measured from the tower to the stored features. <span className="bind">Underlined</span> = at its limit, no slack.</div>
        </>
      ) : null}

      {tab === "crossings" ? (
        <>
          <div className="scroll tw">
            <table data-testid="dc-crossings">
              <thead>
                <tr>
                  <th className="r">km</th><th>Feature</th><th>Towers</th><th className="r">Angle</th><th className="r">Span</th>
                  <th className="r">Tower dist. A / B</th><th className="r">Null point</th><th className="r">Deviation A / B</th><th className="r">Clearance</th><th>Status</th>
                </tr>
              </thead>
              <tbody>
                {d.crossings.map((r) => {
                  const c = r.c;
                  const rng = (v: [number, number] | null, u = "") => (v ? `${v[0]}–${v[1]}${u}` : "any");
                  return (
                    <tr key={r.key} className="pick" onClick={() => onFly(r.lonlat[0], r.lonlat[1])} tabIndex={0} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onFly(r.lonlat[0], r.lonlat[1])}>
                      <td className="r mono">{num(r.km, 2)}</td>
                      <td>{c.name}{c.major ? <span className="lim">major</span> : null}{r.confidence !== null && r.confidence < 0.8 ? <> <ConfPill value={r.confidence} /></> : null}</td>
                      <td className="mono">{r.a.no} {r.a.type} → {r.b.no} {r.b.type}{c.min_type && c.min_type !== "A" ? <span className="lim">min {c.min_type}</span> : null}</td>
                      <td className="r mono"><span className={`${cls(r.issues, "angle")} ${bind(r.binding, "angle")}`}>{num(c.angle_deg, 0)}°</span><span className="lim">{rng(r.angleAllowed, "°")}</span></td>
                      <td className="r mono"><span className={`${cls(r.issues, "span")} ${bind(r.binding, "span")}`}>{num(c.span_m, 0)}</span>{r.spanMax !== null ? <span className="lim">≤ {r.spanMax}</span> : null}</td>
                      <td className="r mono"><span className={`${cls(r.issues, "tower", "warn")} ${bind(r.binding, "side")}`}>{num(c.side_a_m, 0)} / {num(c.side_b_m, 0)}</span>{r.sideWindow ? <span className="lim">{rng(r.sideWindow)}</span> : null}</td>
                      <td className="r mono">{typeof c.null_offset_pct === "number" ? <><span className={`${cls(r.issues, "null")} ${bind(r.binding, "null")}`}>{num(c.null_offset_pct, 1)}%</span><span className="lim">±{r.nullPref}/{r.nullAllowed}</span></> : "—"}</td>
                      <td className="r mono"><span className={`${cls(r.issues, "deflects")} ${bind(r.binding, "deviation")}`}>{num(r.a.deflection_deg, 0)}° / {num(r.b.deflection_deg, 0)}°</span>{r.maxDev !== null ? <span className="lim">≤ {r.maxDev}°</span> : null}</td>
                      <td className="r mono">{r.span.min_clearance_m === null ? "—" : <span className={cls(r.issues, "clearance", "warn")}>{num(r.span.min_clearance_m, 1)} m</span>}</td>
                      <td><span className="dot" style={{ background: STATUS_COLOUR[r.status] }} />{STATUS_LABEL[r.status]}{r.issues.length ? <Flags flags={r.issues} /> : null}</td>
                    </tr>
                  );
                })}
                {!d.crossings.length ? <tr><td colSpan={10} className="hint">This route crosses nothing the rules track.</td></tr> : null}
              </tbody>
            </table>
          </div>
          <div className="foot">Limits in grey come from the crossing's rule. Null point = distance of the crossing from the mid-span of the crossed line, as a share of that span. Preferred = at 90° and the null point; relaxed = allowed at the rule's penalty; violation = outside what the rule allows. A confidence pill marks a crossed feature whose layer came in below 0.80: verify its position on site. <span className="bind">Underlined</span> = at its limit: the crossing cannot move without breaking the rule.</div>
        </>
      ) : null}

      {tab === "rules" ? (
        <>
          <div className="row" style={{ gap: 4, padding: "8px 15px 0" }}>
            <button className="mini" aria-pressed={!allRules} style={!allRules ? { borderColor: "var(--accent-line)", color: "var(--accent-ink)" } : undefined} onClick={() => setAllRules(false)}>Shaped this route</button>
            <button className="mini" aria-pressed={allRules} style={allRules ? { borderColor: "var(--accent-line)", color: "var(--accent-ink)" } : undefined} onClick={() => setAllRules(true)}>All rules loaded</button>
          </div>
          <div className="scroll tw" style={{ marginTop: 8 }}>
            <table data-testid="dc-rules">
              <thead>
                <tr><th>Rule</th><th>Group</th><th>Treatment</th><th className="r">In band</th><th>On this route</th><th>Changed</th></tr>
              </thead>
              <tbody>
                {ruleList.filter((r) => allRules || r.touched).map((r) => {
                  const bits = [
                    r.crossed ? `crossed ${r.crossed}×` : null,
                    r.turned ? `bent the line ${r.turned}×` : null,
                    r.detour ? "detour" : null,
                    r.violated ? `${r.violated} violation${r.violated > 1 ? "s" : ""}` : null,
                  ].filter(Boolean);
                  return (
                    <tr key={r.id} className={r.enabled ? "" : "loff"}>
                      <td>{r.name}<span className="lim">{r.id}</span></td>
                      <td style={{ fontSize: 11.5 }}>{r.group}</td>
                      <td>{r.treatment}</td>
                      <td className="r mono">{r.inBand ? r.inBand.toLocaleString("en-IN") : "—"}</td>
                      <td>{bits.length ? <span className={r.violated ? "bad" : ""}>{bits.join(" · ")}</span> : <span className="hint" style={{ margin: 0 }}>{r.inBand ? "kept clear" : r.enabled ? "nothing in the band" : "off"}</span>}</td>
                      <td>{r.changed ? <span className="pill p-chg">changed</span> : ""}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="foot">In band = features of the rule's layer the compile priced or blocked. On this route = what the rule did to the alignment: crossings, bends made for it, a detour, violations the spotting could not repair. "Kept clear" = features were there and the route stayed outside their buffer.</div>
        </>
      ) : null}

      {tab === "sections" ? (
        <>
          <div className="scroll tw">
            <table data-testid="dc-sections">
              <thead>
                <tr><th className="r">#</th><th>From → to</th><th className="r">Length</th><th className="r">Spans</th><th>Terrain</th><th>Status</th></tr>
              </thead>
              <tbody>
                {d.sections.map((s) => (
                  <tr key={s.n} className="pick" onClick={() => onFly((s.from.lon + s.to.lon) / 2, (s.from.lat + s.to.lat) / 2)}>
                    <td className="r mono">{s.n}</td>
                    <td className="mono">{s.from.no} {s.from.type} → {s.to.no} {s.to.type}<span className="lim">km {num(s.from.chainage_km, 1)} – {num(s.to.chainage_km, 1)}</span></td>
                    <td className="r mono"><span className={`${s.limitKm !== null && s.km > s.limitKm ? "bad" : ""} ${s.binding ? "bind" : ""}`}>{num(s.km, 2)} km</span>{s.limitKm !== null ? <span className="lim">≤ {s.limitKm}</span> : null}</td>
                    <td className="r mono"><span className={s.limitSpans !== null && s.spans > s.limitSpans ? "bad" : ""}>{s.spans}</span>{s.limitSpans !== null ? <span className="lim">≤ {s.limitSpans}</span> : null}</td>
                    <td>{s.hilly ? "Hilly" : "Plain"}</td>
                    <td>{s.ok ? <span className="ok">Within interval</span> : <span className="bad">Over the interval</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="foot">A section runs from one anchor (gantry or tension tower) to the next; the interval limit is the section-point rule, hilly where the section has a hilly span.</div>
        </>
      ) : null}
    </div>
  );
}

const bind = (b: string[], key: string) => (b.includes(key) ? "bind" : "");

function cls(issues: Flag[], word: string, fallback: "bad" | "warn" = "bad"): string {
  const hit = issues.filter((f) => f.text.startsWith(word));
  if (!hit.length) return "";
  return hit.some((f) => f.kind === "bad") ? "bad" : hit[0].kind === "warn" ? "warn" : fallback;
}

function Flags({ flags }: { flags: Flag[] }) {
  return <>{flags.map((f, i) => <span key={i} className={`fl ${f.kind}`}>{f.text}</span>)}</>;
}
