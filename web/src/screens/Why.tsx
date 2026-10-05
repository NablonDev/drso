import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useRunCtx } from "../App";
import { useRules, useSolve } from "../api/hooks";
import type { RouteEvent, Solve } from "../api/types";
import { routeColour } from "../components/map/style";
import { ConfPill, ErrorNote, Head, Loading, Note } from "../components/ui";
import { crore, int, num } from "../lib/format";
import { beelineBlockers, costGroups, eventClass, MODE_LABEL, pointAtKm } from "../lib/solve";
import { useUi } from "../store/ui";
import { useSelectedRoute } from "../store/selection";

function SourceLine({ e }: { e: RouteEvent }) {
  const s = e.source;
  if (!s) return null;
  const low = s.confidence !== null && s.confidence < 0.8;
  const own = s.mode === "replace" || s.mode === "add";
  return (
    <div className="ch" style={{ marginTop: 3, color: low ? "var(--nogo)" : own ? "var(--accent-ink)" : "var(--faint)" }}>
      <span className={`pill ${s.mode === "replace" ? "p-chg" : s.mode === "add" ? "p-acc" : "p-go"}`} style={{ marginRight: 6 }}>
        {MODE_LABEL[s.mode] ?? s.mode}
      </span>
      {s.layer_name}
      {s.confidence !== null ? ` · ${s.confidence.toFixed(2)}` : ""}
    </div>
  );
}

function riskText(e: RouteEvent): string {
  const s = e.source;
  if (!s) return "—";
  if (s.mode === "replace") return "Low risk — your file";
  if (s.confidence !== null && s.confidence < 0.8) return "Verify on site — below 0.80";
  if (s.mode === "add") return "Partly your file — check the merge";
  return "Low risk at this confidence";
}

export default function Why() {
  const run = useRunCtx();
  const q = useSolve(run.id, run.latest_solve_id);
  if (!run.latest_solve_id) return <Head title="Why this route">This run has not been solved yet.</Head>;
  if (q.isPending) return <Loading text="Loading the decision log…" />;
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  return <Body s={q.data} />;
}

function Body({ s }: { s: Solve }) {
  const run = useRunCtx();
  const nav = useNavigate();
  const setFocus = useUi((x) => x.setFocus);
  const rules = useRules(run.id).data;
  const [selId, select] = useSelectedRoute(run, s);
  const routes = s.result.routes;
  const r = routes.find((x) => x.id === selId) ?? routes[0];
  const blockers = useMemo(() => beelineBlockers(s.result.events_global ?? []), [s]);
  const notices = useMemo(() => (s.result.events_global ?? []).filter((e) => e.type !== "BEELINE_BLOCKER"), [s]);
  const statutory = useMemo(() => {
    const m = new Set<string>();
    for (const g of rules?.groups ?? []) if (g.statutory) for (const x of g.rules) m.add(x.id);
    return m;
  }, [rules]);
  const events = r?.events ?? [];
  const sourced = events.filter((e) => e.source);

  if (!r) {
    return (
      <section>
        <Head title="Why this route">The last solve found no viable route — the blockers are listed on the Route map screen.</Head>
        <button className="btn" onClick={() => nav(`/runs/${run.id}/map`)}>See the blockers</button>
        {notices.length ? <Notices notices={notices} /> : null}
      </section>
    );
  }

  const groups = costGroups(r.cost);
  const total = r.cost.total || groups.reduce((a, g) => a + g.rs, 0) || 1;
  const mix = Object.entries(r.stats.tower_mix ?? {}).filter(([, n]) => n > 0).map(([t, n]) => `${n} × ${t}`).join(", ");
  const showAt = (km: number | null) => {
    if (km === null) return;
    const p = pointAtKm(r, km);
    if (p) setFocus({ runId: run.id, lon: p[0], lat: p[1], zoom: 15 });
    nav(`/runs/${run.id}/map`);
  };

  return (
    <section>
      <Head
        title="Why this route"
        right={
          routes.length > 1 ? (
            <div className="seg" role="group" aria-label="Route">
              {routes.map((x) => (
                <button key={x.id} aria-pressed={x.id === r.id} onClick={() => select(x.id)}>
                  <span className="swatch" style={{ background: routeColour(x.id), display: "inline-block", marginRight: 6, verticalAlign: "middle" }} />
                  Route {x.id}
                </button>
              ))}
            </div>
          ) : undefined
        }
      >
        Route {r.id}, {r.name.toLowerCase()}. Every deviation from the bee-line, in order, with its reason and its source.{" "}
        {events.length} event{events.length === 1 ? "" : "s"}, none omitted.
      </Head>

      <div className="grid2" style={{ alignItems: "start" }}>
        <div className="card">
          <div className="ch">
            <h3>What the route does, kilometre by kilometre</h3>
            <span className="hint" style={{ marginLeft: "auto" }}>{events.length} events · source shown per event</span>
          </div>
          <div className="cb">
            <div className="storybox"><div className="story" data-testid="story">
              {events.map((e, i) => (
                <div key={i} className={`ev ${eventClass(e.type)}`} role="button" tabIndex={0} onClick={() => showAt(e.chainage_km)}
                  onKeyDown={(k) => (k.key === "Enter" || k.key === " ") && showAt(e.chainage_km)} title="Show on the route map">
                  <div className="ch">
                    {e.chainage_km !== null ? `km ${num(e.chainage_km, 2)}` : "—"} · {e.type.replace(/_/g, " ").toLowerCase()}
                  </div>
                  <div className="tx">{e.text}</div>
                  <SourceLine e={e} />
                </div>
              ))}
              {events.length === 0 ? <p className="hint">The engine recorded no events for this route.</p> : null}
            </div></div>
          </div>
        </div>
        <div className="stack">
          <div className="card">
            <div className="ch"><h3>Why not just go straight?</h3></div>
            <div className="cb stack">
              <p style={{ fontSize: 13, color: "var(--ink-2)" }}>
                The bee-line is <span className="mono">{num(r.stats.beeline_km, 1)} km</span>. This route is{" "}
                <span className="mono">{num(r.stats.length_km, 1)} km</span> — <b>{num(r.stats.vs_beeline_pct, 1)}% longer</b>.{" "}
                {blockers.length ? "It cannot be straighter because the direct line passes through:" : "Nothing blocks the direct line; the route bends for cost and tower geometry."}
              </p>
              {blockers.length ? (
                <div className="tw">
                  <table data-testid="beeline-blockers">
                    <tbody>
                      {blockers.slice(0, 12).map((b) => (
                        <tr key={b.key}>
                          <td>
                            {statutory.has(b.ruleId) ? <span className="pill p-lock">Statutory</span> : <span className="pill p-nogo">No-Go</span>}
                          </td>
                          <td>
                            {b.feature.charAt(0).toUpperCase() + b.feature.slice(1)}
                            <div className="hint">
                              {num(b.km, 1)} km through{b.count > 1 ? ` in ${b.count} stretches` : ""} · {b.rule} ·{" "}
                              {b.mode ? MODE_LABEL[b.mode] ?? b.mode : "—"}
                              {b.confidence !== null ? `, ${b.confidence.toFixed(2)}` : ""}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {blockers.length > 12 ? <div className="hint">…and {blockers.length - 12} more.</div> : null}
                </div>
              ) : null}
              <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/map`)}>Show the bee-line on the map</button>
            </div>
          </div>
          <div className="card">
            <div className="ch"><h3>Where the money went</h3></div>
            <div className="cb stack">
              <div className="bar" aria-hidden>
                {groups.filter((g) => g.rs > 0).map((g) => {
                  const p = (100 * g.rs) / total;
                  return (
                    <div key={g.key} style={{ width: `${p}%`, background: g.colour }} title={`${g.label} ${Math.round(p)}%`}>
                      {p >= 8 ? `${Math.round(p)}%` : ""}
                    </div>
                  );
                })}
              </div>
              <div className="tw">
                <table data-testid="money">
                  <tbody>
                    {groups.map((g) => (
                      <tr key={g.key}>
                        <td><span className={`pill ${g.pill}`}>{g.label}</span></td>
                        <td>
                          {g.key === "towers" ? `${int(r.stats.towers)} towers · foundations ${crore(r.cost.foundation)}`
                            : g.key === "conductor" ? `${num(r.stats.length_km, 1)} km`
                              : g.key === "land" ? `${int(r.stats.corridor_ha)} ha corridor · crops ${crore(r.cost.crops_trees)} · forest ${crore(r.cost.forest)}`
                                : g.key === "crossings" ? `${int(r.stats.crossings_total)} crossings`
                                  : "clearances"}
                        </td>
                        <td className="mono" style={{ textAlign: "right" }}>{crore(g.rs)}</td>
                      </tr>
                    ))}
                    <tr>
                      <td colSpan={2}><b>Total</b></td>
                      <td className="mono" style={{ textAlign: "right" }}><b>{crore(r.cost.total)}</b></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              {mix ? <div className="hint">Tower mix: {mix}.</div> : null}
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch">
          <h3>How much of this route rests on machine-extracted data</h3>
          <span className="pill p-cross" style={{ marginLeft: "auto" }}>the honest caveat</span>
        </div>
        <div className="tw">
          <table data-testid="provenance">
            <thead>
              <tr>
                <th>Decision</th>
                <th>Rested on</th>
                <th>Source</th>
                <th>Confidence</th>
                <th style={{ textAlign: "right" }}>If it is wrong</th>
              </tr>
            </thead>
            <tbody>
              {sourced.map((e, i) => (
                <tr key={i}>
                  <td style={{ maxWidth: 360 }}>{e.text}</td>
                  <td>{e.source!.layer_name}<div className="hint">{e.source!.source}</div></td>
                  <td>
                    <span className={`pill ${e.source!.mode === "replace" ? "p-chg" : e.source!.mode === "add" ? "p-acc" : "p-go"}`}>
                      {MODE_LABEL[e.source!.mode] ?? e.source!.mode}
                    </span>
                  </td>
                  <td><ConfPill value={e.source!.confidence} /></td>
                  <td style={{ textAlign: "right" }} className="hint">{riskText(e)}</td>
                </tr>
              ))}
              {sourced.length === 0 ? (
                <tr><td colSpan={5} className="hint">No decision on this route rests on an extracted layer.</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch">
          <h3>Constraints overridden in this run</h3>
          <span className="pill p-chg" style={{ marginLeft: "auto" }}>revision record</span>
        </div>
        <div className="tw">
          <table data-testid="overrides">
            <thead>
              <tr>
                <th>Rule · parameter</th>
                <th>Default</th>
                <th>Used</th>
                <th>Reason given</th>
                <th style={{ textAlign: "right" }}>Rev</th>
              </tr>
            </thead>
            <tbody>
              {s.manifest.overrides.map((o, i) => (
                <tr key={i}>
                  <td>
                    {o.rule} · {o.param.toLowerCase()} {o.statutory ? <span className="pill p-lock">statutory</span> : null}
                  </td>
                  <td className="mono">{fmtVal(o.default)}</td>
                  <td className="mono" style={{ color: "var(--changed)" }}>{fmtVal(o.value)}</td>
                  <td>{o.reason ?? "—"}</td>
                  <td className="mono" style={{ textAlign: "right" }}>{o.rev ? `r${o.rev}` : "—"}</td>
                </tr>
              ))}
              {s.manifest.overrides.length === 0 ? (
                <tr><td colSpan={5} className="hint">No rule was overridden — the solve used the default rule set.</td></tr>
              ) : null}
            </tbody>
          </table>
        </div>
        <div className="cb" style={{ borderTop: "1px solid var(--rule-2)" }}>
          <Note kind="info">
            Every override carries the revision it first appeared in, so a reviewer looking at rev {run.rev} can see which departures are
            old and agreed and which are new. The same table is written into the run manifest and the PDF report.
          </Note>
        </div>
      </div>

      {notices.length || s.manifest.notices?.length ? <Notices notices={notices} extra={s.manifest.notices ?? []} /> : null}
    </section>
  );
}

function Notices({ notices, extra = [] }: { notices: RouteEvent[]; extra?: string[] }) {
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <div className="ch">
        <h3>Run-level notices</h3>
        <span className="hint" style={{ marginLeft: "auto" }}>rules that did not fire, coverage, terrain, rates</span>
      </div>
      <div className="tw">
        <table data-testid="notices">
          <tbody>
            {notices.map((e, i) => (
              <tr key={i}>
                <td style={{ width: 130 }}>
                  <span className={`pill ${e.type === "LAYER_EMPTY" || e.type === "NO_COVERAGE" ? "p-cross" : e.type === "RULE_OFF" ? "p-chg" : "p-lock"}`}>
                    {e.type.replace(/_/g, " ").toLowerCase()}
                  </span>
                </td>
                <td>
                  {e.text}
                  {e.source ? <div className="hint">{e.source.layer_name} · {e.source.source}</div> : null}
                </td>
              </tr>
            ))}
            {extra.map((t, i) => (
              <tr key={`x${i}`}>
                <td><span className="pill p-cross">rates</span></td>
                <td>{t}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function fmtVal(v: unknown): string {
  if (Array.isArray(v)) return `${v[0]}–${v[1]}`;
  if (v === null || v === undefined) return "—";
  return String(v);
}
