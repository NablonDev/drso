import { useMemo, useState } from "react";
import { useRunCtx } from "../App";
import { fileUrl, useSolve } from "../api/hooks";
import type { Solve } from "../api/types";
import CopyDialog from "../components/CopyDialog";
import { ErrorNote, Head, Loading } from "../components/ui";
import { int, num } from "../lib/format";
import { useSelectedRoute } from "../store/selection";

const PER_ROUTE: { suffix: string; label: string; desc: string; group: 1 | 2 }[] = [
  { suffix: ".dxf", label: "DXF", desc: "Route centreline, tower points and RoW corridor, layered for AutoCAD", group: 1 },
  { suffix: "_shp.zip", label: "Shapefile", desc: "Route, towers and crossings as zipped .shp sets", group: 1 },
  { suffix: ".kml", label: "KML", desc: "For Google Earth review with the client", group: 1 },
  { suffix: "_towers.csv", label: "Tower schedule (CSV)", desc: "Every tower: chainage, coordinates, type, deviation, span", group: 2 },
  { suffix: "_crossings.csv", label: "Crossing schedule (CSV)", desc: "Each crossing with angle, span and tower types", group: 2 },
];

const RUN_FILES: { name: string; label: string; desc: string; group: 1 | 2 }[] = [
  { name: "layers.gpkg", label: "Extracted layers (GeoPackage)", desc: "Everything the run produced, with source and confidence as attributes", group: 1 },
  { name: "report.pdf", label: "Route report (PDF)", desc: "Map, kilometre-by-kilometre explanation with provenance, cost, every override", group: 2 },
  { name: "manifest.json", label: "Run manifest (JSON)", desc: "Every input, source choice, rule value and override reason", group: 2 },
];

function FileLink({ sid, name, label, desc, present }: { sid: number; name: string; label: string; desc: string; present: boolean }) {
  return (
    <div className="row" style={{ gap: 10, alignItems: "flex-start", flexWrap: "nowrap" }}>
      <span style={{ flex: 1, minWidth: 0 }}>
        <span style={{ fontSize: 13, fontWeight: 500 }}>{label}</span>
        <span className="hint" style={{ display: "block", marginTop: 1 }}>{desc}</span>
        <span className="hint mono" style={{ display: "block" }}>{name}</span>
      </span>
      {present ? (
        <a className="mini" href={fileUrl(sid, name)} download data-testid="file-link">Download</a>
      ) : (
        <span className="pill p-lock">not produced</span>
      )}
    </div>
  );
}

export default function Export() {
  const run = useRunCtx();
  const q = useSolve(run.id, run.latest_solve_id);
  if (!run.latest_solve_id) return <Head title="Export">This run has not been solved yet — nothing to export.</Head>;
  if (q.isPending) return <Loading text="Loading the export list…" />;
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  return <Body s={q.data} />;
}

function Body({ s }: { s: Solve }) {
  const run = useRunCtx();
  const [selId] = useSelectedRoute(run, s);
  const [pick, setPick] = useState<number | "all">(selId);
  const [copy, setCopy] = useState(false);
  const files = useMemo(() => new Set(s.files), [s.files]);
  const routes = s.result.routes;
  const r = routes.find((x) => x.id === (pick === "all" ? selId : pick)) ?? routes[0];
  const allZip = s.files.find((f) => f.endsWith("_all.zip")) ?? `${run.id}_all.zip`;
  const ids = pick === "all" ? routes.map((x) => x.id) : [pick];
  const preview = useMemo(() => {
    const m = s.manifest;
    const short = {
      run_id: m.run_id,
      rev: m.rev,
      ruleset_version: m.ruleset_version,
      costs_version: m.costs_version,
      corridor: m.corridor,
      line: m.line,
      sources: (m.sources ?? []).slice(0, 4).map((x) => ({
        layer: x.layer,
        mode: x.mode,
        model: x.model_version ?? x.model,
        features: x.features,
        confidence: x.confidence,
        ...(x.file ? { file: x.file } : {}),
      })),
      overrides: (m.overrides ?? []).map((o) => ({ rule: `${o.rule_id}.${o.key}`, default: o.default, value: o.value, reason: o.reason })),
    };
    return JSON.stringify(short, null, 1).replace(/\n\s+/g, (x) => (x.length > 3 ? "\n  " : x));
  }, [s.manifest]);

  return (
    <section>
      <Head title="Export">
        {r ? `Route ${r.id} · ` : ""}
        {run.a_name || "A"} → {run.b_name || "B"} · rev {run.rev}
        {r ? ` · ${num(r.stats.length_km, 1)} km · ${int(r.stats.towers)} towers` : ""}.
      </Head>

      <div className="card" style={{ borderColor: "var(--accent-line)", marginBottom: 14 }}>
        <div className="cb">
          <div className="row" style={{ justifyContent: "space-between", gap: 12 }}>
            <div style={{ flex: 1, minWidth: 230 }}>
              <div style={{ fontSize: 13, fontWeight: 500 }}>Everything, all {routes.length} route{routes.length === 1 ? "" : "s"}, one file</div>
              <div className="hint" style={{ marginTop: 3 }}>
                DXF, shapefile and KML for each alignment, plus the extracted layers, the tower and crossing schedules, the report and the
                run manifest — <b className="mono">{allZip}</b>.
              </div>
            </div>
            {files.has(allZip) ? (
              <a className="btn" href={fileUrl(s.id, allZip)} download data-testid="download-all">
                Download all {routes.length > 1 ? `${routes.length} routes` : ""}
              </a>
            ) : (
              <span className="pill p-lock">zip not produced</span>
            )}
          </div>
        </div>
      </div>

      {routes.length ? (
        <div className="row" style={{ marginBottom: 12, gap: 8, flexWrap: "wrap" }}>
          <span className="hint" style={{ margin: 0 }}>Or pick per route:</span>
          <div className="seg" role="group" aria-label="Route">
            {routes.map((x) => (
              <button key={x.id} aria-pressed={pick === x.id} onClick={() => setPick(x.id)}>Route {x.id}</button>
            ))}
            {routes.length > 1 ? <button aria-pressed={pick === "all"} onClick={() => setPick("all")}>All {routes.length === 4 ? "four" : routes.length}</button> : null}
          </div>
        </div>
      ) : null}

      <div className="grid2">
        {[1, 2].map((g) => (
          <div key={g} className="card">
            <div className="ch"><h3>{g === 1 ? "Drawings & GIS" : "Documents & data"}</h3></div>
            <div className="cb stack" data-testid={`exp${g}`}>
              {ids.flatMap((id) =>
                PER_ROUTE.filter((p) => p.group === g).map((p) => {
                  const name = `route${id}${p.suffix}`;
                  return <FileLink key={name} sid={s.id} name={name} label={ids.length > 1 ? `Route ${id} · ${p.label}` : p.label} desc={p.desc} present={files.has(name)} />;
                }),
              )}
              {RUN_FILES.filter((f) => f.group === g).map((f) => (
                <FileLink key={f.name} sid={s.id} name={f.name} label={f.label} desc={f.desc} present={files.has(f.name)} />
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch">
          <h3>Every file in this solve</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>{s.files.length} files · solve {s.id}</span>
        </div>
        <div className="tw">
          <table data-testid="files-table">
            <tbody>
              {s.files.map((f) => (
                <tr key={f}>
                  <td className="mono" style={{ fontSize: 12.5 }}>{f}</td>
                  <td style={{ textAlign: "right" }}><a className="mini" href={fileUrl(s.id, f)} download>Download</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="ch">
          <h3>Run manifest</h3>
          <span className="pill p-acc" style={{ marginLeft: "auto" }}>always included</span>
        </div>
        <div className="cb stack">
          <p style={{ fontSize: 13, color: "var(--ink-2)", maxWidth: "64ch" }}>
            One JSON file holding every input, the source choice and outcome for each layer, every rule value and every override reason —
            including which model version drew each layer.
          </p>
          <div className="tmpl">{preview}</div>
          <div className="row">
            {files.has("manifest.json") ? <a className="btn" href={fileUrl(s.id, "manifest.json")} download>Download the manifest</a> : null}
            <button className="btn ghost" onClick={() => setCopy(true)}>Copy this run and edit</button>
          </div>
        </div>
      </div>
      <CopyDialog open={copy} onClose={() => setCopy(false)} run={run} />
    </section>
  );
}
