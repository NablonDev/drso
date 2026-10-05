import { useFeature } from "../../api/hooks";
import { area, conf, date, humanClass, length, num, when } from "../../lib/format";
import { ConfPill, ErrorNote, SourcePill, Spinner } from "../ui";

/** Provenance of one extracted feature: class, extent, chainage, source, model, dates, confidence. */
export default function FeatureInfo({ runId, fid, layerName }: { runId: string; fid: number; layerName?: string }) {
  const q = useFeature(runId, fid);
  if (q.isPending)
    return (
      <div className="hint">
        <Spinner /> Loading feature {fid}…
      </div>
    );
  if (q.isError) return <ErrorNote error={q.error} />;
  const f = q.data;
  const isLine = f.geom.type.includes("Line");
  const isPoint = f.geom.type.includes("Point");
  const extent = isPoint ? "point" : isLine ? length(f.length_m) : area(f.area_m2);
  const [c0, c1] = f.chainage_km ?? [null, null];
  return (
    <div data-testid="feature-info">
      <div className="eyebrow">Selected feature</div>
      <div style={{ fontSize: 13, fontWeight: 500, margin: "4px 0 2px" }}>{f.name || humanClass(f.class)}</div>
      <div className="tw">
        <table className="kv">
          <tbody>
            <tr><td>Layer</td><td>{layerName ?? f.layer_key}</td></tr>
            <tr><td>Class</td><td>{humanClass(f.class)}</td></tr>
            <tr><td>Extent</td><td className="mono">{extent}</td></tr>
            <tr>
              <td>Chainage</td>
              <td className="mono">{c0 === null ? "—" : c0 === c1 ? `km ${num(c0, 2)}` : `km ${num(c0, 2)} – ${num(c1, 2)}`}</td>
            </tr>
            <tr>
              <td>Source</td>
              <td>
                <SourcePill mode={f.origin === "user" ? "replace" : f.mode} /> <span className="hint">{f.source ?? f.file_name ?? ""}</span>
              </td>
            </tr>
            <tr><td>Model</td><td className="mono" style={{ fontSize: 11.5 }}>{f.model ?? "— (no model)"}{f.model_version ? ` · ${f.model_version}` : ""}</td></tr>
            <tr><td>Image</td><td>{f.image_source ?? "—"}<div className="hint mono">{date(f.image_date)}</div></td></tr>
            <tr><td>Model run</td><td className="mono">{when(f.model_run_at)}</td></tr>
            <tr>
              <td>Confidence</td>
              <td>
                <ConfPill value={f.confidence ?? f.layer_conf} />
                {f.confidence !== null && f.layer_conf !== null && f.confidence !== f.layer_conf ? (
                  <span className="hint"> layer {conf(f.layer_conf)}</span>
                ) : null}
              </td>
            </tr>
            <tr><td>Feature id</td><td className="mono">{f.id}{f.src_id ? ` · src ${f.src_id}` : ""}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
