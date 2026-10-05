import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { apiCalls, qk, useApiMutation, useImageryCatalog } from "../api/hooks";
import type { Job } from "../api/types";
import { date, num } from "../lib/format";
import { Btn, ErrorNote, Note, Spinner } from "./ui";

const LABEL: Record<string, string> = {
  esri_wayback_z18: "Esri Wayback z18 (0.6 m)",
  sentinel2_l2a: "Sentinel-2 L2A (10 m, NIR/SWIR)",
  sentinel1_rtc_vv: "Sentinel-1 RTC VV (10 m)",
  worldcover: "ESA WorldCover 2021 (10 m)",
  cartosat3_pan: "Cartosat-3 PAN (0.45 m)",
  nisar_coh: "NISAR coherence",
};

/**
 * Imagery held on this machine for the run's band — the only imagery the LULC models read.
 * Fetchable products are pulled online under the operator's licences; PAN and NISAR are
 * registered from files already on the machine.
 */
export default function ImageryCatalogPanel({ runId, busy }: { runId: string; busy: boolean }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const q = useImageryCatalog(runId);
  const [pick, setPick] = useState<Record<string, boolean>>({});
  const fetch = useApiMutation<Job, string[]>({
    mutationFn: (products) => apiCalls.fetchImagery(runId, products),
    errorTitle: "Imagery fetch not started",
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.active });
      void qc.invalidateQueries({ queryKey: qk.run(runId) });
      nav(`/runs/${runId}/extracting`);
    },
  });
  if (q.isPending)
    return (
      <div className="hint" style={{ margin: "12px 0" }}>
        <Spinner /> Reading the imagery catalog…
      </div>
    );
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  const chosen = Object.entries(pick).filter(([, v]) => v).map(([k]) => k);
  const cred = q.data.credentials;
  return (
    <div className="card" style={{ marginTop: 16 }} data-testid="imagery-catalog">
      <div className="ch">
        <h3>Imagery on this machine</h3>
        <span className="hint">the LULC models read only this — they never fetch</span>
      </div>
      <div className="tw">
        <table>
          <thead>
            <tr>
              <th />
              <th>Product</th>
              <th>Used by</th>
              <th style={{ textAlign: "right" }}>Items</th>
              <th style={{ textAlign: "right" }}>Band covered</th>
              <th>Newest</th>
              <th>How it gets here</th>
            </tr>
          </thead>
          <tbody>
            {q.data.products.map((p) => (
              <tr key={p.product} data-testid={`imagery-${p.product}`}>
                <td>
                  {p.fetchable ? (
                    <input
                      type="checkbox"
                      aria-label={`Fetch ${LABEL[p.product] ?? p.product}`}
                      checked={Boolean(pick[p.product])}
                      onChange={(e) => setPick((s) => ({ ...s, [p.product]: e.target.checked }))}
                    />
                  ) : null}
                </td>
                <td>{LABEL[p.product] ?? p.product}</td>
                <td style={{ fontSize: 12 }}>{p.used_by.length ? p.used_by.join(", ") : "—"}</td>
                <td className="mono" style={{ textAlign: "right" }}>{p.items}</td>
                <td className="mono" style={{ textAlign: "right", color: p.covered_pct >= 99.5 ? "var(--go)" : p.covered_pct > 0 ? "var(--cross)" : "var(--nogo)" }}>
                  {num(p.covered_pct, 0)}%
                </td>
                <td className="mono" style={{ fontSize: 12 }}>{date(p.newest)}</td>
                <td style={{ fontSize: 12 }}>
                  {p.fetchable ? "Fetch online (licensed)" : "Register files on this machine"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Note kind="info" style={{ marginTop: 10 }}>
        Fetched imagery is stored in the bucket with its release and dates, and every layer records which items it was built
        from. Esri credentials: <b>{cred.esri_token ? "set" : "not set — public endpoint"}</b>. Cartosat-3 PAN scenes and NISAR rasters
        are registered with <span className="mono">python -m beeline_api.imagery_store register-pan / register-raster</span>.
      </Note>
      <div className="actions" style={{ marginTop: 10 }}>
        <Btn
          pending={fetch.isPending}
          disabled={busy || chosen.length === 0}
          onClick={() => fetch.mutate(chosen)}
          data-testid="fetch-imagery"
        >
          Fetch {chosen.length ? `${chosen.length} product${chosen.length > 1 ? "s" : ""}` : "imagery"} for this band
        </Btn>
      </div>
    </div>
  );
}
