import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useConfig, useLayers } from "../api/hooks";
import type { RunDetail, RunLayer, SourceMode, UploadInfo } from "../api/types";
import MappingDialog from "../components/MappingDialog";
import { Btn, ErrorNote, Head, Loading, Note } from "../components/ui";
import { hm, int, plural } from "../lib/format";
import { uploadKey, useUploads } from "../store/uploads";
import { useUi } from "../store/ui";

const MODE_LABEL: Record<SourceMode, string> = { auto: "Auto-fetch", replace: "Replace", add: "Add more" };

function needsExtract(l: RunLayer) {
  return !l.extracted || l.stale || l.status === "failed";
}

export default function Sources() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const cfg = useConfig().data;
  const lq = useLayers(run.id);
  const uploads = useUploads((s) => s.entries);
  const startUpload = useUploads((s) => s.start);
  const clearUpload = useUploads((s) => s.clear);
  const uploadNames = useUi((s) => s.uploadNames);
  const [pending, setPending] = useState<Record<string, SourceMode>>({});
  const [mapFor, setMapFor] = useState<{ key: string; mode: SourceMode; upload: UploadInfo } | null>(null);
  const busyJob = run.job;

  const onRunChanged = (r: RunDetail) => {
    qc.setQueryData(qk.run(run.id), r);
    void qc.invalidateQueries({ queryKey: qk.layers(run.id) });
    void qc.invalidateQueries({ queryKey: qk.runs });
  };

  const setMode = useApiMutation<RunDetail, { key: string; mode: SourceMode; upload_id?: number; mapping?: RunLayer["source_setting"]["mapping"] }>({
    mutationFn: ({ key, mode, upload_id, mapping }) => apiCalls.setSource(run.id, key, mode, upload_id, mapping ?? undefined),
    errorTitle: "Source not changed",
    onSuccess: (r, v) => {
      setPending((p) => {
        const n = { ...p };
        delete n[v.key];
        return n;
      });
      onRunChanged(r);
    },
  });

  const allAuto = useApiMutation<RunDetail | null, string[]>({
    mutationFn: async (keys) => {
      let last: RunDetail | null = null;
      for (const k of keys) last = await apiCalls.setSource(run.id, k, "auto");
      return last;
    },
    errorTitle: "Not every layer was reset",
    onSuccess: (r) => {
      setPending({});
      if (r) onRunChanged(r);
    },
    onError: () => void qc.invalidateQueries({ queryKey: qk.run(run.id) }),
  });

  // open the mapping dialog as soon as an upload started on this screen finishes
  const prompted = useRef<Set<string>>(new Set());
  useEffect(() => {
    for (const up of Object.values(uploads)) {
      if (!up.key.startsWith(`${run.id}:`) || up.status !== "done" || !up.info || prompted.current.has(up.key + up.info.id)) continue;
      prompted.current.add(up.key + up.info.id);
      const key = up.key.slice(run.id.length + 1);
      const saved = lq.data?.layers.find((l) => l.key === key)?.source_setting.upload_id === up.info.id;
      if (saved) continue;
      setMapFor({ key, mode: up.mode, upload: up.info });
      break;
    }
  }, [uploads, run.id, lq.data]);

  const cfgByKey = useMemo(() => Object.fromEntries((cfg?.layers ?? []).map((l) => [l.key, l])), [cfg]);
  const layers = lq.data?.layers ?? [];
  const band = lq.data?.band_km2 ?? 0;

  const modeOf = (l: RunLayer): SourceMode => pending[l.key] ?? l.source_setting.mode ?? "auto";
  const counts = { auto: 0, replace: 0, add: 0 };
  let mins = 0;
  const anyExtracted = layers.some((l) => l.extracted);
  for (const l of layers) {
    const m = modeOf(l);
    counts[m]++;
    const willRun = !anyExtracted || needsExtract(l) || m !== (l.source_setting.mode ?? "auto");
    if (willRun && m !== "replace") mins += (l.minutes_per_km2 ?? 0) * band;
  }
  const carried = layers.filter((l) => l.carried && !l.stale).length;
  const srcRun = run.parent_id ?? run.reuse_from;

  if (lq.isPending) return <Loading text="Loading layers…" />;
  if (lq.isError) return <ErrorNote error={lq.error} retry={() => void lq.refetch()} />;

  const choose = (l: RunLayer, m: SourceMode) => {
    const cur = l.source_setting;
    if (m === "auto") {
      if (cur.mode === "auto" && !pending[l.key]) return;
      setPending((p) => ({ ...p, [l.key]: "auto" }));
      setMode.mutate({ key: l.key, mode: "auto" });
      return;
    }
    if (cur.upload_id && cur.mode !== m) {
      // file already attached — keep it, change only how it is merged
      setPending((p) => ({ ...p, [l.key]: m }));
      setMode.mutate({ key: l.key, mode: m, upload_id: cur.upload_id, mapping: cur.mapping });
      return;
    }
    setPending((p) => ({ ...p, [l.key]: m }));
  };

  return (
    <section>
      <Head title="Where should each layer come from?">
        Set a source for every layer. You will come back to this screen — after the layers are extracted you review them here and
        correct whatever the models got wrong.
      </Head>

      {srcRun ? (
        <div className="card" style={{ borderColor: "var(--accent-line)", marginBottom: 14 }}>
          <div className="cb">
            <div className="row" style={{ gap: 8 }}>
              <span className="pill p-chg">rev {run.rev}</span>
              <b style={{ fontSize: 13 }}>{run.parent_id ? `Copied from ${run.parent_id}` : `Layers reused from ${run.reuse_from}`}</b>
            </div>
            <div className="hint" style={{ marginTop: 4 }}>
              {plural(carried, "layer")} carry over from {srcRun} untouched, with their features, sources and confidences. Only the
              ones you change below are extracted again.
            </div>
          </div>
        </div>
      ) : null}

      {busyJob ? (
        <Note kind="warn" style={{ marginBottom: 14 }}>
          A {busyJob.kind === "extract" ? "layer extraction" : "solve"} is running for this run — sources cannot change until it finishes.
        </Note>
      ) : null}

      <div className="card" style={{ marginBottom: 14 }}>
        <div className="cb">
          <div className="row" style={{ justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            <div className="row" style={{ gap: 16, flexWrap: "wrap" }}>
              <span style={{ fontSize: 12.5 }}><b className="mono">{counts.auto}</b> auto-fetch</span>
              <span style={{ fontSize: 12.5, color: "var(--changed)" }}><b className="mono">{counts.replace}</b> replaced</span>
              <span style={{ fontSize: 12.5, color: "var(--accent-ink)" }}><b className="mono">{counts.add}</b> supplemented</span>
            </div>
            <div className="row" style={{ gap: 8 }}>
              <span className="hint">Extraction</span>
              <b className="mono" data-testid="src-eta">{mins > 0 ? hm(mins) : anyExtracted ? "nothing to re-extract" : "—"}</b>
              <Btn ghost className="mini" style={{ padding: "3px 9px" }} pending={allAuto.isPending}
                disabled={Boolean(busyJob) || counts.auto === layers.length}
                onClick={() => allAuto.mutate(layers.filter((l) => l.source_setting.mode !== "auto").map((l) => l.key))}>
                Set all to auto-fetch
              </Btn>
            </div>
          </div>
          <div className="hint" style={{ marginTop: 6 }}>
            Estimated from each layer's model minutes per km² × the {int(band)} km² band.
          </div>
        </div>
      </div>

      <Note kind="info" style={{ marginBottom: 14 }}>
        <b>Three choices per layer.</b> <b>Auto-fetch</b> lets the system get it — from a cached dataset or by running a vision model
        over the imagery. <b>Replace</b> uses your file instead and skips the model, which is faster and usually more accurate when
        you hold the authoritative data. <b>Add more</b> runs the auto-fetch and merges your file on top; where the two disagree, yours
        wins.
      </Note>

      <div className="card">
        <div className="ch">
          <h3>Layers</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>
            {layers.length} layers · {layers.reduce((a, l) => a + l.rules.length, 0)} rules depend on them
          </span>
        </div>
        <div data-testid="sources-list">
          {layers.map((l) => {
            const m = modeOf(l);
            const up = uploads[uploadKey(run.id, l.key)];
            const cur = l.source_setting;
            const fileName =
              (up?.status === "done" && up.info?.filename) ||
              (cur.upload_id ? uploadNames[cur.upload_id] ?? l.file_name ?? `upload #${cur.upload_id}` : null);
            const attached = Boolean(cur.upload_id) && cur.mode === m;
            const rowBusy = setMode.isPending && setMode.variables?.key === l.key;
            const saves = m === "replace" && l.minutes_per_km2 ? Math.round(l.minutes_per_km2 * band) : 0;
            return (
              <div key={l.key} className={`srcrow ${m === "replace" ? "replaced" : m === "add" ? "added" : ""}`} data-layer={l.key}>
                <div>
                  <div className="nm">
                    {l.name}
                    {saves > 0 ? <span className="pill p-go" style={{ marginLeft: 6 }}>skips {hm(saves)} of model time</span> : null}
                    {l.stale ? <span className="pill p-cross" style={{ marginLeft: 6 }}>re-extract</span> : null}
                    {l.status === "failed" ? <span className="pill p-nogo" style={{ marginLeft: 6 }}>failed</span> : null}
                  </div>
                  <div className="sub">
                    {l.desc} · {plural(l.rules.length, "rule")}
                    {l.extracted && !l.stale && l.status !== "failed" ? ` · ${int(l.feature_count ?? 0)} features extracted` : ""}
                  </div>
                </div>
                <div className="choice" role="group" aria-label={`Source for ${l.name}`}>
                  {(["auto", "replace", "add"] as SourceMode[]).map((x) => (
                    <button key={x} aria-pressed={m === x} disabled={Boolean(busyJob) || rowBusy}
                      onClick={() => choose(l, x)}>
                      {MODE_LABEL[x]}
                    </button>
                  ))}
                </div>
                {m !== "auto" ? (
                  <UploadSlot
                    mode={m}
                    fileName={attached ? fileName : null}
                    upload={up}
                    busy={rowBusy}
                    onFile={(file) => startUpload(run.id, l.key, m, file)}
                    onMap={() => {
                      if (up?.status === "done" && up.info) setMapFor({ key: l.key, mode: m, upload: up.info });
                    }}
                    canRemap={up?.status === "done" && Boolean(up.info)}
                    onRemove={() => {
                      clearUpload(uploadKey(run.id, l.key));
                      choose(l, "auto");
                    }}
                    onCancelUpload={() => {
                      up?.abort?.();
                      clearUpload(uploadKey(run.id, l.key));
                    }}
                  />
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
      <ErrorNote error={setMode.error ?? allAuto.error} />

      <div className="actions">
        <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/images`)}>Back</button>
        <button className="btn" onClick={() => nav(`/runs/${run.id}/rules`)}>Save sources and continue to assumptions →</button>
      </div>

      {mapFor && cfgByKey[mapFor.key] ? (
        <MappingDialog
          open
          onClose={() => setMapFor(null)}
          runId={run.id}
          layer={cfgByKey[mapFor.key]}
          mode={mapFor.mode}
          upload={mapFor.upload}
          initial={layers.find((l) => l.key === mapFor.key)?.source_setting.upload_id === mapFor.upload.id
            ? layers.find((l) => l.key === mapFor.key)?.source_setting.mapping
            : null}
          onSaved={() => {
            setPending((p) => {
              const n = { ...p };
              delete n[mapFor.key];
              return n;
            });
            setMapFor(null);
          }}
        />
      ) : null}
    </section>
  );
}

function UploadSlot({
  mode,
  fileName,
  upload,
  busy,
  onFile,
  onMap,
  canRemap,
  onRemove,
  onCancelUpload,
}: {
  mode: SourceMode;
  fileName: string | null;
  upload?: ReturnType<typeof useUploads.getState>["entries"][string];
  busy: boolean;
  onFile: (f: File) => void;
  onMap: () => void;
  canRemap: boolean;
  onRemove: () => void;
  onCancelUpload: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const merge = mode === "replace" ? "Used instead of auto-fetch" : "Merged over auto-fetch — your data wins where they disagree";
  const picker = (
    <input ref={input} type="file" hidden accept=".gpkg,.geojson,.json,.zip,.kml,.shp"
      onChange={(e) => {
        const f = e.target.files?.[0];
        if (f) onFile(f);
        e.target.value = "";
      }} />
  );
  let body: React.ReactNode;
  if (upload?.status === "uploading") {
    body = (
      <>
        <span className="spin" aria-hidden />
        <span className="filechip">{upload.fileName}</span>
        <span className="hint">Uploading · {Math.round(upload.frac * 100)}% — you can leave this screen</span>
        <button className="mini" onClick={onCancelUpload}>Cancel</button>
      </>
    );
  } else if (upload?.status === "error") {
    body = (
      <>
        <span className="pill p-nogo">upload failed</span>
        <span style={{ color: "var(--nogo)", fontSize: 12 }}>{upload.error}</span>
        <button className="mini" onClick={() => input.current?.click()}>Choose another file</button>
      </>
    );
  } else if (upload?.status === "done" && !fileName) {
    body = (
      <>
        <span className="filechip">{upload.info?.filename}</span>
        <span className="hint">{int(upload.info?.feature_count ?? 0)} features · not saved yet</span>
        <button className="btn" style={{ padding: "4px 11px", fontSize: 12 }} onClick={onMap}>Map the attributes</button>
        <button className="mini" onClick={onRemove}>Remove</button>
      </>
    );
  } else if (fileName) {
    body = (
      <>
        {busy ? <span className="spin" aria-hidden /> : null}
        <span className="filechip">{fileName}</span>
        {canRemap ? <button className="mini" onClick={onMap}>Attribute mapping</button> : null}
        <button className="mini" onClick={() => input.current?.click()}>Replace file</button>
        <button className="mini" onClick={onRemove}>Remove</button>
      </>
    );
  } else {
    body = (
      <>
        <span style={{ color: "var(--muted)" }}>Drop a GeoPackage, GeoJSON, KML or zipped shapefile here, or</span>
        <button className="mini" onClick={() => input.current?.click()}>choose a file</button>
      </>
    );
  }
  return (
    <div
      className={`slot dropzone${over ? " over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
    >
      {picker}
      {body}
      <span className="hint" style={{ marginLeft: "auto" }}>{merge}</span>
    </div>
  );
}
