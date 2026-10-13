import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useLayers } from "../api/hooks";
import type { Job, JobLayerStatus, RunLayer } from "../api/types";
import { Btn, Dialog, ErrorNote, Head, Loading, Note, Prog } from "../components/ui";
import { clock, corridorName, elapsedMin, hm, int } from "../lib/format";
import { useTicker } from "../lib/useDebounced";
import { requestedLayers, useJobLayers } from "../store/jobs";
import { useUi } from "../store/ui";

type RowState = { kind: "running" | "done" | "failed" | "kept" | "queued"; progress?: number; features?: number; error?: string };

function rowState(l: RunLayer, job: Job, acc: Record<string, JobLayerStatus>, requested: string[] | null): RowState {
  const a = acc[l.key];
  if (a) {
    if (a.status === "running") return { kind: "running", progress: a.progress ?? 0 };
    if (a.status === "failed") return { kind: "failed", error: a.error };
    return { kind: "done", features: a.features };
  }
  const started = job.started_at ? new Date(job.started_at).getTime() : Infinity;
  const made = l.created_at ? new Date(l.created_at).getTime() : 0;
  if (l.extracted && made >= started - 1000) {
    return l.status === "failed" ? { kind: "failed", error: l.error ?? undefined } : { kind: "done", features: l.feature_count };
  }
  const inScope = requested ? requested.includes(l.key) || (l.key === "clusters" && requested.includes("buildings")) : !l.extracted || l.stale || l.status === "failed";
  if (!inScope && l.extracted) return { kind: "kept", features: l.feature_count };
  return { kind: job.status === "running" || job.status === "queued" ? "queued" : "kept", features: l.feature_count };
}

export default function Extracting() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const push = useUi((s) => s.push);
  const job: Job | null = run.job?.kind === "extract" ? run.job : run.last_extract;
  const live = Boolean(job && (job.status === "running" || job.status === "queued"));
  const lq = useLayers(run.id, { poll: live });
  const record = useJobLayers((s) => s.record);
  const acc = useJobLayers((s) => (job ? s.byJob[job.id] : undefined)) ?? {};
  const [confirm, setConfirm] = useState(false);
  useTicker(live, 1000);

  useEffect(() => {
    if (job) record(job);
  }, [job, record]);

  // follow the job to the layer review when it finishes while this screen is open
  const sawLive = useRef(false);
  useEffect(() => {
    if (live) sawLive.current = true;
    else if (sawLive.current && job?.status === "done") {
      sawLive.current = false;
      void qc.invalidateQueries({ queryKey: qk.layers(run.id) });
      const t = window.setTimeout(() => nav(`/runs/${run.id}/layers`), 1200);
      return () => window.clearTimeout(t);
    }
    return undefined;
  }, [live, job?.status, nav, qc, run.id]);

  const cancel = useApiMutation<Job, number>({
    mutationFn: (id) => apiCalls.cancelJob(id),
    errorTitle: "Could not cancel",
    onSuccess: () => {
      setConfirm(false);
      push({ kind: "info", text: "Cancel requested. The layer in progress stops at its next checkpoint; finished layers are kept." });
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
      void qc.invalidateQueries({ queryKey: qk.active });
    },
  });

  if (!job) {
    return (
      <section>
        <Head title="Step 1 — extract layers">
          {corridorName(run)} · rev {run.rev}. This starts step 1 only — the layers. The route is not solved until you have
          looked at what came back.
        </Head>
        <StartPanel run={run} />
      </section>
    );
  }

  const pct = Math.round((job.progress ?? 0) * 100);
  const el = elapsedMin(job.started_at ?? job.created_at, live ? null : job.finished_at);
  const layers = lq.data?.layers ?? [];
  const requested = requestedLayers(job);
  const rows = layers.map((l) => ({ l, s: rowState(l, job, acc, requested) }));
  const inJob = rows.filter((r) => r.s.kind !== "kept");
  const doneN = inJob.filter((r) => r.s.kind === "done" || r.s.kind === "failed").length;
  const failed = rows.filter((r) => r.s.kind === "failed");
  const est = inJob.reduce((a, r) => a + (r.l.minutes_per_km2 ?? 0) * (lq.data?.band_km2 ?? 0), 0);

  return (
    <section>
      <Head title="Step 1 — extracting layers">
        {corridorName(run)} · rev {run.rev} · started {clock(job.started_at ?? job.created_at)}
        {live && job.eta_text ? ` · about ${job.eta_text} left` : ""}. The route is not being solved yet.
      </Head>

      {live ? (
        <Note kind="info" style={{ marginBottom: 14 }}>
          <b>You can close this tab.</b> The run continues on the server, and this page picks up where it is when you come back.
          Nothing is lost if your laptop sleeps.
        </Note>
      ) : null}

      <div className="card" style={{ marginBottom: 14 }}>
        <div className="cb stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>
                {job.status === "queued"
                  ? "Waiting for the worker"
                  : job.status === "running"
                    ? job.stage ?? "Extracting"
                    : job.status === "done"
                      ? job.stage ?? "Layers ready for review"
                      : job.status === "cancelled"
                        ? "Extraction cancelled"
                        : "Extraction failed"}
              </div>
              <div className="hint">
                {job.cancel_requested && live ? "Cancel requested — stopping at the next checkpoint · " : ""}
                {doneN} of {inJob.length} layers{est ? ` · estimate ${hm(est)}` : ""}
              </div>
            </div>
            <div className="mono" style={{ fontSize: 19, fontWeight: 500 }} data-testid="extract-pct">{pct}%</div>
          </div>
          <Prog value={job.progress} />
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="hint">
              Elapsed {hm(el)}
              {live && job.eta_text ? ` · about ${job.eta_text} left` : ""}
            </span>
            <div className="row" style={{ gap: 8 }}>
              {live ? (
                <Btn ghost pending={cancel.isPending} disabled={job.cancel_requested} onClick={() => setConfirm(true)}>
                  {job.cancel_requested ? "Cancelling…" : "Cancel run"}
                </Btn>
              ) : null}
              <button className="btn" disabled={!layers.some((l) => l.extracted)} onClick={() => nav(`/runs/${run.id}/layers`)}>
                {live ? "Skip to the layer review" : "Review the layers"}
              </button>
            </div>
          </div>
        </div>
      </div>

      {job.status === "failed" ? (
        <Note kind="stop" style={{ marginBottom: 14 }}>
          <b>The extraction stopped with an error.</b> {(job.error ?? "").split("\n")[0]}
        </Note>
      ) : null}
      {job.status === "done" && failed.length ? (
        <Note kind="warn" style={{ marginBottom: 14 }}>
          <b>{failed.length} layer{failed.length > 1 ? "s" : ""} failed.</b> Retry them from the layer review — only those are redone.
        </Note>
      ) : null}
      {job.status === "done" && !failed.length ? (
        <Note kind="ok" style={{ marginBottom: 14 }}>
          <b>Layers ready for review.</b> Finished {clock(job.finished_at)} after {hm(el)}.
        </Note>
      ) : null}
      {!live ? <StartPanel run={run} compact /> : null}

      <div className="phase">
        <div className="phaseh">
          <span className="pill p-acc">Step 1</span>
          <h3>Extracting land use</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>{doneN} of {inJob.length} layers</span>
        </div>
        {lq.isPending ? (
          <div className="cb"><Loading /></div>
        ) : lq.isError ? (
          <div className="cb"><ErrorNote error={lq.error} retry={() => void lq.refetch()} /></div>
        ) : (
          <div className="tw">
            <table data-testid="extract-table">
              <thead>
                <tr>
                  <th>Layer</th>
                  <th>Found</th>
                  <th style={{ textAlign: "right" }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ l, s }) => {
                  const mode = l.source_setting.mode;
                  const tag = mode === "replace" ? <span className="pill p-chg" style={{ marginRight: 6 }}>your file</span>
                    : mode === "add" ? <span className="pill p-acc" style={{ marginRight: 6 }}>supplemented</span> : null;
                  let cell: React.ReactNode;
                  if (s.kind === "running") {
                    const p = Math.round((s.progress ?? 0) * 100);
                    cell = (
                      <div className="row" style={{ justifyContent: "flex-end", gap: 7, flexWrap: "nowrap" }}>
                        <span className="prog" style={{ width: 54, display: "inline-block" }}><i style={{ width: `${p}%` }} /></span>
                        <span className="mono hint" style={{ width: 30, textAlign: "right", margin: 0 }}>{p}%</span>
                      </div>
                    );
                  } else if (s.kind === "done") cell = <span className="pill p-go">Done</span>;
                  else if (s.kind === "failed") cell = <span className="pill p-nogo" title={s.error}>Failed</span>;
                  else if (s.kind === "kept") cell = <span className="pill p-lock">Kept</span>;
                  else cell = <span className="pill p-lock">Queued</span>;
                  const found =
                    s.kind === "failed" ? (s.error ?? "failed").slice(0, 90)
                      : s.kind === "running" ? "running…"
                        : s.kind === "queued" ? l.desc ?? ""
                          : `${int(s.features ?? l.feature_count ?? 0)} features${s.kind === "kept" ? " · unchanged" : ""}`;
                  return (
                    <tr key={l.key} style={s.kind === "queued" || s.kind === "kept" ? { opacity: 0.6 } : undefined}>
                      <td>{tag}<b>{l.name}</b></td>
                      <td className="mono" style={{ fontSize: 12 }}>{found}</td>
                      <td style={{ textAlign: "right" }}>{cell}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <Note kind="info" style={{ marginTop: 14 }}>
        <b>Nothing is routed when this finishes.</b> The next screen shows you what each model found so you can spot the gaps — a thin
        forest layer, a line the detector missed — and correct them before a single alignment is drawn. You release step 2 yourself.
      </Note>
      <Note kind="warn" style={{ marginTop: 12 }}>
        <b>A failure keeps what it has.</b> If a model fails, the layers already extracted are kept and a retry redoes only the one that
        failed.
      </Note>

      <Dialog open={confirm} onClose={() => setConfirm(false)} labelledBy="cxT">
        <h3 id="cxT">Cancel the extraction?</h3>
        <p>Layers already finished are kept. The layer in progress stops at its next checkpoint and can be retried later.</p>
        <ErrorNote error={cancel.error} />
        <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
          <button className="btn ghost" onClick={() => setConfirm(false)}>Keep running</button>
          <Btn pending={cancel.isPending} onClick={() => cancel.mutate(job.id)}>Cancel the extraction</Btn>
        </div>
      </Dialog>
    </section>
  );
}


/** Start (or re-start) the extraction — the "Review & extract" screen folded into this one (Vivek, 13 Oct 2026). */
function StartPanel({ run, compact }: { run: ReturnType<typeof useRunCtx>; compact?: boolean }) {
  const nav = useNavigate();
  const qc = useQueryClient();
  const lq = useLayers(run.id);
  const [forceModels, setForceModels] = useState(false);
  const extract = useApiMutation<Job, void>({
    mutationFn: () => apiCalls.extract(run.id, undefined, forceModels),
    errorTitle: "Extraction not started",
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.active });
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
      void qc.invalidateQueries({ queryKey: qk.runs });
    },
  });
  if (lq.isPending) return <Loading />;
  if (lq.isError) return <ErrorNote error={lq.error} retry={() => void lq.refetch()} />;
  const L = lq.data;
  const anyExtracted = L.layers.some((l) => l.extracted);
  const todo = L.layers.filter((l) => !l.extracted || l.stale || l.status === "failed");
  const minutes = todo.reduce((a, l) => a + (l.source_setting.mode === "replace" ? 0 : (l.minutes_per_km2 ?? 0) * L.band_km2), 0);
  const finish = new Date(Date.now() + minutes * 60000).toISOString();
  const nothingToDo = anyExtracted && todo.length === 0;
  const img = run.imagery;
  if (compact && nothingToDo) return null;
  return (
    <div className="card" style={{ marginBottom: 14 }}>
      <div className="ch"><h3>{compact ? "Extract again" : "Extract the layers"}</h3></div>
      <div className="cb">
        <div className="tw">
          <table>
            <tbody>
              <tr>
                <td style={{ width: 30 }}><span className="pill p-acc">1</span></td>
                <td>
                  <b>Extract {todo.length} layer{todo.length === 1 ? "" : "s"}</b> over {int(L.band_km2)} km² —{" "}
                  {todo.filter((l) => l.source_setting.mode === "auto").length} auto-fetched,{" "}
                  {todo.filter((l) => l.source_setting.mode === "replace").length} from your files,{" "}
                  {todo.filter((l) => l.source_setting.mode === "add").length} supplemented
                  {anyExtracted ? <div className="hint">{L.layers.length - todo.length} layers are already extracted and are kept.</div> : null}
                  <div className="hint">Unattended. You can close the tab.</div>
                </td>
                <td className="mono" style={{ textAlign: "right" }}>{todo.length ? hm(minutes) : "—"}</td>
              </tr>
              <tr style={{ background: "var(--surface-2)" }}>
                <td><span className="pill p-cross">⏸</span></td>
                <td>
                  <b>You review the layers</b> — check what the models found, add what is missing, then release step 2
                  <div className="hint">The route does not solve on its own.</div>
                </td>
                <td className="mono" style={{ textAlign: "right" }}>—</td>
              </tr>
            </tbody>
          </table>
        </div>
        {img && !img.error && !img.covered && !run.accept_partial ? (
          <Note kind="warn" style={{ marginTop: 12 }}>
            The imagery check found a gap and partial coverage has not been accepted on the <b>Images</b> screen. Extraction can
            still run; the high-resolution layers will be blank over the gap.
          </Note>
        ) : null}
        {nothingToDo ? (
          <Note kind="ok" style={{ marginTop: 12 }}>
            Every layer is already extracted and current. Go straight to the layer review, or change a source to re-extract it.
          </Note>
        ) : null}
        <ErrorNote error={extract.error} />
        <div className="row" style={{ justifyContent: "space-between", marginTop: 14, gap: 12 }}>
          <div className="hint" style={{ maxWidth: "46ch", margin: 0 }}>
            One GPU job runs at a time. If a solve or another extraction is running, this is refused and nothing is queued.
          </div>
          {nothingToDo ? (
            <button className="btn" onClick={() => nav(`/runs/${run.id}/layers`)}>Review the layers →</button>
          ) : (
            <>
              <label className="hint" style={{ display: "flex", gap: 6, alignItems: "center" }}>
                <input type="checkbox" checked={forceModels} onChange={(e) => setForceModels(e.target.checked)}
                  data-testid="force-models" />
                Re-run the LULC models even where a stored output covers the band
              </label>
              <Btn pending={extract.isPending} disabled={Boolean(run.job)} onClick={() => extract.mutate()} data-testid="start-extract">
                Fetch LULCs · layers ready about {clock(finish)}
              </Btn>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
