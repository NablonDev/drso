import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { useRunCtx } from "../App";
import { apiCalls, qk, useApiMutation, useLayers, useRules } from "../api/hooks";
import type { Job } from "../api/types";
import { Btn, Dialog, ErrorNote, Head, Note, Prog } from "../components/ui";
import { clock, elapsedMin, hm, int, seconds } from "../lib/format";
import { useTicker } from "../lib/useDebounced";
import { useUi } from "../store/ui";

const STAGES: [string, RegExp][] = [
  ["Compile the rules into the cost surface", /waiting|starting|compil/i],
  ["Fetch terrain for clearance checks", /terrain|dem/i],
  ["Solve the routes", /solving route/i],
  ["Spot towers and cost them", /spotting/i],
  ["Write the exports and the decision log", /writing|export|done|ready|no viable/i],
];

function stageIndex(stage: string | null, status: Job["status"]): number {
  if (status === "done") return STAGES.length;
  const s = stage ?? "";
  for (let i = STAGES.length - 1; i >= 0; i--) if (STAGES[i][1].test(s)) return i;
  return 0;
}

export function useStartSolve(runId: string) {
  const qc = useQueryClient();
  const nav = useNavigate();
  return useApiMutation<Job, void>({
    mutationFn: () => apiCalls.solve(runId),
    errorTitle: "Solve not started",
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: qk.active });
      void qc.invalidateQueries({ queryKey: qk.run(runId) });
      void qc.invalidateQueries({ queryKey: qk.runs });
      nav(`/runs/${runId}/solve`);
    },
  });
}

export default function Solve() {
  const run = useRunCtx();
  const nav = useNavigate();
  const qc = useQueryClient();
  const push = useUi((s) => s.push);
  const rules = useRules(run.id).data;
  const layers = useLayers(run.id).data;
  const start = useStartSolve(run.id);
  const job: Job | null = run.job?.kind === "solve" ? run.job : run.last_job?.kind === "solve" ? run.last_job : null;
  const live = Boolean(job && (job.status === "running" || job.status === "queued"));
  const [confirm, setConfirm] = useState(false);
  useTicker(live, 1000);

  const saw = useRef(false);
  useEffect(() => {
    if (live) saw.current = true;
    else if (saw.current && job?.status === "done") {
      saw.current = false;
      const t = window.setTimeout(() => nav(`/runs/${run.id}/map`), 900);
      return () => window.clearTimeout(t);
    }
    return undefined;
  }, [live, job?.status, nav, run.id]);

  const cancel = useApiMutation<Job, number>({
    mutationFn: (id) => apiCalls.cancelJob(id),
    errorTitle: "Could not cancel",
    onSuccess: () => {
      setConfirm(false);
      push({ kind: "info", text: "Cancel requested — the solve stops at its next checkpoint." });
      void qc.invalidateQueries({ queryKey: qk.run(run.id) });
    },
  });

  const nRules = rules ? rules.n_rules - rules.n_off : null;
  const features = layers?.features_total;
  const lastS = job?.started_at && job.finished_at ? (new Date(job.finished_at).getTime() - new Date(job.started_at).getTime()) / 1000 : null;
  const cur = job ? stageIndex(job.stage, job.status) : -1;

  return (
    <section>
      <Head title="Step 2 — solving the route">
        Layers are fixed. This step applies the rules and the cost surface to them, and it is the step you will run again after every
        change.
      </Head>

      {live && job ? (
        <div className="card" style={{ marginBottom: 14 }} data-testid="solve-progress">
          <div className="cb stack">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500 }}>
                  Solving {run.routes_wanted} alignment{run.routes_wanted > 1 ? "s" : ""} · {job.stage ?? "starting"}
                </div>
                <div className="hint">
                  {nRules ?? "All"} rules over {features !== undefined ? int(features) : "the extracted"} features · no imagery is read in this step
                </div>
              </div>
              <div className="mono" style={{ fontSize: 19, fontWeight: 500 }}>{Math.round((job.progress ?? 0) * 100)}%</div>
            </div>
            <Prog value={job.progress} />
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span className="hint">
                Started {clock(job.started_at ?? job.created_at)} · elapsed {hm(elapsedMin(job.started_at ?? job.created_at))}
                {job.eta_text ? ` · about ${job.eta_text} left` : ""} · you can close this tab
              </span>
              <Btn ghost pending={cancel.isPending} disabled={job.cancel_requested} onClick={() => setConfirm(true)}>
                {job.cancel_requested ? "Cancelling…" : "Cancel"}
              </Btn>
            </div>
          </div>
        </div>
      ) : (
        <div className="card" style={{ marginBottom: 14, borderColor: "var(--accent-line)" }}>
          <div className="cb stack">
            {job?.status === "failed" ? (
              <Note kind="stop"><b>The last solve failed.</b> {(job.error ?? "").split("\n")[0]}</Note>
            ) : null}
            {job?.status === "cancelled" ? <Note kind="info">The last solve was cancelled.</Note> : null}
            {job?.status === "done" ? (
              <Note kind="ok">
                <b>{job.stage === "No viable route" ? "The last solve found no viable route." : "Routes ready."}</b> Finished{" "}
                {clock(job.finished_at)}{lastS ? ` in ${seconds(lastS)}` : ""}.
              </Note>
            ) : null}
            {!run.layers_ready ? (
              <Note kind="warn">
                {run.layers_stale.length
                  ? `Some layers changed source and must be extracted again first: ${run.layers_stale.join(", ")}.`
                  : "Every layer must be extracted before the route can be solved."}{" "}
                <button className="link" style={{ padding: 0 }} onClick={() => nav(`/runs/${run.id}/review`)}>Go to Review &amp; extract</button>
              </Note>
            ) : null}
            <div className="row" style={{ justifyContent: "space-between", gap: 12 }}>
              <div style={{ maxWidth: "60ch" }}>
                <div style={{ fontSize: 13, fontWeight: 500 }}>
                  {run.routes_wanted} alignment{run.routes_wanted > 1 ? "s" : ""} · {nRules ?? "all"} rules ·{" "}
                  {features !== undefined ? `${int(features)} features` : "extracted layers"}
                </div>
                <div className="hint">
                  Re-uses the extracted layers. Run it as often as you like — changing a rule and re-solving does not re-extract anything.
                </div>
              </div>
              <div className="row" style={{ gap: 8 }}>
                {run.latest_solve_id ? <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/map`)}>See the results</button> : null}
                <Btn pending={start.isPending} disabled={!run.layers_ready || Boolean(run.job)} onClick={() => start.mutate()} data-testid="start-solve">
                  {run.latest_solve_id ? "Solve again" : "Solve the route"}
                  {lastS ? ` · ${seconds(lastS)}` : ""}
                </Btn>
              </div>
            </div>
            <ErrorNote error={start.error} />
          </div>
        </div>
      )}

      <div className="phase">
        <div className="phaseh">
          <span className="pill p-acc">Step 2</span>
          <h3>What the solve does</h3>
          <span className="hint" style={{ marginLeft: "auto" }}>{lastS ? `last run ${seconds(lastS)}` : ""}</span>
        </div>
        <div className="cb" style={{ padding: 15 }}>
          {STAGES.map(([label], i) => {
            const done = live ? i < cur : job?.status === "done";
            const now = live && i === cur;
            return (
              <div key={label} className={`stage${done ? " done" : ""}${now ? " now" : ""}`}>
                <span className="dot" />
                <span>{label}{now && job?.stage ? <span className="hint" style={{ marginLeft: 8 }}>{job.stage}</span> : null}</span>
                <span className="t">{done ? "done" : now ? "running" : "—"}</span>
              </div>
            );
          })}
        </div>
      </div>

      <Note kind="ok" style={{ marginTop: 14 }}>
        <b>This is the cheap half.</b> A solve takes minutes against the extraction's much longer GPU time. Splitting the two means a
        drafter can relax a buffer, re-solve, compare, and relax it again inside a working morning — without touching the imagery
        pipeline once.
      </Note>

      <Dialog open={confirm} onClose={() => setConfirm(false)} labelledBy="csT">
        <h3 id="csT">Cancel the solve?</h3>
        <p>The previous result, if any, is kept. You can solve again at any time.</p>
        <ErrorNote error={cancel.error} />
        <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
          <button className="btn ghost" onClick={() => setConfirm(false)}>Keep running</button>
          <Btn pending={cancel.isPending} onClick={() => job && cancel.mutate(job.id)}>Cancel the solve</Btn>
        </div>
      </Dialog>
    </section>
  );
}
