import { useEffect, useRef } from "react";
import { Link, NavLink, Outlet, useLocation, useMatch, useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { qk, useActiveJob, useCosts, useLayers, useRun } from "../api/hooks";
import type { Job, RunDetail } from "../api/types";
import { corridorName, lineSpec } from "../lib/format";
import { toggleTheme, useUi } from "../store/ui";
import { Prog } from "./ui";
import ErrorBoundary from "./ErrorBoundary";

export type StepKey =
  | "corridor"
  | "images"
  | "sources"
  | "rules"
  | "costs"
  | "extracting"
  | "layers"
  | "solve"
  | "map"
  | "why"
  | "export";

const GROUPS: { title: string; steps: [StepKey, string, string][] }[] = [
  { title: "Set up", steps: [["corridor", "1", "Corridor"], ["images", "2", "Images"], ["sources", "3", "Data sources"]] },
  { title: "Assumptions", steps: [["rules", "4", "Rules"], ["costs", "5", "Costs"]] },
  // "Review & extract" removed (Vivek, 13 Oct 2026): extraction starts from the Extract layers screen
  { title: "Step 1 — layers", steps: [["extracting", "6", "Extract layers"], ["layers", "7", "Review the layers"]] },
  { title: "Step 2 — route", steps: [["solve", "8", "Solve the route"]] },
  { title: "Results", steps: [["map", "9", "Route map"], ["why", "10", "Why this route"], ["export", "11", "Export"]] },
];

export function hasLayers(run: RunDetail): boolean {
  return Boolean(
    run.layers_ready ||
      run.last_extract ||
      run.reuse_from ||
      run.layers_stale?.length ||
      ["review", "solving", "complete", "blocked"].includes(run.status),
  );
}

export function stepEnabled(k: StepKey, run: RunDetail | undefined): boolean {
  if (!run) return k === "corridor";
  switch (k) {
    case "extracting":
      return run.status !== "draft" || run.job?.kind === "extract" || Boolean(run.last_extract);
    case "layers":
      return hasLayers(run);
    case "solve":
      return run.layers_ready || run.job?.kind === "solve" || Boolean(run.latest_solve_id);
    case "map":
    case "why":
    case "export":
      return Boolean(run.latest_solve_id);
    default:
      return true;
  }
}

function stepDone(k: StepKey, run: RunDetail | undefined): boolean {
  if (!run) return false;
  const im = run.imagery;
  switch (k) {
    case "corridor":
      return true;
    case "images":
      return Boolean(im && !im.error && (im.covered || run.accept_partial));
    case "sources":
    case "rules":
    case "costs":
      return run.status !== "draft";
    case "extracting":
      return run.layers_ready && run.job?.kind !== "extract";
    case "layers":
    case "solve":
      return Boolean(run.latest_solve_id) && run.job?.kind !== "solve";
    default:
      return false;
  }
}

function Rail({ run, runId }: { run: RunDetail | undefined; runId: string | null }) {
  const nav = useNavigate();
  const loc = useLocation();
  const isNew = loc.pathname === "/new";
  const costs = useCosts(run ? runId ?? undefined : undefined).data;
  // only read cached layers — do not trigger a fetch from the rail
  const layers = useLayers(runId ?? undefined, { enabled: false }).data;
  const nCustom = run ? Object.values(run.sources ?? {}).filter((s) => s.mode !== "auto").length : 0;
  const nOvr = run ? Object.keys(run.rule_overrides ?? {}).length : 0;
  const nFlags = layers?.flags.filter((f) => f.kind !== "empty").length ?? 0;
  const cur = loc.pathname.split("/").pop();

  return (
    <nav className="rail" aria-label="Steps">
      <NavLink to="/" end className="step" aria-current={loc.pathname === "/" ? "true" : "false"} style={{ textDecoration: "none" }}>
        <span className="n">⌂</span> My routes
      </NavLink>
      {GROUPS.map((g) => (
        <div key={g.title} style={{ display: "contents" }}>
          <div className="railgroup eyebrow">{g.title}</div>
          {g.steps.map(([k, n, label]) => {
            const enabled = isNew ? k === "corridor" : Boolean(runId) && stepEnabled(k, run);
            const current = isNew ? k === "corridor" : cur === k;
            let flag: React.ReactNode = null;
            if (k === "sources" && nCustom) flag = <span className="flag chg">{nCustom}</span>;
            if (k === "rules" && nOvr) flag = <span className="flag chg">{nOvr}</span>;
            if (k === "costs" && costs && costs.n_tcc > 0) flag = <span className="flag warn">TCC</span>;
            if (k === "layers" && nFlags) flag = <span className="flag warn">{nFlags}</span>;
            if (k === "extracting" && run?.job?.kind === "extract") flag = <span className="flag warn">{Math.round((run.job.progress ?? 0) * 100)}%</span>;
            if (k === "solve" && run?.job?.kind === "solve") flag = <span className="flag warn">{Math.round((run.job.progress ?? 0) * 100)}%</span>;
            return (
              <button
                key={k}
                className={`step${stepDone(k, run) && !current ? " done" : ""}`}
                aria-current={current ? "true" : "false"}
                disabled={!enabled}
                title={enabled ? undefined : disabledWhy(k)}
                onClick={() => runId && nav(`/runs/${runId}/${k}`)}
              >
                <span className="n">{n}</span> {label}
                {flag}
              </button>
            );
          })}
        </div>
      ))}
    </nav>
  );
}

function disabledWhy(k: StepKey): string {
  switch (k) {
    case "extracting":
      return "Available once the corridor is saved";
    case "layers":
      return "Layers are reviewed once they are extracted";
    case "solve":
      return "Solve opens once every layer is extracted and current";
    case "map":
    case "why":
    case "export":
      return "Available once the route is solved";
    default:
      return "Create the run on the Corridor screen first";
  }
}

export function jobPath(j: Job) {
  return `/runs/${j.run_id}/${j.kind === "extract" ? "extracting" : "solve"}`;
}

function BusyChip() {
  const job = useActiveJob().data?.job;
  if (!job) return <span className="idlechip">GPU idle</span>;
  const kind = job.kind === "extract" ? "Extracting" : "Solving";
  const pct = Math.round((job.progress ?? 0) * 100);
  const tip = `${kind} ${job.run_id} · ${job.stage ?? job.status} · ${pct}%${job.eta_text ? ` · about ${job.eta_text} left` : ""}`;
  return (
    <Link className="busychip" to={jobPath(job)} title={tip} aria-label={`GPU busy: ${tip}`} data-testid="busy-chip">
      <span className="spin" aria-hidden />
      <span>
        {kind} <b>{job.run_id}</b> · {job.stage ?? job.status}
      </span>
      <b>{pct}%</b>
      <Prog value={job.progress} />
      {job.eta_text ? <span className="mono">{job.eta_text} left</span> : null}
    </Link>
  );
}

/** When the active job ends, refresh what it changed and say how it ended. */
function useJobWatcher() {
  const job = useActiveJob().data?.job ?? null;
  const prev = useRef<Job | null>(null);
  const qc = useQueryClient();
  const push = useUi((s) => s.push);
  useEffect(() => {
    const p = prev.current;
    prev.current = job;
    if (p && (!job || job.id !== p.id)) {
      void qc.invalidateQueries({ queryKey: qk.run(p.run_id) });
      void qc.invalidateQueries({ queryKey: qk.runs });
      void qc.invalidateQueries({ queryKey: qk.layers(p.run_id) });
      api<Job>(`/api/jobs/${p.id}`)
        .then((j) => {
          const what = j.kind === "extract" ? "Layer extraction" : "Route solve";
          if (j.status === "done") {
            const failed = (j.detail?.failed as string[] | undefined) ?? [];
            push({
              kind: failed.length ? "warn" : "ok",
              text: `${what} for ${j.run_id} finished${j.stage ? ` — ${j.stage}` : ""}.`,
            });
          } else if (j.status === "failed") {
            push({ kind: "stop", text: `${what} for ${j.run_id} failed: ${(j.error ?? "unknown error").split("\n")[0]}` });
          } else if (j.status === "cancelled") {
            push({ kind: "info", text: `${what} for ${j.run_id} was cancelled.` });
          }
        })
        .catch(() => {
          /* the run refresh above is what matters; a missing toast is harmless */
        });
    }
  }, [job, qc, push]);
}

export function Toasts() {
  const toasts = useUi((s) => s.toasts);
  const dismiss = useUi((s) => s.dismiss);
  return (
    <div className="toasts" aria-live="assertive" role="region" aria-label="Notifications">
      {toasts.map((t) => (
        <div key={t.id} className={`note toast n-${t.kind}`} role={t.kind === "stop" ? "alert" : "status"} data-testid="toast">
          <div>
            {t.title ? <b>{t.title}. </b> : null}
            {t.text}
            {t.job ? (
              <div style={{ marginTop: 4 }}>
                <Link to={jobPath(t.job)} className="link" onClick={() => dismiss(t.id)}>
                  See the running job
                </Link>
              </div>
            ) : null}
          </div>
          <button className="x" aria-label="Dismiss" onClick={() => dismiss(t.id)}>
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

export default function Shell() {
  const m = useMatch("/runs/:runId/*");
  const runId = m?.params.runId ?? null;
  const run = useRun(runId ?? undefined).data;
  const loc = useLocation();
  useJobWatcher();

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [loc.pathname]);

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand" style={{ textDecoration: "none" }}>
          <span className="brand-logo">
            <img
              src="/brand/adani-logo.png"
              alt="Adani"
              onError={(e) => {
                (e.currentTarget.parentElement as HTMLElement).hidden = true;
              }}
            />
          </span>
          <span className="brand-name">
            <b>Beeline</b>
            <span>transmission route planner</span>
          </span>
        </Link>
        {run ? (
          <div className="runchip" data-testid="runchip">
            <b data-testid="corridor-name">{corridorName(run)}</b> · <span className="mono">{run.id}</span> ·{" "}
            <b className="mono">rev {run.rev}</b> · <span className="mono">{lineSpec(run.kv, run.circuit)}</span>
          </div>
        ) : null}
        <BusyChip />
        <div className="spacer" />
        <button className="tbtn" onClick={toggleTheme}>
          Theme
        </button>
      </header>
      <Rail run={run} runId={runId} />
      <main className="main" id="main">
        <ErrorBoundary key={loc.pathname}>
          <Outlet />
        </ErrorBoundary>
      </main>
      <Toasts />
    </div>
  );
}
