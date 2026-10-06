import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useActiveJob, useRuns } from "../api/hooks";
import type { Job, RunListItem } from "../api/types";
import CopyDialog from "../components/CopyDialog";
import { jobPath } from "../components/Shell";
import { ErrorNote, Head, Loading, Prog, StatusPill } from "../components/ui";
import { clock, corridorName, crore, km, lineSpec, when } from "../lib/format";

function RunningCard({ job, run }: { job: Job; run?: RunListItem }) {
  const pct = Math.round((job.progress ?? 0) * 100);
  const step = job.kind === "extract" ? "Step 1 of 2 — extracting layers" : "Step 2 of 2 — solving the route";
  const tail =
    job.kind === "extract" ? "The route will not solve until you review them." : "The layers are fixed; this step can be re-run.";
  return (
    <div className="card" style={{ borderColor: "var(--accent-line)", marginBottom: 14 }} data-testid="running-card">
      <div className="cb">
        <div className="row" style={{ justifyContent: "space-between", gap: 12 }}>
          <div style={{ flex: 1, minWidth: 210 }}>
            <div className="row" style={{ gap: 8 }}>
              <span className="pill p-acc">Running</span>
              <b style={{ fontSize: 13 }}>
                {job.run_id}
                {run ? ` · ${corridorName(run)}` : ""}
              </b>
            </div>
            <div className="hint" style={{ marginTop: 4 }}>
              {step} · {pct}% · {job.stage ?? job.status} · started {clock(job.started_at ?? job.created_at)}
              {job.eta_text ? ` · about ${job.eta_text} left` : ""}. {tail}
            </div>
            <div style={{ marginTop: 8 }}>
              <Prog value={job.progress} />
            </div>
          </div>
          <Link className="btn ghost" to={jobPath(job)}>
            Status
          </Link>
        </div>
      </div>
    </div>
  );
}

function statusExtra(r: RunListItem): string | null {
  if (r.job) {
    const pct = Math.round((r.job.progress ?? 0) * 100);
    return r.job.eta_text ? `${r.job.eta_text} left` : `${pct}%`;
  }
  return null;
}

function action(r: RunListItem, copy: (r: RunListItem) => void, nav: (p: string) => void) {
  if (r.job) return <button className="mini" onClick={() => nav(jobPath(r.job!))}>Status</button>;
  switch (r.status) {
    case "complete":
    case "blocked":
      return (
        <button className="mini" onClick={() => copy(r)}>
          Copy &amp; edit
        </button>
      );
    case "review":
      return (
        <button className="mini" onClick={() => nav(`/runs/${r.id}/layers`)}>
          Review layers
        </button>
      );
    case "draft":
      return (
        <button className="mini" onClick={() => nav(`/runs/${r.id}/corridor`)}>
          Continue
        </button>
      );
    default:
      return (
        <button className="mini" onClick={() => nav(`/runs/${r.id}`)}>
          Open
        </button>
      );
  }
}

export default function Home() {
  const q = useRuns();
  const active = useActiveJob().data?.job ?? q.data?.active_job ?? null;
  const nav = useNavigate();
  const [copyOf, setCopyOf] = useState<RunListItem | null>(null);
  const [copyAny, setCopyAny] = useState(false);
  const runs = q.data?.runs ?? [];
  const copyable = runs.filter((r) => !r.job);

  return (
    <section>
      <Head title="My routes">
        Every run in one list. Layers are extracted first and reviewed; the route is solved only when you say so. No sign-in —
        everyone who opens the tool sees the same list.
      </Head>

      {active ? <RunningCard job={active} run={runs.find((r) => r.id === active.run_id)} /> : null}

      <div className="row" style={{ marginBottom: 14 }}>
        <button className="btn" onClick={() => nav("/new")}>
          New route
        </button>
        <button className="btn ghost" onClick={() => setCopyAny(true)} disabled={!copyable.length}>
          Copy and edit a run
        </button>
        <div className="spacer" />
        <span className="hint">
          {active
            ? `One GPU job at a time — ${active.kind === "extract" ? "an extraction" : "a solve"} is running${active.eta_text ? `, about ${active.eta_text} left` : ""}. Nothing is queued.`
            : "GPU free — nothing is running or queued"}
        </span>
      </div>

      <div className="card">
        {q.isPending ? (
          <div className="cb">
            <Loading text="Loading runs…" />
          </div>
        ) : q.isError ? (
          <div className="cb">
            <ErrorNote error={q.error} retry={() => void q.refetch()} />
          </div>
        ) : runs.length === 0 ? (
          <div className="cb">
            <p style={{ color: "var(--muted)", fontSize: 13 }}>No runs yet. Start with <b>New route</b>.</p>
          </div>
        ) : (
          <div className="tw">
            <table data-testid="runs-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Rev</th>
                  <th>Corridor</th>
                  <th>Voltage</th>
                  <th>Sources</th>
                  <th>Length</th>
                  <th>Cost</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.id} data-testid="run-row">
                    <td>
                      <Link to={`/runs/${r.id}`} style={{ textDecoration: "none" }}>
                        <b>{corridorName(r)}</b>
                      </Link>
                      <div className="hint">
                        <span className="mono">{r.id}</span> · {when(r.created_at)}
                        {r.parent_id ? ` · copied from ${r.parent_id}` : ""}
                        {r.note ? ` · ${r.note}` : ""}
                      </div>
                      {r.last_error ? <div className="err">Last job failed: {r.last_error}</div> : null}
                    </td>
                    <td>
                      <span className={`pill ${r.rev > 1 ? "p-chg" : "p-go"}`}>r{r.rev}</span>
                    </td>
                    <td className="hint">
                      {r.a_name || r.b_name ? `${r.a_name || "A"} → ${r.b_name || "B"}` : "A → B"}
                    </td>
                    <td className="mono">{lineSpec(r.kv, r.circuit)}</td>
                    <td>
                      {r.custom_sources ? (
                        <span className="pill p-chg">{r.custom_sources} custom</span>
                      ) : (
                        <span className="pill p-go">All auto</span>
                      )}
                    </td>
                    <td className="mono">{r.r1_length_km && r.feasible !== false ? km(Number(r.r1_length_km)) : "—"}</td>
                    <td className="mono">{r.r1_cost_rs && r.feasible !== false ? crore(r.r1_cost_rs) : "—"}</td>
                    <td>
                      <StatusPill status={r.job ? (r.job.kind === "extract" ? "extracting" : "solving") : r.status} extra={statusExtra(r)} />
                    </td>
                    <td>{action(r, setCopyOf, nav)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      <div className="note n-info" style={{ marginTop: 14 }}>
        <div>
          <b>Routing is an iteration, not a one-shot.</b> A corridor is rarely right the first time — the forest layer comes back
          thin, or an existing line the team knows about was never detected. <b>Copy and edit</b> makes the next revision: the
          extracted layers carry over, only your corrections are re-applied, and the route re-solves from there. Every revision
          keeps its own record of which constraints were overridden.
        </div>
      </div>

      <CopyDialog open={Boolean(copyOf)} onClose={() => setCopyOf(null)} run={copyOf} />
      <CopyDialog open={copyAny} onClose={() => setCopyAny(false)} runs={copyable} />
    </section>
  );
}
