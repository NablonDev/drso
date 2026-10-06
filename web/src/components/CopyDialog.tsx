import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { apiCalls, qk, useApiMutation } from "../api/hooks";
import type { Run, RunDetail } from "../api/types";
import { Btn, Dialog, ErrorNote } from "./ui";
import { corridorName } from "../lib/format";

/**
 * "Copy and edit — make revision N". If `runs` is given (home screen, no run chosen yet) a picker
 * is shown. `then` decides where the new revision opens.
 */
export default function CopyDialog({
  open,
  onClose,
  run,
  runs,
  then = "sources",
  defaultNote = "",
  extra,
}: {
  open: boolean;
  onClose: () => void;
  run?: Run | null;
  runs?: Run[];
  then?: string;
  defaultNote?: string;
  extra?: React.ReactNode;
}) {
  const [pick, setPick] = useState<string>(run?.id ?? runs?.[0]?.id ?? "");
  const [note, setNote] = useState(defaultNote);
  const nav = useNavigate();
  const qc = useQueryClient();
  useEffect(() => {
    if (open) {
      setPick(run?.id ?? runs?.[0]?.id ?? "");
      setNote(defaultNote);
    }
  }, [open, run, runs, defaultNote]);
  const src = run ?? runs?.find((r) => r.id === pick) ?? null;
  const m = useApiMutation<RunDetail, { id: string; note: string }>({
    mutationFn: ({ id, note }) => apiCalls.copyRun(id, note.trim() || null),
    silent: true,
    onSuccess: (nr) => {
      qc.setQueryData(qk.run(nr.id), nr);
      void qc.invalidateQueries({ queryKey: qk.runs });
      onClose();
      nav(`/runs/${nr.id}/${then}`);
    },
  });
  const nextRev = (src?.rev ?? 0) + 1;
  return (
    <Dialog open={open} onClose={onClose} id="revDlg" labelledBy="revDlgT">
      <h3 id="revDlgT">Copy and edit — make revision {nextRev}</h3>
      <p>A new run that starts from this one. Nothing is re-extracted unless you change the layer it came from.</p>
      {runs && !run ? (
        <label className="fld" style={{ marginBottom: 10 }}>
          <span>Run to copy</span>
          <select value={pick} onChange={(e) => setPick(e.target.value)}>
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {corridorName(r)} · {r.id} · rev {r.rev}
              </option>
            ))}
          </select>
        </label>
      ) : null}
      <div className="tw" style={{ border: "1px solid var(--rule)", borderRadius: 8, marginBottom: 12 }}>
        <table>
          <tbody>
            <tr>
              <td><span className="pill p-go">carries over</span></td>
              <td>All extracted layers, with their features, sources and confidences<div className="hint">Model time you do not pay again</div></td>
            </tr>
            <tr>
              <td><span className="pill p-go">carries over</span></td>
              <td>Corridor, band, imagery, line spec<div className="hint">The corridor is locked once layers exist — copying keeps it fixed</div></td>
            </tr>
            <tr>
              <td><span className="pill p-go">carries over</span></td>
              <td>Every rule value and every override, with its reason and the revision it started in</td>
            </tr>
            <tr>
              <td><span className="pill p-go">carries over</span></td>
              <td>All eight cost categories</td>
            </tr>
            <tr>
              <td><span className="pill p-chg">you change</span></td>
              <td>Whatever you want — a layer source, a buffer, a rate</td>
            </tr>
            <tr>
              <td><span className="pill p-acc">re-runs</span></td>
              <td>The solve. Only layers whose source you change are extracted again.</td>
            </tr>
          </tbody>
        </table>
      </div>
      {extra}
      <label className="fld" style={{ marginBottom: 10 }}>
        <span>What is this revision for? (appears in the comparison)</span>
        <input
          type="text"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Client asked to keep out of reserve forest entirely"
        />
      </label>
      <div className="note n-info" style={{ marginBottom: 12 }}>
        <div>
          Revision {src?.rev ?? "—"} is kept exactly as it is. Both stay on the list and the route map can show the previous one
          next to the new one.
        </div>
      </div>
      <ErrorNote error={m.error} />
      <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
        <button className="btn ghost" onClick={onClose}>
          Cancel
        </button>
        <Btn pending={m.isPending} disabled={!src} onClick={() => src && m.mutate({ id: src.id, note })}>
          Create revision {nextRev}
        </Btn>
      </div>
    </Dialog>
  );
}
