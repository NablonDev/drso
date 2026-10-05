import { create } from "zustand";
import type { Job, JobLayerStatus } from "../api/types";

/**
 * The worker writes per-layer progress as `detail || {"layers": {key: ...}}`, a shallow jsonb
 * merge, so each poll only carries the layer being worked on. We accumulate what we have seen
 * per job so the table can show every layer's outcome.
 */
interface JobLayersState {
  byJob: Record<number, Record<string, JobLayerStatus>>;
  record: (job: Job) => void;
}

export const useJobLayers = create<JobLayersState>((set, get) => ({
  byJob: {},
  record: (job) => {
    const l = job.detail?.layers;
    if (!l || Array.isArray(l) || typeof l !== "object") return;
    const prev = get().byJob[job.id] ?? {};
    let changed = false;
    const next = { ...prev };
    for (const [k, v] of Object.entries(l)) {
      const p = prev[k];
      // never downgrade a finished layer back to running
      if (p && p.status !== "running" && v.status === "running") continue;
      if (!p || p.status !== v.status || p.progress !== v.progress || p.features !== v.features) {
        next[k] = v;
        changed = true;
      }
    }
    if (changed) set((s) => ({ byJob: { ...s.byJob, [job.id]: next } }));
  },
}));

/** Layers a job was asked to extract, when the request named them. */
export function requestedLayers(job: Job): string[] | null {
  const l = job.detail?.requested ?? job.detail?.layers;
  return Array.isArray(l) ? (l as string[]) : null;
}
