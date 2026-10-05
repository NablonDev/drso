import { create } from "zustand";
import { errorText, uploadFile } from "../api/client";
import type { SourceMode, UploadInfo } from "../api/types";
import { useUi } from "./ui";

/**
 * Uploads live outside the screen so they keep going (and are still visible) when the user
 * leaves the Data sources screen and comes back.
 */
export interface UploadEntry {
  key: string; // `${runId}:${layerKey}`
  mode: SourceMode;
  fileName: string;
  status: "uploading" | "done" | "error";
  frac: number;
  info?: UploadInfo;
  error?: string;
  abort?: () => void;
}

interface UploadState {
  entries: Record<string, UploadEntry>;
  start: (runId: string, layerKey: string, mode: SourceMode, file: File) => void;
  clear: (key: string) => void;
}

export const uploadKey = (runId: string, layerKey: string) => `${runId}:${layerKey}`;

export const useUploads = create<UploadState>((set, get) => ({
  entries: {},
  start: (runId, layerKey, mode, file) => {
    const key = uploadKey(runId, layerKey);
    get().entries[key]?.abort?.();
    const ctl = new AbortController();
    set((s) => ({
      entries: {
        ...s.entries,
        [key]: { key, mode, fileName: file.name, status: "uploading", frac: 0, abort: () => ctl.abort() },
      },
    }));
    const patch = (p: Partial<UploadEntry>) =>
      set((s) => (s.entries[key] ? { entries: { ...s.entries, [key]: { ...s.entries[key], ...p } } } : s));
    uploadFile<UploadInfo>("/api/uploads", file, (frac) => patch({ frac }), ctl.signal)
      .then((info) => {
        useUi.getState().rememberUpload(info.id, info.filename);
        patch({ status: "done", frac: 1, info, abort: undefined });
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === "AbortError") {
          get().clear(key);
          return;
        }
        patch({ status: "error", error: errorText(e), abort: undefined });
      });
  },
  clear: (key) =>
    set((s) => {
      const next = { ...s.entries };
      delete next[key];
      return { entries: next };
    }),
}));
