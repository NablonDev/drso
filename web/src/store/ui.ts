import { create } from "zustand";
import type { Job } from "../api/types";

export type ToastKind = "stop" | "warn" | "ok" | "info";

export interface Toast {
  id: number;
  kind: ToastKind;
  title?: string;
  text: string;
  job?: Job | null;
  sticky?: boolean;
}

interface UiState {
  toasts: Toast[];
  push: (t: Omit<Toast, "id">) => number;
  dismiss: (id: number) => void;
  /** chosen route per run, shared by map / why / export */
  selectedRoute: Record<string, number>;
  selectRoute: (runId: string, id: number) => void;
  /** a place the route map should fly to on next open (from "Why this route") */
  focus: { runId: string; lon: number; lat: number; zoom: number } | null;
  setFocus: (f: UiState["focus"]) => void;
  /** upload id → file name, so the sources screen can name files the server does not echo back */
  uploadNames: Record<number, string>;
  rememberUpload: (id: number, name: string) => void;
}

let seq = 1;

export const useUi = create<UiState>((set, get) => ({
  toasts: [],
  push: (t) => {
    const id = seq++;
    // one toast per identical message
    const dup = get().toasts.find((x) => x.text === t.text);
    if (dup) return dup.id;
    set((s) => ({ toasts: [...s.toasts.slice(-3), { ...t, id }] }));
    if (!t.sticky) {
      window.setTimeout(() => get().dismiss(id), t.kind === "stop" ? 12000 : 6000);
    }
    return id;
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
  selectedRoute: {},
  selectRoute: (runId, id) => set((s) => ({ selectedRoute: { ...s.selectedRoute, [runId]: id } })),
  focus: null,
  setFocus: (focus) => set({ focus }),
  uploadNames: {},
  rememberUpload: (id, name) => set((s) => ({ uploadNames: { ...s.uploadNames, [id]: name } })),
}));

export type Theme = "light" | "dark" | null;

export function currentTheme(): "light" | "dark" {
  const t = document.documentElement.getAttribute("data-theme");
  if (t === "dark" || t === "light") return t;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try {
    localStorage.setItem("beeline-theme", next);
  } catch {
    /* storage blocked — theme still applies for this visit */
  }
}
