import { create } from "zustand";
import type { Route, Tower } from "../api/types";

/**
 * Client-side tower store. Each spotted tower gets a stable id `${routeId}-${no}` so a later
 * drag-to-move feature can update one tower in place (moveTower) without rebuilding the
 * route data or the map layers.
 */
export interface TowerFeatureProps {
  fid: string;
  route: number;
  no: number;
  type: string;
  angle: boolean;
  gantry: boolean;
  label: string;
}

export interface StoredTower extends Tower {
  fid: string;
  route: number;
  /** true once the user has moved it (future drag editing) */
  moved?: boolean;
}

interface TowerState {
  solveId: number | null;
  byRoute: Record<number, StoredTower[]>;
  /** replace the store contents from a solve; no-op if the same solve is already loaded */
  load: (solveId: number, routes: Route[]) => void;
  moveTower: (fid: string, lon: number, lat: number) => void;
  clear: () => void;
}

export const towerId = (route: number, no: number) => `${route}-${no}`;

export const useTowers = create<TowerState>((set, get) => ({
  solveId: null,
  byRoute: {},
  load: (solveId, routes) => {
    if (get().solveId === solveId) return;
    const byRoute: Record<number, StoredTower[]> = {};
    for (const r of routes) {
      byRoute[r.id] = r.towers.map((t) => ({ ...t, fid: towerId(r.id, t.no), route: r.id }));
    }
    set({ solveId, byRoute });
  },
  moveTower: (fid, lon, lat) =>
    set((s) => {
      const [rs] = fid.split("-");
      const rid = Number(rs);
      const list = s.byRoute[rid];
      if (!list) return s;
      return {
        byRoute: { ...s.byRoute, [rid]: list.map((t) => (t.fid === fid ? { ...t, lon, lat, moved: true } : t)) },
      };
    }),
  clear: () => set({ solveId: null, byRoute: {} }),
}));

export function towersToGeoJSON(towers: StoredTower[]): GeoJSON.FeatureCollection<GeoJSON.Point, TowerFeatureProps> {
  return {
    type: "FeatureCollection",
    features: towers.map((t) => ({
      type: "Feature",
      id: undefined,
      geometry: { type: "Point", coordinates: [t.lon, t.lat] },
      properties: {
        fid: t.fid,
        route: t.route,
        no: t.no,
        type: t.type,
        angle: Boolean(t.angle_point),
        gantry: t.type === "G",
        label: t.type === "G" ? "Gantry" : `${t.no} · ${t.type}`,
      },
    })),
  };
}
