import { useCallback, useEffect, useRef, useState } from "react";
import type maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useQueryClient } from "@tanstack/react-query";
import { api, qk } from "../../api/hooks";
import type { Route, Solve } from "../../api/types";
import { towerId, useTowers } from "../../store/towers";
import { useUi } from "../../store/ui";
import { Btn } from "../ui";

/**
 * Move towers by hand (Vivek, 10 Oct 2026). While the mode is on, towers of the selected route can be
 * dragged; every position is checked live against the solve's rules (the list beside the tower names
 * each rule it breaks, red for problems the move adds), the two spans redraw with their lengths and
 * the turns at the three affected towers, and cells where no tower may stand are shaded. A drop that
 * breaks a rule is allowed and flagged. Saving writes a new solve version; the original is kept.
 */
export interface Check {
  ok: boolean | null;
  preparing?: boolean;
  tower_rules: { rule: string; name: string; feature: string | null; limit_m: number; distance_m: number | null }[];
  spans: { a: number; b: number; length_m: number; ok: boolean; problems: string[]; crossings: { name: string; angle_deg: number }[] }[];
  turns: { tower: number; deflection_deg: number; type: string; problems: string[] }[];
  new_problems?: string[];
  existing_problems?: string[];
}

const NOGO_SRC = "nogo", NOGO_LAYER = "nogo-shade", PREV_SRC = "move-preview", PREV_LAYER = "move-preview-line";

/** Drag handlers TowerLayer calls while a tower is being moved (set only while the mode is on). */
export const moveHandlers: { drag: ((fid: string, lon: number, lat: number) => void) | null; drop: ((fid: string, lon: number, lat: number) => void) | null } = { drag: null, drop: null };

export default function TowerMoveTool({ map, sid, runId, route, active, onExit }: {
  map: maplibregl.Map | null; sid: number; runId: string; route: Route; active: boolean; onExit: () => void;
}) {
  const qc = useQueryClient();
  const push = useUi((s) => s.push);
  const moveTower = useTowers((s) => s.moveTower);
  const byRoute = useTowers((s) => s.byRoute);
  const [ready, setReady] = useState(false);
  const [moves, setMoves] = useState<{ tower_no: number; lon: number; lat: number }[]>([]);
  const [check, setCheck] = useState<{ fid: string; px: { x: number; y: number }; c: Check } | null>(null);
  const [saving, setSaving] = useState(false);
  const inflight = useRef<AbortController | null>(null);
  const lastCall = useRef(0);
  const original = useRef<Record<number, [number, number]>>({});

  // the solve's rules load in the API on first use (tens of seconds); poll until ready
  useEffect(() => {
    if (!active) return;
    let stop = false;
    const poll = async () => {
      try {
        const st = await api<{ ready: boolean }>(`/api/solves/${sid}/edit/status?prepare=true`);
        if (stop) return;
        setReady(st.ready);
        if (!st.ready) window.setTimeout(poll, 2000);
      } catch { if (!stop) window.setTimeout(poll, 4000); }
    };
    poll();
    original.current = Object.fromEntries(route.towers.map((t) => [t.no, [t.lon, t.lat] as [number, number]]));
    return () => { stop = true; };
  }, [active, sid, route]);

  // no-tower shading while the mode is on
  useEffect(() => {
    if (!map || !active || !ready) return;
    if (!map.getSource(NOGO_SRC)) {
      map.addSource(NOGO_SRC, { type: "raster", tiles: [`${window.location.origin}/api/solves/${sid}/nogo/{z}/{x}/{y}.png`], tileSize: 256, minzoom: 10, maxzoom: 16 });
      map.addLayer({ id: NOGO_LAYER, type: "raster", source: NOGO_SRC, paint: { "raster-opacity": 0.75 } }, "towers-circle");
    }
    if (!map.getSource(PREV_SRC)) {
      map.addSource(PREV_SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      map.addLayer({ id: PREV_LAYER, type: "line", source: PREV_SRC, paint: { "line-color": ["case", ["get", "bad"], "#e0473a", "#ffd166"], "line-width": 3, "line-dasharray": [2, 1.5] } });
    }
    return () => {
      for (const id of [NOGO_LAYER, PREV_LAYER]) if (map.getLayer(id)) map.removeLayer(id);
      for (const id of [NOGO_SRC, PREV_SRC]) if (map.getSource(id)) map.removeSource(id);
    };
  }, [map, active, ready, sid]);

  const preview = useCallback((no: number, lon: number, lat: number, c: Check | null) => {
    if (!map) return;
    const src = map.getSource(PREV_SRC) as GeoJSONSource | undefined;
    if (!src) return;
    const list = byRoute[route.id] ?? [];
    const t = (n: number) => list.find((x) => x.no === n);
    const a = t(no - 1), b = t(no + 1);
    const feats: GeoJSON.Feature[] = [];
    if (a) feats.push({ type: "Feature", properties: { bad: Boolean(c && !c.spans[0]?.ok) }, geometry: { type: "LineString", coordinates: [[a.lon, a.lat], [lon, lat]] } });
    if (b) feats.push({ type: "Feature", properties: { bad: Boolean(c && !c.spans[1]?.ok) }, geometry: { type: "LineString", coordinates: [[lon, lat], [b.lon, b.lat]] } });
    src.setData({ type: "FeatureCollection", features: feats });
  }, [map, byRoute, route.id]);

  const onDrag = useCallback(async (fid: string, lon: number, lat: number) => {
    if (!map) return;
    const no = Number(fid.split("-")[1]);
    preview(no, lon, lat, null);
    const now = Date.now();
    if (now - lastCall.current < 140) return;
    lastCall.current = now;
    inflight.current?.abort();
    const ac = new AbortController(); inflight.current = ac;
    try {
      const res = await fetch(`/api/solves/${sid}/edit/check`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ route_id: route.id, tower_no: no, lon, lat }), signal: ac.signal });
      if (!res.ok) return;
      const c = (await res.json()) as Check;
      const px = map.project([lon, lat]);
      setCheck({ fid, px: { x: px.x, y: px.y }, c });
      preview(no, lon, lat, c);
    } catch { /* aborted */ }
  }, [map, sid, route.id, preview]);

  const onDrop = useCallback((fid: string, lon: number, lat: number) => {
    const no = Number(fid.split("-")[1]);
    setMoves((m) => [...m.filter((x) => x.tower_no !== no), { tower_no: no, lon, lat }]);
    onDrag(fid, lon, lat);
  }, [onDrag]);

  useEffect(() => {
    if (!active) return;
    moveHandlers.drag = onDrag; moveHandlers.drop = onDrop;
    return () => { moveHandlers.drag = null; moveHandlers.drop = null; };
  }, [active, onDrag, onDrop]);

  const undo = () => {
    const last = moves[moves.length - 1];
    if (!last) return;
    const o = original.current[last.tower_no];
    if (o) moveTower(towerId(route.id, last.tower_no), o[0], o[1]);
    setMoves((m) => m.slice(0, -1));
    setCheck(null);
    preview(last.tower_no, o?.[0] ?? 0, o?.[1] ?? 0, null);
  };
  const cancel = () => {
    for (const mv of moves) { const o = original.current[mv.tower_no]; if (o) moveTower(towerId(route.id, mv.tower_no), o[0], o[1]); }
    setMoves([]); setCheck(null); onExit();
  };
  const save = async () => {
    setSaving(true);
    try {
      const out = await api<{ solve_id: number; solve: Solve }>(`/api/solves/${sid}/edit/apply`, { method: "POST", json: { route_id: route.id, moves, note: `${moves.length} tower(s) moved by hand` } });
      push({ kind: "ok", text: `Saved as solve ${out.solve_id} — route ${route.id} with ${moves.length} moved tower${moves.length > 1 ? "s" : ""}.` });
      useTowers.getState().clear();
      setMoves([]); setCheck(null); onExit();
      await qc.invalidateQueries({ queryKey: qk.run(runId) });
      await qc.invalidateQueries({ queryKey: qk.runs });
      void qc.invalidateQueries({ queryKey: ["solve", runId] });
    } catch (e) {
      push({ kind: "warn", text: e instanceof Error ? e.message : String(e) });
    } finally { setSaving(false); }
  };

  if (!active) return null;
  return (
    <>
      <div className="move-bar" data-testid="move-bar">
        <b>Move towers</b>
        {!ready ? <span className="hint">Preparing the solve's rules for live checks…</span> : <span className="hint">Drag a tower of route {route.id}. Red cells: no tower may stand there.</span>}
        <span style={{ marginLeft: "auto" }} className="mono">{moves.length} moved</span>
        <button className="btn ghost" onClick={undo} disabled={!moves.length}>Undo</button>
        <button className="btn ghost" onClick={cancel}>Cancel</button>
        <Btn onClick={save} disabled={!moves.length || !ready} pending={saving} data-testid="move-save">Save as new version</Btn>
      </div>
      {check && check.c.ok !== null ? (
        <div className={`move-check${check.c.ok ? " ok" : ""}`} style={{ left: check.px.x + 14, top: check.px.y + 14 }} data-testid="move-check">
          <div className="mc-head">Tower {check.fid.split("-")[1]} · {check.c.ok ? "no new rule broken" : `${check.c.new_problems?.length ?? 0} rule${(check.c.new_problems?.length ?? 0) === 1 ? "" : "s"} broken`}</div>
          {(check.c.new_problems ?? []).map((p, i) => <div key={`n${i}`} className="mc-bad">{p}</div>)}
          {(check.c.existing_problems ?? []).map((p, i) => <div key={`e${i}`} className="mc-old">{p} <em>(already there)</em></div>)}
          <div className="mc-spans mono">
            {check.c.spans.map((s) => <span key={`${s.a}-${s.b}`} className={s.ok ? "" : "bad"}>span {s.a}→{s.b} {s.length_m.toFixed(0)} m{s.crossings.length ? ` · ${s.crossings.map((c) => `${c.name} ${c.angle_deg.toFixed(0)}°`).join(", ")}` : ""}</span>)}
            {check.c.turns.map((t) => <span key={t.tower} className={t.problems.length ? "bad" : ""}>#{t.tower} {t.deflection_deg.toFixed(1)}° → {t.type}</span>)}
          </div>
        </div>
      ) : null}
    </>
  );
}
