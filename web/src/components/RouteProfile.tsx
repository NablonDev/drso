import { useEffect, useMemo, useRef, useState } from "react";
import type maplibregl from "maplibre-gl";
import { useRouteProfile } from "../api/hooks";
import type { Route } from "../api/types";
import { Loading } from "./ui";

const W = 1100, H = 240, PAD = { l: 44, r: 12, t: 10, b: 26 };

/**
 * Longitudinal section of the selected route (10 Oct 2026): ground from the GLO-30 DEM, towers with
 * their attachment height (extensions included), the conductor sag per span and the clearance line.
 * Hover shows chainage / ground / nearest tower and marks the spot on the map; drag selects a window.
 */
export default function RouteProfile({ sid, route, map, enabled }: { sid: number; route: Route; map: maplibregl.Map | null; enabled: boolean }) {
  const q = useRouteProfile(sid, route.id, enabled);
  const [win, setWin] = useState<[number, number] | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const drag = useRef<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const p = q.data;

  useEffect(() => setWin(null), [route.id, sid]);

  const view = useMemo(() => {
    if (!p) return null;
    const [x0, x1] = win ?? [0, p.length_m];
    const gx = p.ground.x, gz = p.ground.z;
    const zs: number[] = [];
    for (let i = 0; i < gx.length; i++) if (gx[i] >= x0 && gx[i] <= x1 && gz[i] !== null) zs.push(gz[i] as number);
    for (const t of p.towers) if (t.chainage_m >= x0 && t.chainage_m <= x1 && t.attach_z !== null) zs.push(t.attach_z);
    if (!zs.length) return null;
    const zmin = Math.min(...zs) - 5, zmax = Math.max(...zs) + 5;
    const sx = (x: number) => PAD.l + ((x - x0) / Math.max(1, x1 - x0)) * (W - PAD.l - PAD.r);
    const sy = (z: number) => PAD.t + (1 - (z - zmin) / Math.max(1, zmax - zmin)) * (H - PAD.t - PAD.b);
    const inv = (px: number) => x0 + ((px - PAD.l) / (W - PAD.l - PAD.r)) * (x1 - x0);
    return { x0, x1, zmin, zmax, sx, sy, inv };
  }, [p, win]);

  // the hovered chainage as a marker on the map
  useEffect(() => {
    if (!map || !p) return;
    const id = "profile-cursor";
    const src = map.getSource(id) as maplibregl.GeoJSONSource | undefined;
    const feature = (() => {
      if (hover === null) return { type: "FeatureCollection" as const, features: [] };
      const T = p.towers;
      let i = T.findIndex((t) => t.chainage_m >= hover);
      if (i <= 0) i = Math.min(Math.max(i, 1), T.length - 1);
      const a = T[i - 1], b = T[i];
      const f = (hover - a.chainage_m) / Math.max(1, b.chainage_m - a.chainage_m);
      return { type: "FeatureCollection" as const, features: [{ type: "Feature" as const, properties: {}, geometry: { type: "Point" as const, coordinates: [a.lon + f * (b.lon - a.lon), a.lat + f * (b.lat - a.lat)] } }] };
    })();
    if (!src) {
      if (!map.isStyleLoaded()) return;
      map.addSource(id, { type: "geojson", data: feature });
      map.addLayer({ id, type: "circle", source: id, paint: { "circle-radius": 7, "circle-color": "#ffd166", "circle-stroke-color": "#1b1b1b", "circle-stroke-width": 2 } });
    } else {
      src.setData(feature);
    }
  }, [map, p, hover]);

  if (!enabled) return null;
  if (q.isPending) return <div className="card" style={{ marginTop: 10 }}><div className="cb"><Loading text="Reading the ground profile…" /></div></div>;
  if (q.isError || !p) return null;
  if (!p.dem) return <div className="card" style={{ marginTop: 10 }}><div className="cb hint">No terrain on disk for this route — the profile needs the DEM the solver fetched.</div></div>;

  const v = view;
  const px = (e: React.MouseEvent) => {
    const r = svgRef.current!.getBoundingClientRect();
    return ((e.clientX - r.left) / r.width) * W;
  };
  const onMove = (e: React.MouseEvent) => {
    if (!v) return;
    const x = Math.max(v.x0, Math.min(v.x1, v.inv(px(e))));
    setHover(x);
  };
  const onDown = (e: React.MouseEvent) => { if (v) drag.current = v.inv(px(e)); };
  const onUp = (e: React.MouseEvent) => {
    if (!v || drag.current === null) return;
    const a = drag.current, b = v.inv(px(e));
    drag.current = null;
    if (Math.abs(b - a) > 200) setWin([Math.max(0, Math.min(a, b)), Math.min(p.length_m, Math.max(a, b))]);
    else if (map) {
      // click: fly the map to this chainage
      const T = p.towers; let i = T.findIndex((t) => t.chainage_m >= b); i = Math.min(Math.max(i, 1), T.length - 1);
      const t0 = T[i - 1], t1 = T[i]; const f = (b - t0.chainage_m) / Math.max(1, t1.chainage_m - t0.chainage_m);
      map.flyTo({ center: [t0.lon + f * (t1.lon - t0.lon), t0.lat + f * (t1.lat - t0.lat)], zoom: Math.max(map.getZoom(), 15), duration: 600 });
    }
  };
  const nearTower = hover === null ? null : p.towers.reduce((best, t) => (Math.abs(t.chainage_m - hover) < Math.abs(best.chainage_m - hover) ? t : best), p.towers[0]);
  const groundAt = (x: number) => {
    const gx = p.ground.x; let i = gx.findIndex((g) => g >= x); if (i < 0) i = gx.length - 1;
    return p.ground.z[i];
  };
  const short = p.spans.filter((s) => s.min_clearance_m !== null && (s.min_clearance_m as number) < p.clearance_m);

  return (
    <div className="card profile" style={{ marginTop: 10 }} data-testid="route-profile">
      <div className="ch">
        <h3>Ground profile — route {route.id}</h3>
        <span className="hint" style={{ marginLeft: "auto" }}>
          {p.dem} · clearance {p.clearance_m} m · {short.length ? `${short.length} span${short.length > 1 ? "s" : ""} below it` : "every span clears"}
          {win ? <> · <button className="link" onClick={() => setWin(null)}>Whole route</button></> : " · drag to zoom, click to fly there"}
        </span>
      </div>
      {v ? (
        <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="profile-svg" onMouseMove={onMove} onMouseLeave={() => { setHover(null); drag.current = null; }} onMouseDown={onDown} onMouseUp={onUp}>
          {/* axes */}
          {[0, 0.25, 0.5, 0.75, 1].map((f) => {
            const z = v.zmin + f * (v.zmax - v.zmin);
            return <g key={f}><line x1={PAD.l} x2={W - PAD.r} y1={v.sy(z)} y2={v.sy(z)} className="pf-grid" /><text x={PAD.l - 4} y={v.sy(z) + 3} className="pf-tick" textAnchor="end">{z.toFixed(0)}</text></g>;
          })}
          {Array.from({ length: 11 }, (_, i) => v.x0 + (i / 10) * (v.x1 - v.x0)).map((x) => (
            <text key={x} x={v.sx(x)} y={H - 8} className="pf-tick" textAnchor="middle">{(x / 1000).toFixed(v.x1 - v.x0 < 5000 ? 2 : 1)} km</text>
          ))}
          {/* clearance band: ground + required clearance */}
          <polyline className="pf-clear" points={p.ground.x.map((x, i) => (p.ground.z[i] === null || x < v.x0 || x > v.x1 ? null : `${v.sx(x)},${v.sy((p.ground.z[i] as number) + p.clearance_m)}`)).filter(Boolean).join(" ")} />
          {/* ground */}
          <polyline className="pf-ground" points={p.ground.x.map((x, i) => (p.ground.z[i] === null || x < v.x0 || x > v.x1 ? null : `${v.sx(x)},${v.sy(p.ground.z[i] as number)}`)).filter(Boolean).join(" ")} />
          {/* conductor per span: parabola between attachments with the span's sag */}
          {p.spans.map((s) => {
            if (s.attach_a === null || s.attach_b === null || s.chainage_b < v.x0 || s.chainage_a > v.x1) return null;
            const pts: string[] = [];
            for (let k = 0; k <= 24; k++) {
              const t = k / 24; const x = s.chainage_a + t * (s.chainage_b - s.chainage_a);
              const z = (1 - t) * s.attach_a + t * s.attach_b - 4 * s.sag_m * t * (1 - t);
              pts.push(`${v.sx(x)},${v.sy(z)}`);
            }
            const bad = s.min_clearance_m !== null && (s.min_clearance_m as number) < p.clearance_m;
            return <polyline key={`${s.a}-${s.b}`} className={bad ? "pf-wire bad" : "pf-wire"} points={pts.join(" ")}><title>{`Span ${s.a}→${s.b}: ${s.length_m.toFixed(0)} m, sag ${s.sag_m} m${s.min_clearance_m !== null ? `, clearance ${s.min_clearance_m} m` : ""}${s.crossings.length ? `, crosses ${s.crossings.join(", ")}` : ""}`}</title></polyline>;
          })}
          {/* towers */}
          {p.towers.map((t) => {
            if (t.ground_z === null || t.attach_z === null || t.chainage_m < v.x0 || t.chainage_m > v.x1) return null;
            const x = v.sx(t.chainage_m);
            const cls = t.violations.length ? "pf-tower viol" : t.type === "A" ? "pf-tower" : "pf-tower tension";
            return <g key={t.no} className={cls}><line x1={x} x2={x} y1={v.sy(t.ground_z)} y2={v.sy(t.attach_z)} /><title>{`#${t.no} ${t.type}${t.extension_m ? ` +${t.extension_m} m` : ""} · ground ${t.ground_z} m · km ${(t.chainage_m / 1000).toFixed(2)}${t.violations.length ? ` · ${t.violations.join(", ")}` : ""}`}</title></g>;
          })}
          {/* hover cursor */}
          {hover !== null ? <line className="pf-cursor" x1={v.sx(hover)} x2={v.sx(hover)} y1={PAD.t} y2={H - PAD.b} /> : null}
        </svg>
      ) : <div className="cb hint">No ground data in this window.</div>}
      <div className="profile-foot mono">
        {hover !== null && nearTower ? (
          <>km {(hover / 1000).toFixed(2)} · ground {groundAt(hover) ?? "—"} m · nearest tower #{nearTower.no} {nearTower.type}{nearTower.extension_m ? ` +${nearTower.extension_m} m` : ""} at km {(nearTower.chainage_m / 1000).toFixed(2)}</>
        ) : (
          <>{p.towers.length - 2} towers · {(p.length_m / 1000).toFixed(1)} km · attachment {p.attachment_m} m above ground at standard height · sag {p.spans.length ? `${Math.min(...p.spans.map((s) => s.sag_m)).toFixed(1)}–${Math.max(...p.spans.map((s) => s.sag_m)).toFixed(1)} m` : "—"}</>
        )}
      </div>
    </div>
  );
}
