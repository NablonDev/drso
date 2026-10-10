import { useEffect, useRef, useState } from "react";
import type maplibregl from "maplibre-gl";
import { useDemCoverage } from "../../api/hooks";

/**
 * Ground height on the map (10 Oct 2026): hillshade overlay with opacity, optional 3D terrain, and a
 * cursor readout "lat, lon · elevation". Both come from the Copernicus GLO-30 tiles the solver used.
 */
export default function TerrainControl({ map, runId }: { map: maplibregl.Map | null; runId: string }) {
  const cov = useDemCoverage(runId);
  const [shade, setShade] = useState(false);
  const [opacity, setOpacity] = useState(0.7);
  const [threeD, setThreeD] = useState(false);
  const [readout, setReadout] = useState<{ lat: number; lon: number; z: number | null } | null>(null);
  const timer = useRef<number | null>(null);
  const abort = useRef<AbortController | null>(null);
  const ready = Boolean(map && cov.data?.complete);

  useEffect(() => {
    if (!map) return;
    const apply = () => {
      if (!map.getLayer("hillshade")) return;
      map.setLayoutProperty("hillshade", "visibility", shade ? "visible" : "none");
      map.setPaintProperty("hillshade", "hillshade-exaggeration", 0.65 * opacity);
    };
    if (map.isStyleLoaded()) apply();
    else map.once("styledata", apply);
  }, [map, shade, opacity]);

  useEffect(() => {
    if (!map) return;
    try {
      map.setTerrain(threeD ? { source: "dem", exaggeration: 1.5 } : null);
      if (threeD) map.easeTo({ pitch: 55, duration: 600 });
      else if (map.getPitch() > 0) map.easeTo({ pitch: 0, duration: 400 });
    } catch {
      /* terrain unsupported in this browser */
    }
    return () => {
      try { map.setTerrain(null); } catch { /* map gone */ }
    };
  }, [map, threeD]);

  // cursor elevation: from the 3D terrain when it is on, else from the API (debounced)
  useEffect(() => {
    if (!map || !ready) return;
    const onMove = (e: maplibregl.MapMouseEvent) => {
      const { lng, lat } = e.lngLat;
      if (threeD) {
        const z = map.queryTerrainElevation(e.lngLat);
        setReadout({ lat, lon: lng, z: z === null || z === undefined ? null : Math.round(z * 10) / 10 });
        return;
      }
      setReadout((r) => ({ lat, lon: lng, z: r?.z ?? null }));
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(async () => {
        abort.current?.abort();
        const ac = new AbortController();
        abort.current = ac;
        try {
          const res = await fetch(`/api/dem/elevation?lon=${lng}&lat=${lat}`, { signal: ac.signal });
          if (res.ok) {
            const d = (await res.json()) as { elevation_m: number | null };
            setReadout({ lat, lon: lng, z: d.elevation_m });
          }
        } catch {
          /* aborted or offline */
        }
      }, 120);
    };
    const onOut = () => setReadout(null);
    map.on("mousemove", onMove);
    map.getCanvas().addEventListener("mouseleave", onOut);
    return () => {
      map.off("mousemove", onMove);
      map.getCanvas().removeEventListener("mouseleave", onOut);
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [map, ready, threeD]);

  if (!map) return null;
  if (cov.data && !cov.data.complete) {
    return <div className="terrain-ctl" data-testid="terrain-control"><span className="hint">No terrain fetched for this run</span></div>;
  }
  return (
    <div className="terrain-ctl" data-testid="terrain-control">
      <div className="terrain-row">
        <button aria-pressed={shade} onClick={() => setShade((x) => !x)} disabled={!ready} title="Hillshade from the Copernicus GLO-30 DEM" data-testid="terrain-hillshade">Hillshade</button>
        <button aria-pressed={threeD} onClick={() => setThreeD((x) => !x)} disabled={!ready} title="Tilt the map over the terrain (1.5× exaggeration)" data-testid="terrain-3d">3D</button>
        {shade ? (
          <input type="range" min={0.2} max={1} step={0.1} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} aria-label="Hillshade strength" />
        ) : null}
      </div>
      {readout ? (
        <div className="terrain-readout mono" data-testid="terrain-readout">
          {readout.lat.toFixed(5)}, {readout.lon.toFixed(5)} · {readout.z === null ? "—" : `${readout.z.toFixed(1)} m`}
        </div>
      ) : null}
    </div>
  );
}
