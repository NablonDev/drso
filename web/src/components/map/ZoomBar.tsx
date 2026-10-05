import { useEffect, useState } from "react";
import type maplibregl from "maplibre-gl";
import { ESRI_MAX_ZOOM, metresPerPixel } from "./style";
import { useZoom } from "./useMap";

/**
 * The prototype's zoom readout: − z15 + with the native-resolution note. Zooming in past z18
 * (button, wheel or pinch) is refused and the "zoom stops here" note appears.
 */
export function useCeiling(map: maplibregl.Map | null) {
  const [blocked, setBlocked] = useState(false);
  const z = useZoom(map);
  useEffect(() => {
    if (z < ESRI_MAX_ZOOM - 0.05) setBlocked(false);
  }, [z]);
  useEffect(() => {
    if (!map) return;
    const el = map.getCanvasContainer();
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY < 0 && map.getZoom() >= ESRI_MAX_ZOOM - 0.01) setBlocked(true);
    };
    el.addEventListener("wheel", onWheel, { passive: true });
    return () => el.removeEventListener("wheel", onWheel);
  }, [map]);
  return { z, blocked, setBlocked };
}

export function ZoomBar({ map, ceiling }: { map: maplibregl.Map | null; ceiling: ReturnType<typeof useCeiling> }) {
  const { z, blocked, setBlocked } = ceiling;
  const zi = Math.round(z * 10) / 10;
  const atMax = z >= ESRI_MAX_ZOOM - 0.05;
  return (
    <span className="m8z" data-testid="zoombar">
      <button className="zb" aria-label="Zoom out" disabled={!map || z <= 4} onClick={() => map?.zoomOut()}>
        −
      </button>
      <span className="zl" aria-live="polite">z{Number.isInteger(zi) ? zi : zi.toFixed(1)}</span>
      <button
        className="zb"
        aria-label="Zoom in"
        disabled={!map}
        onClick={() => {
          if (!map) return;
          if (atMax) setBlocked(true);
          else map.zoomIn();
        }}
      >
        +
      </button>
      <span className={`nat ${blocked || atMax ? "natbad" : ""}`}>
        {atMax ? `at the ceiling · native z${ESRI_MAX_ZOOM}` : z < 13 ? "zoom in for 0.5 m detail" : `native z${ESRI_MAX_ZOOM} · Esri World Imagery`}
      </span>
    </span>
  );
}

export function CeilingNote({ ceiling }: { ceiling: ReturnType<typeof useCeiling> }) {
  if (!ceiling.blocked) return null;
  return (
    <div className="note n-stop" style={{ margin: "8px 0 0" }} role="status" data-testid="ceiling-note">
      <div>
        <b>Zoom stops at z{ESRI_MAX_ZOOM} here.</b> The imagery under this view is Esri World Imagery at about 0.5 m, so anything finer
        is interpolation, not detail. The viewer refuses to upscale rather than show you pixels the models never saw.
      </div>
    </div>
  );
}

export function ScaleBar({ map }: { map: maplibregl.Map | null }) {
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!map) return;
    const f = () => setTick((x) => x + 1);
    map.on("moveend", f);
    map.on("zoomend", f);
    return () => {
      map.off("moveend", f);
      map.off("zoomend", f);
    };
  }, [map]);
  if (!map) return null;
  const m = metresPerPixel(map.getCenter().lat, map.getZoom());
  const targets = [5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000];
  let best = targets[0];
  for (const t of targets) if (t / m <= 190) best = t;
  const px = best / m;
  return (
    <div className="sbar">
      <i style={{ width: `${px.toFixed(0)}px` }} />
      <b>{best >= 1000 ? `${best / 1000} km` : `${best} m`}</b>
      <span>{m < 10 ? m.toFixed(2) : m.toFixed(0)} m/px</span>
    </div>
  );
}
