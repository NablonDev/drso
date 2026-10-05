import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl from "maplibre-gl";
import { addLowConfLayer, addRunLayers, baseStyle, ESRI_MAX_ZOOM, removeRunLayers } from "./style";

export type Bounds = [number, number, number, number];

/**
 * Create a MapLibre map in `container` and return it once its style has loaded.
 * Zoom is capped at the Esri imagery's native z18 — the viewer refuses to upscale.
 */
export function useMapLibre(
  container: React.RefObject<HTMLDivElement>,
  opts: { bounds?: Bounds | null; interactive?: boolean; padding?: number } = {},
) {
  const [map, setMap] = useState<maplibregl.Map | null>(null);
  const boundsRef = useRef(opts.bounds);
  boundsRef.current = opts.bounds;
  const interactive = opts.interactive ?? true;
  const padding = opts.padding ?? 30;

  useEffect(() => {
    const el = container.current;
    if (!el) return;
    const b = boundsRef.current;
    const m = new maplibregl.Map({
      container: el,
      style: baseStyle(),
      maxZoom: ESRI_MAX_ZOOM,
      minZoom: 3,
      interactive,
      attributionControl: interactive ? { compact: true } : false,
      ...(b ? { bounds: b as maplibregl.LngLatBoundsLike, fitBoundsOptions: { padding } } : { center: [72.5, 22.8], zoom: 6 }),
      dragRotate: false,
      pitchWithRotate: false,
      touchPitch: false,
      fadeDuration: 0,
    });
    m.touchZoomRotate?.disableRotation();
    // A listener keeps MapLibre from printing tile errors to the console; the screens decide what to show.
    m.on("error", () => undefined);
    let alive = true;
    m.once("load", () => {
      if (!alive) return;
      m.resize();
      setMap(m);
    });
    // MapLibre only tracks window resizes; layout changes (grid, sidebar, fonts) need this.
    const ro = new ResizeObserver(() => {
      if (alive) m.resize();
    });
    ro.observe(el);
    return () => {
      alive = false;
      ro.disconnect();
      setMap(null);
      m.remove();
    };
  }, [container, interactive, padding]);

  return map;
}

/** Fit once when bounds first become known (or change identity by value). */
export function useFitBounds(map: maplibregl.Map | null, bounds: Bounds | null | undefined, padding = 30) {
  const key = bounds ? bounds.map((x) => x.toFixed(5)).join(",") : "";
  const done = useRef<string>("");
  useEffect(() => {
    if (!map || !bounds || done.current === key) return;
    // wait until the container has a real size, or the fit lands at a useless zoom
    const tryFit = () => {
      const c = map.getContainer();
      if (c.clientWidth < 200 || c.clientHeight < 150) return;
      done.current = key;
      map.off("resize", tryFit);
      map.fitBounds(bounds as maplibregl.LngLatBoundsLike, { padding, duration: 0 });
    };
    map.on("resize", tryFit);
    tryFit();
    return () => {
      map.off("resize", tryFit);
    };
  }, [map, key, bounds, padding]);
}

/**
 * The run's vector tiles. If the tile server fails, stop requesting tiles (remove the source)
 * and report the error so the screen can say so; `retry` re-adds it.
 */
export function useRunTiles(map: maplibregl.Map | null, runId: string, enabled: boolean, onReady?: (m: maplibregl.Map) => void) {
  const [error, setError] = useState<string | null>(null);
  const [epoch, setEpoch] = useState(0);
  const ready = useRef(onReady);
  ready.current = onReady;

  useEffect(() => {
    if (!map || !enabled) return;
    setError(null);
    const firstSymbol = map.getLayer("band-line") ? "band-line" : undefined;
    addRunLayers(map, runId, { before: firstSymbol });
    addLowConfLayer(map);
    ready.current?.(map);
    const onErr = (e: { error?: unknown; sourceId?: string }) => {
      if (e.sourceId !== "run") return;
      const st = (e.error as { status?: number } | undefined)?.status;
      setError(
        st
          ? `The server could not draw the extracted layers as map tiles (HTTP ${st}).`
          : "The extracted layers could not be loaded as map tiles.",
      );
      // stop hammering a failing endpoint
      window.setTimeout(() => {
        try {
          removeRunLayers(map);
        } catch {
          /* map already gone */
        }
      }, 0);
    };
    map.on("error", onErr);
    return () => {
      map.off("error", onErr);
      try {
        if (map.getStyle()) removeRunLayers(map);
      } catch {
        /* map removed */
      }
    };
  }, [map, runId, enabled, epoch]);

  const retry = useCallback(() => setEpoch((x) => x + 1), []);
  return { error, retry };
}

export function useZoom(map: maplibregl.Map | null) {
  const [z, setZ] = useState<number>(map?.getZoom() ?? 0);
  useEffect(() => {
    if (!map) return;
    const f = () => setZ(map.getZoom());
    f();
    map.on("zoomend", f);
    map.on("zoom", f);
    return () => {
      map.off("zoomend", f);
      map.off("zoom", f);
    };
  }, [map]);
  return z;
}
