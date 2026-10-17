import { useEffect, useMemo } from "react";
import type maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useBuffers } from "../../api/hooks";
import type { BufferItem, LonLat, Route } from "../../api/types";
import { DENSE_MINZOOM, LAYER_COLOURS } from "./style";

const SRC = "near-route";
const IDS = ["near-bld", "near-road-casing", "near-road"];
const NEAR_M = 150;

function thin(geom: LonLat[]): LonLat[] {
  const step = Math.max(1, Math.ceil(geom.length / 400));
  const out = geom.filter((_, i) => i % step === 0);
  if (out[out.length - 1] !== geom[geom.length - 1]) out.push(geom[geom.length - 1]);
  return out;
}

/**
 * Village roads and buildings within 150 m of the selected route, drawn from z10 until the full layers
 * arrive in the tiles at z12 (17 Oct 2026): the tiles leave both out below z12 because they are too heavy
 * for the whole band, but these few near the line are what a route is about to meet.
 */
export default function NearRouteLayer({ map, runId, sid, route, roads, buildings }: {
  map: maplibregl.Map | null; runId: string; sid: number; route: Route; roads: boolean; buildings: boolean;
}) {
  const line = useMemo(() => thin(route.geometry), [route.geometry]);
  const items = useMemo<BufferItem[]>(() => [
    { kind: "near", layer: "village_roads", buffer_m: 0, near_m: NEAR_M, limit: 2000 },
    { kind: "near", layer: "buildings", buffer_m: 0, near_m: NEAR_M, limit: 2000 },
  ], []);
  const q = useBuffers(runId, `${sid}-${route.id}-near`, { route: line, items }, roads || buildings);

  useEffect(() => {
    if (!map) return;
    if (!map.getSource(SRC)) {
      map.addSource(SRC, { type: "geojson", data: { type: "FeatureCollection", features: [] } });
      const before = map.getLayer("routes-casing") ? "routes-casing" : undefined;
      const z = { minzoom: 10, maxzoom: DENSE_MINZOOM } as const;
      // a building is a fraction of a pixel here: a fixed-width stroke round it makes a visible dot
      map.addLayer({ id: "near-bld", type: "line", source: SRC, ...z, filter: ["==", ["get", "layer"], "buildings"],
        paint: { "line-color": LAYER_COLOURS.buildings, "line-width": 3, "line-opacity": 0.95 } }, before);
      map.addLayer({ id: "near-road-casing", type: "line", source: SRC, ...z, filter: ["==", ["get", "layer"], "village_roads"],
        layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": "#1d1708", "line-width": 2.6, "line-opacity": 0.5 } }, before);
      map.addLayer({ id: "near-road", type: "line", source: SRC, ...z, filter: ["==", ["get", "layer"], "village_roads"],
        layout: { "line-cap": "round", "line-join": "round" }, paint: { "line-color": LAYER_COLOURS.village_roads, "line-width": 1.4, "line-opacity": 0.95 } }, before);
    }
    return () => {
      try {
        if (map.getStyle()) {
          for (const id of IDS) if (map.getLayer(id)) map.removeLayer(id);
          if (map.getSource(SRC)) map.removeSource(SRC);
        }
      } catch {
        /* map gone */
      }
    };
  }, [map]);

  useEffect(() => {
    if (!map) return;
    (map.getSource(SRC) as GeoJSONSource | undefined)?.setData(q.data ?? { type: "FeatureCollection", features: [] });
  }, [map, q.data]);

  useEffect(() => {
    if (!map || !map.getLayer("near-bld")) return;
    map.setLayoutProperty("near-bld", "visibility", buildings ? "visible" : "none");
    for (const id of ["near-road-casing", "near-road"]) map.setLayoutProperty(id, "visibility", roads ? "visible" : "none");
  }, [map, roads, buildings, q.data]);

  return null;
}
