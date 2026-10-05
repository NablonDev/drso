import type maplibregl from "maplibre-gl";
import type { ExpressionSpecification, FilterSpecification, StyleSpecification } from "maplibre-gl";

export const ESRI_MAX_ZOOM = 18;

/** Layer colours from the prototype's MLAY array, keyed by API layer key. */
export const LAYER_COLOURS: Record<string, string> = {
  lulc: "#8FA36B",
  buildings: "#9A8574",
  clusters: "#A33A31",
  water: "#6E93AC",
  forest: "#3B7350",
  soil: "#A08B5B",
  plotted: "#9C5C14",
  roads: "#8C8274",
  rail: "#46596A",
  power: "#7A5EA8",
  telecom: "#2BB0A6",
  pipelines: "#B07A2A",
  wetlands: "#4E8C7A",
  aviation: "#9E6811",
  defence: "#5B6B7A",
  monuments: "#8A7A9A",
  protected: "#3B7350",
  substations: "#B5523B",
  solar: "#C9A227",
};

/** Transmission voltage colours (OSM power layer). */
export const VOLTAGE_COLOURS: { kv: number; colour: string }[] = [
  { kv: 765, colour: "#E0473A" },
  { kv: 400, colour: "#F28C28" },
  { kv: 220, colour: "#3FA7D6" },
  { kv: 132, colour: "#B28DFF" },
];

/** Highway class colours (OSM highways, the roads layer). */
export const HIGHWAY_COLOURS: { cls: string; label: string; colour: string }[] = [
  { cls: "nh", label: "NH", colour: "#F4C430" },
  { cls: "sh", label: "SH", colour: "#E8875B" },
];

function highwayColour(fallback: string): ExpressionSpecification {
  const m: unknown[] = ["match", ["get", "class"]];
  for (const h of HIGHWAY_COLOURS) m.push(h.cls, h.colour);
  m.push(fallback);
  return m as ExpressionSpecification;
}

function voltageColour(fallback: string): ExpressionSpecification {
  const m: unknown[] = ["match", ["coalesce", ["get", "voltage_kv"], 0]];
  for (const v of VOLTAGE_COLOURS) m.push(v.kv, v.colour);
  m.push(fallback);
  return m as ExpressionSpecification;
}

/** LULC class colours from the prototype's LCLS array, keyed by API class. */
export const LULC_CLASSES: { key: string; label: string; colour: string }[] = [
  { key: "agriculture", label: "Agriculture — kharif", colour: "#B9C79A" },
  { key: "agriculture_fallow", label: "Agriculture — fallow", colour: "#D6D2A8" },
  { key: "orchard", label: "Orchard / plantation", colour: "#7E9B63" },
  { key: "shrub", label: "Shrub", colour: "#C2C49A" },
  { key: "barren", label: "Barren", colour: "#D9CFA6" },
  { key: "hard_rock", label: "Hard rock", colour: "#C6C3BC" },
  { key: "marsh", label: "Marsh", colour: "#9DB0A0" },
  { key: "salt_pan", label: "Salt pan", colour: "#E2E0D4" },
  { key: "sand_dunes", label: "Sand dunes", colour: "#DCCFA8" },
  { key: "social_forest", label: "Social forest", colour: "#6E8A5E" },
  { key: "reserve_forest", label: "Reserve forest", colour: "#3B7350" },
  { key: "water_body", label: "Water body", colour: "#9FBACD" },
  { key: "built_up", label: "Built-up", colour: "#B9A796" },
  { key: "unclassified", label: "Unclassified", colour: "#CFCFC6" },
];

/** Bottom-to-top drawing order. */
export const LAYER_ORDER = [
  "lulc",
  "soil",
  "forest",
  "protected",
  "wetlands",
  "defence",
  "aviation",
  "solar",
  "substations",
  "plotted",
  "clusters",
  "buildings",
  "water",
  "roads",
  "rail",
  "pipelines",
  "power",
  "telecom",
  "monuments",
];

export const DEFAULT_ON = new Set([
  "lulc",
  "buildings",
  "clusters",
  "water",
  "forest",
  "plotted",
  "roads",
  "rail",
  "power",
  "telecom",
  "wetlands",
  "substations",
  "solar",
]);

/** Layers the route map shows under "No-Go features". */
export const NOGO_LAYERS = ["clusters", "buildings", "protected", "wetlands", "defence", "aviation", "substations", "solar", "monuments"];

/** Route colours: the prototype's four hues, lifted for contrast over satellite imagery. */
export const ROUTE_COLOURS = ["#4FA3CF", "#9C82D6", "#E3AA45", "#6CC08A"];
export const routeColour = (id: number) => ROUTE_COLOURS[(id - 1) % ROUTE_COLOURS.length];

export const IS_POLY: FilterSpecification = ["match", ["geometry-type"], ["Polygon", "MultiPolygon"], true, false];
export const IS_LINE: FilterSpecification = ["match", ["geometry-type"], ["LineString", "MultiLineString"], true, false];
export const IS_POINT: FilterSpecification = ["match", ["geometry-type"], ["Point", "MultiPoint"], true, false];

const origin = () => (typeof window !== "undefined" ? window.location.origin : "");

export function baseStyle(): StyleSpecification {
  return {
    version: 8,
    sources: {
      esri: {
        type: "raster",
        tiles: [`${origin()}/api/tiles/esri/{z}/{x}/{y}`],
        tileSize: 256,
        maxzoom: ESRI_MAX_ZOOM,
        attribution: "Imagery © Esri World Imagery (Vantor, Maxar and the GIS user community)",
      },
    },
    layers: [
      { id: "bg", type: "background", paint: { "background-color": "#1d2421" } },
      { id: "esri", type: "raster", source: "esri", paint: { "raster-fade-duration": 120 } },
    ],
  };
}

function lulcColour(): ExpressionSpecification {
  const m: unknown[] = ["match", ["get", "class"]];
  for (const c of LULC_CLASSES) m.push(c.key, c.colour);
  m.push("#CFCFC6");
  return m as ExpressionSpecification;
}

export const layerIds = (key: string) => [`L-${key}-fill`, `L-${key}-outline`, `L-${key}-line`, `L-${key}-point`];

/** Add the run's vector tile source and four styled sublayers per layer key. */
export function addRunLayers(map: maplibregl.Map, runId: string, opts: { before?: string } = {}) {
  if (map.getSource("run")) return;
  map.addSource("run", {
    type: "vector",
    tiles: [`${origin()}/api/tiles/${encodeURIComponent(runId)}/{z}/{x}/{y}.pbf`],
    minzoom: 6,
    maxzoom: 16,
  });
  for (const key of LAYER_ORDER) {
    const c = LAYER_COLOURS[key] ?? "#888";
    const fillColour = key === "lulc" ? lulcColour() : c;
    const minzoom = key === "buildings" ? 12 : 6;
    const common = { source: "run", "source-layer": key, minzoom } as const;
    map.addLayer(
      {
        id: `L-${key}-fill`,
        type: "fill",
        ...common,
        filter: IS_POLY,
        paint: {
          "fill-color": fillColour,
          "fill-opacity": key === "lulc" ? 0.42 : 0.3,
        },
      },
      opts.before,
    );
    map.addLayer(
      {
        id: `L-${key}-outline`,
        type: "line",
        ...common,
        filter: IS_POLY,
        paint: {
          "line-color": key === "lulc" ? fillColour : c,
          "line-width": ["case", ["boolean", ["feature-state", "sel"], false], 3, key === "lulc" ? 0.6 : 1.4],
          "line-opacity": 0.95,
        },
      },
      opts.before,
    );
    map.addLayer(
      {
        id: `L-${key}-line`,
        type: "line",
        ...common,
        filter: IS_LINE,
        layout: { "line-cap": "round", "line-join": "round" },
        paint: {
          "line-color": key === "power" ? voltageColour(c) : key === "roads" ? highwayColour(c) : c,
          "line-width": ["case", ["boolean", ["feature-state", "sel"], false], 5, key === "water" ? 3 : key === "rail" ? 2.4 : key === "roads" ? ["match", ["get", "class"], "nh", 3.2, 2.2] : 2.2],
          "line-opacity": 0.95,
          ...(key === "rail" ? { "line-dasharray": [3, 2] } : key === "pipelines" ? { "line-dasharray": [1, 2] } : {}),
        },
      },
      opts.before,
    );
    map.addLayer(
      {
        id: `L-${key}-point`,
        type: "circle",
        ...common,
        filter: IS_POINT,
        paint: {
          "circle-color": key === "power" ? voltageColour(c) : c,
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 8, 1.6, 13, 3.2, 16, 5],
          "circle-stroke-color": ["case", ["boolean", ["feature-state", "sel"], false], "#ffffff", "rgba(255,255,255,.7)"],
          "circle-stroke-width": ["case", ["boolean", ["feature-state", "sel"], false], 2.5, 0.8],
        },
      },
      opts.before,
    );
  }
}

export function removeRunLayers(map: maplibregl.Map) {
  for (const key of LAYER_ORDER) {
    for (const id of [...layerIds(key), `L-lowconf-${key}`]) if (map.getLayer(id)) map.removeLayer(id);
  }
  if (map.getSource("run")) map.removeSource("run");
}

export function setRunLayerVisibility(map: maplibregl.Map, visible: (key: string) => boolean) {
  for (const key of LAYER_ORDER) {
    const v = visible(key) ? "visible" : "none";
    for (const id of layerIds(key)) if (map.getLayer(id) && map.getLayoutProperty(id, "visibility") !== v) map.setLayoutProperty(id, "visibility", v);
  }
}

export function setRunLayerOpacity(map: maplibregl.Map, op: number) {
  for (const key of LAYER_ORDER) {
    const [fill, outline, line, point] = layerIds(key);
    if (!map.getLayer(fill)) continue;
    map.setPaintProperty(fill, "fill-opacity", (key === "lulc" ? 0.75 : 0.55) * op);
    map.setPaintProperty(outline, "line-opacity", Math.min(1, 0.4 + op));
    map.setPaintProperty(line, "line-opacity", Math.min(1, 0.3 + op));
    map.setPaintProperty(point, "circle-opacity", Math.min(1, 0.3 + op));
  }
}

export function setLulcClasses(map: maplibregl.Map, enabled: string[] | null) {
  for (const id of layerIds("lulc")) {
    if (!map.getLayer(id)) continue;
    const geom = id.endsWith("-line") ? IS_LINE : id.endsWith("-point") ? IS_POINT : IS_POLY;
    const f: FilterSpecification = enabled
      ? (["all", geom, ["in", ["get", "class"], ["literal", enabled]]] as FilterSpecification)
      : geom;
    map.setFilter(id, f);
  }
}

/** Red hatch-free outline over every feature below 0.80 confidence. */
export function addLowConfLayer(map: maplibregl.Map) {
  if (!map.getSource("run")) return;
  for (const key of LAYER_ORDER) {
    const id = `L-lowconf-${key}`;
    if (map.getLayer(id)) continue;
    map.addLayer({
      id,
      type: "line",
      source: "run",
      "source-layer": key,
      filter: ["<", ["coalesce", ["get", "confidence"], 1], 0.8],
      layout: { visibility: "none" },
      paint: { "line-color": "#E0473A", "line-width": 2, "line-dasharray": [2, 1.5], "line-opacity": 0.9 },
    });
  }
}

export function setLowConfVisible(map: maplibregl.Map, on: boolean) {
  for (const key of LAYER_ORDER) {
    const id = `L-lowconf-${key}`;
    if (map.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  }
}

export function bboxOf(coords: Iterable<[number, number]>): [number, number, number, number] | null {
  let x0 = Infinity,
    y0 = Infinity,
    x1 = -Infinity,
    y1 = -Infinity;
  for (const [x, y] of coords) {
    if (x < x0) x0 = x;
    if (y < y0) y0 = y;
    if (x > x1) x1 = x;
    if (y > y1) y1 = y;
  }
  return Number.isFinite(x0) ? [x0, y0, x1, y1] : null;
}

export function* geomCoords(g: GeoJSON.Geometry | null | undefined): Generator<[number, number]> {
  if (!g) return;
  switch (g.type) {
    case "Point":
      yield g.coordinates as [number, number];
      break;
    case "LineString":
    case "MultiPoint":
      for (const c of g.coordinates) yield c as [number, number];
      break;
    case "Polygon":
    case "MultiLineString":
      for (const r of g.coordinates) for (const c of r) yield c as [number, number];
      break;
    case "MultiPolygon":
      for (const p of g.coordinates) for (const r of p) for (const c of r) yield c as [number, number];
      break;
    case "GeometryCollection":
      for (const x of g.geometries) yield* geomCoords(x);
      break;
  }
}

/** metres per CSS pixel at a latitude and MapLibre zoom (512 px tiles) */
export function metresPerPixel(lat: number, zoom: number) {
  return (40075016.686 * Math.cos((lat * Math.PI) / 180)) / (512 * Math.pow(2, zoom));
}
