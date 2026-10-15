import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useRunCtx } from "../App";
import { useCompare, useLayers, useRules, useSolve } from "../api/hooks";
import type { RunDetail, Solve } from "../api/types";
import CopyDialog from "../components/CopyDialog";
import TowerLayer from "../components/map/TowerLayer";
import CrossingLayer from "../components/map/CrossingLayer";
import BendLayer, { BEND_COLOUR } from "../components/map/BendLayer";
import DesignCheck from "../components/DesignCheck";
import { designOf, STATUS_COLOUR } from "../lib/design";
import {
  bboxOf,
  DEFAULT_ON,
  geomCoords,
  LAYER_COLOURS,
  LAYER_ORDER,
  layerIds,
  NOGO_LAYERS,
  routeColour,
  setLowConfVisible,
  setRunLayerOpacity,
  setRunLayerVisibility,
} from "../components/map/style";
import { useFitBounds, useMapLibre, useRunTiles, type Bounds } from "../components/map/useMap";
import TerrainControl from "../components/map/TerrainControl";
import RouteProfile from "../components/RouteProfile";
import TowerMoveTool from "../components/map/TowerMoveTool";
import MapTools, { mapToolState } from "../components/map/MapTools";
import { CeilingNote, ScaleBar, useCeiling, ZoomBar } from "../components/map/ZoomBar";
import { ErrorNote, Head, Kpi, Loading, Note } from "../components/ui";
import { crore, int, num } from "../lib/format";
import { endpoints, linesGeoJSON, routesGeoJSON, tradeOff } from "../lib/solve";
import { useTowers } from "../store/towers";
import { useUi } from "../store/ui";
import { useSelectedRoute } from "../store/selection";
import { useStartSolve } from "./Solve";

type Toggles = { towers: boolean; angles: boolean; spans: boolean; crossings: boolean; bends: boolean; beeline: boolean; lowconf: boolean; prev: boolean; hills: boolean };

export default function Results() {
  const run = useRunCtx();
  const nav = useNavigate();
  const q = useSolve(run.id, run.latest_solve_id);
  const start = useStartSolve(run.id);
  if (!run.latest_solve_id) {
    return (
      <section>
        <Head title="Route map">This run has not been solved yet.</Head>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/layers`)}>Review the layers</button>
          <button className="btn" disabled={!run.layers_ready || Boolean(run.job) || start.isPending} onClick={() => start.mutate()}>Solve the route</button>
        </div>
      </section>
    );
  }
  if (q.isPending) return <Loading text="Loading the routes…" />;
  if (q.isError) return <ErrorNote error={q.error} retry={() => void q.refetch()} />;
  return <Body s={q.data} />;
}

function Body({ s }: { s: Solve }) {
  const run = useRunCtx();
  const nav = useNavigate();
  const layersQ = useLayers(run.id);
  const cmp = useCompare(run.id, run.latest_solve_id, Boolean(run.parent_id)).data;
  const [selId, select] = useSelectedRoute(run, s);
  const loadTowers = useTowers((x) => x.load);
  const focus = useUi((x) => x.focus);
  const setFocus = useUi((x) => x.setFocus);
  const [t, setT] = useState<Toggles>({ towers: true, angles: false, spans: false, crossings: true, bends: false, beeline: false, lowconf: false, prev: false, hills: false });
  // every run layer can be shown under the routes (6 Oct 2026); starts as on Review the layers
  const [on, setOn] = useState<Record<string, boolean>>(() => Object.fromEntries(LAYER_ORDER.map((k) => [k, DEFAULT_ON.has(k)])));
  const [widen, setWiden] = useState(false);
  const [maxed, setMaxed] = useState(false);
  const [moveMode, setMoveMode] = useState(false);
  // routes hidden on the map (8 Oct 2026), like the layer toggles
  const [hidden, setHidden] = useState<number[]>([]);
  const routes = s.result.routes;
  const feasible = s.result.feasible && routes.length > 0;
  const sel = routes.find((r) => r.id === selId) ?? routes[0];
  // design check of the selected route (15 Oct 2026): towers, sections and crossings against the rules
  const rulesQ = useRules(run.id);
  const design = useMemo(() => (sel ? designOf(sel, rulesQ.data, s.manifest, run.kv) : null), [sel, rulesQ.data, s.manifest, run.kv]);
  const flyTo = (lon: number, lat: number) => map?.flyTo({ center: [lon, lat], zoom: Math.max(map.getZoom(), 15.5), duration: 700 });

  useEffect(() => {
    loadTowers(s.id, routes);
  }, [s.id, routes, loadTowers]);

  const ends = useMemo(() => endpoints(run), [run.a_lon, run.a_lat, run.b_lon, run.b_lat]); // eslint-disable-line react-hooks/exhaustive-deps
  const hwM = s.result.line?.half_width_m ?? 0;
  const midLat = (ends.a[1] + ends.b[1]) / 2;
  const routesFC = useMemo(() => routesGeoJSON(routes), [routes]);
  const prevFC = useMemo(() => linesGeoJSON(cmp?.parent_routes ?? []), [cmp]);
  const band = layersQ.data?.band;
  const bounds = useMemo<Bounds | null>(() => {
    const pts: [number, number][] = [ends.a, ends.b];
    for (const r of routes) for (let i = 0; i < r.geometry.length; i += 5) pts.push(r.geometry[i]);
    if (band) for (const c of geomCoords(band)) pts.push(c);
    return bboxOf(pts);
  }, [routes, ends, band]);

  const mapRef = useRef<HTMLDivElement>(null);
  const map = useMapLibre(mapRef, { bounds });
  useFitBounds(map, bounds);
  const ceiling = useCeiling(map);
  const wantTiles = t.lowconf || Object.values(on).some(Boolean);
  const applyTiles = (m: maplibregl.Map) => {
    setRunLayerVisibility(m, (k) => Boolean(on[k]));
    setRunLayerOpacity(m, 0.55);
    setLowConfVisible(m, t.lowconf);
  };
  const applyRef = useRef(applyTiles);
  applyRef.current = applyTiles;
  const tiles = useRunTiles(map, run.id, wantTiles, (m) => applyRef.current(m));
  useEffect(() => {
    if (map && map.getSource("run")) applyRef.current(map);
  }, [map, on, t.lowconf]);

  // static overlays: band, bee-line, previous revision, routes, end points
  useEffect(() => {
    if (!map) return;
    const add = (id: string, data: GeoJSON.GeoJSON) => {
      const src = map.getSource(id) as GeoJSONSource | undefined;
      if (src) src.setData(data);
      else map.addSource(id, { type: "geojson", data });
    };
    if (band) add("band", { type: "Feature", properties: {}, geometry: band });
    add("beeline", { type: "Feature", properties: {}, geometry: { type: "LineString", coordinates: [ends.a, ends.b] } });
    add("prev", prevFC);
    add("routes", routesFC);
    if (band && !map.getLayer("band-line"))
      map.addLayer({ id: "band-line", type: "line", source: "band", paint: { "line-color": "#ffffff", "line-width": 1.4, "line-dasharray": [4, 3], "line-opacity": 0.7 } });
    if (!map.getLayer("beeline-line"))
      map.addLayer({ id: "beeline-line", type: "line", source: "beeline", layout: { visibility: "none" }, paint: { "line-color": "#ffffff", "line-width": 1.6, "line-dasharray": [4, 3], "line-opacity": 0.85 } });
    if (!map.getLayer("prev-line"))
      map.addLayer({ id: "prev-line", type: "line", source: "prev", layout: { visibility: "none", "line-join": "round" }, paint: { "line-color": "#C8D0CC", "line-width": 2.6, "line-dasharray": [3, 2], "line-opacity": 0.9 } });
    if (!map.getLayer("row-strip") && hwM > 0) {
      // right of way along each route: a line whose pixel width tracks 2 x half-width in metres
      const mpp0 = 156543.03392 * Math.cos((midLat * Math.PI) / 180);
      const w0 = (2 * hwM) / mpp0;
      map.addLayer({
        id: "row-strip",
        type: "line",
        source: "routes",
        layout: { "line-join": "round", "line-cap": "butt" },
        paint: {
          "line-color": ["match", ["get", "id"], 1, routeColour(1), 2, routeColour(2), 3, routeColour(3), 4, routeColour(4), "#cccccc"],
          "line-width": ["interpolate", ["exponential", 2], ["zoom"], 0, w0, 22, w0 * 2 ** 22],
          "line-opacity": 0.14,
        },
      });
    }
    if (!map.getLayer("routes-casing")) {
      map.addLayer({
        id: "routes-casing",
        type: "line",
        source: "routes",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: { "line-color": "#0d1418", "line-width": 5, "line-opacity": 0.55 },
      });
      map.addLayer({
        id: "routes-line",
        type: "line",
        source: "routes",
        layout: { "line-join": "round", "line-cap": "round" },
        paint: {
          "line-color": ["match", ["get", "id"], 1, routeColour(1), 2, routeColour(2), 3, routeColour(3), 4, routeColour(4), "#cccccc"],
          "line-width": 2.6,
          "line-opacity": 0.6,
        },
      });
    }
  }, [map, band, ends, prevFC, routesFC, hwM, midLat]);

  // selection styling
  useEffect(() => {
    if (!map || !map.getLayer("routes-line")) return;
    if (map.getLayer("row-strip")) map.setPaintProperty("row-strip", "line-opacity", ["case", ["==", ["get", "id"], selId], 0.38, 0.12]);
    map.setPaintProperty("routes-line", "line-width", ["case", ["==", ["get", "id"], selId], 5, 2.4]);
    map.setPaintProperty("routes-line", "line-opacity", ["case", ["==", ["get", "id"], selId], 1, 0.55]);
    map.setPaintProperty("routes-casing", "line-width", ["case", ["==", ["get", "id"], selId], 8.5, 4.5]);
    map.setLayoutProperty("routes-line", "line-sort-key", ["case", ["==", ["get", "id"], selId], 10, 0]);
    map.setLayoutProperty("routes-casing", "line-sort-key", ["case", ["==", ["get", "id"], selId], 10, 0]);
  }, [map, selId, routesFC]);

  useEffect(() => {
    if (!map) return;
    const f: maplibregl.FilterSpecification = ["!", ["in", ["get", "id"], ["literal", hidden]]];
    for (const id of ["routes-line", "routes-casing", "row-strip"]) if (map.getLayer(id)) map.setFilter(id, f);
  }, [map, hidden, routesFC]);

  useEffect(() => {
    if (!map) return;
    if (map.getLayer("beeline-line")) map.setLayoutProperty("beeline-line", "visibility", t.beeline ? "visible" : "none");
    if (map.getLayer("prev-line")) map.setLayoutProperty("prev-line", "visibility", t.prev ? "visible" : "none");
  }, [map, t.beeline, t.prev, prevFC]);

  // hill ground the search avoids (14 Oct 2026): DEM relief + mean slope, from the solve
  const hillFC = (s.result as { hill_areas?: GeoJSON.FeatureCollection | null }).hill_areas ?? null;
  useEffect(() => {
    if (!map || !hillFC) return;
    const src = map.getSource("hills") as GeoJSONSource | undefined;
    if (src) src.setData(hillFC);
    else map.addSource("hills", { type: "geojson", data: hillFC });
    if (!map.getLayer("hills-fill")) {
      const before = map.getLayer("routes-casing") ? "routes-casing" : undefined;
      map.addLayer({ id: "hills-fill", type: "fill", source: "hills", layout: { visibility: "none" }, paint: { "fill-color": "#C08A4B", "fill-opacity": 0.35 } }, before);
      map.addLayer({ id: "hills-line", type: "line", source: "hills", layout: { visibility: "none" }, paint: { "line-color": "#C08A4B", "line-width": 1.2 } }, before);
    }
    for (const id of ["hills-fill", "hills-line"]) map.setLayoutProperty(id, "visibility", t.hills ? "visible" : "none");
  }, [map, hillFC, t.hills]);

  // A / B markers like the prototype: a ring at A, a square at B
  useEffect(() => {
    if (!map) return;
    const mk = (cls: string, title: string) => {
      const el = document.createElement("div");
      el.title = title;
      el.style.cssText =
        cls === "a"
          ? "width:16px;height:16px;border-radius:50%;background:#fff;border:3px solid #14202A;box-shadow:0 0 0 2px rgba(255,255,255,.6)"
          : "width:15px;height:15px;background:#fff;border:3px solid #14202A;box-shadow:0 0 0 2px rgba(255,255,255,.6)";
      return el;
    };
    const a = new maplibregl.Marker({ element: mk("a", run.a_name ? `A · ${run.a_name}` : "A") }).setLngLat(ends.a).addTo(map);
    const b = new maplibregl.Marker({ element: mk("b", run.b_name ? `B · ${run.b_name}` : "B") }).setLngLat(ends.b).addTo(map);
    return () => {
      a.remove();
      b.remove();
    };
  }, [map, ends, run.a_name, run.b_name]);

  // click a route to select it
  useEffect(() => {
    if (!map) return;
    const onClick = (e: maplibregl.MapLayerMouseEvent) => {
      if (e.defaultPrevented || mapToolState.busy) return;
      const id = Number(e.features?.[0]?.properties?.id);
      if (Number.isFinite(id)) select(id);
    };
    const enter = () => (map.getCanvas().style.cursor = "pointer");
    const leave = () => (map.getCanvas().style.cursor = "");
    map.on("click", "routes-casing", onClick);
    map.on("mouseenter", "routes-casing", enter);
    map.on("mouseleave", "routes-casing", leave);
    return () => {
      map.off("click", "routes-casing", onClick);
      map.off("mouseenter", "routes-casing", enter);
      map.off("mouseleave", "routes-casing", leave);
    };
  }, [map, select]);

  // maximise the map; Escape brings the page back
  useEffect(() => {
    if (!maxed) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setMaxed(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [maxed]);
  useEffect(() => {
    if (!map) return;
    const id = requestAnimationFrame(() => map.resize());
    return () => cancelAnimationFrame(id);
  }, [map, maxed]);

  // hover: what is under the pointer on the layers shown
  const layerName = useMemo(() => Object.fromEntries((layersQ.data?.layers ?? []).map((l) => [l.key, l.name])), [layersQ.data]);
  useEffect(() => {
    if (!map) return;
    const popup = new maplibregl.Popup({ closeButton: false, closeOnClick: false, offset: 12, className: "hovertip", maxWidth: "280px" });
    let raf = 0;
    const onMove = (e: maplibregl.MapMouseEvent) => {
      cancelAnimationFrame(raf);
      if (mapToolState.busy) {
        popup.remove();
        return;
      }
      raf = requestAnimationFrame(() => {
        const ids = LAYER_ORDER.flatMap(layerIds).filter((i) => map.getLayer(i) && map.getLayoutProperty(i, "visibility") !== "none");
        const hit = ids.length ? map.queryRenderedFeatures(e.point, { layers: ids }) : [];
        if (!hit.length) {
          popup.remove();
          return;
        }
        // the topmost feature of each layer under the pointer, at most three layers
        const seen = new Set<string>();
        const rows: string[] = [];
        for (const f of hit) {
          const key = f.sourceLayer ?? "";
          if (seen.has(key) || seen.size >= 3) continue;
          seen.add(key);
          const pr = (f.properties ?? {}) as Record<string, unknown>;
          const cls = typeof pr.class === "string" ? pr.class.replace(/_/g, " ") : "";
          const bits = [
            pr.name ? String(pr.name) : null,
            cls && cls !== String(pr.name ?? "").toLowerCase() ? cls : null,
            pr.voltage_kv ? `${pr.voltage_kv} kV` : null,
            typeof pr.confidence === "number" ? `confidence ${(pr.confidence as number).toFixed(2)}` : null,
          ].filter(Boolean);
          rows.push(`<div class="ht-row"><b>${esc(layerName[key] ?? key)}</b>${bits.length ? `<span>${esc(bits.join(" · "))}</span>` : ""}</div>`);
        }
        popup.setLngLat(e.lngLat).setHTML(rows.join("")).addTo(map);
      });
    };
    const onOut = () => {
      cancelAnimationFrame(raf);
      popup.remove();
    };
    map.on("mousemove", onMove);
    map.getCanvas().addEventListener("mouseleave", onOut);
    return () => {
      cancelAnimationFrame(raf);
      map.off("mousemove", onMove);
      map.getCanvas().removeEventListener("mouseleave", onOut);
      popup.remove();
    };
  }, [map, layerName]);

  // fly to a spot requested from "Why this route"
  useEffect(() => {
    if (!map || !focus || focus.runId !== run.id) return;
    map.flyTo({ center: [focus.lon, focus.lat], zoom: focus.zoom, duration: 800 });
    setFocus(null);
  }, [map, focus, run.id, setFocus]);

  const lowConf = (s.manifest.sources ?? []).filter((x) => (x.confidence ?? 1) < 0.8 && x.features > 0);
  const srcCounts = { auto: 0, replace: 0, add: 0 };
  for (const x of s.manifest.sources ?? []) srcCounts[x.mode] = (srcCounts[x.mode] ?? 0) + 1;
  const withFeat = (s.manifest.sources ?? []).filter((x) => x.features > 0 && x.confidence !== null);
  const meanConf = withFeat.length ? withFeat.reduce((a, x) => a + (x.confidence ?? 0), 0) / withFeat.length : null;
  const tog = (k: keyof Toggles) => setT((x) => ({ ...x, [k]: !x[k] }));
  const nWord = ["No", "One", "Two", "Three", "Four"][routes.length] ?? String(routes.length);

  return (
    <section>
      <Head title={feasible ? `${nWord} route${routes.length === 1 ? "" : "s"}` : "No viable route"}>
        {feasible
          ? "Each optimises something different. Compare them on the table below the map, pick one, or change a constraint and solve again."
          : "The solve could not connect A to B inside the band with the rules as they stand."}
      </Head>

      {!feasible ? <Failure s={s} onWiden={() => setWiden(true)} /> : null}

      {feasible && lowConf.length ? (
        <Note kind="warn" style={{ marginBottom: 12 }}>
          <b>
            {lowConf.length} layer{lowConf.length > 1 ? "s" : ""} extracted below 0.80 confidence
          </b>{" "}
          — {lowConf.map((x) => `${x.layer} ${x.confidence?.toFixed(2)}`).join(", ")}. The route treats their features as certain; the
          report does not. Full breakdown on{" "}
          <button className="link" style={{ padding: 0 }} onClick={() => nav(`/runs/${run.id}/why`)}><b>Why this route</b></button>.
        </Note>
      ) : null}

      <div className="withnotes" style={{ gridTemplateColumns: "minmax(0,1fr) 288px" }}>
        <div>
          <div className="maptools">
            <button aria-pressed={t.towers} onClick={() => tog("towers")} disabled={!feasible}>Towers</button>
            <button aria-pressed={t.angles} onClick={() => tog("angles")} disabled={!feasible || !t.towers} data-testid="toggle-angles"
              title="Deflection on each angle tower's label (from z14)">Angles</button>
            <button aria-pressed={t.spans} onClick={() => tog("spans")} disabled={!feasible || !t.towers} data-testid="toggle-spans"
              title="Span length at the midpoint of every span (from z14.5)">Spans</button>
            <button aria-pressed={t.crossings} onClick={() => tog("crossings")} disabled={!feasible} data-testid="toggle-crossings"
              title="Where the route crosses a feature, coloured by how the crossing stands against its rule; click one for the metrics">Crossings</button>
            <button aria-pressed={t.bends} onClick={() => tog("bends")} disabled={!feasible} data-testid="toggle-bends"
              title="A tie from each bend to the feature it reacts to, with the distance and the rule's setback (from z13)">Why bends</button>
            <button aria-pressed={t.beeline} onClick={() => tog("beeline")}>Bee-line</button>
            <button aria-pressed={t.lowconf} onClick={() => tog("lowconf")}>Low-confidence</button>
            <button aria-pressed={t.hills} onClick={() => tog("hills")} disabled={!hillFC}
              title={hillFC ? "Hill ground the search avoids (relief + slope from the DEM)" : "This solve has no hill layer"}>Hills</button>
            <button aria-pressed={t.prev} onClick={() => tog("prev")} disabled={!cmp?.parent_routes?.length}
              title={cmp?.parent_routes?.length ? undefined : "No solved previous revision"}>
              Previous revision
            </button>
            <ZoomBar map={map} ceiling={ceiling} />
            <button aria-pressed={maxed} onClick={() => setMaxed((x) => !x)} data-testid="map-maximise" title="Fill the screen with the map (Esc to return)">
              Maximise map
            </button>
            <button aria-pressed={moveMode} onClick={() => setMoveMode((x) => !x)} disabled={!feasible || !sel} data-testid="move-towers"
              title="Drag the selected route's towers; each position is checked against the rules and saved as a new version">
              Move towers
            </button>
          </div>
          <div className={`mapcanvas${maxed ? " maxed" : ""}`} data-testid="route-map">
            <div ref={mapRef} className="mlmap" />
            <MapTools map={map} runId={run.id} onLayersChanged={tiles.retry} />
            <TerrainControl map={map} runId={run.id} />
            {feasible && sel ? <TowerMoveTool map={map} sid={s.id} runId={run.id} route={sel} active={moveMode} onExit={() => setMoveMode(false)} /> : null}
            {maxed ? (
              <button className="mapexit" onClick={() => setMaxed(false)} data-testid="map-restore">Exit full screen · Esc</button>
            ) : null}
            {!map ? <div className="mapnote"><Loading text="Loading imagery…" /></div> : null}
          </div>
          {(t.towers || moveMode) && feasible && !hidden.includes(sel.id) ? <TowerLayer map={map} routes={[sel.id]} editable={moveMode} angles={t.angles} spans={t.spans} /> : null}
          {feasible && design ? <CrossingLayer map={map} rows={design.crossings} visible={t.crossings && !hidden.includes(sel.id)} /> : null}
          {feasible && sel ? <BendLayer map={map} runId={run.id} sid={s.id} route={sel} rules={rulesQ.data} visible={t.bends && !hidden.includes(sel.id)} /> : null}
          <ScaleBar map={map} />
          <CeilingNote ceiling={ceiling} />
          {feasible && sel ? <RouteProfile sid={s.id} route={sel} map={map} enabled={!maxed} limits={design?.limits} /> : null}
          {tiles.error && wantTiles ? (
            <div className="note n-stop" style={{ marginTop: 8 }} role="alert" data-testid="tile-error">
              <div style={{ flex: 1 }}>
                <b>{tiles.error}</b> Routes, towers and the bee-line are drawn from the solve and are unaffected; the layer and
                low-confidence overlays are unavailable until the tile server is fixed.
              </div>
              <button className="mini" onClick={tiles.retry}>Try again</button>
            </div>
          ) : null}
          <div className="legend">
            {routes.map((r) => (
              <div key={r.id}><i style={{ background: routeColour(r.id) }} />Route {r.id}</div>
            ))}
            {hwM > 0 ? <div><i style={{ background: routeColour(sel?.id ?? 1), opacity: 0.4, height: 8 }} />Right of way ±{num(hwM, 1)} m</div> : null}
            <div><i style={{ background: "#D69A4E", borderRadius: "50%", width: 8, height: 8 }} />Angle tower</div>
            <div><i style={{ background: "#fff", border: "1px solid #14202A", borderRadius: "50%", width: 8, height: 8 }} />Suspension tower</div>
            {t.bends ? (
              <div>
                <i style={{ background: BEND_COLOUR.avoid }} />
                <i style={{ background: BEND_COLOUR.cross, marginLeft: -2 }} />
                <i style={{ background: BEND_COLOUR.clear, marginLeft: -2 }} />Bend: avoids / crosses / keeps clear
              </div>
            ) : null}
            {t.crossings ? (
              <div>
                <i style={{ background: STATUS_COLOUR.ok, borderRadius: "50%", width: 8, height: 8 }} />
                <i style={{ background: STATUS_COLOUR.relaxed, borderRadius: "50%", width: 8, height: 8, marginLeft: -2 }} />
                <i style={{ background: STATUS_COLOUR.violation, borderRadius: "50%", width: 8, height: 8, marginLeft: -2 }} />Crossing: preferred / relaxed / violation
              </div>
            ) : null}
            <div><i style={{ background: "repeating-linear-gradient(90deg,var(--ink-2) 0 4px,transparent 4px 7px)" }} />Bee-line / band</div>
          </div>
          <div className="row" style={{ marginTop: 8, gap: 8 }}>
            <span className="hint" style={{ margin: 0, flex: 1 }}>
              Click a route to select it, a tower or a crossing for its details. Tower labels appear from z14, deflections and span lengths a little closer in.
            </span>
            <button className="mini" onClick={() => nav(`/runs/${run.id}/layers`)}>Inspect layers at 0.5 m</button>
          </div>
        </div>
        <div>
          {feasible ? (
            <>
              <div className="eyebrow" style={{ marginBottom: 8 }}>Routes</div>
              <div data-testid="route-cards">
                {routes.map((r) => {
                  const shown = !hidden.includes(r.id);
                  return (
                    <div key={r.id} className={`routecard ${shown ? "" : "loff"}`} data-selected={r.id === sel.id}>
                      <button className="rsel" aria-pressed={r.id === sel.id} onClick={() => select(r.id)}>
                        <span className="swatch" style={{ background: routeColour(r.id) }} />
                        <span>
                          <span className="nm">Route {r.id} — {r.name}</span>
                          <span className="sub">
                            {num(r.stats.length_km, 1)} km · {int(r.stats.towers)} towers · {num(r.stats.forest_ha, 1)} ha forest
                            {r.stats.approval_months ? ` · approvals ~${r.stats.approval_months} months` : ""}
                          </span>
                        </span>
                        <span className="cost">{crore(r.cost.total, { unit: false, digits: 1 })}<span>₹ Cr</span></span>
                      </button>
                      <button className={`eye ${shown ? "" : "off"}`} aria-pressed={shown} data-testid={`show-route-${r.id}`}
                        aria-label={shown ? `Hide route ${r.id} on the map` : `Show route ${r.id} on the map`}
                        title={shown ? "Hide on the map" : "Show on the map"}
                        onClick={() => setHidden((h) => (h.includes(r.id) ? h.filter((x) => x !== r.id) : [...h, r.id]))}>
                        <EyeIcon off={!shown} />
                      </button>
                    </div>
                  );
                })}
              </div>
            </>
          ) : null}
          <RouteLayers layers={layersQ.data?.layers ?? []} on={on} setOn={setOn} />
          <div className="card" style={{ marginTop: 10 }}>
            <div className="cb stack">
              <div className="eyebrow">Where the data came from</div>
              <div className="tw">
                <table>
                  <tbody>
                    <tr><td>Auto-fetched</td><td className="mono" style={{ textAlign: "right" }}>{srcCounts.auto} layers</td></tr>
                    <tr><td>Your files</td><td className="mono" style={{ textAlign: "right" }}>{srcCounts.replace} layers</td></tr>
                    <tr><td>Supplemented</td><td className="mono" style={{ textAlign: "right" }}>{srcCounts.add} layers</td></tr>
                    <tr><td>Mean confidence</td><td className="mono" style={{ textAlign: "right" }}>{meanConf === null ? "—" : meanConf.toFixed(2)}</td></tr>
                  </tbody>
                </table>
              </div>
              <div className="hint">Solve {s.id} · {s.manifest.ruleset_version} · {s.manifest.costs_version}</div>
            </div>
          </div>
        </div>
      </div>

      {feasible ? (
        <>
          <div className="kpis" style={{ marginTop: 14 }} data-testid="kpis">
            <Kpi k="Length" v={num(sel.stats.length_km, 1)} unit="km" />
            <Kpi k="Towers" v={int(sel.stats.towers)} />
            <Kpi k="Angle points" v={int(sel.stats.angle_points)} />
            <Kpi k="Crossings" v={int(sel.stats.crossings_total)} unit={sel.stats.crossings_relaxed ? `${int(sel.stats.crossings_relaxed)} relaxed` : undefined} />
            <Kpi k="Forest" v={num(sel.stats.forest_ha, 1)} unit="ha" />
            <Kpi k="Cost" v={crore(sel.cost.total, { unit: false, digits: 1 })} unit="₹Cr" />
          </div>
          {design ? <DesignCheck route={sel} d={design} onFly={flyTo} /> : null}
          <CompareTable s={s} selId={sel.id} select={select} />
          <RevisionCard run={run} cmp={cmp} selId={sel.id} prevOn={t.prev} togglePrev={() => tog("prev")} />
          <div className="actions">
            <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/why`)}>Why this route?</button>
            <button className="btn ghost" onClick={() => nav(`/runs/${run.id}/rules`)}>Change a constraint and re-solve</button>
            <button className="btn" onClick={() => nav(`/runs/${run.id}/export`)}>Export</button>
          </div>
        </>
      ) : null}

      <CopyDialog
        open={widen}
        onClose={() => setWiden(false)}
        run={run}
        then="corridor"
        defaultNote={`Widen the band beyond ± ${num(run.band_km, 0)} km`}
        extra={
          <Note kind="warn" style={{ marginBottom: 12 }}>
            <b>Widening the band means extracting the layers again</b> over the new ground. In this version a copy keeps the layers, so its
            corridor stays locked — the Corridor screen will show the server's refusal if you try. To widen now, start a <b>New route</b> with
            the same end points and a wider band.
          </Note>
        }
      />
    </section>
  );
}

function CompareTable({ s, selId, select }: { s: Solve; selId: number; select: (n: number) => void }) {
  const routes = s.result.routes;
  const cols = useMemo(() => {
    const v = (f: (r: (typeof routes)[number]) => number) => Math.min(...routes.map(f));
    return {
      len: v((r) => r.stats.length_km),
      vs: v((r) => r.stats.vs_beeline_pct),
      ang: v((r) => r.stats.angle_points),
      forest: v((r) => r.stats.forest_ha),
      agri: v((r) => r.stats.agri_ha),
      cross: v((r) => r.stats.crossings_total),
      cost: v((r) => r.cost.total),
    };
  }, [routes]);
  const b = (x: number, best: number) => (Math.abs(x - best) < 1e-9 ? "best" : "");
  const n = routes.length;
  const note = tradeOff(routes);
  return (
    <div className="card" style={{ marginTop: 14 }}>
      <div className="ch">
        <h3>Compare the {n === 4 ? "four" : n === 3 ? "three" : n === 2 ? "two" : n}</h3>
        <span className="hint" style={{ marginLeft: "auto" }}>best value in each column marked · click a row to select</span>
      </div>
      <div className="tw">
        <table className="cmp" data-testid="compare-table">
          <thead>
            <tr>
              <th>Route</th>
              <th>Optimises</th>
              <th style={{ textAlign: "right" }}>Length</th>
              <th style={{ textAlign: "right" }}>vs bee-line</th>
              <th style={{ textAlign: "right" }}>Angle twr</th>
              <th style={{ textAlign: "right" }}>Susp. twr</th>
              <th style={{ textAlign: "right" }}>Forest</th>
              <th style={{ textAlign: "right" }}>Agri</th>
              <th style={{ textAlign: "right" }}>Crossings</th>
              <th style={{ textAlign: "right" }}>Total cost</th>
            </tr>
          </thead>
          <tbody>
            {routes.map((r) => (
              <tr key={r.id} className={r.id === selId ? "selrow" : ""} style={{ cursor: "pointer" }} onClick={() => select(r.id)}
                tabIndex={0} onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && select(r.id)} aria-selected={r.id === selId}>
                <td>
                  <span className="swatch" style={{ background: routeColour(r.id), display: "inline-block", verticalAlign: "middle", marginRight: 7 }} />
                  <b>Route {r.id}</b>
                </td>
                <td style={{ fontSize: 12.5 }}>{r.name}</td>
                <td className={`mono ${b(r.stats.length_km, cols.len)}`} style={{ textAlign: "right" }}>{num(r.stats.length_km, 1)} km</td>
                <td className={`mono ${b(r.stats.vs_beeline_pct, cols.vs)}`} style={{ textAlign: "right" }}>+{num(r.stats.vs_beeline_pct, 1)}%</td>
                <td className={`mono ${b(r.stats.angle_points, cols.ang)}`} style={{ textAlign: "right" }}>{int(r.stats.angle_points)}</td>
                <td className="mono" style={{ textAlign: "right" }}>{int(r.stats.suspension_towers)}</td>
                <td className={`mono ${b(r.stats.forest_ha, cols.forest)}`} style={{ textAlign: "right" }}>{num(r.stats.forest_ha, 1)} ha</td>
                <td className={`mono ${b(r.stats.agri_ha, cols.agri)}`} style={{ textAlign: "right" }}>{int(r.stats.agri_ha)} ha</td>
                <td className={`mono ${b(r.stats.crossings_total, cols.cross)}`} style={{ textAlign: "right" }}>{int(r.stats.crossings_total)}</td>
                <td className={`mono ${b(r.cost.total, cols.cost)}`} style={{ textAlign: "right" }}>{crore(r.cost.total)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="cb" style={{ borderTop: "1px solid var(--rule-2)" }}>
        <Note kind="info">
          Nothing here is a recommendation. {note} That trade-off belongs to the business, which is why the table sits next to the map
          rather than behind it.
        </Note>
      </div>
    </div>
  );
}

function RevisionCard({
  run,
  cmp,
  selId,
  prevOn,
  togglePrev,
}: {
  run: RunDetail;
  cmp: ReturnType<typeof useCompare>["data"];
  selId: number;
  prevOn: boolean;
  togglePrev: () => void;
}) {
  if (!run.parent_id || !cmp || !cmp.parent) return null;
  const rows = cmp.rows?.find((r) => r.route === selId) ?? cmp.rows?.[0];
  const fmt = (v: number | null, unit: string) => {
    if (v === null || v === undefined) return "—";
    if (unit === "₹ Cr") return `₹${num(v, 2)} Cr`;
    if (unit === "km") return `${num(v, 2)} km`;
    if (unit === "ha") return `${num(v, 1)} ha`;
    return int(v);
  };
  const fmtD = (d: number | null, unit: string) => {
    if (d === null || d === undefined) return "—";
    const sgn = d > 0 ? "+" : d < 0 ? "−" : "±";
    const a = Math.abs(d);
    if (unit === "₹ Cr") return `${sgn}₹${num(a, 2)} Cr`;
    if (unit === "km") return `${sgn}${num(a, 2)} km`;
    if (unit === "ha") return `${sgn}${num(a, 1)} ha`;
    return `${sgn}${int(a)}`;
  };
  return (
    <div className="card" style={{ marginTop: 14, borderColor: "var(--accent-line)" }} data-testid="revision-card">
      <div className="ch">
        <h3>What changed since the last revision</h3>
        <span className="pill p-chg" style={{ marginLeft: "auto" }}>rev {cmp.parent_rev ?? "?"} → rev {cmp.rev ?? run.rev}</span>
      </div>
      <div className="cb stack">
        {rows ? (
          <div className="tw">
            <table>
              <thead>
                <tr>
                  <th>Route {rows.route} — {rows.name}</th>
                  <th style={{ textAlign: "right" }}>rev {cmp.parent_rev}</th>
                  <th style={{ textAlign: "right" }}>rev {cmp.rev}</th>
                  <th style={{ textAlign: "right" }}>Change</th>
                </tr>
              </thead>
              <tbody>
                {rows.rows.map((r) => (
                  <tr key={r.label}>
                    <td>{r.label}</td>
                    <td className="mono" style={{ textAlign: "right" }}>{fmt(r.prev, r.unit)}</td>
                    <td className="mono" style={{ textAlign: "right" }}>{fmt(r.now, r.unit)}</td>
                    <td className="mono" style={{ textAlign: "right", color: r.delta === null || r.delta === 0 ? undefined : r.delta < 0 ? "var(--go)" : "var(--nogo)" }}>
                      {fmtD(r.delta, r.unit)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="hint">{cmp.parent} has not been solved, so there is nothing to compare numbers against yet.</p>
        )}
        <Note kind="ok">
          <div>
            <b>What changed in the inputs.</b>{" "}
            {cmp.changes?.length ? (
              <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
                {cmp.changes.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            ) : (
              "Nothing — the inputs match the previous revision."
            )}
          </div>
        </Note>
        <div className="row" style={{ gap: 8 }}>
          <button className="mini" aria-pressed={prevOn} onClick={togglePrev} disabled={!cmp.parent_routes?.length}>
            {prevOn ? "Hide" : "Show"} rev {cmp.parent_rev} on the map
          </button>
        </div>
      </div>
    </div>
  );
}

function Failure({ s, onWiden }: { s: Solve; onWiden: () => void }) {
  const run = useRunCtx();
  const nav = useNavigate();
  const bl = s.result.blocked;
  return (
    <div data-testid="no-route">
      <Note kind="stop" style={{ marginBottom: 14 }}>
        <b>No route could be found inside the ± {num(run.band_km, 0)} km band.</b> {bl?.event?.text ?? ""}{" "}
        {bl ? `The search reached km ${num(bl.reach_km, 1)} before the band closed.` : ""} The extraction is kept, so fixing this does not
        mean starting over.
      </Note>
      <div className="card">
        <div className="tw">
          <table data-testid="blockers">
            <thead>
              <tr>
                <th>Blocking constraint</th>
                <th>Type</th>
                <th>Where</th>
                <th style={{ textAlign: "right" }}>Fix</th>
              </tr>
            </thead>
            <tbody>
              {(bl?.blockers ?? []).map((b) => (
                <tr key={b.rule_id}>
                  <td>
                    {b.rule} — {b.feature}
                    <div className="hint">
                      spans {b.band_cover_pct}% of the band width
                      {b.source ? ` · ${b.source.layer_name}, ${b.source.mode}${b.source.confidence !== null ? ` ${b.source.confidence.toFixed(2)}` : ""}` : ""}
                    </div>
                  </td>
                  <td>{b.statutory ? <span className="pill p-lock">Statutory</span> : <span className="pill p-nogo">No-Go</span>}</td>
                  <td className="mono">km {num(b.from_km, 1)} – {num(b.to_km, 1)}</td>
                  <td style={{ textAlign: "right" }}>
                    <button className="mini" onClick={() => nav(`/runs/${run.id}/rules?q=${encodeURIComponent(b.rule)}`)}>
                      Relax rule{b.statutory ? " (flagged)" : ""}
                    </button>
                  </td>
                </tr>
              ))}
              <tr>
                <td>Band too narrow to route around it</td>
                <td><span className="pill p-cross">Band</span></td>
                <td className="mono">{bl?.blockers[0] ? `km ${num(bl.blockers[0].from_km, 1)} – ${num(bl.blockers[0].to_km, 1)}` : "—"}</td>
                <td style={{ textAlign: "right" }}>
                  <button className="mini" onClick={onWiden}>Widen the band</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <Note kind="info" style={{ marginTop: 14, marginBottom: 14 }}>
        Relaxing a rule re-uses the extraction and re-solves in minutes; widening the band is the only fix that runs the vision models
        again.
      </Note>
    </div>
  );
}



/** Eye / eye-off, the route's visibility toggle on its card (Vivek, 13 Oct 2026). */
function EyeIcon({ off }: { off: boolean }) {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6S2 12 2 12z" />
      <circle cx="12" cy="12" r="3" />
      {off ? <path d="M4 4l16 16" /> : null}
    </svg>
  );
}

/** Every layer of the run, toggled under the routes. Presets: as on Review the layers, No-Go only, all, none. */
function RouteLayers({ layers, on, setOn }: {
  layers: { key: string; name: string; extracted?: boolean; feature_count?: number | null }[];
  on: Record<string, boolean>;
  setOn: (f: (o: Record<string, boolean>) => Record<string, boolean>) => void;
}) {
  const ordered = [...layers].sort((a, b) => LAYER_ORDER.indexOf(a.key) - LAYER_ORDER.indexOf(b.key));
  const set = (pick: (k: string) => boolean) => setOn(() => Object.fromEntries(LAYER_ORDER.map((k) => [k, pick(k)])));
  const nOn = ordered.filter((l) => on[l.key]).length;
  return (
    <div className="card" style={{ marginTop: 10 }} data-testid="route-layers">
      <div className="ch">
        <h3 style={{ fontSize: 13 }}>Layers</h3>
        <span className="hint" style={{ marginLeft: "auto" }}>{nOn} of {ordered.length} shown</span>
      </div>
      <div className="row" style={{ gap: 4, padding: "6px 9px", flexWrap: "wrap" }}>
        <button className="mini" onClick={() => set((k) => DEFAULT_ON.has(k))}>Default</button>
        <button className="mini" onClick={() => set((k) => NOGO_LAYERS.includes(k))}>No-Go only</button>
        <button className="mini" onClick={() => set(() => true)}>All</button>
        <button className="mini" onClick={() => set(() => false)}>None</button>
      </div>
      <div style={{ maxHeight: 340, overflowY: "auto" }}>
        {ordered.map((l) => {
          const v = Boolean(on[l.key]);
          const empty = l.extracted && (l.feature_count ?? 0) === 0;
          return (
            <div key={l.key} className={`lrow ${v ? "" : "loff"}`}>
              <label className="ln">
                <input type="checkbox" checked={v} onChange={() => setOn((o) => ({ ...o, [l.key]: !o[l.key] }))} aria-label={`Show ${l.name}`} />
                <i className="sw" style={{ background: LAYER_COLOURS[l.key] ?? "#888" }} />
                <span>
                  <b>{l.name}</b>
                  <em>{!l.extracted ? "not extracted" : empty ? "nothing in the band" : `${(l.feature_count ?? 0).toLocaleString("en-IN")} features`}{NOGO_LAYERS.includes(l.key) ? " · No-Go" : ""}</em>
                </span>
              </label>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function esc(t: string): string {
  return t.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] ?? c);
}
