import { useEffect, useMemo } from "react";
import type maplibregl from "maplibre-gl";
import type { GeoJSONSource } from "maplibre-gl";
import { useTowerWhy } from "../../api/hooks";
import type { Route, RulesResponse, Tower, TowerWhy } from "../../api/types";
import { crore, num } from "../../lib/format";
import { turnEvent, typeReasons, type Limits } from "../../lib/design";
import { Spinner } from "../ui";

export interface WhyCtx {
  sid: number;
  route: Route;
  rules?: RulesResponse;
  limits: Limits | null;
}

const SRC = "why-straight";
const IDS = ["why-leg", "why-stop", "why-onl"];

/** The straight alternative on the map while the popup is open: the leg dashed, the stretches where no
 *  tower may stand (red when too long to span, amber when they can be spanned), the tower's spot on the line. */
function useStraightOverlay(map: maplibregl.Map | null, w: TowerWhy | undefined) {
  useEffect(() => {
    if (!map || !w || w.preparing || !w.straight) return;
    const s = w.straight;
    const fc: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: [
        { type: "Feature", properties: { k: "leg" }, geometry: { type: "LineString", coordinates: [s.from, s.to] } },
        ...s.no_tower_stretches.map((r) => ({ type: "Feature" as const, properties: { k: r.spannable ? "span" : "stop" }, geometry: { type: "LineString" as const, coordinates: [r.a, r.b] } })),
        { type: "Feature", properties: { k: "onl" }, geometry: { type: "Point", coordinates: w.on_line.lonlat } },
      ],
    };
    const src = map.getSource(SRC) as GeoJSONSource | undefined;
    if (src) src.setData(fc);
    else {
      map.addSource(SRC, { type: "geojson", data: fc });
      map.addLayer({ id: "why-leg", type: "line", source: SRC, filter: ["==", ["get", "k"], "leg"], paint: { "line-color": "#ffffff", "line-width": 2, "line-dasharray": [2, 2] } });
      map.addLayer({ id: "why-stop", type: "line", source: SRC, filter: ["in", ["get", "k"], ["literal", ["stop", "span"]]], layout: { "line-cap": "round" },
        paint: { "line-color": ["case", ["==", ["get", "k"], "stop"], "#D7263D", "#F0A202"], "line-width": 5, "line-opacity": 0.9 } });
      map.addLayer({ id: "why-onl", type: "circle", source: SRC, filter: ["==", ["get", "k"], "onl"], paint: { "circle-radius": 5, "circle-color": "#ffffff", "circle-stroke-color": "#14202A", "circle-stroke-width": 2 } });
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
  }, [map, w]);
}

/**
 * Why this tower (15 Oct 2026): its type, the turn the solver logged, and — for an angle tower — the
 * straight alternative tested with the solve's own compiled rules: what would break if the line ran
 * straight from the angle point before to the one after, and what that would have saved.
 */
export default function TowerWhyPanel({ t, ctx, map }: { t: Tower; ctx: WhyCtx; map: maplibregl.Map | null }) {
  const isAngle = t.type !== "G" && (t.angle_point || t.deflection_deg > 0.05);
  const q = useTowerWhy(ctx.sid, ctx.route.id, t.no, isAngle);
  const w = q.data && !q.data.preparing ? q.data : undefined;
  useStraightOverlay(map, w);
  const reasons = useMemo(() => (ctx.limits ? typeReasons(t, ctx.limits, ctx.rules) : []), [t, ctx.limits, ctx.rules]);
  const ev = useMemo(() => turnEvent(ctx.route, t.no), [ctx.route, t.no]);
  if (t.type === "G") return null;
  const s = w?.straight;
  const longStops = s?.no_tower_stretches.filter((r) => !r.spannable) ?? [];
  const shortStops = s?.no_tower_stretches.filter((r) => r.spannable) ?? [];
  const badX = s?.crossings.filter((c) => !c.angle_ok) ?? [];

  return (
    <div className="why" data-testid="tower-why">
      <div className="why-h">Why this type</div>
      <ul>{reasons.map((r) => <li key={r}>{r}</li>)}</ul>

      {isAngle ? (
        <>
          <div className="why-h">Why the line turns here</div>
          {ev ? (
            <p>{ev.text}<span className="why-note"> Logged after the solve by looking straight ahead from the tower; the test below checks it.</span></p>
          ) : (
            <p className="why-note">The solver logged no feature for this turn.</p>
          )}

          <div className="why-h">Straight alternative</div>
          {q.isPending || q.data?.preparing ? (
            <p className="why-note"><Spinner /> Loading this solve's rule checker. The first tower takes about three minutes; after that each one is instant.</p>
          ) : q.isError ? (
            <p className="why-note">The straight-line test is not available: {String((q.error as Error)?.message ?? q.error)}</p>
          ) : s && w ? (
            <>
              <p>
                Straight from tower {s.from_tower} to tower {s.to_tower}: <b>{num(s.length_m, 0)} m</b> instead of {num(s.bend_length_m, 0)} m.{" "}
                {s.allowed ? <span className="ok"><b>The rules allow it.</b></span> : <span className="bad"><b>The rules do not allow it:</b></span>}
              </p>
              {!s.allowed ? (
                <ul className="bad-list">
                  {s.wire_buffers.map((x) => (
                    <li key={x.id}>{x.buffer_m > 0 ? `the wire would pass within ${num(x.buffer_m, 1)} m of ${x.feature ?? "a feature"} (${x.name})` : `the wire would enter the ${x.name.toLowerCase()} no-go zone (${x.feature ?? "a feature"})`}</li>
                  ))}
                  {badX.map((c, i) => <li key={`x${i}`}>it would cross the {c.name.toLowerCase()} at {num(c.angle_deg, 0)}°, outside the rule's angle window</li>)}
                  {longStops.map((r, i) => (
                    <li key={`s${i}`}>{num(r.length_m, 0)} m where no tower may stand ({r.rules.map((x) => x.name).join(", ")}{r.feature ? `, ${r.feature}` : ""}), longer than a span can bridge</li>
                  ))}
                </ul>
              ) : null}
              {shortStops.length ? (
                <p className="why-note">{shortStops.length} short no-tower stretch{shortStops.length > 1 ? "es" : ""} on the straight line could be spanned over ({[...new Set(shortStops.flatMap((r) => r.rules.map((x) => x.name)))].join(", ")}).</p>
              ) : null}
              <p className="why-note">
                Moving only this tower {num(w.on_line.shift_m, 0)} m onto the line through its neighbours{" "}
                {w.on_line.new_problems.length ? <>would add: {w.on_line.new_problems.join("; ")}.</> : "adds no new rule problem."}
              </p>
              <p>
                Straightening would save about <b>₹{crore(s.saving_rs.total, { unit: false, digits: 2 })} Cr</b>
                <span className="why-note"> (length ₹{crore(s.saving_rs.length, { unit: false, digits: 2 })} Cr, lighter tower type ₹{crore(s.saving_rs.tower_type, { unit: false, digits: 2 })} Cr, {s.saving_rs.towers_bend - s.saving_rs.towers_straight > 0 ? `${s.saving_rs.towers_bend - s.saving_rs.towers_straight} fewer tower ₹${crore(s.saving_rs.fewer_towers, { unit: false, digits: 2 })} Cr` : "same tower count"}). Land, forest, foundation and existing-line pricing are not in this estimate.</span>
              </p>
              {s.allowed ? (
                <p className="why-note">
                  The straight line is legal, so the bend is a cost decision: the solver judged the turn cheaper once land, forest, hill and existing-line pricing are counted, or its 25–50 m search grid missed the straight option. Test it with Move towers.
                </p>
              ) : null}
              <p className="why-note">On the map: the white dashed line is the straight alternative, red marks stretches no span can bridge, amber marks short ones a span can cross, the white dot is this tower's spot on the line.</p>
            </>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
