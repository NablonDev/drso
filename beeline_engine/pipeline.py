"""Solve orchestration: layers + rules + costs -> four routes, spotted, costed and explained."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features as rfeatures
from shapely import STRtree
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union

from . import dem as demmod
from .compile import attach_terrain, compile_rules
from .config import LineSpec, load_engine, load_layers, resolve_costs, resolve_rules
from .events import event
from .frame import Frame
from .solver import SolverParams, solve, tower_type_cost
from .spotting import SpotParams, spot

log = logging.getLogger(__name__)


POWER_LINE_RULES = ("power_line", "power_line_lt")          # high / low tension (6 Oct 2026)
CROSSING_RULES = {"railway", "national_highway", "river", "canal", *POWER_LINE_RULES, "pipeline_crossing"}


class Cancelled(Exception):
    pass


@dataclass
class LayerInput:
    key: str
    gdf: gpd.GeoDataFrame | None
    mode: str = "auto"                 # auto | replace | add
    source: str = ""
    confidence: float | None = None
    coverage: object | None = None     # shapely lon/lat polygon; None = whole band covered
    model: str | None = None


@dataclass
class SolveInput:
    a: tuple[float, float]             # lon, lat
    b: tuple[float, float]
    band_km: float
    kv: int
    circuit: str
    layers: dict[str, LayerInput]
    a_name: str = "A"
    b_name: str = "B"
    routes_wanted: int = 4
    rule_overrides: dict = field(default_factory=dict)
    cost_overrides: dict = field(default_factory=dict)
    dem_dir: Path | None = None


# ----------------------------------------------------------------------------------------
class FeatureIndex:
    """Raw features (frame coords) for naming the thing a decision was about."""

    def __init__(self, frame: Frame, layers: dict[str, LayerInput], layer_names: dict[str, str]):
        self.items = []
        geoms = []
        for key, li in layers.items():
            if li.gdf is None or not len(li.gdf):
                continue
            g = li.gdf.to_crs(frame.epsg)
            for geom, cls, name, conf in zip(g.geometry, g["class"], g.get("name", [None] * len(g)),
                                             g.get("confidence", [None] * len(g))):
                if geom is None or geom.is_empty:
                    continue
                fg = frame.geom_utm_to_frame(geom)
                label = name if isinstance(name, str) and name.strip() else cls.replace("_", " ")
                c = conf if conf is not None and not (isinstance(conf, float) and math.isnan(conf)) else li.confidence
                self.items.append({"layer": key, "layer_name": layer_names.get(key, key), "class": cls,
                                   "label": label, "confidence": None if c is None else float(c),
                                   "mode": li.mode, "source": li.source})
                geoms.append(fg)
        self.geoms = geoms
        self.tree = STRtree(geoms) if geoms else None

    def nearest(self, geom, layers: set[str] | None = None, classes: set[str] | None = None,
                max_dist: float = 5000.0):
        if self.tree is None:
            return None, None
        idx = self.tree.query(geom.buffer(max_dist))
        best, bd = None, None
        for i in idx:
            it = self.items[i]
            if layers and it["layer"] not in layers:
                continue
            if classes and it["class"] not in classes:
                continue
            d = self.geoms[i].distance(geom)
            if bd is None or d < bd:
                best, bd = i, d
        if best is None:
            return None, None
        it = dict(self.items[best])
        it["_idx"] = int(best)
        return it, bd


def _source(it: dict | None, layers: dict[str, LayerInput]) -> dict | None:
    if it is None:
        return None
    return {"layer": it["layer"], "layer_name": it["layer_name"], "mode": it["mode"],
            "confidence": it["confidence"], "source": it["source"], "feature": it["label"]}


# ----------------------------------------------------------------------------------------
def run_solve(inp: SolveInput, progress=None, cancel=None) -> dict:
    t_start = time.time()
    timings = {}

    def prog(frac, stage):
        if progress:
            progress(min(0.999, frac), stage)

    def check_cancel():
        if cancel and cancel():
            raise Cancelled()

    eng = load_engine()
    layer_reg = {l["key"]: l for l in load_layers()["layers"]}
    layer_names = {k: v["name"] for k, v in layer_reg.items()}
    spec = LineSpec(inp.kv, inp.circuit)
    rules = resolve_rules(spec, inp.rule_overrides)
    costs = resolve_costs(spec, inp.cost_overrides)
    hw = float(rules["corridor_half_width"].params["half_width_m"])
    design = float(rules["design_span"].params["plain_m"])
    # Shortest span = the rule's crossing minimum (falling-tower distance). Short spans are
    # allowed anywhere; their extra tower cost is what keeps them rare.
    smin = float(rules["crossing_span_limits"].params["min_span_m"])
    smax = design          # "Design span" is the hard maximum span (hilly value applied on hilly ground)
    bands = {k: float(rules["tower_type_bands"].params[k]) for k in ("A", "B1", "B2", "C1", "C2", "D45", "D60")}
    max_dev = float(rules["max_deviation"].params["limit_deg"])
    if not rules["max_deviation"].enabled:
        max_dev = 60.0
    go_radius = float(rules["substations"].params.get("gantry_go_radius_m", 150))

    frame = Frame(tuple(inp.a), tuple(inp.b), inp.band_km * 1000.0, eng["grid"]["band_margin_m"],
                  eng["grid"]["raster_res_m"])
    prog(0.02, "Compiling rules")
    gdfs = {k: li.gdf for k, li in inp.layers.items()}
    t = time.time()
    comp = compile_rules(frame, gdfs, rules, costs, inp.kv, hw, endpoints_go_radius_m=go_radius,
                         design_span_m=design,
                         single_angle_tol_deg=eng["crossings"]["single_angle_tolerance_deg"],
                         gantry_buffer_m=float(rules["substations"].params.get("gantry_buffer_m", 10)),
                         raster_margin_m=float(eng["grid"].get("raster_margin_m", 0.0)),
                         angle_tower_rules=tuple(eng["crossings"].get("angle_tower_rules") or ()))
    timings["compile_s"] = round(time.time() - t, 2)
    check_cancel()

    # DEM
    prog(0.12, "Fetching terrain")
    dem = None
    dem_note = None
    if inp.dem_dir is not None:
        try:
            bb = frame.band_polygon_lonlat(margin=True).bounds
            tiles = demmod.ensure_tiles(bb, Path(inp.dem_dir))
            dem = demmod.DemSampler(tiles, bb)
        except Exception as e:  # network or file trouble: solve without clearance checks
            log.warning("DEM unavailable: %s", e)
            dem_note = f"DEM unavailable ({type(e).__name__}); ground clearance not checked"
    check_cancel()

    tower_cost = tower_type_cost(costs, inp.kv)
    # terrain in the search: the DEM on the rule grid; costs from the same tables the spotting uses
    terrain = dem is not None and dem.ok
    if terrain:
        t = time.time()
        attach_terrain(comp, dem, float(eng["spotting"].get("hilly_slope_deg", 10)))
        timings["terrain_s"] = round(time.time() - t, 2)
    steel_a = float(costs["tonnage"]["A"]) * float(costs["tower"]["rate_per_tonne_rs"])
    plain_km = float(rules["section_interval"].params["plain_km"])
    hilly_km = float(rules["section_interval"].params["hilly_km"])
    base = SolverParams(site_spacing_m=eng["grid"]["site_spacing_m"], span_min_m=smin, span_max_m=smax,
                        max_dev_deg=max_dev, type_bands=bands, tower_cost_rs=tower_cost,
                        parallel_angle_deg=eng["crossings"]["parallel_angle_deg"],
                        river_max_span_m=float(eng["spans"].get("river_crossing_max_span_m", 0)),
                        long_heading_limit_deg=float(eng["spans"].get("long_heading_limit_deg", 60)),
                        long_heading_step_deg=float(eng["spans"].get("long_heading_step_deg", 5)),
                        long_length_step_m=float(eng["spans"].get("long_length_step_m", 100)),
                        terrain=terrain, heading_slack=eng["grid"].get("heading_slack", False),
                        ground_clearance_m=float(eng["spotting"]["ground_clearance_m"][inp.kv]),
                        lowest_attachment_m=float(eng["spotting"]["lowest_attachment_m"][inp.kv]),
                        sag_at_design_m=float(eng["spotting"]["sag_at_design_span_m"][inp.kv]),
                        design_span_m=design,
                        extension_cost_rs=steel_a * float(costs["tower"]["extension_pct"]) / 100.0,
                        hilly_span_max_m=float(rules["design_span"].params["hilly_m"]),
                        hilly_section_rs_per_m=max(0.0, tower_cost["B1"] - tower_cost["A"])
                        * max(0.0, 1.0 / hilly_km - 1.0 / plain_km) / 1000.0)
    sp = SpotParams(design_span_m=design, span_max_m=smax,
                    min_span_m=float(rules["crossing_span_limits"].params["min_span_m"]),
                    max_dev_deg=max_dev, type_bands=bands,
                    section_km=float(rules["section_interval"].params["plain_km"]),
                    parallel_angle_deg=eng["crossings"]["parallel_angle_deg"], tower_cost_rs=tower_cost,
                    ground_clearance_m=float(eng["spotting"]["ground_clearance_m"][inp.kv]),
                    lowest_attachment_m=float(eng["spotting"]["lowest_attachment_m"][inp.kv]),
                    sag_at_design_m=float(eng["spotting"]["sag_at_design_span_m"][inp.kv]),
                    go_radius_m=go_radius,
                    max_dev_adjacent_power_deg={rid: float(rules[rid].params.get("max_dev_adjacent_deg", 45))
                                                for rid in POWER_LINE_RULES if rid in rules},
                    river_max_span_m=float(eng["spans"].get("river_crossing_max_span_m", 0)),
                    nudge_radius_m=float(eng["grid"].get("nudge_radius_m", 30)),
                    hilly_design_span_m=float(rules["design_span"].params["hilly_m"]),
                    hilly_section_km=float(rules["section_interval"].params["hilly_km"]),
                    hilly_slope_deg=float(eng["spotting"].get("hilly_slope_deg", 10)))

    findex = FeatureIndex(frame, inp.layers, layer_names)
    rule_layer = {r.id: r.layer for r in rules.values()}
    global_events = _global_events(inp, frame, rules, comp, findex, layer_names, rule_layer, dem, dem_note)

    variants = eng["routes"][: max(1, min(4, inp.routes_wanted))]
    routes = []
    prev_lines = []
    n_var = len(variants)
    for vi, var in enumerate(variants):
        check_cancel()
        frac0 = 0.15 + 0.8 * vi / n_var
        span_frac = 0.8 / n_var
        prog(frac0, f"Solving route {var['id']} — {var['name']}")
        p = _variant_params(base, var, design)

        def sub(f, _f0=frac0, _sf=span_frac, _v=var):
            prog(_f0 + _sf * 0.7 * f, f"Solving route {_v['id']} — {_v['name']}")

        t = time.time()
        out = solve(comp, p, cancel_check=check_cancel, progress=sub)
        solve_s = time.time() - t
        diversity_ev = None
        if out.feasible and prev_lines:
            line = _outcome_line(out)
            overlap, other = _max_overlap(line, prev_lines, eng["diversity"]["corridor_m"])
            if overlap > 1 - eng["diversity"]["min_distinct_fraction"]:
                pen = _penalty_raster(comp, prev_lines, eng["diversity"])
                p.extra_strip_rs_per_m = pen
                out2 = solve(comp, p, cancel_check=check_cancel, progress=sub)
                if out2.feasible:
                    out = out2
                    diversity_ev = event("DIVERSITY", {"route": var["id"], "overlap_pct": overlap * 100,
                                                       "other": other + 1})
                solve_s = time.time() - t
        if not out.feasible:
            blocked = _blocked(inp, frame, comp, out, findex, rules)
            prog(1.0, "No viable route")
            return {"feasible": False, "blocked": blocked, "routes": [], "events_global": global_events,
                    "timings": {**timings, "total_s": round(time.time() - t_start, 2)},
                    "frame": _frame_info(frame), "rules_fired": comp.rules_fired}
        prog(frac0 + span_frac * 0.75, f"Spotting towers — route {var['id']}")
        pts = [out.grid.site_xy(c, r) for c, r in out.sites]
        pts = [(float(x), float(y)) for x, y in pts]
        t = time.time()
        sr = spot(comp, pts, sp, dem=dem)
        spot_s = time.time() - t
        line = LineString([(tw.x, tw.y) for tw in sr.towers])
        prev_lines.append(line)
        evs = _route_events(sr, frame, comp, findex, rules, sp, layer_names, inp, dem,
                            [e for e in global_events if e["type"] == "BEELINE_BLOCKER"])
        if diversity_ev:
            evs.insert(0, diversity_ev)
        routes.append(_route_json(var, sr, frame, evs, solve_s, spot_s, out))
    prog(1.0, "Done")
    timings["total_s"] = round(time.time() - t_start, 2)
    return {"feasible": True, "routes": routes, "events_global": global_events, "timings": timings,
            "frame": _frame_info(frame), "rules_fired": comp.rules_fired,
            "line": {"kv": inp.kv, "circuit": inp.circuit, "half_width_m": hw, "design_span_m": design,
                     "span_min_m": smin, "span_max_m": smax}}


def _variant_params(base: SolverParams, var: dict, design: float) -> SolverParams:
    from dataclasses import replace
    p = replace(base)
    if "angle_tower_penalty_rs" in var:
        p.angle_tower_penalty_rs = float(var["angle_tower_penalty_rs"])
    if "lateral_penalty_rs_per_m_per_km" in var:
        p.lateral_penalty_rs_per_m = float(var["lateral_penalty_rs_per_m_per_km"]) * design / 1000.0
    if "sensitive_multiplier" in var:
        p.sensitive_multiplier = float(var["sensitive_multiplier"])
    return p


def _outcome_line(out):
    return LineString([out.grid.site_xy(c, r) for c, r in out.sites])


def _max_overlap(line, prev_lines, corridor_m):
    best, who = 0.0, None
    for i, pl in enumerate(prev_lines):
        inside = line.intersection(pl.buffer(corridor_m)).length / max(line.length, 1)
        if inside > best:
            best, who = inside, i
    return best, who


def _penalty_raster(comp, prev_lines, div):
    f = comp.frame
    from .spotting import _frame_geom_to_utm
    geom = unary_union([l.buffer(div["corridor_m"]) for l in prev_lines])
    m = rfeatures.rasterize([(_frame_geom_to_utm(f, geom), 1)], out_shape=comp.cost_class.shape,
                            transform=f.transform, fill=0, dtype="uint8")
    return m.astype(np.float32) * float(div["penalty_rs_per_m"])


def _frame_info(frame: Frame) -> dict:
    return {"epsg": frame.epsg, "length_km": round(frame.length_m / 1000, 3),
            "half_width_km": frame.half_width_m / 1000, "res_m": frame.res_m}


# ----------------------------------------------------------------------------------------
def _global_events(inp, frame, rules, comp, findex, layer_names, rule_layer, dem, dem_note):
    evs = []
    # beeline blockers
    bee = LineString([(0, 0), (frame.length_m, 0)])
    seen = set()
    skip = CROSSING_RULES | {"open_water", "terminal_yard", "pipeline_footing"}
    found = []
    for vr in comp.tower_nogo + comp.conductor_nogo:
        rid = vr.rule_id
        if rid in skip:
            continue      # crossable features and the terminal yards are not reasons to leave the bee-line
        idx = vr.near_idx(bee, limit=3000)
        if not len(idx):
            continue
        inter = bee.intersection(vr.buffered(idx))
        if inter.is_empty or inter.length < 1:
            continue
        parts = getattr(inter, "geoms", [inter])
        for part in parts:
            key = (rid, round(part.bounds[0] / 200))
            if key in seen:
                continue
            seen.add(key)
            it, _ = findex.nearest(part.centroid, layers={vr.layer} if vr.layer else None)
            found.append((vr.layer, part, rid, it))
    # one entry per stretch of the bee-line and layer: the statutory clearance and the no-go buffer
    # on the same buildings are one obstacle, reported under the wider buffer
    found.sort(key=lambda f: -f[1].length)
    kept = []
    for lyr, part, rid, it in found:
        if any(k[0] == lyr and k[1].intersects(part.buffer(1)) for k in kept):
            continue
        kept.append((lyr, part, rid, it))
    # aggregate per feature (named features) or per layer (unnamed, e.g. individual buildings)
    agg: dict[tuple, dict] = {}
    for lyr, part, rid, it in kept:
        label = it["label"] if it else (rules[rid].name if rid in rules else rid)
        key = (lyr, label)
        a = agg.setdefault(key, {"parts": [], "rid": rid, "it": it, "by_rule": {}})
        a["parts"].append(part)
        a["by_rule"][rid] = a["by_rule"].get(rid, 0.0) + part.length
    for (lyr, label), a in sorted(agg.items(), key=lambda kv: min(p.bounds[0] for p in kv[1]["parts"])):
        rid = max(a["by_rule"], key=a["by_rule"].get)
        it = a["it"]
        length = sum(p.length for p in a["parts"])
        x0 = min(p.bounds[0] for p in a["parts"])
        x1 = max(p.bounds[2] for p in a["parts"])
        evs.append(event("BEELINE_BLOCKER", {
            "feature": label, "rule": rules[rid].name if rid in rules else rid, "length_km": length / 1000,
            "stretches": len(a["parts"]), "rule_id": rid, "from_km": x0 / 1000, "to_km": x1 / 1000,
            "feature_idx": it["_idx"] if it else None, "layer": lyr},
            chainage_km=round(x0 / 1000, 2), source=_source(it, inp.layers)))
    # layers
    rules_by_layer: dict[str, list[str]] = {}
    for r in rules.values():
        if r.enabled:
            for lk in ([r.layer] if r.layer else []) + (r.also_layers or []):
                rules_by_layer.setdefault(lk, []).append(r.name)
    band = frame.band_polygon_lonlat()
    band_km2 = _area_km2(frame, band)
    for key, li in inp.layers.items():
        nm = layer_names.get(key, key)
        if li.gdf is None or len(li.gdf) == 0:
            if key in rules_by_layer:
                evs.append(event("LAYER_EMPTY", {"layer": nm, "rules": " · ".join(rules_by_layer[key]),
                                                 "layer_key": key},
                                 source={"layer": key, "layer_name": nm, "mode": li.mode,
                                         "confidence": li.confidence, "source": li.source}))
        if li.coverage is not None:
            unc = band.difference(li.coverage)
            a = _area_km2(frame, unc) if not unc.is_empty else 0.0
            if a > 0.5:
                fx = frame.geom_utm_to_frame(frame.geom_to_utm(unc))
                parts = getattr(fx, "geoms", [fx])
                ranges = ", ".join(f"km {max(0, p.bounds[0]) / 1000:.1f} – {min(frame.length_m, p.bounds[2]) / 1000:.1f}"
                                   for p in sorted(parts, key=lambda g: g.bounds[0])[:6])
                evs.append(event("NO_COVERAGE", {"layer": nm, "uncovered_km2": a, "ranges": ranges,
                                                 "band_km2": band_km2, "layer_key": key}, severity="warn",
                                 source={"layer": key, "layer_name": nm, "mode": li.mode,
                                         "confidence": li.confidence, "source": li.source}))
    for r in rules.values():
        if not r.enabled:
            evs.append(event("RULE_OFF", {"rule": r.name, "rule_id": r.id}))
    if dem is not None and dem.ok:
        evs.append(event("DEM_NOTICE", {"dem": demmod.SOURCE}))
    elif dem_note:
        evs.append(event("DEM_NOTICE", {"dem": dem_note}, severity="warn"))
    return evs


def _area_km2(frame, geom_lonlat):
    return frame.geom_to_utm(geom_lonlat).area / 1e6


def _detours(T, blockers, findex, frame):
    """Stretches where the route leaves the bee-line around a blocker: (from_x, to_x, blocker, side, max_off)."""
    out = []
    xs = np.array([t.x for t in T])
    ys = np.array([t.y for t in T])
    for b in blockers:
        mid = (b["data"]["from_km"] + b["data"]["to_km"]) / 2 * 1000
        y_mid = float(np.interp(mid, xs, ys))
        if abs(y_mid) < 25:
            continue
        side = np.sign(y_mid)
        i = int(np.searchsorted(xs, mid))
        lo = i - 1
        while lo > 0 and np.sign(ys[lo]) == side and abs(ys[lo]) >= 25:
            lo -= 1
        hi = min(i, len(T) - 1)
        while hi < len(T) - 1 and np.sign(ys[hi]) == side and abs(ys[hi]) >= 25:
            hi += 1
        seg = slice(lo, hi + 1)
        out.append((float(xs[lo]), float(xs[hi]), b, "left" if side > 0 else "right", float(np.abs(ys[seg]).max())))
    return out


def _route_events(sr, frame, comp, findex, rules, sp, layer_names, inp, dem, blockers=()):
    evs = []
    T = sr.towers
    detours = _detours(T, blockers, findex, frame)
    route_line = LineString([(t.x, t.y) for t in T])
    groups: dict[tuple, list] = {}
    for x0, x1, b, side, max_off in detours:
        groups.setdefault((round(x0), round(x1), side, round(max_off)), []).append(b)
    detour_info = []
    for (x0, x1, side, max_off), bs in groups.items():
        cleared = []
        for b in bs:
            fi = b["data"].get("feature_idx")
            if fi is None:
                continue
            d = route_line.distance(findex.geoms[fi])
            if d > 5:
                cleared.append((d, b))
        if not cleared:
            continue
        cleared.sort(key=lambda c: -c[1]["data"]["length_km"])
        names = []
        for _, b in cleared:
            n = f"{b['data']['feature']} ({b['data']['rule']})"
            if n not in names:
                names.append(n)
        label = ", ".join(names[:3]) + (f" and {len(names) - 3} more" if len(names) > 3 else "")
        main = cleared[0][1]
        detour_info.append((x0, x1, main, label))
        evs.append(event("DETOUR", {"from_km": x0 / 1000, "to_km": x1 / 1000, "side": side, "max_offset_m": max_off,
                                    "feature": label, "rule": main["data"]["rule"],
                                    "clearance_m": min(c[0] for c in cleared)},
                         chainage_km=round(x0 / 1000, 2), source=main.get("source")))
    vec_rules = comp.tower_nogo + comp.angle_nogo + comp.conductor_nogo
    search = load_engine()["events"]["turn_feature_search_m"]
    low = load_engine()["events"]["low_confidence"]
    rule_layer = {r.id: r.layer for r in rules.values()}
    for i in range(1, len(T) - 1):
        t = T[i]
        if not t.angle_point:
            continue
        if "dead_end" in t.reasons and t.deflection_deg < 30:
            continue      # leaving / entering the gantry, not a routing decision
        p0, p1 = T[i - 1], t
        dx, dy = p1.x - p0.x, p1.y - p0.y
        n = math.hypot(dx, dy) or 1
        # look along the incoming heading up to the next angle point (+ the search distance):
        # the feature that straight-on would have hit is the one this turn avoids
        nxt = next((T[j] for j in range(i + 1, len(T)) if T[j].angle_point or T[j].type == "G"), T[-1])
        reach = max(search, (nxt.x - p1.x) / max(dx / n, 0.2) + search) if dx > 0 else search
        ray = LineString([(p1.x, p1.y), (p1.x + dx / n * reach, p1.y + dy / n * reach)])
        hit = None
        P = Point(p1.x, p1.y)
        for vr in vec_rules:
            idx = vr.near_idx(ray, limit=200)
            if not len(idx):
                continue
            ip = ray.intersection(vr.buffered(idx))
            if ip.is_empty:
                continue
            d = P.distance(ip)
            if hit is None or d < hit[1]:
                hit = (vr.rule_id, d, ip, vr.layer)
        data = {"tower": t.no, "type": t.type, "deflection": t.deflection_deg,
                "dir": {"L": "left", "R": "right"}.get(t.turn, ""), "chainage_km": round(t.x / 1000, 2)}
        if hit:
            rid, d, ip, lyr = hit
            it, dist = findex.nearest(ip, layers={lyr} if lyr else None)
            line_here = LineString([(T[i - 1].x, T[i - 1].y), (t.x, t.y), (T[i + 1].x, T[i + 1].y)])
            clearance = route_line.distance(findex.geoms[it["_idx"]]) if it else d
            local = line_here.distance(findex.geoms[it["_idx"]]) if it else d
            data.update(feature=(it["label"] if it else rid.replace("_", " ")) + (f" ({it['class'].replace('_', ' ')})" if it and it["label"] != it["class"].replace("_", " ") else ""),
                        clearance_m=clearance, rule_id=rid)
            if rid in CROSSING_RULES and min(local, clearance) < 1.0:
                kind = "TURN_TO_CROSS"
            elif clearance > 300:
                kind = "TURN_TO_KEEP_CLEAR"
            else:
                kind = "TURN_TO_AVOID"
            evs.append(event(kind, data, chainage_km=data["chainage_km"], source=_source(it, inp.layers)))
        elif (d := next((d for d in detour_info if d[0] - 1 <= t.x <= d[1] + 1), None)) is not None:
            x0, x1, b, label = d
            fi = b["data"].get("feature_idx")
            data.update(feature=b["data"]["feature"] + f" ({b['data']['rule']})", rule_id=b["data"].get("rule_id"),
                        clearance_m=route_line.distance(findex.geoms[fi]) if fi is not None else 0.0)
            evs.append(event("TURN_TO_AVOID", data, chainage_km=data["chainage_km"], source=b.get("source")))
        else:
            o_dx, o_dy = T[i + 1].x - t.x, T[i + 1].y - t.y
            heading_back = abs(t.y) > 1.0 and (o_dy * t.y) < 0 and abs(o_dy / (math.hypot(o_dx, o_dy) or 1)) > abs(dy / n)
            evs.append(event("TURN_BACK" if heading_back else "TURN", data, chainage_km=data["chainage_km"]))
    for s in sr.spans:
        for c in s.crossings:
            a, b = T[s.a], T[s.b]
            mid = Point((a.x + b.x) / 2, (a.y + b.y) / 2)
            lyr = rule_layer.get(c["rule"])
            it, _ = findex.nearest(mid, layers={lyr} if lyr else None, max_dist=s.length_m)
            label = (it["label"] if it else c["name"])
            if c.get("major"):
                label += " (major crossing)"
            evs.append(event("CROSSING", {"feature": label, "angle_deg": c["angle_deg"], "span_m": c["span_m"],
                                          "tower_a": a.no, "tower_b": b.no, "type_a": a.type, "type_b": b.type,
                                          "rule_id": c["rule"], "chainage_km": round(mid.x / 1000, 2)},
                             chainage_km=round(mid.x / 1000, 2), source=_source(it, inp.layers)))
    for t in T:
        if "section" in t.reasons or "section_hilly" in t.reasons:
            km = sp.hilly_section_km if "section_hilly" in t.reasons else sp.section_km
            evs.append(event("SECTION_POINT", {"tower": t.no, "type": t.type, "section_km": km},
                             chainage_km=round(t.x / 1000, 2)))
        if "clearance" in t.reasons:
            evs.append(event("CLEARANCE_TOWER", {"tower": t.no, "clearance_req": sp.ground_clearance_m, "dem": demmod.SOURCE},
                             chainage_km=round(t.x / 1000, 2)))
        if t.extension_m > 0 and "crossing_tower" in t.reasons:
            s_long = next((s for s in sr.spans if s.a == t.no and s.length_m > sp.span_max_m), None)
            if s_long:
                evs.append(event("CROSSING_TOWERS", {"span_m": s_long.length_m, "tower_a": t.no, "tower_b": T[s_long.b].no,
                                                     "extension_m": t.extension_m, "clearance_req": sp.ground_clearance_m,
                                                     "dem": demmod.SOURCE}, chainage_km=round(t.x / 1000, 2)))
        elif t.extension_m > 0:
            evs.append(event("EXTENSION", {"tower": t.no, "extension_m": t.extension_m,
                                           "clearance_req": sp.ground_clearance_m, "dem": demmod.SOURCE},
                             chainage_km=round(t.x / 1000, 2)))
        for w in t.warnings:
            if w == "ground_clearance":
                sp_i = next((s for s in sr.spans if s.a == t.no), None)
                evs.append(event("CLEARANCE_SHORT", {"tower_a": t.no, "clearance_m": sp_i.min_clearance_m if sp_i else 0,
                                                     "clearance_req": sp.ground_clearance_m, "dem": demmod.SOURCE},
                                 chainage_km=round(t.x / 1000, 2), severity="warn"))
            elif w.startswith("power_adjacent_deviation"):
                rid = w.partition(":")[2] or "power_line"
                kind = "low tension" if rid == "power_line_lt" else "high tension"
                evs.append(event("RULE_WARNING", {"tower": t.no, "what": f"deviates {t.deflection_deg:.0f}° next to a {kind} power-line crossing; the rule allows {sp.max_dev_adjacent_power_deg.get(rid, 45):g}° at n±1",
                                                  "code": w}, chainage_km=round(t.x / 1000, 2), severity="warn"))
        for v in t.violations:
            what = {"tower_buffer": "tower stands inside a no-go buffer",
                    "conductor_buffer": "span passes through a conductor no-go buffer"}.get(
                v.split(":")[0], v.replace("_", " "))
            evs.append(event("VIOLATION", {"tower": t.no, "what": what, "code": v},
                             chainage_km=round(t.x / 1000, 2), severity="warn"))
    fr = rules["forest"]
    fha = sr.stats["forest_ha"]
    if fha > 0.01:
        ro = float(fr.params.get("ro_threshold_ha", 5))
        cons = float(fr.params.get("conservator_threshold_ha", 40))
        note = ""
        if fha > cons:
            note = f" Above the {cons:g} ha conservator inspection threshold."
        elif fha > ro:
            note = f" Above the {ro:g} ha Regional Office threshold."
        it = None
        line = LineString([(t.x, t.y) for t in T])
        it, _ = findex.nearest(line, layers={"reserve_forest", "forest_in_pa"}, max_dist=comp.half_width_m + 50)
        evs.append(event("FOREST_ENTRY", {"forest_ha": fha, "threshold_note": note}, source=_source(it, inp.layers)))
    # airport NOC
    noc = rules.get("airport_noc")
    if noc and noc.enabled:
        rad = float(noc.params["radius_km"]) * 1000
        fg = [(i, g) for i, g in enumerate(findex.geoms) if findex.items[i]["layer"] == "aviation"]
        for i, g in fg:
            n_t = sum(1 for t in T if Point(t.x, t.y).distance(g) <= rad)
            if n_t:
                evs.append(event("NOC_REQUIRED", {"n_towers": n_t, "radius_km": rad / 1000, "feature": findex.items[i]["label"],
                                                  "height_source": noc.params.get("height_source", "CCZM")},
                                 source=_source(findex.items[i], inp.layers)))
    # low confidence: every decision event whose source layer is below the threshold
    for e in list(evs):
        s = e.get("source")
        if s and s.get("confidence") is not None and s["confidence"] < low and e["type"] in (
                "TURN_TO_AVOID", "TURN_TO_CROSS", "TURN_TO_KEEP_CLEAR", "DETOUR", "CROSSING", "FOREST_ENTRY"):
            decision = {"TURN_TO_AVOID": f"The turn at km {e['chainage_km']:.1f}",
                        "TURN_TO_CROSS": f"The turn at km {e['chainage_km']:.1f}",
                        "TURN_TO_KEEP_CLEAR": f"The turn at km {e['chainage_km']:.1f}",
                        "DETOUR": f"The detour from km {e['chainage_km']:.1f}",
                        "CROSSING": f"The crossing at km {e['chainage_km']:.1f}",
                        "FOREST_ENTRY": "The forest area taken"}[e["type"]]
            evs.append(event("LOW_CONF", {"decision": decision, "layer": s["layer_name"], "confidence": s["confidence"]},
                             chainage_km=e["chainage_km"], source=s, severity="warn"))
    evs.sort(key=lambda e: (e["chainage_km"] is None, e["chainage_km"] or 0))
    return evs


def _route_json(var, sr, frame, evs, solve_s, spot_s, out) -> dict:
    T = sr.towers
    xs = np.array([t.x for t in T])
    ys = np.array([t.y for t in T])
    ux, uy = frame.frame_to_utm(xs, ys)
    lon, lat = frame.utm_to_lonlat(ux, uy)
    towers = []
    for i, t in enumerate(T):
        towers.append({
            "no": t.no, "lon": round(float(lon[i]), 7), "lat": round(float(lat[i]), 7),
            "x_utm": round(float(ux[i]), 2), "y_utm": round(float(uy[i]), 2),
            "chainage_km": round(t.x / 1000, 3), "offset_m": round(t.y, 1),
            "type": t.type, "deflection_deg": t.deflection_deg, "turn": t.turn, "angle_point": t.angle_point,
            "reasons": t.reasons, "ground_z": t.ground_z, "extension_m": t.extension_m, "violations": t.violations,
            "warnings": t.warnings,
            "span_ahead_m": round(sr.spans[i].length_m, 1) if i < len(sr.spans) else None,
        })
    spans = [{"a": T[s.a].no, "b": T[s.b].no, "length_m": round(s.length_m, 1), "crossings": s.crossings,
              "min_clearance_m": s.min_clearance_m} for s in sr.spans]
    return {
        "id": var["id"], "key": var["key"], "name": var["name"], "feasible": True,
        "geometry": [[t["lon"], t["lat"]] for t in towers],
        "towers": towers, "spans": spans, "cost": {k: round(v, 0) for k, v in sr.cost.items()},
        "stats": sr.stats, "events": evs, "spot_method": sr.method,
        "timings": {"solve_s": round(solve_s, 2), "spot_s": round(spot_s, 2)},
        "grid_cost_rs": round(out.cost_rs, 0),
    }


def _blocked(inp, frame, comp, out, findex, rules) -> dict:
    """Name the features that close the band where the DP stopped reaching."""
    reach = out.reach_chainage_m
    lo, hi = max(0.0, reach - 500), min(frame.length_m, reach + 3000)
    slab = box(lo, -frame.half_width_m, hi, frame.half_width_m)
    scores = []
    for vr in comp.tower_nogo + comp.conductor_nogo:
        idx = vr.near_idx(slab, limit=5000)
        if not len(idx):
            continue
        inter = vr.buffered(idx).intersection(slab)
        if inter.is_empty:
            continue
        lat_cover = (inter.bounds[3] - inter.bounds[1]) / (2 * frame.half_width_m)
        scores.append((lat_cover, vr.rule_id, inter, vr.layer))
    scores.sort(key=lambda s: -s[0])
    blockers = []
    seen = set()
    for cov, rid, inter, vlyr in scores[:6]:
        if rid in seen:
            continue
        seen.add(rid)
        r = rules.get(rid)
        lyr = vlyr or (r.layer if r else None)      # the layer this buffer came from (also_layers too)
        it, _ = findex.nearest(inter.centroid, layers={lyr} if lyr else None)
        blockers.append({"rule_id": rid, "rule": r.name if r else rid, "statutory": bool(r and r.statutory),
                         "feature": it["label"] if it else (r.name if r else rid),
                         "from_km": round(inter.bounds[0] / 1000, 2), "to_km": round(inter.bounds[2] / 1000, 2),
                         "band_cover_pct": round(cov * 100), "source": _source(it, inp.layers)})
    first = blockers[0] if blockers else None
    ev = event("BAND_BLOCKED", {"band_km": inp.band_km, "from_km": first["from_km"] if first else reach / 1000,
                                "to_km": first["to_km"] if first else reach / 1000,
                                "feature": first["feature"] if first else "the combined constraints"}, severity="error")
    return {"reach_km": round(reach / 1000, 2), "blockers": blockers, "event": ev}
