"""Tower spotting on the solved alignment, and the route's true cost.

1. String-pulling: replace runs of grid towers by straight legs between angle points, as long
   as every leg passes the exact (vector) conductor checks and its towers can be spotted.
2. Spot suspension towers along each leg at equal spans (design span or shorter).
3. Section points: upgrade a suspension tower to B1 where tension towers are further apart
   than the section interval.
4. Forced crossing tower types at both ends of every controlled crossing span.
5. Ground clearance on the DEM; a short tower gets the +18 m extension if that clears it.
6. True cost and statistics of the route.

The DP alignment is costed the same way and kept if the pulled one is not cheaper.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from rasterio import features as rfeatures
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .compile import COST_CLASS_CODES, RS_PER_CR, RS_PER_L, Compiled, class_density_rs_m2
from .config import TOWER_TYPES, max_type, tower_type_for

log = logging.getLogger(__name__)
EXTENSION_M = 18.0


@dataclass
class SpotParams:
    design_span_m: float
    span_max_m: float
    min_span_m: float
    max_dev_deg: float
    type_bands: dict[str, float]
    section_km: float
    parallel_angle_deg: float
    tower_cost_rs: dict[str, float]
    ground_clearance_m: float
    lowest_attachment_m: float
    sag_at_design_m: float
    go_radius_m: float
    max_dev_adjacent_power_deg: float = 45.0
    river_max_span_m: float = 0.0
    nudge_radius_m: float = 30.0


@dataclass
class Tower:
    no: int
    x: float
    y: float
    type: str
    deflection_deg: float
    turn: str                      # L / R / -
    angle_point: bool
    reasons: list[str] = field(default_factory=list)   # section, crossing:<rule>, dead_end
    ground_z: float | None = None
    extension_m: float = 0.0
    violations: list[str] = field(default_factory=list)   # hard: geometry breaks a rule
    warnings: list[str] = field(default_factory=list)     # soft: verify on site


@dataclass
class Span:
    a: int
    b: int
    length_m: float
    crossings: list[dict] = field(default_factory=list)
    min_clearance_m: float | None = None


@dataclass
class SpottedRoute:
    towers: list[Tower]
    spans: list[Span]
    cost: dict[str, float]
    stats: dict
    method: str                    # pulled | grid


# ----------------------------------------------------------------------------------------
class Checker:
    """Exact and raster checks in frame coordinates."""

    def __init__(self, comp: Compiled, sp: SpotParams):
        self.c = comp
        self.sp = sp
        f = comp.frame
        self.go = comp.terminal_zone if comp.terminal_zone is not None else \
            unary_union([Point(0, 0).buffer(sp.go_radius_m), Point(f.length_m, 0).buffer(sp.go_radius_m)])
        self.yards = comp.terminal_yards
        self.tower_rules = comp.tower_nogo
        self.angle_rules = comp.angle_nogo
        self.cond_rules = comp.conductor_nogo

    def in_go_zone(self, p: Point) -> bool:
        return self.go.contains(p)

    def tower_ok(self, x, y, angle_tower: bool) -> bool:
        if abs(y) > self.c.frame.half_width_m + 1e-6:
            return False
        p = Point(x, y)
        if self.in_go_zone(p):
            return self.yards is None or not self.yards.contains(p)
        if any(r.hits(p) for r in self.tower_rules):
            return False
        if angle_tower and any(r.hits(p) for r in self.angle_rules):
            return False
        return True

    def conductor_ok(self, p, q) -> bool:
        line = LineString([p, q])
        if line.intersects(self.go):
            line = line.difference(self.go)
            if line.is_empty:
                return True
        return not any(r.hits(line) for r in self.cond_rules)

    def _sample(self, arr, p, q, step=None):
        f = self.c.frame
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(math.ceil(L / (step or f.res_m / 2))) + 1)
        t = np.linspace(0, 1, n)
        fx = p[0] + t * (q[0] - p[0])
        fy = p[1] + t * (q[1] - p[1])
        col = np.clip(((fx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
        row = np.clip(((f.y_top - fy) / f.res_m).astype(int), 0, f.height - 1)
        return arr[row, col], L / (n - 1)

    def span_crossings(self, p, q) -> tuple[bool, list[dict]]:
        """(valid, crossings) for one span, by the same rules the solver applied."""
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        head = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        out, ok = [], True
        for cr in self.c.crossings:
            a, step = self._sample(cr.angle, p, q)
            hit = ~np.isnan(a)
            if not hit.any():
                continue
            fa = float(a[np.argmax(hit)])
            d = (head - fa) % 180.0
            acute = min(d, 180.0 - d)
            item = {"rule": cr.rule_id, "name": cr.name, "angle_deg": round(acute, 1), "span_m": round(L, 1),
                    "min_type": cr.min_type, "cost_rs": cr.cost_rs, "major": False}
            if cr.window is not None and acute < cr.window[0] - 1e-3:
                ok = False
                item["violation"] = "angle"
            if cr.max_span_m is not None and L > cr.max_span_m + 1e-3:
                ok = False
                item["violation"] = "span"
            if cr.width_raster is not None and cr.major_threshold_m is not None:
                w, st = self._sample(cr.width_raster, p, q)
                width = float(w.sum()) * st
                item["width_m"] = round(width, 1)
                if width >= cr.major_threshold_m:
                    item.update(major=True, min_type=cr.major_type, cost_rs=cr.major_cost_rs)
            out.append(item)
        if L > self.sp.span_max_m + 1e-3:
            if L > self.sp.river_max_span_m + 1e-3 or not any(c["rule"] == "river" for c in out):
                ok = False
        for pr in self.c.parallels:
            a, _ = self._sample(pr.angle, p, q)
            near = ~np.isnan(a)
            if near.any():
                d = np.abs((head - a[near]) % 180.0)
                acute = np.minimum(d, 180.0 - d)
                if (acute < self.sp.parallel_angle_deg).any():
                    ok = False
        return ok, out


def _deflection(p0, p1, p2) -> tuple[float, str]:
    h1 = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    h2 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    d = math.degrees((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
    return abs(d), ("L" if d > 0 else "R" if d < 0 else "-")


def _spot_leg(chk: Checker, p, q, sp: SpotParams):
    """Intermediate tower positions on a straight leg p->q, or None if no spotting works."""
    if not chk.conductor_ok(p, q):
        return None
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    n0 = max(1, math.ceil(d / sp.design_span_m - 1e-9), math.ceil(d / sp.span_max_m - 1e-9))
    cands = list(range(n0, n0 + 5))
    if n0 > 1 and d <= sp.river_max_span_m:
        cands = [1] + cands       # a single long span across a river
    for n in cands:
        span = d / n
        if span < sp.min_span_m and n > 1:
            break
        pts = [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(1, n)]
        if not all(chk.tower_ok(x, y, angle_tower=False) for x, y in pts):
            continue
        seq = [p] + pts + [q]
        good = True
        for a, b in zip(seq[:-1], seq[1:]):
            ok, _ = chk.span_crossings(a, b)
            if not ok:
                good = False
                break
        if good:
            return pts
    return None


def pull_string(chk: Checker, grid_pts: list[tuple[float, float]], sp: SpotParams, window: int = 60):
    """Greedy farthest-reachable angle points over the solver's towers.

    From each angle point, take the farthest later tower that a straight, fully spotted leg can
    reach. If none can (the exact check disagrees with the grid at this tower), keep the solver's
    own next span so the alignment stays complete; the violation is reported downstream."""
    n = len(grid_pts)
    aps = [grid_pts[0]]
    towers = [grid_pts[0]]
    i = 0
    prev = None
    pulled_any = False
    while i < n - 1:
        chosen = None
        for j in range(min(n - 1, i + window), i + 1, -1):
            q = grid_pts[j]
            if prev is not None:
                dfl, _ = _deflection(prev, grid_pts[i], q)
                if dfl > sp.max_dev_deg + 1e-6:
                    continue
            if j < n - 1 and not chk.tower_ok(q[0], q[1], angle_tower=True):
                continue
            inner = _spot_leg(chk, grid_pts[i], q, sp)
            if inner is None:
                continue
            chosen = (j, inner)
            pulled_any = True
            break
        if chosen is None:
            # the exact check rejects every leg from here: nudge the next grid tower
            q = _nudge(chk, grid_pts[i], grid_pts[i + 1], grid_pts[i + 2] if i + 2 < n else None, prev, sp,
                       is_last=(i + 1 == n - 1))
            if q is not None:
                grid_pts = list(grid_pts)
                grid_pts[i + 1] = q
            chosen = (i + 1, [])
        j, inner = chosen
        towers.extend(inner)
        towers.append(grid_pts[j])
        aps.append(grid_pts[j])
        prev = towers[-2] if len(towers) >= 2 else grid_pts[i]
        i = j
    if not pulled_any:
        return None
    return aps, towers


def _nudge(chk: Checker, p, q, nxt, prev, sp: SpotParams, is_last: bool):
    """Move q by up to nudge_radius_m so that the span p->q (and q->nxt) pass the exact checks."""
    best = None
    for rad in (5, 10, 15, 20, 25, 30):
        if rad > sp.nudge_radius_m:
            break
        for k in range(16):
            a = 2 * math.pi * k / 16
            c = (q[0] + rad * math.cos(a), q[1] + rad * math.sin(a))
            if not chk.tower_ok(c[0], c[1], angle_tower=True):
                continue
            L = math.hypot(c[0] - p[0], c[1] - p[1])
            if L < sp.min_span_m or not chk.conductor_ok(p, c) or not chk.span_crossings(p, c)[0]:
                continue
            if prev is not None and _deflection(prev, p, c)[0] > sp.max_dev_deg:
                continue
            if nxt is not None:
                L2 = math.hypot(nxt[0] - c[0], nxt[1] - c[1])
                if L2 < sp.min_span_m or not chk.conductor_ok(c, nxt) or not chk.span_crossings(c, nxt)[0]:
                    continue
            best = c
            break
        if best is not None:
            return best
    return None


def build_route(chk: Checker, pts: list[tuple[float, float]], angle_set: set[int] | None,
                sp: SpotParams, dem=None) -> SpottedRoute:
    """Tower list from points; angle_set = indices that are angle points (None = all with deflection)."""
    comp = chk.c
    towers: list[Tower] = []
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        if i == 0 or i == n - 1:
            towers.append(Tower(i, x, y, "G", 0.0, "-", False, ["gantry"]))
            continue
        dfl, turn = _deflection(pts[i - 1], pts[i], pts[i + 1])
        if i == 1 or i == n - 2:
            towers.append(Tower(i, x, y, "D60", round(dfl, 2), turn, True, ["dead_end"]))
            continue
        t = tower_type_for(dfl, sp.type_bands) or "D60"
        is_ap = t != "A"
        towers.append(Tower(i, x, y, t, round(dfl, 2), turn if is_ap else "-", is_ap))
    spans: list[Span] = []
    for i in range(n - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        gantry_span = i == 0 or i == n - 2
        cr = [] if gantry_span else chk.span_crossings(pts[i], pts[i + 1])[1]
        s = Span(i, i + 1, L, cr)
        spans.append(s)
        for c in cr:
            if c.get("violation"):
                towers[i].violations.append(f"crossing_{c['violation']}:{c['rule']}")
        if not chk.conductor_ok(pts[i], pts[i + 1]):
            towers[i if i > 0 else 1].violations.append("conductor_buffer")
    # forced crossing types
    for s in spans:
        for c in s.crossings:
            if c["min_type"] != "A":
                for ti in (s.a, s.b):
                    t = towers[ti]
                    if t.type == "G":
                        continue
                    nt = max_type(t.type, c["min_type"])
                    if nt != t.type:
                        t.type = nt
                    t.reasons.append(f"crossing:{c['rule']}")
            if c["rule"] == "power_line":
                for ti in (s.a - 1, s.b + 1):
                    if 0 < ti < len(towers) - 1 and towers[ti].deflection_deg > sp.max_dev_adjacent_power_deg:
                        towers[ti].warnings.append("power_adjacent_deviation")
    # section points
    last_tension = 0.0
    run = 0.0
    for i in range(1, len(towers)):
        run += spans[i - 1].length_m
        t = towers[i]
        if t.type != "A":
            run = 0.0
            continue
        nxt = spans[i].length_m if i < len(spans) else 0
        if run + nxt > sp.section_km * 1000.0:
            t.type = "B1"
            t.reasons.append("section")
            run = 0.0
    # exact tower checks
    for i, t in enumerate(towers):
        if i in (0, len(towers) - 1):
            continue
        if not chk.tower_ok(t.x, t.y, angle_tower=t.type != "A"):
            t.violations.append("tower_buffer")
    # ground clearance
    if dem is not None and dem.ok:
        xs = np.array([t.x for t in towers])
        ys = np.array([t.y for t in towers])
        lon, lat = comp.frame.frame_to_lonlat(xs, ys)
        gz = dem.sample(lon, lat)
        for t, z in zip(towers, gz):
            t.ground_z = None if not np.isfinite(z) else round(float(z), 1)
        for s in spans:
            _clearance(comp, dem, towers, s, sp)
    cost, stats = route_cost(comp, towers, spans, sp)
    return SpottedRoute(towers, spans, cost, stats, "pulled" if angle_set is None else "grid")


def _clearance(comp, dem, towers, s: Span, sp: SpotParams):
    ta, tb = towers[s.a], towers[s.b]
    if ta.type == "G" or tb.type == "G" or ta.ground_z is None or tb.ground_z is None:
        return
    n = max(3, int(s.length_m // 15) + 1)
    t = np.linspace(0, 1, n)
    fx = ta.x + t * (tb.x - ta.x)
    fy = ta.y + t * (tb.y - ta.y)
    lon, lat = comp.frame.frame_to_lonlat(fx, fy)
    ground = dem.sample(lon, lat)
    sag = sp.sag_at_design_m * (s.length_m / sp.design_span_m) ** 2

    def min_clear():
        za = ta.ground_z + sp.lowest_attachment_m + ta.extension_m
        zb = tb.ground_z + sp.lowest_attachment_m + tb.extension_m
        line = (1 - t) * za + t * zb - 4 * sag * t * (1 - t)
        return float(np.nanmin(line - ground))

    c = min_clear()
    if c < sp.ground_clearance_m:
        low = ta if (ta.ground_z + ta.extension_m) <= (tb.ground_z + tb.extension_m) else tb
        if low.extension_m == 0:
            low.extension_m = EXTENSION_M
            c = min_clear()
        if c < sp.ground_clearance_m:
            ta.warnings.append("ground_clearance")
    s.min_clearance_m = round(c, 1)


def route_cost(comp: Compiled, towers: list[Tower], spans: list[Span], sp: SpotParams):
    costs = comp.costs
    f = comp.frame
    rate = float(costs["tower"]["rate_per_tonne_rs"])
    ext_pct = float(costs["tower"]["extension_pct"]) / 100.0
    fnd_base = float(costs["foundation"]["base_cr_tower"]) * RS_PER_CR
    base_area = float(costs["base_area_m2"])
    base_pct = costs["land"]["tower_base_pct"] / 100.0
    earthing = float(costs["tower"]["earthing_rs"])
    c_tower = c_found = c_land = 0.0
    for t in towers:
        if t.type == "G":
            continue
        px = int(np.clip((t.x - f.x0) / f.res_m, 0, f.width - 1))
        py = int(np.clip((f.y_top - t.y) / f.res_m, 0, f.height - 1))
        steel_ins = sp.tower_cost_rs[t.type]
        steel = costs["tonnage"][t.type] * rate
        c_tower += steel_ins + earthing + (steel * ext_pct if t.extension_m > 0 else 0.0)
        c_found += fnd_base * float(comp.soil_mult[py, px])
        c_land += base_area * float(comp.land_rate_m2[py, px]) * base_pct
    length = sum(s.length_m for s in spans)
    c_cond = length * comp.conductor_rs_per_m

    # RoW corridor statistics on the raster
    line = LineString([(t.x, t.y) for t in towers])
    corridor = line.buffer(comp.half_width_m, cap_style="flat", join_style="mitre")
    utm = _frame_geom_to_utm(f, corridor)
    mask = rfeatures.rasterize([(utm, 1)], out_shape=comp.cost_class.shape, transform=f.transform,
                               fill=0, dtype="uint8").astype(bool)
    cell = f.res_m ** 2
    dens = class_density_rs_m2(costs)
    areas = {k: float((mask & (comp.cost_class == v)).sum()) * cell / 1e4 for k, v in COST_CLASS_CODES.items()}
    land = costs["land"]
    crops = costs["crops"]
    forest = costs["forest"]
    row = land["row_corridor_pct"] / 100.0
    stamp = land["stamp_legal_l_ha"] * RS_PER_L
    agri_ha = areas["agriculture"] + areas["orchard"]
    nonagri_ha = areas["nonagri"] + areas["social_forest"]
    c_row = (agri_ha * land["agri_land_cr_ha"] * RS_PER_CR * row + nonagri_ha * land["nonagri_land_cr_ha"] * RS_PER_CR * row
             + (agri_ha + nonagri_ha) * stamp)
    c_crops = (areas["agriculture"] * crops["standing_crop_l_ha"] * RS_PER_L
               + areas["orchard"] * crops["orchard_trees_per_ha"] * crops["orchard_tree_rs"]
               + areas["social_forest"] * crops["social_forest_trees_per_ha"] * crops["timber_tree_rs"])
    c_forest = areas["forest"] * ((forest["npv_l_ha"] + forest["ca_l_ha"]) * RS_PER_L + forest["rf_land_cr_ha"] * RS_PER_CR * row)
    if areas["forest"] > 0.01:
        c_forest += forest["diversion_l_case"] * RS_PER_L
    c_cross = sum(c["cost_rs"] for s in spans for c in s.crossings)
    cost = {
        "towers": c_tower, "foundation": c_found, "conductor": c_cond,
        "land_row": c_row + c_land, "crops_trees": c_crops, "forest": c_forest,
        "crossings": c_cross, "environment": 0.0,
    }
    cost["total"] = sum(cost.values())
    mix = {t: 0 for t in TOWER_TYPES}
    real = [t for t in towers if t.type != "G"]
    for t in real:
        mix[t.type] += 1
    cross_counts: dict[str, int] = {}
    for s in spans:
        for c in s.crossings:
            cross_counts[c["rule"]] = cross_counts.get(c["rule"], 0) + 1
    stats = {
        "length_km": round(length / 1000, 3),
        "beeline_km": round(f.length_m / 1000, 3),
        "vs_beeline_pct": round((length / f.length_m - 1) * 100, 2),
        "towers": len(real),
        "angle_points": sum(1 for t in real if t.angle_point and "dead_end" not in t.reasons),
        "angle_towers": len(real) - mix["A"],
        "suspension_towers": mix["A"],
        "tension_towers": len(real) - mix["A"],
        "tower_mix": mix,
        "forest_ha": round(areas["forest"], 2),
        "agri_ha": round(areas["agriculture"] + areas["orchard"], 2),
        "corridor_ha": round(sum(areas.values()), 2),
        "areas_ha": {k: round(v, 2) for k, v in areas.items()},
        "crossings": cross_counts,
        "crossings_total": sum(cross_counts.values()),
        "extensions": sum(1 for t in real if t.extension_m > 0),
        "violations": sum(len(t.violations) for t in towers),
        "warnings": sum(len(t.warnings) for t in towers),
        "max_lateral_m": round(max(abs(t.y) for t in towers), 1),
    }
    return cost, stats


def _frame_geom_to_utm(f, g):
    from shapely import affinity
    g = affinity.rotate(g, f.bearing_rad, origin=(0, 0), use_radians=True)
    return affinity.translate(g, f.A[0], f.A[1])


def spot(comp: Compiled, grid_pts: list[tuple[float, float]], sp: SpotParams, dem=None) -> SpottedRoute:
    """grid_pts: the solver's towers, first dead-end to last. Gantries A and B are added here."""
    chk = Checker(comp, sp)
    A = (0.0, 0.0)
    B = (comp.frame.length_m, 0.0)
    grid_route = build_route(chk, [A] + grid_pts + [B], angle_set=set(), sp=sp, dem=dem)
    pulled = pull_string(chk, grid_pts, sp) if len(grid_pts) > 2 else None
    if pulled is None:
        log.info("string pulling found no valid simplification; keeping the grid alignment")
        return grid_route
    _, towers = pulled
    pulled_route = build_route(chk, [A] + towers + [B], angle_set=None, sp=sp, dem=dem)
    better = (pulled_route.stats["violations"], pulled_route.cost["total"]) <= \
             (grid_route.stats["violations"], grid_route.cost["total"])
    return pulled_route if better else grid_route
