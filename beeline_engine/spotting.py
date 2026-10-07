"""Tower spotting on the solved alignment, and the route's true cost.

1. String-pulling: replace runs of grid towers by straight legs between angle points, as long
   as every leg passes the exact (vector) conductor checks, its towers can be spotted, and the
   straight leg is not dearer than the grid towers it replaces (towers, foundations, land, RoW
   strip incl. forest): a cost-blind straightening cut corners through forest and rock and made
   a hill route ₹14 Cr dearer than the search had priced it (6 Oct 2026).
2. Spot suspension towers along each leg at equal spans (design span or shorter).
3. Section points: upgrade a suspension tower to B1 where tension towers are further apart
   than the section interval (the hilly interval once the stretch includes a hilly span).
   Hilly: the DEM slope reaches hilly_slope_deg over 90 m somewhere on the span / leg; hilly legs
   are spotted at the hilly design span (Vivek, 6 Oct 2026).
4. Forced crossing tower types at both ends of every controlled crossing span.
5. Ground clearance on the DEM: a span that the +18 m extension on both towers cannot clear gets
   an extra suspension tower at its worst point (the ridge), repeated as needed; otherwise a
   short tower gets the +18 m extension if that clears it.
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

from .compile import COST_CLASS_CODES, Compiled
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
    # power-line crossing rule id -> max deviation at n±1 (high and low tension rules, 6 Oct 2026)
    max_dev_adjacent_power_deg: dict[str, float] = field(default_factory=dict)
    river_max_span_m: float = 0.0
    nudge_radius_m: float = 30.0
    # hilly terrain (rules design_span.hilly_m, section_interval.hilly_km; engine hilly_slope_deg)
    hilly_design_span_m: float | None = None
    hilly_section_km: float | None = None
    hilly_slope_deg: float | None = None
    pull_cost_tol: float = 0.01          # a straightened leg may cost at most this much more than the grid towers it replaces


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
    hilly: bool = False


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
        self.dem = None                    # set by spot(): hilly legs and clearance re-spotting

    def profile(self, p, q, step=30.0):
        """(t in 0..1, ground z) along p->q on the DEM, or None without a DEM."""
        if self.dem is None or not self.dem.ok:
            return None
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(3, int(L // step) + 1)
        t = np.linspace(0, 1, n)
        lon, lat = self.c.frame.frame_to_lonlat(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))
        return t, self.dem.sample(lon, lat), L

    def is_hilly(self, p, q) -> bool:
        if self.sp.hilly_slope_deg is None:
            return False
        pr = self.profile(p, q)
        if pr is None:
            return False
        t, z, L = pr
        if len(z) < 4 or L <= 0:
            return False
        step = L / (len(z) - 1)
        k = max(1, int(round(90.0 / step)))
        if len(z) <= k:
            k = len(z) - 1
        dz = np.abs(z[k:] - z[:-k])
        slope = np.degrees(np.arctan(dz / (k * step)))
        return bool(np.nanmax(slope) >= self.sp.hilly_slope_deg) if np.isfinite(slope).any() else False

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
        # hard maximum: the plain design span, or the (longer) hilly one on hilly ground; rivers excepted
        limit = self.sp.span_max_m
        if L > self.sp.design_span_m + 1e-3 and not self.is_hilly(p, q):
            limit = self.sp.design_span_m
        if L > limit + 1e-3:
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
    design = sp.design_span_m
    hilly_leg = bool(sp.hilly_design_span_m) and chk.is_hilly(p, q)
    if hilly_leg:
        design = sp.hilly_design_span_m
    n0 = max(1, math.ceil(d / design - 1e-9), math.ceil(d / sp.span_max_m - 1e-9))
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
        if hilly_leg and span > sp.design_span_m + 1e-3 and n > 1 and \
                not all(chk.is_hilly(a, b) for a, b in zip(seq[:-1], seq[1:])):
            continue      # a span longer than the plain value must itself be on hilly ground
        good = True
        for a, b in zip(seq[:-1], seq[1:]):
            ok, _ = chk.span_crossings(a, b)
            if not ok:
                good = False
                break
        if good:
            return pts
    return None


def _tower_site_rs(comp: Compiled, t: str, px: int, py: int) -> float:
    """Exact per-tower extras of type t at a raster cell: foundation extra over a normal one
    (soil multiplier − 1) and tower-base compensation on the type's base area (Route Cost Parameters 3)."""
    tw = comp.costs["towers"][t]
    return (tw["fdn_rs"] * (float(comp.soil_mult[py, px]) - 1.0)
            + tw["area_m2"] * float(comp.land_rate_m2[py, px]) * comp.costs["land"]["tower_base_pct"] / 100.0)


def _chain_cost(comp: Compiled, sp: SpotParams, pts, inner_types) -> float:
    """Approximate exact cost of a tower chain (frame coords): inner towers (all-in price by type,
    soil extra, base land) + spans (line ₹/m + RoW strip density incl. forest)."""
    f = comp.frame
    c = 0.0
    for (x, y), t in zip(pts[1:-1], inner_types):
        px = int(np.clip((x - f.x0) / f.res_m, 0, f.width - 1))
        py = int(np.clip((f.y_top - y) / f.res_m, 0, f.height - 1))
        c += sp.tower_cost_rs[t] + _tower_site_rs(comp, t, px, py)
    width = 2 * comp.half_width_m
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(L // (f.res_m / 2)) + 1)
        t = np.linspace(0, 1, n)
        fx = x0 + t * (x1 - x0)
        fy = y0 + t * (y1 - y0)
        dens = 0.0
        for off in (-comp.half_width_m / 2, 0.0, comp.half_width_m / 2):     # three lanes across the RoW
            col = np.clip(((fx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
            row = np.clip(((f.y_top - (fy + off)) / f.res_m).astype(int), 0, f.height - 1)
            dens += float(comp.strip_density[row, col].mean()) / 3.0
        c += L * (comp.conductor_rs_per_m + dens * width)
    return c


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
            # not dearer than the grid towers it replaces (their types from their own deflections)
            seg = grid_pts[i:j + 1]
            gtypes = [tower_type_for(_deflection(seg[m - 1], seg[m], seg[m + 1])[0], sp.type_bands) or "D60"
                      for m in range(1, len(seg) - 1)]
            if _chain_cost(chk.c, sp, [grid_pts[i]] + inner + [q], ["A"] * len(inner)) > \
                    _chain_cost(chk.c, sp, seg, gtypes) * (1.0 + sp.pull_cost_tol) + 1.0:
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
    added = set()
    if dem is not None and dem.ok:
        pts = _split_hilly(chk, pts, sp)
        pts, added = _respot_for_clearance(chk, pts, sp)
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
        s = Span(i, i + 1, L, cr, hilly=(not gantry_span) and chk.is_hilly(pts[i], pts[i + 1]))
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
            if c["rule"] in sp.max_dev_adjacent_power_deg:
                lim = sp.max_dev_adjacent_power_deg[c["rule"]]
                for ti in (s.a - 1, s.b + 1):
                    if 0 < ti < len(towers) - 1 and towers[ti].deflection_deg > lim:
                        towers[ti].warnings.append(f"power_adjacent_deviation:{c['rule']}")
    for i in added:
        towers[i].reasons.append("clearance")
    # hilly ground: every tower on a hilly span is a tension tower, B1 at least (Vivek, 7 Oct 2026)
    for s in spans:
        if s.hilly:
            for ti in (s.a, s.b):
                t = towers[ti]
                if t.type != "G" and t.type == "A":
                    t.type = "B1"
                    t.reasons.append("hilly")
    # section points (hilly interval once the stretch since the last tension tower has a hilly span)
    run = 0.0
    hilly_run = False
    for i in range(1, len(towers)):
        run += spans[i - 1].length_m
        hilly_run = hilly_run or spans[i - 1].hilly
        t = towers[i]
        if t.type != "A":
            run = 0.0
            hilly_run = False
            continue
        nxt = spans[i].length_m if i < len(spans) else 0
        limit = sp.hilly_section_km if (sp.hilly_section_km and (hilly_run or (i < len(spans) and spans[i].hilly))) \
            else sp.section_km
        if run + nxt > limit * 1000.0:
            t.type = "B1"
            t.reasons.append("section_hilly" if limit != sp.section_km else "section")
            run = 0.0
            hilly_run = False
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


def _span_clearance(chk: Checker, p, q, sp: SpotParams, ext_a=0.0, ext_b=0.0):
    """(min clearance m, t at the worst point) of the conductor over the DEM for a straight span."""
    pr = chk.profile(p, q, step=15.0)
    if pr is None:
        return None, None
    t, ground, L = pr
    if not np.isfinite(ground[0]) or not np.isfinite(ground[-1]):
        return None, None
    sag = sp.sag_at_design_m * (L / sp.design_span_m) ** 2
    za = ground[0] + sp.lowest_attachment_m + ext_a
    zb = ground[-1] + sp.lowest_attachment_m + ext_b
    c = (1 - t) * za + t * zb - 4 * sag * t * (1 - t) - ground
    if not np.isfinite(c).any():
        return None, None
    k = int(np.nanargmin(c))
    return float(c[k]), float(t[k])


def _split_hilly(chk: Checker, pts, sp: SpotParams):
    """Hilly spans longer than the hilly design span are split into equal spans (each new suspension
    tower nudged along the span if its spot is not allowed). Gantry spans and long single crossing
    spans (over span_max) are left alone."""
    if not sp.hilly_design_span_m:
        return pts
    out = [pts[0]]
    n = len(pts)
    for i in range(n - 1):
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if 0 < i < n - 2 and sp.hilly_design_span_m < L <= sp.span_max_m + 1e-3 and chk.is_hilly(p, q):
            k = math.ceil(L / sp.hilly_design_span_m - 1e-9)
            for j in range(1, k):
                base = j / k
                for dt in (0.0, 0.03, -0.03, 0.06, -0.06):
                    tt = base + dt
                    x, y = p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1])
                    if chk.tower_ok(x, y, angle_tower=False):
                        out.append((x, y))
                        break
        out.append(q)
    return out


def _respot_for_clearance(chk: Checker, pts, sp: SpotParams, max_new: int = 200):
    """Insert suspension towers where a span cannot clear the ground even with the +18 m extension
    on both towers: at the worst point (a ridge), or the nearest allowed spot along the span.
    Gantry spans and long single crossing spans are left alone. Returns (points, inserted indices)."""
    pts = list(pts)
    flags = [False] * len(pts)
    i, new = 1, 0                      # span pts[i] -> pts[i+1]; skip the gantry spans
    while i < len(pts) - 2 and new < max_new:
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        c, tw = _span_clearance(chk, p, q, sp, EXTENSION_M, EXTENSION_M)
        if c is None or c >= sp.ground_clearance_m or L < 2 * sp.min_span_m or L > sp.span_max_m + 1e-3:
            i += 1
            continue
        placed = None
        for dt in (0.0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2):
            tt = min(1 - sp.min_span_m / L, max(sp.min_span_m / L, tw + dt))
            x, y = p[0] + tt * (q[0] - p[0]), p[1] + tt * (q[1] - p[1])
            if chk.tower_ok(x, y, angle_tower=False) and chk.span_crossings(p, (x, y))[0] \
                    and chk.span_crossings((x, y), q)[0]:
                placed = (x, y)
                break
        if placed is None:
            i += 1                     # nothing allowed on this span: reported as a clearance warning
            continue
        pts.insert(i + 1, placed)
        flags.insert(i + 1, True)
        new += 1                       # re-check p -> new tower before moving on
    return pts, {k for k, f in enumerate(flags) if f}


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
    if c < sp.ground_clearance_m and s.length_m > sp.span_max_m + 1e-3:
        # long crossing span: both crossing towers are raised by what the sag needs (6 m steps)
        need = sp.ground_clearance_m - c
        ext = math.ceil(need / 6.0) * 6.0
        for tw in (ta, tb):
            tw.extension_m = max(tw.extension_m, ext)
            if "crossing_tower" not in tw.reasons:
                tw.reasons.append("crossing_tower")
        c = min_clear()
    if c < sp.ground_clearance_m:
        low = ta if (ta.ground_z + ta.extension_m) <= (tb.ground_z + tb.extension_m) else tb
        high = tb if low is ta else ta
        if low.extension_m < EXTENSION_M:
            low.extension_m = EXTENSION_M
            c = min_clear()
        if c < sp.ground_clearance_m and high.extension_m < EXTENSION_M:
            high.extension_m = EXTENSION_M        # both towers extended (the re-spotting assumed so)
            c = min_clear()
        if c < sp.ground_clearance_m:
            ta.warnings.append("ground_clearance")
    s.min_clearance_m = round(c, 1)


def route_cost(comp: Compiled, towers: list[Tower], spans: list[Span], sp: SpotParams):
    """Route cost, Route Cost Parameters 3 (7 Oct 2026): towers at their all-in price by type
    (normal foundation included); foundation = soil extra only; line ₹/km × length; RoW corridor
    compensation + tower-base land; forest NPV + CA + plans + reserve-forest land (× PA multiplier).
    Body extensions at extension_pct of the tower's price per +18 m. No crossing, earthing, crop,
    stamp or diversion items."""
    costs = comp.costs
    f = comp.frame
    base_pct = costs["land"]["tower_base_pct"] / 100.0
    ext_pct = float(costs["tower"]["extension_pct"]) / 100.0
    c_tower = c_found = c_land = 0.0
    for t in towers:
        if t.type == "G":
            continue
        px = int(np.clip((t.x - f.x0) / f.res_m, 0, f.width - 1))
        py = int(np.clip((f.y_top - t.y) / f.res_m, 0, f.height - 1))
        tw = costs["towers"][t.type]
        # the extension rate is quoted per +18 m on the tower's own price; taller towers pro rata
        c_tower += sp.tower_cost_rs[t.type] * (1.0 + ext_pct * t.extension_m / EXTENSION_M)
        c_found += tw["fdn_rs"] * (float(comp.soil_mult[py, px]) - 1.0)
        c_land += tw["area_m2"] * float(comp.land_rate_m2[py, px]) * base_pct
    length = sum(s.length_m for s in spans)
    c_cond = length * comp.conductor_rs_per_m

    # RoW corridor statistics on the raster
    line = LineString([(t.x, t.y) for t in towers])
    corridor = line.buffer(comp.half_width_m, cap_style="flat", join_style="mitre")
    utm = _frame_geom_to_utm(f, corridor)
    mask = rfeatures.rasterize([(utm, 1)], out_shape=comp.cost_class.shape, transform=f.transform,
                               fill=0, dtype="uint8").astype(bool)
    # raster cells scaled to the exact corridor area: a corridor along the grid otherwise counts whole
    # cells (67 m at 765 kV becomes 80 m on the 20 m grid) and is overstated by up to ~19 %
    n_cells = int(mask.sum())
    cell = f.res_m ** 2 * (corridor.area / (n_cells * f.res_m ** 2) if n_cells else 1.0)
    areas = {k: float((mask & (comp.cost_class == v)).sum()) * cell / 1e4 for k, v in COST_CLASS_CODES.items()}
    # corridor ₹ from the compiled strip density (same figures the search used, incl. the PA multiplier)
    fmask = mask & (comp.cost_class == COST_CLASS_CODES["forest"])
    c_forest = float(comp.strip_density[fmask].sum()) * cell
    c_row = float(comp.strip_density[mask & ~fmask].sum()) * cell
    cost = {
        "towers": c_tower, "foundation": c_found, "conductor": c_cond,
        "land_row": c_row + c_land, "forest": c_forest,
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
    chk.dem = dem
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
