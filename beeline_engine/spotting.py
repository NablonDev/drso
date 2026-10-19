"""Tower spotting on the solved alignment, and the route's true cost.

1. String-pulling: replace runs of grid towers by straight legs between angle points, as long
   as every leg passes the exact (vector) conductor checks, its towers can be spotted, and the
   straight leg is not dearer than the grid towers it replaces (towers, foundations, land, RoW
   strip incl. forest): a cost-blind straightening cut corners through forest and rock and made
   a hill route ₹14 Cr dearer than the search had priced it (6 Oct 2026).
2. Spot suspension towers along each leg at equal spans (design span or shorter; up to the longest span
   allowed where that is cheaper at the search charge per metre over the design span, 18 Oct 2026).
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
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
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
    cost_polish: bool = True               # final single-tower cost descent (14 Oct 2026)
    raster_prefilter: bool = True         # skip the vector tower rules in clearly free cells (11 Oct 2026)
    gantry_min_span_m: float = 0.0        # gantry to first / last tower, at least (Vivek, 13 Oct 2026)
    dead_end_a: tuple | None = None       # fixed dead-end positions (frame coords), square to the substation face (19 Oct 2026)
    dead_end_b: tuple | None = None
    refine_crossings: bool = True         # fine local re-optimisation round power-line crossings (14 Oct 2026)
    refine_step_m: float = 10.0
    # hilly terrain (rules design_span.hilly_m, section_interval.hilly_km; engine hilly_slope_deg)
    hilly_design_span_m: float | None = None
    hilly_section_km: float | None = None
    hilly_slope_deg: float | None = None
    hilly_rise_m: float | None = None      # hilly span: ground rises more than this between its towers (Route_Rules 3)
    tsd: dict | None = None                # AESL Tower Spotting Data tables for this voltage (engine.yaml tsd)
    approach_length_m: float = 0.0         # approach section within this of each gantry (CEA Schedule VII)
    approach_span_m: float | None = None
    proximity_discount: float = 0.0         # search-only discount near existing lines (string pulling)
    proximity_tower_discount: float = 0.0   # the same on towers standing near them (Vivek, 13 Oct 2026)
    angle_point_rs: float = 0.0             # search-only surcharge on every tower that is not an A (what-if, 15 Oct 2026)
    hilly_surcharge_rs_per_m: float = 0.0   # search-only hill avoidance (Vivek, 14 Oct 2026), string pulling too
    hilly_tower_surcharge_rs: float = 0.0
    extension_pct: float = 9.4              # body extension price, % of the tower per +18 m
    sag_ref_span_m: float | None = None     # span the sag figure belongs to (None = design span)
    section_spans: int | None = None       # TSD D: ≤ 15 spans between section points (plain)
    hilly_section_spans: int | None = None # ≤ 10 spans (hilly)
    pull_cost_tol: float = 0.01          # a straightened leg may cost at most this much more than the grid towers it replaces
    # longest span (Vivek, 18 Oct 2026): ordinary spans on plain ground up to plain_span_max_m (None = design_span_m;
    # hilly ground up to span_max_m); each span over design_span_m pays excess_span_rs_per_m per metre, search only
    # (chain costs and the leg programme, never route_cost), counted up to span_max_m
    plain_span_max_m: float | None = None
    excess_span_rs_per_m: float = 0.0


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


def _plain_max(sp: SpotParams) -> float:
    """Longest ordinary span on plain ground (rule design_span.max_span_m, 765 kV 450 m; 18 Oct 2026)."""
    return float(sp.plain_span_max_m) if sp.plain_span_max_m else float(sp.design_span_m)


def _excess_rs(sp: SpotParams, L: float) -> float:
    """Search-only charge of a span over the plain design span (Vivek, 18 Oct 2026), as the search prices it."""
    from .solver import excess_span_rs
    return excess_span_rs(L, sp.design_span_m, sp.span_max_m, sp.excess_span_rs_per_m)


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
        # Raster pre-filter for the exact tower checks (11 Oct 2026): a cell two cells clear of every
        # forbidden cell is more than the rule distance from any feature (the rule rasters are built from
        # cell-centre distances, so a point can be up to ~28 m nearer a feature than its cell says; two
        # 20 m cells cover that). Only "clearly free" skips the vector rules; anything else is checked.
        self._clear = ~ndimage.binary_dilation(comp.tower_forbidden, iterations=2)
        self._clear_angle = ~ndimage.binary_dilation(comp.angle_forbidden, iterations=2)
        self._verify = os.environ.get("BEELINE_VERIFY_PREFILTER") == "1"
        # Route-neutral speed-ups (16 Oct 2026; measured on BHU-LAK-0224: identical towers and cost, route building
        # 1.5x faster): one mask of every crossing / parallel raster, so a span that meets none of them skips the
        # per-rule sampling; memoised tower, conductor and hilly checks (the leg spotting and the repairs test the same
        # spots and spans many times over). Results are exactly those of the unmemoised checks.
        self._any_cross = None
        if comp.crossings:
            m = np.zeros(comp.tower_forbidden.shape, bool)
            for cr in comp.crossings:
                m |= ~np.isnan(cr.angle)
            self._any_cross = m
        self._any_par = None
        if comp.parallels:
            m = np.zeros(comp.tower_forbidden.shape, bool)
            for pr in comp.parallels:
                m |= ~np.isnan(pr.angle)
            self._any_par = m
        self._tower_memo: dict = {}
        self._cond_memo: dict = {}
        self._hilly_memo: dict = {}
        # (A combined shortlist STRtree over every rule's buffered features was tried on 11 Oct 2026 and
        # dropped: ~900 k buffers took 500 s to build per compile and the line queries were slower than
        # the per-rule dwithin queries.)

    @property
    def dem(self):
        return self.__dict__.get("_dem")

    @dem.setter
    def dem(self, v):
        # the hilly memo depends on the DEM: setting it (spot() and EditContext do, right after construction)
        # clears the memo, so a caller that sets the DEM late still gets the DEM's answer
        self.__dict__["_dem"] = v
        self.__dict__.setdefault("_hilly_memo", {}).clear()

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
        key = (round(p[0], 3), round(p[1], 3), round(q[0], 3), round(q[1], 3))
        hit = self._hilly_memo.get(key)
        if hit is None:
            hit = self._is_hilly(p, q)
            if len(self._hilly_memo) > 1_000_000:
                self._hilly_memo.clear()
            self._hilly_memo[key] = hit
        return hit

    def _is_hilly(self, p, q) -> bool:
        if self.sp.hilly_rise_m is not None:
            # Route_Rules item 3 (8 Oct 2026): the ground rises more than hilly_rise_m between the towers
            if self.dem is None or not self.dem.ok:
                return False
            lon, lat = self.c.frame.frame_to_lonlat(np.array([p[0], q[0]]), np.array([p[1], q[1]]))
            z = self.dem.sample(lon, lat)
            return bool(np.isfinite(z).all() and abs(float(z[1] - z[0])) > self.sp.hilly_rise_m)
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

    def in_approach(self, p, q) -> bool:
        """The span's midpoint lies in an approach section (within approach_length_m of a gantry)."""
        if not self.sp.approach_length_m:
            return False
        mx, my = (p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0
        L = self.c.frame.length_m
        return math.hypot(mx, my) <= self.sp.approach_length_m or math.hypot(mx - L, my) <= self.sp.approach_length_m

    def in_go_zone(self, p: Point) -> bool:
        return self.go.contains(p)

    def tower_ok(self, x, y, angle_tower: bool) -> bool:
        key = (round(float(x), 3), round(float(y), 3), bool(angle_tower))
        hit = self._tower_memo.get(key)
        if hit is None:
            hit = self._tower_ok(x, y, angle_tower)
            if len(self._tower_memo) > 2_000_000:
                self._tower_memo.clear()
            self._tower_memo[key] = hit
        return hit

    def _tower_ok(self, x, y, angle_tower: bool) -> bool:
        if abs(y) > self.c.frame.half_width_m + 1e-6:
            return False
        p = Point(x, y)
        if self.in_go_zone(p):
            # go zone: only a substation / existing tower overlap stops a tower (9 Oct 2026)
            if self.yards is not None and self.yards.contains(p):
                return False
            return not any(r.hits(p) for r in self.c.go_overlap)
        f = self.c.frame
        col = int(np.clip((x - f.x0) / f.res_m, 0, f.width - 1))
        row = int(np.clip((f.y_top - y) / f.res_m, 0, f.height - 1))
        clear = self.sp.raster_prefilter and bool(self._clear[row, col]) and (not angle_tower or bool(self._clear_angle[row, col]))
        if clear and not self._verify:
            return True
        ok = not any(r.hits(p) for r in self.tower_rules) and not (angle_tower and any(r.hits(p) for r in self.angle_rules))
        if clear and not ok:
            log.warning("prefilter disagreement at (%.0f, %.0f): raster clear, vector hit", x, y)
        return ok

    def _rule_intersects(self, rule_id: str, p, q):
        """True / False when the rule's vector features do / do not meet the span; None without features."""
        trees = self.__dict__.setdefault("_rule_trees", {})
        if rule_id not in trees:
            from shapely import STRtree
            geoms = []
            for r in list(self.tower_rules) + list(self.cond_rules):
                if r.rule_id == rule_id:
                    geoms.extend(list(r.geoms))
            trees[rule_id] = (STRtree(geoms), geoms) if geoms else None
        t = trees[rule_id]
        if t is None:
            return None
        tree, geoms = t
        seg = LineString([p, q])
        return any(geoms[j].intersects(seg) for j in tree.query(seg))

    def conductor_ok(self, p, q) -> bool:
        key = (round(p[0], 3), round(p[1], 3), round(q[0], 3), round(q[1], 3))
        hit = self._cond_memo.get(key)
        if hit is None:
            hit = self._conductor_ok(p, q)
            if len(self._cond_memo) > 1_000_000:
                self._cond_memo.clear()
            self._cond_memo[key] = hit
        return hit

    def _conductor_ok(self, p, q) -> bool:
        line = LineString([p, q])
        # go zone: crossings of >= 220 kV lines near their towers stay banned (clipped to the zone)
        L0 = self.c.frame.length_m
        gantry = any(abs(pt[1]) < 1e-6 and (abs(pt[0]) < 1e-6 or abs(pt[0] - L0) < 1e-6) for pt in (p, q))
        if self.c.go_conductor and any(r.hits(line) for r in self.c.go_conductor
                                       if not (gantry and r.rule_id == "terminal_substation")):
            return False
        if line.intersects(self.go):
            line = line.difference(self.go)
            if line.is_empty:
                return True
        return not any(r.hits(line) for r in self.cond_rules)

    def _sample_idx(self, p, q, step=None):
        f = self.c.frame
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(2, int(math.ceil(L / (step or f.res_m / 2))) + 1)
        t = np.linspace(0, 1, n)
        fx = p[0] + t * (q[0] - p[0])
        fy = p[1] + t * (q[1] - p[1])
        col = np.clip(((fx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
        row = np.clip(((f.y_top - fy) / f.res_m).astype(int), 0, f.height - 1)
        return row, col, L / (n - 1)

    def _sample(self, arr, p, q, step=None):
        row, col, st = self._sample_idx(p, q, step)
        return arr[row, col], st

    def span_crossings(self, p, q) -> tuple[bool, list[dict]]:
        """(valid, crossings) for one span, by the same rules the solver applied (memoised per span;
        callers get copies of the items)."""
        key = (round(p[0], 3), round(p[1], 3), round(q[0], 3), round(q[1], 3))
        cache = self.__dict__.setdefault("_sc_cache", {})
        hit = cache.get(key)
        if hit is None:
            hit = self._span_crossings(p, q)
            if len(cache) > 400000:
                cache.clear()
            cache[key] = hit
        return hit[0], [dict(x) for x in hit[1]]

    def _span_crossings(self, p, q) -> tuple[bool, list[dict]]:
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        head = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
        out, ok = [], True
        row_, col_, step_ = self._sample_idx(p, q)               # one sample index for every raster of this span
        any_cross = self._any_cross is None or bool(self._any_cross[row_, col_].any())
        for cr in (self.c.crossings if any_cross else ()):
            a, step = cr.angle[row_, col_], step_
            hit = ~np.isnan(a)
            if not hit.any():
                continue
            first = int(np.argmax(hit))
            last = len(hit) - 1 - int(np.argmax(hit[::-1]))
            fa = float(a[first])
            d = (head - fa) % 180.0
            acute = min(d, 180.0 - d)
            # the crossed feature sits at the middle of its hit samples; distances from each tower to it
            mid = (first + last) / 2.0 * step
            # perpendicular distance from each crossing tower to the wire (Vivek, 14 Oct 2026: to the wire itself,
            # not to the edge of its raster band): the wire is the middle of the first / last run of hit samples
            # (one span may cross two lines side by side); along-span distance x sin(crossing angle)
            idx = np.flatnonzero(hit)
            brk = np.flatnonzero(np.diff(idx) > 2)
            r0_end = idx[brk[0]] if len(brk) else idx[-1]
            rl_start = idx[brk[-1] + 1] if len(brk) else idx[0]
            along_a = (first + r0_end) / 2.0 * step
            along_b = L - (rl_start + last) / 2.0 * step
            sin_a = max(math.sin(math.radians(acute)), 0.05)
            sa, sb = along_a * sin_a, along_b * sin_a
            null_exact = None
            if cr.rule_id in ("power_line", "power_line_lt"):
                ex = _wire_sides(self, cr.rule_id, p, q)          # exact, from the wire geometry
                if ex is not None:
                    sa, sb, along_a, along_b, null_exact, ac_min = ex
                    if ac_min is not None:
                        acute = min(acute, ac_min)                # the tightest angle over every wire crossed
            item = {"rule": cr.rule_id, "name": cr.name, "angle_deg": round(acute, 1), "span_m": round(L, 1),
                    "min_type": cr.min_type, "cost_rs": cr.cost_rs, "major": False,
                    "side_a_m": round(sa, 1), "side_b_m": round(sb, 1),
                    "along_a_m": round(along_a, 1), "along_b_m": round(along_b, 1)}
            if getattr(cr, "extension_m", 0):
                item["extension_m"] = cr.extension_m
            if getattr(cr, "long_span", False):
                item["long_span"] = True
            if getattr(cr, "max_dev_deg", None) is not None:
                item["max_dev_deg"] = float(cr.max_dev_deg)
            # crossing preferences (Vivek, 13 Oct 2026): angle and, for power lines, the null point
            pw = getattr(cr, "pref_window", None)
            item["angle_preferred"] = pw is None or acute >= pw[0] - 1e-3
            pm = getattr(cr, "pref_mask", None)
            if pm is not None:
                mk = pm[row_, col_]
                item["point_preferred"] = bool(mk[(first + last) // 2])
            no = getattr(cr, "null_offset_wide", None)
            if no is not None:
                if null_exact is not None:
                    # exact from the wire and its towers (14 Oct 2026: the raster copy read 26 % for an 8.7 % crossing)
                    item["null_offset_pct"] = round(null_exact * 100, 1)
                    item["point_preferred"] = null_exact <= float(getattr(cr, "null_pref", 0.05)) + 1e-9
                    if null_exact > float(getattr(cr, "null_allowed", 0.2)) + 0.005:
                        ok = False
                        item["violation"] = "null_point"
                else:
                    ov = no[row_, col_]
                    v = ov[(first + last) // 2]
                    if np.isfinite(v):
                        item["null_offset_pct"] = round(float(v) * 100, 1)   # distance from the null point, % of their span
            if cr.window is not None and acute < cr.window[0] - 1e-3:
                ok = False
                item["violation"] = "angle"
            if cr.max_span_m is not None and L > cr.max_span_m + 1e-3:
                ok = False
                item["violation"] = "span"
            band = getattr(cr, "span_band", None)
            if band is not None:
                # SH (Vivek, 16 Oct 2026): a crossing span outside the band is allowed with an approval record
                item["span_band_m"] = [band[0], band[1]]
                if L < band[0] - 1e-3 or L > band[1] + 1e-3:
                    item["approval"] = "span_band"
                    item["band_penalty_rs"] = float(getattr(cr, "band_penalty_rs", 0.0) or 0.0)
            side = getattr(cr, "side_m", None)
            if side is not None:
                lo, hi = side
                item["side_window_m"] = [lo, hi]
                item["side_relax_rs_per_m"] = float(getattr(cr, "side_relax_rs_per_m", 0.0) or 0.0)
                if sa < lo - 1e-3 or sb < lo - 1e-3:
                    ok = False
                    item["violation"] = "side"
                elif sa > hi + 1e-3 or sb > hi + 1e-3:
                    if getattr(cr, "side_relax_rs_per_m", 0) > 0:
                        item["side_relaxed_m"] = round(max(0.0, sa - hi) + max(0.0, sb - hi), 1)   # 14 Oct 2026
                    else:
                        ok = False
                        item["violation"] = "side"
            if cr.width_raster is not None and cr.major_threshold_m is not None:
                w, st = cr.width_raster[row_, col_], step_
                width = float(w.sum()) * st
                item["width_m"] = round(width, 1)
                if width >= cr.major_threshold_m:
                    item.update(major=True, min_type=cr.major_type, cost_rs=cr.major_cost_rs)
            out.append(item)
        # hard maximum: the longest plain span (18 Oct 2026: max_span_m, over the design span at a search charge),
        # or the (longer) hilly one on hilly ground; rivers excepted
        limit = self.sp.span_max_m
        if self.sp.approach_span_m is not None and self.in_approach(p, q):
            limit = min(limit, self.sp.approach_span_m)
        if L > _plain_max(self.sp) + 1e-3 and not self.is_hilly(p, q):
            limit = min(limit, _plain_max(self.sp))
        if L > limit + 1e-3:
            if L > self.sp.river_max_span_m + 1e-3 or not any(c.get("long_span") and c.get("geom_crossed") is not False for c in out):
                ok = False
        # a crossing is a crossing only where the feature's geometry meets the span (audit, 14 Oct 2026: the
        # raster band of 1.5 cells round a canal / pipeline made a near-miss a "crossing", which exempted a
        # parallel run and excused a 498 m span); rules without vector features keep the raster's word
        for c in out:
            c["geom_crossed"] = self._rule_intersects(c["rule"], p, q)
            if c.get("approval") and c["geom_crossed"] is False:
                c.pop("approval", None)          # a near miss of the raster band is no crossing
                c.pop("band_penalty_rs", None)
        crossed_rules = {c.get("rule") for c in out if c.get("geom_crossed") is not False}
        any_par = self._any_par is None or bool(self._any_par[row_, col_].any())
        for pr in (self.c.parallels if any_par else ()):
            if getattr(pr, "crossing_exempt", False) and pr.rule_id in crossed_rules:
                continue                    # the crossing span itself is not a parallel run (canal)
            a = pr.angle[row_, col_]
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


def _spot_leg(chk: Checker, p, q, sp: SpotParams, exact: bool = False):
    """Intermediate tower positions on a straight leg p->q, or None if no spotting works.

    exact=False (string pulling, repairs: a feasibility probe): even spacing, else uneven with the same
    tower count — the old, fast behaviour. exact=True (the final re-spot of every straight run, merges,
    the best straighten candidates): Vivek, 14 Oct 2026, no preference for even spacing — the towers stand
    wherever the checks allow at the positions that make the leg cheapest (suspension towers, site extras,
    the body extensions each span's clearance needs, the B1 upgrade on a hilly span): a 1-D programme over
    10 m positions, spans between the minimum and the longest span (over the design span at the search charge,
    18 Oct 2026); a span the crossing checks reject is struck and the programme repeated. A single long crossing
    span is tried first either way."""
    if not chk.conductor_ok(p, q):
        return None
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    design = sp.design_span_m
    leg_max = _plain_max(sp)
    hilly_leg = bool(sp.hilly_design_span_m) and chk.is_hilly(p, q)
    if hilly_leg:
        design = sp.hilly_design_span_m
        leg_max = sp.span_max_m
    design = min(design, leg_max)
    ok1, items1 = chk.span_crossings(p, q)
    if ok1 and d <= design + 1e-6:
        return []
    if ok1 and d <= sp.river_max_span_m and any(it.get("long_span") for it in items1):
        return []                             # one long span over the crossing
    if exact:
        return _leg_dp(chk, p, q, sp, design, hilly_leg)
    tA = float(sp.tower_cost_rs.get("A", 0.0) or 0.0)
    if ok1 and d <= leg_max + 1e-6 and _excess_rs(sp, d) <= tA:
        return []                             # one span over the design span: its search charge is below a tower
    # tower counts from the design span up, and (18 Oct 2026) fewer, longer spans up to the longest span, tried in
    # the order of their price — suspension towers plus the search charge of the even spans
    n0 = max(1, math.ceil(d / design - 1e-9))
    n_lo = max(1, math.ceil(d / leg_max - 1e-9))
    counts = sorted(range(min(n_lo, n0), n0 + 5), key=lambda n: ((n - 1) * tA + n * _excess_rs(sp, d / n), n))
    for n in counts:
        span = d / n
        if span < sp.min_span_m and n > 1:
            continue
        pts = [(p[0] + (q[0] - p[0]) * i / n, p[1] + (q[1] - p[1]) * i / n) for i in range(1, n)]
        if _even_leg_ok(chk, p, q, pts, span, n, hilly_leg, sp):
            return pts
        if 1 < n <= min(n0 + 1, UNEVEN_MAX_SPANS):
            un = _uneven_leg(chk, p, q, n, design if n >= n0 else leg_max, sp)
            if un is not None:
                return un
    return None


def _even_leg_ok(chk: Checker, p, q, pts, span: float, n: int, hilly_leg: bool, sp: SpotParams) -> bool:
    if not all(chk.tower_ok(x, y, angle_tower=False) for x, y in pts):
        return False
    seq = [p] + pts + [q]
    if sp.approach_span_m is not None and span > sp.approach_span_m + 1e-3 and \
            any(chk.in_approach(a, b) for a, b in zip(seq[:-1], seq[1:])):
        return False      # approach section: 250 m spans
    if hilly_leg and span > _plain_max(sp) + 1e-3 and n > 1 and \
            not all(chk.is_hilly(a, b) for a, b in zip(seq[:-1], seq[1:])):
        return False      # a span longer than the plain value must itself be on hilly ground
    return all(chk.span_crossings(a, b)[0] for a, b in zip(seq[:-1], seq[1:]))


UNEVEN_MAX_SPANS = 8          # uneven spacing is tried on legs of up to this many spans


def _uneven_leg(chk: Checker, p, q, n: int, design: float, sp: SpotParams, step: float = 10.0):
    """n spans on the straight leg p->q with towers where they may stand, spans as even as possible.
    Every span within ±15 % of the even span (and the span cap): two neighbouring spans then keep
    the TSD rule one span ≤ 60 % of their sum. Minimises the squared deviation from even spacing;
    a span the exact checks reject is struck and the search repeated (a few times at most)."""
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    ideal = d / n
    lo, hi = max(sp.min_span_m, ideal * 0.85), min(design, ideal * 1.15)
    if lo > hi:
        return None
    m = int(d // step)
    xs = [i * step for i in range(1, m + 1) if i * step < d - lo + 1e-6]
    pos = [0.0] + xs + [d]
    at = lambda t: (p[0] + (q[0] - p[0]) * t / d, p[1] + (q[1] - p[1]) * t / d)
    _ok: dict = {0: True, len(pos) - 1: True}

    def ok_at(j):                         # tower_ok, evaluated on first use only
        if j not in _ok:
            _ok[j] = chk.tower_ok(*at(pos[j]), angle_tower=False)
        return _ok[j]
    bad: set = set()
    for _ in range(6):
        INF_ = float("inf")
        best = [[INF_] * len(pos) for _ in range(n + 1)]
        arg = [[-1] * len(pos) for _ in range(n + 1)]
        best[0][0] = 0.0
        for k in range(1, n + 1):
            for j in range(1, len(pos)):
                # tower k can only stand between k·lo and k·hi from p, and (n-k)·lo..(n-k)·hi from q
                if pos[j] < k * lo - 1e-6 or pos[j] > k * hi + 1e-6 or \
                        d - pos[j] < (n - k) * lo - 1e-6 or d - pos[j] > (n - k) * hi + 1e-6:
                    continue
                if (k < n and j == len(pos) - 1) or (k == n and j != len(pos) - 1) or not ok_at(j):
                    continue
                for i in range(j - 1, -1, -1):
                    L = pos[j] - pos[i]
                    if L > hi + 1e-6:
                        break
                    if L < lo - 1e-6 or best[k - 1][i] == INF_ or (i, j) in bad:
                        continue
                    c = best[k - 1][i] + (L - ideal) ** 2
                    if c < best[k][j]:
                        best[k][j], arg[k][j] = c, i
        if best[n][len(pos) - 1] == INF_:
            return None
        idx, j = [], len(pos) - 1
        for k in range(n, 0, -1):
            idx.append(j)
            j = arg[k][j]
        idx.append(0)
        idx = idx[::-1]
        seq = [at(pos[i]) for i in idx]
        seq[0], seq[-1] = p, q
        failed = None
        for a_i, b_i, a, b in zip(idx[:-1], idx[1:], seq[:-1], seq[1:]):
            if not chk.span_crossings(a, b)[0]:
                failed = (a_i, b_i)
                break
        if failed is None:
            return seq[1:-1]
        bad.add(failed)
    return None




def _leg_model(chk: Checker, p, q, sp: SpotParams, design: float, hilly_leg: bool, pos):
    """Cost pieces of a straight leg p->q with candidate tower positions `pos` (m from p, 0 and d included):
    tower_rs(j) = a suspension tower at pos[j] (price + site extras); span_rs(i, j) = the body extensions the
    span's ground clearance needs on both towers (closest step; None = needs a taller type), the B1 upgrade
    when the ground rises more than the hilly limit, the plain-span rule on hilly legs, and (18 Oct 2026) the
    search charge of a span over the plain design span."""
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    pos = np.asarray(pos, float)
    at = lambda t: (p[0] + (q[0] - p[0]) * t / d, p[1] + (q[1] - p[1]) * t / d)
    z = None
    if chk.dem is not None and getattr(chk.dem, "ok", False):
        lon, lat = chk.c.frame.frame_to_lonlat(p[0] + (q[0] - p[0]) * pos / d, p[1] + (q[1] - p[1]) * pos / d)
        z = np.asarray(chk.dem.sample(lon, lat), float)
        if not np.isfinite(z).all():
            z = None
    tA = sp.tower_cost_rs.get("A", 0.0)
    tB = sp.tower_cost_rs.get("B1", tA)
    steps_a = _ext_steps(sp, "A")
    f = chk.c.frame
    _site: dict = {}
    # village-cluster breach band along the leg (Vivek, 18 Oct 2026): a candidate span pays the breach charge
    # when any 10 m sample between its towers lies in the band (prefix counts make each span O(1))
    br_rs = float(getattr(chk.c, "breach_rs", 0.0) or 0.0)
    br_pref = None
    if br_rs > 0 and getattr(chk.c, "breach_band", None) is not None:
        nb = max(2, int(d // 10.0) + 2)
        tb = np.linspace(0.0, d, nb)
        bx = p[0] + (q[0] - p[0]) * tb / d
        by = p[1] + (q[1] - p[1]) * tb / d
        bc = np.clip(((bx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
        brw = np.clip(((f.y_top - by) / f.res_m).astype(int), 0, f.height - 1)
        hitb = chk.c.breach_band[brw, bc].astype(np.int32)
        br_pref = (tb, np.concatenate([[0], np.cumsum(hitb)]))
    # buildings soft zone along the leg (Vivek, 19 Oct 2026): (position along the leg, perpendicular distance)
    # of each building near it; a candidate span is charged for the buildings whose foot point lies on it
    bs_x = []
    if _bld_soft_tree(chk.c) is not None:
        leg_ = LineString([p, q])
        lim_ = float(chk.c.bld_soft_normal_m)
        for jb in _bld_soft_tree(chk.c).query(leg_, predicate="dwithin", distance=lim_):
            g_ = chk.c.bld_soft_geoms[jb][0]
            from shapely.ops import nearest_points as _np_
            pa_, _ = _np_(leg_, g_)
            bs_x.append((leg_.project(pa_), g_))
    # span-band features (SH) the leg crosses: (distance from p, band, penalty) — 16 Oct 2026
    band_x = []
    if _band_index(chk.c):
        leg = LineString([p, q])
        for (lo, hi), pen, tree, lines in _band_index(chk.c):
            for j in tree.query(leg):
                ip = lines[j].intersection(leg)
                for g in getattr(ip, "geoms", [ip]):
                    if not g.is_empty:
                        band_x.append((leg.project(g.centroid if g.geom_type != "Point" else g), lo, hi, pen))

    def tower_rs(j):
        if j not in _site:
            x, y = at(pos[j])
            px = int(np.clip((x - f.x0) / f.res_m, 0, f.width - 1))
            py = int(np.clip((f.y_top - y) / f.res_m, 0, f.height - 1))
            _site[j] = tA + _tower_site_rs(chk.c, "A", px, py)
        return _site[j]

    def span_rs(i, j):
        L = pos[j] - pos[i]
        c = 0.0
        for sx, lo, hi, pen in band_x:
            if pos[i] < sx < pos[j] and not (lo - 1e-3 <= L <= hi + 1e-3):
                c += pen
        if bs_x:
            seg_ = None
            for sx_, g_ in bs_x:
                if pos[i] < sx_ <= pos[j] or (i == 0 and sx_ <= pos[j]):
                    seg_ = seg_ or LineString([at(pos[i]), at(pos[j])])
                    dd_ = g_.distance(seg_)
                    if dd_ < chk.c.bld_soft_normal_m - 1e-6:
                        c += chk.c.bld_soft_a_rs + chk.c.bld_soft_rate_rs_per_m * (chk.c.bld_soft_normal_m - dd_)
        if br_pref is not None:
            tb_, cs_ = br_pref
            ia = int(np.searchsorted(tb_, pos[i], side="left")); ib = int(np.searchsorted(tb_, pos[j], side="right"))
            if cs_[ib] - cs_[ia] > 0:
                c += br_rs
        if z is not None:
            hilly_span = sp.hilly_rise_m is not None and abs(z[j] - z[i]) > sp.hilly_rise_m
            if hilly_leg and L > _plain_max(sp) + 1e-3 and not hilly_span:
                return None                   # a span longer than the plain value must itself be on hilly ground
            sag = sp.sag_at_design_m * (L / (sp.sag_ref_span_m or sp.design_span_m)) ** 2
            t = (pos[i:j + 1] - pos[i]) / L
            line = (1 - t) * (z[i] + sp.lowest_attachment_m) + t * (z[j] + sp.lowest_attachment_m) - 4 * sag * t * (1 - t)
            need = max(0.0, sp.ground_clearance_m - float(np.min(line - z[i:j + 1])))
            ext = nearest_step(need, steps_a) if need > 0 else 0.0
            if ext is None and steps_a and need <= rounding_cover(list(steps_a) + [EXTENSION_M]) + 1e-9:
                ext = max(steps_a)            # the largest A step serves up to halfway to the D step (as the search)
            if ext is None:
                return None
            c += 2 * tA * sp.extension_pct / 100.0 * ext / 18.0
            if hilly_span:
                c += tB - tA
        elif hilly_leg and L > _plain_max(sp) + 1e-3:
            return None
        if sp.approach_span_m is not None and L > sp.approach_span_m + 1e-3 and chk.in_approach(at(pos[i]), at(pos[j])):
            return None
        return c + _excess_rs(sp, L)
    return at, tower_rs, span_rs


def _leg_objective(chk: Checker, sp: SpotParams, seq) -> float:
    """The leg programme's own cost of a given straight tower sequence (first and last are the anchors)."""
    p, q = seq[0], seq[-1]
    pos = [math.hypot(x - p[0], y - p[1]) for x, y in seq]
    hilly_leg = bool(sp.hilly_design_span_m) and chk.is_hilly(p, q)
    design = sp.hilly_design_span_m if hilly_leg else sp.design_span_m
    _, tower_rs, span_rs = _leg_model(chk, p, q, sp, design, hilly_leg, pos)
    tot = 0.0
    for j in range(1, len(pos)):
        sc = span_rs(j - 1, j)
        if sc is None:
            return float("inf")
        tot += sc + (tower_rs(j) if j < len(pos) - 1 else 0.0)
    if sp.tsd and sp.tsd.get("span_sum") and sp.span_max_m > sp.design_span_m:
        # 18 Oct 2026: with spans over 400 m two neighbours can pass the A table (863 m at 0°); the type the
        # TSD check will give that suspension tower is priced here, so a re-spot is kept only when it truly pays
        from .solver import fit_type_for_spans
        tA = sp.tower_cost_rs.get("A", 0.0)
        frac = float(sp.tsd.get("single_span_fraction", 0.6))
        for j in range(1, len(pos) - 1):
            t = fit_type_for_spans(sp.tsd["span_sum"], "A", 0.0, pos[j] - pos[j - 1], pos[j + 1] - pos[j], frac)
            if t is None:
                return float("inf")
            if t != "A":
                tot += sp.tower_cost_rs.get(t, tA) - tA
    return tot


def _leg_dp(chk: Checker, p, q, sp: SpotParams, design: float, hilly_leg: bool, step: float = 10.0):
    d = math.hypot(q[0] - p[0], q[1] - p[1])
    # spans up to the longest span (18 Oct 2026): over the design span each pays the search charge (span_rs)
    lo, hi = sp.min_span_m, (sp.span_max_m if hilly_leg else min(_plain_max(sp), sp.span_max_m))
    if d > hi + 1e-6 and d < 2 * lo - 1e-6:
        return None
    m = int(d // step)
    pos = np.array([0.0] + [i * step for i in range(1, m + 1) if i * step < d - lo + 1e-6 and i * step >= lo - 1e-6] + [d])
    n = len(pos)
    at, tower_rs, span_rs = _leg_model(chk, p, q, sp, design, hilly_leg, pos)
    _tok: dict = {0: True, n - 1: True}

    def tower_ok(j):
        if j not in _tok:
            x, y = at(pos[j])
            _tok[j] = chk.tower_ok(x, y, angle_tower=False)
        return _tok[j]

    bad: set = set()
    for _ in range(8):
        INF_ = float("inf")
        best = np.full(n, INF_)
        arg = np.full(n, -1, int)
        best[0] = 0.0
        for j in range(1, n):
            if j < n - 1 and not tower_ok(j):
                continue
            tc = 0.0 if j == n - 1 else tower_rs(j)
            for i in range(j - 1, -1, -1):
                L = pos[j] - pos[i]
                if L > hi + 1e-6:
                    break
                if L < lo - 1e-6 or best[i] == INF_ or (i, j) in bad:
                    continue
                sc = span_rs(i, j)
                if sc is None:
                    continue
                c = best[i] + sc + tc
                if c < best[j]:
                    best[j], arg[j] = c, i
        if best[n - 1] == INF_:
            return None
        idx, j = [], n - 1
        while j > 0:
            idx.append(j)
            j = int(arg[j])
        idx.append(0)
        idx = idx[::-1]
        seq = [at(pos[i]) for i in idx]
        seq[0], seq[-1] = p, q
        failed = None
        for a_i, b_i, a, b in zip(idx[:-1], idx[1:], seq[:-1], seq[1:]):
            if not chk.span_crossings(a, b)[0]:
                failed = (a_i, b_i)
                break
        if failed is None:
            return seq[1:-1]
        bad.add(failed)
    return None


def _respot_runs(chk: Checker, pts, sp: SpotParams, min_gain_rs: float = 1e5):
    """Final placement (Vivek, 14 Oct 2026: no even spans): every straight run between anchors (vertices,
    crossing towers, dead ends) is spotted again by the leg programme — towers wherever they may stand, at
    the cheapest positions — and kept when the programme's own cost drops and every check passes."""
    pts = list(pts)
    moves, saved = 0, 0.0
    n = len(pts)
    anchors = [k for k in range(1, n - 1) if k in (1, n - 2) or _is_vertex(pts, k) or _is_crossing_tower(chk, pts, k)]
    out = []
    for a, b in zip(anchors[:-1], anchors[1:]):
        if b - a < 2:
            continue
        cur = _leg_objective(chk, sp, pts[a:b + 1])
        if any(not chk.tower_ok(pts[k][0], pts[k][1], angle_tower=False) for k in range(a + 1, b)):
            # a tower of the run fails its exact check (a lattice site the snap could not move, 16 Oct 2026):
            # any legal re-spot of the run replaces it, whatever it costs
            cur = float("inf")
        new = _spot_leg(chk, pts[a], pts[b], sp, exact=True)
        if new is None:
            continue
        cost = _leg_objective(chk, sp, [pts[a]] + list(new) + [pts[b]])
        if cost < cur - min_gain_rs:
            # the current run can price at infinity (its spans fail the programme's own checks): the move is
            # kept, but its saving is not a number to report (15 Oct 2026: Infinity broke the result JSON)
            out.append((a, b, list(new), cur - cost if math.isfinite(cur) else 0.0))
    for a, b, new, gain in reversed(out):            # apply from the end: earlier indices stay valid
        trial = pts[:a + 1] + new + pts[b:]
        if _run_legal(chk, sp, trial, a + 1, a + len(new), fixed=set()) if new else True:
            pts = trial
            moves += 1
            saved += gain
    if moves:
        log.info("re-spot runs: %d runs, ₹%.2f Cr (leg programme prices)", moves, saved / 1e7)
    return pts, saved, moves

def _tower_site_rs(comp: Compiled, t: str, px: int, py: int) -> float:
    """Exact per-tower extras of type t at a raster cell: foundation extra over a normal one
    (soil multiplier − 1) and tower-base compensation on the type's base area (Route Cost Parameters 3)."""
    tw = comp.costs["towers"][t]
    return (tw["fdn_rs"] * (float(comp.soil_mult[py, px]) - 1.0)
            + tw["area_m2"] * float(comp.land_rate_m2[py, px]) * comp.costs["land"]["tower_base_pct"] / 100.0)



def _hill_stats(comp: Compiled, towers) -> dict:
    """Line length and towers on hill ground (compile.attach_hill_area; 14 Oct 2026)."""
    m = getattr(comp, "hill_area", None)
    if m is None:
        return {"hill_km": None, "hill_towers": None}
    f = comp.frame

    def on(x, y):
        return bool(m[int(np.clip((f.y_top - y) / f.res_m, 0, f.height - 1)), int(np.clip((x - f.x0) / f.res_m, 0, f.width - 1))])
    L = 0.0
    for a, b in zip(towers[:-1], towers[1:]):
        d = math.hypot(b.x - a.x, b.y - a.y)
        n = max(2, int(d // 10) + 1)
        k = sum(on(a.x + (b.x - a.x) * i / (n - 1), a.y + (b.y - a.y) * i / (n - 1)) for i in range(n))
        L += d * k / n
    return {"hill_km": round(L / 1000, 3), "hill_towers": sum(1 for t in towers[1:-1] if on(t.x, t.y))}

def _chain_cost(comp: Compiled, sp: SpotParams, pts, inner_types) -> float:
    """Approximate exact cost of a tower chain (frame coords): inner towers (all-in price by type,
    soil extra, base land) + spans (line ₹/m + RoW strip density incl. forest)."""
    f = comp.frame
    c = 0.0
    for k, ((x, y), t) in enumerate(zip(pts[1:-1], inner_types)):
        px = int(np.clip((x - f.x0) / f.res_m, 0, f.width - 1))
        py = int(np.clip((f.y_top - y) / f.res_m, 0, f.height - 1))
        tc = sp.tower_cost_rs[t] + _tower_site_rs(comp, t, px, py)
        if sp.angle_point_rs and t != "A":
            # the surcharge goes with the turn, as in the search (handoff D, 16 Oct 2026): a tower heavy only
            # because a crossing, a section point or hill ground forces it carries none
            d, _ = _deflection(pts[k], pts[k + 1], pts[k + 2])
            if (tower_type_for(d, sp.type_bands) or "D60") != "A":
                tc += sp.angle_point_rs
        if comp.hill_tower_rs and comp.hill_area is not None and comp.hill_area[py, px]:
            tc += comp.hill_tower_rs                 # search-only hill surcharge (14 Oct 2026)
        if sp.proximity_tower_discount > 0 and comp.near_existing is not None and comp.near_existing[py, px]:
            tc *= 1.0 - sp.proximity_tower_discount
        c += tc
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
        span_c = L * (comp.conductor_rs_per_m + dens * width)
        if sp.proximity_discount > 0 and comp.near_existing is not None:
            col = np.clip(((fx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
            row = np.clip(((f.y_top - fy) / f.res_m).astype(int), 0, f.height - 1)
            span_c *= 1.0 - sp.proximity_discount * float(comp.near_existing[row, col].mean())
        c += span_c
        c += _band_penalty(comp, (x0, y0), (x1, y1), L)
        c += _breach_rs(comp, (x0, y0), (x1, y1))
        c += _excess_rs(sp, L)                      # span over the plain design span (search only, 18 Oct 2026)
    c += _bld_soft_rs(comp, _bld_soft_hits(comp, pts))   # buildings soft zone, once per building (19 Oct 2026)
    return c


def _bld_soft_tree(comp: Compiled):
    """STRtree over the buildings of the soft zone (Vivek, 19 Oct 2026), or None when the zone is off."""
    bg = getattr(comp, "bld_soft_geoms", None)
    if not bg or getattr(comp, "bld_soft_normal_m", 0) <= 0:
        return None
    tr = comp.__dict__.get("_bld_soft_tree")
    if tr is None:
        from shapely import STRtree
        tr = comp.__dict__["_bld_soft_tree"] = STRtree([g for g, _ in bg])
    return tr


def _bld_soft_hits(comp: Compiled, pts) -> dict:
    """{building index: closest distance} for the buildings inside the soft zone of a chain of spans (frame coords);
    each building counted once at its closest distance to the chain (Vivek, 19 Oct 2026)."""
    tr = _bld_soft_tree(comp)
    if tr is None:
        return {}
    bg, lim = comp.bld_soft_geoms, float(comp.bld_soft_normal_m)
    out: dict = {}
    for a, b in zip(pts[:-1], pts[1:]):
        seg = LineString([a, b])
        for j in tr.query(seg, predicate="dwithin", distance=lim):
            dd = bg[j][0].distance(seg)
            if dd < lim - 1e-6 and dd < out.get(j, 1e18):
                out[j] = dd
    return out


def _bld_soft_rs(comp: Compiled, hits: dict) -> float:
    lim = float(getattr(comp, "bld_soft_normal_m", 0) or 0)
    return sum(comp.bld_soft_a_rs + comp.bld_soft_rate_rs_per_m * max(0.0, lim - d) for d in hits.values())


def _breach_rs(comp: Compiled, a, b) -> float:
    """Village-cluster breach charge of one span (Vivek, 18 Oct 2026): the span's centre line sampled on the band
    raster, as the search does; search price only."""
    band = getattr(comp, "breach_band", None)
    rs = float(getattr(comp, "breach_rs", 0.0) or 0.0)
    if band is None or rs <= 0:
        return 0.0
    f = comp.frame
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(2, int(math.ceil(L / (f.res_m / 2))) + 1)
    t = np.linspace(0, 1, n)
    col = np.clip(((a[0] + t * (b[0] - a[0]) - f.x0) / f.res_m).astype(int), 0, f.width - 1)
    row = np.clip(((f.y_top - (a[1] + t * (b[1] - a[1]))) / f.res_m).astype(int), 0, f.height - 1)
    return rs if bool(band[row, col].any()) else 0.0


def _band_index(comp: Compiled):
    """[(span band, penalty, STRtree, lines)] of the crossing rules with a span band (SH), built once."""
    idx = comp.__dict__.get("_band_idx")
    if idx is None:
        from shapely import STRtree
        idx = []
        for cr in comp.crossings:
            if getattr(cr, "span_band", None) is not None and getattr(cr, "band_penalty_rs", 0.0) > 0 and cr.band_lines:
                idx.append((cr.span_band, float(cr.band_penalty_rs), STRtree(cr.band_lines), cr.band_lines))
        comp.__dict__["_band_idx"] = idx
    return idx


def _band_penalty(comp: Compiled, a, b, L: float) -> float:
    """Search cost of a span that crosses a span-band feature (SH) with a length outside the band (16 Oct 2026)."""
    idx = _band_index(comp)
    if not idx:
        return 0.0
    c = 0.0
    seg = None
    for (lo, hi), pen, tree, lines in idx:
        if lo - 1e-3 <= L <= hi + 1e-3:
            continue
        seg = seg or LineString([a, b])
        if any(lines[j].intersects(seg) for j in tree.query(seg)):
            c += pen
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
                if dfl > sp.max_dev_deg + 1e-6 or dfl > _dev_cap(chk, sp, prev, grid_pts[i], q) + 1e-6:
                    continue
            if j < n - 1 and not chk.tower_ok(q[0], q[1], angle_tower=True):
                continue
            inner = _spot_leg(chk, grid_pts[i], q, sp)
            if inner is None:
                continue
            if sp.tsd and not _tsd_leg_ok(chk, sp, prev, grid_pts[i], inner, q, is_last=(j == n - 1)):
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


def _pin_dead_ends(pts: list, sp: SpotParams) -> list:
    """Put the first and last towers on their fixed points (Vivek, 19 Oct 2026: the dead-end tower stands exactly
    dead_end_distance_m from the gantry, square to the substation face)."""
    if len(pts) < 4 or (sp.dead_end_a is None and sp.dead_end_b is None):
        return pts
    pts = list(pts)
    if sp.dead_end_a is not None:
        pts[1] = (float(sp.dead_end_a[0]), float(sp.dead_end_a[1]))
    if sp.dead_end_b is not None:
        pts[-2] = (float(sp.dead_end_b[0]), float(sp.dead_end_b[1]))
    return pts


def _tsd_leg_ok(chk: Checker, sp: SpotParams, prev, start, inner, q, is_last: bool) -> bool:
    """A straightened leg must keep the TSD limits the search kept: the dead-end turn at the first
    and last towers (≤ dead_end_max_dev_deg against the gantry slack span), and at the leg's start
    tower the adjacent-span table and the wind span for its deviation."""
    from .config import tower_type_for
    from .solver import fit_type_for_spans
    T = sp.tsd
    lim = T.get("dead_end_max_dev_deg")
    first = inner[0] if inner else q
    A, B = (0.0, 0.0), (chk.c.frame.length_m, 0.0)
    if prev is None:
        if lim is not None and _deflection(A, start, first)[0] > float(lim) + 1e-6:
            return False
    else:
        d, _ = _deflection(prev, start, first)
        L1 = math.hypot(start[0] - prev[0], start[1] - prev[1])
        L2 = math.hypot(first[0] - start[0], first[1] - start[1])
        long_span = max(L1, L2) > sp.span_max_m + 1e-3          # a long crossing span (river)
        if not long_span:
            t = tower_type_for(d, sp.type_bands) or "D60"
            if fit_type_for_spans(T["span_sum"], t, d, L1, L2, float(T.get("single_span_fraction", 0.6))) is None:
                return False
            if (L1 + L2) / 2.0 > float((T.get("wind_span_m") or {}).get("base", 400)) + 1e-3:
                return False
    if is_last and lim is not None:
        last_prev = inner[-1] if inner else start
        if _deflection(last_prev, q, B)[0] > float(lim) + 1e-6:
            return False
    return True


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
            if prev is not None and _deflection(prev, p, c)[0] > _dev_cap(chk, sp, prev, p, c):
                continue
            if nxt is not None:
                L2 = math.hypot(nxt[0] - c[0], nxt[1] - c[1])
                if L2 < sp.min_span_m or not chk.conductor_ok(c, nxt) or not chk.span_crossings(c, nxt)[0]:
                    continue
                # the moved tower's own turn stays within the limit (10 Oct 2026: nudges of up to 30 m on
                # 100-250 m spans had pushed turns to 68-70°); 45° next to a crossing (13 Oct 2026)
                if _deflection(p, c, nxt)[0] > _dev_cap(chk, sp, p, c, nxt):
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
    pts = _snap_to_legal(chk, pts, sp)
    pts = _pin_dead_ends(pts, sp)
    pts = _crossing_span_fix(chk, pts, sp)
    if sp.refine_crossings and angle_set is None:      # the final, string-pulled alignment
        pts = _refine_crossings(chk, pts, sp)
    if sp.tsd:
        pts = _wind_span_fix(chk, pts, sp)
        pts = _angle_span_fix(chk, pts, sp)
    if sp.refine_crossings:
        pts = _split_relaxed_crossings(chk, pts, sp)
    if dem is not None and dem.ok:
        pts = _split_hilly(chk, pts, sp)
        pts, added = _respot_for_clearance(chk, pts, sp)
    added_pts = {pts[k] for k in added}
    pts = _crossing_turn_split(chk, pts, sp)
    polish = None
    if sp.cost_polish and angle_set is None:         # the final alignment only (the grid route is the fallback)
        pts, s_saved, s_moves = _straighten_runs(chk, pts, sp)
        pts, m_saved, m_moves = _merge_bends(chk, pts, sp)
        pts, v_saved, v_moves = _vertex_descent(chk, pts, sp)
        pts, saved, moves = _cost_polish(chk, pts, sp)
        pts, r_saved, r_moves = _respot_runs(chk, pts, sp)
        pts, b_saved, b_moves = _straighten_between_bends(chk, pts, sp)
        polish = {"moves": moves, "saved_cr": round(saved / 1e7, 3),
                  "straighten_moves": s_moves, "straighten_saved_cr": round(s_saved / 1e7, 3),
                  "merge_moves": m_moves, "merge_saved_cr": round(m_saved / 1e7, 3),
                  "vertex_moves": v_moves, "vertex_saved_cr": round(v_saved / 1e7, 3),
                  "respot_runs": r_moves, "respot_saved_cr": round(r_saved / 1e7, 3),
                  "bend_straighten_moves": b_moves, "bend_straighten_saved_cr": round(b_saved / 1e7, 3)}
    pts = _pin_dead_ends(pts, sp)            # every pass above works round them; this keeps them exact
    added = {k for k, pt in enumerate(pts) if pt in added_pts}
    towers: list[Tower] = []
    n = len(pts)
    for i, (x, y) in enumerate(pts):
        if i == 0 or i == n - 1:
            towers.append(Tower(i, x, y, "G", 0.0, "-", False, ["gantry"]))
            continue
        dfl, turn = _deflection(pts[i - 1], pts[i], pts[i + 1])
        over_dev = dfl > sp.max_dev_deg + 0.5
        if i == 1 or i == n - 2:
            towers.append(Tower(i, x, y, "D60", round(dfl, 2), turn, True, ["dead_end"]))
            continue
        t = tower_type_for(dfl, sp.type_bands) or "D60"
        is_ap = t != "A"
        towers.append(Tower(i, x, y, t, round(dfl, 2), turn if is_ap else "-", is_ap))
        if over_dev:
            towers[-1].violations.append(f"max_deviation:{dfl:.0f}deg")
    spans: list[Span] = []
    for i in range(n - 1):
        L = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        gantry_span = i == 0 or i == n - 2
        okv, cr = (True, []) if gantry_span else chk.span_crossings(pts[i], pts[i + 1])
        s = Span(i, i + 1, L, cr, hilly=(not gantry_span) and chk.is_hilly(pts[i], pts[i + 1]))
        spans.append(s)
        for c in cr:
            if c.get("violation"):
                towers[i].violations.append(f"crossing_{c['violation']}:{c['rule']}")
        if not okv and not any(c.get("violation") for c in cr):
            towers[i].violations.append("span_rule")     # over the span limit or a parallel run (audit, 14 Oct 2026)
        if not chk.conductor_ok(pts[i], pts[i + 1]):
            towers[i if i > 0 else 1].violations.append("conductor_buffer")
        if gantry_span and L < sp.gantry_min_span_m - 1e-6:
            towers[1 if i == 0 else n - 2].violations.append(f"gantry_span:{L:.0f}m")
    # village clusters inside the normal distance (Vivek, 18 Oct 2026): a warning with the exact distance
    bg = getattr(comp, "breach_geoms", None)
    breaches = 0
    if bg and getattr(comp, "breach_normal_m", 0) > 0:
        from shapely import STRtree
        btree = comp.__dict__.get("_breach_tree")
        if btree is None:
            btree = comp.__dict__["_breach_tree"] = STRtree([g for g, _ in bg])
        lim = float(comp.breach_normal_m)
        for i in range(1, n - 2):
            seg = LineString([pts[i], pts[i + 1]])
            for j in btree.query(seg.buffer(lim)):
                dd = bg[j][0].distance(seg)
                if dd < lim - 1e-6:
                    towers[i].warnings.append(f"cluster_breach:{bg[j][1]} {dd:.0f}m")
                    breaches += 1
    # buildings inside the soft zone (Vivek, 19 Oct 2026): a warning with the exact distance, each building once
    bsh = {}
    if _bld_soft_tree(comp) is not None:
        lim_b = float(comp.bld_soft_normal_m)
        for i in range(1, n - 2):
            seg = LineString([pts[i], pts[i + 1]])
            for j in _bld_soft_tree(comp).query(seg, predicate="dwithin", distance=lim_b):
                dd = comp.bld_soft_geoms[j][0].distance(seg)
                if dd < lim_b - 1e-6 and dd < bsh.get(j, (1e18, 0))[0]:
                    bsh[j] = (dd, i)
        for j, (dd, i) in bsh.items():
            towers[i].warnings.append(f"building_soft:{comp.bld_soft_geoms[j][1]} {dd:.1f}m")
    # small ponds spanned (Vivek, 18 Oct 2026): a warning on each span over or within the pond buffer of one
    spd = getattr(comp, "small_ponds", None)
    if spd:
        from shapely import STRtree
        ptree = comp.__dict__.get("_pond_tree")
        if ptree is None:
            ptree = comp.__dict__["_pond_tree"] = STRtree([g for g, _ in spd])
        wm = float(getattr(comp, "small_pond_warn_m", 0.0) or 0.0)
        for i in range(1, n - 2):
            seg = LineString([pts[i], pts[i + 1]])
            for j in ptree.query(seg.buffer(max(wm, 0.01))):
                if spd[j][0].distance(seg) < wm + 1e-6:
                    towers[i].warnings.append(f"small_pond_spanned:{spd[j][1]}")
    # road structures with no NH / SH on them (Vivek, 17 Oct 2026): a warning to classify the structure
    us = getattr(comp, "unclassified_structures", None)
    if us:
        from shapely import STRtree
        tree = comp.__dict__.get("_us_tree")
        if tree is None:
            tree = comp.__dict__["_us_tree"] = STRtree([g for g, _ in us])
        for i in range(1, n - 2):
            seg = LineString([pts[i], pts[i + 1]])
            for j in tree.query(seg):
                if us[j][0].intersects(seg):
                    towers[i].warnings.append(f"bridge_unclassified:{us[j][1]}")
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
            if c.get("extension_m"):
                for ti in (s.a, s.b):
                    t = towers[ti]
                    if t.type != "G" and t.extension_m < c["extension_m"]:
                        t.extension_m = float(c["extension_m"])
                        t.reasons.append(f"crossing_extension:{c['rule']}")
            lim = c.get("max_dev_deg")
            if lim is not None:
                # the two crossing towers turn at most max_dev_deg (Vivek, 13 Oct 2026; was a warning at n±1)
                for ti in (s.a, s.b):
                    if 0 < ti < len(towers) - 1 and towers[ti].deflection_deg > lim + 1e-6:
                        towers[ti].violations.append(f"crossing_deviation:{c['rule']}:{towers[ti].deflection_deg:.0f}deg")
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
    nspans = 0
    hilly_run = False
    for i in range(1, len(towers)):
        run += spans[i - 1].length_m
        nspans += 1
        hilly_run = hilly_run or spans[i - 1].hilly
        t = towers[i]
        if t.type != "A":
            run = 0.0
            nspans = 0
            hilly_run = False
            continue
        nxt = spans[i].length_m if i < len(spans) else 0
        hilly_now = bool(sp.hilly_section_km) and (hilly_run or (i < len(spans) and spans[i].hilly))
        limit = sp.hilly_section_km if hilly_now else sp.section_km
        max_spans = sp.hilly_section_spans if hilly_now else sp.section_spans
        # TSD D: ≤ 15 spans or 5 km (plain), ≤ 10 spans or 3 km (hilly) between section points
        if run + nxt > limit * 1000.0 or (max_spans and nspans + 1 > max_spans):
            t.type = "B1"
            t.reasons.append("section_hilly" if hilly_now else "section")
            run = 0.0
            nspans = 0
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
    if sp.tsd:
        _tsd_checks(towers, spans, sp)
    cost, stats = route_cost(comp, towers, spans, sp)
    if bg:
        stats["cluster_breach_spans"] = breaches
    if _bld_soft_tree(comp) is not None:
        # search-only charge, reported apart from the cost (Vivek, 19 Oct 2026)
        stats["building_soft_count"] = len(bsh)
        stats["building_soft_charge_cr"] = round(_bld_soft_rs(comp, {j: v[0] for j, v in bsh.items()}) / 1e7, 3)
    if polish is not None:
        stats["cost_polish"] = polish
    return SpottedRoute(towers, spans, cost, stats, "pulled" if angle_set is None else "grid")


def _legal_here(chk: Checker, pts, i, sp: SpotParams, cand) -> bool:
    """Would tower i at `cand` pass every exact check with its current neighbours?"""
    n = len(pts)
    if not chk.tower_ok(cand[0], cand[1], angle_tower=True):
        return False
    prev_, next_ = pts[i - 1], pts[i + 1]
    for a, b in ((prev_, cand), (cand, next_)):
        if math.hypot(b[0] - a[0], b[1] - a[1]) < sp.min_span_m - 1e-6:
            return False
        if not chk.conductor_ok(a, b) or not chk.span_crossings(a, b)[0]:
            return False
    de = float(sp.tsd.get("dead_end_max_dev_deg", 180) or 180) if sp.tsd else 180.0     # audit, 14 Oct 2026
    cap = lambda k: min(sp.max_dev_deg, de) if k in (1, n - 2) else sp.max_dev_deg
    if _deflection(prev_, cand, next_)[0] > cap(i):
        return False
    if i >= 2 and _deflection(pts[i - 2], prev_, cand)[0] > cap(i - 1):
        return False
    if i + 2 < n and _deflection(cand, next_, pts[i + 2])[0] > cap(i + 1):
        return False
    if not _crossing_dev_ok(chk, pts, i, cand):
        return False
    if sp.tsd:
        # TSD wind span (mean of the two adjacent spans) at the moved tower and both neighbours — the
        # search honoured it on the lattice; a move must not undo that
        wind = float((sp.tsd.get("wind_span_m") or {}).get("base", 400))
        trial = list(pts); trial[i] = cand
        for m in (i - 1, i, i + 1):
            if 0 < m < n - 1:
                L1 = math.hypot(trial[m][0] - trial[m - 1][0], trial[m][1] - trial[m - 1][1])
                L2 = math.hypot(trial[m + 1][0] - trial[m][0], trial[m + 1][1] - trial[m][1])
                if (L1 + L2) / 2.0 > wind + 1e-3:
                    return False
    return True


def _dev_cap(chk: Checker, sp: SpotParams, prev, p, nxt) -> float:
    """The deflection a tower at p may take: the route maximum, or a crossing rule's cap when the span to
    prev or to nxt crosses a feature that caps its crossing towers (Vivek, 13 Oct 2026: 45°)."""
    cap = sp.max_dev_deg
    for a, b in ((prev, p), (p, nxt)):
        if a is None or b is None:
            continue
        for it in chk.span_crossings(a, b)[1]:
            lim = it.get("max_dev_deg")
            if lim is not None and lim < cap:
                cap = lim
    return cap


def _crossing_dev_ok(chk: Checker, pts, i, cand) -> bool:
    """With tower i at `cand`: every tower at the end of a span that crosses a capped feature (towers
    i-1, i, i+1 can be such ends) turns at most that feature's cap (Vivek, 13 Oct 2026)."""
    n = len(pts)
    trial = list(pts); trial[i] = cand
    caps: dict[int, float] = {}
    for a in (i - 1, i):
        if 0 <= a < n - 1:
            for it in chk.span_crossings(trial[a], trial[a + 1])[1]:
                lim = it.get("max_dev_deg")
                if lim is not None:
                    for ti in (a, a + 1):
                        caps[ti] = min(caps.get(ti, 1e9), lim)
    for ti, lim in caps.items():
        if 0 < ti < n - 1 and _deflection(trial[ti - 1], trial[ti], trial[ti + 1])[0] > lim + 1e-6:
            return False
    return True


def _shift_chain(chk: Checker, pts, i, j, d: float, sp: SpotParams, depth: int = 6):
    """Move tower i by d towards its neighbour j along the line between them (10 Oct 2026 'ripple'),
    used when a repair needs a few metres that a span at the cap cannot give. If tower i's other span
    then exceeds the cap, the tower beyond slides by the excess, and so on along the run (turns
    < 15°, never a gantry, at most `depth` towers). Returns the new point list, or None."""
    n = len(pts)
    if i <= 0 or i >= n - 1 or d <= 0 or depth < 0:
        return None
    if _deflection(pts[i - 1], pts[i], pts[i + 1])[0] >= 15.0:      # up to the B1 band: a slide of a few metres barely changes it
        return None
    p, q = pts[i], pts[j]
    L = math.hypot(q[0] - p[0], q[1] - p[1])
    if L <= d + sp.min_span_m:
        return None
    cand = (p[0] + (q[0] - p[0]) * d / L, p[1] + (q[1] - p[1]) * d / L)
    trial = list(pts)
    trial[i] = cand
    other = i - 1 if j == i + 1 else i + 1
    Lo = math.hypot(trial[other][0] - cand[0], trial[other][1] - cand[1])
    if Lo > sp.span_max_m + 1e-6:
        if depth == 0 or other <= 0 or other >= n - 1:
            return None
        deeper = _shift_chain(chk, trial, other, i, Lo - sp.span_max_m + 1.0, sp, depth - 1)
        if deeper is None:
            return None
        trial = deeper
    return trial if _legal_here(chk, trial, i, sp, cand) else None



def _crossing_turn_split(chk: Checker, pts, sp: SpotParams, radius_m: float = 200.0, step_m: float = 10.0):
    """Last repair (14 Oct 2026): a crossing tower that still turns more than its crossing cap (45°) —
    typically the sideways jog over an existing line our route runs beside — shares the turn with a
    neighbour: the neighbour moves (nearest first, up to radius_m) to where the crossing tower is
    within its cap and every exact check at the neighbour passes (_legal_here). Indices are kept, so
    the clearance insertions stay valid."""
    pts = list(pts)
    n = len(pts)
    rings = [r * step_m for r in range(1, int(radius_m / step_m) + 1)]
    dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    for i in range(1, n - 1):
        cap = _dev_cap(chk, sp, pts[i - 1], pts[i], pts[i + 1])
        if cap >= sp.max_dev_deg or _deflection(pts[i - 1], pts[i], pts[i + 1])[0] <= cap + 1e-6:
            continue
        # one objective (14 Oct 2026): the cheapest legal neighbour move, by exact local cost
        cands = []
        for r in rings:
            for j in (i - 1, i + 1):
                if j <= 0 or j >= n - 1:
                    continue
                for dx, dy in dirs:
                    cand = (pts[j][0] + dx * r, pts[j][1] + dy * r)
                    trial = list(pts); trial[j] = cand
                    if _deflection(trial[i - 1], trial[i], trial[i + 1])[0] > cap + 1e-6:
                        continue
                    cands.append((_local_cost(chk, sp, trial, j, r=3), j, r, cand, trial))
        for cost, j, r, cand, trial in sorted(cands, key=lambda z: z[0]):
            if _legal_here(chk, pts, j, sp, cand) and _clearance_kept(chk, sp, pts, trial, j):
                log.info("crossing turn split: tower %d moved %.0f m so tower %d turns <= %.0f deg", j, r, i, cap)
                pts = trial
                break
    return pts


# ---- one objective for the repairs (Vivek, 14 Oct 2026) ---------------------------------------------
# Every repair picks the cheapest legal candidate by the exact local cost (tower types from the deflections
# that result, crossing / hilly minimum types, spans, RoW and the search surcharges), and a final polish
# moves single towers wherever that lowers the cost; before, repairs took the nearest legal spot.

def _type_at(chk: Checker, sp: SpotParams, pts, k: int) -> str | None:
    n = len(pts)
    if k <= 0 or k >= n - 1:
        return None
    if k == 1 or k == n - 2:
        return "D60"
    dfl, _ = _deflection(pts[k - 1], pts[k], pts[k + 1])
    t = tower_type_for(dfl, sp.type_bands) or "D60"
    for a, b in ((pts[k - 1], pts[k]), (pts[k], pts[k + 1])):
        for it in chk.span_crossings(a, b)[1]:
            if it.get("min_type"):
                t = max_type(t, it["min_type"])
        if sp.hilly_design_span_m and chk.is_hilly(a, b):
            t = max_type(t, "B1")
    return t


def _local_cost(chk: Checker, sp: SpotParams, pts, i: int, r: int = 2) -> float:
    """Exact cost of towers i-r+1 .. i+r-1 and the spans from i-r to i+r (types from the deflections)."""
    a, b = max(0, i - r), min(len(pts) - 1, i + r)
    types = [_type_at(chk, sp, pts, k) for k in range(a + 1, b)]
    if any(t is None for t in types):
        return float("inf")
    return _chain_cost(chk.c, sp, pts[a:b + 1], types)


def _clearance_kept(chk: Checker, sp: SpotParams, old, new, i: int) -> bool:
    """Moving tower i must not make either of its spans clear the ground worse than before (below the
    required clearance) — the extensions that would buy it back are not in the local cost."""
    if chk.dem is None or not getattr(chk.dem, "ok", False):
        return True
    for a, b in ((i - 1, i), (i, i + 1)):
        c1, _ = _span_clearance(chk, new[a], new[b], sp)
        if c1 is None or c1 >= sp.ground_clearance_m:
            continue
        c0, _ = _span_clearance(chk, old[a], old[b], sp)
        if c0 is None or c1 < c0 - 0.05:
            return False
    return True


def _is_crossing_tower(chk: Checker, pts, i: int) -> bool:
    """Tower i ends a span over a power line with a crossing-tower window (placed by the refinement)."""
    for a, b in ((i - 1, i), (i, i + 1)):
        if 0 <= a and b < len(pts) and any(it.get("side_window_m") for it in chk.span_crossings(pts[a], pts[b])[1]):
            return True
    return False


def _cost_polish(chk: Checker, pts, sp: SpotParams, passes: int = 3, min_gain_rs: float = 1e5):
    """Final pass: each tower (not a gantry, dead end or power-line crossing tower) tries spots along the
    line (±10..50 m), across it (±5..40 m), both, and — for a bend — the exact inward shift that brings its
    deflection down to the next type band (idea 3); the cheapest that passes every exact check and keeps
    the ground clearance replaces it when it saves at least min_gain_rs. Repeats until nothing improves.
    Returns (points, ₹ saved, moves)."""
    pts = list(pts)
    n = len(pts)
    saved, moves = 0.0, 0
    along = (10, 20, 35, 50)
    across = (5, 10, 15, 20, 30, 40)
    edges = sorted(v for k, v in sp.type_bands.items() if k != "D60") if sp.type_bands else [2.0, 15.0, 30.0, 45.0]
    for _ in range(passes):
        gain_pass = 0.0
        for i in range(2, n - 2):
            if _is_crossing_tower(chk, pts, i):
                continue
            hx, hy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
            hn = math.hypot(hx, hy) or 1.0
            ux, uy = hx / hn, hy / hn
            vx, vy = -uy, ux
            offs = [(s * a, 0.0) for a in along for s in (1, -1)] + [(0.0, s * c) for c in across for s in (1, -1)] + \
                   [(s * a, t * c) for a in (10, 20) for c in (5, 10) for s in (1, -1) for t in (1, -1)]
            cand_pts = [(pts[i][0] + ux * da + vx * dc, pts[i][1] + uy * da + vy * dc) for da, dc in offs]
            # angle redistribution to a band edge (Vivek, 14 Oct 2026: idea 3): a bend just above a type band
            # (16° -> C1) moves inward along its bisector by the exact amount that brings it to the edge
            # (14.7° -> B1); its neighbours take the difference, which the local cost prices
            dfl = _deflection(pts[i - 1], pts[i], pts[i + 1])[0]
            if dfl > 2.0:
                L1 = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                L2 = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                mx, my = (pts[i - 1][0] + pts[i + 1][0]) / 2 - pts[i][0], (pts[i - 1][1] + pts[i + 1][1]) / 2 - pts[i][1]
                mn = math.hypot(mx, my) or 1.0
                for e in edges:
                    if 0 < dfl - e <= 8.0:
                        delta = math.radians(dfl - e + 0.3) / (1.0 / L1 + 1.0 / L2)
                        if delta <= 60.0:
                            cand_pts.append((pts[i][0] + mx / mn * delta, pts[i][1] + my / mn * delta))
            base = _local_cost(chk, sp, pts, i)
            cands = []
            for c in cand_pts:
                trial = list(pts); trial[i] = c
                if min(math.hypot(c[0] - pts[i - 1][0], c[1] - pts[i - 1][1]),
                       math.hypot(pts[i + 1][0] - c[0], pts[i + 1][1] - c[1])) < sp.min_span_m:
                    continue
                cost = _local_cost(chk, sp, trial, i)
                if cost < base - min_gain_rs:
                    cands.append((cost, c, trial))
            for cost, c, trial in sorted(cands, key=lambda z: z[0]):
                if _legal_here(chk, pts, i, sp, c) and _clearance_kept(chk, sp, pts, trial, i):
                    gain_pass += base - cost
                    moves += 1
                    pts = trial
                    break
        saved += gain_pass
        if gain_pass < min_gain_rs:
            break
    if moves:
        log.info("cost polish: %d moves, ₹%.2f Cr saved (search prices)", moves, saved / 1e7)
    return pts, saved, moves


def _relax_cost(chk: Checker, pts, a: int, b: int) -> float:
    """Search price of the crossing-tower relaxation (beyond the window maximum) on spans a..b-1."""
    tot = 0.0
    for k in range(max(0, a), min(len(pts) - 1, b)):
        for it in chk.span_crossings(pts[k], pts[k + 1])[1]:
            # the item's own rate (14 Oct 2026: a lookup by rule id took the go-zone copy's zero rate, so the
            # repair priced a 180 m relaxation at nothing and kept it)
            tot += float(it.get("side_relaxed_m") or 0.0) * float(it.get("side_relax_rs_per_m") or 0.0)
    return tot


def _run_legal(chk: Checker, sp: SpotParams, trial, a: int, b: int, fixed: set) -> bool:
    """Every exact check for towers a..b of `trial` (those not in `fixed` are new or moved) and the spans and
    turns they touch: tower buffers, span limits, conductor buffers, crossings, turn limits and crossing caps,
    TSD wind span and ground clearance with the largest A extension."""
    n = len(trial)
    if a < 2 or b > n - 3:
        return False
    for k in range(a, b + 1):
        if k not in fixed and not chk.tower_ok(trial[k][0], trial[k][1], angle_tower=True):
            return False
    e = _max_ext(sp, "A")
    for k in range(a - 1, b + 1):
        p, q = trial[k], trial[k + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L < sp.min_span_m - 1e-6 or L > sp.span_max_m + 1e-3:
            return False
        if not chk.conductor_ok(p, q) or not chk.span_crossings(p, q)[0]:
            return False
        if chk.dem is not None and getattr(chk.dem, "ok", False):
            c, _ = _span_clearance(chk, p, q, sp, e, e)
            if c is not None and c < sp.ground_clearance_m - 1e-6:
                return False
    wind = float((sp.tsd.get("wind_span_m") or {}).get("base", 400)) if sp.tsd else None
    for k in range(a - 1, b + 2):
        if not 0 < k < n - 1:
            continue
        if _deflection(trial[k - 1], trial[k], trial[k + 1])[0] > sp.max_dev_deg + 1e-6:
            return False
        if not _crossing_dev_ok(chk, trial, k, trial[k]):
            return False
        if wind is not None:
            L1 = math.hypot(trial[k][0] - trial[k - 1][0], trial[k][1] - trial[k - 1][1])
            L2 = math.hypot(trial[k + 1][0] - trial[k][0], trial[k + 1][1] - trial[k][1])
            if (L1 + L2) / 2.0 > wind + 1e-3:
                return False
    return True


def _split_relaxed_crossings(chk: Checker, pts, sp: SpotParams, min_excess_m: float = 12.5):
    """A crossing tower beyond the window maximum (relaxed) gets a new tower inside the window on the same
    span line, when that is cheaper than the relaxation (14 Oct 2026: one tower stood 303 m from a 220 kV
    wire between two lines, where a second tower between them was the cheap fix). The old tower stays as
    the next tower along."""
    pts = list(pts)
    i = 1
    while i < len(pts) - 2:
        p, q = pts[i], pts[i + 1]
        items = [it for it in chk.span_crossings(p, q)[1] if it.get("side_window_m") and (it.get("side_relaxed_m") or 0) > min_excess_m]
        if not items:
            i += 1
            continue
        it = items[0]
        lo, hi = it["side_window_m"]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        s_ang = max(math.sin(math.radians(it["angle_deg"])), 0.05)
        best = None
        for side, d_perp in (("b", it["side_b_m"]), ("a", it["side_a_m"])):
            if d_perp <= hi + min_excess_m:
                continue
            for d in [hi - 4 * k for k in range(int((hi - lo) // 4) + 1)]:
                if side == "b":
                    x = it["along_b_m"] - d / s_ang       # back from q towards the wire
                    new = (q[0] - ux * x, q[1] - uy * x)
                    trial = pts[:i + 1] + [new] + pts[i + 1:]
                    k_new = i + 1
                else:
                    x = it["along_a_m"] - d / s_ang
                    new = (p[0] + ux * x, p[1] + uy * x)
                    trial = pts[:i + 1] + [new] + pts[i + 1:]
                    k_new = i + 1
                if x < sp.min_span_m - 1e-6:
                    continue
                if not _run_legal(chk, sp, trial, k_new, k_new, fixed=set()):
                    continue
                before = _chain_cost(chk.c, sp, pts[i - 1:i + 3], [_type_at(chk, sp, pts, k) for k in range(i, i + 2)]) \
                    + _relax_cost(chk, pts, i - 1, i + 2)
                after = _chain_cost(chk.c, sp, trial[i - 1:i + 4], [_type_at(chk, sp, trial, k) for k in range(i, i + 3)]) \
                    + _relax_cost(chk, trial, i - 1, i + 3)
                if after < before and (best is None or after - before < best[0]):
                    best = (after - before, trial)
                break                                     # the first legal spot from the maximum inwards
        if best is not None:
            log.info("relaxed crossing %d-%d: a tower added inside the window (₹%.2f Cr)", i, i + 1, best[0] / 1e7)
            pts = best[1]
        i += 1
    return pts


def _straighten_runs(chk: Checker, pts, sp: SpotParams, max_run: int = 5, reach_m: float = 40.0, min_gain_rs: float = 1e5):
    """Multi-tower move (Vivek, 14 Oct 2026): a run of towers between anchors i and j (2..max_run spans) is
    replaced by a straight run from i to j, j moved up to reach_m, its towers spotted by the leg programme
    (uneven, wherever they may stand) — the cheapest by exact cost that passes every check. Merges two nearby bends into one, removes a
    tower a bend forced, straightens a bow round an obstacle. Power-line crossing towers are not moved."""
    pts = list(pts)
    offs = [0.0, -10.0, 10.0, -20.0, 20.0, -35.0, 35.0]
    moves, saved = 0, 0.0
    i = 2
    while i < len(pts) - 4:
        n = len(pts)
        done = False
        for m in range(max_run, 1, -1):
            j = i + m
            if j > n - 3:
                continue
            inner = range(i + 1, j)
            if all(_deflection(pts[k - 1], pts[k], pts[k + 1])[0] <= 2.0 for k in inner):
                continue                                  # already straight
            if any(_is_crossing_tower(chk, pts, k) for k in list(inner) + [j]):
                continue
            base = _chain_cost(chk.c, sp, pts[i - 1:j + 2], [_type_at(chk, sp, pts, k) for k in range(i, j + 1)])
            hx, hy = pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]
            hn = math.hypot(hx, hy) or 1.0
            ux, uy = hx / hn, hy / hn
            cands = []
            for da in offs:
                for dc in offs:
                    if math.hypot(da, dc) > reach_m + 1e-6:
                        continue
                    jq = (pts[j][0] + ux * da - uy * dc, pts[j][1] + uy * da + ux * dc)
                    run = _spot_leg(chk, pts[i], jq, sp)       # towers where they may stand, cheapest (uneven)
                    if run is None:
                        continue
                    nsp = len(run) + 1
                    trial = pts[:i + 1] + list(run) + [jq] + pts[j + 1:]
                    jn = i + nsp
                    cost = _chain_cost(chk.c, sp, trial[i - 1:jn + 2], [_type_at(chk, sp, trial, k) for k in range(i, jn + 1)])
                    if cost < base - min_gain_rs:
                        cands.append((cost, trial, jn, da == 0 and dc == 0))
            for cost, trial, jn, j_kept in sorted(cands, key=lambda z: z[0]):
                if _run_legal(chk, sp, trial, i + 1, jn, fixed={jn} if j_kept else set()):
                    # the winner's run placed by the leg programme (uneven), when that is cheaper and legal too
                    run2 = _spot_leg(chk, trial[i], trial[jn], sp, exact=True)
                    if run2 is not None:
                        trial2 = trial[:i + 1] + list(run2) + trial[jn:]
                        jn2 = i + len(run2) + 1
                        cost2 = _chain_cost(chk.c, sp, trial2[i - 1:jn2 + 2], [_type_at(chk, sp, trial2, k) for k in range(i, jn2 + 1)])
                        if cost2 <= cost and _run_legal(chk, sp, trial2, i + 1, jn2, fixed={jn2} if j_kept else set()):
                            trial, jn, cost = trial2, jn2, cost2
                    log.info("straighten %d..%d: %d -> %d spans, ₹%.2f Cr", i, j, j - i, jn - i, (base - cost) / 1e7)
                    saved += base - cost
                    moves += 1
                    pts = trial
                    done = True
                    break
            if done:
                break
        if not done:
            i += 1
    return pts, saved, moves


def _straighten_between_bends(chk: Checker, pts, sp: SpotParams, max_len_m: float = 2000.0, min_gain_rs: float = 1e5):
    """Final pass (Vivek, 19 Oct 2026): for every pair of bends i < j at most max_len_m apart along the line, the
    towers between them are replaced by a straight run spotted by the exact leg programme; kept when every check
    passes and the chain cost (both end bends' new turns included) drops by min_gain_rs. Runs after the other
    polish steps, so it sees the final bends (the 5-span straighten probes with even spacing, before the bends
    settle, and missed T164 -> T169 on rev 44). Crossing towers inside the run are never removed."""
    pts = list(pts)
    moves, saved = 0, 0.0
    i = 2
    while i < len(pts) - 4:
        if not _is_vertex(pts, i):
            i += 1
            continue
        best = None
        run_len = 0.0
        for j in range(i + 1, len(pts) - 2):
            run_len += math.dist(pts[j - 1], pts[j])
            if run_len > max_len_m:
                break
            if _is_crossing_tower(chk, pts, j - 1) and j - 1 > i:
                break
            if j - i < 2 or not _is_vertex(pts, j):
                continue
            leg = _spot_leg(chk, pts[i], pts[j], sp, exact=True)
            if leg is None:
                continue
            trial = pts[:i + 1] + list(leg) + pts[j:]
            jn = i + len(leg) + 1
            a, b = max(1, i - 1), min(len(pts) - 2, j + 1)
            base = _chain_cost(chk.c, sp, pts[a - 1:b + 2], [_type_at(chk, sp, pts, k) or "D60" for k in range(a, b + 1)])
            bn = min(len(trial) - 2, jn + 1)
            cost = _chain_cost(chk.c, sp, trial[a - 1:bn + 2], [_type_at(chk, sp, trial, k) or "D60" for k in range(a, bn + 1)])
            if cost < base - min_gain_rs and (best is None or base - cost > best[0]):
                if _run_legal(chk, sp, trial, i, jn, fixed={i, jn}):
                    best = (base - cost, trial, j)
        if best is not None:
            log.info("straighten bends %d..%d: ₹%.2f Cr", i, best[2], best[0] / 1e7)
            pts, saved, moves = best[1], saved + best[0], moves + 1
            continue                                   # retry from the same bend on the new line
        i += 1
    return pts, saved, moves


# ---- angle-point moves (Vivek, 14 Oct 2026: reduce angle towers under the rules) ----------------------

def _is_vertex(pts, k: int) -> bool:
    return 0 < k < len(pts) - 1 and _deflection(pts[k - 1], pts[k], pts[k + 1])[0] > 2.0


def _turn_sign(pts, k: int) -> int:
    ax, ay = pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]
    bx, by = pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]
    c = ax * by - ay * bx
    return 1 if c > 0 else -1 if c < 0 else 0


def _ray_intersection(a, b, c, d):
    """X on the line a->b beyond b and on the line d->c beyond c (the two outer legs converge), else None."""
    r = (b[0] - a[0], b[1] - a[1])
    s_ = (c[0] - d[0], c[1] - d[1])
    den = r[0] * s_[1] - r[1] * s_[0]
    if abs(den) < 1e-9:
        return None
    qp = (d[0] - a[0], d[1] - a[1])
    t = (qp[0] * s_[1] - qp[1] * s_[0]) / den
    u = (qp[0] * r[1] - qp[1] * r[0]) / den
    if t <= 1.0 + 1e-9 or u <= 1.0 + 1e-9:
        return None
    return (a[0] + t * r[0], a[1] + t * r[1])


def _merge_bends(chk: Checker, pts, sp: SpotParams, max_gap_m: float = 1200.0, min_gain_rs: float = 1e5):
    """Two same-direction bends within max_gap_m become one bend where their outer legs meet (the tangent
    intersection): C1 + C1 -> one D45, B1 + B1 -> one C1. The towers between the outer anchors are spotted
    again by the leg programme; the merged vertex may sit up to 30 m off the exact intersection. Kept when
    cheaper by exact cost and every check passes. Crossing towers and dead ends are never merged."""
    pts = list(pts)
    moves, saved = 0, 0.0
    offs = [(0.0, 0.0)] + [(r * math.cos(a), r * math.sin(a)) for r in (15.0, 30.0) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    i = 2
    while i < len(pts) - 3:
        n = len(pts)
        if not _is_vertex(pts, i) or _is_crossing_tower(chk, pts, i):
            i += 1
            continue
        j = i + 1
        while j < n - 2 and not _is_vertex(pts, j):
            j += 1
        if j >= n - 2 or _is_crossing_tower(chk, pts, j) or _turn_sign(pts, i) != _turn_sign(pts, j):
            i += 1
            continue
        if sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(i, j)) > max_gap_m:
            i += 1
            continue
        X = _ray_intersection(pts[i - 1], pts[i], pts[j], pts[j + 1])
        if X is None or _deflection(pts[i - 1], X, pts[j + 1])[0] > sp.max_dev_deg + 1e-6:
            i += 1
            continue
        # the straight runs on both sides are spotted again from the neighbouring vertices (or dead ends), so
        # the extra tangent length is absorbed by their span slack instead of costing a tower
        ip = i - 1
        while ip > 1 and not _is_vertex(pts, ip) and not _is_crossing_tower(chk, pts, ip):
            ip -= 1
        jn = j + 1
        while jn < n - 2 and not _is_vertex(pts, jn) and not _is_crossing_tower(chk, pts, jn):
            jn += 1
        base = _chain_cost(chk.c, sp, pts[ip - 1:jn + 2], [_type_at(chk, sp, pts, k) for k in range(ip, jn + 1)])
        best = None
        for ox, oy in offs:
            Xc = (X[0] + ox, X[1] + oy)
            if not chk.tower_ok(Xc[0], Xc[1], angle_tower=True):
                continue
            left = _spot_leg(chk, pts[ip], Xc, sp, exact=True)
            right = _spot_leg(chk, Xc, pts[jn], sp, exact=True) if left is not None else None
            if left is None or right is None:
                continue
            trial = pts[:ip + 1] + list(left) + [Xc] + list(right) + pts[jn:]
            kend = ip + len(left) + len(right) + 1       # last new tower; pts[jn] sits at kend + 1
            cost = _chain_cost(chk.c, sp, trial[ip - 1:kend + 3], [_type_at(chk, sp, trial, k) for k in range(ip, kend + 2)])
            if cost < base - min_gain_rs and (best is None or cost < best[0]):
                best = (cost, trial, kend, ip)
        if best is not None and _run_legal(chk, sp, best[1], best[3] + 1, best[2], fixed=set()):
            log.info("merge bends %d+%d: ₹%.2f Cr", i, j, (base - best[0]) / 1e7)
            saved += base - best[0]
            moves += 1
            pts = best[1]
            i = max(2, best[3])
            continue
        i += 1
    return pts, saved, moves


def _vertex_descent(chk: Checker, pts, sp: SpotParams, passes: int = 2, min_gain_rs: float = 1e5,
                    radii=(20.0, 40.0, 80.0, 120.0)):
    """Idea 4 (Vivek, 14 Oct 2026): continuous angle-point optimisation. Every vertex (a bend that is not a
    crossing tower or dead end) tries, in turn, each spot on a star round it (8 directions x radii) and its
    own removal; both legs out to the neighbouring anchors (vertices, crossing towers, dead ends) are spotted
    again. The cheapest candidate by exact cost that passes every check replaces it; the winner's legs are
    then placed by the leg programme when that is cheaper still. Passes repeat until nothing improves."""
    pts = list(pts)
    moves, saved = 0, 0.0
    dirs = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    for _ in range(passes):
        gain_pass = 0.0
        v = 2
        while v < len(pts) - 2:
            n = len(pts)
            if not _is_vertex(pts, v) or _is_crossing_tower(chk, pts, v):
                v += 1
                continue
            ip = v - 1
            while ip > 1 and not _is_vertex(pts, ip) and not _is_crossing_tower(chk, pts, ip):
                ip -= 1
            jn = v + 1
            while jn < n - 2 and not _is_vertex(pts, jn) and not _is_crossing_tower(chk, pts, jn):
                jn += 1
            base = _chain_cost(chk.c, sp, pts[ip - 1:jn + 2], [_type_at(chk, sp, pts, k) for k in range(ip, jn + 1)])
            cands = []                                   # (cost, trial, kend)
            spots = [None] + [(pts[v][0] + dx * r, pts[v][1] + dy * r) for r in radii for dx, dy in dirs]
            for c in spots:
                if c is None:                            # remove the vertex: one straight leg ip -> jn
                    run = _spot_leg(chk, pts[ip], pts[jn], sp)
                    if run is None:
                        continue
                    trial = pts[:ip + 1] + list(run) + pts[jn:]
                    kend = ip + len(run)
                else:
                    if not chk.tower_ok(c[0], c[1], angle_tower=True):
                        continue
                    left = _spot_leg(chk, pts[ip], c, sp)
                    right = _spot_leg(chk, c, pts[jn], sp) if left is not None else None
                    if left is None or right is None:
                        continue
                    trial = pts[:ip + 1] + list(left) + [c] + list(right) + pts[jn:]
                    kend = ip + len(left) + len(right) + 1
                cost = _chain_cost(chk.c, sp, trial[ip - 1:kend + 3], [_type_at(chk, sp, trial, k) for k in range(ip, kend + 2)])
                if cost < base - min_gain_rs:
                    cands.append((cost, trial, kend))
            done = False
            for cost, trial, kend in sorted(cands, key=lambda z: z[0])[:6]:
                if not _run_legal(chk, sp, trial, ip + 1, kend, fixed=set()):
                    continue
                # the winner's legs by the leg programme (uneven), kept when cheaper and legal
                nv = next((k for k in range(ip + 1, kend + 1) if _is_vertex(trial, k)), None)
                if nv is not None:
                    left2 = _spot_leg(chk, trial[ip], trial[nv], sp, exact=True)
                    right2 = _spot_leg(chk, trial[nv], trial[kend + 1], sp, exact=True)
                    if left2 is not None and right2 is not None:
                        trial2 = trial[:ip + 1] + list(left2) + [trial[nv]] + list(right2) + trial[kend + 1:]
                        kend2 = ip + len(left2) + len(right2) + 1
                        cost2 = _chain_cost(chk.c, sp, trial2[ip - 1:kend2 + 3], [_type_at(chk, sp, trial2, k) for k in range(ip, kend2 + 2)])
                        if cost2 <= cost and _run_legal(chk, sp, trial2, ip + 1, kend2, fixed=set()):
                            trial, kend, cost = trial2, kend2, cost2
                log.info("vertex %d: %s, ₹%.2f Cr", v, "removed" if kend < jn - 1 + (kend - jn + 1) and len(trial) < n else "moved", (base - cost) / 1e7)
                gain_pass += base - cost
                moves += 1
                pts = trial
                v = max(2, ip)
                done = True
                break
            if not done:
                v += 1
        saved += gain_pass
        if gain_pass < min_gain_rs:
            break
    if moves:
        log.info("vertex descent: %d moves, ₹%.2f Cr", moves, saved / 1e7)
    return pts, saved, moves

def _snap_to_legal(chk: Checker, pts, sp: SpotParams):
    """The search accepts a lattice site when a legal cell lies within nudge_radius_m of it (10 Oct
    2026); a grid tower that fails the exact check is moved to the nearest point within that radius
    where it and its two spans pass every check. Gantries stay; a tower with no such point is left
    for the violation report."""
    pts = list(pts)
    n = len(pts)
    for i in range(1, n - 1):
        x, y = pts[i]
        if chk.tower_ok(x, y, angle_tower=True):
            continue
        prev_, next_ = pts[i - 1], pts[i + 1]
        # local route direction: moves across it barely change the span lengths, so try them first
        hx, hy = next_[0] - prev_[0], next_[1] - prev_[1]
        hn = math.hypot(hx, hy) or 1.0
        found = None
        # one objective (14 Oct 2026): every legal spot in reach, the cheapest by exact local cost (a slide
        # along the line that keeps a suspension tower beats a sideways step that makes it a B1)
        best = None
        for rad in range(5, int(sp.nudge_radius_m) + 6, 5):
            for k in range(24):
                a = 2 * math.pi * k / 24
                c = (x + rad * math.cos(a), y + rad * math.sin(a))
                if not chk.tower_ok(c[0], c[1], angle_tower=True):
                    continue
                if _legal_here(chk, pts, i, sp, c):
                    trial = list(pts); trial[i] = c
                    cost = _local_cost(chk, sp, trial, i)
                    if best is None or cost < best[0]:
                        best = (cost, c)
        if best is not None:
            pts[i] = best[1]
            continue
        for rad in range(5, int(sp.nudge_radius_m) + 6, 5):
            angles = sorted((2 * math.pi * k / 24 for k in range(24)),
                            key=lambda a: abs(math.cos(a) * hx / hn + math.sin(a) * hy / hn))
            for a in angles:
                c = (x + rad * math.cos(a), y + rad * math.sin(a))
                if not chk.tower_ok(c[0], c[1], angle_tower=True):
                    continue
                # a neighbouring span at the cap: let a straight neighbour slide to make room
                trial = list(pts); trial[i] = c
                for nb, far in ((i - 1, i - 2), (i + 1, i + 2)):
                    if nb <= 0 or nb >= n - 1 or far < 0 or far >= n:
                        continue
                    Lnb = math.hypot(c[0] - pts[nb][0], c[1] - pts[nb][1])
                    over = Lnb - sp.span_max_m
                    if over <= 0:
                        continue
                    moved = _shift_chain(chk, trial, nb, i, over + 1.0, sp)
                    if moved is not None:
                        trial = moved
                if trial[i - 1] != pts[i - 1] or trial[i + 1] != pts[i + 1]:
                    if _legal_here(chk, trial, i, sp, c):
                        pts[:] = trial
                        found = c
                        break
            if found:
                break
        if found:
            pts[i] = found
    return pts


def _crossing_span_fix(chk: Checker, pts, sp: SpotParams):
    """The search lets a crossing span exceed its limit by half a site spacing (10 Oct 2026); here the
    two towers of such a span slide towards each other along it until the exact limit holds, each new
    position passing the tower, conductor, crossing, span-length and deviation checks."""
    pts = list(pts)
    n = len(pts)
    pts = _crossing_side_fix(chk, pts, sp)
    for i in range(1, n - 2):                      # gantry spans are left alone
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        items = chk.span_crossings(p, q)[1]
        rules = {it.get("rule") for it in items}
        lims = [cr.max_span_m for cr in chk.c.crossings if cr.max_span_m is not None and cr.rule_id in rules]
        if not lims or L <= min(lims) + 1e-3:
            continue
        lim = min(lims)
        need = L - lim
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        prev_, next_ = pts[i - 1], pts[i + 2]
        done = False
        for extra in (0.5, 3.0, 6.0, 10.0):
            for split in (0.5, 1.0, 0.0, 0.25, 0.75):
                dp, dq = (need + extra) * split, (need + extra) * (1 - split)
                p2 = (p[0] + ux * dp, p[1] + uy * dp)
                q2 = (q[0] - ux * dq, q[1] - uy * dq)
                segs = [(prev_, p2), (p2, q2), (q2, next_)]
                if any(math.hypot(b[0] - a[0], b[1] - a[1]) < sp.min_span_m - 1e-6 for a, b in segs):
                    continue
                if not (chk.tower_ok(p2[0], p2[1], True) and chk.tower_ok(q2[0], q2[1], True)):
                    continue
                # neighbours pushed over the cap by the slide: let a straight neighbour follow
                trial = list(pts); trial[i], trial[i + 1] = p2, q2
                for nb, k_, mv in ((i - 1, i, dp), (i + 2, i + 1, dq)):
                    if mv <= 0 or nb <= 0 or nb >= n - 1:
                        continue
                    Lnb = math.hypot(trial[k_][0] - trial[nb][0], trial[k_][1] - trial[nb][1])
                    if Lnb > sp.span_max_m + 1e-6:
                        moved = _shift_chain(chk, trial, nb, k_, Lnb - sp.span_max_m + 1.0, sp)
                        if moved is not None:
                            trial = moved
                prev_t, next_t = trial[i - 1], trial[i + 2]
                segs = [(prev_t, p2), (p2, q2), (q2, next_t)]
                if not all(chk.conductor_ok(a, b) and chk.span_crossings(a, b)[0] for a, b in segs):
                    continue
                prev_, next_ = prev_t, next_t
                if any(_deflection(a, b, c)[0] > _dev_cap(chk, sp, a, b, c) for a, b, c in
                       ((pts[i - 2] if i >= 2 else None, prev_, p2), (prev_, p2, q2), (p2, q2, next_),
                        (q2, next_, pts[i + 3] if i + 3 < n else None)) if a is not None and c is not None):
                    continue
                pts[:] = trial
                pts[i], pts[i + 1] = p2, q2
                done = True
                break
            if done:
                break
    return pts


def _crossing_side_fix(chk: Checker, pts, sp: SpotParams):
    """Crossing towers into their window (Vivek, 13 Oct 2026): where a span crosses a wire whose rule sets
    crossing_side_m, both towers are placed along the alignment lo..hi from the wire (the middle of the
    window first) and the straight run on each side is spotted again between the crossing tower and the
    run's end (its spans were at the design span, so a ripple could not absorb the move). A crossing that
    cannot be placed is left for the violation report."""
    pts = list(pts)
    i = 1
    while i < len(pts) - 2:
        n = len(pts)
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        items = [it for it in chk.span_crossings(p, q)[1] if it.get("side_window_m")] if L > 0 else []
        # placed again when a tower is outside its window, or stands in another buffer next to the crossing
        # (14 Oct 2026: e.g. a second line beside the crossed one — the span then relaxes as far as needed)
        towers_bad = not (chk.tower_ok(p[0], p[1], True) and chk.tower_ok(q[0], q[1], True))
        # 14 Oct 2026: a relaxed crossing (towers beyond the maximum) is placed again too, accepting only less excess
        relaxed_now = max((it.get("side_relaxed_m") or 0.0 for it in items), default=0.0)
        if not items or (all(it.get("violation") != "side" for it in items) and not towers_bad and relaxed_now <= 0.5):
            i += 1
            continue
        only_better = relaxed_now > 0.5 and all(it.get("violation") != "side" for it in items) and not towers_bad
        it = items[0]
        lo, hi = it["side_window_m"]
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        X = (p[0] + ux * it["along_a_m"], p[1] + uy * it["along_a_m"])     # the (first) crossed wire on the alignment
        gap = max(0.0, L - it["along_a_m"] - it["along_b_m"])                # to the last wire, when one span crosses two
        s_ang = max(math.sin(math.radians(it["angle_deg"])), 0.05)          # window distances are perpendicular to the wire
        # the straight run this span belongs to: from the previous turn (or dead-end) to the next
        run_start = i
        while run_start > 1 and _deflection(pts[run_start - 1], pts[run_start], pts[run_start + 1])[0] < 1.0:
            run_start -= 1
        run_end = i + 1
        while run_end < n - 2 and _deflection(pts[run_end - 1], pts[run_end], pts[run_end + 1])[0] < 1.0:
            run_end += 1
        done = False
        mid = (lo + hi) / 2
        # every pair of distances in the window on a 4 m step, nearest the middle first; then beyond the
        # maximum (Vivek, 14 Oct 2026: relaxed only as far as needed), least excess first, up to the span cap
        steps = [lo + 2 + 4 * k for k in range(int((hi - lo - 4) // 4) + 1)]
        combos = sorted(((da, db) for da in steps for db in steps), key=lambda c_: abs(c_[0] - mid) + abs(c_[1] - mid))
        far = [hi + 4 * k for k in range(1, int(max(0.0, sp.span_max_m - lo - hi - gap) // 4) + 1)]
        ext = [(da, db) for da in steps + far for db in steps + far if (da > hi or db > hi) and (da + db) / s_ang + gap <= sp.span_max_m]
        combos += sorted(ext, key=lambda c_: (max(0, c_[0] - hi) + max(0, c_[1] - hi), abs(c_[0] - c_[1])))
        if only_better:
            combos = [c_ for c_ in combos if max(0, c_[0] - hi) + max(0, c_[1] - hi) < relaxed_now - 0.5]
        for da, db in combos:
            p2 = (X[0] - ux * da / s_ang, X[1] - uy * da / s_ang)
            q2 = (X[0] + ux * (gap + db / s_ang), X[1] + uy * (gap + db / s_ang))
            if not (chk.tower_ok(p2[0], p2[1], True) and chk.tower_ok(q2[0], q2[1], True)):
                log.debug("side fix %d: towers at %g/%g m not legal", i, da, db)
                continue
            if not (chk.conductor_ok(p2, q2) and chk.span_crossings(p2, q2)[0]):
                log.debug("side fix %d: crossing span at %g/%g m not legal", i, da, db)
                continue
            # re-spot a window of a few spans each side (the whole run would include the approach
            # sections, whose shorter spans a single even spacing cannot give), widening if needed
            for w in (3, 5, 8):
                start = max(run_start, i - w)
                end = min(run_end, i + 1 + w)
                left = _spot_leg(chk, pts[start], p2, sp) if start < i else []
                right = _spot_leg(chk, q2, pts[end], sp) if end > i + 1 else []
                if left is None or right is None:
                    log.debug("side fix %d: window %d..%d cannot be re-spotted (left %s, right %s)", i, start, end, left is not None, right is not None)
                    continue
                head = pts[:start + 1] if start < i else pts[:i]          # the old crossing towers go
                tail = pts[end:] if end > i + 1 else pts[i + 2:]
                trial = head + list(left) + [p2, q2] + list(right) + tail
                k = len(head) + len(left)                               # p2's index in the trial
                if any(math.hypot(trial[m + 1][0] - trial[m][0], trial[m + 1][1] - trial[m][1]) < sp.min_span_m - 1e-6
                       for m in range(max(0, len(head) - 1), min(len(trial) - 1, k + 2 + len(right)))):
                    log.debug("side fix %d: a span under the minimum at %g/%g m", i, da, db)
                    continue
                if not (_crossing_dev_ok(chk, trial, k, p2) and _crossing_dev_ok(chk, trial, k + 1, q2)):
                    log.debug("side fix %d: crossing-tower deviation over the cap at %g/%g m", i, da, db)
                    continue
                pts = trial
                i = k + 1
                done = True
                break
            if done:
                break
        if not done:
            log.info("crossing span %d-%d: no placement within %g-%g m of the crossed wire", i, i + 1, lo, hi)
        i += 1
    return pts


# ---- fine local re-optimisation round power-line crossings (Vivek, 14 Oct 2026) -------------------

def _wire_sides(chk, rule_id, p, q):
    """Exact perpendicular distances (sa, sb) from towers p and q to the first / last existing wire of rule_id
    their span crosses, and the along-span distances to those crossing points (14 Oct 2026). None if the
    geometry finds no crossing (the raster then decides)."""
    from shapely.geometry import LineString as _LS, Point as _Pt
    tree = getattr(chk, "_wire_tree", None)
    if tree is None:
        from shapely import STRtree
        ws = _wires(chk)
        chk._wire_list = ws
        chk._wire_tree = tree = STRtree([ln for _, ln, _ in ws]) if ws else False
    if tree is False:
        return None
    seg = _LS([p, q])
    hits = []
    for j in tree.query(seg):
        rid, ln, _ = chk._wire_list[j]
        if rid != rule_id:
            continue
        x = seg.intersection(ln)
        if x.is_empty:
            continue
        for g in getattr(x, "geoms", [x]):
            if g.geom_type == "Point":
                hits.append((seg.project(g), ln))
    if not hits:
        return None
    hits.sort(key=lambda h: h[0])
    (ta, la), (tb, lb) = hits[0], hits[-1]
    # exact distance of each crossing from the null point of the crossed span, as a share of that span
    # (0 = mid-span, 0.5 = at an existing tower); the largest over the wires this span crosses (14 Oct 2026);
    # and the crossing angle at EACH wire (audit, 14 Oct 2026: a span over two 66 kV ways was measured once)
    pos_of = getattr(chk, "_wire_pos_of", None)
    if pos_of is None:                     # built once per checker (it was rebuilt on every call)
        pos_of = chk._wire_pos_of = {id(ln): pos for _, ln, pos in chk._wire_list}
    head = math.degrees(math.atan2(q[1] - p[1], q[0] - p[0]))
    null, acute_min = None, None
    for t_, ln in hits:
        pos = pos_of.get(id(ln)) or []
        sx = ln.project(seg.interpolate(t_))
        w0, w1 = ln.interpolate(max(0.0, sx - 5.0)), ln.interpolate(min(ln.length, sx + 5.0))
        wa = math.degrees(math.atan2(w1.y - w0.y, w1.x - w0.x))
        d_ = (head - wa) % 180.0
        ac = min(d_, 180.0 - d_)
        acute_min = ac if acute_min is None else min(acute_min, ac)
        lo = max((v for v in pos if v <= sx), default=None)
        hi = min((v for v in pos if v >= sx), default=None)
        if lo is None or hi is None or hi - lo < 1.0:
            continue
        off = abs(sx - (lo + hi) / 2.0) / (hi - lo)
        null = off if null is None else max(null, off)
    return la.distance(_Pt(p)), lb.distance(_Pt(q)), ta, seg.length - tb, null, acute_min

def _wires(chk: Checker):
    """[(rule_id, wire LineString, sorted tower positions along it)] of the existing power lines (frame)."""
    if getattr(chk, "_wire_cache", None) is not None:
        return chk._wire_cache
    from shapely.geometry import LineString as _LS
    from shapely import STRtree
    lines, towers, seen = [], {}, set()
    for r in chk.tower_rules:                 # the approach and the normal compile carry the same features
        if r.rule_id not in ("power_line", "power_line_lt"):
            continue
        for g in r.geoms:
            if g.geom_type == "Point":
                key = (r.rule_id, round(g.x, 1), round(g.y, 1))
                if key not in seen:
                    seen.add(key)
                    towers.setdefault(r.rule_id, []).append(g)
            else:
                for x in getattr(g, "geoms", [g]):
                    key = (r.rule_id, x.wkb)
                    if x.geom_type == "LineString" and x.length > 0 and key not in seen:
                        seen.add(key)
                        lines.append((r.rule_id, x))
    out = []
    for rid, ln in lines:
        tw = towers.get(rid, [])
        tree = STRtree(tw) if tw else None
        pos = sorted(ln.project(tw[j]) for j in tree.query(ln.buffer(15.0)) if ln.distance(tw[j]) <= 15.0) if tree is not None else []
        out.append((rid, ln, pos))
    chk._wire_cache = out
    return out


def _refine_crossings(chk: Checker, pts, sp: SpotParams):
    """Each power-line crossing re-optimised on a fine grid: the crossing point slides along the existing wire
    within the allowed band (refine_step_m), the line's angle to it varies over the allowed window, the two
    crossing towers stand in (or, at a price, beyond) their distance window, and the spans out to the towers
    two before and three after are spotted again. Every candidate passes the exact checks; the cheapest
    (towers, line, RoW, plus the crossing preferences priced as in the search) replaces the coarse layout."""
    from shapely.geometry import LineString as _LS, Point as _Pt
    from .config import tower_type_for
    pts = list(pts)
    crs_all: dict = {}
    for cr_ in chk.c.crossings:
        if getattr(cr_, "side_m", None) is not None:
            crs_all.setdefault(cr_.rule_id, []).append(cr_)
    crs = {rid: lst[0] for rid, lst in crs_all.items()}
    if not crs:
        return pts
    fr_ = chk.c.frame

    def raster_at(rid, X):
        """The rule's raster whose angle field carries this wire (the parts by voltage share a rule id)."""
        col = int(np.clip((X.x - fr_.x0) / fr_.res_m, 0, fr_.width - 1))
        row = int(np.clip((fr_.y_top - X.y) / fr_.res_m, 0, fr_.height - 1))
        for cr_ in crs_all[rid]:
            if np.isfinite(cr_.angle[row, col]):
                return cr_
        return crs_all[rid][0]
    wires = _wires(chk)
    i = 3
    while i < len(pts) - 4:
        p0, q0 = pts[i], pts[i + 1]
        seg = _LS([p0, q0])
        hit = [(rid, ln, pos) for rid, ln, pos in wires if rid in crs and ln.intersects(seg)]
        if len(hit) != 1:
            i += 1
            continue
        rid, ln, pos = hit[0]
        X0 = ln.intersection(seg)
        X0 = X0 if X0.geom_type == "Point" else X0.centroid
        cr = raster_at(rid, X0)                      # audit, 14 Oct 2026: was the last part by rule id
        lo, hi = cr.side_m
        s0 = ln.project(X0)
        a_ = max([v for v in pos if v <= s0], default=None)
        b_ = min([v for v in pos if v >= s0], default=None)
        if a_ is not None and b_ is not None and b_ > a_:
            span = b_ - a_
            mid = (a_ + b_) / 2
            band = (mid - cr.null_allowed * span, mid + cr.null_allowed * span)
        else:
            span, mid, band = None, s0, (s0 - 60.0, s0 + 60.0)
        A, B = pts[i - 2], pts[i + 3]
        AB = (B[0] - A[0], B[1] - A[1])
        step = sp.refine_step_m
        n_s = int((band[1] - band[0]) // step) + 1
        sides = [lo + 2 + 6 * k for k in range(int((hi - lo - 4) // 6) + 1)]
        far = [hi + 10 * k for k in range(1, 7)] if getattr(cr, "side_relax_rs_per_m", 0) > 0 else []

        def cost_of(chain, sx, ang, da, db):
            inner = [tower_type_for(_deflection(chain[k - 1], chain[k], chain[k + 1])[0], sp.type_bands) or "D60"
                     for k in range(1, len(chain) - 1)]
            c = _chain_cost(chk.c, sp, chain, inner)
            if span:
                c += min(1.0, abs(sx - mid) / span / max(cr.null_allowed, 1e-6)) * cr.penalty_rs
            if cr.pref_window is not None and ang < cr.pref_window[0] - 1e-3:
                c += cr.penalty_rs
            c += (max(0.0, da - hi) + max(0.0, db - hi)) * getattr(cr, "side_relax_rs_per_m", 0.0)
            return c

        best = None
        legs_l: dict = {}
        legs_r: dict = {}
        j0 = i - 2

        def leg(cache, a, b, key, from_a):
            """(inner towers, cost of the leg's spans and inner towers) — cached per crossing-tower spot."""
            if key not in cache:
                inner = _spot_leg(chk, a, b, sp)
                if inner is None:
                    cache[key] = None
                else:
                    ch = [a] + list(inner) + [b]
                    ty = [tower_type_for(_deflection(ch[k - 1], ch[k], ch[k + 1])[0], sp.type_bands) or "D60"
                          for k in range(1, len(ch) - 1)]
                    cache[key] = (inner, _chain_cost(chk.c, sp, ch, ty))
            return cache[key]

        f_ = chk.c.frame

        def tower_rs(pt, prev, nxt):
            t = tower_type_for(_deflection(prev, pt, nxt)[0], sp.type_bands) or "D60"
            px = int(np.clip((pt[0] - f_.x0) / f_.res_m, 0, f_.width - 1))
            py = int(np.clip((f_.y_top - pt[1]) / f_.res_m, 0, f_.height - 1))
            return sp.tower_cost_rs[t] + _tower_site_rs(chk.c, t, px, py)

        for k in range(n_s):
            sx = band[0] + k * step
            if sx < 0 or sx > ln.length:
                continue
            X = ln.interpolate(sx)
            t1 = ln.interpolate(min(ln.length, sx + 1.0))
            t0 = ln.interpolate(max(0.0, sx - 1.0))
            wx, wy = t1.x - t0.x, t1.y - t0.y
            wn = math.hypot(wx, wy) or 1.0
            wx, wy = wx / wn, wy / wn
            for ang in range(int(cr.window[0]) if cr.window else 70, 91, 4):
                for sgn in ((1, -1) if ang < 90 else (1,)):
                    th = math.radians(ang) * sgn
                    s_th = max(math.sin(math.radians(ang)), 0.05)      # da / db are perpendicular to the wire (14 Oct 2026)
                    ux, uy = wx * math.cos(th) - wy * math.sin(th), wx * math.sin(th) + wy * math.cos(th)
                    if ux * AB[0] + uy * AB[1] < 0:
                        ux, uy = -ux, -uy
                    # legal crossing-tower spots on each side, with their legs out to A / B (cached per spot)
                    P_ok = []
                    for da in sides + far:
                        p = (X.x - ux * da / s_th, X.y - uy * da / s_th)
                        if chk.tower_ok(p[0], p[1], True):
                            lg = leg(legs_l, A, p, (round(p[0], 1), round(p[1], 1)), True)
                            if lg is not None:
                                P_ok.append((da, p, lg))
                    if not P_ok:
                        continue
                    Q_ok = []
                    for db in sides + far:
                        q = (X.x + ux * db / s_th, X.y + uy * db / s_th)
                        if chk.tower_ok(q[0], q[1], True):
                            lg = leg(legs_r, q, B, (round(q[0], 1), round(q[1], 1)), False)
                            if lg is not None:
                                Q_ok.append((db, q, lg))
                    for da, p, (left, lcost) in P_ok:
                        for db, q, (right, rcost) in Q_ok:
                            if (da + db) / s_th > sp.span_max_m:
                                continue
                            chain = [A] + list(left) + [p, q] + list(right) + [B]
                            if any(math.hypot(chain[m + 1][0] - chain[m][0], chain[m + 1][1] - chain[m][1]) < sp.min_span_m - 1e-6
                                   for m in range(len(chain) - 1)):
                                continue
                            full = pts[:i - 2] + chain + pts[i + 4:]
                            # every tower whose turn the new layout changes: A, the chain, B and B's successor
                            dev_rng = range(max(1, j0), min(len(full) - 1, j0 + len(chain) + 1))
                            if any(_deflection(full[m - 1], full[m], full[m + 1])[0] > sp.max_dev_deg + 1e-6 for m in dev_rng):
                                continue
                            # legs cached; the crossing span and the two crossing towers per pair
                            c = (lcost + rcost + _chain_cost(chk.c, sp, [p, q], [])
                                 + tower_rs(p, chain[len(left)], q) + tower_rs(q, p, chain[len(left) + 3]))
                            if span:
                                c += min(1.0, abs(sx - mid) / span / max(cr.null_allowed, 1e-6)) * cr.penalty_rs
                            if cr.pref_window is not None and ang < cr.pref_window[0] - 1e-3:
                                c += cr.penalty_rs
                            c += (max(0.0, da - hi) + max(0.0, db - hi)) * getattr(cr, "side_relax_rs_per_m", 0.0)
                            if best is not None and c >= best[0]:
                                continue
                            # the exact checks last, only for a candidate that would win
                            if not (chk.conductor_ok(p, q) and chk.span_crossings(p, q)[0]):
                                continue
                            if any(_deflection(full[m - 1], full[m], full[m + 1])[0] > _dev_cap(chk, sp, full[m - 1], full[m], full[m + 1]) + 1e-6
                                   for m in dev_rng):
                                continue
                            # the spans either side of the chain must still be legal with the new end towers
                            if j0 >= 1 and not (chk.conductor_ok(full[j0 - 1], full[j0]) and chk.span_crossings(full[j0 - 1], full[j0])[0]):
                                continue
                            best = (c, full, j0 + 1 + len(left))
        if best is not None:
            cur = pts[i - 2:i + 4]
            cur_ok = all(chk.tower_ok(v[0], v[1], True) for v in cur[1:-1]) and all(
                chk.conductor_ok(cur[m], cur[m + 1]) and chk.span_crossings(cur[m], cur[m + 1])[0] for m in range(len(cur) - 1))
            it0 = [x for x in chk.span_crossings(p0, q0)[1] if x.get("rule") == rid]
            cur_cost = cost_of(cur, s0, it0[0]["angle_deg"], it0[0]["side_a_m"], it0[0]["side_b_m"]) if it0 else float("inf")
            log.debug("refine %d-%d: coarse %.0f (legal %s), best %.0f", i, i + 1, cur_cost, cur_ok, best[0])
            if not cur_ok or best[0] < cur_cost - 1.0:
                pts = best[1]
                i = best[2] + 2
                continue
        else:
            log.info("crossing %d-%d (%s): no legal layout within the refinement reach", i, i + 1, rid)
        i += 1
    return pts


def _wind_span_fix(chk: Checker, pts, sp: SpotParams):
    """TSD wind span ≤ 450 m (base): next to a span longer than the longest span (a river crossing), put a
    tower on the neighbouring span close enough that the crossing tower's mean span stays within the
    limit (e.g. 700 m crossing → neighbour ≤ 100 m). Gantry spans are left alone."""
    wind = float((sp.tsd.get("wind_span_m") or {}).get("base", 400))
    pts = list(pts)
    i = 1
    while i < len(pts) - 2:
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L <= sp.span_max_m + 1e-3:
            i += 1
            continue
        w = wind
        if chk.dem is not None and getattr(chk.dem, "ok", False):
            # the crossing towers will carry the extension this span's sag needs; its step sets
            # their wind span (TSD: +18 / +33 / +45 m → 300 / 270 / 250 m)
            c0, _ = _span_clearance(chk, p, q, sp)
            need = 0.0 if c0 is None else max(0.0, sp.ground_clearance_m - c0)
            ws = sp.tsd.get("wind_span_m", {})
            # the closest step, as _clearance will give the crossing towers (Vivek, 9 Oct 2026); the
            # smallest step >= need here put towers 2 x 300 - L from the crossing towers needlessly
            st = nearest_step(need, _ext_steps(sp, "D45"))
            st = 45.0 if st is None else st
            if need > 0 and st >= 18:
                w = float(ws.get("d18", 300) if st < 33 else ws.get("d33", 270) if st < 45 else ws.get("d45", 250))
        allow = 2 * w - L                            # longest neighbour span at each crossing tower
        for side in (-1, +1):
            k = i if side < 0 else i + 1             # the crossing tower
            nb = k - 1 if side < 0 else k + 1        # its other neighbour
            if nb <= 0 or nb >= len(pts) - 1:
                continue
            a, b = pts[k], pts[nb]
            Ln = math.hypot(b[0] - a[0], b[1] - a[1])
            if Ln <= allow + 1e-3 or allow < sp.min_span_m:
                continue
            if Ln - allow < sp.min_span_m:
                # too little over the allowance for a new tower: slide the neighbour towards the
                # crossing tower instead (10 Oct 2026; this made 3-5 m spans before)
                moved = _shift_chain(chk, pts, nb, k, Ln - allow + 2.0, sp)
                if moved is not None:
                    pts[:] = moved
                    continue
            for dd in (allow - 2.0, allow - 10, allow - 20, allow - 30):     # 2 m margin: later passes may shift a tower slightly
                if dd < sp.min_span_m:
                    break
                if Ln - dd < sp.min_span_m - 1e-6:
                    continue        # the rest of the neighbour span would be below the minimum (9 Oct 2026 fix)
                t = dd / Ln
                x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
                if chk.tower_ok(x, y, angle_tower=False) and chk.span_crossings(a, (x, y))[0] \
                        and chk.span_crossings((x, y), b)[0]:
                    pos = k if side < 0 else k + 1
                    pts.insert(pos, (x, y))
                    if side < 0:
                        i += 1                       # the crossing span moved one place on
                    break
        i += 1
    return pts


def _angle_span_fix(chk: Checker, pts, sp: SpotParams):
    """TSD adjacent-span tables at angle towers: where the two spans at a tower exceed its type's
    allowance at that deviation (e.g. 189 m in total at 60°), put towers on the neighbouring spans,
    in line, close enough to the angle tower that the pair fits. The dead-end towers are left alone."""
    from .config import tower_type_for
    from .solver import fit_type_for_spans, span_sum_limit
    T = sp.tsd
    frac = float(T.get("single_span_fraction", 0.6))
    pts = list(pts)
    i = 2
    while i < len(pts) - 2:
        p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
        d, _ = _deflection(p0, p1, p2)
        t = tower_type_for(d, sp.type_bands) or "D60"
        L1 = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        L2 = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if fit_type_for_spans(T["span_sum"], t, d, L1, L2, frac) is not None:
            i += 1
            continue
        lim = span_sum_limit(T["span_sum"], "D60" if t in ("D45", "D60") and d > 45 else t, d) or 0.0
        a = max(sp.min_span_m, min(lim / 2.0, frac * lim))
        inserted = 0
        for side, (q, L) in ((-1, (p0, L1)), (+1, (p2, L2))):
            if L <= a + 1e-3:
                continue
            for dd in (a, a - 10, a - 20):
                if dd < sp.min_span_m:
                    break
                if L - dd < sp.min_span_m - 1e-6:
                    continue        # the rest of the span would be shorter than the minimum (9 Oct 2026 fix)
                tt = dd / L
                x, y = p1[0] + tt * (q[0] - p1[0]), p1[1] + tt * (q[1] - p1[1])
                if chk.tower_ok(x, y, angle_tower=False) and chk.span_crossings(p1, (x, y))[0] \
                        and chk.span_crossings((x, y), q)[0]:
                    if side < 0:
                        pts.insert(i, (x, y))
                        i += 1
                    else:
                        pts.insert(i + 1, (x, y))
                    inserted += 1
                    break
        i += 1
    return pts


def _span_clearance(chk: Checker, p, q, sp: SpotParams, ext_a=0.0, ext_b=0.0):
    """(min clearance m, t at the worst point) of the conductor over the DEM for a straight span."""
    pr = chk.profile(p, q, step=15.0)
    if pr is None:
        return None, None
    t, ground, L = pr
    if not np.isfinite(ground[0]) or not np.isfinite(ground[-1]):
        return None, None
    sag = sp.sag_at_design_m * (L / (sp.sag_ref_span_m or sp.design_span_m)) ** 2
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
        # 18 Oct 2026: a hilly span up to the longest plain span stays (it pays the search charge instead)
        if 0 < i < n - 2 and max(sp.hilly_design_span_m, _plain_max(sp)) < L <= sp.span_max_m + 1e-3 and chk.is_hilly(p, q):
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
    """Insert suspension towers where a span cannot clear the ground even with the largest A/B/C
    extension (+6 m, TSD; +18 m without TSD data) on both towers: at the worst point (a ridge), or the nearest allowed spot along the span.
    Gantry spans and long single crossing spans are left alone. Returns (points, inserted indices)."""
    pts = list(pts)
    flags = [False] * len(pts)
    i, new = 1, 0                      # span pts[i] -> pts[i+1]; skip the gantry spans
    while i < len(pts) - 2 and new < max_new:
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        e = _max_ext(sp, "C1")
        c, tw = _span_clearance(chk, p, q, sp, e, e)
        # closest-step rule (9 Oct 2026): +6 m on both meets a shortfall up to halfway to the next step
        nxt = [x for x in _ext_steps(sp, "D45") if x > e] if sp.tsd else []
        tol = (min(nxt) - e) / 2.0 if nxt else 0.0
        if c is None or c >= sp.ground_clearance_m - tol or L < 2 * sp.min_span_m or L > sp.span_max_m + 1e-3:
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
    sag = sp.sag_at_design_m * (s.length_m / (sp.sag_ref_span_m or sp.design_span_m)) ** 2

    def min_clear():
        za = ta.ground_z + sp.lowest_attachment_m + ta.extension_m
        zb = tb.ground_z + sp.lowest_attachment_m + tb.extension_m
        line = (1 - t) * za + t * zb - 4 * sag * t * (1 - t)
        return float(np.nanmin(line - ground))

    c = min_clear()
    if c < sp.ground_clearance_m:
        # Closest-step extensions (9 Oct 2026): one raised tower or both, whichever is cheaper; each
        # extension is the step closest to what the shortfall asks, so a small shortfall may remain
        # (the DEM and the sag are estimates). Gain at the worst point: e × (1 - t) on a, e × t on b.
        line_needed = sp.ground_clearance_m - c
        za = ta.ground_z + sp.lowest_attachment_m + ta.extension_m
        zb = tb.ground_z + sp.lowest_attachment_m + tb.extension_m
        lin = (1 - t) * za + t * zb - 4 * sag * t * (1 - t)
        k = int(np.nanargmin(lin - ground))
        fa, fb = max(1e-3, 1.0 - t[k]), max(1e-3, t[k])
        reason = "crossing_tower" if s.length_m > sp.span_max_m + 1e-3 else "extension"
        cands = []
        reachable = False
        for tw, frac in ((ta, fa), (tb, fb)):
            o = _ext_option(tw, tw.extension_m + line_needed / frac, sp)
            reachable |= o is not None
            if o is not None and o[1] > tw.extension_m + 1e-9:
                cands.append(((_tower_rs(sp, *o) - _tower_rs(sp, tw.type, tw.extension_m)), [(tw, o)]))
        oa = _ext_option(ta, ta.extension_m + line_needed, sp)
        ob = _ext_option(tb, tb.extension_m + line_needed, sp)
        reachable |= oa is not None and ob is not None
        if oa is not None and ob is not None and (oa[1] > ta.extension_m + 1e-9 or ob[1] > tb.extension_m + 1e-9):
            cands.append(((_tower_rs(sp, *oa) - _tower_rs(sp, ta.type, ta.extension_m))
                          + (_tower_rs(sp, *ob) - _tower_rs(sp, tb.type, tb.extension_m)), [(ta, oa), (tb, ob)]))
        if cands:
            cands.sort(key=lambda x: (x[0], len(x[1])))
            for tw, o in cands[0][1]:
                _apply_ext(tw, o, reason)
            c = min_clear()
        if not reachable:
            ta.warnings.append("ground_clearance")      # beyond what the closest step can meet
    s.min_clearance_m = round(c, 1)


def _ext_steps(sp: SpotParams, t: str) -> list[float]:
    if sp.tsd and sp.tsd.get("extensions_m"):
        return [float(x) for x in sp.tsd["extensions_m"].get(t, [])]
    return [EXTENSION_M] if t != "G" else []


def nearest_step(need_m: float, steps) -> float | None:
    """Body extension rounded to the closest available step, 0 included (Vivek, 9 Oct 2026: 6-12 m
    is met by +6 m, above 12 m by +18 m — the DEM and sag figures are estimates, so the extension
    is not maximised). Beyond the largest step: the largest if within half the last gap, else None."""
    opts = sorted({0.0, *[float(x) for x in steps]})
    if need_m <= 0:
        return 0.0
    if need_m > opts[-1]:
        gap = opts[-1] - opts[-2] if len(opts) > 1 else 0.0
        return opts[-1] if need_m - opts[-1] <= gap / 2 + 1e-9 else None
    best = None
    for o in opts:            # ties go to the higher step
        if best is None or abs(need_m - o) < abs(need_m - best) - 1e-9 or \
                (abs(abs(need_m - o) - abs(need_m - best)) <= 1e-9 and o > best):
            best = o
    return best


def rounding_cover(steps) -> float:
    """The largest shortfall the closest-step rule still meets with these steps (e.g. +6 m below an
    +18 m step: up to 12 m)."""
    opts = sorted({0.0, *[float(x) for x in steps]})
    return opts[-1] + ((opts[-1] - opts[-2]) / 2 if len(opts) > 1 else 0.0)


def _max_ext(sp: SpotParams, t: str) -> float:
    st = _ext_steps(sp, t)
    return max(st) if st else 0.0


def _ext_option(tw, need_m: float, sp: SpotParams):
    """(type, extension) for a tower that needs need_m in total: its own type's closest step, else a
    D45's closest step; None if neither reaches by rounding."""
    if tw.type == "G":
        return None
    own = _ext_steps(sp, tw.type)
    e = nearest_step(need_m, own)
    if own and sp.tsd and tw.type not in ("D45", "D60"):
        # one ladder: above the type's largest step the next rung is a D's (+18 m), so the largest
        # step serves up to halfway there (6-12 m -> +6 m; Vivek, 9 Oct 2026)
        top = max(own)
        nxt = [x for x in _ext_steps(sp, "D45") if x > top]
        if need_m > top:
            e = top if nxt and need_m <= (top + min(nxt)) / 2.0 + 1e-9 else None
    if e is not None:
        return tw.type, max(tw.extension_m, e)
    if tw.type not in ("D45", "D60") and sp.tsd:
        e = nearest_step(need_m, _ext_steps(sp, "D45"))
        if e is not None:
            return "D45", max(tw.extension_m, e)
    return None


def _apply_ext(tw, opt, reason: str) -> None:
    t, e = opt
    if t != tw.type:
        tw.type = t
        tw.reasons.append("extension_type")
    if e > tw.extension_m + 1e-9:
        tw.extension_m = e
        if reason not in tw.reasons:
            tw.reasons.append(reason)


def _tower_rs(sp: SpotParams, t: str, ext: float) -> float:
    return sp.tower_cost_rs.get(t, 0.0) * (1.0 + sp.extension_pct / 100.0 * ext / 18.0)


def _raise(tw, need_m: float, sp: SpotParams, reason: str) -> None:
    """Give the tower the closest extension step to need_m (rounded, not maximised); a D45 when its
    own type's steps cannot reach."""
    if tw.type == "G" or need_m <= tw.extension_m:
        return
    opt = _ext_option(tw, need_m, sp)
    if opt is None:
        st = _ext_steps(sp, "D45" if sp.tsd else tw.type)
        opt = ("D45" if sp.tsd and tw.type not in ("D45", "D60") else tw.type, max(st) if st else tw.extension_m)
    _apply_ext(tw, opt, reason)


def _tsd_checks(towers: list[Tower], spans: list[Span], sp: SpotParams) -> None:
    """AESL Tower Spotting Data checks on the spotted line (8 Oct 2026):
    - sum of adjacent spans by tower type and deviation, one span ≤ 60 % of it;
    - wind span (mean of the adjacent spans) ≤ 450 m (base, 18 Oct 2026), D with +18/+33/+45 m: 300/270/250 m;
    - weight span within the type's limits — a tower in a dip (uplift) is upgraded to a type
      whose limit holds (A → B1 → D45), else flagged;
    - dead-end towers: deviation ≤ 15° line side + 30° slack side (note J).
    Breaches that a type upgrade cannot fix are warnings ("verify on site")."""
    from .solver import fit_type_for_spans, span_sum_limit
    T = sp.tsd
    n = len(towers)
    frac = float(T.get("single_span_fraction", 0.6))
    ws = T.get("wind_span_m", {})
    wl = T.get("weight_span_m", {})
    hw = float(T.get("tension_over_weight_m", 1605))
    order = ["A", "B1", "C1", "D45", "D60"]

    def wind_limit(t):
        if t.type in ("D45", "D60") and t.extension_m >= 45:
            return ws.get("d45", 250)
        if t.type in ("D45", "D60") and t.extension_m >= 33:
            return ws.get("d33", 270)
        if t.type in ("D45", "D60") and t.extension_m >= 18:
            return ws.get("d18", 300)
        return ws.get("base", 400)

    for i in range(1, n - 1):
        t = towers[i]
        if t.type == "G":
            continue
        L1, L2 = spans[i - 1].length_m, spans[i].length_m
        dead_end = "dead_end" in t.reasons
        if dead_end:
            if t.deflection_deg > float(T.get("dead_end_max_dev_deg", 180)) + 1e-6:
                t.violations.append(f"dead_end_deviation:{t.deflection_deg:.0f}deg")     # Vivek's 15° rule (audit, 14 Oct 2026)
            lim = span_sum_limit(T["span_sum"], t.type, t.deflection_deg)
            if lim is not None and max(L1, L2) > frac * lim + 1e-3:
                t.warnings.append(f"span_sum_single:{max(L1, L2):.0f}/{frac * lim:.0f}m")
            continue
        nt = fit_type_for_spans(T["span_sum"], t.type, t.deflection_deg, L1, L2, frac)
        if nt is None:
            lim = span_sum_limit(T["span_sum"], t.type, t.deflection_deg)
            t.warnings.append(f"span_sum:{L1 + L2:.0f}/{lim:.0f}m")
        elif nt != t.type:
            t.type = nt
            t.reasons.append("span_table")
        if (L1 + L2) / 2.0 > wind_limit(t) + 1e-3:
            t.warnings.append(f"wind_span:{(L1 + L2) / 2:.0f}m")
        a, b = towers[i - 1], towers[i + 1]
        if t.ground_z is None or a.ground_z is None or b.ground_z is None or L1 <= 0 or L2 <= 0:
            continue
        att = sp.lowest_attachment_m
        h = t.ground_z + att + t.extension_m
        ha = a.ground_z + (0.0 if a.type == "G" else att + a.extension_m)
        hb = b.ground_z + (0.0 if b.type == "G" else att + b.extension_m)
        w = (L1 + L2) / 2.0 + hw * ((h - ha) / L1 + (h - hb) / L2)
        lo, hi = wl.get(t.type, [-1e9, 1e9])
        if w < lo:
            # uplift (a tower lower than its neighbours, e.g. next to a raised crossing tower): the
            # cheapest of raising this tower by a body extension step or a stronger type (9 Oct 2026:
            # fewer angle-type towers on straight line). Raising it by e adds hw·e·(1/L1 + 1/L2).
            gain = hw * (1.0 / L1 + 1.0 / L2)
            best = None
            for nt in order[order.index(t.type):]:
                for e in sorted({t.extension_m, *[x for x in _ext_steps(sp, nt) if x >= t.extension_m]}):
                    if w + gain * (e - t.extension_m) >= wl.get(nt, [-1e9])[0] - 1e-6:
                        rs = _tower_rs(sp, nt, e)
                        if best is None or rs < best[0] - 1e-6:
                            best = (rs, nt, e)
                        break                       # the smallest step that holds for this type
            if best is None:
                t.warnings.append(f"weight_span:{w:.0f}m")
            else:
                _, nt, e = best
                if nt != t.type:
                    t.type = nt
                    t.reasons.append("weight_span")
                if e > t.extension_m + 1e-9:
                    t.extension_m = e
                    t.reasons.append("weight_span_extension")
        elif w > hi:
            t.warnings.append(f"weight_span:{w:.0f}m")


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
    relaxed = 0
    for s in spans:
        for c in s.crossings:
            cross_counts[c["rule"]] = cross_counts.get(c["rule"], 0) + 1
            if not c.get("angle_preferred", True) or c.get("point_preferred") is False:
                relaxed += 1
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
        **_hill_stats(comp, towers),
        "agri_ha": round(areas["agriculture"] + areas["orchard"], 2),
        "corridor_ha": round(sum(areas.values()), 2),
        "areas_ha": {k: round(v, 2) for k, v in areas.items()},
        "crossings": cross_counts,
        "crossings_total": sum(cross_counts.values()),
        "crossings_relaxed": relaxed,                 # outside the preferred angle / null point (13 Oct 2026)
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
