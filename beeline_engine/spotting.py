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
    raster_prefilter: bool = True         # skip the vector tower rules in clearly free cells (11 Oct 2026)
    # hilly terrain (rules design_span.hilly_m, section_interval.hilly_km; engine hilly_slope_deg)
    hilly_design_span_m: float | None = None
    hilly_section_km: float | None = None
    hilly_slope_deg: float | None = None
    hilly_rise_m: float | None = None      # hilly span: ground rises more than this between its towers (Route_Rules 3)
    tsd: dict | None = None                # AESL Tower Spotting Data tables for this voltage (engine.yaml tsd)
    approach_length_m: float = 0.0         # approach section within this of each gantry (CEA Schedule VII)
    approach_span_m: float | None = None
    proximity_discount: float = 0.0         # search-only discount near existing lines (string pulling)
    extension_pct: float = 9.4              # body extension price, % of the tower per +18 m
    sag_ref_span_m: float | None = None     # span the sag figure belongs to (None = design span)
    section_spans: int | None = None       # TSD D: ≤ 15 spans between section points (plain)
    hilly_section_spans: int | None = None # ≤ 10 spans (hilly)
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
        # Raster pre-filter for the exact tower checks (11 Oct 2026): a cell two cells clear of every
        # forbidden cell is more than the rule distance from any feature (the rule rasters are built from
        # cell-centre distances, so a point can be up to ~28 m nearer a feature than its cell says; two
        # 20 m cells cover that). Only "clearly free" skips the vector rules; anything else is checked.
        self._clear = ~ndimage.binary_dilation(comp.tower_forbidden, iterations=2)
        self._clear_angle = ~ndimage.binary_dilation(comp.angle_forbidden, iterations=2)
        self._verify = os.environ.get("BEELINE_VERIFY_PREFILTER") == "1"
        # (A combined shortlist STRtree over every rule's buffered features was tried on 11 Oct 2026 and
        # dropped: ~900 k buffers took 500 s to build per compile and the line queries were slower than
        # the per-rule dwithin queries.)

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

    def conductor_ok(self, p, q) -> bool:
        line = LineString([p, q])
        # go zone: crossings of >= 220 kV lines near their towers stay banned (clipped to the zone)
        if self.c.go_conductor and any(r.hits(line) for r in self.c.go_conductor):
            return False
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
            first = int(np.argmax(hit))
            last = len(hit) - 1 - int(np.argmax(hit[::-1]))
            fa = float(a[first])
            d = (head - fa) % 180.0
            acute = min(d, 180.0 - d)
            # the crossed feature sits at the middle of its hit samples; distances from each tower to it
            mid = (first + last) / 2.0 * step
            item = {"rule": cr.rule_id, "name": cr.name, "angle_deg": round(acute, 1), "span_m": round(L, 1),
                    "min_type": cr.min_type, "cost_rs": cr.cost_rs, "major": False,
                    "side_a_m": round(mid, 1), "side_b_m": round(L - mid, 1)}
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
                mk, _ = self._sample(pm, p, q)
                item["point_preferred"] = bool(mk[(first + last) // 2])
            if cr.window is not None and acute < cr.window[0] - 1e-3:
                ok = False
                item["violation"] = "angle"
            if cr.max_span_m is not None and L > cr.max_span_m + 1e-3:
                ok = False
                item["violation"] = "span"
            side = getattr(cr, "side_m", None)
            if side is not None:
                lo, hi = side
                item["side_window_m"] = [lo, hi]
                if not (lo - 1e-3 <= mid <= hi + 1e-3 and lo - 1e-3 <= L - mid <= hi + 1e-3):
                    ok = False
                    item["violation"] = "side"
            if cr.width_raster is not None and cr.major_threshold_m is not None:
                w, st = self._sample(cr.width_raster, p, q)
                width = float(w.sum()) * st
                item["width_m"] = round(width, 1)
                if width >= cr.major_threshold_m:
                    item.update(major=True, min_type=cr.major_type, cost_rs=cr.major_cost_rs)
            out.append(item)
        # hard maximum: the plain design span, or the (longer) hilly one on hilly ground; rivers excepted
        limit = self.sp.span_max_m
        if self.sp.approach_span_m is not None and self.in_approach(p, q):
            limit = min(limit, self.sp.approach_span_m)
        if L > self.sp.design_span_m + 1e-3 and not self.is_hilly(p, q):
            limit = self.sp.design_span_m
        if L > limit + 1e-3:
            if L > self.sp.river_max_span_m + 1e-3 or not any(c.get("long_span") for c in out):
                ok = False
        crossed_rules = {c.get("rule") for c in out}
        for pr in self.c.parallels:
            if getattr(pr, "crossing_exempt", False) and pr.rule_id in crossed_rules:
                continue                    # the crossing span itself is not a parallel run (canal)
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
        if _even_leg_ok(chk, p, q, pts, span, n, hilly_leg, sp):
            return pts
        if 1 < n <= min(n0 + 1, UNEVEN_MAX_SPANS):
            # equal spacing fails: uneven spacing with the same tower count (9 Oct 2026)
            un = _uneven_leg(chk, p, q, n, design, sp)
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
    if hilly_leg and span > sp.design_span_m + 1e-3 and n > 1 and \
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
        span_c = L * (comp.conductor_rs_per_m + dens * width)
        if sp.proximity_discount > 0 and comp.near_existing is not None:
            col = np.clip(((fx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
            row = np.clip(((f.y_top - fy) / f.res_m).astype(int), 0, f.height - 1)
            span_c *= 1.0 - sp.proximity_discount * float(comp.near_existing[row, col].mean())
        c += span_c
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
        long_span = max(L1, L2) > sp.design_span_m + 1e-3
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
    pts = _crossing_span_fix(chk, pts, sp)
    if sp.tsd:
        pts = _wind_span_fix(chk, pts, sp)
        pts = _angle_span_fix(chk, pts, sp)
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
    if _deflection(prev_, cand, next_)[0] > sp.max_dev_deg:
        return False
    if i >= 2 and _deflection(pts[i - 2], prev_, cand)[0] > sp.max_dev_deg:
        return False
    if i + 2 < n and _deflection(cand, next_, pts[i + 2])[0] > sp.max_dev_deg:
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
        for rad in range(5, int(sp.nudge_radius_m) + 6, 5):
            angles = sorted((2 * math.pi * k / 24 for k in range(24)),
                            key=lambda a: abs(math.cos(a) * hx / hn + math.sin(a) * hy / hn))
            for a in angles:
                c = (x + rad * math.cos(a), y + rad * math.sin(a))
                if not chk.tower_ok(c[0], c[1], angle_tower=True):
                    continue
                if _legal_here(chk, pts, i, sp, c):
                    found = c
                    break
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
        if not items or all(it.get("violation") != "side" for it in items):
            i += 1
            continue
        it = items[0]
        lo, hi = it["side_window_m"]
        ux, uy = (q[0] - p[0]) / L, (q[1] - p[1]) / L
        X = (p[0] + ux * it["side_a_m"], p[1] + uy * it["side_a_m"])       # the crossed wire on the alignment
        # the straight run this span belongs to: from the previous turn (or dead-end) to the next
        run_start = i
        while run_start > 1 and _deflection(pts[run_start - 1], pts[run_start], pts[run_start + 1])[0] < 1.0:
            run_start -= 1
        run_end = i + 1
        while run_end < n - 2 and _deflection(pts[run_end - 1], pts[run_end], pts[run_end + 1])[0] < 1.0:
            run_end += 1
        done = False
        mid = (lo + hi) / 2
        # every pair of distances in the window on a 4 m step, nearest the middle first
        steps = [lo + 2 + 4 * k for k in range(int((hi - lo - 4) // 4) + 1)]
        combos = sorted(((da, db) for da in steps for db in steps), key=lambda c_: abs(c_[0] - mid) + abs(c_[1] - mid))
        for da, db in combos:
            p2 = (X[0] - ux * da, X[1] - uy * da)
            q2 = (X[0] + ux * db, X[1] + uy * db)
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


def _wind_span_fix(chk: Checker, pts, sp: SpotParams):
    """TSD wind span ≤ 400 m: next to a span longer than the design span (a river crossing), put a
    tower on the neighbouring span close enough that the crossing tower's mean span stays within the
    limit (e.g. 700 m crossing → neighbour ≤ 100 m). Gantry spans are left alone."""
    wind = float((sp.tsd.get("wind_span_m") or {}).get("base", 400))
    pts = list(pts)
    i = 1
    while i < len(pts) - 2:
        p, q = pts[i], pts[i + 1]
        L = math.hypot(q[0] - p[0], q[1] - p[1])
        if L <= sp.design_span_m + 1e-3:
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
    - wind span (mean of the adjacent spans) ≤ 400 m, D with +18/+33/+45 m: 300/270/250 m;
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
                t.warnings.append(f"dead_end_deviation:{t.deflection_deg:.0f}deg")
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
