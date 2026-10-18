"""Pylon-graph solver: layer-wise dynamic programme on GPU.

Candidate tower sites sit on a grid in the chainage frame (columns = chainage, rows = lateral
offset). A span is one of K offsets (dc, dr) whose length is within [Smin, Smax]; dc >= 1, so
the graph is a DAG ordered by chainage. State = (site, incoming offset k). The tower at a site
gets its type, hence its cost, from the deflection between the incoming and outgoing offsets.

  M[c, r, k'] = S[c, r] + min_k ( V[c, r, k] + T[k, k'] )        tower at (c, r)
  W[c, r, k'] = M[c, r, k'] + E[c, r, k']                          span leaving (c, r)
  V[c+dc, r+dr, k'] = W[c, r, k']                                  arrival

Exact for this graph. Detours that go backwards along the bee-line are not representable (v1).
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage
from shapely.geometry import Point

from .compile import Compiled
from .config import TOWER_TYPES, max_type, tower_type_for

log = logging.getLogger(__name__)
INF = float("inf")


@dataclass
class SolverParams:
    site_spacing_m: float
    span_min_m: float
    span_max_m: float
    max_dev_deg: float
    type_bands: dict[str, float]
    tower_cost_rs: dict[str, float]          # by type, excluding site cost
    parallel_angle_deg: float = 30.0
    # long spans, valid only across a river (0 disables)
    river_max_span_m: float = 0.0
    site_tolerance_m: float = 0.0            # search only: a site counts as allowed if a legal cell lies within
                                             # this distance (the spotting snaps the tower to it)
    crossing_span_slack_m: float = 0.0       # search only: a crossing span may exceed its limit by this
                                             # much (half a site spacing); the spotting enforces the exact limit
    long_heading_limit_deg: float = 60.0
    long_heading_step_deg: float = 5.0
    long_length_step_m: float = 100.0
    # variant penalties (route 2-4 and diversity)
    angle_tower_penalty_rs: float = 0.0
    lateral_penalty_rs_per_m: float = 0.0    # per tower, per metre of lateral offset
    sensitive_multiplier: float = 1.0
    extra_strip_rs_per_m: np.ndarray | None = None   # raster, ₹ per metre of line (diversity)
    proximity_discount: float = 0.0          # search-only discount on line + RoW cost near existing lines
    ext_zero_tol_m: float = 0.0              # closest-step rounding: shortfall met by no extension
    ext_round_tol_m: float = 0.0             # ... beyond the largest A/B/C step, met by that step
    # Heading quantisation (6 Oct 2026): consecutive span vectors that differ by no more than the grid's
    # own heading step (atan(site spacing / span length), 7.1° at 400 m) stand for a straight line at
    # an in-between heading, which the spotting straightens into suspension towers. Pricing them as
    # B1/B2 made every stretch not parallel to the bee-line ~₹0.3 Cr per tower too dear (the search
    # ranked a hill route ₹22 Cr below a flat one that was ₹3 Cr cheaper once spotted).
    heading_slack: str | bool = False        # False | True (any bend <= step) | "adjacent" (same span length, one row apart)
    # terrain in the search (6 Oct 2026): clearance per span, hilly span limit, hilly section towers
    terrain: bool = False
    ground_clearance_m: float = 0.0
    lowest_attachment_m: float = 0.0
    sag_at_design_m: float = 0.0
    design_span_m: float = 400.0
    sag_ref_span_m: float | None = None       # span the sag figure belongs to (None = design span)
    extension_m: float = 18.0
    extension_cost_rs: float = 0.0           # one body extension on a suspension (A) tower
    extension_cost_hilly_rs: float = 0.0     # one body extension on a hilly-ground (B1) tower
    hilly_span_max_m: float | None = None
    plain_span_max_m: float | None = None    # spans with no hilly ground stay within this (hilly may be longer)
    hilly_tower_extra_rs: float = 0.0        # hilly ground: towers are tension (B1), ₹ over an A per span
    hilly_surcharge_rs_per_m: float = 0.0    # search-only: avoid hills (Vivek, 14 Oct 2026)
    hilly_tower_surcharge_rs: float = 0.0
    hilly_rise_m: float = 10.0               # hilly span: ground rises more than this between its towers (Route_Rules 3)
    extension_pct: float = 0.0               # body extension price, % of the tower price per +18 m
    # TSD span tables (765 kV): {type: [[deg, sum_m], ...]}; None = not applied
    span_sum: dict | None = None
    single_span_fraction: float = 0.6
    wind_span_m: float | None = None
    # long (river) spans under the TSD: D extension steps and the wind span each step allows
    long_ext_steps: tuple = ()                 # e.g. ((6, 400), (18, 300), (33, 270), (45, 250))
    long_ext_cost_rs_per_m: float = 0.0        # ₹ per metre of extension on one D tower
    dead_end_max_dev_deg: float | None = None  # TSD note J: 15° line side + 30° slack side
    approach_length_m: float = 0.0             # approach section within this of each gantry (0 = none)
    approach_span_m: float | None = None       # its design span (CEA Schedule VII)
    edge_cache_limit_bytes: float = 8e9        # GPU bytes the edge-geometry cache may take (11 Oct 2026)
    crossing_dev_max_deg: float | None = None  # deviation cap at the two towers of a crossing span (Vivek, 13 Oct 2026)
    gantry_min_span_m: float = 30.0            # gantry to the first / last tower, at least (Vivek, 13 Oct 2026: 100 m)
    proximity_tower_discount: float = 0.0      # search-only discount on towers near existing lines (Vivek, 13 Oct 2026)


@dataclass
class Grid:
    nc: int
    nr: int
    r0: int
    sx: float
    sy: float
    offsets: np.ndarray      # (K, 2) int  (dc, dr)
    heading_deg: np.ndarray  # (K,)
    length_m: np.ndarray     # (K,)
    is_long: np.ndarray      # (K,) bool: river-only long span

    def site_xy(self, c, r):
        return np.asarray(c) * self.sx, (np.asarray(r) - self.r0) * self.sy


@dataclass
class SolveOutcome:
    feasible: bool
    cost_rs: float
    sites: list[tuple[int, int]]          # (c, r) towers from the first dead-end to the last
    headings: list[int]                   # offset index of each span
    grid: Grid
    reach_chainage_m: float               # furthest chainage reached (for BAND_BLOCKED)
    seconds: float
    reach_map: np.ndarray | None = None   # (nc+1, R) bool: a tower can stand here on some feasible path from A


def make_grid(compiled: Compiled, p: SolverParams) -> Grid:
    f = compiled.frame
    nc = max(2, int(round(f.length_m / p.site_spacing_m)))
    sx = f.length_m / nc
    sy = p.site_spacing_m
    half_rows = int(math.floor(f.half_width_m / sy))
    nr = 2 * half_rows + 1
    r0 = half_rows
    dcmax = int(math.ceil(p.span_max_m / sx))
    drmax = int(math.ceil(p.span_max_m / sy))
    offs = []
    for dc in range(1, dcmax + 1):
        for dr in range(-drmax, drmax + 1):
            L = math.hypot(dc * sx, dr * sy)
            if p.span_min_m - 1e-6 <= L <= p.span_max_m + 1e-6:
                offs.append((dc, dr))
    n_short = len(offs)
    if p.river_max_span_m > p.span_max_m:
        seen = set(offs)
        lengths = []
        Lv = p.span_max_m + p.long_length_step_m
        while Lv <= p.river_max_span_m + 1e-6:
            lengths.append(Lv)
            Lv += p.long_length_step_m
        if not lengths or lengths[-1] < p.river_max_span_m - 1e-6:
            lengths.append(p.river_max_span_m)          # the limit itself (TSD: 720 m)
        for Lv in lengths:
            th = -p.long_heading_limit_deg
            while th <= p.long_heading_limit_deg + 1e-6:
                dc = int(round(Lv * math.cos(math.radians(th)) / sx))
                dr = int(round(Lv * math.sin(math.radians(th)) / sy))
                L = math.hypot(dc * sx, dr * sy)
                if dc >= 1 and p.span_max_m + 1e-6 < L <= p.river_max_span_m + 1e-6 and (dc, dr) not in seen:
                    seen.add((dc, dr))
                    offs.append((dc, dr))
                th += p.long_heading_step_deg
    offs = np.array(offs, dtype=np.int64)
    head = np.degrees(np.arctan2(offs[:, 1] * sy, offs[:, 0] * sx))
    length = np.hypot(offs[:, 0] * sx, offs[:, 1] * sy)
    is_long = np.zeros(len(offs), bool)
    is_long[n_short:] = True
    return Grid(nc=nc, nr=nr, r0=r0, sx=sx, sy=sy, offsets=offs, heading_deg=head, length_m=length,
                is_long=is_long)


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_strip(compiled: Compiled, p: SolverParams) -> np.ndarray:
    """₹ per metre of line at each raster cell: corridor-width box mean of the ₹/m² density."""
    res = compiled.frame.res_m
    width_px = max(1, int(round(2 * compiled.half_width_m / res)))
    dens = compiled.strip_density.astype(np.float32)
    if p.sensitive_multiplier != 1.0:
        dens = dens * (1.0 + (p.sensitive_multiplier - 1.0) * compiled.sensitive)
    sm = ndimage.uniform_filter(dens, size=width_px, mode="nearest")
    strip = sm * (2 * compiled.half_width_m)
    if p.extra_strip_rs_per_m is not None:
        strip = strip + p.extra_strip_rs_per_m
    return strip.astype(np.float32)


class _Rasters:
    def __init__(self, compiled: Compiled, p: SolverParams, dev, strip_np: np.ndarray | None = None):
        self.dev = dev
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=dev)
        self.cond = t(compiled.conductor_forbidden, torch.bool)
        self.strip = t(strip_np if strip_np is not None else prepare_strip(compiled, p))
        self.near = t(compiled.near_existing, torch.float32) if p.proximity_discount > 0 and \
            getattr(compiled, "near_existing", None) is not None else None
        self.cross = [(cr, t(np.nan_to_num(cr.angle, nan=-1.0)),
                       None if cr.width_raster is None else t(cr.width_raster, torch.bool),
                       None if getattr(cr, "null_offset_wide", None) is None else t(np.nan_to_num(cr.null_offset_wide, nan=-1.0)))
                      for cr in compiled.crossings]
        self.par = [t(np.nan_to_num(pr.angle, nan=-1.0)) for pr in compiled.parallels]
        # village-cluster breach band (Vivek, 18 Oct 2026): a span sampled inside it costs breach_rs
        self.breach = t(compiled.breach_band, torch.bool) if getattr(compiled, "breach_band", None) is not None and \
            getattr(compiled, "breach_rs", 0.0) > 0 else None
        self.breach_rs = float(getattr(compiled, "breach_rs", 0.0) or 0.0)
        self.par_meta = [(pr.rule_id, getattr(pr, "crossing_exempt", False)) for pr in compiled.parallels]
        self.h, self.w = compiled.tower_forbidden.shape
        self.f_x0, self.f_y_top, self.res_m = float(compiled.frame.x0), float(compiled.frame.y_top), float(compiled.frame.res_m)
        self.river_cols = None
        self.z = self.hilly = None
        if p.terrain and compiled.ground_z is not None:
            self.z = t(compiled.ground_z)
            self.hilly = t(compiled.hilly, torch.bool)

    def set_river_cols(self, compiled: Compiled, grid: "Grid", p: SolverParams):
        """Grid columns from which a long span could reach a river."""
        f = compiled.frame
        riv = [cr for cr in compiled.crossings if getattr(cr, "long_span", cr.rule_id == "river")]
        if not riv or p.river_max_span_m <= p.span_max_m:
            return
        has = np.zeros(f.width, bool)
        for cr in riv:
            has |= ~np.isnan(cr.angle).all(axis=0)
        xs = f.x0 + (np.nonzero(has)[0] + 0.5) * f.res_m
        cols = np.zeros(grid.nc + 1, bool)
        cx = np.arange(grid.nc + 1) * grid.sx
        for x in xs:
            cols |= (cx <= x + f.res_m) & (cx >= x - p.river_max_span_m)
        self.river_cols = cols


def _acute(a_deg, b_deg):
    d = torch.remainder(a_deg - b_deg, 180.0)
    return torch.minimum(d, 180.0 - d)


_CACHE_MUTEX = threading.Lock()


def _edge_index(grid: Grid, rasters: "_Rasters", c_from: int, c_to: int, kidx: np.ndarray, n_samples: int, rows=None):
    """Raster rows / cols sampled along every edge of the block, plus the edge geometry tensors."""
    dev = rasters.dev
    f_x0, f_y_top, res = rasters.f_x0, rasters.f_y_top, rasters.res_m
    C = c_to - c_from
    rlo, rhi = (0, grid.nr) if rows is None else rows
    R, K = rhi - rlo, len(kidx)
    offs = grid.offsets[kidx]
    cs = torch.arange(c_from, c_to, device=dev, dtype=torch.float32)
    rs = torch.arange(rlo, rhi, device=dev, dtype=torch.float32)
    x0 = (cs * grid.sx).view(C, 1, 1, 1)
    y0 = ((rs - grid.r0) * grid.sy).view(1, R, 1, 1)
    dx = torch.as_tensor(offs[:, 0] * grid.sx, device=dev, dtype=torch.float32).view(1, 1, K, 1)
    dy = torch.as_tensor(offs[:, 1] * grid.sy, device=dev, dtype=torch.float32).view(1, 1, K, 1)
    tt = torch.linspace(0, 1, n_samples, device=dev).view(1, 1, 1, n_samples)
    col = ((x0 + tt * dx - f_x0) / res).long().clamp_(0, rasters.w - 1)
    row = ((f_y_top - (y0 + tt * dy)) / res).long().clamp_(0, rasters.h - 1)
    return row, col, x0, y0, dx, dy, tt


def _edge_strip(grid: Grid, rasters: "_Rasters", c_from: int, c_to: int, kidx: np.ndarray, n_samples: int) -> torch.Tensor:
    """Mean RoW strip ₹/m along every edge of the block -> (C, R, K). The only part of the edge cost
    that changes between the forest-cap passes and the route variants (11 Oct 2026)."""
    row, col, *_ = _edge_index(grid, rasters, c_from, c_to, kidx, n_samples)
    return rasters.strip[row, col].mean(dim=-1)


def _edge_block(compiled: Compiled, grid: Grid, p: SolverParams, rasters: "_Rasters",
                c_from: int, c_to: int, kidx: np.ndarray, n_samples: int, require_river: bool,
                cache: dict | None = None, rows=None) -> torch.Tensor:
    """E for columns c_from..c_to-1 and offsets kidx -> (C, R, len(kidx)) ₹, inf = invalid.
    E = B + A × strip, with (A, B) the strip-independent geometry of the block; `cache` (shared by the
    solves on one compile: the route variants and the forest-cap passes) keeps (A, B) on the GPU."""
    key = (c_from, c_to, int(require_river), int(n_samples), int(len(kidx)), grid.nc, grid.nr, len(grid.offsets), "u1")
    if rows is not None:
        cache = None                                   # a row window (refinement pass) is never cached
    geom = cache.get(key) if cache is not None else None
    if geom is None:
        if cache is None:
            geom = _edge_geom(compiled, grid, p, rasters, c_from, c_to, kidx, n_samples, require_river, rows)
        else:
            # one route computes a block while the others wait for it instead of computing it too
            with _CACHE_MUTEX:
                lock = cache.setdefault("__locks__", {}).setdefault(key, threading.Lock())
            with lock:
                geom = cache.get(key)
                if geom is None:
                    geom = _edge_geom(compiled, grid, p, rasters, c_from, c_to, kidx, n_samples, require_river)
                    if rasters.dev.type == "cuda":
                        # the routes solve on their own CUDA streams: the block must be complete before
                        # another stream reads it from the cache
                        torch.cuda.current_stream(rasters.dev).synchronize()
                    cache[key] = geom
    A, B, X, U = geom
    strip = _edge_strip(grid, rasters, c_from, c_to, kidx, n_samples)
    if rows is not None:
        strip = strip[:, rows[0]:rows[1]]
    return B + A * strip, X, U


def _edge_geom(compiled: Compiled, grid: Grid, p: SolverParams, rasters: "_Rasters",
               c_from: int, c_to: int, kidx: np.ndarray, n_samples: int, require_river: bool, rows=None):
    """(A, B, X, U) for the block: E = B + A × mean strip ₹/m; B carries inf where the edge is invalid; X marks
    edges that cross a feature with a deviation cap at its crossing towers; U is the ₹ over an A that a
    crossing forces on each end tower (charged once at the tower, in solve())."""
    dev = rasters.dev
    f = compiled.frame
    C = c_to - c_from
    rlo, rhi = (0, grid.nr) if rows is None else rows
    R, K = rhi - rlo, len(kidx)
    offs = grid.offsets[kidx]
    cs = torch.arange(c_from, c_to, device=dev, dtype=torch.float32)
    rs = torch.arange(rlo, rhi, device=dev, dtype=torch.float32)
    x0 = (cs * grid.sx).view(C, 1, 1, 1)
    y0 = ((rs - grid.r0) * grid.sy).view(1, R, 1, 1)
    dx = torch.as_tensor(offs[:, 0] * grid.sx, device=dev, dtype=torch.float32).view(1, 1, K, 1)
    dy = torch.as_tensor(offs[:, 1] * grid.sy, device=dev, dtype=torch.float32).view(1, 1, K, 1)
    tt = torch.linspace(0, 1, n_samples, device=dev).view(1, 1, 1, n_samples)
    col = ((x0 + tt * dx - f.x0) / f.res_m).long().clamp_(0, rasters.w - 1)
    row = ((f.y_top - (y0 + tt * dy)) / f.res_m).long().clamp_(0, rasters.h - 1)
    length = torch.as_tensor(grid.length_m[kidx], device=dev, dtype=torch.float32).view(1, 1, K)
    heading = torch.as_tensor(grid.heading_deg[kidx], device=dev, dtype=torch.float32).view(1, 1, K, 1)
    A = length.expand(C, R, K)                       # ₹ per (₹/m of strip): the span length …

    invalid = rasters.cond[row, col].any(dim=-1)
    if p.approach_length_m > 0 and p.approach_span_m is not None and not require_river:
        # approach section: spans whose midpoint lies within approach_length_m of a gantry keep to
        # the approach design span
        xm = (x0 + dx / 2.0).squeeze(-1)
        ym = (y0 + dy / 2.0).squeeze(-1)
        in_ap = (torch.hypot(xm, ym) <= p.approach_length_m) | \
                (torch.hypot(xm - f.length_m, ym) <= p.approach_length_m)
        invalid |= in_ap & (length > p.approach_span_m + 1e-3)
    E = (length * compiled.conductor_rs_per_m).expand(C, R, K)
    U = torch.zeros(E.shape, device=dev)            # ₹ over an A forced on EACH end tower (hilly span, crossing)
    if rasters.near is not None:
        # stay close to existing lines: search-only discount by the share of the span near one
        fac = 1.0 - p.proximity_discount * rasters.near[row, col].mean(dim=-1)
        E = E * fac
        A = A * fac                                  # … times the same discount
    if rasters.z is not None and not require_river:
        # ground clearance of this span over the DEM (towers at standard height): extensions cost
        # steel; a span that extensions on both towers cannot clear is not allowed (a tower must
        # stand on the high ground instead, or the line goes round)
        zs = rasters.z[row, col]                                         # (C,R,K,S)
        za, zb = zs[..., :1], zs[..., -1:]
        sag = p.sag_at_design_m * (length.unsqueeze(-1) / (p.sag_ref_span_m or p.design_span_m)) ** 2
        line = (1 - tt) * (za + p.lowest_attachment_m) + tt * (zb + p.lowest_attachment_m) - 4 * sag * tt * (1 - tt)
        clear = torch.nan_to_num(line - zs, nan=1e9).amin(dim=-1)        # no DEM value: no penalty
        need = p.ground_clearance_m - clear
        # closest-step extensions (9 Oct 2026): a shortfall below half the smallest step needs none;
        # one raised tower gains ~half its extension at mid-span; the largest A/B/C step (+6 m) is
        # taken up to halfway to the next step (+18 m), i.e. a 12 m shortfall
        z0, cover = p.ext_zero_tol_m, p.extension_m + p.ext_round_tol_m
        one = (need > z0) & (need <= cover * 0.5)
        two = (need > cover * 0.5) & (need <= cover)
        invalid |= need > cover
        # hilly span: the ground rises more than hilly_rise_m between its towers (Route_Rules item 3)
        hil_any = (torch.nan_to_num((zb - za).abs().squeeze(-1), nan=0.0) > p.hilly_rise_m)
        ext = torch.where(hil_any, torch.full_like(E, p.extension_cost_hilly_rs), torch.full_like(E, p.extension_cost_rs))
        E = E + one.float() * ext + two.float() * (2 * ext)
        if p.hilly_span_max_m is not None:
            invalid |= hil_any & (length > p.hilly_span_max_m + 1e-3)
        if p.plain_span_max_m is not None:
            invalid |= ~hil_any & (length > p.plain_span_max_m + 1e-3)
        # hilly ground: every tower is a tension tower (Vivek, 7 Oct 2026) — charged at the tower as a forced
        # type, once (audit, 14 Oct 2026: it was added per span on top of the deflection type at bent towers)
        U = torch.maximum(U, hil_any.float() * p.hilly_tower_extra_rs)
    if rasters.z is not None and require_river and p.long_ext_steps:
        # long crossing span (TSD): both crossing towers raised to clear the ground at this span's
        # sag; the extension step sets the wind span the towers allow, so the span plus a minimum
        # neighbour span must fit 2 × wind span; beyond the largest step the span is not allowed
        zs = rasters.z[row, col]
        za, zb = zs[..., :1], zs[..., -1:]
        sag = p.sag_at_design_m * (length.unsqueeze(-1) / (p.sag_ref_span_m or p.design_span_m)) ** 2
        line = (1 - tt) * (za + p.lowest_attachment_m) + tt * (zb + p.lowest_attachment_m) - 4 * sag * tt * (1 - tt)
        need = (p.ground_clearance_m - torch.nan_to_num(line - zs, nan=1e9).amin(dim=-1)).clamp(min=0.0)
        ok_any = torch.zeros_like(invalid)
        ext_used = torch.zeros_like(E)
        # the closest step (0 included): step s serves shortfalls between the midpoints to its neighbours
        steps = sorted(p.long_ext_steps)
        sv = [0.0] + [float(s_) for s_, _ in steps]
        ok_any |= need <= sv[1] / 2.0 + 1e-6                    # rounds to no extension
        for j, (step, wind) in enumerate(steps, start=1):
            hi_ = (sv[j] + sv[j + 1]) / 2.0 if j + 1 < len(sv) else sv[j] + (sv[j] - sv[j - 1]) / 2.0
            fits = (~ok_any) & (need <= hi_ + 1e-6) & (length + p.span_min_m <= 2 * wind + 1e-6)
            ext_used = torch.where(fits, torch.full_like(E, float(step)), ext_used)
            ok_any |= fits
        invalid |= ~ok_any
        E = E + 2 * ext_used * p.long_ext_cost_rs_per_m
    if rasters.breach is not None:
        # village cluster within the normal distance (Vivek, 18 Oct 2026): search cost per breached span
        hit_b = rasters.breach[row, col].any(dim=-1)
        E = E + torch.where(hit_b, torch.full_like(E, rasters.breach_rs), torch.zeros_like(E))
    crossed_river = torch.zeros_like(invalid)
    X = torch.zeros_like(invalid)                   # crosses a feature with a deviation cap at its towers

    tA = p.tower_cost_rs["A"]
    crossed_by: dict = {}
    for cr, ang_t, width_t, pref_t in rasters.cross:
        a = ang_t[row, col]                        # (C,R,K,S), -1 = none
        hit = a >= 0
        crossed = hit.any(dim=-1)
        if not bool(crossed.any()):
            continue
        first = torch.argmax(hit.to(torch.int8), dim=-1, keepdim=True)
        fa = torch.gather(a, -1, first).squeeze(-1)
        acute = _acute(heading.squeeze(-1), fa)
        bad = torch.zeros_like(crossed)
        if cr.window is not None:
            bad |= acute < cr.window[0] - 1e-3
        if cr.max_span_m is not None:
            bad |= length > cr.max_span_m + p.crossing_span_slack_m + 1e-3
        if cr.side_m is not None:
            # crossing towers this far from the crossed wire, each side (Vivek, 13 Oct 2026); the wire sits at
            # the middle of the hit samples; the lattice gets the crossing-span slack, the spotting is exact
            # tower a to the first wire it crosses, tower b to the last (one span may cross two lines side by side)
            last = (n_samples - 1) - torch.argmax(hit.flip(-1).to(torch.int8), dim=-1)
            step = length / float(n_samples - 1)
            # perpendicular distance to the wire itself (Vivek, 14 Oct 2026): along-span distance to the band edge
            # x sin(crossing angle) + the band half-width; the spotting measures it exactly
            sin_c = torch.sin(torch.deg2rad(acute)).clamp(min=0.05)
            d1 = first.squeeze(-1).float() * step * sin_c + float(cr.band_half_m)
            d2 = (length - last.float() * step) * sin_c + float(cr.band_half_m)
            lo, hi = cr.side_m
            sl = p.crossing_span_slack_m
            # the minimum keeps the lattice slack: with it removed the search closed the Shivlaka end (rev 30);
            # the fine re-optimisation round each crossing (spotting) places the towers exactly (14 Oct 2026)
            bad |= (d1 < lo - sl) | (d2 < lo - sl)
            if cr.side_relax_rs_per_m > 0:
                # beyond the maximum only as far as needed (Vivek, 14 Oct 2026): priced per metre of excess
                excess = (d1 - hi - sl).clamp(min=0.0) + (d2 - hi - sl).clamp(min=0.0)
                side_cost = excess * float(cr.side_relax_rs_per_m)
            else:
                bad |= (d1 > hi + sl) | (d2 > hi + sl)
        if cr.side_m is None or cr.side_relax_rs_per_m <= 0:
            side_cost = None
        pen = torch.zeros_like(crossed)
        if cr.pref_window is not None:
            pen |= acute < cr.pref_window[0] - 1e-3
        # graded null-point penalty (Vivek, 13 Oct 2026): 0 at the null point rising linearly to the full
        # allowance at the edge of the allowed band; read at the middle of the hit samples (the wire itself)
        pos_frac = torch.zeros_like(E)
        if pref_t is not None:
            lastp = (n_samples - 1) - torch.argmax(hit.flip(-1).to(torch.int8), dim=-1, keepdim=True)
            midp = ((first + lastp) // 2).clamp_(0, n_samples - 1)
            off = torch.gather(pref_t[row, col], -1, midp).squeeze(-1)
            pos_frac = torch.where(off >= 0, (off / max(cr.null_allowed, 1e-6)).clamp(0.0, 1.0), torch.zeros_like(off))
        if cr.max_dev_deg is not None:
            X |= crossed
        min_type = cr.min_type
        cost = cr.cost_rs
        if width_t is not None and cr.major_threshold_m is not None:
            step = length / (n_samples - 1)
            width = width_t[row, col].sum(dim=-1).float() * step
            major = width >= cr.major_threshold_m
            add = torch.where(major, torch.full_like(E, cr.major_cost_rs), torch.full_like(E, cost))
            up = torch.where(major, torch.full_like(E, max(0.0, p.tower_cost_rs[cr.major_type] - tA)),
                             torch.full_like(E, max(0.0, p.tower_cost_rs[min_type] - tA)))
        else:
            add = torch.full_like(E, cost)
            up = torch.full_like(E, max(0.0, p.tower_cost_rs[min_type] - tA))
        # the forced type is charged at the tower, once, as max(type by deflection, forced type) — see solve()
        # (14 Oct 2026: it used to be added here for both ends on top of the deflection type, so a bend at a
        # crossing tower cost the search twice and the route bent elsewhere)
        U = torch.maximum(U, torch.where(crossed, up, torch.zeros_like(E)))
        if getattr(cr, "extension_m", 0):
            add = add + 2 * p.tower_cost_rs[min_type] * p.extension_pct / 100.0 * cr.extension_m / 18.0
        if getattr(cr, "long_span", cr.rule_id == "river"):
            crossed_river |= crossed
        crossed_by[cr.rule_id] = crossed_by.get(cr.rule_id, torch.zeros_like(crossed)) | crossed
        E = E + torch.where(crossed, add, torch.zeros_like(E))
        if cr.penalty_rs > 0:
            # non-preferred angle: the full allowance; position: graded by the distance from the null point
            E = E + torch.where(crossed & pen, torch.full_like(E, float(cr.penalty_rs)), torch.zeros_like(E))
            E = E + torch.where(crossed, pos_frac * float(cr.penalty_rs), torch.zeros_like(E))
        if side_cost is not None:
            E = E + torch.where(crossed, side_cost, torch.zeros_like(E))
        if getattr(cr, "span_band", None) is not None and getattr(cr, "band_penalty_rs", 0.0) > 0:
            # SH crossing span band (Vivek, 16 Oct 2026): outside it needs an approval; the lattice gets the slack
            blo, bhi = cr.span_band
            sl = p.crossing_span_slack_m
            off_band = (length < blo - sl - 1e-3) | (length > bhi + sl + 1e-3)
            E = E + torch.where(crossed & off_band, torch.full_like(E, float(cr.band_penalty_rs)), torch.zeros_like(E))
        invalid |= crossed & bad
    for pa, (rid, exempt) in zip(rasters.par, rasters.par_meta):
        a = pa[row, col]
        near = a >= 0
        if not bool(near.any()):
            continue
        acute = _acute(heading.expand_as(a), a)
        par_bad = (near & (acute < p.parallel_angle_deg)).any(dim=-1)
        if exempt and rid in crossed_by:
            par_bad = par_bad & ~crossed_by[rid]          # the crossing span itself is not a parallel run
        invalid |= par_bad
    if require_river:
        invalid |= ~crossed_river
    return A, torch.where(invalid, torch.full_like(E, INF), E), X, U


def edge_costs(compiled: Compiled, grid: Grid, p: SolverParams, rasters: "_Rasters",
               c_from: int, c_to: int, n_samples: int, cache: dict | None = None, rows=None) -> torch.Tensor:
    """(E, X, U): E[c, r, k] for columns c_from..c_to-1 -> tensor (C, R, K) of ₹ (inf = invalid); X marks edges
    that cross a feature whose crossing towers carry a deviation cap; U the forced-type ₹ over an A per end. Long offsets are evaluated only
    where a long-span feature lies within reach."""
    dev = rasters.dev
    C, R, K = c_to - c_from, grid.nr, len(grid.offsets)
    short = np.nonzero(~grid.is_long)[0]
    E = torch.full((C, R, K), INF, device=dev)
    X = torch.zeros((C, R, K), dtype=torch.bool, device=dev)
    U = torch.zeros((C, R, K), device=dev)
    rsl = slice(0, R) if rows is None else slice(rows[0], rows[1])     # refinement pass: this row window only
    E[:, rsl, short], X[:, rsl, short], U[:, rsl, short] = _edge_block(compiled, grid, p, rasters, c_from, c_to, short, n_samples, False, cache, rows)
    long_ = np.nonzero(grid.is_long)[0]
    if len(long_) and rasters.river_cols is not None and rasters.river_cols[c_from:c_to].any():
        n_long = int(math.ceil(p.river_max_span_m / (compiled.frame.res_m / 2))) + 1
        sub = 8
        for c0 in range(c_from, c_to, sub):
            c1 = min(c_to, c0 + sub)
            if not rasters.river_cols[c0:c1].any():
                continue
            Eb, Xb, Ub = _edge_block(compiled, grid, p, rasters, c0, c1, long_, n_long, True, cache, rows)
            E[c0 - c_from:c1 - c_from, rsl, long_] = Eb
            X[c0 - c_from:c1 - c_from, rsl, long_] = Xb
            U[c0 - c_from:c1 - c_from, rsl, long_] = Ub
    return E, X, U


def site_costs(compiled: Compiled, grid: Grid, p: SolverParams) -> tuple[np.ndarray, np.ndarray]:
    """(S, angle_only): S[c, r] ₹ type-independent tower cost (inf if forbidden);
    angle_only[c, r] True where only type A towers may stand."""
    f = compiled.frame
    cc, rr = np.meshgrid(np.arange(grid.nc + 1), np.arange(grid.nr), indexing="ij")
    fx, fy = grid.site_xy(cc, rr)
    px, py = f.frame_to_pixel(fx, fy)
    col = np.clip(px.astype(int), 0, f.width - 1)
    row = np.clip(py.astype(int), 0, f.height - 1)
    S = compiled.site_cost[row, col].astype(np.float64)
    if p.proximity_tower_discount > 0 and getattr(compiled, "near_existing", None) is not None:
        S = S * (1.0 - p.proximity_tower_discount * compiled.near_existing[row, col])
    S = S + p.lateral_penalty_rs_per_m * np.abs(fy)
    forbidden = compiled.tower_forbidden
    if p.site_tolerance_m > 0:
        # a lattice site is usable when a legal cell lies within site_tolerance_m of it (10 Oct 2026:
        # a 47 m legal strip between two lines' setbacks fell between 50 m sites); the spotting then
        # snaps the tower onto that cell under the exact checks
        from scipy import ndimage
        r = int(math.ceil(p.site_tolerance_m / f.res_m))
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        forbidden = ndimage.binary_erosion(forbidden, structure=(xx * xx + yy * yy) <= r * r, border_value=1)
    S[forbidden[row, col]] = INF
    return S.astype(np.float32), compiled.angle_forbidden[row, col]


def span_sum_limit(table: dict, t: str, d: float) -> float | None:
    """Permissible sum of adjacent spans (m) for tower type t at deviation d (TSD table 3). Below
    the first tabulated angle the first value applies; between entries the stricter (next) one."""
    rows = table.get(t)
    if not rows:
        return None
    if d <= rows[0][0]:
        return float(rows[0][1])
    for (a0, v0), (a1, v1) in zip(rows[:-1], rows[1:]):
        if d <= a1:
            return float(v1)
    return float(rows[-1][1])


def fit_type_for_spans(table: dict, t: str, d: float, L1: float, L2: float, frac: float) -> str | None:
    """Lightest type from t upward whose adjacent-span allowance at deviation d takes spans L1, L2."""
    from .config import TOWER_TYPES
    for tt in TOWER_TYPES[TOWER_TYPES.index(t):]:
        lim = span_sum_limit(table, tt, d)
        if lim is None or (L1 + L2 <= lim + 1e-6 and max(L1, L2) <= frac * lim + 1e-6):
            return tt
    return None


def transition_matrix(grid: Grid, p: SolverParams) -> tuple[np.ndarray, np.ndarray]:
    """T[k, k'] ₹ tower cost by deflection type; TA: same but only type A allowed."""
    h = grid.heading_deg
    defl = np.abs(h[:, None] - h[None, :])
    K = len(h)
    step = np.degrees(np.arctan(grid.sy / np.maximum(grid.length_m, 1.0)))     # heading step per vector
    if p.heading_slack == "adjacent":
        # only the two vectors a straight line at an in-between heading alternates between: same
        # chainage advance, one row apart (e.g. (8,0) and (8,1): 7.1° at 400 m)
        dc, dr = grid.offsets[:, 0], grid.offsets[:, 1]
        adj = (dc[:, None] == dc[None, :]) & (np.abs(dr[:, None] - dr[None, :]) == 1)
        slack = np.where(adj, np.maximum(step[:, None], step[None, :]) + 0.5, 0.0)
    elif p.heading_slack:
        slack = np.maximum(step[:, None], step[None, :]) + 0.5
    else:
        slack = np.zeros((K, K))
    T = np.full((K, K), INF, np.float32)
    TA = np.full((K, K), INF, np.float32)
    for i in range(K):
        for j in range(K):
            d = defl[i, j]
            if d > p.max_dev_deg + 1e-9:
                continue
            t = tower_type_for(0.0 if d <= slack[i, j] else d, p.type_bands)
            if t is None:
                continue
            if p.span_sum is not None:
                # TSD: the span pair must fit the type's adjacent-span table at this deviation; a
                # heavier type with a longer allowance is taken when the banded one does not fit
                Li, Lj = float(grid.length_m[i]), float(grid.length_m[j])
                if p.wind_span_m is not None and (Li + Lj) / 2.0 > p.wind_span_m + 1e-6:
                    continue
                t = fit_type_for_spans(p.span_sum, t, d, Li, Lj, p.single_span_fraction)
                if t is None:
                    continue
            c = p.tower_cost_rs[t]
            if t != "A":
                c += p.angle_tower_penalty_rs
            T[i, j] = c
            if t == "A":
                TA[i, j] = c
    return T, TA


def gantry_seeds(compiled: Compiled, grid: Grid, p: SolverParams, S_np: np.ndarray, strip_np: np.ndarray,
                 at_end: bool, dead_end_rs: float) -> np.ndarray:
    """Cost of gantry -> first (or last -> gantry) dead-end tower at every site, inf if not allowed.

    The gantry is A (or B). The slack span may run over the terminal yard, so its reach is the
    max span plus the yard's extent around the gantry. The span must keep clear of conductor
    no-go cells and must not cross a controlled feature."""
    f = compiled.frame
    ex = f.length_m if at_end else 0.0
    yard_extent = 0.0
    if compiled.terminal_yards is not None:
        ep = Point(ex, 0.0)
        for y in getattr(compiled.terminal_yards, "geoms", [compiled.terminal_yards]):
            if y.distance(ep) < 1.0:
                minx, miny, maxx, maxy = y.bounds
                yard_extent = max(yard_extent, max(abs(minx - ex), abs(maxx - ex), abs(miny), abs(maxy)) * math.sqrt(2))
    reach = min(3000.0, p.span_max_m + yard_extent)
    if p.approach_length_m > 0 and p.approach_span_m is not None:
        reach = min(reach, p.approach_span_m)     # the gantry span keeps the approach span (9 Oct 2026)
    seeds = np.full(S_np.shape, INF, np.float32)
    cs = np.arange(grid.nc + 1)
    xs = cs * grid.sx
    col_sel = np.nonzero(np.abs(xs - ex) <= reach)[0]
    rs = np.arange(grid.nr)
    ys = (rs - grid.r0) * grid.sy
    row_sel = np.nonzero(np.abs(ys) <= reach)[0]
    cross = [np.nan_to_num(cr.angle, nan=-1.0) for cr in compiled.crossings]
    for c in col_sel:
        for r in row_sel:
            if not np.isfinite(S_np[c, r]):
                continue
            x, y = xs[c], ys[r]
            L = math.hypot(x - ex, y)
            if L < p.gantry_min_span_m or L > reach:
                continue
            n = max(2, int(math.ceil(L / (f.res_m / 2))) + 1)
            t = np.linspace(0, 1, n)
            fx = ex + t * (x - ex)
            fy = t * y
            col = np.clip(((fx - f.x0) / f.res_m).astype(int), 0, f.width - 1)
            row = np.clip(((f.y_top - fy) / f.res_m).astype(int), 0, f.height - 1)
            cf = compiled.conductor_forbidden[row, col]
            tc = getattr(compiled, "terminal_cond", None)
            if tc is not None:
                cf = cf & ~tc[row, col]                  # the gantry span itself may leave the substation
            if cf.any():
                continue
            if any((a[row, col] >= 0).any() for a in cross):
                continue
            seeds[c, r] = S_np[c, r] + dead_end_rs + L * (compiled.conductor_rs_per_m + float(strip_np[row, col].mean()))
    return seeds


def solve(compiled: Compiled, p: SolverParams, cancel_check=None, progress=None, corridor=None) -> SolveOutcome:
    """A and B are gantries. The route is gantry A -> dead-end tower -> ... -> dead-end tower -> gantry B.
    corridor = (line, half_width_m): the search is confined to the sites within half_width_m of the
    frame-coordinate line (the targeted refinement pass on a finer lattice, 14 Oct 2026)."""
    t0 = time.time()
    dev = _device()
    grid = make_grid(compiled, p)
    R, K = grid.nr, len(grid.offsets)
    nc = grid.nc
    win = None                                   # (nc+1, 2) row window per column, or None = every row
    if corridor is not None:
        line, half = corridor
        xs = np.asarray([pt[0] for pt in line.coords]); ys = np.asarray([pt[1] for pt in line.coords])
        cx = np.arange(nc + 1) * grid.sx
        yc = np.interp(cx, xs, ys)
        lo = np.clip(np.floor((yc - half) / grid.sy).astype(int) + grid.r0, 0, R - 1)
        hi = np.clip(np.ceil((yc + half) / grid.sy).astype(int) + grid.r0 + 1, 1, R)
        win = np.stack([lo, hi], axis=1)
    S_np, angle_only_np = site_costs(compiled, grid, p)
    T_np, TA_np = transition_matrix(grid, p)
    T45_np = TA45_np = None
    if p.crossing_dev_max_deg is not None and p.crossing_dev_max_deg < p.max_dev_deg - 1e-9:
        # the two towers of a crossing span turn at most crossing_dev_max_deg (Vivek, 13 Oct 2026)
        from dataclasses import replace as _replace
        T45_np, TA45_np = transition_matrix(grid, _replace(p, max_dev_deg=float(p.crossing_dev_max_deg)))
    strip_np = prepare_strip(compiled, p)
    dead_end = p.tower_cost_rs["D60"]
    seedA = gantry_seeds(compiled, grid, p, S_np, strip_np, at_end=False, dead_end_rs=dead_end)
    seedB = gantry_seeds(compiled, grid, p, S_np, strip_np, at_end=True, dead_end_rs=dead_end)
    if not np.isfinite(seedA).any() or not np.isfinite(seedB).any():
        side = 0.0 if not np.isfinite(seedA).any() else compiled.frame.length_m
        return SolveOutcome(False, INF, [], [], grid, side, time.time() - t0)
    S = torch.as_tensor(S_np, device=dev)
    # tower price discount per site near an existing line (search only, Vivek 13 Oct 2026): scales the
    # type-by-deflection cost the DP adds at that site
    tfac = None
    if p.proximity_tower_discount > 0 and getattr(compiled, "near_existing", None) is not None:
        f_ = compiled.frame
        cc_, rr_ = np.meshgrid(np.arange(nc + 1), np.arange(R), indexing="ij")
        fx_, fy_ = grid.site_xy(cc_, rr_)
        px_, py_ = f_.frame_to_pixel(fx_, fy_)
        nr_ = compiled.near_existing[np.clip(py_.astype(int), 0, f_.height - 1), np.clip(px_.astype(int), 0, f_.width - 1)]
        tfac = torch.as_tensor(1.0 - p.proximity_tower_discount * nr_.astype(np.float32), device=dev)
    angle_only = torch.as_tensor(angle_only_np, device=dev)
    T = torch.as_tensor(T_np, device=dev)
    TA = torch.as_tensor(TA_np, device=dev)
    T45 = torch.as_tensor(T45_np, device=dev) if T45_np is not None else None
    TA45 = torch.as_tensor(TA45_np, device=dev) if TA45_np is not None else None
    SA = torch.as_tensor(seedA, device=dev)
    SB = torch.as_tensor(seedB, device=dev)
    rasters = _Rasters(compiled, p, dev, strip_np)
    rasters.set_river_cols(compiled, grid, p)
    n_samples = int(math.ceil(p.span_max_m / (compiled.frame.res_m / 2))) + 1

    dcs = torch.as_tensor(grid.offsets[:, 0], device=dev)
    drs = torch.as_tensor(grid.offsets[:, 1], device=dev)
    dcmax = int(grid.offsets[:, 0].max())
    ring = dcmax + 1
    Wring = torch.full((ring, R, K), INF, device=dev)
    Xring = torch.zeros((ring, R, K), dtype=torch.bool, device=dev)       # the edge into each site crossed a capped feature
    Uring = torch.zeros((ring, R, K), device=dev)                         # forced-type ₹ the edge into each site carries
    tA_f = float(p.tower_cost_rs["A"])
    arg = torch.zeros((nc + 1, R, K), dtype=torch.int16, device=dev) if win is None else None
    arg_cols: list = [None] * (nc + 1)           # (lo, (hi-lo, K) int16) per column with a row window
    rows = torch.arange(R, device=dev).view(R, 1)
    src_rows = rows - drs.view(1, K)                 # (R, K)
    row_ok = (src_rows >= 0) & (src_rows < R)
    src_rows_c = src_rows.clamp(0, R - 1)
    kidx = torch.arange(K, device=dev).view(1, K).expand(R, K)
    reach = torch.zeros(nc + 1, dtype=torch.bool, device=dev)
    reach_map = torch.zeros((nc + 1, R), dtype=torch.bool, device=dev)
    end_tot = torch.full((nc + 1, R), INF, device=dev)
    end_k = torch.zeros((nc + 1, R), dtype=torch.int64, device=dev)
    end_cols = set(np.nonzero(np.isfinite(seedB).any(axis=1))[0].tolist())
    start_marker = torch.tensor(-1, dtype=torch.int16, device=dev)
    ys_t = torch.as_tensor((np.arange(R) - grid.r0) * grid.sy, device=dev, dtype=torch.float32)
    head_t = torch.as_tensor(grid.heading_deg, device=dev, dtype=torch.float32)

    budget = 1.25e8 if rasters.z is not None else 2.5e8               # terrain doubles the gathered samples
    if win is None:
        chunk = max(1, min(32, int(budget // max(1, R * K * n_samples))))
    else:
        # the refinement pass gathers its row window only; the window of a chunk is the union over its columns
        # (the corridor drifts sideways), so size for twice the tallest column window and cap the chunk — a
        # 64-column chunk asked for 14 GB on a shared GPU (14 Oct 2026)
        R_eff = int((win[:, 1] - win[:, 0]).max())
        chunk = max(1, min(16, int(budget // max(1, 2 * R_eff * K * n_samples))))
    # edge-geometry cache (11 Oct 2026): (A, B) per column block, shared by every solve on this compile;
    # two float32 (nc, R, K) tensors on the GPU — skipped when that would exceed the configured limit
    cache = getattr(compiled, "edge_cache", None)
    if cache is not None and (nc + 1) * R * K * 8 > p.edge_cache_limit_bytes:
        cache = None
    E_chunk, E_start = None, -1
    for c in range(nc + 1):
        if cancel_check and c % 64 == 0:
            cancel_check()
        # arrivals at column c
        if c == 0:
            V = torch.full((R, K), INF, device=dev)
            cross_in = torch.zeros((R, K), dtype=torch.bool, device=dev)
            up_in = torch.zeros((R, K), device=dev)
        else:
            src_c = c - dcs
            ok_c = src_c >= 0
            ring_idx = torch.remainder(src_c, ring).view(1, K).expand(R, K)
            V = Wring[ring_idx, src_rows_c, kidx]
            valid_in = row_ok & ok_c.view(1, K)
            V = torch.where(valid_in, V, torch.full_like(V, INF))
            cross_in = Xring[ring_idx, src_rows_c, kidx] & valid_in
            up_in = torch.where(valid_in, Uring[ring_idx, src_rows_c, kidx], torch.zeros((R, K), device=dev))
            reach[c] = torch.isfinite(V).any()
        if c in end_cols:
            if p.dead_end_max_dev_deg is not None:
                hs = torch.rad2deg(torch.atan2(-ys_t, torch.full_like(ys_t, max(compiled.frame.length_m - c * grid.sx, 1.0)))).view(R, 1)
                dev_ok = (head_t.view(1, K) - hs).abs() <= p.dead_end_max_dev_deg + 1e-6
                Vend = torch.where(dev_ok, V, torch.full_like(V, INF))
            else:
                Vend = V
            vmin, vk = Vend.min(dim=1)
            end_tot[c] = vmin + SB[c]
            end_k[c] = vk
        if c == nc:
            break
        if E_chunk is None or c >= E_start + E_chunk.shape[0]:
            E_start = c
            c_end = min(nc, c + chunk)
            rows = None if win is None else (int(win[c:c_end + 1, 0].min()), int(win[c:c_end + 1, 1].max()))
            E_chunk, X_chunk, U_chunk = edge_costs(compiled, grid, p, rasters, c, c_end, n_samples, cache, rows)
        cross_out = X_chunk[c - E_start]                  # (R, K): the edge leaving this site crosses a capped feature
        up_out = U_chunk[c - E_start]
        lo_r, hi_r = (0, R) if win is None else (int(win[c, 0]), int(win[c, 1]))
        Rw = hi_r - lo_r
        rw = slice(lo_r, hi_r)
        # tower at column c: from an arrival (type by deflection) or as the first dead-end. A crossing tower pays
        # max(type by deflection, the type its crossing forces) — once (14 Oct 2026; the forced type was also on
        # the edge before, so a bend at a crossing tower, free up to 45°, cost the search a second tower premium)
        sc = tfac[c, rw].view(Rw, 1, 1) if tfac is not None else 1.0
        Vw = V[rw]
        forced = tA_f + torch.maximum(up_in[rw].unsqueeze(2), up_out[rw].unsqueeze(1))     # (Rw, K_in, K_out)
        tot = Vw.unsqueeze(2) + torch.maximum(T.unsqueeze(0), forced) * sc
        totA = Vw.unsqueeze(2) + torch.maximum(TA.unsqueeze(0), forced) * sc
        if T45 is not None:
            capped = cross_in[rw].unsqueeze(2) | cross_out[rw].unsqueeze(1)          # (Rw, K_in, K_out)
            tot = torch.where(capped, Vw.unsqueeze(2) + torch.maximum(T45.unsqueeze(0), forced) * sc, tot)
            totA = torch.where(capped, Vw.unsqueeze(2) + torch.maximum(TA45.unsqueeze(0), forced) * sc, totA)
        del forced
        tot = torch.where(angle_only[c, rw].view(Rw, 1, 1), totA, tot)
        mval, am = tot.min(dim=1)
        Mw = mval + S[c, rw].view(Rw, 1)
        am = am.to(torch.int16)
        del tot, totA
        seed = SA[c, rw].view(Rw, 1).expand(Rw, K)
        if p.dead_end_max_dev_deg is not None:
            # TSD note J: the first dead-end tower turns at most dead_end_max_dev_deg between the
            # gantry slack span and the line
            hs = torch.rad2deg(torch.atan2(ys_t[rw], torch.full_like(ys_t[rw], max(c * grid.sx, 1.0)))).view(Rw, 1)
            dev_ok = (head_t.view(1, K) - hs).abs() <= p.dead_end_max_dev_deg + 1e-6
            seed = torch.where(dev_ok, seed, torch.full_like(seed, INF))
        use_seed = seed < Mw
        Mw = torch.where(use_seed, seed, Mw)
        am = torch.where(use_seed, start_marker, am)
        if win is None:
            M = Mw
            arg[c] = am
        else:
            M = torch.full((R, K), INF, device=dev)
            M[rw] = Mw
            arg_cols[c] = (lo_r, am)
        reach_map[c] = torch.isfinite(M).any(dim=1)
        E = E_chunk[c - E_start]
        Wring[c % ring] = M + E
        Xring[c % ring] = cross_out
        Uring[c % ring] = up_out
        if progress and c % 100 == 0:
            progress(c / nc)

    best = float(end_tot.min().item())
    reach_np = reach.cpu().numpy()
    reach_idx = int(np.nonzero(reach_np)[0].max()) if reach_np.any() else 0
    if not math.isfinite(best):
        rm = reach_map.cpu().numpy()
        cols = np.nonzero(rm.any(axis=1))[0]
        reach_m = float(cols.max() * grid.sx) if len(cols) else 0.0
        return SolveOutcome(False, INF, [], [], grid, reach_m, time.time() - t0, rm)
    flat = int(end_tot.argmin().item())
    c, r = divmod(flat, R)
    k = int(end_k[c, r].item())
    sites = [(c, r)]
    heads = []
    guard = 0
    while True:
        guard += 1
        if guard > 100000:
            raise RuntimeError("backtrack did not terminate")
        heads.append(k)
        dc, dr = grid.offsets[k]
        pc, pr = c - int(dc), r - int(dr)
        sites.append((pc, pr))
        if win is None:
            k_prev = int(arg[pc, pr, k].item())
        else:
            lo_p, am_p = arg_cols[pc]
            k_prev = int(am_p[pr - lo_p, k].item())
        if k_prev == -1:
            break
        c, r, k = pc, pr, k_prev
    sites.reverse()
    heads.reverse()
    return SolveOutcome(True, best, sites, heads, grid, compiled.frame.length_m, time.time() - t0)


def refine(compiled: Compiled, p: SolverParams, out: SolveOutcome, spacing_m: float = 25.0, half_width_m: float = 300.0,
           cancel_check=None, progress=None) -> SolveOutcome:
    """Targeted finer lattice (Vivek, 14 Oct 2026): the search again at spacing_m, confined to the sites within
    half_width_m of the coarse solution's line; the lattice slack (site tolerance, crossing-span slack) scales
    with the spacing. Returns the finer outcome when it is feasible and no dearer, else the coarse one."""
    from dataclasses import replace as _replace
    from shapely.geometry import LineString as _LS
    if not out.feasible:
        return out
    line = _LS([out.grid.site_xy(c, r) for c, r in out.sites])
    p2 = _replace(p, site_spacing_m=float(spacing_m), site_tolerance_m=float(spacing_m) / 2.0,
                  crossing_span_slack_m=float(spacing_m) / 2.0, edge_cache_limit_bytes=0)
    out2 = solve(compiled, p2, cancel_check=cancel_check, progress=progress, corridor=(line, float(half_width_m)))
    if out2.feasible and out2.cost_rs <= out.cost_rs + 1e-6:
        log.info("refine %gm: ₹%.2f Cr -> ₹%.2f Cr (search prices), %d -> %d towers, %.0f s",
                 spacing_m, out.cost_rs / 1e7, out2.cost_rs / 1e7, len(out.sites), len(out2.sites), out2.seconds)
        return out2
    log.info("refine %gm: kept the coarse solution (%s)", spacing_m, "infeasible" if not out2.feasible else "dearer")
    return out


def tower_type_cost(costs: dict, kv: int) -> dict[str, float]:
    """₹ per tower by type: the all-in price of Route Cost Parameters 3 (steel, erection and a normal
    foundation). Soil extras and tower-base land are site costs."""
    return {t: float(costs["towers"][t]["price_rs"]) for t in TOWER_TYPES}
