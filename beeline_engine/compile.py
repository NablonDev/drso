"""Rule compile: reviewed layers + resolved rules + costs -> rasters on the chainage frame.

Outputs (all float32/bool arrays of shape (frame.height, frame.width)):
  tower_forbidden      no tower may stand here
  angle_forbidden      only suspension (type A) towers may stand here
  conductor_forbidden  no span may pass over here
  strip_density        ₹ per m² of RoW corridor (land comp, crops, trees, forest)
  sensitive            1 where land is forest / agriculture / orchard / social forest
  site_cost            ₹ per tower that does not depend on tower type (foundation, land, earthing)
  crossings            list[CrossingRaster]: angle raster (NaN = no feature) per crossing rule
  parallels            list[ParallelRaster]: nearest-feature angle within the parallel setback

Every raster buffer is a distance-transform threshold. The 20 m grid makes it accurate to
about ±14 m; spotting re-checks every tower against the exact vector buffers.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from rasterio import features as rfeatures
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union

from .config import ResolvedRule, forced_min_type, load_layers
from .frame import Frame

log = logging.getLogger(__name__)

RS_PER_CR = 1e7
RS_PER_L = 1e5
M2_PER_HA = 1e4


@dataclass
class CrossingRaster:
    rule_id: str
    name: str
    angle: np.ndarray            # feature direction in degrees [0,180), NaN where no feature
    window: tuple[float, float] | None   # allowed crossing angle (deg, 0-90 measure) or None = any
    max_span_m: float | None
    min_type: str
    cost_rs: float
    width_raster: np.ndarray | None = None  # 1 inside polygon features (river width measure)
    major_threshold_m: float | None = None
    major_type: str | None = None
    major_cost_rs: float = 0.0


@dataclass
class ParallelRaster:
    rule_id: str
    angle: np.ndarray            # nearest feature angle where within setback, NaN elsewhere


class VecRule:
    """Exact vector form of one rule buffer: features (frame coords) + distance, behind an STRtree.
    'X is forbidden' == X lies within `d` of any feature. No buffering or unions up front."""

    def __init__(self, rule_id: str, layer: str | None, geoms, d: float):
        from shapely import STRtree
        self.rule_id = rule_id
        self.layer = layer
        self.geoms = np.asarray(list(geoms), dtype=object)
        self.d = max(float(d), 0.0)
        self.tree = STRtree(self.geoms)

    def hits(self, geom) -> bool:
        if self.d > 0:
            return len(self.tree.query(geom, predicate="dwithin", distance=self.d)) > 0
        return len(self.tree.query(geom, predicate="intersects")) > 0

    def near_idx(self, geom, limit: int | None = None):
        idx = (self.tree.query(geom, predicate="dwithin", distance=self.d) if self.d > 0
               else self.tree.query(geom, predicate="intersects"))
        return idx if limit is None else idx[:limit]

    def buffered(self, idx) -> object:
        gs = [self.geoms[i].buffer(self.d, resolution=6) if self.d > 0 else self.geoms[i] for i in idx]
        return unary_union(gs) if gs else None


@dataclass
class Compiled:
    frame: Frame
    tower_forbidden: np.ndarray
    angle_forbidden: np.ndarray
    conductor_forbidden: np.ndarray
    strip_density: np.ndarray
    sensitive: np.ndarray
    site_cost: np.ndarray
    soil_mult: np.ndarray
    land_rate_m2: np.ndarray
    cost_class: np.ndarray       # uint8 COST_CLASS_CODES, for area statistics
    crossings: list[CrossingRaster]
    parallels: list[ParallelRaster]
    half_width_m: float
    conductor_rs_per_m: float
    # vector buffers for exact checks in spotting, in frame coords
    tower_nogo: list[VecRule] = field(default_factory=list)
    angle_nogo: list[VecRule] = field(default_factory=list)
    conductor_nogo: list[VecRule] = field(default_factory=list)
    rules_fired: dict[str, int] = field(default_factory=dict)
    costs: dict = field(default_factory=dict)
    terminal_zone: object = None     # frame coords: go-zone discs + terminal yards (+ gantry buffer)
    terminal_yards: object = None


# ----------------------------------------------------------------------------------------
def _to_frame(frame: Frame, gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    utm = gdf.geometry.to_crs(frame.epsg)
    return utm.apply(frame.geom_utm_to_frame)


class _Rasterizer:
    """Caches per-(layer, classes) feature masks and distance transforms."""

    def __init__(self, frame: Frame, layers: dict[str, gpd.GeoDataFrame]):
        self.frame = frame
        self.layers = layers
        self.shape = (frame.height, frame.width)
        self._mask: dict = {}
        self._dist: dict = {}
        self._frame_geoms: dict = {}

    def subset(self, layer: str | None, classes: list[str] | None) -> gpd.GeoDataFrame | None:
        if layer is None or layer not in self.layers:
            return None
        g = self.layers[layer]
        if g is None or len(g) == 0:
            return None
        if classes:
            g = g[g["class"].isin(classes)]
        return g if len(g) else None

    def frame_geoms(self, layer, classes):
        key = (layer, tuple(classes or ()))
        if key not in self._frame_geoms:
            g = self.subset(layer, classes)
            self._frame_geoms[key] = None if g is None else _to_frame(self.frame, g)
        return self._frame_geoms[key]

    def mask(self, layer, classes) -> np.ndarray | None:
        key = (layer, tuple(classes or ()))
        if key not in self._mask:
            g = self.subset(layer, classes)
            if g is None:
                self._mask[key] = None
            else:
                utm = g.geometry.to_crs(self.frame.epsg)
                shapes = [(geom, 1) for geom in utm if geom is not None and not geom.is_empty]
                m = rfeatures.rasterize(shapes, out_shape=self.shape, transform=self.frame.transform,
                                        fill=0, all_touched=True, dtype="uint8") if shapes else None
                self._mask[key] = None if m is None or not m.any() else m.astype(bool)
        return self._mask[key]

    def buffer_mask(self, layer, classes, d: float) -> np.ndarray | None:
        """Cells within d of the features, from the buffered vectors: unlike dist(), features
        outside the frame (an airport beyond the band) still reach into it."""
        g = self.subset(layer, classes)
        if g is None:
            return None
        utm = g.geometry.to_crs(self.frame.epsg)
        shapes = [(geom.buffer(max(d, 0.0) + RASTER_MARGIN_M), 1) for geom in utm if geom is not None and not geom.is_empty]
        if not shapes:
            return None
        m = rfeatures.rasterize(shapes, out_shape=self.shape, transform=self.frame.transform,
                                fill=0, all_touched=True, dtype="uint8")
        return m.astype(bool) if m.any() else None

    def dist(self, layer, classes) -> np.ndarray | None:
        """Distance (m) from each cell to the nearest feature cell."""
        key = (layer, tuple(classes or ()))
        if key not in self._dist:
            m = self.mask(layer, classes)
            self._dist[key] = None if m is None else (
                ndimage.distance_transform_edt(~m).astype(np.float32) * self.frame.res_m)
        return self._dist[key]

    def angle_nearest(self, layer, classes, within_m: float) -> np.ndarray | None:
        """Direction (deg in [0,180)) of the nearest line segment, for cells within `within_m`."""
        g = self.subset(layer, classes)
        if g is None:
            return None
        utm = g.geometry.to_crs(self.frame.epsg)
        shapes = []
        for geom in utm:
            for seg_start, seg_end in _segments(geom):
                ang = math.degrees(math.atan2(seg_end[1] - seg_start[1], seg_end[0] - seg_start[0]))
                ang = (ang - math.degrees(self.frame.bearing_rad)) % 180.0
                shapes.append((LineString([seg_start, seg_end]), ang + 1.0))  # +1 so 0 means empty
        if not shapes:
            return None
        ang_r = rfeatures.rasterize(shapes, out_shape=self.shape, transform=self.frame.transform,
                                    fill=0, all_touched=True, dtype="float32")
        feat = ang_r > 0
        if not feat.any():
            return None
        d, idx = ndimage.distance_transform_edt(~feat, return_indices=True)
        out = ang_r[idx[0], idx[1]] - 1.0
        out[d * self.frame.res_m > within_m] = np.nan
        return out.astype(np.float32)


def _segments(geom):
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        cs = list(geom.coords)
        for i in range(len(cs) - 1):
            yield cs[i], cs[i + 1]
    elif isinstance(geom, MultiLineString):
        for g in geom.geoms:
            yield from _segments(g)
    elif geom.geom_type in ("Polygon", "MultiPolygon"):
        yield from _segments(geom.boundary)
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _segments(g)


RASTER_MARGIN_M = 0.0  # set by compile_rules from the raster resolution


def _within(dist: np.ndarray | None, d: float) -> np.ndarray | None:
    """Cells within d of a feature, padded by half a cell diagonal so the raster never
    allows a site or span that the exact vector check would reject."""
    if dist is None:
        return None
    return dist <= max(d, 0.0) + RASTER_MARGIN_M + 1e-6


# ----------------------------------------------------------------------------------------
LULC_COST_CLASS = {
    "agriculture": "agriculture", "agriculture_fallow": "agriculture", "orchard": "orchard",
    "social_forest": "social_forest", "reserve_forest": "forest",
    "shrub": "nonagri", "barren": "nonagri", "hard_rock": "nonagri", "marsh": "nonagri",
    "salt_pan": "nonagri", "sand_dunes": "nonagri", "built_up": "nonagri", "unclassified": "nonagri",
    "water_body": "water",
}
COST_CLASS_CODES = {"nonagri": 0, "agriculture": 1, "orchard": 2, "social_forest": 3, "forest": 4, "water": 5}
SOIL_KEYS = ("normal_dry", "wet_submerged", "black_cotton", "hard_rock", "sandy_fissured")


def class_density_rs_m2(costs: dict, sensitive_mult: float = 1.0) -> dict[str, float]:
    """₹ per m² of RoW corridor by cost class (true cost, sensitive_mult applied by caller)."""
    land = costs["land"]
    crops = costs["crops"]
    forest = costs["forest"]
    row = land["row_corridor_pct"] / 100.0
    stamp = land["stamp_legal_l_ha"] * RS_PER_L / M2_PER_HA
    agri = land["agri_land_cr_ha"] * RS_PER_CR / M2_PER_HA
    nonagri = land["nonagri_land_cr_ha"] * RS_PER_CR / M2_PER_HA
    crop = crops["standing_crop_l_ha"] * RS_PER_L / M2_PER_HA
    orchard_trees = crops["orchard_trees_per_ha"] * crops["orchard_tree_rs"] / M2_PER_HA
    sf_trees = crops["social_forest_trees_per_ha"] * crops["timber_tree_rs"] / M2_PER_HA
    fr = (forest["npv_l_ha"] + forest["ca_l_ha"]) * RS_PER_L / M2_PER_HA
    rf_land = forest["rf_land_cr_ha"] * RS_PER_CR / M2_PER_HA
    return {
        "agriculture": agri * row + stamp + crop,
        "orchard": agri * row + stamp + orchard_trees,
        "social_forest": nonagri * row + stamp + sf_trees,
        "forest": rf_land * row + fr,
        "nonagri": nonagri * row + stamp,
        "water": 0.0,
    }


def compile_rules(frame: Frame, layers: dict[str, gpd.GeoDataFrame], rules: dict[str, ResolvedRule],
                  costs: dict, spec_kv: int, half_width_m: float, endpoints_go_radius_m: float | None = None,
                  design_span_m: float = 400.0, single_angle_tol_deg: float = 15.0,
                  gantry_buffer_m: float = 10.0, raster_margin_m: float = 0.0) -> Compiled:
    # Terminal yards: the substation(s) the line starts / ends in. They are carved out of the
    # substation rule; the yard (+ gantry buffer) and the go-zone disc are clear for the conductor,
    # and no tower may stand inside the yard other than the dead-end towers at A and B.
    global RASTER_MARGIN_M
    RASTER_MARGIN_M = raster_margin_m
    go_r = endpoints_go_radius_m or 0.0
    a_pt = Point(*frame.A)
    b_pt = Point(*frame.B)
    yards_frame = []
    subs = layers.get("substations")
    if subs is not None and len(subs):
        utm = subs.to_crs(frame.epsg)
        term = utm.geometry.apply(lambda g: g is not None and (g.distance(a_pt) <= go_r or g.distance(b_pt) <= go_r))
        if term.any():
            yards_frame = [frame.geom_utm_to_frame(g) for g in utm.geometry[term]]
            layers = dict(layers)
            layers["substations"] = subs[~term.values]
    zone = unary_union([Point(0, 0).buffer(go_r), Point(frame.length_m, 0).buffer(go_r)]
                       + [y.buffer(gantry_buffer_m) for y in yards_frame])
    yards = unary_union(yards_frame) if yards_frame else None
    R = _Rasterizer(frame, layers)
    shape = R.shape
    tower_forbidden = np.zeros(shape, bool)
    angle_forbidden = np.zeros(shape, bool)
    conductor_forbidden = np.zeros(shape, bool)
    fired: dict[str, int] = {}
    tower_geoms, angle_geoms, cond_geoms = [], [], []

    def add(mask_arr, target, rule_id):
        if mask_arr is not None and mask_arr.any():
            target |= mask_arr
            fired[rule_id] = fired.get(rule_id, 0) + 1

    def vec(lst, rule_id, layer, classes, d):
        fg = R.frame_geoms(layer, classes)
        if fg is not None and len(fg):
            lst.append(VecRule(rule_id, layer, [g for g in fg if g is not None and not g.is_empty], d))

    hw = half_width_m

    def cond_buffer(r: ResolvedRule) -> float:
        b = r.params.get("buffer") or r.params.get("conductor_buffer")
        if b == "Corridor half-width":
            return hw
        if b == "Fixed":
            return float(r.params.get("buffer_m", r.params.get("conductor_buffer_m", 0)))
        if b == "None":
            return 0.0
        return float(r.params.get("conductor_buffer_m", 0))

    crossings: list[CrossingRaster] = []
    parallels: list[ParallelRaster] = []
    cross_costs = costs["crossings"]
    tol = single_angle_tol_deg

    for r in rules.values():
        if not r.enabled:
            continue
        p = r.params
        m = r.mechanism
        if m == "conductor_clearance":
            extra = p.get("increment_per_33kv_m", 0) * max(0, (spec_kv - 33) / 33.0) if spec_kv > 400 else 0
            d = float(p["horizontal_m"]) + extra + hw
            # conductor clearance is measured from the outer phase; the corridor half-width
            # bounds the outer phase position, so the line axis keeps hw + horizontal away.
            add(_within(R.dist(r.layer, r.classes), d), conductor_forbidden, r.id)
            vec(cond_geoms, r.id, r.layer, r.classes, d)
        elif m == "airport_prohibition":
            applies = p.get("applies_to", "EHT and HT")
            if applies == "EHT only" and spec_kv < 220:
                continue
            d = float(p["distance_m"])
            mk = R.buffer_mask(r.layer, r.classes, d)
            add(mk, tower_forbidden, r.id)
            add(mk, conductor_forbidden, r.id)
            vec(tower_geoms, r.id, r.layer, r.classes, d)
            vec(cond_geoms, r.id, r.layer, r.classes, d)
        elif m == "flag_radius":
            pass  # events only
        elif m == "tower_setback":
            d = float(p["setback_m"])
            add(_within(R.dist(r.layer, r.classes), d), tower_forbidden, r.id)
            vec(tower_geoms, r.id, r.layer, r.classes, d)
        elif m == "wetland":
            if p.get("tower_placement") == "Prohibited":
                d = float(p["band_m"])
                add(_within(R.dist(r.layer, r.classes), d), tower_forbidden, r.id)
                vec(tower_geoms, r.id, r.layer, r.classes, d)
            if p.get("conductor_may_span") == "No":
                add(R.mask(r.layer, r.classes), conductor_forbidden, r.id)
                vec(cond_geoms, r.id, r.layer, r.classes, 0)
        elif m == "nogo":
            if p.get("treatment") == "Allowed":
                continue
            td = float(p.get("tower_setback_m", 0))
            dist = R.dist(r.layer, r.classes)
            if r.also_layers and r.id not in ("buildings", "dams", "creek"):
                # same buffers on further layers, read whole (e.g. plotted land under the
                # built-up cluster rule — Vivek, 6 Oct 2026)
                cd = float(p.get("conductor_buffer_m", 0))
                for lyr, cls in ((r.layer, r.classes), *((extra, None) for extra in r.also_layers)):
                    d_l = R.dist(lyr, cls)
                    if d_l is None:
                        continue
                    add(_within(d_l, td), tower_forbidden, r.id)
                    vec(tower_geoms, r.id, lyr, cls, td)
                    add(_within(d_l, cd), conductor_forbidden, r.id)
                    vec(cond_geoms, r.id, lyr, cls, cd)
                continue
            if dist is None:
                continue
            if r.id == "buildings":
                cb = cond_buffer(r)
                add(_within(dist, cb), conductor_forbidden, r.id)
                vec(cond_geoms, r.id, r.layer, r.classes, cb)
                target, glist = (angle_forbidden, angle_geoms) if p.get("applies_to") == "Angular towers" \
                    else (tower_forbidden, tower_geoms)
                add(_within(dist, td), target, r.id)
                vec(glist, r.id, r.layer, r.classes, td)
                continue
            add(_within(dist, td), tower_forbidden, r.id)
            vec(tower_geoms, r.id, r.layer, r.classes, td)
            if r.id == "dams":
                if p.get("crossing") == "Prohibited":
                    add(_within(dist, td), conductor_forbidden, r.id)
                    vec(cond_geoms, r.id, r.layer, r.classes, td)
            elif r.id == "creek":
                add(_within(dist, td), conductor_forbidden, r.id)
                vec(cond_geoms, r.id, r.layer, r.classes, td)
            else:
                cd = float(p.get("conductor_buffer_m", 0))
                add(_within(dist, cd), conductor_forbidden, r.id)
                vec(cond_geoms, r.id, r.layer, r.classes, cd)
        elif m == "landclass":
            if p.get("treatment") != "No-Go":
                continue
            cb = cond_buffer(r)
            for lyr, cls in ((r.layer, r.classes), ("lulc", r.also_lulc_classes),
                             *((extra, None) for extra in (r.also_layers or []))):
                if lyr == "lulc" and cls is None and r.layer != "lulc":
                    continue
                dist = R.dist(lyr, cls)
                add(_within(dist, cb), conductor_forbidden, r.id)
                add(_within(dist, 0), tower_forbidden, r.id)
                vec(cond_geoms, r.id, lyr, cls, cb)
                vec(tower_geoms, r.id, lyr, cls, 0)
        elif m == "crossing":
            td = float(p.get("tower_setback_m", 0))
            classes = r.classes
            if r.id == "power_line":
                # towers keep clear of existing towers and of the existing conductors;
                # the new conductor keeps the null-point radius from existing towers
                # "Tower to existing tower" applies to towers. Existing conductors: a new tower
                # keeps the new line's corridor half-width clear of them (interpretation, flagged).
                add(_within(R.dist(r.layer, ["power_tower"]), td), tower_forbidden, r.id)
                vec(tower_geoms, r.id, r.layer, ["power_tower"], td)
                add(_within(R.dist(r.layer, ["power_line"]), hw), tower_forbidden, r.id)
                vec(tower_geoms, r.id, r.layer, ["power_line"], hw)
                nr = float(p.get("null_point_radius_m", 0))
                add(_within(R.dist(r.layer, ["power_tower"]), nr), conductor_forbidden, r.id)
                vec(cond_geoms, r.id, r.layer, ["power_tower"], nr)
                line_classes = ["power_line"]
            else:
                if td > 0:
                    add(_within(R.dist(r.layer, classes), td), tower_forbidden, r.id)
                    vec(tower_geoms, r.id, r.layer, classes, td)
                line_classes = classes
            ang = R.angle_nearest(r.layer, line_classes, within_m=frame.res_m * 1.5)
            if ang is None:
                continue
            fired[r.id] = fired.get(r.id, 0) + 1
            window = _angle_window(r, tol)
            max_span = p.get("max_span_m")
            if p.get("pct_design_span") is not None:
                pct_span = float(p["pct_design_span"]) / 100.0 * design_span_m
                max_span = pct_span if max_span is None else min(float(max_span), pct_span)
            cost = float(cross_costs.get(r.crossing_cost, 0)) * RS_PER_L if r.crossing_cost else 0.0
            cr = CrossingRaster(rule_id=r.id, name=r.name, angle=ang, window=window,
                                max_span_m=None if max_span is None else float(max_span),
                                min_type=forced_min_type(p.get("tower_type")), cost_rs=cost)
            if r.id == "river":
                cr.width_raster = R.mask(r.layer, classes)
                cr.major_threshold_m = float(p.get("major_threshold_m", 600))
                cr.major_type = forced_min_type(p.get("major_tower_type"))
                cr.major_cost_rs = float(cross_costs.get("major_river", 0)) * RS_PER_L
                cr.cost_rs = 0.0
                cr.min_type = forced_min_type(p.get("tower_type"))
            crossings.append(cr)
            ps = p.get("parallel_setback_m")
            if ps:
                pa = R.angle_nearest(r.layer, line_classes, within_m=float(ps))
                if pa is not None:
                    parallels.append(ParallelRaster(rule_id=r.id, angle=pa))

    # Implicit engineering constraint: no tower stands in open water.
    for lyr, cls in (("pond", None), ("lake", None), ("river", None), ("canal", None), ("reservoir", None),
                     ("nalla", None), ("lulc", ["water_body"])):
        mk = R.mask(lyr, cls)
        if mk is not None:
            g = R.subset(lyr, cls)
            if g is not None and g.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
                poly = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
                mp = rfeatures.rasterize([(x, 1) for x in poly.geometry.to_crs(frame.epsg)],
                                         out_shape=shape, transform=frame.transform, fill=0,
                                         all_touched=False, dtype="uint8").astype(bool)
                tower_forbidden |= mp
                tower_geoms.append(VecRule("open_water", lyr, list(_to_frame(frame, poly)), 0.0))

    # ---- cost rasters -------------------------------------------------------------------
    dens = class_density_rs_m2(costs)
    strip_density = np.full(shape, dens["nonagri"], np.float32)
    sensitive = np.zeros(shape, np.float32)
    land_rate = np.full(shape, costs["land"]["nonagri_land_cr_ha"] * RS_PER_CR / M2_PER_HA, np.float32)
    cost_class = np.zeros(shape, np.uint8)
    lulc = layers.get("lulc")
    if lulc is not None and len(lulc):
        utm = lulc.to_crs(frame.epsg)
        codes = {c: i + 1 for i, c in enumerate(LULC_COST_CLASS)}
        shapes = [(g, codes.get(c, codes["unclassified"])) for g, c in zip(utm.geometry, utm["class"])
                  if g is not None and not g.is_empty]
        cls_r = rfeatures.rasterize(shapes, out_shape=shape, transform=frame.transform, fill=0,
                                    dtype="uint8") if shapes else np.zeros(shape, np.uint8)
        for c, code in codes.items():
            sel = cls_r == code
            if not sel.any():
                continue
            cc = LULC_COST_CLASS[c]
            strip_density[sel] = dens[cc]
            cost_class[sel] = COST_CLASS_CODES[cc]
            if cc in ("agriculture", "orchard", "social_forest", "forest"):
                sensitive[sel] = 1.0
            if cc in ("agriculture", "orchard"):
                land_rate[sel] = costs["land"]["agri_land_cr_ha"] * RS_PER_CR / M2_PER_HA
    fm = R.mask("forest", None)
    if fm is not None:
        strip_density[fm] = dens["forest"]
        cost_class[fm] = COST_CLASS_CODES["forest"]
        sensitive[fm] = 1.0
        land_rate[fm] = costs["forest"]["rf_land_cr_ha"] * RS_PER_CR / M2_PER_HA

    # Foundation multiplier: the `soil` layer's per-feature class, plus every layer whose registry
    # entry names a foundation_class (the soil-type layers). Where two disagree the dearer wins.
    soil_mult = np.ones(shape, np.float32)
    fnd = costs["foundation"]
    soil_layers = [("soil", None)] + [(l["key"], l["foundation_class"]) for l in load_layers()["layers"]
                                      if l.get("foundation_class") in SOIL_KEYS]
    for key, fixed in soil_layers:
        soil = layers.get(key)
        if soil is None or not len(soil):
            continue
        utm = soil.to_crs(frame.epsg)
        shapes = [(g, float(fnd.get(fixed or c, 1.0))) for g, c in zip(utm.geometry, utm["class"])
                  if g is not None and not g.is_empty and (fixed or c) in SOIL_KEYS]
        if shapes:
            sm = rfeatures.rasterize(shapes, out_shape=shape, transform=frame.transform, fill=0,
                                     dtype="float32")
            soil_mult = np.maximum(soil_mult, np.where(sm > 0, sm, 1.0)).astype(np.float32)

    fnd_base = float(costs["foundation"]["base_cr_tower"]) * RS_PER_CR
    base_area = float(costs["base_area_m2"])
    earthing = float(costs["tower"]["earthing_rs"])
    site_cost = (fnd_base * soil_mult + base_area * land_rate * costs["land"]["tower_base_pct"] / 100.0
                 + earthing).astype(np.float32)

    cnd = costs["conductor"]
    conductor_rs_per_m = (cnd["conductor_cr_km"] + cnd["earthwire_cr_km"] + cnd["stringing_cr_km"]
                          + cnd["spacer_cr_km"]) * RS_PER_CR / 1000.0

    # ---- terminal zone (gantry go-zone + terminal yards) --------------------------------
    if not zone.is_empty:
        utm_zone = _frame_to_utm(frame, zone)
        # cell centres only: the cleared raster area stays inside the exact zone
        zm = rfeatures.rasterize([(utm_zone, 1)], out_shape=shape, transform=frame.transform, fill=0,
                                 all_touched=False, dtype="uint8").astype(bool)
        tower_forbidden[zm] = False
        angle_forbidden[zm] = False
        conductor_forbidden[zm] = False
        # lines that end in the terminal yard are not crossings of the new line there
        for cr in crossings:
            cr.angle[zm] = np.nan
        for pr in parallels:
            pr.angle[zm] = np.nan
        if yards is not None:
            ym = rfeatures.rasterize([(_frame_to_utm(frame, yards), 1)], out_shape=shape,
                                     transform=frame.transform, fill=0, dtype="uint8").astype(bool)
            tower_forbidden |= ym
            tower_geoms.append(VecRule("terminal_yard", "substations", list(getattr(yards, "geoms", [yards])), 0.0))

    return Compiled(frame=frame, tower_forbidden=tower_forbidden, angle_forbidden=angle_forbidden,
                    conductor_forbidden=conductor_forbidden, strip_density=strip_density,
                    sensitive=sensitive, site_cost=site_cost, soil_mult=soil_mult,
                    land_rate_m2=land_rate, cost_class=cost_class, crossings=crossings, parallels=parallels,
                    half_width_m=hw, conductor_rs_per_m=conductor_rs_per_m,
                    tower_nogo=tower_geoms, angle_nogo=angle_geoms,
                    conductor_nogo=cond_geoms, rules_fired=fired, costs=costs,
                    terminal_zone=zone, terminal_yards=yards)


def _frame_to_utm(frame: Frame, g):
    from shapely import affinity
    g = affinity.rotate(g, frame.bearing_rad, origin=(0, 0), use_radians=True)
    return affinity.translate(g, frame.A[0], frame.A[1])


def _angle_window(r: ResolvedRule, tol: float) -> tuple[float, float] | None:
    """Allowed crossing angle as (lo, hi) on the 0-90° acute-angle scale; None = any."""
    p = r.params
    if "min_angle_deg" in p:
        return (float(p["min_angle_deg"]), 90.0)
    a = p.get("angle_deg")
    if a is None or a == "Any":
        return None
    if isinstance(a, (list, tuple)):
        lo, hi = float(a[0]), float(a[1])
        # a range like 70-110 means the acute angle must be >= min(lo, 180-hi)
        return (min(lo, 180.0 - hi), 90.0)
    if isinstance(a, str):
        if a == "90°":
            return (90.0 - tol, 90.0)
        if "–" in a:
            lo, hi = a.replace("°", "").split("–")
            return (min(float(lo), 180.0 - float(hi)), 90.0)
        return None
    return (max(0.0, float(a) - tol), 90.0)
