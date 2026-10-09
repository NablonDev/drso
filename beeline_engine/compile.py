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

from .config import ResolvedRule, forced_min_type, load_layers, max_type
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
    extension_m: float = 0.0               # body extension each crossing tower carries (flyovers)
    long_span: bool = False                # spans longer than the design span may cross it (rivers; relaxed water)


@dataclass
class ParallelRaster:
    rule_id: str
    angle: np.ndarray            # nearest feature angle where within setback, NaN elsewhere
    crossing_exempt: bool = False   # a span that crosses the feature is not a parallel run (canal, 9 Oct 2026)


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
        # approach section (8 Oct 2026): a rule compiled for the approach applies only inside it,
        # the normal one only outside it (frame-coordinate polygons)
        self.inside = None
        self.outside = None

    def _clip(self, geom):
        if self.outside is not None:
            geom = geom.difference(self.outside)
        if self.inside is not None and not geom.is_empty:
            geom = geom.intersection(self.inside)
        return geom

    def hits(self, geom) -> bool:
        if self.inside is not None or self.outside is not None:
            geom = self._clip(geom)
            if geom.is_empty:
                return False
        if self.d > 0:
            return len(self.tree.query(geom, predicate="dwithin", distance=self.d)) > 0
        return len(self.tree.query(geom, predicate="intersects")) > 0

    def near_idx(self, geom, limit: int | None = None):
        if self.inside is not None or self.outside is not None:
            geom = self._clip(geom)
            if geom.is_empty:
                return np.array([], dtype=int)
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
    relaxed: bool = False                    # compiled for the Relaxed route (last-resort rules opened)
    approach_zone: object = None             # frame polygon of the approach sections (None = none)
    rasterizer: object = None                # the _Rasterizer, reused by further compiles of the same layers
    go_overlap: list = field(default_factory=list)   # VecRules still checked inside the go zone (overlap only)
    go_conductor: list = field(default_factory=list)  # conductor VecRules kept in the go zone (>= 220 kV crossings)
    precedent_items: list = field(default_factory=list)  # polygons opened by "learn from existing lines"
    near_existing: np.ndarray | None = None  # bool raster: within reach of an existing line (search discount)
    # last-resort relaxations taken in this compile: (VecRule of what was opened, approval months, label)
    relaxed_items: list = field(default_factory=list)
    # terrain on the rule raster grid (set by attach_terrain; None = no DEM)
    ground_z: np.ndarray | None = None       # m, NaN where the DEM has no value
    hilly: np.ndarray | None = None          # bool: slope >= hilly_slope_deg over ~100 m


def attach_terrain(comp: "Compiled", dem, hilly_slope_deg: float) -> None:
    """Ground height and hilly mask on the 20 m rule grid, from the DEM sampler (6 Oct 2026)."""
    f = comp.frame
    h, w = comp.tower_forbidden.shape
    cols = np.arange(w)
    z = np.full((h, w), np.nan, np.float32)
    for r0 in range(0, h, 256):                     # row blocks: bounded memory for wide bands
        rows = np.arange(r0, min(h, r0 + 256))
        fx = np.broadcast_to(f.x0 + (cols + 0.5) * f.res_m, (len(rows), w))
        fy = np.broadcast_to((f.y_top - (rows + 0.5) * f.res_m)[:, None], (len(rows), w))
        lon, lat = f.frame_to_lonlat(fx.ravel(), fy.ravel())
        z[rows] = np.asarray(dem.sample(lon, lat), np.float32).reshape(len(rows), w)
    zf = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0.0)
    zs = ndimage.uniform_filter(zf, size=5, mode="nearest")            # ~100 m, as the 90 m span check
    gy, gx = np.gradient(zs, f.res_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    comp.ground_z = z
    comp.hilly = (slope >= hilly_slope_deg) & np.isfinite(z)


# ----------------------------------------------------------------------------------------
def _to_frame(frame: Frame, gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """UTM → frame coordinates for a whole layer at once (one vectorised coordinate transform;
    per-geometry shapely affinity calls took ~1 min on 250k features)."""
    utm = gdf.geometry.to_crs(frame.epsg)
    return gpd.GeoSeries(frame.geoms_utm_to_frame(np.asarray(utm.values)), index=utm.index)


class _Rasterizer:
    """Caches per-(layer, classes) feature masks and distance transforms."""

    def __init__(self, frame: Frame, layers: dict[str, gpd.GeoDataFrame], cache=None):
        self.frame = frame
        self.layers = layers
        self.shape = (frame.height, frame.width)
        self._mask: dict = {}
        self._dist: dict = {}
        self._angle: dict = {}
        self._frame_geoms: dict = {}
        self._fp: dict = {}
        self.base_layer: dict = {}        # filtered key -> source layer key
        self.eased: dict = {}             # precedent key -> polygons opened by existing towers
        self.cache = cache                # RasterCache (on disk, across solves) or None

    def precedent(self, layer: str, towers) -> str:
        """Key of a copy of `layer` without the polygons an existing tower (UTM points) stands in;
        the removed polygons are kept in self.eased[key]."""
        key = f"{layer}|precedent"
        if key not in self.layers:
            g = self.layers.get(layer)
            self.eased[key] = None
            if g is None or not len(g) or towers is None or not len(towers):
                self.layers[key] = g
            else:
                from shapely import STRtree
                utm = np.asarray(g.geometry.to_crs(self.frame.epsg).values)
                poly = np.array([x is not None and x.geom_type in ("Polygon", "MultiPolygon") for x in utm])
                pairs = STRtree(towers).query(utm, predicate="contains")
                hit = np.zeros(len(g), bool)
                hit[np.unique(pairs[0])] = True
                hit &= poly
                self.layers[key] = g[~hit]
                self.eased[key] = g[hit] if hit.any() else None
            self.base_layer[key] = self.base_layer.get(layer, layer)
        return key

    def fingerprint(self, layer, classes) -> str:
        from .rastercache import geometry_fingerprint
        key = (layer, tuple(classes or ()))
        if key not in self._fp:
            self._fp[key] = geometry_fingerprint(self.subset(layer, classes), classes)
        return self._fp[key]

    def by_voltage(self, layer: str, min_kv: float, at_least: bool) -> str:
        """Key of a copy of `layer` holding only features whose attrs.voltage_kv is >= min_kv
        (at_least) or < min_kv. Features without a voltage count as >= (the stricter side)."""
        key = f"{layer}|kv{'>=' if at_least else '<'}{min_kv:g}"
        if key not in self.layers:
            g = self.layers.get(layer)
            if g is None or not len(g) or "attrs" not in g:
                self.layers[key] = g if at_least else None
            else:
                kv = g["attrs"].map(lambda a: (a or {}).get("voltage_kv")).astype(float)
                hi = kv.isna() | (kv >= min_kv)
                self.layers[key] = g[hi] if at_least else g[~hi]
            self.base_layer[key] = self.base_layer.get(layer, layer)
        return key

    def class_subset(self, layer: str, classes: list[str]) -> str:
        """Key of a copy of `layer` holding only the given classes (e.g. flyovers among bridges)."""
        key = f"{layer}|cls={','.join(classes)}"
        if key not in self.layers:
            g = self.layers.get(layer)
            self.layers[key] = None if g is None or not len(g) else g[g["class"].isin(classes)]
            self.base_layer[key] = self.base_layer.get(layer, layer)
        return key

    def filtered(self, layer: str | None, min_area_m2: float) -> str | None:
        """Key of a copy of `layer` without polygons smaller than min_area_m2 (UTM m²); registered
        on first use. Rules with a `min_area_m2` param read this copy (e.g. buildings: a field hut
        is spanned, not avoided)."""
        if layer is None or layer not in self.layers or not min_area_m2:
            return layer
        key = f"{layer}|min{min_area_m2:g}"
        if key not in self.layers:
            g = self.layers[layer]
            if g is None or not len(g):
                self.layers[key] = g
            else:
                poly = g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                area = g.geometry.to_crs(self.frame.epsg).area
                self.layers[key] = g[~poly | (area >= min_area_m2)]
            self.base_layer[key] = layer
        return key

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
        if key not in self._mask and self.cache is not None and self.subset(layer, classes) is not None:
            m, found = self.cache.get_mask(self.fingerprint(layer, classes), self.shape)
            if found:
                self._mask[key] = m
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
                if self.cache is not None:
                    self.cache.put_mask(self.fingerprint(layer, classes), self._mask[key])
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
        from .rastercache import dist_from_sq, sq_pixels
        if key not in self._dist and self.cache is not None and self.subset(layer, classes) is not None:
            sq, found = self.cache.get_dist(self.fingerprint(layer, classes))
            if found:
                self._dist[key] = None if sq is None else dist_from_sq(sq, self.frame.res_m)
        if key not in self._dist:
            m = self.mask(layer, classes)
            if m is None:
                self._dist[key] = None
                sq = None
            else:
                e = ndimage.distance_transform_edt(~m)
                self._dist[key] = e.astype(np.float32) * self.frame.res_m
                sq = sq_pixels(e) if self.cache is not None else None
            if self.cache is not None and self.subset(layer, classes) is not None:
                self.cache.put_dist(self.fingerprint(layer, classes), sq)
        return self._dist[key]

    def angle_nearest(self, layer, classes, within_m: float) -> np.ndarray | None:
        """Direction (deg in [0,180)) of the nearest line segment, for cells within `within_m`.
        The unthresholded field is cached (memory and disk); the threshold is applied per call."""
        g = self.subset(layer, classes)
        if g is None:
            return None
        key = (layer, tuple(classes or ()))
        if key not in self._angle:
            hit = None
            if self.cache is not None:
                ang, sq, found = self.cache.get_angle(self.fingerprint(layer, classes))
                if found:
                    hit = (ang, sq)
            if hit is None:
                hit = self._angle_field(g)
                if self.cache is not None:
                    self.cache.put_angle(self.fingerprint(layer, classes), *hit)
            self._angle[key] = hit
        ang, sq = self._angle[key]
        if ang is None:
            return None
        out = ang.copy()
        out[np.sqrt(sq.astype(np.float64)) * self.frame.res_m > within_m] = np.nan   # float64, as before caching
        return out

    def _angle_field(self, g):
        """(nearest-segment direction deg, squared pixel distance) over the whole raster, unthresholded."""
        utm = g.geometry.to_crs(self.frame.epsg)
        shapes = []
        for geom in utm:
            for seg_start, seg_end in _segments(geom):
                ang = math.degrees(math.atan2(seg_end[1] - seg_start[1], seg_end[0] - seg_start[0]))
                ang = (ang - math.degrees(self.frame.bearing_rad)) % 180.0
                shapes.append((LineString([seg_start, seg_end]), ang + 1.0))  # +1 so 0 means empty
        if not shapes:
            return None, None
        ang_r = rfeatures.rasterize(shapes, out_shape=self.shape, transform=self.frame.transform,
                                    fill=0, all_touched=True, dtype="float32")
        feat = ang_r > 0
        if not feat.any():
            return None, None
        from .rastercache import sq_pixels
        d, idx = ndimage.distance_transform_edt(~feat, return_indices=True)
        out = (ang_r[idx[0], idx[1]] - 1.0).astype(np.float32)
        return out, sq_pixels(d)


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
GO_ZONE_CROSSING_KV = 220.0   # go zone: crossings of existing lines from this voltage up keep their rules
# "Learn from existing lines" (9 Oct 2026): layers whose polygons an existing tower may open, and the
# rule mechanisms that are eased (crossing rules never are)
PRECEDENT_LAYERS = {"wet_land", "pond", "lake", "reservoir", "other_water", "crz", "salt_pans",
                    "reserve_forest", "social_forest", "forest_in_pa", "national_park", "wildlife_sanctuary"}
PRECEDENT_PA_LAYERS = {"reserve_forest", "social_forest", "forest_in_pa", "national_park", "wildlife_sanctuary"}
PRECEDENT_MECHANISMS = {"nogo", "landclass", "wetland", "tower_setback"}


def _within(dist: np.ndarray | None, d: float) -> np.ndarray | None:
    """Cells within d of a feature, padded by half a cell diagonal so the raster never
    allows a site or span that the exact vector check would reject."""
    if dist is None:
        return None
    return dist <= max(d, 0.0) + RASTER_MARGIN_M + 1e-6


# ----------------------------------------------------------------------------------------
def _tower_crossing_bans(lines, towers, epsg: int, clear_m: float) -> list:
    """Parts of the existing conductors the new line may not cross (UTM): the wire within clear_m
    of an existing tower. The rest of the wire may be crossed anywhere (Vivek, 8 Oct 2026; replaces
    the mid-span null-point window)."""
    if lines is None or towers is None or not len(lines) or not len(towers) or clear_m <= 0:
        return []
    lu = unary_union([g for g in lines.geometry.to_crs(epsg) if g is not None and not g.is_empty])
    out = []
    for t in towers.geometry.to_crs(epsg):
        if t is None or t.is_empty:
            continue
        piece = lu.intersection(t.buffer(clear_m, quad_segs=16))
        for g in getattr(piece, "geoms", [piece]):
            if g.geom_type == "LineString" and g.length > 0:
                out.append(g)
    return out


# rule parameters whose value is the corridor half-width (CHW) or CHW + a fixed width (Rule_Table)
CHW_TIED = {("lake", "tower_setback_m"), ("lake", "conductor_buffer_m"), ("crz", "tower_setback_m"),
            ("crz", "conductor_buffer_m"), ("defence", "tower_setback_m"), ("defence", "conductor_buffer_m"),
            ("power_line", "parallel_setback_m"), ("power_line_lt", "parallel_setback_m"),
            ("railway", "parallel_setback_m"), ("state_highway", "parallel_setback_m")}


def merge_approach(norm: "Compiled", app: "Compiled", zone) -> "Compiled":
    """Rules inside the approach zone (frame polygon) from the approach compile, elsewhere from the
    normal one: rasters cell by cell, vector checks by region."""
    f = norm.frame
    zm = rfeatures.rasterize([(_frame_to_utm(f, zone), 1)], out_shape=norm.tower_forbidden.shape,
                             transform=f.transform, fill=0, all_touched=False, dtype="uint8").astype(bool)
    for attr in ("tower_forbidden", "angle_forbidden", "conductor_forbidden"):
        setattr(norm, attr, np.where(zm, getattr(app, attr), getattr(norm, attr)))
    app_par = {p.rule_id: p for p in app.parallels}
    for p in norm.parallels:
        if p.rule_id in app_par:
            p.angle = np.where(zm, app_par[p.rule_id].angle, p.angle)
    for lst_name in ("tower_nogo", "angle_nogo", "conductor_nogo"):
        merged = []
        for vr in getattr(norm, lst_name):
            vr.outside = zone
            merged.append(vr)
        for vr in getattr(app, lst_name):
            vr.inside = zone
            merged.append(vr)
        setattr(norm, lst_name, merged)
    norm.approach_zone = zone
    return norm


LULC_COST_CLASS = {
    "agriculture": "agriculture", "agriculture_fallow": "agriculture", "orchard": "orchard",
    "social_forest": "social_forest", "reserve_forest": "forest",
    "shrub": "nonagri", "barren": "nonagri", "hard_rock": "nonagri", "marsh": "nonagri",
    "salt_pan": "nonagri", "sand_dunes": "nonagri", "built_up": "nonagri", "unclassified": "nonagri",
    "water_body": "water",
}
COST_CLASS_CODES = {"nonagri": 0, "agriculture": 1, "orchard": 2, "social_forest": 3, "forest": 4, "water": 5}
SOIL_KEYS = ("normal_dry", "wet_submerged", "black_cotton", "hard_rock")


def class_density_rs_m2(costs: dict, sensitive_mult: float = 1.0) -> dict[str, float]:
    """₹ per m² of RoW corridor by cost class (true cost, sensitive_mult applied by caller).
    Route Cost Parameters 3 (7 Oct 2026): one rural land value; corridor compensation is a share of
    it; forest adds NPV + compensatory afforestation + plans per m² diverted and the reserve-forest
    land share. No crop, tree or stamp items."""
    land = costs["land"]
    forest = costs["forest"]
    row = land["row_corridor_pct"] / 100.0
    value = land["land_cr_ha"] * RS_PER_CR / M2_PER_HA
    fr = (forest["npv_l_ha"] + forest["ca_l_ha"] + forest["plans_l_ha"]) * RS_PER_L / M2_PER_HA
    rf_land = forest["rf_land_cr_ha"] * RS_PER_CR / M2_PER_HA
    corridor = value * row
    return {
        "agriculture": corridor,
        "orchard": corridor,
        "social_forest": corridor,
        "forest": rf_land * row + fr,
        "nonagri": corridor,
        "water": 0.0,
    }


def compile_rules(frame: Frame, layers: dict[str, gpd.GeoDataFrame], rules: dict[str, ResolvedRule],
                  costs: dict, spec_kv: int, half_width_m: float, endpoints_go_radius_m: float | None = None,
                  design_span_m: float = 400.0, single_angle_tol_deg: float = 15.0,
                  gantry_buffer_m: float = 10.0, raster_margin_m: float = 0.0,
                  angle_tower_rules: tuple[str, ...] = (), relaxed: bool = False,
                  chw_ref_m: float | None = None, rasterizer=None, raster_cache=None,
                  terminal_yard_radius_m: float | None = None, precedent_towers=None) -> Compiled:
    # Terminal yards: the substation(s) the line starts / ends in. They are carved out of the
    # substation rule; the yard (+ gantry buffer) and the go-zone disc are clear for the conductor,
    # and no tower may stand inside the yard other than the dead-end towers at A and B.
    global RASTER_MARGIN_M
    RASTER_MARGIN_M = raster_margin_m
    go_r = endpoints_go_radius_m or 0.0
    yard_r = go_r if terminal_yard_radius_m is None else min(float(terminal_yard_radius_m), go_r)
    all_layers = layers
    a_pt = Point(*frame.A)
    b_pt = Point(*frame.B)
    yards_frame = []
    subs = layers.get("substations")
    if subs is not None and len(subs):
        utm = subs.to_crs(frame.epsg)
        term = utm.geometry.apply(lambda g: g is not None and (g.distance(a_pt) <= yard_r or g.distance(b_pt) <= yard_r))
        if term.any():
            yards_frame = [frame.geom_utm_to_frame(g) for g in utm.geometry[term]]
            layers = dict(layers)
            layers["substations"] = subs[~term.values]
    zone = unary_union([Point(0, 0).buffer(go_r), Point(frame.length_m, 0).buffer(go_r)]
                       + [y.buffer(gantry_buffer_m) for y in yards_frame])
    yards = unary_union(yards_frame) if yards_frame else None
    R = rasterizer if rasterizer is not None else _Rasterizer(frame, layers, cache=raster_cache)
    shape = R.shape
    tower_forbidden = np.zeros(shape, bool)
    angle_forbidden = np.zeros(shape, bool)
    conductor_forbidden = np.zeros(shape, bool)
    fired: dict[str, int] = {}
    precedent_items: dict = {}                                 # layer -> eased polygons (learn from existing lines)
    go_hv_near = np.zeros((frame.height, frame.width), bool)   # cells next to a >= 220 kV line
    go_hv_bans: list = []                                      # UTM pieces of those lines near their towers
    go_conductor: list = []
    tower_geoms, angle_geoms, cond_geoms = [], [], []

    def add(mask_arr, target, rule_id):
        if mask_arr is not None and mask_arr.any():
            target |= mask_arr
            fired[rule_id] = fired.get(rule_id, 0) + 1

    def vec(lst, rule_id, layer, classes, d):
        fg = R.frame_geoms(layer, classes)
        if fg is not None and len(fg):
            lst.append(VecRule(rule_id, R.base_layer.get(layer, layer),
                               [g for g in fg if g is not None and not g.is_empty], d))

    hw = half_width_m
    relaxed_items: list = []

    def relaxed_note(r: ResolvedRule, layer, classes, d: float, months: float, label: str | None = None):
        fg = R.frame_geoms(layer, classes)
        if fg is not None and len(fg):
            relaxed_items.append((VecRule(r.id, R.base_layer.get(layer, layer),
                                          [g for g in fg if g is not None and not g.is_empty], d),
                                  float(months or 0), label or r.name))

    def last_resort(r: ResolvedRule) -> str:
        """'nogo' (normal routes), 'open' (relaxed: towers may stand inside), 'span' (relaxed: the
        conductor may cross, towers keep the buffer)."""
        if r.params.get("treatment") != "Last resort":
            return "nogo"
        if not relaxed:
            return "nogo"
        return "open" if r.params.get("last_resort_towers_inside") == "Yes" else "span"

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
    tol = single_angle_tol_deg

    for r in rules.values():
        if not r.enabled:
            continue
        p = r.params
        if chw_ref_m is not None and chw_ref_m > half_width_m:
            # approach-section compile: buffers set at the corridor half-width shrink with it
            delta = chw_ref_m - half_width_m
            keys = [k for (rid, k) in CHW_TIED if rid == r.id and p.get(k) is not None]
            if keys:
                p = {**p, **{k: max(0.0, float(p[k]) - delta) for k in keys}}
        if relaxed and r.id in ("buildings", "building_clearance"):
            # last resort (Rule_Table 18): single structures up to 10 m² may be crossed (12 months)
            bl = rules.get("buildings")
            lr = float((bl.params if bl else {}).get("last_resort_max_area_m2", 0) or 0)
            if lr > float(p.get("min_area_m2") or 0):
                p = {**p, "min_area_m2": lr}
                if r.id == "buildings":
                    relaxed_note(r, r.layer, r.classes, cond_buffer(r), 12, f"Structures up to {lr:g} m² crossed")
        if p.get("min_area_m2"):
            from dataclasses import replace as _replace
            r = _replace(r, layer=R.filtered(r.layer, float(p["min_area_m2"])))
        m = r.mechanism
        if precedent_towers is not None and m in PRECEDENT_MECHANISMS:
            # learn from existing lines: polygons existing towers stand in are opened (9 Oct 2026)
            from dataclasses import replace as _replace

            def _eased(lyr):
                if lyr is None or R.base_layer.get(lyr, lyr) not in PRECEDENT_LAYERS:
                    return lyr
                k = R.precedent(lyr, precedent_towers)
                ez = R.eased.get(k)
                base = R.base_layer.get(lyr, lyr)
                if ez is not None and base not in precedent_items:
                    names = sorted({str(n) for n in ez.get("name", []) if isinstance(n, str) and n.strip()})
                    precedent_items[base] = {
                        "layer": base, "rule": r.id, "n": int(len(ez)), "names": names[:5],
                        "vr": VecRule(r.id, base, [x for x in _to_frame(frame, ez) if x is not None], 0.0),
                        "protected": base in PRECEDENT_PA_LAYERS}
                return k
            r = _replace(r, layer=_eased(r.layer), also_layers=[_eased(x) for x in (r.also_layers or [])] or r.also_layers)
        if m == "conductor_clearance":
            # Horizontal clearance is the per-voltage figure (no per-33 kV increment on top of it)
            # and is measured from the outer phase, which sits outer_phase_m from the line axis
            # (765 kV D/C ~15 m — Vivek, 7 Oct 2026). Earlier: + increment + corridor half-width
            # (49 m at 765 kV), which closed every farm lane.
            d = float(p["horizontal_m"]) + float(p.get("outer_phase_m", hw))
            add(_within(R.dist(r.layer, r.classes), d), conductor_forbidden, r.id)
            vec(cond_geoms, r.id, r.layer, r.classes, d)
        elif m == "airport_prohibition":
            applies = p.get("applies_to", "EHT and HT")
            if applies == "EHT only" and spec_kv < 220:
                continue
            d = float(p["distance_m"])
            fd = float(p.get("funnel_m") or 0)
            if relaxed and p.get("last_resort_distance_m"):
                # last resort (Rule_Table 14-15): 1.5 km / 6 km, 12 months' approval
                nd, nf = float(p["last_resort_distance_m"]), float(p.get("last_resort_funnel_m") or fd)
                zone_full = _airport_zone(R, frame, r.layer, r.classes, d, fd)
                if zone_full is not None:
                    relaxed_items.append((VecRule(r.id, r.layer, [zone_full], 0.0), 12.0,
                                          f"Airport buffer reduced to {nd / 1000:g} km / {nf / 1000:g} km"))
                d, fd = nd, nf
            z = _airport_zone(R, frame, r.layer, r.classes, d, fd)
            if z is None:
                continue
            mk = rfeatures.rasterize([(_frame_to_utm(frame, z), 1)], out_shape=shape,
                                     transform=frame.transform, fill=0, all_touched=True, dtype="uint8").astype(bool)
            add(mk, tower_forbidden, r.id)
            add(mk, conductor_forbidden, r.id)
            tower_geoms.append(VecRule(r.id, r.layer, [z], 0.0))
            cond_geoms.append(VecRule(r.id, r.layer, [z], 0.0))
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
            lr = last_resort(r)
            if lr != "nogo":
                relaxed_note(r, r.layer, r.classes, float(p.get("conductor_buffer_m", p.get("tower_setback_m", 0)) or 0),
                             p.get("approval_months", 0))
                if lr == "open":
                    continue
                td = float(p.get("tower_setback_m", 0))         # "span": towers keep the buffer
                add(_within(R.dist(r.layer, r.classes), td), tower_forbidden, r.id)
                vec(tower_geoms, r.id, r.layer, r.classes, td)
                # the water may be spanned like a river (the TSD span tables still apply)
                ang = R.angle_nearest(r.layer, r.classes, within_m=frame.res_m * 1.5)
                if ang is not None:
                    crossings.append(CrossingRaster(rule_id=r.id, name=r.name, angle=ang, window=None,
                                                    max_span_m=None, min_type="A", cost_rs=0.0, long_span=True))
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
            elif p.get("conductor_may_cross") == "Yes":
                pass                    # tower setback only; spans cross freely (nalla — Vivek, 7 Oct 2026)
            else:
                cd = float(p.get("conductor_buffer_m", 0))
                add(_within(dist, cd), conductor_forbidden, r.id)
                vec(cond_geoms, r.id, r.layer, r.classes, cd)
        elif m == "landclass":
            if p.get("treatment") not in ("No-Go", "Last resort"):
                continue
            cb = cond_buffer(r)
            lr = last_resort(r)
            if lr != "nogo":
                for lyr, cls in ((r.layer, r.classes), *((extra, None) for extra in (r.also_layers or []))):
                    relaxed_note(r, lyr, cls, cb, p.get("approval_months", 0))
                    if lr == "span":
                        add(_within(R.dist(lyr, cls), cb), tower_forbidden, r.id)
                        vec(tower_geoms, r.id, lyr, cls, cb)
                continue
            tb = cb if p.get("treatment") == "Last resort" else 0.0      # Rule_Table: the buffer holds towers too
            for lyr, cls in ((r.layer, r.classes), ("lulc", r.also_lulc_classes),
                             *((extra, None) for extra in (r.also_layers or []))):
                if lyr == "lulc" and cls is None and r.layer != "lulc":
                    continue
                dist = R.dist(lyr, cls)
                add(_within(dist, cb), conductor_forbidden, r.id)
                add(_within(dist, tb), tower_forbidden, r.id)
                vec(cond_geoms, r.id, lyr, cls, cb)
                vec(tower_geoms, r.id, lyr, cls, tb)
        elif m == "crossing":
            td = float(p.get("tower_setback_m", 0))
            classes = r.classes
            if r.id in ("power_line", "power_line_lt"):
                # towers keep clear of existing towers and of the existing conductors;
                # "Tower to existing tower" applies to towers. Existing conductors: a new tower
                # keeps the new line's corridor half-width clear of them (interpretation, flagged).
                add(_within(R.dist(r.layer, ["power_tower"]), td), tower_forbidden, r.id)
                vec(tower_geoms, r.id, r.layer, ["power_tower"], td)
                add(_within(R.dist(r.layer, ["power_line"]), hw), tower_forbidden, r.id)
                vec(tower_geoms, r.id, r.layer, ["power_line"], hw)
                # the new conductor may cross the existing wire anywhere except within
                # `crossing_tower_clear_m` of an existing tower (Vivek, 8 Oct 2026)
                cl = float(p.get("crossing_tower_clear_m", 0))
                bans = _tower_crossing_bans(R.subset(r.layer, ["power_line"]), R.subset(r.layer, ["power_tower"]),
                                            frame.epsg, cl)
                # go zone (9 Oct 2026): crossings of lines from GO_ZONE_CROSSING_KV up stay governed there
                hv = R.by_voltage(r.layer, GO_ZONE_CROSSING_KV, True)
                d_hv = R.dist(hv, ["power_line"])
                if d_hv is not None:
                    go_hv_near |= d_hv <= frame.res_m * 1.5
                    hv_bans = _tower_crossing_bans(R.subset(hv, ["power_line"]), R.subset(hv, ["power_tower"]),
                                                   frame.epsg, cl)
                    if hv_bans:
                        go_hv_bans.extend(hv_bans)
                        go_conductor.append(VecRule(r.id, r.layer, [frame.geom_utm_to_frame(g) for g in hv_bans], 0.0))
                if bans:
                    # raster: one cell shorter at each end (a 20 m raster cannot hold the ends exactly);
                    # the exact pieces go to the spotting's vector check
                    from shapely.ops import substring as _sub
                    loose = [_sub(g, frame.res_m, g.length - frame.res_m) for g in bans if g.length > 2 * frame.res_m]
                    loose = [g for g in loose if g is not None and not g.is_empty]
                    if loose:
                        mk = rfeatures.rasterize([(g, 1) for g in loose], out_shape=shape,
                                                 transform=frame.transform, fill=0, all_touched=False,
                                                 dtype="uint8").astype(bool)
                        add(mk, conductor_forbidden, r.id)
                    cond_geoms.append(VecRule(r.id, r.layer, [frame.geom_utm_to_frame(g) for g in bans], 0.0))
                line_classes = ["power_line"]
            else:
                if td > 0 or r.id == "bridges":         # bridges: no tower on the structure
                    add(_within(R.dist(r.layer, classes), td), tower_forbidden, r.id)
                    vec(tower_geoms, r.id, r.layer, classes, td)
                line_classes = classes
            window = _angle_window(r, tol)
            # Power lines: the crossing-angle window applies only from `angle_min_kv` upwards; lower
            # voltages are crossed at any angle (Vivek, 7 Oct 2026). Two rasters, same rule id.
            base_type = forced_min_type(p.get("tower_type"))
            parts = [(r.layer, window, base_type, 0.0)]
            if r.id in ("power_line", "power_line_lt"):
                # crossing-angle window from angle_min_kv (Rule_Table 9: lines ≥ 66 kV); crossing
                # towers by the crossed line's voltage (TSD B): D at ≥ 400 kV, the rule's type below
                akv = float(p["angle_min_kv"]) if p.get("angle_min_kv") is not None and window is not None else None
                hi_type = forced_min_type(p.get("tower_type_400kv")) if p.get("tower_type_400kv") else None
                parts = []
                above = r.layer
                if akv is not None:
                    parts.append((R.by_voltage(r.layer, akv, False), None, base_type, 0.0))
                    above = R.by_voltage(r.layer, akv, True)
                if hi_type is not None:
                    parts.append((R.by_voltage(above, 400.0, False), window if akv is not None else None, base_type, 0.0))
                    parts.append((R.by_voltage(above, 400.0, True), window if akv is not None else None,
                                  max_type(base_type, hi_type), 0.0))
                else:
                    parts.append((above, window if akv is not None else None, base_type, 0.0))
            elif r.id == "bridges" and p.get("flyover_extension_m"):
                ext = float(p["flyover_extension_m"])
                parts = [(r.layer if not classes else R.class_subset(r.layer, [c for c in classes if c != "flyover"]),
                          window, base_type, 0.0),
                         (R.class_subset(r.layer, ["flyover"]), window, base_type, ext)]
            max_span = p.get("max_span_m")
            if p.get("pct_design_span") is not None:
                pct_span = float(p["pct_design_span"]) / 100.0 * design_span_m
                max_span = pct_span if max_span is None else min(float(max_span), pct_span)
            cost = 0.0          # no crossing charges (Route Cost Parameters 3); the heavier towers are priced by type
            built = []
            for lyr, win, mt, ext in parts:
                ang = R.angle_nearest(lyr, None if (r.id == "bridges") else line_classes, within_m=frame.res_m * 1.5)
                if ang is None:
                    continue
                built.append(CrossingRaster(rule_id=r.id, name=r.name, angle=ang, window=win,
                                            max_span_m=None if max_span is None else float(max_span),
                                            min_type=mt, cost_rs=cost, extension_m=ext))
            if not built:
                continue
            fired[r.id] = fired.get(r.id, 0) + 1
            cr = built[0]
            for extra in built[1:]:
                if r.id in angle_tower_rules:
                    extra.min_type = max_type(extra.min_type, "B1")
                crossings.append(extra)
            if r.id == "river":
                cr.long_span = True
                cr.width_raster = R.mask(r.layer, classes)
                cr.major_threshold_m = float(p.get("major_threshold_m", 600))
                cr.major_type = forced_min_type(p.get("major_tower_type"))
                cr.major_cost_rs = 0.0
                cr.cost_rs = 0.0
                cr.min_type = forced_min_type(p.get("tower_type"))
            if r.id in angle_tower_rules:        # angle tower each side of the crossing (engine.yaml)
                cr.min_type = max_type(cr.min_type, "B1")
                if cr.major_type is not None:
                    cr.major_type = max_type(cr.major_type, "B1")
            crossings.append(cr)
            ps = p.get("parallel_setback_m")
            if ps:
                pa = R.angle_nearest(r.layer, line_classes, within_m=float(ps))
                if pa is not None:
                    parallels.append(ParallelRaster(rule_id=r.id, angle=pa,
                                                    crossing_exempt=p.get("crossing_exempt_from_parallel") == "Yes"))

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
    land_rate = np.full(shape, costs["land"]["land_cr_ha"] * RS_PER_CR / M2_PER_HA, np.float32)
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
    # Delivered forest layers override the LULC classes (forest package, 6 Oct 2026): social forest
    # first, reserve forest (incl. reserve forest inside a PA) over it.
    sm = R.mask("social_forest", None)
    if sm is not None:
        strip_density[sm] = dens["social_forest"]
        cost_class[sm] = COST_CLASS_CODES["social_forest"]
        sensitive[sm] = 1.0
    fm = None
    for key in ("reserve_forest", "forest_in_pa"):
        mk = R.mask(key, None)
        if mk is not None:
            fm = mk if fm is None else (fm | mk)
    if fm is not None:
        strip_density[fm] = dens["forest"]
        cost_class[fm] = COST_CLASS_CODES["forest"]
        sensitive[fm] = 1.0
        land_rate[fm] = costs["forest"]["rf_land_cr_ha"] * RS_PER_CR / M2_PER_HA
        # NPV multiplier on forest diverted inside a protected area (applies only where the
        # protected-areas rule is set to Allowed; No-Go keeps the route out anyway)
        npv = costs["forest"]["npv_l_ha"] * RS_PER_L / M2_PER_HA
        for key, mult in (("wildlife_sanctuary", "pa_wls_mult"), ("national_park", "pa_np_mult")):
            pm = R.mask(key, None)
            if pm is not None:
                sel = fm & pm
                strip_density[sel] += np.float32(npv * (float(costs["forest"][mult]) - 1.0))

    # Foundation multiplier: every layer whose registry entry names a foundation_class (the
    # soil-type layers). Where two disagree the dearer wins. (Generic `soil` layer removed, 6 Oct 2026.)
    soil_mult = np.ones(shape, np.float32)
    fnd = costs["foundation"]
    soil_layers = [(l["key"], l["foundation_class"]) for l in load_layers()["layers"]
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

    # Type-independent part of a tower's cost at each cell (the search adds the all-in price by type):
    # soil extra over a normal foundation and tower-base compensation, both on the suspension
    # tower's figures; the spotting costs every tower with its own type's foundation and area.
    ta = costs["towers"]["A"]
    site_cost = (ta["fdn_rs"] * (soil_mult - 1.0)
                 + ta["area_m2"] * land_rate * costs["land"]["tower_base_pct"] / 100.0).astype(np.float32)

    conductor_rs_per_m = float(costs["line_rs_per_m"])

    # ---- soft: near an existing line (search-only discount, 9 Oct 2026) -------------------
    near_existing = None
    pr = rules.get("existing_line_proximity")
    if pr is not None and pr.enabled and float(pr.params.get("discount_pct", 0) or 0) > 0:
        reach = float(pr.params.get("distance_m", 1000))
        mkv = float(pr.params.get("min_kv", 220))
        for lyr in ("power", "power_lt"):
            d = R.dist(R.by_voltage(lyr, mkv, True), ["power_line", "power_tower"])
            if d is not None:
                near_existing = (d <= reach) if near_existing is None else (near_existing | (d <= reach))

    # ---- terminal zone (gantry go-zone + terminal yards) --------------------------------
    go_overlap: list = []
    if not zone.is_empty:
        # inside the go zone but off the terminal yard: crossings of >= GO_ZONE_CROSSING_KV lines and
        # the no-crossing-near-an-existing-tower ban stay (Vivek, 9 Oct 2026)
        yb = yards.buffer(gantry_buffer_m) if yards is not None else None
        ybm = rfeatures.rasterize([(_frame_to_utm(frame, yb), 1)], out_shape=shape, transform=frame.transform,
                                  fill=0, dtype="uint8").astype(bool) if yb is not None else np.zeros(shape, bool)
        go_keep_zone = zone.difference(yb) if yb is not None else zone
        for vr in go_conductor:
            vr.inside = go_keep_zone
        hv_ban_mask = np.zeros(shape, bool)
        if go_hv_bans:
            from shapely.ops import substring as _sub
            loose = [_sub(g, frame.res_m, g.length - frame.res_m) for g in go_hv_bans if g.length > 2 * frame.res_m]
            loose = [g for g in loose if g is not None and not g.is_empty]
            if loose:
                hv_ban_mask = rfeatures.rasterize([(g, 1) for g in loose], out_shape=shape, transform=frame.transform,
                                                  fill=0, all_touched=False, dtype="uint8").astype(bool)
        utm_zone = _frame_to_utm(frame, zone)
        # cell centres only: the cleared raster area stays inside the exact zone
        zm = rfeatures.rasterize([(utm_zone, 1)], out_shape=shape, transform=frame.transform, fill=0,
                                 all_touched=False, dtype="uint8").astype(bool)
        tower_forbidden[zm] = False
        angle_forbidden[zm] = False
        conductor_forbidden[zm] = False
        keep = zm & ~ybm
        conductor_forbidden |= hv_ban_mask & keep
        # lines that end in the terminal yard are not crossings of the new line there; elsewhere in the
        # go zone only crossings of lines below GO_ZONE_CROSSING_KV are waived
        for cr in crossings:
            if cr.rule_id in ("power_line", "power_line_lt"):
                cr.angle[zm & ~(keep & go_hv_near)] = np.nan
            else:
                cr.angle[zm] = np.nan
        for pr in parallels:
            pr.angle[zm] = np.nan
        if yards is not None:
            ym = rfeatures.rasterize([(_frame_to_utm(frame, yards), 1)], out_shape=shape,
                                     transform=frame.transform, fill=0, dtype="uint8").astype(bool)
            tower_forbidden |= ym
            tower_geoms.append(VecRule("terminal_yard", "substations", list(getattr(yards, "geoms", [yards])), 0.0))
        # go zone (Vivek, 9 Oct 2026): the only check left is that a tower base does not overlap a
        # substation or an existing tower. Base = the largest type's square (centre to corner); an
        # existing tower point is taken as a base of the same size.
        half_diag = math.sqrt(max(float(t["area_m2"]) for t in costs["towers"].values())) / math.sqrt(2.0)
        for rid, key, classes, d in (("go_zone_substation", "substations", None, half_diag),
                                     ("go_zone_tower", "power", ["power_tower"], 2 * half_diag),
                                     ("go_zone_tower", "power_lt", ["power_tower"], 2 * half_diag),
                                     ("go_zone_tower", "telecom", None, 2 * half_diag),
                                     ("go_zone_tower", "wind_turbines", None, 2 * half_diag)):
            g = all_layers.get(key)
            if g is None or not len(g):
                continue
            if classes:
                g = g[g["class"].isin(classes)]
            fg = [x for x in _to_frame(frame, g) if x is not None and not x.is_empty and x.distance(zone) <= d]
            if not fg:
                continue
            vr = VecRule(rid, key, fg, d)
            vr.inside = zone
            go_overlap.append(vr)
            utm_g = [_frame_to_utm(frame, x).buffer(d + RASTER_MARGIN_M) for x in fg]
            om = rfeatures.rasterize([(x, 1) for x in utm_g], out_shape=shape, transform=frame.transform,
                                     fill=0, all_touched=True, dtype="uint8").astype(bool)
            tower_forbidden |= om & zm
            fired[rid] = fired.get(rid, 0) + int((om & zm).sum())

    return Compiled(frame=frame, tower_forbidden=tower_forbidden, angle_forbidden=angle_forbidden,
                    conductor_forbidden=conductor_forbidden, strip_density=strip_density,
                    sensitive=sensitive, site_cost=site_cost, soil_mult=soil_mult,
                    land_rate_m2=land_rate, cost_class=cost_class, crossings=crossings, parallels=parallels,
                    half_width_m=hw, conductor_rs_per_m=conductor_rs_per_m,
                    tower_nogo=tower_geoms, angle_nogo=angle_geoms,
                    conductor_nogo=cond_geoms, rules_fired=fired, costs=costs,
                    terminal_zone=zone, terminal_yards=yards, relaxed=relaxed, relaxed_items=relaxed_items,
                    rasterizer=R, go_overlap=go_overlap, go_conductor=go_conductor,
                    precedent_items=list(precedent_items.values()), near_existing=near_existing)


def _airport_zone(R: "_Rasterizer", frame: Frame, layer: str, classes, d: float, funnel_m: float):
    """No-go zone around airports, frame coords (Rule_Table 14-15): within d of the airport boundary,
    and within funnel_m of it inside the approach funnel of each runway end. The funnel follows the
    ICAO Annex 14 approach surface (code 4): inner edge 300 m wide, 60 m beyond the runway end,
    diverging 15 % each side."""
    from shapely import affinity
    from shapely.geometry import Polygon
    ap = R.subset(layer, classes)
    if ap is None or not len(ap):
        return None
    au = ap.to_crs(frame.epsg)
    zone = unary_union([g.buffer(d) for g in au.geometry if g is not None and not g.is_empty])
    if funnel_m > d:
        rw = R.subset("approach", ["runway"])
        if rw is not None and len(rw):
            reach = unary_union([g.buffer(funnel_m) for g in au.geometry if g is not None and not g.is_empty])
            funnels = []
            for g in rw.to_crs(frame.epsg).geometry:
                if g is None or g.is_empty or g.distance(reach) > 0:
                    continue
                rect = g.minimum_rotated_rectangle
                xs, ys = rect.exterior.coords.xy
                c = [(xs[k], ys[k]) for k in range(4)]
                e = [((c[k][0] + c[(k + 1) % 4][0]) / 2, (c[k][1] + c[(k + 1) % 4][1]) / 2) for k in range(4)]
                l01 = math.dist(c[0], c[1]); l12 = math.dist(c[1], c[2])
                ends = (e[1], e[3]) if l01 > l12 else (e[0], e[2])     # midpoints of the short sides
                for p0, p1 in (ends, ends[::-1]):
                    ux, uy = (p0[0] - p1[0]), (p0[1] - p1[1])
                    n = math.hypot(ux, uy) or 1.0
                    ux, uy = ux / n, uy / n
                    vx, vy = -uy, ux
                    L = funnel_m
                    s0x, s0y = p0[0] + 60 * ux, p0[1] + 60 * uy
                    w0, w1 = 150.0, 150.0 + 0.15 * L
                    funnels.append(Polygon([(s0x + w0 * vx, s0y + w0 * vy), (s0x - w0 * vx, s0y - w0 * vy),
                                            (s0x + L * ux - w1 * vx, s0y + L * uy - w1 * vy),
                                            (s0x + L * ux + w1 * vx, s0y + L * uy + w1 * vy)]))
            if funnels:
                zone = unary_union([zone, unary_union(funnels).intersection(reach)])
    return frame.geom_utm_to_frame(zone)


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
