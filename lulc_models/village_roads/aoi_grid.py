"""AOI -> z18 block plan on the global 16-tile grid.

Replaces guj_mask_z18.npz: the tile mask is rasterised from the request AOI (EPSG:4326),
dilated by buffer_m, and kept on a grid whose origin is a multiple of NT, so block tags
(xyz col,row of the top-left tile) match the statewide run (/mnt/cog-data/guj/prob_u).
Needs numpy, scipy, rasterio only.
"""
from __future__ import annotations

import math

import numpy as np

Z = 18
N = 2 ** Z
O = 20037508.342789244          # half the web-mercator extent
TM = 2 * O / N                  # tile size (m, EPSG:3857)
RES = TM / 256                  # pixel size (m, EPSG:3857)
NT = 16                         # tiles per block side
H = 2                           # halo tiles each side for inference
R_EARTH = 6378137.0


# ---- AOI -------------------------------------------------------------------------------
def aoi_polygons(aoi) -> list:
    """Accept a Polygon / MultiPolygon geometry, a Feature or a FeatureCollection.
    Returns a list of polygon coordinate arrays (rings of [lon, lat])."""
    t = aoi.get("type")
    if t == "FeatureCollection":
        out = []
        for f in aoi["features"]:
            out += aoi_polygons(f)
        return out
    if t == "Feature":
        return aoi_polygons(aoi["geometry"])
    if t == "Polygon":
        return [aoi["coordinates"]]
    if t == "MultiPolygon":
        return list(aoi["coordinates"])
    if t == "GeometryCollection":
        out = []
        for g in aoi["geometries"]:
            out += aoi_polygons(g)
        return out
    raise ValueError("AOI must be a Polygon/MultiPolygon (got %r)" % t)


def aoi_geometry(aoi) -> dict:
    polys = aoi_polygons(aoi)
    if not polys:
        raise ValueError("empty AOI")
    return {"type": "MultiPolygon", "coordinates": polys}


def lonlat_to_3857(lon, lat):
    lon = np.asarray(lon, float); lat = np.clip(np.asarray(lat, float), -85.05112878, 85.05112878)
    x = R_EARTH * np.radians(lon)
    y = R_EARTH * np.log(np.tan(math.pi / 4 + np.radians(lat) / 2))
    return x, y


def x3857_to_lonlat(x, y):
    lon = np.degrees(np.asarray(x, float) / R_EARTH)
    lat = np.degrees(2 * np.arctan(np.exp(np.asarray(y, float) / R_EARTH)) - math.pi / 2)
    return lon, lat


def geom_to_3857(geom: dict) -> dict:
    def ring(r):
        a = np.asarray(r, float)
        x, y = lonlat_to_3857(a[:, 0], a[:, 1])
        return np.stack([x, y], 1).tolist()
    return {"type": "MultiPolygon",
            "coordinates": [[ring(r) for r in poly] for poly in aoi_geometry(geom)["coordinates"]]}


def aoi_bounds(geom: dict):
    pts = np.concatenate([np.asarray(r, float)[:, :2] for poly in aoi_polygons(geom) for r in poly])
    return float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())


def lonlat_tile(lon, lat):
    x, y = lonlat_to_3857(lon, lat)
    return int(math.floor((float(x) + O) / TM)), int(math.floor((O - float(y)) / TM))


# ---- grid / plan ------------------------------------------------------------------------
class Plan:
    """Tile grid aligned to NT. Arrays are indexed [row - r0, col - c0] (xyz rows)."""

    def __init__(self, c0, r0, h, w, aoi_mask, buf_mask):
        assert c0 % NT == 0 and r0 % NT == 0 and h % NT == 0 and w % NT == 0
        self.c0, self.r0, self.h, self.w = c0, r0, h, w
        self.aoi_mask = aoi_mask        # tiles touched by the AOI
        self.buf_mask = buf_mask        # AOI dilated by buffer_m (superset)

    @property
    def transform(self):
        from rasterio.transform import Affine
        return Affine(TM, 0, -O + self.c0 * TM, 0, -TM, O - self.r0 * TM)

    def blocks(self) -> list[tuple[int, int]]:
        """Block tags (bx, by) with any buffered-mask tile, row-major north -> south."""
        hb, wb = self.h // NT, self.w // NT
        m = self.buf_mask.reshape(hb, NT, wb, NT).any(axis=(1, 3))
        ii, jj = np.nonzero(m)
        return [(self.c0 + int(j) * NT, self.r0 + int(i) * NT) for i, j in zip(ii, jj)]

    def block_slice(self, bx, by):
        i, j = by - self.r0, bx - self.c0
        return slice(i, i + NT), slice(j, j + NT)


def plan_blocks(aoi, buffer_m: float = 2400.0) -> Plan:
    """Rasterise the AOI onto z18 tiles (all_touched) and dilate by buffer_m.

    The dilation is conservative: a tile is kept if its centre is within buffer + sqrt(2)
    tiles of an AOI tile centre, which guarantees every point within buffer_m of the AOI
    lies in a kept tile. buffer_m is ground metres, converted to mercator with the AOI's
    largest |lat| (the most stretched latitude)."""
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    import scipy.ndimage as ndi

    geom = aoi_geometry(aoi)
    lon0, lat0, lon1, lat1 = aoi_bounds(geom)
    k = 1.0 / math.cos(math.radians(max(abs(lat0), abs(lat1))))
    rad_t = max(0.0, float(buffer_m)) * k / TM
    pad = int(math.ceil(rad_t)) + 2
    cmin, rmin = lonlat_tile(lon0, lat1)
    cmax, rmax = lonlat_tile(lon1, lat0)
    c0 = ((cmin - pad) // NT) * NT
    r0 = ((rmin - pad) // NT) * NT
    c1 = -(-(cmax + pad + 1) // NT) * NT
    r1 = -(-(rmax + pad + 1) // NT) * NT
    c0, r0 = max(c0, 0), max(r0, 0)
    c1, r1 = min(c1, N), min(r1, N)
    h, w = r1 - r0, c1 - c0
    tr = Affine(TM, 0, -O + c0 * TM, 0, -TM, O - r0 * TM)
    aoi_mask = rasterize([(geom_to_3857(geom), 1)], out_shape=(h, w), transform=tr,
                         all_touched=True, fill=0, dtype="uint8").astype(bool)
    if rad_t > 0 and aoi_mask.any():
        d = ndi.distance_transform_edt(~aoi_mask)
        buf_mask = d <= rad_t + math.sqrt(2)
    else:
        buf_mask = aoi_mask.copy()
    return Plan(c0, r0, h, w, aoi_mask, buf_mask)


def neighbours(bx, by):
    return [(bx + dx * NT, by + dy * NT) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if dx or dy]


def block_transform(bx, by):
    from rasterio.transform import from_origin
    return from_origin(-O + bx * TM, O - by * TM, RES, RES)


def block_lat(by):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (by + NT / 2) / N))))
