"""CPU post-processing for the solar plugin: polygonise, OSM classify, frozen-band final filter,
MBTiles coverage. No torch, no network — importable from the beeline env for tests.

Ported from /mnt/drso-data/solar_lora/scripts:
  08b (unscripted)  prob >= 153 -> gdal_polygonize -> merc_area_m2 >= 5000   -> polygonize_prob()
  09_classify.py    EPSG:7755 area >= 5000, intersects OSM solar = known      -> classify_osm()
  16_final_nir.py   NIR band filter on new detections (band FROZEN here)      -> final_classes()
"""
from __future__ import annotations

import math
import sqlite3

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import shape
from shapely.ops import unary_union

EQ_CRS = "EPSG:7755"           # equal-area for India (Web Mercator overstates ~19 % here)
R_MERC = 6378137.0 * math.pi

CLS_CONFIRMED = "confirmed_OSM"
CLS_VERIFIED = "new_verified_NIR"
CLS_REJECTED = "rejected_NIR"
CLS_EXCLUDED = "excluded_below_20m"
CLS_UNVERIFIED = "new_unverified"

CONFIDENCE = {CLS_CONFIRMED: 1.0, CLS_VERIFIED: 0.8, CLS_UNVERIFIED: 0.5}
KEEP = tuple(CONFIDENCE)


# ---- 08b: threshold + polygonise + area filter ------------------------------------------
def polygonize_prob(prob_path, thr_u8: int = 153, min_merc_area_m2: float = 5000.0,
                    window: Window | None = None, block: int = 4096, cancelled=None) -> gpd.GeoDataFrame:
    """Polygonise prob >= thr_u8 (4-connected, as gdal_polygonize) and keep merc area >= min.

    Streams in block x block tiles. Per tile, 4-connected components smaller than the area floor
    that do not touch an interior seam are dropped before vectorising (polygon area == pixel
    count, so this is exact and skips the speckle). Vectorising is done in integer pixel
    coordinates so pieces split by a seam share exact edges and union back into one polygon —
    the result equals polygonising the whole raster at once. Returns EPSG:3857 polygons with
    merc_area_m2.
    """
    from scipy import ndimage as nd
    from shapely import affinity

    whole, seam = [], []
    with rasterio.open(prob_path) as ds:
        crs, T = ds.crs, ds.transform
        px_area = abs(T.a * T.e)
        min_px = min_merc_area_m2 / px_area - 1e-6
        win = window or Window(0, 0, ds.width, ds.height)
        c0, r0 = int(win.col_off), int(win.row_off)
        W, H = int(win.width), int(win.height)
        for y0 in range(0, H, block):
            for x0 in range(0, W, block):
                if cancelled and cancelled():
                    raise SystemExit("cancelled")
                w, h = min(block, W - x0), min(block, H - y0)
                m = ds.read(1, window=Window(c0 + x0, r0 + y0, w, h)) >= thr_u8
                if not m.any():
                    continue
                lab, n = nd.label(m)                       # default structure = 4-connected
                sizes = np.bincount(lab.ravel(), minlength=n + 1)
                edge = np.zeros(n + 1, bool)
                if x0 > 0:
                    edge[lab[:, 0]] = True
                if x0 + w < W:
                    edge[lab[:, -1]] = True
                if y0 > 0:
                    edge[lab[0, :]] = True
                if y0 + h < H:
                    edge[lab[-1, :]] = True
                keep = (sizes >= min_px) | edge
                keep[0] = False
                if not keep[1:].any():
                    continue
                km = keep[lab]
                gx, gy = c0 + x0, r0 + y0                  # global pixel offset (integers)
                tr = Affine(1, 0, gx, 0, 1, gy)
                for geom, v in shapes(km.astype(np.uint8), mask=km, transform=tr, connectivity=4):
                    g = shape(geom)
                    bx0, by0, bx1, by1 = g.bounds
                    touches = ((x0 > 0 and bx0 <= gx) or (x0 + w < W and bx1 >= gx + w) or
                               (y0 > 0 and by0 <= gy) or (y0 + h < H and by1 >= gy + h))
                    (seam if touches else whole).append(g)
    if seam:
        u = unary_union(seam)
        whole.extend(getattr(u, "geoms", [u]))
    mat = [T.a, T.b, T.d, T.e, T.c, T.f]
    whole = [affinity.affine_transform(g, mat) for g in whole]
    g = gpd.GeoDataFrame(geometry=whole, crs=crs)
    g["merc_area_m2"] = g.area
    g = g[g["merc_area_m2"] >= min_merc_area_m2].reset_index(drop=True)
    return g[["merc_area_m2", "geometry"]]


# ---- 09: OSM classification --------------------------------------------------------------
def read_osm_solar(path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read the OSM solar reference. Without a layer name: the only layer, else every layer
    whose name contains 'solar'."""
    import pyogrio
    if layer:
        layers = [layer]
    else:
        names = [n for n, _ in pyogrio.list_layers(path)]
        layers = names if len(names) == 1 else [n for n in names if "solar" in n.lower()]
        if not layers:
            raise ValueError(f"no solar layer in {path} (layers: {names}); set params.osm_layer")
    parts = [gpd.read_file(path, layer=n)[["geometry"]].to_crs(4326) for n in layers]
    osm = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    return osm[osm.geometry.notnull() & ~osm.geometry.is_empty]


def classify_osm(det: gpd.GeoDataFrame, osm: gpd.GeoDataFrame | None, min_area_m2: float = 5000.0):
    """09_classify: drop null/invalid, EPSG:7755 area >= min, known = intersects any OSM solar.
    Returns the detections in EPSG:7755 with area_m2 and known."""
    d = det[det.geometry.notnull() & det.geometry.is_valid].to_crs(EQ_CRS)
    d = d.copy()
    d["area_m2"] = d.area
    d = d[d["area_m2"] >= min_area_m2].reset_index(drop=True)
    if osm is None or len(osm) == 0 or len(d) == 0:
        d["known"] = False
        return d
    o = osm.to_crs(EQ_CRS)[["geometry"]]
    j = gpd.sjoin(d[["geometry"]], o, how="left", predicate="intersects")
    j = j[~j.index.duplicated(keep="first")]
    d["known"] = j["index_right"].notna().reindex(d.index).values
    return d


# ---- 16: final classes with the frozen NIR band -----------------------------------------
def final_classes(g: pd.DataFrame, s2_available: bool, nir_band=(1159, 2351), min_obs: int = 4,
                  gate_known: bool = True) -> pd.Series:
    """Class per detection. Needs columns known, resolvable_20m, n_obs, nir_med.

    With Sentinel-2 (14_s2_seasonal + 16_final_nir, band frozen):
      not resolvable at 20 m      -> excluded_below_20m  (known ones too, when gate_known —
                                     the research run dropped them; see report)
      resolvable, n_obs < min_obs -> known: confirmed_OSM; new: new_unverified (no S2 evidence)
      resolvable, n_obs >= min    -> known: confirmed_OSM; new: nir_med in band ? verified : rejected
    Without Sentinel-2: known -> confirmed_OSM, new -> new_unverified.
    """
    known = g["known"].astype(bool).values
    out = np.where(known, CLS_CONFIRMED, CLS_UNVERIFIED).astype(object)
    if not s2_available:
        return pd.Series(out, index=g.index, name="cls")
    lo, hi = float(nir_band[0]), float(nir_band[1])
    res = g["resolvable_20m"].astype(bool).values
    nobs = g["n_obs"].fillna(0).astype(int).values
    nir = g["nir_med"].astype(float).values
    obs_ok = res & (nobs >= min_obs)
    inband = np.isfinite(nir) & (nir >= lo) & (nir <= hi)       # pandas between(): inclusive
    new_ok = ~known & obs_ok
    out[new_ok & inband] = CLS_VERIFIED
    out[new_ok & ~inband] = CLS_REJECTED
    excl = ~res if gate_known else (~res & ~known)
    out[excl] = CLS_EXCLUDED
    return pd.Series(out, index=g.index, name="cls")


# ---- coverage: AOI ∩ MBTiles data footprint ---------------------------------------------
def mbtiles_footprint_3857(path, bounds_3857=None):
    """Union of the tiles actually present (max zoom) as a shapely geometry in EPSG:3857,
    optionally restricted to a bbox. Read-only sqlite; no pixel decoding."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        z = con.execute("select max(zoom_level) from tiles").fetchone()[0]
        if z is None:
            return None
        n = 2 ** z
        size = 2 * R_MERC / n
        q = "select tile_column, tile_row from tiles where zoom_level=?"
        args = [z]
        if bounds_3857 is not None:
            x0, y0, x1, y1 = bounds_3857
            cx0 = max(0, int(math.floor((x0 + R_MERC) / size)))
            cx1 = min(n - 1, int(math.floor((x1 + R_MERC) / size)))
            ry0 = max(0, int(math.floor((R_MERC - y1) / size)))     # XYZ rows
            ry1 = min(n - 1, int(math.floor((R_MERC - y0) / size)))
            q += " and tile_column between ? and ? and tile_row between ? and ?"
            args += [cx0, cx1, n - 1 - ry1, n - 1 - ry0]                # TMS rows
        rows = np.array(con.execute(q, args).fetchall(), dtype=np.int64)
    finally:
        con.close()
    if rows.size == 0:
        return None
    cols, ys = rows[:, 0], (n - 1) - rows[:, 1]
    c0, r0 = cols.min(), ys.min()
    mask = np.zeros((ys.max() - r0 + 1, cols.max() - c0 + 1), np.uint8)
    mask[ys - r0, cols - c0] = 1
    tr = Affine(size, 0, -R_MERC + c0 * size, 0, -size, R_MERC - r0 * size)
    polys = [shape(gm) for gm, v in shapes(mask, mask=mask.astype(bool), transform=tr) if v == 1]
    return unary_union(polys)


def _polygonal(g):
    """Drop line/point slivers an AOI clip can leave in a GeometryCollection."""
    if g is None or g.is_empty or g.geom_type in ("Polygon", "MultiPolygon"):
        return g
    import shapely
    parts = [p for p in shapely.get_parts(g) if p.geom_type in ("Polygon", "MultiPolygon")]
    return unary_union(parts) if parts else None


def to_features(g: gpd.GeoDataFrame, aoi_4326) -> gpd.GeoDataFrame:
    """Kept classes -> contract layer `features` (EPSG:4326, clipped to the AOI)."""
    k = g[g["cls"].isin(KEEP)].to_crs(4326).copy()
    k["geometry"] = [_polygonal(g) for g in k.geometry.intersection(aoi_4326)]
    k = k[k.geometry.notnull() & ~k.geometry.is_empty]
    cols = ["cls", "area_m2", "n_obs", "nir_med", "ndvi_med"]
    out = gpd.GeoDataFrame({
        "class": "solar_park",
        "confidence": k["cls"].map(CONFIDENCE).astype(float).values,
        "name": pd.Series([None] * len(k), dtype=object).values,
        "src_id": k["src_id"].values,
        **{c: k[c].values for c in cols},
    }, geometry=k.geometry.values, crs=4326)
    return out
