"""CPU side of the wind-turbine plugin: Exp-G on a mask, cross-scene NMS, OSM confirmation,
scene footprints and acquisition dates.

expg_scene() is the detection half of /mnt/drso-data/devesh/turbine_expg/run_expg_new.py
(blobs -> azimuth -> turbine_detect.detect -> pixel centre -> map point), with the hard-coded
EPSG:32642 output replaced by the scene's own (projected) CRS. The OSM diagnostic half of that
script is research scoring and is not part of the plugin; OSM is used here only to confirm.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from shapely.geometry import box, shape

import turbine_blobs as XB
import turbine_detect as TD
import turbine_harness as H


# ---------------------------------------------------------------------------- Exp-G per scene
def expg_scene(mask_path: str, scene_id: str, thresh: int = 160, min_px: int = 15,
               cache_path: str | None = None, on_strip=None) -> tuple[gpd.GeoDataFrame, dict]:
    """Detections (Point, scene CRS) with columns area, n_frag; plus scene stats."""
    d = XB.extract(mask_path, thresh=thresh, min_px=min_px, cache_path=cache_path, tile=scene_id,
                   on_strip=on_strip)
    B = np.array(d["blobs"], dtype=float).reshape(-1, 6)
    gsd = d["gsd"]
    az, naz = H.azimuth(B)
    C = TD.detect(B, gsd, az)
    tr = rasterio.Affine(*d["transform"])
    pts = [shapely.Point(*rasterio.transform.xy(tr, c["cy"], c["cx"])) for c in C]
    det = gpd.GeoDataFrame({"area": [round(c.get("area", 0.0)) for c in C],
                            "n_frag": [c.get("n", c.get("n_frag", 1)) for c in C]},
                           geometry=pts, crs=d["crs"])
    return det, {"blobs": len(B), "azimuth": round(az, 1), "azimuth_blobs": naz, "detections": len(det)}


# ---------------------------------------------------------------------------- geometry helpers
def utm_epsg(lon: float, lat: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def aoi_utm(aoi_geom) -> int:
    c = aoi_geom.representative_point()
    return utm_epsg(c.x, c.y)


def _looks_lonlat(g) -> bool:
    x0, y0, x1, y1 = g.bounds
    return -180.0 <= x0 <= 180.0 and -180.0 <= x1 <= 180.0 and -90.0 <= y0 <= 90.0 and -90.0 <= y1 <= 90.0


def scene_footprint_4326(item: dict, raster_crs=None, raster_bounds=None):
    """Footprint of a catalog item in EPSG:4326.

    Uses item['footprint'] (GeoJSON; EPSG:4326 per the contract; coordinates outside lon/lat range
    are taken to be in the raster's CRS). Falls back to the BAND.shp delivered with the scene (its
    .prj is not readable by PROJ, so the raster CRS is used), then to the raster bounds."""
    g, crs = None, None
    fp = item.get("footprint")
    if fp:
        try:
            g = shape(fp.get("geometry", fp) if isinstance(fp, dict) and fp.get("type") == "Feature" else fp)
            crs = "EPSG:4326" if _looks_lonlat(g) else raster_crs
        except Exception:   # noqa: BLE001
            g = None
    if g is None and item.get("path"):
        shp = os.path.join(os.path.dirname(item["path"]), "BAND.shp")
        if os.path.exists(shp) and raster_crs is not None:
            try:
                import fiona
                with fiona.open(shp) as f:
                    geoms = [shape(r["geometry"]) for r in f if r["geometry"]]
                if geoms:
                    g = shapely.unary_union(geoms); crs = raster_crs
                    if _looks_lonlat(g):
                        crs = "EPSG:4326"
            except Exception:   # noqa: BLE001
                g = None
    if g is None and raster_bounds is not None:
        g, crs = box(*raster_bounds), raster_crs
    if g is None:
        return None
    if not g.is_valid:
        g = g.buffer(0)
    if crs and str(crs) != "EPSG:4326":
        g = gpd.GeoSeries([g], crs=crs).to_crs(4326).iloc[0]
    return g


# ---------------------------------------------------------------------------- truncation
def blocks_past_eof(path: str) -> int | None:
    """Number of TIFF tiles/strips whose bytes lie beyond the end of the file (an interrupted
    copy: 2660409141's raw BAND.tif opens fine but 4989 of its 35512 tiles are missing). Reads the
    IFD only. None when tifffile is unavailable or the file is not a plain TIFF."""
    try:
        import tifffile
        with tifffile.TiffFile(path) as tf:
            pg = tf.pages[0]
            o = np.asarray(pg.dataoffsets, dtype=np.int64); c = np.asarray(pg.databytecounts, dtype=np.int64)
        return int(((o + c) > os.path.getsize(path)).sum())
    except Exception:   # noqa: BLE001
        return None


# ---------------------------------------------------------------------------- acquisition date
_MON = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep",
                                    "oct", "nov", "dec"], 1)}


def _parse_date(s: str) -> str | None:
    s = s.strip()
    m = re.match(r"(\d{1,2})-([A-Za-z]{3})-(\d{4})", s)
    if m and m.group(2).lower() in _MON:
        return f"{int(m.group(3)):04d}-{_MON[m.group(2).lower()]:02d}-{int(m.group(1)):02d}"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return m.group(0)
    for fmt in ("%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s[:10], fmt).date().isoformat()
        except ValueError:
            pass
    return None


def acquisition_date(item: dict) -> str | None:
    """item['date'] if the catalog carries one, else DateOfPass / SceneCentreTime from
    BAND_META.txt or BAND_META.xml next to BAND.tif."""
    if item.get("date"):
        return _parse_date(str(item["date"])) or str(item["date"])
    d = os.path.dirname(item.get("path") or "")
    txt, xml = os.path.join(d, "BAND_META.txt"), os.path.join(d, "BAND_META.xml")
    try:
        if os.path.exists(txt):
            kv = dict(l.split("=", 1) for l in open(txt, errors="replace").read().splitlines() if "=" in l)
            for k in ("DateOfPass", "SceneCentreTime", "SceneStartTime"):
                if kv.get(k) and _parse_date(kv[k]):
                    return _parse_date(kv[k])
        if os.path.exists(xml):
            s = open(xml, errors="replace").read()
            for k in ("DateOfPass", "SceneCentreTime", "SceneStartTime"):
                m = re.search(rf"<{k}>([^<]+)</{k}>", s)
                if m and _parse_date(m.group(1)):
                    return _parse_date(m.group(1))
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------- cross-scene NMS
def cross_scene_nms(det: gpd.GeoDataFrame, nms_m: float) -> gpd.GeoDataFrame:
    """Greedy NMS across scenes in a metric CRS: a detection is dropped when a larger one from a
    DIFFERENT scene lies within nms_m (identical-footprint duplicate scenes exist). Within a scene
    Exp-G has already applied its own 95 m spacing NMS. Adds n_views = number of scenes that saw
    the kept turbine (itself included)."""
    if det.empty:
        out = det.copy(); out["n_views"] = []
        return out
    assert det.crs is not None and det.crs.is_projected, "NMS needs a metric CRS"
    det = det.reset_index(drop=True)
    order = det.sort_values(["area", "scene_id", "src_id"], ascending=[False, True, True]).index.to_numpy()
    geoms = det.geometry.values
    scene = det["scene_id"].astype(str).to_numpy()
    tree = shapely.STRtree(geoms)
    suppressed = np.zeros(len(det), bool)
    kept, views = [], []
    for i in order:
        if suppressed[i]:
            continue
        seen = {scene[i]}
        for j in tree.query(geoms[i], predicate="dwithin", distance=nms_m):
            # a larger neighbour from another scene would already have suppressed i, so every
            # unsuppressed cross-scene neighbour here is smaller (or equal, later in order)
            if j != i and scene[j] != scene[i] and not suppressed[j]:
                suppressed[j] = True; seen.add(scene[j])
        kept.append(i); views.append(len(seen))
    out = det.loc[kept].copy()
    out["n_views"] = views
    return out


# ---------------------------------------------------------------------------- OSM confirmation
def load_osm_wind(path: str | None) -> gpd.GeoDataFrame | None:
    if not path or not os.path.exists(path):
        return None
    o = gpd.read_file(path)
    if o.crs is None:
        o = o.set_crs(4326)
    o = o[o.geometry.notna() & ~o.geometry.is_empty]
    if "generator:source" in o.columns and o["generator:source"].notna().any():
        o = o[o["generator:source"].astype(str).str.lower().eq("wind") | o["generator:source"].isna()]
    o = o.copy()
    o["geometry"] = [g if g.geom_type == "Point" else g.representative_point() for g in o.geometry]
    return o


def osm_confirm(det: gpd.GeoDataFrame, osm: gpd.GeoDataFrame | None, match_m: float) -> tuple[list, list]:
    """Greedy one-to-one match (closest pairs first) within match_m in det's metric CRS.
    Returns (confirmed bools, matched osm_id or None) aligned with det rows."""
    n = len(det)
    conf, oid = [False] * n, [None] * n
    if osm is None or osm.empty or n == 0:
        return conf, oid
    o = osm.to_crs(det.crs)
    tree = shapely.STRtree(o.geometry.values)
    pairs = []
    for i, g in enumerate(det.geometry.values):
        for j in tree.query(g, predicate="dwithin", distance=match_m):
            pairs.append((g.distance(o.geometry.values[j]), i, int(j)))
    pairs.sort()
    used_d, used_o = set(), set()
    ids = o["osm_id"].to_numpy() if "osm_id" in o.columns else None
    for _, i, j in pairs:
        if i in used_d or j in used_o:
            continue
        used_d.add(i); used_o.add(j)
        conf[i] = True
        oid[i] = None if ids is None else (str(ids[j]) if ids[j] is not None else None)
    return conf, oid


def to_json(o) -> str:
    return json.dumps(o, indent=1, default=str)
