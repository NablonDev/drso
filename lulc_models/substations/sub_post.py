"""Post-processing of the Stage B probability raster into gated substation polygons.

Provenance (source scripts and sha256 in lulc_models/PROVENANCE.md):
  polygonise      sA04_score.polygonise L44-81 (p >= thresh, first-pass size filter on mercator
                  pixel area, dissolve/explode, EPSG:7755 area recheck). Parameterised only.
  OSM match       sA06_layers.py L41-46 (detection intersects an OSM substation geometry).
  minwidth        sA09_cascade_real.py L46-53, attribute only, EPSG:7755 metres.
  NISAR coh       sA08_cascade.coh_of L28-41 (median coh_min3 inside the polygon shifted +15 m E,
                  +5 m N for radar layover; values outside (0, 1] ignored).
  S1 VV           sA10_s1.py L51-96, with the Planetary Computer STAC search replaced by the local
                  scenes in request.imagery.sentinel1_rtc_vv: one scene per (relative orbit, month),
                  newest first, at most `s1_scenes`; per scene the mean LINEAR gamma0 of the valid
                  (finite, > 0) pixels inside the polygon; dB per scene; nanmean / nanstd of dB.
  gates           replaces sA11_final.py. sA11 did ((coh >= 0.337) & (vv_db >= -10.10)).fillna(False),
                  which silently dropped every detection without SAR, then applied 69 manual rejects.
                  Here (decision, Vivek): S1 gate everywhere, NISAR gate only where a coherence raster
                  covers the polygon, no manual rejects; a gate that cannot be evaluated lowers the
                  confidence and flags drafter review instead of dropping (see evaluate_gates).
"""
from __future__ import annotations

import math
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize, shapes
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from shapely import affinity
from shapely.geometry import shape

EQ = "EPSG:7755"           # WGS 84 / India NSF LCC, the area CRS of the production chain
NISAR_DX, NISAR_DY = 15.0, 5.0   # radar layover, measured over 993 towers (sA07/sA08)


# ---- polygonise (sA04_score.polygonise) ------------------------------------------------------
def polygonise(path, thresh, min_ha, eq_crs=EQ, step=8192, ov=256):
    ds = rasterio.open(path)
    cut = int(round(thresh * 255))
    px = abs(ds.transform.a * ds.transform.e)
    min_px = int(min_ha * 1e4 / px)
    polys = []
    H, W, STEP, OV = ds.height, ds.width, step, ov
    for y in range(0, H, STEP - OV):
        for x in range(0, W, STEP - OV):
            h, w = min(STEP, H - y), min(STEP, W - x)
            if h < 8 or w < 8:
                continue
            win = rasterio.windows.Window(x, y, w, h)
            a = ds.read(1, window=win)
            m = a >= cut
            if not m.any():
                continue
            lab, n = ndimage.label(m)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())
            lut = np.zeros(sizes.size, dtype=bool)
            lut[1:] = sizes[1:] >= min_px
            keep = lut[lab]
            if not keep.any():
                continue
            tf = ds.window_transform(win)
            for geom, v in shapes(keep.astype(np.uint8), mask=keep, transform=tf):
                if v == 1:
                    polys.append(geom)
    g = gpd.GeoDataFrame(geometry=[shape(p) for p in polys], crs=ds.crs)
    ds.close()
    if not len(g):
        g["ha"] = pd.Series(dtype=float)
        return g
    g = g.dissolve().explode(index_parts=False).reset_index(drop=True)
    g["ha"] = g.to_crs(eq_crs).area / 1e4
    return g[g["ha"] >= min_ha].reset_index(drop=True)


# ---- OSM match (sA06_layers L41-46) ----------------------------------------------------------
def read_osm(path):
    """OSM substations from an upstream gpkg; layer `features` if present, else the first layer
    whose name mentions substation, else the only/first layer. Returns 4326 or None."""
    if not path:
        return None
    import pyogrio
    layers = [l[0] for l in pyogrio.list_layers(path)]
    if not layers:
        return None
    lay = ("features" if "features" in layers else
           next((l for l in layers if "substation" in l.lower()), layers[0]))
    g = gpd.read_file(path, layer=lay)
    if g.crs is None:
        g = g.set_crs(4326)
    return g.to_crs(4326)


def osm_match(det, osm):
    """det/osm in 4326. Returns (matched bool array, name list, osm_id list)."""
    n = len(det)
    matched = np.zeros(n, bool)
    names = [None] * n
    ids = [None] * n
    if osm is None or not len(osm) or not n:
        return matched, names, ids
    oidx = osm.sindex
    has_name = "name" in osm.columns
    idcol = next((c for c in ("osm_id", "id", "src_id") if c in osm.columns), None)
    for j, geom in enumerate(det.geometry):
        cand = osm.iloc[list(oidx.intersection(geom.bounds))]
        if not len(cand):
            continue
        hit = cand[cand.intersects(geom)]
        if not len(hit):
            continue
        matched[j] = True
        # largest overlap first so a detection over two OSM rows takes the main one's name
        g3857 = gpd.GeoSeries([geom], crs=4326).to_crs(3857).iloc[0]
        ov = hit.geometry.to_crs(3857).intersection(g3857).area.values
        hit = hit.iloc[np.argsort(-ov, kind="stable")]
        if has_name:
            nm = [v for v in hit["name"] if isinstance(v, str) and v.strip()]
            names[j] = nm[0] if nm else None
        if idcol:
            ids[j] = str(hit[idcol].iloc[0])
    return matched, names, ids


# ---- minimum width (sA09_cascade_real L46-53) ------------------------------------------------
def minwidth(poly):
    mrr = poly.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return 0.0
    x, y = mrr.exterior.coords.xy
    pts = list(zip(x, y))
    L = [math.dist(pts[i], pts[i + 1]) for i in range(4)]
    return min(L[0], L[1])


# ---- NISAR coherence (sA08_cascade.coh_of) ---------------------------------------------------
def _shift_to(ds_crs, gdf, dx, dy):
    """Polygons in ds_crs, shifted dx east / dy north in metres."""
    crs = CRS.from_user_input(ds_crs)
    if crs.is_projected and (crs.linear_units or "").lower() in ("metre", "meter", "m"):
        g = gdf.to_crs(crs)
        return [affinity.translate(geom, xoff=dx, yoff=dy) for geom in g.geometry]
    g = gdf.to_crs(EQ)
    sh = gpd.GeoSeries([affinity.translate(geom, xoff=dx, yoff=dy) for geom in g.geometry],
                       crs=EQ).to_crs(crs)
    return list(sh)


def coh_of(ds, gdf, dx=NISAR_DX, dy=NISAR_DY):
    out = []
    for gs in _shift_to(ds.crs, gdf, dx, dy):
        try:
            a, _ = mask(ds, [gs.__geo_interface__], crop=True, filled=False)
            v = a.compressed() if hasattr(a, "compressed") else a.ravel()
            v = v[np.isfinite(v)]
            v = v[(v > 0) & (v <= 1)]
            out.append(float(np.median(v)) if v.size else np.nan)
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)


def _footprint_4326(entry, ds):
    from shapely.geometry import box
    fp = entry.get("footprint")
    if fp:
        g = shape(fp.get("geometry", fp)) if fp.get("type") == "Feature" else shape(fp)
        return g
    b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    return box(*b)


def nisar_coh(entries, det4326, log=print):
    """Returns (coh array, applicable bool array). A polygon is applicable when it meets the
    footprint of a registered coherence raster; coh is NaN where applicable but no valid pixel
    was found (counted as a gate that could not be evaluated)."""
    n = len(det4326)
    coh = np.full(n, np.nan)
    appl = np.zeros(n, bool)
    for e in entries or []:
        try:
            ds = rasterio.open(e["path"])
        except Exception as ex:
            log("NISAR open failed %s: %r" % (e.get("path"), ex))
            continue
        with ds:
            fp = _footprint_4326(e, ds)
            here = det4326.intersects(fp).values
            todo = here & ~np.isfinite(coh)
            appl |= here
            if todo.any():
                v = coh_of(ds, det4326[todo])
                coh[np.where(todo)[0]] = v
    return coh, appl


# ---- Sentinel-1 RTC VV (sA10_s1 L51-96) ------------------------------------------------------
def select_s1_scenes(entries, n_max, aoi=None):
    """One scene per (relative orbit, month), newest first, at most n_max (sA10 L51-55)."""
    es = [e for e in (entries or []) if e.get("path")]
    if aoi is not None:
        es = [e for e in es if not e.get("footprint") or shape(
            e["footprint"].get("geometry", e["footprint"])
            if e["footprint"].get("type") == "Feature" else e["footprint"]).intersects(aoi)]
    es.sort(key=lambda e: str(e.get("date") or ""), reverse=True)
    byorb = {}
    for e in es:
        orb = e.get("relative_orbit", e.get("sat:relative_orbit", e.get("orbit")))
        k = (orb, str(e.get("date") or "")[:7])
        byorb.setdefault(k, e)
    return list(byorb.values())[:n_max]


def _looks_like_db(ds) -> bool:
    """Linear gamma0 is > 0; a raster whose valid pixels are mostly negative is already dB."""
    h, w = min(ds.height, 512), min(ds.width, 512)
    a = ds.read(1, out_shape=(h, w), masked=True).astype("float32")
    v = np.asarray(a.compressed() if hasattr(a, "compressed") else a).ravel()
    v = v[np.isfinite(v) & (v != 0)]
    return bool(v.size) and float((v < 0).mean()) > 0.5


def s1_scene_means(ds, gdf, is_db=False):
    """Mean linear gamma0 of valid pixels inside each polygon for one scene (NaN if none)."""
    vals = np.full(len(gdf), np.nan)
    g = gdf.to_crs(ds.crs)
    wb = ds.bounds
    for i, geom in enumerate(g.geometry):
        b = geom.bounds
        if b[2] < wb[0] or b[0] > wb[2] or b[3] < wb[1] or b[1] > wb[3]:
            continue
        try:
            w = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
            w = w.intersection(Window(0, 0, ds.width, ds.height))
            if w.width < 1 or w.height < 1:
                continue
            a = ds.read(1, window=w).astype("float32")
            if ds.nodata is not None and np.isfinite(ds.nodata):
                a[a == ds.nodata] = np.nan
            if is_db:
                a = np.power(10.0, a / 10.0, dtype="float32")
            m = rasterize([(geom, 1)], out_shape=a.shape,
                          transform=ds.window_transform(w), fill=0, dtype="uint8").astype(bool)
            m &= np.isfinite(a) & (a > 0)
            if m.sum():
                vals[i] = float(a[m].mean())
        except Exception:
            pass
    return vals


def s1_aggregate(stack):
    """stack (n_scenes, n_polys) linear means -> (vv_db mean, vv_db std, n valid scenes)."""
    n_polys = stack.shape[1] if stack.ndim == 2 else 0
    if stack.size == 0:
        return np.full(n_polys, np.nan), np.full(n_polys, np.nan), np.zeros(n_polys, int)
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10 * np.log10(stack)
    db[~np.isfinite(db)] = np.nan
    nvalid = np.isfinite(db).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(db, axis=0)
        std = np.nanstd(db, axis=0)
    return mean, std, nvalid


def s1_vv(scenes, det4326, log=print):
    stack, used = [], []
    for e in scenes:
        try:
            ds = rasterio.open(e["path"])
        except Exception as ex:
            log("S1 open failed %s: %r" % (e.get("path"), ex))
            continue
        with ds:
            is_db = _looks_like_db(ds)
            if is_db:
                log("S1 scene %s looks like dB, converting to linear" % e.get("scene_id"))
            v = s1_scene_means(ds, det4326, is_db)
        stack.append(v)
        used.append(e)
        log("S1 scene %s %s: %d/%d polygons covered"
            % (e.get("scene_id"), e.get("date"), int(np.isfinite(v).sum()), len(v)))
    A = np.vstack(stack) if stack else np.empty((0, len(det4326)))
    mean, std, nvalid = s1_aggregate(A)
    return mean, std, nvalid, used


# ---- gates / confidence ----------------------------------------------------------------------
# Vivek, 5 Oct 2026: without the manual reject step, new detections that pass the SAR gates
# were ~29% correct on Bhuj–Lakadia (67 of 94 were rejected by hand) -> low confidence + review.
# Ungated (not evaluable) ones rank below gated ones. Overridable via params conf_gated /
# conf_unchecked in config/models.yaml.
CONF_OSM, CONF_GATED, CONF_REVIEW = 0.90, 0.30, 0.25


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def evaluate_gates(vv_db, coh, nisar_applicable, osm_matched, vv_db_min, coh_min,
                   conf_gated=None, conf_unchecked=None):
    """One detection -> dict(keep, confidence, review, gates, reason).

    S1 VV gate everywhere; NISAR gate only where a coherence raster covers the polygon.
      a gate that is evaluated and fails     -> dropped (kept in `rejected` with the reason),
                                                OSM-matched or not (production gated TPs too)
      OSM-matched, no failed gate            -> 0.90
      every applicable gate evaluated, pass  -> 0.30, review = True
      an applicable gate not evaluable (no S1 pixels; NISAR registered but no valid pixel)
                                             -> 0.25, review = True
    Comparisons are inclusive (>=), as sA11. NaN / None / inf mean "not evaluable".
    """
    parts, fails, na = [], [], False
    if _finite(vv_db):
        ok = float(vv_db) >= vv_db_min
        parts.append("s1_vv:%s(%.2f%s%.2f)" % ("pass" if ok else "fail", float(vv_db),
                                                ">=" if ok else "<", vv_db_min))
        if not ok:
            fails.append("s1_vv %.2f < %.2f dB" % (float(vv_db), vv_db_min))
    else:
        parts.append("s1_vv:na")
        na = True
    if not nisar_applicable:
        parts.append("nisar_coh:not_covered")
    elif _finite(coh):
        ok = float(coh) >= coh_min
        parts.append("nisar_coh:%s(%.3f%s%.3f)" % ("pass" if ok else "fail", float(coh),
                                                    ">=" if ok else "<", coh_min))
        if not ok:
            fails.append("nisar_coh %.3f < %.3f" % (float(coh), coh_min))
    else:
        parts.append("nisar_coh:na")
        na = True
    gates = ";".join(parts)
    if fails:
        return dict(keep=False, confidence=None, review=False, gates=gates, reason="; ".join(fails))
    if osm_matched:
        return dict(keep=True, confidence=CONF_OSM, review=False, gates=gates, reason=None)
    if na:
        return dict(keep=True, confidence=CONF_REVIEW if conf_unchecked is None else conf_unchecked,
                    review=True, gates=gates, reason=None)
    return dict(keep=True, confidence=CONF_GATED if conf_gated is None else conf_gated,
                review=True, gates=gates, reason=None)


def apply_gates(det, vv_db_min, coh_min, conf_gated=None, conf_unchecked=None):
    """det has columns vv_db, coh, nisar_applicable, matches_osm -> adds confidence/review/
    gates/reject_reason/keep."""
    rows = [evaluate_gates(v, c, bool(a), bool(m), vv_db_min, coh_min, conf_gated, conf_unchecked)
            for v, c, a, m in zip(det["vv_db"], det["coh"], det["nisar_applicable"],
                                  det["matches_osm"])]
    det = det.copy()
    det["keep"] = [r["keep"] for r in rows]
    det["confidence"] = [r["confidence"] for r in rows]
    det["review"] = [r["review"] for r in rows]
    det["gates"] = [r["gates"] for r in rows]
    det["reject_reason"] = [r["reason"] for r in rows]
    return det
