"""Seasonal Sentinel-2 features per detection polygon, from LOCAL band files only.

Port of /mnt/drso-data/solar_lora/scripts/14_s2_seasonal.py with the STAC search and /vsicurl
reads replaced by the request's `imagery.sentinel2_l2a` items:
    {"scene_id", "date", "cloud", "bands": {"B04","B08","B11","B12": path}, "footprint", ...}

Kept from the research code:
  - scene choice: per calendar month, per MGRS tile, the lowest-cloud scene with cloud < max_cloud
  - resolvability gate: rasterise each polygon on a 20 m grid in EPSG:7755, erode one pixel per
    20 m, resolvable if >= min_pure_px whole pixels remain
  - per scene: per-band median of the pixels inside the polygon (whole window when the polygon
    covers no pixel centre), values <= 0 dropped; an observation needs all four bands
  - features over observations: nir_med, ndvi_med, ndvi_std, swir_ratio_med (B11/B12), n_obs
Changed: read/open errors are counted and returned as warnings instead of being swallowed.

Value convention: L2A DN = reflectance x 10000 with the BOA offset already removed (Earth Search
v1 COGs with `earthsearch:boa_offset_applied: true`, which is what the frozen NIR band was
calibrated on). An item carrying `"boa_offset_applied": false` has 1000 subtracted.
"""
from __future__ import annotations

import collections
import re

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

PX = 20.0
BANDS = {"nir": "B08", "red": "B04", "swir16": "B11", "swir22": "B12"}
_MGRS = re.compile(r"(?:^|[_T])(\d{2}[C-X][A-Z]{2})(?:_|$)")


def tile_key(item: dict) -> str:
    for k in ("mgrs_tile", "mgrs", "tile", "grid_code"):
        if item.get(k):
            return str(item[k]).replace("MGRS-", "")
    m = _MGRS.search(str(item.get("scene_id", "")))
    if m:
        return m.group(1)
    return "nogrid:" + str(item.get("bands", {}).get("B08", item.get("scene_id")))


def select_scenes(items: list[dict], max_cloud: float = 20.0) -> tuple[list[dict], list[str]]:
    """One lowest-cloud scene per (month, MGRS tile) with cloud < max_cloud."""
    warns = []
    best: dict = {}
    for it in items:
        bands = it.get("bands") or {}
        missing = [b for b in BANDS.values() if not bands.get(b)]
        if missing:
            warns.append(f"S2 {it.get('scene_id')}: missing bands {missing}, skipped")
            continue
        c = it.get("cloud")
        c = 100.0 if c is None else float(c)
        if c >= max_cloud:
            continue
        key = (str(it.get("date", ""))[:7], tile_key(it))
        if key not in best or c < best[key][0]:
            best[key] = (c, it)
    return [v[1] for k, v in sorted(best.items())], warns


def resolvable(gdf, min_pure_px: int = 4, erode_m: float = 20.0):
    """Whole 20 m pixels surviving an erode_m inward erosion (rasterised, as in 14)."""
    from rasterio.transform import from_origin
    from scipy import ndimage as nd

    eq = gdf.to_crs("EPSG:7755")
    iters = max(1, int(round(erode_m / PX)))
    areas = np.zeros(len(eq), dtype=float)
    for i, geom in enumerate(eq.geometry):
        if geom is None or geom.is_empty:
            continue
        x0, y0, x1, y1 = geom.bounds
        w = int(np.ceil((x1 - x0) / PX)) + 4
        h = int(np.ceil((y1 - y0) / PX)) + 4
        if w < 3 or h < 3 or w * h > 4_000_000:
            continue
        tr = from_origin(x0 - 2 * PX, y1 + 2 * PX, PX, PX)
        m = rasterize([(geom, 1)], out_shape=(h, w), transform=tr, fill=0, dtype="uint8",
                      all_touched=False).astype(bool)
        if not m.any():
            continue
        er = nd.binary_erosion(m, structure=nd.generate_binary_structure(2, 1), iterations=iters,
                               border_value=0)
        areas[i] = er.sum() * PX * PX
    return areas >= min_pure_px * PX * PX, areas


def _sample_scene(gdf, item, store, errs, cache=None) -> int:
    paths = {k: item["bands"][b] for k, b in BANDS.items()}
    srcs = {}
    try:
        for k, p in paths.items():
            srcs[k] = rasterio.open(p)
    except Exception as e:  # noqa: BLE001 — recorded, scene skipped
        errs.append(f"S2 {item.get('scene_id')}: cannot open {k} ({p}): {type(e).__name__}: {e}")
        for s in srcs.values():
            s.close()
        return 0
    off = 1000.0 if item.get("boa_offset_applied") is False else 0.0
    n = 0
    fails = 0
    first = None
    try:
        ref = srcs["nir"]
        key = ref.crs.to_string()
        masks = cache.setdefault("masks", {}) if cache is not None else None
        if cache is not None and key in cache:
            g = cache[key]
        else:
            g = gdf.to_crs(ref.crs)
            if cache is not None:
                cache[key] = g
        wb = ref.bounds
        for i, geom in enumerate(g.geometry):
            if geom is None or geom.is_empty:
                continue
            b = geom.bounds
            if b[2] < wb[0] or b[0] > wb[2] or b[3] < wb[1] or b[1] > wb[3]:
                continue
            vals = {}
            ok = True
            for k, ds in srcs.items():
                try:
                    w = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
                    if w.width < 1 or w.height < 1:
                        w = from_bounds(b[0] - 30, b[1] - 30, b[2] + 30, b[3] + 30,
                                        transform=ds.transform).round_offsets().round_lengths()
                    arr = ds.read(1, window=w, boundless=True, fill_value=0).astype(float)
                    tr = ds.window_transform(w)
                    # scenes of one MGRS tile share the pixel grid: rasterise each polygon once
                    mk = (key, i, tuple(tr)[:6], arr.shape)
                    m = masks.get(mk) if masks is not None else None
                    if m is None:
                        m = rasterize([(geom, 1)], out_shape=arr.shape, transform=tr, fill=0,
                                      dtype="uint8").astype(bool)
                        if masks is not None:
                            masks[mk] = m
                    v = arr[m] if m.any() else arr.ravel()
                    v = v[v > 0] - off
                    if v.size == 0:          # nodata over the polygon: no observation (not an error)
                        ok = False
                        break
                    vals[k] = float(np.median(v))
                except Exception as e:  # noqa: BLE001 — counted and reported
                    fails += 1
                    first = first or f"{k}: {type(e).__name__}: {e}"
                    ok = False
                    break
            if ok:
                store[i].append(vals)
                n += 1
    finally:
        for s in srcs.values():
            s.close()
    if fails:
        errs.append(f"S2 {item.get('scene_id')}: {fails} polygon read(s) failed, first: {first}")
    return n


def seasonal_features(gdf, items: list[dict], max_cloud: float = 20.0, progress=None, cancelled=None):
    """Adds n_obs, nir_med, ndvi_med, ndvi_std, swir_ratio_med to a copy of gdf.
    Returns (gdf, used_scenes, warnings)."""
    scenes, warns = select_scenes(items, max_cloud)
    store = collections.defaultdict(list)
    cache: dict = {}                       # polygons reprojected once per band CRS
    for k, sc in enumerate(scenes):
        if cancelled and cancelled():
            raise SystemExit("cancelled")
        _sample_scene(gdf, sc, store, warns, cache)
        if progress:
            progress((k + 1) / max(len(scenes), 1), f"S2 {sc.get('date')} {tile_key(sc)}")
    cols = {c: [] for c in ("ndvi_med", "ndvi_std", "nir_med", "swir_ratio_med", "n_obs")}
    for i in range(len(gdf)):
        rows = store.get(i, [])
        if not rows:
            for c in ("ndvi_med", "ndvi_std", "nir_med", "swir_ratio_med"):
                cols[c].append(np.nan)
            cols["n_obs"].append(0)
            continue
        nd = np.array([(r["nir"] - r["red"]) / max(r["nir"] + r["red"], 1) for r in rows])
        nir = np.array([r["nir"] for r in rows])
        sr = np.array([r["swir16"] / max(r["swir22"], 1) for r in rows])
        cols["ndvi_med"].append(float(np.median(nd)))
        cols["ndvi_std"].append(float(np.std(nd)))
        cols["nir_med"].append(float(np.median(nir)))
        cols["swir_ratio_med"].append(float(np.median(sr)))
        cols["n_obs"].append(len(rows))
    out = gdf.copy()
    for c, v in cols.items():
        out[c] = v
    return out, scenes, warns
