"""Buildings (footprint) model plugin — Esri India SamLoRA, Bhuj–Lakadia fine-tune (Vivek, 6 Oct 2026).

Model: arcgis.learn SamLoRA (SAM ViT-L image encoder, LoRA rank 4, 256 px chips at 0.305 m/px)
from BUILDING_FOOTPRINT_EXTRACTION_-_INDIA.dlpk, fine-tuned on the corridor
(bhuj-lakadia/output/finetune_dataset/checkpoints/best.pth = epoch 5, lowest false-positive rate;
see FINETUNE_REPORT.md). The architecture is vendored in sam_lora.py.

Inference follows bhuj-lakadia/scripts/05_run_building_footprint_model.py:
  each Esri Wayback z18 tile is upsampled to the model's 0.305 m/px (latitude-corrected), cut
  into 4 overlapping 256 px chips, run in bf16, averaged, thresholded at 0.5 and polygonised.
Clean-up follows 13_clean_gpkg.py: polygons split by tile seams are merged, repaired, measured
in the local UTM zone, < 4 m² dropped, > 5000 m² flagged `oversize` (yards, salt pans, merged
roofs) but kept.

Imagery: only the MBTiles listed in request.imagery.esri_wayback_z18 (catalog items on this
machine). Nothing is fetched. Where releases overlap, the newest one is used for a tile.

    python run_aoi.py [--check] --request request.json
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.environ.get("PYTHONPATH", str(HERE.parent)))
import common  # noqa: E402

REPO = HERE.parent.parent
MODEL = "buildings-samlora-vitl"
MODEL_VERSION = "Esri India SamLoRA r4 + Bhuj–Lakadia fine-tune epoch 5 (best.pth)"
ZOOM = 18
MODEL_CELL_M = 0.30517578125               # dlpk training resolution
WEB_MERC_CELL_Z0 = 156543.03392            # EPSG:3857 m/px at zoom 0
CHIP = 256
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
R = 6378137.0 * math.pi

DEFAULTS = {"threshold": 0.5, "min_area_m2": 4.0, "simplify_m": 0.5, "oversize_m2": 5000.0,
            "batch_tiles": 8, "decode_workers": 6}
COLS = ["class", "name", "confidence", "src_id", "area_m2", "perimeter_m", "oversize",
        "image_release", "image_date"]


# ---- request helpers ---------------------------------------------------------------------
def _resolve(p):
    if not p:
        return p
    q = Path(p)
    if q.is_absolute():
        return str(q)
    if q.exists():
        return str(q.resolve())
    r = REPO / q
    return str(r) if r.exists() else str(q.resolve())


def _aoi_geom(aoi):
    from shapely.geometry import shape
    from shapely.ops import unary_union
    if aoi.get("type") == "FeatureCollection":
        return unary_union([shape(f["geometry"]) for f in aoi["features"]])
    if aoi.get("type") == "Feature":
        return shape(aoi["geometry"])
    return shape(aoi)


def _connect(path):
    # immutable=1: never create a journal/lock next to a read-only archive
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)


def _zoom_of(path):
    with _connect(path) as c:
        md = dict(c.execute("select name, value from metadata").fetchall())
        if md.get("maxzoom") is not None:
            return int(md["maxzoom"])
        return int(c.execute("select max(zoom_level) from tiles").fetchone()[0])


def _tile_lonlat_bounds(x, y, z):
    n = 2 ** z
    w, e = x / n * 360.0 - 180.0, (x + 1) / n * 360.0 - 180.0
    nlat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    slat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return w, slat, e, nlat


def plan_tiles(req, aoi, warnings):
    """[(item, path, [(x, y), ...])], newest release first; a tile is read from one release only."""
    from shapely.geometry import box
    from shapely.prepared import prep
    items = sorted((req.get("imagery") or {}).get("esri_wayback_z18") or [],
                   key=lambda e: str(e.get("date") or ""), reverse=True)
    paoi = prep(aoi)
    minx, miny, maxx, maxy = aoi.bounds
    x0, y0 = common.lonlat_to_tile(minx, maxy, ZOOM)
    x1, y1 = common.lonlat_to_tile(maxx, miny, ZOOM)
    n = 2 ** ZOOM
    taken, plan = set(), []
    for e in items:
        path = _resolve(e.get("path"))
        if not path or not Path(path).is_file():
            warnings.append(f"Wayback MBTiles missing: {path}")
            continue
        z = _zoom_of(path)
        if z != ZOOM:
            warnings.append(f"skipped {Path(path).name}: max zoom {z}, the model needs z{ZOOM}")
            continue
        with _connect(path) as c:
            rows = c.execute("select tile_column, tile_row from tiles where zoom_level=? and "
                             "tile_column between ? and ? and tile_row between ? and ?",
                             (ZOOM, x0, x1, n - 1 - y1, n - 1 - y0)).fetchall()
        tiles = []
        for col, trow in rows:
            xy = (col, n - 1 - trow)
            if xy in taken or not paoi.intersects(box(*_tile_lonlat_bounds(*xy, ZOOM))):
                continue
            taken.add(xy)
            tiles.append(xy)
        if tiles:
            plan.append((e, path, sorted(tiles)))
            common.log(f"release {e.get('release')} ({e.get('date')}): {len(tiles):,} tiles")
    if len(plan) > 1:
        warnings.append(f"AOI spans {len(plan)} Wayback releases; each tile is read from the newest one")
    return plan


def coverage_3857(tiles):
    """Union of processed tile squares, polygonised on the tile grid (EPSG:3857)."""
    import numpy as np
    from affine import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union
    xs, ys = [t[0] for t in tiles], [t[1] for t in tiles]
    x0, y0 = min(xs), min(ys)
    grid = np.zeros((max(ys) - y0 + 1, max(xs) - x0 + 1), np.uint8)
    for x, y in tiles:
        grid[y - y0, x - x0] = 1
    size = 2 * R / 2 ** ZOOM
    tf = Affine(size, 0, -R + x0 * size, 0, -size, R - y0 * size)
    return unary_union([shape(g) for g, v in shapes(grid, mask=grid.astype(bool), transform=tf) if v == 1])


# ---- model -------------------------------------------------------------------------------
def load_model(ckpt):
    import torch
    from sam_lora import LoRA_Sam, sam_model_registry
    sam, _ = sam_model_registry["vit_l"](image_size=CHIP, num_classes=1, checkpoint=None)
    model = LoRA_Sam(sam, 4)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
    return model.cuda().eval()


def prepare(conn, lock, x, y):
    """Decode one tile, upsample to the model's ground resolution, cut 4 overlapping chips."""
    import io
    import cv2
    import numpy as np
    from PIL import Image
    n = 2 ** ZOOM
    with lock:
        row = conn.execute("select tile_data from tiles where zoom_level=? and tile_column=? and tile_row=?",
                           (ZOOM, x, n - 1 - y)).fetchone()
    if row is None:
        return x, y, None, None, None
    rgb = np.asarray(Image.open(io.BytesIO(row[0])).convert("RGB"))
    w, s, e, nl = _tile_lonlat_bounds(x, y, ZOOM)
    ground = WEB_MERC_CELL_Z0 / 2 ** ZOOM * math.cos(math.radians((s + nl) / 2))
    size = max(CHIP, round(CHIP * ground / MODEL_CELL_M))
    up = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    st = size - CHIP
    origins = [(0, 0)] if st == 0 else [(0, 0), (0, st), (st, 0), (st, st)]
    mean, std = np.array(IMAGENET_MEAN, np.float32), np.array(IMAGENET_STD, np.float32)
    chips = np.stack([(up[oy:oy + CHIP, ox:ox + CHIP].astype(np.float32) / 255.0 - mean) / std
                      for oy, ox in origins]).transpose(0, 3, 1, 2)
    return x, y, chips, origins, size


def bounded_map(pool, fn, items, window):
    """pool.map with backpressure (05's run() hit 98 GB RSS without it)."""
    it, futs = iter(items), deque()
    for _ in range(window):
        try:
            futs.append(pool.submit(fn, next(it)))
        except StopIteration:
            break
    while futs:
        f = futs.popleft()
        try:
            futs.append(pool.submit(fn, next(it)))
        except StopIteration:
            pass
        yield f.result()


def vectorize(prob, x, y, size, threshold):
    """Threshold one tile's probability map -> [(polygon EPSG:3857, mean prob)]."""
    import numpy as np
    from rasterio.features import shapes
    from rasterio.transform import from_bounds
    from scipy import ndimage
    from shapely.geometry import shape
    mask = prob >= threshold
    if not mask.any():
        return []
    lab, k = ndimage.label(mask)                     # 4-connected, same as rasterio shapes
    mean = ndimage.mean(prob, lab, np.arange(1, k + 1))
    tf = from_bounds(*common.tile_bounds_3857(x, y, ZOOM), size, size)
    return [(shape(g), float(mean[int(v) - 1]))
            for g, v in shapes(lab.astype(np.int32), mask=mask, transform=tf) if v > 0]


def infer(plan, ckpt, params, frac0=0.05, frac1=0.85):
    """-> list of (polygon 3857, mean prob, tile key, release, date)."""
    import threading
    import numpy as np
    import torch
    model = load_model(ckpt)
    total = sum(len(t) for _, _, t in plan)
    out, done, empty = [], 0, 0
    step = max(1, total // 100)
    for e, path, tiles in plan:
        conn, lock = _connect(path), threading.Lock()
        group = []

        def flush():
            nonlocal group
            if not group:
                return
            batch = torch.from_numpy(np.concatenate([g[2] for g in group])).float().cuda()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            i = 0
            for x, y, _, origins, size in group:
                acc = np.zeros((size, size), np.float32)
                cnt = np.zeros((size, size), np.float32)
                for oy, ox in origins:
                    acc[oy:oy + CHIP, ox:ox + CHIP] += probs[i]
                    cnt[oy:oy + CHIP, ox:ox + CHIP] += 1
                    i += 1
                for g, p in vectorize(acc / cnt, x, y, size, params["threshold"]):
                    out.append((g, p, (x, y), e.get("release"), e.get("date")))
            group = []

        with ThreadPoolExecutor(max_workers=int(params["decode_workers"])) as pool:
            for x, y, chips, origins, size in bounded_map(
                    pool, lambda xy: prepare(conn, lock, *xy), tiles, int(params["decode_workers"]) * 3):
                done += 1
                if chips is None:
                    empty += 1
                else:
                    group.append((x, y, chips, origins, size))
                    if len(group) >= int(params["batch_tiles"]):
                        flush()
                if done % step == 0:
                    common.progress(frac0 + (frac1 - frac0) * done / total,
                                    f"footprints: {done:,}/{total:,} tiles, {len(out):,} raw polygons")
            flush()
        conn.close()
    return out, empty


def clean(raw, aoi, params):
    """Seam merge, repair, UTM measures, speckle drop, simplify, oversize flag, clip to AOI."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    g = gpd.GeoDataFrame({"conf": [r[1] for r in raw], "tile": [f"{r[2][0]}_{r[2][1]}" for r in raw],
                          "image_release": [str(r[3]) for r in raw], "image_date": [r[4] for r in raw]},
                         geometry=[r[0] for r in raw], crs=3857)
    # polygons that touch across a tile edge are one building cut by the seam
    a, b = g.sindex.query(g.geometry, predicate="intersects")
    keep = (a < b) & (g["tile"].values[a] != g["tile"].values[b])
    adj = coo_matrix((np.ones(int(keep.sum())), (a[keep], b[keep])), shape=(len(g), len(g)))
    g["comp"] = connected_components(adj, directed=False)[1]
    g["w"] = g.geometry.area
    g["cw"] = g["conf"] * g["w"]
    m = g.dissolve(by="comp", aggfunc={"cw": "sum", "w": "sum", "image_release": "first", "image_date": "first"})
    m["confidence"] = (m["cw"] / m["w"]).round(3)
    lon, lat = aoi.centroid.x, aoi.centroid.y
    utm = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
    m = m.to_crs(utm)
    m["geometry"] = m.geometry.make_valid()
    m = m.explode(index_parts=False)
    m = m[m.geom_type == "Polygon"]
    m = m[m.geometry.area >= float(params["min_area_m2"])]
    m["geometry"] = m.geometry.simplify(float(params["simplify_m"]), preserve_topology=True)
    m = m.to_crs(4326)
    m = gpd.clip(m, aoi).explode(index_parts=False)
    m = m[m.geom_type == "Polygon"].to_crs(utm)
    m = m[m.geometry.area >= float(params["min_area_m2"])]
    m["area_m2"] = m.geometry.area.round(1)
    m["perimeter_m"] = m.geometry.length.round(1)
    m["oversize"] = m["area_m2"] > float(params["oversize_m2"])
    m = m.to_crs(4326).reset_index(drop=True)
    c = m.geometry.representative_point()
    m = m.assign(_x=c.x, _y=c.y).sort_values(["_x", "_y"]).reset_index(drop=True)
    m["class"] = "building"
    m["name"] = pd.Series([None] * len(m), dtype="object")
    m["src_id"] = [f"bldg-m-{i + 1}" for i in range(len(m))]
    return gpd.GeoDataFrame(m[COLS], geometry=m.geometry, crs=4326)


def _empty():
    import geopandas as gpd
    import pandas as pd
    return gpd.GeoDataFrame({c: pd.Series(dtype="object") for c in COLS},
                            geometry=gpd.GeoSeries([], crs=4326), crs=4326)


def _write(path, g):
    if Path(path).exists():
        Path(path).unlink()
    g.to_file(path, layer="features", driver="GPKG", engine="pyogrio", geometry_type="Polygon")


# ---- entry -------------------------------------------------------------------------------
def check(req, aoi, params):
    warnings = []
    plan = plan_tiles(req, aoi, warnings)
    ckpt = _resolve((req.get("weights") or {}).get("checkpoint"))
    rep = {"plugin": "buildings", "model": MODEL, "model_version": MODEL_VERSION, "params": params,
           "checkpoint": ckpt, "checkpoint_ok": bool(ckpt and Path(ckpt).is_file()),
           "imagery": [{"release": e.get("release"), "date": e.get("date"), "path": p, "tiles": len(t)}
                       for e, p, t in plan],
           "tiles": sum(len(t) for _, _, t in plan), "warnings": warnings}
    try:
        import cv2, scipy, torch  # noqa: F401,E401
        import sam_lora  # noqa: F401
        rep["imports_ok"], rep["cuda"] = True, bool(torch.cuda.is_available())
    except Exception as ex:  # noqa: BLE001
        rep["imports_ok"], rep["import_error"] = False, repr(ex)
    rep["ok"] = rep["checkpoint_ok"] and rep["imports_ok"]
    return rep


def main():
    req = common.load_request()
    for p in ("work_dir", "out_gpkg", "result_json"):
        req[p] = _resolve(req[p])
    params = dict(DEFAULTS)
    params.update(req.get("params") or {})
    aoi = _aoi_geom(req["aoi"])

    if "--check" in sys.argv:
        rep = check(req, aoi, params)
        txt = json.dumps(rep, indent=1, default=str)
        Path(req["work_dir"], "check.json").write_text(txt)
        print(txt, flush=True)
        sys.exit(0 if rep["ok"] else 1)

    common.progress(0.0, "start")
    import geopandas as gpd
    from shapely.geometry import GeometryCollection, mapping
    warnings = []
    plan = plan_tiles(req, aoi, warnings)
    if not plan:
        warnings.append("no Esri Wayback z18 imagery on this machine for the AOI; no footprints produced")
        _write(req["out_gpkg"], _empty())
        common.write_result(req, coverage=mapping(GeometryCollection()), image_source=None, image_dates=[],
                            model=MODEL, model_version=MODEL_VERSION, counts={"features": 0},
                            warnings=warnings, params=params)
        common.progress(1.0, "no imagery")
        return

    ckpt = _resolve((req.get("weights") or {}).get("checkpoint"))
    if not ckpt or not Path(ckpt).is_file():
        raise SystemExit(f"checkpoint not found: {ckpt} — run scripts/install_models.py")
    vram = common.gpu_env(req)
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; the building footprint model needs a GPU")
    common.torch_vram_cap(vram)
    common.progress(0.02, "loading model")
    raw, empty = infer(plan, ckpt, params)
    if empty:
        warnings.append(f"{empty:,} listed tile(s) had no image data")
    common.progress(0.87, f"cleaning {len(raw):,} raw polygons")
    feats = clean(raw, aoi, params) if raw else _empty()
    _write(req["out_gpkg"], feats)

    tiles = [t for _, _, ts in plan for t in ts]
    cov = gpd.GeoSeries([coverage_3857(tiles)], crs=3857).to_crs(4326).iloc[0].intersection(aoi)
    rel = sorted({str(e.get("release")) for e, _, _ in plan})
    n_over = int(feats["oversize"].astype(bool).sum()) if len(feats) else 0
    common.write_result(
        req, coverage=mapping(cov), image_source="Esri Wayback z18 release " + ", ".join(rel),
        image_dates=sorted({str(e.get("date")) for e, _, _ in plan if e.get("date")}),
        model=MODEL, model_version=MODEL_VERSION,
        counts={"features": len(feats), "oversize": n_over, "raw_polygons": len(raw), "tiles": len(tiles)},
        warnings=warnings, params=params,
        notes=["model trained at 0.305 m/px; z18 tiles (~0.55 m/px) are upsampled ~1.8x as in the "
               "corridor run. Corridor validation: precision 0.41, recall 0.33 (FINETUNE_REPORT.md)."])
    common.progress(1.0, f"done: {len(feats):,} footprints ({n_over} oversize)")


if __name__ == "__main__":
    main()
