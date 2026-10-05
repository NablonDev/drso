"""Wind-turbine LULC plugin: Cartosat-3 PAN 0.45 m -> DINOv3-LoRA round4 mask -> Exp-G -> points.

    <terratorch python> lulc_models/wind_turbines/run_aoi.py --request <work_dir>/request.json [--check]

Per request PAN scene whose footprint intersects the AOI (whole scene, never clipped first: the
stretch and Exp-G calibrate on scene statistics):
  1. mask (GPU, turbine_mask.py) -> <work_dir>/masks/<scene>_round4_mask.tif. Skipped when a
     mask on the same grid already exists in params.mask_dir (str or list) or <work_dir>/masks.
  2. Exp-G (CPU, turbine_post.expg_scene) -> <work_dir>/<scene>_expG_detections.gpkg
     (scene CRS, columns area, n_frag).
Then: cross-scene NMS at params.nms_m in the AOI's UTM zone, clip to the AOI, OSM confirmation
against inputs.osm_wind_gpkg within params.osm_match_m (confidence 0.9 confirmed / 0.4
unconfirmed + review flag), out.gpkg layer `features` in EPSG:4326.

--check validates the environment without loading weights or touching the GPU: imports,
checkpoint + HF cache present, PAN files open (CRS, GSD, last block readable), footprints
intersect the AOI. Prints a JSON report and writes <work_dir>/check.json; exit 1 on problems.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))           # lulc_models/common.py

import common  # noqa: E402

MODEL = "turbine-dinov3-lora-round4"
MODEL_VERSION = "epoch25 + Exp-G"
IMAGE_SOURCE = "Cartosat-3 PAN 0.45 m"
HF_REPO_DIR = "hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m"
DEFAULTS = dict(thresh=160, min_px=15, nms_m=95.0, osm_match_m=60.0, gsd_m=0.45, gsd_tol=0.01,
                conf_confirmed=0.9, conf_unconfirmed=0.4, mask_dir=None)
OUT_COLUMNS = ["class", "confidence", "name", "src_id", "scene_id", "area", "n_frag", "n_views",
               "osm_confirmed", "osm_id", "review"]


def _resolve(p: str | None) -> str | None:
    """Absolute as given; a relative weights path (config/models.yaml style 'models/...') is tried
    against BEELINE_DATA."""
    if not p:
        return p
    if os.path.isabs(p) or os.path.exists(p):
        return p
    base = os.environ.get("BEELINE_DATA")
    return os.path.join(base, p) if base else p


def _params(req: dict) -> dict:
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (req.get("params") or {}).items() if v is not None})
    return p


def _mask_dirs(req: dict, p: dict) -> list[str]:
    md = p.get("mask_dir")
    dirs = [md] if isinstance(md, str) else list(md or [])
    return dirs + [os.path.join(req["work_dir"], "masks")]


def _same_grid(mask_path: str, scene: dict) -> bool:
    import rasterio
    try:
        with rasterio.open(mask_path) as m:
            return (m.width, m.height) == (scene["width"], scene["height"]) and \
                m.crs == scene["crs"] and m.transform.almost_equals(scene["transform"], precision=1e-3)
    except Exception:   # noqa: BLE001
        return False


def existing_mask(req: dict, p: dict, scene: dict) -> str | None:
    for d in _mask_dirs(req, p):
        f = os.path.join(d, f"{scene['scene_id']}_round4_mask.tif")
        if os.path.exists(f) and _same_grid(f, scene):
            return f
    return None


def survey_scenes(req: dict, aoi, p: dict, deep: bool = False) -> tuple[list[dict], list[str]]:
    """Open every cartosat3_pan item (header + TIFF tile index, so truncated copies are caught
    before a GPU pass; deep=True probes pixels when tifffile is unavailable). Returns scene
    records and warnings."""
    import rasterio
    import turbine_post as TP
    items = (req.get("imagery") or {}).get("cartosat3_pan") or []
    scenes, warns, seen = [], [], set()
    for it in items:
        sid = str(it.get("scene_id") or Path(it.get("path", "")).parent.name)
        rec = {"scene_id": sid, "path": it.get("path"), "opens": False, "use": False}
        if sid in seen:
            rec["skip"] = "duplicate scene_id in request"; scenes.append(rec); continue
        seen.add(sid)
        try:
            with rasterio.open(it["path"]) as src:
                rec.update(opens=True, crs=src.crs, transform=src.transform, width=src.width, height=src.height,
                           gsd=abs(src.res[0]), bounds=tuple(src.bounds), dtype=src.dtypes[0])
            nb = TP.blocks_past_eof(it["path"])
            rec["blocks_past_eof"] = nb
            if nb is None and deep:        # no tifffile: probe the last block and the centre
                import rasterio.windows as rw
                try:
                    with rasterio.open(it["path"]) as src:
                        src.read(1, window=rw.Window(src.width - 1, src.height - 1, 1, 1))
                        src.read(1, window=rw.Window(src.width // 2, src.height // 2, 1, 1))
                except Exception as e:   # noqa: BLE001
                    rec["blocks_past_eof"] = -1; rec["read_error"] = str(e)[:300]
        except Exception as e:   # noqa: BLE001
            rec["error"] = str(e)[:300]
            warns.append(f"{sid}: cannot open {it.get('path')}: {rec['error']}")
            scenes.append(rec); continue
        crs = rec["crs"]
        rec["crs_projected"] = bool(crs and crs.is_projected)
        rec["gsd_ok"] = rec["crs_projected"] and abs(rec["gsd"] - float(p["gsd_m"])) < float(p["gsd_tol"])
        fp = TP.scene_footprint_4326(it, raster_crs=crs, raster_bounds=rec["bounds"])
        rec["footprint"] = fp
        rec["intersects_aoi"] = bool(fp is not None and fp.intersects(aoi))
        rec["date"] = TP.acquisition_date(it)
        if not rec["intersects_aoi"]:
            rec["skip"] = "footprint does not intersect the AOI"
        elif rec.get("blocks_past_eof"):
            rec["skip"] = (f"raster truncated ({rec['blocks_past_eof']} tiles beyond end of file)"
                           if rec["blocks_past_eof"] > 0 else f"raster unreadable: {rec.get('read_error')}")
            warns.append(f"{sid}: skipped, {rec['skip']}")
        elif not rec["gsd_ok"]:
            rec["skip"] = f"GSD {rec['gsd']:.3f} (CRS {crs}) is not {p['gsd_m']} m"
            warns.append(f"{sid}: skipped, {rec['skip']}")
        else:
            rec["use"] = True
        scenes.append(rec)
    return scenes, warns


# ------------------------------------------------------------------------------------ --check
def run_check(req: dict) -> int:
    p = _params(req)
    from shapely.geometry import shape
    rep = {"plugin": "wind_turbines", "imports": {}, "problems": []}
    w = req.get("weights") or {}
    hf = _resolve(w.get("hf_home"))
    if hf:
        os.environ["HF_HOME"] = hf
    os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for mod in ("numpy", "scipy", "sklearn", "rasterio", "geopandas", "shapely", "fiona", "torch",
                "transformers", "terratorch"):
        try:
            m = __import__(mod)
            rep["imports"][mod] = getattr(m, "__version__", "ok")
        except Exception as e:   # noqa: BLE001
            rep["imports"][mod] = f"FAIL {e}"; rep["problems"].append(f"import {mod}: {e}")
    try:
        sys.path.insert(0, str(HERE.parent / "_shared"))
        import custom_modules  # noqa: F401
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
        ok = "dinov3_hf_vitl16" in TERRATORCH_BACKBONE_REGISTRY
        rep["imports"]["custom_modules"] = "ok" if ok else "imported, backbone not registered"
        if not ok:
            rep["problems"].append("custom_modules: dinov3_hf_vitl16 not in the terratorch registry")
    except Exception as e:   # noqa: BLE001
        rep["imports"]["custom_modules"] = f"FAIL {e}"; rep["problems"].append(f"import custom_modules: {e}")
    for mod in ("turbine_mask", "turbine_post", "turbine_blobs", "turbine_detect", "turbine_harness"):
        try:
            __import__(mod); rep["imports"][mod] = "ok"
        except Exception as e:   # noqa: BLE001
            rep["imports"][mod] = f"FAIL {e}"; rep["problems"].append(f"import {mod}: {e}")

    ck = _resolve(w.get("checkpoint"))
    rep["checkpoint"] = {"path": ck, "exists": bool(ck and os.path.isfile(ck)),
                         "size_mb": round(os.path.getsize(ck) / 2**20) if ck and os.path.isfile(ck) else None}
    if not rep["checkpoint"]["exists"]:
        rep["problems"].append(f"checkpoint missing: {ck}")
    hfr = {"path": hf, "repo_dir": None, "config.json": False, "model.safetensors": False}
    if hf:
        rd = Path(hf) / HF_REPO_DIR
        hfr["repo_dir"] = str(rd)
        for name in ("config.json", "model.safetensors"):
            hfr[name] = any(f.exists() for f in rd.glob(f"snapshots/*/{name}"))   # follows symlinks
    rep["hf_home"] = hfr
    if not (hfr["config.json"] and hfr["model.safetensors"]):
        rep["problems"].append(f"HF cache incomplete for facebook/dinov3-vitl16-pretrain-lvd1689m under {hf}")

    osm = (req.get("inputs") or {}).get("osm_wind_gpkg")
    rep["osm_wind_gpkg"] = {"path": osm, "exists": bool(osm and os.path.exists(osm))}
    if not rep["osm_wind_gpkg"]["exists"]:
        rep["osm_wind_gpkg"]["note"] = "absent: every detection would be unconfirmed (0.4)"

    aoi = shape(req["aoi"])
    scenes, warns = survey_scenes(req, aoi, p, deep=True)
    rep["scenes"] = []
    for s in scenes:
        r = {k: s.get(k) for k in ("scene_id", "path", "opens", "error", "gsd", "gsd_ok", "intersects_aoi",
                                   "blocks_past_eof", "date", "use", "skip")}
        r["crs"] = str(s["crs"]) if s.get("crs") else None
        if s.get("opens"):
            r["shape"] = [s["height"], s["width"]]
            r["existing_mask"] = existing_mask(req, p, s)
        rep["scenes"].append(r)
        if s.get("intersects_aoi") and s.get("blocks_past_eof"):
            rep["problems"].append(f"{s['scene_id']}: {s['skip']}")
    rep["warnings"] = warns
    usable = [s for s in scenes if s.get("use")]
    if not usable:
        rep["problems"].append("no PAN scene at 0.45 m intersects the AOI")
    rep["scenes_usable"] = [s["scene_id"] for s in usable]
    rep["ok"] = not rep["problems"]
    txt = json.dumps(rep, indent=1, default=str)
    print(txt, flush=True)
    Path(req["work_dir"], "check.json").write_text(txt)
    return 0 if rep["ok"] else 1


# ------------------------------------------------------------------------------------ run
def _empty_features():
    import geopandas as gpd
    import pandas as pd
    cols = {"class": pd.Series(dtype=str), "confidence": pd.Series(dtype=float), "name": pd.Series(dtype=str),
            "src_id": pd.Series(dtype=str), "scene_id": pd.Series(dtype=str), "area": pd.Series(dtype="int64"),
            "n_frag": pd.Series(dtype="int64"), "n_views": pd.Series(dtype="int64"),
            "osm_confirmed": pd.Series(dtype=bool), "osm_id": pd.Series(dtype=str), "review": pd.Series(dtype=bool)}
    return gpd.GeoDataFrame(cols, geometry=gpd.GeoSeries([], crs=4326), crs=4326)


def write_features(gdf, path: str):
    if os.path.exists(path):
        os.remove(path)
    gdf = gdf.copy()
    gdf["name"] = gdf["name"].astype(object)
    gdf.to_file(path, layer="features", driver="GPKG")


def postprocess(per_scene, aoi, p: dict, osm_path: str | None, warns: list):
    """per_scene: list of (scene_id, GeoDataFrame in any CRS with area, n_frag).
    Returns the final features GeoDataFrame in EPSG:4326 and the UTM EPSG used."""
    import geopandas as gpd
    import pandas as pd
    import turbine_post as TP
    epsg = TP.aoi_utm(aoi)
    parts = []
    for sid, det in per_scene:
        if det is None or det.empty:
            continue
        d = det.reset_index(drop=True).to_crs(epsg)
        d["scene_id"] = str(sid)
        d["src_id"] = [f"{sid}_{k}" for k in range(len(d))]
        parts.append(d[["scene_id", "src_id", "area", "n_frag", "geometry"]])
    if not parts:
        return _empty_features(), epsg
    det = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=epsg)
    det = TP.cross_scene_nms(det, float(p["nms_m"]))
    aoi_m = gpd.GeoSeries([aoi], crs=4326).to_crs(epsg).iloc[0]
    det = det[det.geometry.intersects(aoi_m)].reset_index(drop=True)
    if det.empty:
        return _empty_features(), epsg
    osm = TP.load_osm_wind(osm_path)
    if osm is None:
        warns.append("inputs.osm_wind_gpkg missing: no detection is OSM-confirmed (all 0.4, flagged for review)")
    conf, oid = TP.osm_confirm(det, osm, float(p["osm_match_m"]))
    det["osm_confirmed"] = conf
    det["osm_id"] = oid
    det["confidence"] = [float(p["conf_confirmed"]) if c else float(p["conf_unconfirmed"]) for c in conf]
    det["review"] = [not c for c in conf]
    det["class"] = "wind_turbine"
    det["name"] = None
    det["area"] = det["area"].astype("int64"); det["n_frag"] = det["n_frag"].astype("int64")
    det["n_views"] = det["n_views"].astype("int64")
    return det[OUT_COLUMNS + ["geometry"]].to_crs(4326), epsg


def run(req: dict) -> int:
    import geopandas as gpd
    from shapely.geometry import mapping, shape
    from shapely.ops import unary_union
    import turbine_post as TP
    p = _params(req)
    wd = Path(req["work_dir"])
    aoi = shape(req["aoi"])
    if not aoi.is_valid:
        aoi = aoi.buffer(0)
    common.progress(0.0, "surveying Cartosat-3 PAN scenes")
    scenes, warns = survey_scenes(req, aoi, p)
    todo = [s for s in scenes if s.get("use")]
    common.log(f"{len(todo)} of {len(scenes)} PAN scenes intersect the AOI at {p['gsd_m']} m: "
               f"{[s['scene_id'] for s in todo]}")

    w = req.get("weights") or {}
    model = None
    per_scene, done, dates, stats = [], [], set(), {}
    n = max(1, len(todo))
    for k, s in enumerate(todo):
        sid = s["scene_id"]; base = k / n; share = 1.0 / n
        mask = existing_mask(req, p, s)
        if mask is None and model is None:     # failure to load the model is fatal for the run
            import turbine_mask as TM
            hf = _resolve(w.get("hf_home"))
            if hf:
                os.environ["HF_HOME"] = hf
            os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"
            vram = common.gpu_env(req)
            common.progress(base, "loading turbine model")
            model = TM.load_model(_resolve(w.get("checkpoint")), vram)
        try:
            if mask is None:
                import turbine_mask as TM
                (wd / "masks").mkdir(exist_ok=True)
                mask = str(wd / "masks" / f"{sid}_round4_mask.tif")
                common.progress(base, f"{sid}: mask")
                TM.build_mask(model, s["path"], mask, gsd_m=float(p["gsd_m"]), gsd_tol=float(p["gsd_tol"]),
                              on_progress=lambda f: common.progress(base + share * 0.85 * f, f"{sid}: mask"),
                              log=common.log)
                e0 = base + share * 0.85
            else:
                common.log(f"{sid}: reusing mask {mask}")
                e0 = base
            common.progress(e0, f"{sid}: Exp-G")
            (wd / "blob_cache").mkdir(exist_ok=True)
            det, st = TP.expg_scene(mask, sid, int(p["thresh"]), int(p["min_px"]),
                                    cache_path=str(wd / "blob_cache" / f"blobs_t{int(p['thresh'])}_{sid}.json"),
                                    on_strip=lambda f: common.progress(e0 + (base + share - e0) * f, f"{sid}: Exp-G"))
        except Exception as e:   # noqa: BLE001  (unreadable / truncated raster, OOM, ...): skip the scene
            warns.append(f"{sid}: failed ({type(e).__name__}: {e}); scene skipped")
            common.log(traceback.format_exc())
            continue
        det.to_file(wd / f"{sid}_expG_detections.gpkg", driver="GPKG")
        common.log(f"{sid}: {st}")
        stats[sid] = st
        per_scene.append((sid, det)); done.append(s)
        if s.get("date"):
            dates.add(s["date"])
        else:
            warns.append(f"{sid}: no acquisition date in catalog or BAND_META")

    if todo and not done:
        raise RuntimeError("every intersecting scene failed: " + "; ".join(warns))
    common.progress(0.97, "cross-scene NMS, AOI clip, OSM confirmation")
    feats, epsg = postprocess(per_scene, aoi, p, (req.get("inputs") or {}).get("osm_wind_gpkg"), warns)
    write_features(feats, req["out_gpkg"])
    cov = None
    if done:
        c = aoi.intersection(unary_union([s["footprint"] for s in done]))
        cov = mapping(c) if not c.is_empty else None
    if not todo:
        warns.append("no Cartosat-3 PAN 0.45 m scene intersects the AOI: empty output")
    n_conf = int(feats["osm_confirmed"].sum()) if len(feats) else 0
    common.write_result(
        req, coverage=cov, image_source=IMAGE_SOURCE, image_dates=sorted(dates), model=MODEL,
        model_version=MODEL_VERSION,
        counts={"features": int(len(feats)), "osm_confirmed": n_conf, "unconfirmed": int(len(feats)) - n_conf,
                "scenes_used": len(done), "scenes_skipped": len(scenes) - len(done),
                "per_scene_detections": {k: v["detections"] for k, v in stats.items()}},
        warnings=warns, scenes=[s["scene_id"] for s in done], utm_epsg=epsg,
        artifacts={})
    common.progress(1.0, f"{len(feats)} turbines ({n_conf} OSM-confirmed)")
    return 0


def main() -> int:
    req = common.load_request()
    if "--check" in sys.argv:
        return run_check(req)
    return run(req)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:   # noqa: BLE001
        traceback.print_exc()
        print(f"wind_turbines failed: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
