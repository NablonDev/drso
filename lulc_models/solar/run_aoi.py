#!/usr/bin/env python
"""SOLAR PARKS plugin (lulc_models/CONTRACT.md).

    <terratorch env>/bin/python lulc_models/solar/run_aoi.py --request <work_dir>/request.json
    <terratorch env>/bin/python lulc_models/solar/run_aoi.py --check --request request.json [--check-gpu]
    ... run_aoi.py --request request.json --prob corridor_prob.tif   # dev/test: skip inference

Pipeline (research: /mnt/drso-data/solar_lora/scripts 07, 08b, 09, 14, 16):
  1. DINOv3-L/16 LoRA segmentation over each Esri Wayback z18 MBTiles, AOI windows only -> prob tif
  2. prob >= threshold -> polygonise -> Web-Mercator area >= min_area_m2
  3. EPSG:7755 area >= min_area_m2; known = intersects OSM solar (inputs.osm_solar_gpkg)
  4. Sentinel-2 seasonal features from local band files (imagery.sentinel2_l2a) + 20 m gate
  5. new detections kept when nir_med is inside the FROZEN params.nir_band
Output `features`: confirmed_OSM 1.0, new_verified_NIR 0.8, new_unverified 0.5 (no S2 evidence).
Stage layers SOLAR_confirmed / SOLAR_new_verified / SOLAR_rejected / SOLAR_excluded_below_20m /
SOLAR_new_unverified / SOLAR_FINAL_all are written to the same GeoPackage for audit.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # lulc_models/common.py
sys.path.insert(0, str(HERE))            # sibling modules
import common  # noqa: E402

MODEL = "solar-dinov3-lora"
MODEL_VERSION = "v1 epoch07"
DEFAULTS = {"threshold": 0.60, "min_area_m2": 5000.0, "nir_band": [1159, 2351], "min_obs": 4,
            "min_pure_px": 4, "erode_m": 20.0, "max_cloud": 20.0, "block": 3072, "tile": 512,
            "overlap": 128, "batch": 24, "osm_layer": None, "gate_known": True, "allow_cpu": False}
HF_REPO_DIR = "hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m"
STAGE_LAYERS = {"confirmed_OSM": "SOLAR_confirmed", "new_verified_NIR": "SOLAR_new_verified",
                "rejected_NIR": "SOLAR_rejected", "excluded_below_20m": "SOLAR_excluded_below_20m",
                "new_unverified": "SOLAR_new_unverified"}
STAGE_COLS = ["src_id", "cls", "known", "area_m2", "merc_area_m2", "eroded_m2", "resolvable_20m", "n_obs",
              "nir_med", "ndvi_med", "ndvi_std", "swir_ratio_med", "geometry"]


def _params(req):
    p = dict(DEFAULTS)
    p.update(req.get("params") or {})
    return p


def _aoi(req):
    from shapely.geometry import shape
    from shapely.validation import make_valid
    return make_valid(shape(req["aoi"]))


def _to3857(geom):
    import geopandas as gpd
    return gpd.GeoSeries([geom], crs=4326).to_crs(3857).iloc[0]


def _to4326(geom):
    import geopandas as gpd
    return gpd.GeoSeries([geom], crs=3857).to_crs(4326).iloc[0]


def _empty_features():
    import geopandas as gpd
    import pandas as pd
    cols = {"class": pd.Series(dtype=str), "confidence": pd.Series(dtype=float),
            "name": pd.Series(dtype=str), "src_id": pd.Series(dtype=str), "cls": pd.Series(dtype=str),
            "area_m2": pd.Series(dtype=float), "n_obs": pd.Series(dtype=int),
            "nir_med": pd.Series(dtype=float), "ndvi_med": pd.Series(dtype=float)}
    return gpd.GeoDataFrame(cols, geometry=gpd.GeoSeries([], crs=4326), crs=4326)


def _write_gpkg(path, layer, gdf):
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio", promote_to_multi=True,
                geometry_type="MultiPolygon")


# ---- main run ------------------------------------------------------------------------------
def run(req, prob_override: str | None = None):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import mapping
    from shapely.ops import unary_union

    import solar_post as pp
    import solar_s2 as s2f

    P = _params(req)
    warnings: list[str] = []
    work = Path(req["work_dir"])
    out_gpkg = Path(req["out_gpkg"])
    img = req.get("imagery") or {}
    wb_items = list(img.get("esri_wayback_z18") or [])
    s2_items = list(img.get("sentinel2_l2a") or [])
    aoi = _aoi(req)
    aoi3857 = _to3857(aoi)

    if out_gpkg.exists():
        out_gpkg.unlink()
    common.progress(0.0, "solar: start")

    # ---- coverage per MBTiles (newest first, later items only fill what is still uncovered)
    wb_items.sort(key=lambda it: str(it.get("date", "")), reverse=True)
    plan, covered = [], None
    for it in wb_items:
        p = it.get("path")
        if not p or not Path(p).exists():
            warnings.append(f"Wayback MBTiles missing on disk: {p}")
            continue
        fp = pp.mbtiles_footprint_3857(p, aoi3857.bounds)
        if fp is None:
            continue
        part = fp.intersection(aoi3857)
        if covered is not None:
            part = part.difference(covered)
        if part.is_empty or part.area < 1.0:
            continue
        plan.append((it, part))
        covered = part if covered is None else unary_union([covered, part])

    if not plan:
        warnings.append("no Esri Wayback z18 imagery over the AOI: empty output")
        _write_gpkg(out_gpkg, "features", _empty_features())
        common.write_result(req, coverage={"type": "GeometryCollection", "geometries": []},
                            image_source=None, image_dates=[], model=MODEL, model_version=MODEL_VERSION,
                            counts={"features": 0}, warnings=warnings, artifacts={"prob_raster": []})
        common.progress(1.0, "solar: no imagery")
        return 0

    # ---- 1. inference (or, dev/test only, a supplied probability raster)
    import solar_infer as infer
    model = device = None
    if prob_override:
        warnings.append(f"inference skipped: --prob {prob_override} supplied (dev/test mode)")
    else:
        vram = common.gpu_env(req)
        w = req.get("weights") or {}
        ckpt, hf_home = w.get("checkpoint"), w.get("hf_home")
        if not ckpt or not Path(ckpt).exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        infer.set_offline_env(hf_home)
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            common.torch_vram_cap(vram)
        elif P["allow_cpu"]:
            device = "cpu"
            warnings.append("CUDA unavailable: inference ran on CPU (params.allow_cpu)")
        else:
            raise RuntimeError("CUDA unavailable (set params.allow_cpu to run on CPU)")
        common.progress(0.01, "solar: loading model")
        model = infer.load_model(ckpt, hf_home, device)
        common.log(f"model loaded: {Path(ckpt).name} on {device}")

    probs, dets, dates, releases = [], [], [], []
    thr8 = int(round(float(P["threshold"]) * 255))
    n_plan = len(plan)
    for k, (it, part) in enumerate(plan):
        tif = work / f"solar_prob_{k}_{it.get('release', 'r')}.tif"
        lo, hi = 0.02 + 0.78 * k / n_plan, 0.02 + 0.78 * (k + 1) / n_plan
        if prob_override:
            st = infer.crop_prob(prob_override, part, str(tif))
        else:
            st = infer.run(it["path"], part, str(tif), model, device=device, block=int(P["block"]),
                           tile=int(P["tile"]), overlap=int(P["overlap"]), batch_size=int(P["batch"]),
                           progress=lambda f, m: common.progress(lo + (hi - lo) * f, m), log=common.log)
        if not st:
            continue
        common.log(f"inference {Path(it['path']).name}: {st}")
        probs.append(str(tif))
        dates.append(str(it.get("date")))
        releases.append(str(it.get("release")))
        # ---- 2. polygonise + merc area filter (08b)
        common.progress(hi, "solar: polygonising")
        d = pp.polygonize_prob(str(tif), thr8, float(P["min_area_m2"]))
        d["release"] = str(it.get("release"))
        dets.append(d)
    del model
    det = gpd.GeoDataFrame(pd.concat(dets, ignore_index=True), crs=3857) if dets else \
        gpd.GeoDataFrame({"merc_area_m2": []}, geometry=gpd.GeoSeries([], crs=3857), crs=3857)
    n_raw = len(det)

    # ---- 3. OSM classification (09)
    common.progress(0.82, "solar: OSM classification")
    osm_path = (req.get("inputs") or {}).get("osm_solar_gpkg")
    osm = None
    if osm_path:
        if not Path(osm_path).exists():
            raise FileNotFoundError(f"inputs.osm_solar_gpkg not found: {osm_path}")
        osm = pp.read_osm_solar(osm_path, P.get("osm_layer"))
    else:
        warnings.append("no OSM solar input: every detection is treated as new")
    g = pp.classify_osm(det, osm, float(P["min_area_m2"]))
    # deterministic src_id: north-to-south, west-to-east
    if len(g):
        c = g.geometry.centroid
        g = g.assign(_y=-c.y, _x=c.x).sort_values(["_y", "_x"]).drop(columns=["_y", "_x"]).reset_index(drop=True)
    g["src_id"] = [f"solar_{i:05d}" for i in range(len(g))]

    # ---- 4. Sentinel-2 seasonal features (14)
    s2_dates = []
    ok, er = s2f.resolvable(g, int(P["min_pure_px"]), float(P["erode_m"])) if len(g) else ([], [])
    g["resolvable_20m"] = list(ok)
    g["eroded_m2"] = list(er)
    s2_available = bool(s2_items)
    if s2_available and len(g):
        common.progress(0.86, "solar: Sentinel-2 features")
        g, used, w2 = s2f.seasonal_features(g, s2_items, float(P["max_cloud"]),
                                            progress=lambda f, m: common.progress(0.86 + 0.1 * f, m))
        warnings.extend(w2)
        s2_dates = sorted({str(s.get("date"))[:10] for s in used})
        if not used:
            warnings.append(f"no Sentinel-2 scene below {P['max_cloud']}% cloud with all four bands: "
                            "new detections left unverified")
            s2_available = False
    else:
        for col in ("n_obs", "nir_med", "ndvi_med", "ndvi_std", "swir_ratio_med"):
            g[col] = 0 if col == "n_obs" else float("nan")
        if not s2_items:
            warnings.append("no Sentinel-2 L2A supplied: OSM-confirmed kept, new detections unverified "
                            "(confidence 0.5)")

    # ---- 5. frozen NIR band (16)
    g["cls"] = pp.final_classes(g, s2_available, P["nir_band"], int(P["min_obs"]), bool(P["gate_known"])) \
        if len(g) else pd.Series([], dtype=object)
    if s2_available:
        n_noobs = int(((g["cls"] == pp.CLS_UNVERIFIED)).sum())
        if n_noobs:
            warnings.append(f"{n_noobs} new detection(s) had fewer than {P['min_obs']} Sentinel-2 "
                            "observations: kept unverified (confidence 0.5)")
        n_kx = int(((g["cls"] == pp.CLS_EXCLUDED) & g["known"]).sum())
        if n_kx:
            warnings.append(f"{n_kx} OSM-confirmed detection(s) too narrow for the 20 m gate were dropped "
                            "(params.gate_known)")

    # ---- outputs
    common.progress(0.97, "solar: writing output")
    feats = pp.to_features(g, aoi) if len(g) else _empty_features()
    _write_gpkg(out_gpkg, "features", feats if len(feats) else _empty_features())
    g4 = g.to_crs(4326) if len(g) else None
    for cls, lay in STAGE_LAYERS.items():
        if g4 is not None:
            sub = g4[g4["cls"] == cls]
            if len(sub):
                _write_gpkg(out_gpkg, lay, sub[[c for c in STAGE_COLS if c in sub.columns]])
    if g4 is not None and len(feats):
        fin = g4[g4["cls"].isin(pp.KEEP)]
        _write_gpkg(out_gpkg, "SOLAR_FINAL_all", fin[[c for c in STAGE_COLS if c in fin.columns]])

    counts = {"features": int(len(feats)), "detections_raw": n_raw, "detections": int(len(g))}
    for cls in STAGE_LAYERS:
        counts[cls] = int((g["cls"] == cls).sum()) if len(g) else 0
    cov4326 = _to4326(covered)
    rel = "/".join(dict.fromkeys(releases)) or "?"
    src = f"Esri Wayback z18 release {rel}" + (" + Sentinel-2 L2A" if s2_dates else "")
    common.write_result(req, coverage=mapping(cov4326), image_source=src,
                        image_dates=sorted(set(dates) | set(s2_dates)), model=MODEL,
                        model_version=MODEL_VERSION, counts=counts, warnings=warnings,
                        artifacts={"prob_raster": probs},
                        params={k: P[k] for k in ("threshold", "min_area_m2", "nir_band", "min_obs",
                                                  "min_pure_px", "erode_m", "max_cloud")})
    common.progress(1.0, f"solar: {len(feats)} features")
    return 0


# ---- --check -------------------------------------------------------------------------------
def check(req, check_gpu=False) -> dict:
    """Environment / input readiness without loading weights or touching the GPU."""
    rep = {"ok": True, "checks": {}}

    def put(name, ok, **info):
        rep["checks"][name] = {"ok": bool(ok), **info}
        if not ok:
            rep["ok"] = False

    w = req.get("weights") or {}
    import solar_infer as infer
    infer.set_offline_env(w.get("hf_home"))
    for mod in ("numpy", "rasterio", "geopandas", "shapely", "scipy", "pyogrio", "torch", "lightning",
                "transformers", "peft", "terratorch"):
        try:
            m = __import__(mod)
            put(f"import:{mod}", True, version=getattr(m, "__version__", None))
        except Exception as e:  # noqa: BLE001
            put(f"import:{mod}", False, error=f"{type(e).__name__}: {e}")
    try:
        infer.import_custom_modules()
        from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
        put("custom_modules", "dinov3_hf_vitl16" in TERRATORCH_BACKBONE_REGISTRY,
            path=str(infer.SHARED / "custom_modules"))
    except Exception as e:  # noqa: BLE001
        put("custom_modules", False, error=f"{type(e).__name__}: {e}")
    try:
        import rasterio
        with rasterio.Env() as env:
            put("gdal:MBTiles_driver", "MBTiles" in env.drivers())
    except Exception as e:  # noqa: BLE001
        put("gdal:MBTiles_driver", False, error=str(e))

    ck = w.get("checkpoint")
    put("checkpoint", bool(ck) and Path(ck).is_file(), path=ck,
        size_mb=round(Path(ck).stat().st_size / 2**20, 1) if ck and Path(ck).is_file() else None)
    hf = w.get("hf_home")
    snaps = sorted((Path(hf) / HF_REPO_DIR / "snapshots").glob("*")) if hf else []
    good = [s for s in snaps if (s / "config.json").exists() and
            any((s / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))]
    put("hf_cache", bool(good), hf_home=hf, snapshot=str(good[0]) if good else None)

    if check_gpu:
        try:
            import torch
            put("gpu", torch.cuda.is_available(), device_count=torch.cuda.device_count())
        except Exception as e:  # noqa: BLE001
            put("gpu", False, error=str(e))

    import solar_post as pp
    import rasterio
    from shapely.geometry import shape
    aoi = _aoi(req)
    aoi3857 = _to3857(aoi)
    img = req.get("imagery") or {}
    tot = 0.0
    for i, it in enumerate(img.get("esri_wayback_z18") or []):
        p = it.get("path")
        try:
            with rasterio.open(p) as ds:
                info = {"driver": ds.driver, "bands": ds.count, "crs": str(ds.crs), "size": [ds.width, ds.height],
                        "res_m": round(ds.res[0], 4)}
            fp = pp.mbtiles_footprint_3857(p, aoi3857.bounds)
            inter = fp.intersection(aoi3857) if fp is not None else None
            km2 = round(inter.area / 1e6, 3) if inter is not None else 0.0
            # Web-Mercator area; scale by cos^2(lat) for ground area
            import math
            lat = aoi.centroid.y
            km2_ground = round(km2 * math.cos(math.radians(lat)) ** 2, 3)
            tot += km2_ground
            put(f"mbtiles[{i}]", info["driver"] == "MBTiles" and km2 > 0, path=p, release=it.get("release"),
                date=it.get("date"), aoi_overlap_km2=km2_ground, **info)
        except Exception as e:  # noqa: BLE001
            put(f"mbtiles[{i}]", False, path=p, error=f"{type(e).__name__}: {e}")
    if not (img.get("esri_wayback_z18") or []):
        put("mbtiles", False, error="no esri_wayback_z18 item in request")
    rep["aoi_overlap_km2"] = round(tot, 3)

    bad = []
    s2 = img.get("sentinel2_l2a") or []
    for it in s2:
        for b in ("B04", "B08", "B11", "B12"):
            p = (it.get("bands") or {}).get(b)
            try:
                with rasterio.open(p) as ds:
                    _ = ds.crs, ds.transform
            except Exception as e:  # noqa: BLE001
                bad.append(f"{it.get('scene_id')}:{b}: {type(e).__name__}: {e}")
        fpg = it.get("footprint")
        if fpg and not shape(fpg).intersects(aoi):
            bad.append(f"{it.get('scene_id')}: footprint does not intersect the AOI")
    # S2 is optional (no S2 -> unverified), so its absence is info, not failure
    put("sentinel2_l2a", not bad, scenes=len(s2), errors=bad[:20], optional=True)
    osm = (req.get("inputs") or {}).get("osm_solar_gpkg")
    if osm:
        try:
            n = len(pp.read_osm_solar(osm, (req.get("params") or {}).get("osm_layer")))
            put("osm_solar_gpkg", True, path=osm, features=n)
        except Exception as e:  # noqa: BLE001
            put("osm_solar_gpkg", False, path=osm, error=f"{type(e).__name__}: {e}")
    return rep


def main():
    if "--check" in sys.argv:
        req = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text())
        rep = check(req, "--check-gpu" in sys.argv)
        print(json.dumps(rep, indent=1, default=str), flush=True)
        if req.get("work_dir") and Path(req["work_dir"]).is_dir():
            (Path(req["work_dir"]) / "check.json").write_text(json.dumps(rep, indent=1, default=str))
        return 0 if rep["ok"] else 1
    req = common.load_request()
    prob = sys.argv[sys.argv.index("--prob") + 1] if "--prob" in sys.argv else None
    try:
        return run(req, prob)
    except SystemExit as e:
        print(f"solar: {e}", file=sys.stderr, flush=True)
        return 143 if str(e) == "cancelled" else 1
    except Exception as e:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        print(f"solar failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
