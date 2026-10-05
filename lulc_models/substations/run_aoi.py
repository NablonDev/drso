#!/usr/bin/env python3
"""SUBSTATIONS plugin: Stage B DINOv3-LoRA segmentation + SAR gates (contract: ../CONTRACT.md).

    run_aoi.py --request request.json            # full run (GPU)
    run_aoi.py --request request.json --check    # readiness report, no weights, no GPU
    run_aoi.py --request request.json --prob a.tif [--prob b.tif]
                                                 # dev/test: skip inference, post-process the
                                                 # given uint8 prob rasters (EPSG:3857)

Pipeline (production chain of /mnt/cog-data/vivek/stageA, see sub_infer.py / sub_post.py):
  AOI ∩ MBTiles tile footprint -> inference on the blocks meeting it -> polygonise (p >= threshold,
  >= min_area_ha) -> clip to AOI -> OSM match -> minwidth attribute -> Sentinel-1 VV dB per polygon
  -> NISAR coherence where registered -> gates / confidence -> out.gpkg (`features`, `rejected`).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LULC = HERE.parent
REPO = LULC.parent
for p in (str(HERE), str(LULC / "_shared"), str(LULC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402

MODEL = "substation-dinov3-lora-stageB"
MODEL_VERSION = "epoch22"
CLASS = "substation"
DEFAULTS = {"threshold": 0.50, "min_area_ha": 0.05, "vv_db_min": -10.10, "coh_min": 0.337,
            "s1_scenes": 8, "block": 3072, "batch": 24, "equal_area_crs": "EPSG:7755"}
COLS = ["class", "confidence", "name", "src_id", "source", "area_ha", "vv_db", "vv_std",
        "s1_n", "coh", "minwidth", "gates", "review", "osm_id", "image_release"]


def _arg_list(flag):
    out, a = [], sys.argv
    for i, v in enumerate(a[:-1]):
        if v == flag:
            out.append(a[i + 1])
    return out


def _resolve(p):
    """Absolute paths pass through; relative ones are tried against cwd, then the repo root."""
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


def _empty_frame(extra=()):
    import geopandas as gpd
    import pandas as pd
    cols = COLS + list(extra)
    return gpd.GeoDataFrame({c: pd.Series(dtype="object") for c in cols},
                            geometry=gpd.GeoSeries([], crs=4326), crs=4326)


def _write_layers(path, feats, rej):
    import geopandas as gpd
    if Path(path).exists():
        Path(path).unlink()
    for name, g in (("features", feats), ("rejected", rej)):
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
        g.to_file(path, layer=name, driver="GPKG", engine="pyogrio", geometry_type="Polygon")


def main():
    req = common.load_request()          # installs the SIGTERM -> SystemExit handler
    for p in ("work_dir", "out_gpkg", "result_json"):
        req[p] = _resolve(req[p])
    params = dict(DEFAULTS)
    params.update(req.get("params") or {})
    aoi = _aoi_geom(req["aoi"])

    if "--check" in sys.argv:
        import sub_check
        rep = sub_check.run_check(req, _resolve, aoi)
        txt = json.dumps(rep, indent=1, default=str)
        Path(req["work_dir"], "check.json").write_text(txt)
        print(txt, flush=True)
        sys.exit(0 if rep["ok"] else 1)

    common.progress(0.0, "start")
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import GeometryCollection, mapping
    from shapely.ops import unary_union
    import sub_mbtiles
    import sub_post

    warnings = []
    img = req.get("imagery") or {}
    work = Path(req["work_dir"])
    aoi3857 = gpd.GeoSeries([aoi], crs=4326).to_crs(3857).iloc[0]

    # ---- coverage: AOI ∩ tile footprint, newest release first, no double coverage ----------
    prob_override = [_resolve(p) for p in _arg_list("--prob")]
    wb = sorted(img.get("esri_wayback_z18") or [], key=lambda e: str(e.get("date") or ""),
                reverse=True)
    regions, covered = [], None
    for e in wb:
        path = _resolve(e.get("path"))
        if not path or not Path(path).exists():
            warnings.append("Wayback MBTiles missing: %s" % path)
            continue
        fp, n = sub_mbtiles.footprint_3857(path, aoi3857.bounds)
        reg = fp.intersection(aoi3857) if n else None
        if reg is not None and covered is not None:
            reg = reg.difference(covered)
        if reg is None or reg.is_empty or reg.area < 1.0:
            continue
        regions.append((e, path, reg))
        covered = reg if covered is None else covered.union(reg)
        common.log("release %s (%s): %d tiles, %.2f km2 of AOI"
                   % (e.get("release"), e.get("date"), n, reg.area / 1e6))
    if len(regions) > 1:
        warnings.append("AOI spans %d Wayback releases; seams between them are not merged"
                        % len(regions))
    if prob_override and len(prob_override) != len(regions):
        raise SystemExit("--prob given %d rasters for %d imagery regions"
                         % (len(prob_override), len(regions)))

    if not regions:
        warnings.append("no Esri Wayback z18 imagery for the AOI")
        _write_layers(req["out_gpkg"], _empty_frame(), _empty_frame(["reject_reason"]))
        common.write_result(req, coverage=mapping(GeometryCollection()), image_source=None,
                            image_dates=[], model=MODEL, model_version=MODEL_VERSION,
                            counts={"features": 0, "rejected": 0}, warnings=warnings)
        common.progress(1.0, "no imagery")
        return

    # ---- inference -----------------------------------------------------------------------
    probs = []
    if prob_override:
        probs = prob_override
        warnings.append("inference skipped: --prob rasters supplied (dev/test mode)")
    else:
        w = req.get("weights") or {}
        ckpt, hf = _resolve(w.get("checkpoint")), _resolve(w.get("hf_home"))
        if not ckpt or not Path(ckpt).is_file():
            raise SystemExit("checkpoint not found: %s" % ckpt)
        if hf:
            os.environ["HF_HOME"] = hf
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        vram = common.gpu_env(req)
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA not available; the substation model needs a GPU")
        common.torch_vram_cap(vram)
        import sub_infer
        common.progress(0.02, "loading model")
        model = sub_infer.load_model(ckpt, "cuda")
        tot = sum(r.area for _, _, r in regions)
        done = 0.0
        for e, path, reg in regions:
            out = str(work / ("prob_%s.tif" % (e.get("release") or Path(path).stem)))
            base, span = done, reg.area / tot

            def prog(f, msg, base=base, span=span):
                common.progress(0.05 + 0.75 * (base + span * f), msg)
            st = sub_infer.run(path, reg, model, out, block=int(params["block"]),
                               batch=int(params["batch"]), device="cuda", progress=prog,
                               log=common.log)
            common.log("inference %s: %s" % (e.get("release"), st))
            probs.append(out)
            done += span
        del model
        torch.cuda.empty_cache()

    # ---- polygonise, clip ----------------------------------------------------------------
    common.progress(0.80, "polygonise")
    eq = params["equal_area_crs"]
    parts = []
    for (e, _, _), pr in zip(regions, probs):
        g = sub_post.polygonise(pr, float(params["threshold"]), float(params["min_area_ha"]), eq)
        if len(g):
            g = g.to_crs(4326)
            g["image_release"] = str(e.get("release"))
            parts.append(g)
    det = (gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326) if parts else
           gpd.GeoDataFrame({"image_release": []}, geometry=gpd.GeoSeries([], crs=4326)))
    n_raw = len(det)
    if len(det):
        det = gpd.clip(det, aoi).explode(index_parts=False).reset_index(drop=True)
        det = det[det.geometry.geom_type == "Polygon"].reset_index(drop=True)
        det["area_ha"] = det.to_crs(eq).area / 1e4
        det = det[det["area_ha"] >= float(params["min_area_ha"])].reset_index(drop=True)
    else:
        det["area_ha"] = pd.Series(dtype=float)
    common.log("detections: %d polygonised, %d after AOI clip" % (n_raw, len(det)))

    # ---- OSM, minwidth -------------------------------------------------------------------
    common.progress(0.83, "OSM match")
    osm_path = _resolve((req.get("inputs") or {}).get("osm_substations_gpkg"))
    osm = None
    if osm_path and Path(osm_path).exists():
        osm = sub_post.read_osm(osm_path)
    else:
        warnings.append("no OSM substations input: no detection can be OSM-confirmed")
    m, names, ids = sub_post.osm_match(det, osm)
    det["matches_osm"] = m
    det["name"] = names
    det["osm_id"] = ids
    det["source"] = np.where(m, "model_confirms_osm", "model_new")
    det["minwidth"] = [sub_post.minwidth(g) for g in det.to_crs(eq).geometry] if len(det) else []

    # ---- Sentinel-1 VV -------------------------------------------------------------------
    common.progress(0.86, "Sentinel-1 VV")
    scenes = sub_post.select_s1_scenes(
        [dict(e, path=_resolve(e.get("path"))) for e in img.get("sentinel1_rtc_vv") or []],
        int(params["s1_scenes"]), aoi)
    if not scenes:
        warnings.append("no Sentinel-1 RTC VV scenes: S1 gate not evaluated, all non-OSM "
                        "detections are low-confidence review items")
    vv, vstd, vn, used = sub_post.s1_vv(scenes, det, log=common.log) if len(det) else \
        (np.array([]), np.array([]), np.array([], int), scenes)
    det["vv_db"], det["vv_std"], det["s1_n"] = vv, vstd, vn

    # ---- NISAR coherence -----------------------------------------------------------------
    common.progress(0.92, "NISAR coherence")
    nis = [dict(e, path=_resolve(e.get("path"))) for e in img.get("nisar_coh") or []]
    coh, appl = sub_post.nisar_coh(nis, det, log=common.log) if len(det) else \
        (np.array([]), np.array([], bool))
    det["coh"], det["nisar_applicable"] = coh, appl

    # ---- gates, ids, outputs -------------------------------------------------------------
    common.progress(0.96, "gates")
    det = sub_post.apply_gates(det, float(params["vv_db_min"]), float(params["coh_min"]),
                               params.get("conf_gated"), params.get("conf_unchecked"))
    det["class"] = CLASS
    if len(det):
        c = det.geometry.representative_point()
        order = np.lexsort((c.x.values, -c.y.values))      # north to south, then west to east
        det = det.iloc[order].reset_index(drop=True)
    det["src_id"] = ["substation-%04d" % (i + 1) for i in range(len(det))]
    for col in ("vv_db", "vv_std", "coh", "minwidth", "area_ha"):
        det[col] = pd.to_numeric(det[col], errors="coerce").astype(float)
    det["s1_n"] = pd.to_numeric(det["s1_n"], errors="coerce").fillna(0).astype(int)
    det["review"] = det["review"].astype(bool)
    keep = det["keep"].astype(bool).values
    feats = det[keep][COLS + ["geometry"]].reset_index(drop=True) if len(det) else _empty_frame()
    rej = det[~keep][COLS + ["reject_reason", "geometry"]].reset_index(drop=True) \
        if len(det) else _empty_frame(["reject_reason"])
    _write_layers(req["out_gpkg"], feats, rej)

    nisar_used = bool(len(det) and np.isfinite(det["coh"].values).any())
    rel = sorted({str(e.get("release")) for e, _, _ in regions})
    src = "Esri Wayback z18 release %s" % ", ".join(rel)
    if used:
        src += " + Sentinel-1 RTC"
    if nisar_used:
        src += " + NISAR"
    cov = gpd.GeoSeries([unary_union([r for _, _, r in regions])], crs=3857).to_crs(4326).iloc[0]
    counts = {"features": int(keep.sum()), "rejected": int((~keep).sum()),
              "raw_detections": n_raw,
              "confirms_osm": int((feats["source"] == "model_confirms_osm").sum()),
              "new": int((feats["source"] == "model_new").sum()),
              "review": int(feats["review"].astype(bool).sum())}
    common.write_result(
        req, coverage=mapping(cov), image_source=src,
        image_dates=sorted({str(e.get("date")) for e, _, _ in regions if e.get("date")}),
        model=MODEL, model_version=MODEL_VERSION, counts=counts, warnings=warnings,
        artifacts={"prob_raster": probs},
        sar_dates=sorted({str(e.get("date")) for e in used if e.get("date")}),
        params=params,
        notes=["inference on the native EPSG:3857 z18 grid (0.597 m/px) as in production; "
               "the model was trained on UTM 0.60 m chips"])
    common.progress(1.0, "done: %d features (%d review), %d rejected"
                    % (counts["features"], counts["review"], counts["rejected"]))


if __name__ == "__main__":
    main()
