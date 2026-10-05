"""SUBSTATIONS plugin tests on small real windows of the Bhuj-Lakadia production run.
No network, no GPU (inference itself is not exercised; the end-to-end test feeds a crop of the
production probability raster through `run_aoi.py --prob`).

pytest lives in the beeline env, the plugin runs in the terratorch env:
    /home/vivek_jain/miniforge3/envs/beeline/bin/python -m pytest -q tests/plugins/test_substations.py
Pure-function tests import the plugin modules directly (numpy/rasterio/geopandas only); --check,
end-to-end and SIGTERM tests run run_aoi.py under SUB_TEST_PY (default terratorch python).
Tests that need the production files skip when they are absent (e.g. on the prod machine).
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import signal
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PLUG = REPO / "lulc_models" / "substations"
RUN = PLUG / "run_aoi.py"
PY = os.environ.get("SUB_TEST_PY", "/home/vivek_jain/miniforge3/envs/terratorch/bin/python")

STAGE = Path("/mnt/cog-data/vivek/stageA")
PROB = STAGE / "corridor_sub_prob_stageB.tif"
REVIEW = STAGE / "STAGEB_review.gpkg"
CASCADE = STAGE / "STAGEB_cascade_s1.gpkg"
MBT = Path("/mnt/drso-data/wayback/bhuj_lakadia_L18_2026-08-05.mbtiles")
NISAR = Path("/mnt/cog-data/nisar/out_coh/NISAR_coh_HH_min3.tif")
OSM = Path("/mnt/drso-data/solar_lora/data/osm_ref.gpkg")
CKPT = STAGE / "checkpoints/stageB/lightning_logs/version_0/checkpoints/epoch22.ckpt"
HF = Path.home() / ".cache/huggingface"
# 4096 x 4096 px window of the corridor prob raster (row, col) with 10 Stage B detections,
# 2 of them on OSM substations (Essar, Global Hitech)
WIN = (81920, 72704, 4096, 4096)

need_stage = pytest.mark.skipif(not (PROB.exists() and REVIEW.exists() and CASCADE.exists()),
                                reason="Stage B production files absent")
need_mbt = pytest.mark.skipif(not MBT.exists(), reason="Wayback MBTiles absent")
need_py = pytest.mark.skipif(not Path(PY).exists(), reason="terratorch env absent")


def _load(name):
    spec = importlib.util.spec_from_file_location("substations_" + name, PLUG / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


post = _load("sub_post")
mbt = _load("sub_mbtiles")
infer = _load("sub_infer")


# ------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def window(tmp_path_factory):
    """Crop of the production prob raster + its 4326 bounds."""
    if not PROB.exists():
        pytest.skip("prob raster absent")
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, bounds
    d = tmp_path_factory.mktemp("subwin")
    r, c, h, w = WIN
    with rasterio.open(PROB) as src:
        win = Window(c, r, w, h)
        prof = src.profile.copy()
        prof.update(width=w, height=h, transform=src.window_transform(win), BIGTIFF="NO")
        out = d / "prob_crop.tif"
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(src.read(1, window=win), 1)
        b3857 = bounds(win, src.transform)
        b4326 = transform_bounds(src.crs, "EPSG:4326", *b3857)
    return {"dir": d, "prob": out, "b3857": b3857, "b4326": b4326}


def _ref_detections():
    import geopandas as gpd
    import pandas as pd
    return gpd.GeoDataFrame(pd.concat([gpd.read_file(REVIEW, layer=l)
                                       for l in ("detected_TP", "detected_FP")],
                                      ignore_index=True), crs=4326)


def _iou(a, b):
    return a.intersection(b).area / a.union(b).area


# ------------------------------------------------------------------------- polygonise
@need_stage
def test_polygonise_matches_stageB_review(window):
    import geopandas as gpd
    from shapely.geometry import box
    det = post.polygonise(str(window["prob"]), 0.50, 0.05).to_crs(4326)
    ref = _ref_detections()
    x0, y0, x1, y1 = window["b4326"]
    inner = box(x0 + 0.001, y0 + 0.001, x1 - 0.001, y1 - 0.001)   # ~100 m off the crop edge
    ref_in = ref[ref.within(inner)]
    det_in = det[det.within(inner)]
    assert len(ref_in) == 10
    assert len(det_in) == len(ref_in)
    for _, r in ref_in.iterrows():
        cand = det_in[det_in.intersects(r.geometry)]
        assert len(cand) == 1
        q = cand.iloc[0]
        assert _iou(r.geometry, q.geometry) > 0.9999
        assert q["ha"] == pytest.approx(r["ha"], rel=1e-6)
    assert isinstance(det, gpd.GeoDataFrame)


@need_stage
def test_polygonise_empty_and_threshold(window):
    g = post.polygonise(str(window["prob"]), 1.01, 0.05)   # cut 258 > any uint8
    assert len(g) == 0 and "ha" in g.columns
    hi = post.polygonise(str(window["prob"]), 0.90, 0.05)
    lo = post.polygonise(str(window["prob"]), 0.50, 0.05)
    assert hi["ha"].sum() < lo["ha"].sum()


# ------------------------------------------------------------------------- gates
VV, COH = -10.10, 0.337
NAN = float("nan")


@pytest.mark.parametrize("vv,coh,appl,osm,keep,conf,review,gate_s1,gate_n", [
    # S1 present, NISAR covers
    (-8.0, 0.50, True, False, True, 0.75, False, "pass", "pass"),
    (-8.0, 0.50, True, True, True, 0.90, False, "pass", "pass"),
    (-12.0, 0.50, True, False, False, None, False, "fail", "pass"),
    (-8.0, 0.20, True, False, False, None, False, "pass", "fail"),
    (-8.0, 0.20, True, True, False, None, False, "pass", "fail"),   # OSM does not bypass a gate
    # S1 present, no NISAR raster for the polygon -> NISAR gate skipped, no penalty
    (-8.0, NAN, False, False, True, 0.75, False, "pass", "not_covered"),
    (-12.0, NAN, False, False, False, None, False, "fail", "not_covered"),
    (-8.0, 0.10, False, False, True, 0.75, False, "pass", "not_covered"),  # coh ignored
    # no S1 -> review
    (NAN, 0.50, True, False, True, 0.50, True, "na", "pass"),
    (None, NAN, False, False, True, 0.50, True, "na", "not_covered"),
    (NAN, NAN, False, True, True, 0.90, False, "na", "not_covered"),
    (NAN, 0.20, True, False, False, None, False, "na", "fail"),
    # NISAR registered but no valid pixel -> not evaluable -> review
    (-8.0, NAN, True, False, True, 0.50, True, "pass", "na"),
    (float("inf"), 0.50, True, False, True, 0.50, True, "na", "pass"),
    # boundaries are inclusive (sA11 used >=)
    (VV, COH, True, False, True, 0.75, False, "pass", "pass"),
    (VV - 1e-9, COH, True, False, False, None, False, "fail", "pass"),
])
def test_gate_table(vv, coh, appl, osm, keep, conf, review, gate_s1, gate_n):
    r = post.evaluate_gates(vv, coh, appl, osm, VV, COH)
    assert r["keep"] is keep
    assert r["confidence"] == conf
    assert r["review"] is review
    s1, n = r["gates"].split(";")
    assert s1.startswith("s1_vv:" + gate_s1)
    assert n.startswith("nisar_coh:" + gate_n)
    assert (r["reason"] is None) == keep


def test_apply_gates_frame_nan_handling():
    """The production .fillna(False) dropped detections without SAR; here they stay, flagged."""
    import pandas as pd
    d = pd.DataFrame({"vv_db": [np.nan, -5.0, -20.0, None],
                      "coh": [np.nan, np.nan, 0.6, 0.6],
                      "nisar_applicable": [False, True, True, True],
                      "matches_osm": [False, False, False, True]})
    g = post.apply_gates(d, VV, COH)
    assert g["keep"].tolist() == [True, True, False, True]
    assert g["confidence"].tolist()[:2] == [0.5, 0.5] and g["confidence"].iloc[3] == 0.9
    assert g["review"].tolist() == [True, True, False, False]
    assert "s1_vv" in g["reject_reason"].iloc[2]


@need_stage
def test_gates_reproduce_production_sar_cascade():
    """With full SAR, the kept set equals sA11's SAR pass set (before manual rejects)."""
    import geopandas as gpd
    d = gpd.read_file(CASCADE, layer="detections")
    d["nisar_applicable"] = True
    g = post.apply_gates(d, VV, COH)
    prod = ((d.coh >= COH) & (d.vv_db >= VV)).fillna(False)
    assert (g["keep"].values == prod.values).all()
    assert int(g["keep"].sum()) == 155
    assert int((g["confidence"] == 0.9).sum()) == 61


# ------------------------------------------------------------------------- S1
def _s1_raster(path, value_fn, crs="EPSG:32642", x0=600000.0, y0=2600000.0, n=40, res=10.0,
               nodata=None, dtype="float32"):
    import rasterio
    from rasterio.transform import from_origin
    a = np.fromfunction(value_fn, (n, n)).astype(dtype)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype=dtype, crs=crs,
                       transform=from_origin(x0, y0, res, res), nodata=nodata) as d:
        d.write(a, 1)
    return path


def _polys_utm(boxes):
    import geopandas as gpd
    from shapely.geometry import box
    return gpd.GeoDataFrame(geometry=[box(*b) for b in boxes], crs=32642).to_crs(4326)


def test_s1_db_aggregation_synthetic(tmp_path):
    # scene A: left half 0.1, right half 0.3 (linear); scene B: constant 0.05; scene C: dB file
    a = _s1_raster(tmp_path / "a.tif", lambda r, c: np.where(c < 20, 0.1, 0.3))
    b = _s1_raster(tmp_path / "b.tif", lambda r, c: np.full(r.shape, 0.05))
    cdb = _s1_raster(tmp_path / "c.tif", lambda r, c: np.full(r.shape, -10.0))   # 0.1 linear
    # polygon 1 straddles the halves (10 px each side), polygon 2 left half only,
    # polygon 3 outside every scene
    det = _polys_utm([(600100, 2599800, 600300, 2599900),     # cols 10..29, rows 10..19
                      (600000, 2599700, 600100, 2599800),
                      (700000, 2500000, 700100, 2500100)])
    import rasterio
    with rasterio.open(a) as ds:
        va = post.s1_scene_means(ds, det)
    assert va[0] == pytest.approx(0.2, rel=1e-5)          # mean LINEAR inside the polygon
    assert va[1] == pytest.approx(0.1, rel=1e-5)
    assert math.isnan(va[2])
    with rasterio.open(cdb) as ds:
        assert post._looks_like_db(ds)
    with rasterio.open(a) as ds:
        assert not post._looks_like_db(ds)

    scenes = [{"scene_id": "A", "date": "2026-03-01", "path": str(a)},
              {"scene_id": "B", "date": "2026-02-01", "path": str(b)},
              {"scene_id": "C", "date": "2026-01-01", "path": str(cdb)}]
    mean, std, n, used = post.s1_vv(scenes, det, log=lambda *_: None)
    exp0 = np.mean([10 * math.log10(0.2), 10 * math.log10(0.05), -10.0])   # mean of per-scene dB
    exp1 = np.mean([-10.0, 10 * math.log10(0.05), -10.0])
    assert mean[0] == pytest.approx(exp0, abs=1e-4)
    assert mean[1] == pytest.approx(exp1, abs=1e-4)
    assert std[0] == pytest.approx(np.std([10 * math.log10(0.2), 10 * math.log10(0.05), -10.0]),
                                   abs=1e-4)
    assert n.tolist() == [3, 3, 0]
    assert math.isnan(mean[2]) and len(used) == 3


def test_s1_invalid_pixels_ignored(tmp_path):
    # zeros, negatives and NaN are not gamma0; nodata value masked too
    def f(r, c):
        v = np.full(r.shape, 0.2)
        v[:, :15] = np.nan
        v[:, 15:18] = 0.0
        v[:, 18:20] = -9999.0
        return v
    p = _s1_raster(tmp_path / "x.tif", f, nodata=-9999.0)
    det = _polys_utm([(600100, 2599800, 600300, 2599900), (600000, 2599700, 600100, 2599800)])
    import rasterio
    with rasterio.open(p) as ds:
        v = post.s1_scene_means(ds, det)
    assert v[0] == pytest.approx(0.2, rel=1e-6)
    assert math.isnan(v[1])                                # only NaN under polygon 2
    st = np.vstack([v, v])
    mean, _, n = post.s1_aggregate(st)
    assert mean[0] == pytest.approx(10 * math.log10(0.2), abs=1e-5) and n.tolist() == [2, 0]
    m0, s0, n0 = post.s1_aggregate(np.empty((0, 3)))
    assert np.isnan(m0).all() and n0.tolist() == [0, 0, 0]


def test_select_s1_scenes():
    es = [{"path": "p%d" % i, "date": d, "relative_orbit": o}
          for i, (d, o) in enumerate([("2026-01-03", 5), ("2026-01-15", 5), ("2026-01-20", 78),
                                       ("2026-02-02", 5), ("2026-03-01", 78), ("2025-12-30", 5)])]
    s = post.select_s1_scenes(es, 8)
    assert [e["date"] for e in s] == ["2026-03-01", "2026-02-02", "2026-01-20", "2026-01-15",
                                      "2025-12-30"]
    assert len(post.select_s1_scenes(es, 2)) == 2
    assert post.select_s1_scenes([], 8) == []


# ------------------------------------------------------------------------- NISAR
@need_stage
@pytest.mark.skipif(not NISAR.exists(), reason="NISAR coherence raster absent")
def test_nisar_coh_matches_production(window):
    import geopandas as gpd
    from shapely.geometry import box
    d = gpd.read_file(CASCADE, layer="detections")
    d = d[d.within(box(*window["b4326"]))].reset_index(drop=True)
    assert len(d) == 10
    coh, appl = post.nisar_coh([{"path": str(NISAR)}], d, log=lambda *_: None)
    assert appl.all()
    np.testing.assert_allclose(coh, d["coh"].values, rtol=0, atol=1e-6)
    # a registered footprint that misses the polygons -> not applicable, NaN
    far = {"type": "Polygon", "coordinates": [[[60, 10], [61, 10], [61, 11], [60, 11], [60, 10]]]}
    coh2, appl2 = post.nisar_coh([{"path": str(NISAR), "footprint": far}], d, log=lambda *_: None)
    assert not appl2.any() and np.isnan(coh2).all()


def test_minwidth_rectangle():
    from shapely.geometry import box
    assert post.minwidth(box(0, 0, 50, 20)) == pytest.approx(20.0)


# ------------------------------------------------------------------------- inference grid / footprint
def test_block_grid_matches_production():
    H, W, B = 7000, 9000, 3072
    blocks = infer.block_origins(H, W, B)
    assert blocks[:3] == [(0, 0), (0, 2944), (0, 5888)]
    assert len(blocks) == len(range(0, H, 2944)) * len(range(0, W, 2944))
    cys, cxs = infer.tile_offsets(3072, 1000)
    assert cys[0] == 0 and cys[-1] == 3072 - 512 and all(b - a <= 384 for a, b in zip(cys, cys[1:]))
    assert cxs[-1] + 512 >= 1000
    assert infer.tile_offsets(300, 300) == ([0], [0])


def test_mbtiles_footprint_synthetic(tmp_path):
    p = tmp_path / "t.mbtiles"
    z, n = 18, 2 ** 18
    with sqlite3.connect(p) as c:
        c.execute("create table metadata (name text, value text)")
        c.execute("create table tiles (zoom_level int, tile_column int, tile_row int, tile_data blob)")
        c.executemany("insert into metadata values (?,?)", [("maxzoom", "18"), ("minzoom", "18")])
        xyz = [(182000, 113000), (182001, 113000), (182005, 113003)]
        c.executemany("insert into tiles values (18,?,?,x'00')", [(x, n - 1 - y) for x, y in xyz])
    R = 6378137.0 * math.pi
    size = 2 * R / n
    bb = (-R + 181990 * size, R - 113010 * size, -R + 182010 * size, R - 112990 * size)
    fp, cnt = mbt.footprint_3857(str(p), bb)
    assert cnt == 3
    assert fp.area == pytest.approx(3 * size * size, rel=1e-9)
    assert len(getattr(fp, "geoms", [fp])) == 2               # two adjacent tiles merge
    minx, _, _, maxy = fp.bounds
    assert minx == pytest.approx(-R + 182000 * size) and maxy == pytest.approx(R - 113000 * size)
    fp2, cnt2 = mbt.footprint_3857(str(p), (0, 0, 1, 1))
    assert cnt2 == 0 and fp2.is_empty


@need_stage
@need_mbt
def test_mbtiles_footprint_real_window(window):
    fp, cnt = mbt.footprint_3857(str(MBT), window["b3857"])
    assert cnt > 0
    from shapely.geometry import box
    inter = fp.intersection(box(*window["b3857"])).area
    assert inter == pytest.approx(box(*window["b3857"]).area, rel=1e-6)   # interior of corridor


# ------------------------------------------------------------------------- subprocess tests
def _request(d, b4326, *, s1=None, nisar=True, ckpt=CKPT, mbtiles=MBT):
    x0, y0, x1, y1 = b4326
    aoi = {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
    wd = Path(d) / "wd"
    req = {"layer_key": "substations", "aoi": aoi, "work_dir": str(wd),
           "out_gpkg": str(wd / "out.gpkg"), "result_json": str(wd / "result.json"),
           "imagery": {"esri_wayback_z18": [{"path": str(mbtiles), "release": "26334",
                                             "date": "2026-08-05"}],
                       "sentinel1_rtc_vv": s1 or [],
                       "nisar_coh": [{"path": str(NISAR)}] if nisar else []},
           "inputs": {"osm_substations_gpkg": str(OSM) if OSM.exists() else None},
           "weights": {"checkpoint": str(ckpt), "hf_home": str(HF)},
           "gpu": {"device": 0, "vram_mb": 14000},
           "params": {"threshold": 0.50, "min_area_ha": 0.05, "vv_db_min": -10.10,
                      "coh_min": 0.337, "s1_scenes": 8}}
    Path(d).mkdir(parents=True, exist_ok=True)
    p = Path(d) / "request.json"
    p.write_text(json.dumps(req))
    return p, req


def _env():
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = ""          # belt and braces: nothing here may touch a GPU
    e["HF_HUB_OFFLINE"] = "1"
    e["TRANSFORMERS_OFFLINE"] = "1"
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        e[k] = "http://127.0.0.1:9"
    return e


@need_py
@need_mbt
@need_stage
def test_check_mode(window, tmp_path):
    p, req = _request(tmp_path, window["b4326"])
    r = subprocess.run([PY, str(RUN), "--request", str(p), "--check"], capture_output=True,
                       text=True, env=_env(), timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    rep = json.loads(r.stdout[r.stdout.index("{"):])
    byname = {c["name"]: c for c in rep["checks"]}
    assert rep["ok"] is True
    assert byname["checkpoint"]["ok"] and byname["import custom_modules"]["ok"]
    assert any(k.startswith("hf_cache") and v["ok"] for k, v in byname.items())
    mb = byname["mbtiles " + MBT.name]
    assert mb["ok"] and "tiles in AOI bbox" in mb["detail"]
    assert byname["sentinel1_rtc_vv"]["level"] == "warn"
    assert json.loads((Path(req["work_dir"]) / "check.json").read_text())["ok"] is True

    # a missing checkpoint and an AOI with no tiles both fail the check
    p2, _ = _request(tmp_path / "bad", (60.0, 10.0, 60.01, 10.01), ckpt="/nonexistent.ckpt")
    r2 = subprocess.run([PY, str(RUN), "--request", str(p2), "--check"], capture_output=True,
                        text=True, env=_env(), timeout=300)
    assert r2.returncode == 1
    rep2 = json.loads(r2.stdout[r2.stdout.index("{"):])
    bad = {c["name"] for c in rep2["checks"] if not c["ok"]}
    assert "checkpoint" in bad and "mbtiles " + MBT.name in bad


@need_py
@need_mbt
@need_stage
def test_end_to_end_prob_override(window, tmp_path):
    """Post-processing chain on the production prob crop, with a synthetic S1 scene covering the
    western half of the window (bright -> passes) and nothing in the east (-> review)."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_bounds as tfb
    x0, y0, x1, y1 = window["b4326"]
    xm = (x0 + x1) / 2
    s1p = tmp_path / "s1_west.tif"
    with rasterio.open(s1p, "w", driver="GTiff", width=200, height=200, count=1, dtype="float32",
                       crs="EPSG:4326", transform=tfb(x0 - 0.01, y0 - 0.01, xm, y1 + 0.01,
                                                      200, 200)) as d:
        d.write(np.full((200, 200), 0.5, np.float32), 1)       # -3.0 dB
    s1 = [{"scene_id": "SYN", "date": "2026-03-04", "path": str(s1p)}]
    p, req = _request(tmp_path, window["b4326"], s1=s1)
    r = subprocess.run([PY, str(RUN), "--request", str(p), "--prob", str(window["prob"])],
                       capture_output=True, text=True, env=_env(), timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "PROGRESS 1.0000" in r.stdout
    res = json.loads(Path(req["result_json"]).read_text())
    assert res["status"] == "ok" and res["model"] == "substation-dinov3-lora-stageB"
    assert res["model_version"] == "epoch22" and res["image_dates"] == ["2026-08-05"]
    assert res["image_source"] == "Esri Wayback z18 release 26334 + Sentinel-1 RTC + NISAR"
    assert res["artifacts"]["prob_raster"] == [str(window["prob"])]
    assert res["coverage"]["type"] in ("Polygon", "MultiPolygon")
    f = gpd.read_file(req["out_gpkg"], layer="features")
    rj = gpd.read_file(req["out_gpkg"], layer="rejected")
    assert f.crs.to_epsg() == 4326 and rj.crs.to_epsg() == 4326
    for c in ("class", "confidence", "name", "src_id", "source", "area_ha", "vv_db", "coh",
              "minwidth", "gates"):
        assert c in f.columns
    assert "reject_reason" in rj.columns
    assert len(f) + len(rj) == 10 and res["counts"]["features"] == len(f)
    assert set(f["class"]) == {"substation"}
    assert f["src_id"].is_unique and not set(f["src_id"]) & set(rj["src_id"])
    west = f.geometry.representative_point().x < xm
    assert (f.loc[west & (f.source == "model_new"), "confidence"] == 0.75).all()
    assert (f.loc[~west & (f.source == "model_new"), "confidence"] == 0.5).all()
    assert f.loc[~west & (f.source == "model_new"), "review"].all()
    osm = f[f.source == "model_confirms_osm"]
    assert set(osm["name"]) == {"Essar", "Global Hitech"} and (osm.confidence == 0.9).all()
    # the one low-coherence detection (coh 0.151 in production) is rejected on NISAR
    assert (rj["reject_reason"].str.contains("nisar_coh")).any()
    from shapely.geometry import box
    assert f.geometry.within(box(x0, y0, x1, y1).buffer(1e-9)).all()


@need_py
@need_mbt
def test_no_imagery_empty_output(tmp_path):
    p, req = _request(tmp_path, (60.0, 10.0, 60.01, 10.01), nisar=False)
    r = subprocess.run([PY, str(RUN), "--request", str(p)], capture_output=True, text=True,
                       env=_env(), timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(Path(req["result_json"]).read_text())
    assert res["counts"]["features"] == 0
    assert res["coverage"] == {"type": "GeometryCollection", "geometries": []}
    import geopandas as gpd
    assert len(gpd.read_file(req["out_gpkg"], layer="features")) == 0


@need_py
@need_mbt
@need_stage
def test_sigterm_exits_promptly(window, tmp_path):
    p, _ = _request(tmp_path, window["b4326"])
    pr = subprocess.Popen([PY, str(RUN), "--request", str(p), "--prob", str(window["prob"])],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(),
                          start_new_session=True)
    line = pr.stdout.readline()                 # first PROGRESS line: handler is installed
    assert line.startswith("PROGRESS")
    t0 = time.time()
    os.killpg(pr.pid, signal.SIGTERM)
    rc = pr.wait(timeout=60)
    assert rc != 0 and time.time() - t0 < 60


_STUB = r'''
import sys, json, numpy as np, rasterio, torch
sys.path.insert(0, sys.argv[1])
import sub_infer
from shapely.geometry import box
class Stub(torch.nn.Module):            # p(class 1) = 0.5 everywhere; no weights involved
    def forward(self, x):
        return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])
reg = box(*json.loads(sys.argv[3]))
st = sub_infer.run(sys.argv[2], reg, Stub(), sys.argv[4], block=3072, batch=8, device="cpu",
                   log=lambda *a: None)
print(json.dumps(st))
'''


@need_py
@need_mbt
@need_stage
def test_inference_windowing_with_stub_model(window, tmp_path):
    """sub_infer.run grid / masking / writing on CPU with a stub network (not the real model):
    the output is aligned to the MBTiles pixel grid, 127 inside the region, 0 outside."""
    x0, y0, x1, y1 = window["b3857"]
    reg = (x0 + 300.0, y0 + 300.0, x0 + 1300.0, y0 + 900.0)     # 1000 x 600 m inside the window
    out = tmp_path / "stub_prob.tif"
    env = _env()
    r = subprocess.run([PY, "-c", _STUB, str(PLUG), str(MBT), json.dumps(reg), str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    st = json.loads(r.stdout.strip().splitlines()[-1])
    assert st["blocks"] >= 1 and st["skipped"] == 0
    import rasterio
    with rasterio.open(out) as d, rasterio.open(MBT) as m:
        assert d.crs.to_epsg() == 3857 and d.nodata == 0
        # same pixel size, origin on the MBTiles grid
        assert d.transform.a == pytest.approx(m.transform.a)
        assert ((d.transform.c - m.transform.c) / m.transform.a) == pytest.approx(
            round((d.transform.c - m.transform.c) / m.transform.a), abs=1e-6)
        a = d.read(1)
    vals = set(np.unique(a).tolist())
    assert vals <= {0, 127} and 127 in vals
    inside = (a == 127).sum() * d.transform.a ** 2
    assert inside == pytest.approx(1000 * 600, rel=0.01)


@need_py
@need_mbt
@need_stage
def test_imagery_but_no_detections(window, tmp_path):
    p, req = _request(tmp_path, window["b4326"])
    req["params"]["threshold"] = 1.01                      # nothing can pass
    p.write_text(json.dumps(req))
    r = subprocess.run([PY, str(RUN), "--request", str(p), "--prob", str(window["prob"])],
                       capture_output=True, text=True, env=_env(), timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(Path(req["result_json"]).read_text())
    assert res["counts"]["features"] == 0 and res["coverage"]["type"] == "Polygon"
    import geopandas as gpd
    f = gpd.read_file(req["out_gpkg"], layer="features")
    assert len(f) == 0 and "confidence" in f.columns
