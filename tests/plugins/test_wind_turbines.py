"""Wind-turbine plugin tests on real Cartosat-3 PAN scenes - CPU only, no network, no GPU.

The mask step (GPU) is not exercised: every run reuses an existing round4 mask from
/mnt/drso-data/devesh/turbine_expg/masks_edge through params.mask_dir, and the plugin is started
with CUDA_VISIBLE_DEVICES="" so a missing mask would fail the run instead of touching a GPU.

pytest lives in the beeline env, the plugin runs in the terratorch env, so this file is dual-use:
  * pytest:  /home/vivek_jain/miniforge3/envs/beeline/bin/python -m pytest tests/plugins/test_wind_turbines.py
  * helper, run by the tests under the terratorch interpreter:
        <terratorch python> test_wind_turbines.py request <work_dir> <aoi_mode> <scene_id>[,<path>] ...
        <terratorch python> test_wind_turbines.py nms
        <terratorch python> test_wind_turbines.py osm
        <terratorch python> test_wind_turbines.py compare <out.gpkg> <scene_gpkg> <reference_gpkg>
        <terratorch python> test_wind_turbines.py coarse <work_dir>     (1 m synthetic scene)
        <terratorch python> test_wind_turbines.py count <out.gpkg>
Tests skip when the research data is absent (e.g. on the prod machine).
The Exp-G run on one full scene takes ~2 min (2.2-gigapixel mask).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO / "lulc_models" / "wind_turbines"
PLUGIN = PLUGIN_DIR / "run_aoi.py"
PY = os.environ.get("WT_TEST_PY", "/home/vivek_jain/miniforge3/envs/terratorch/bin/python")
D = "/mnt/drso-data/devesh/turbine_expg"
MASKS = D + "/masks_edge"
OUT_ALL = D + "/out_all"
OSM = "/home/vivek_jain/seggpt-local/turbines_work/generator_source_wind_32642.gpkg"
CKPT = ("/home/vivek_jain/seggpt-local/turbines_work/dinov3_lora_turbine_round4/lightning_logs/version_0/"
        "checkpoints/epoch=25.ckpt")
HF_HOME = str(Path.home() / ".cache" / "huggingface")
RAW141_TRUNCATED = "/mnt/drso-data/Order_60409_PAN/2660409141/BAND.tif"
ORDERS = ("57715", "58898", "60409")
TRUTH_SCENE = "2660409341"           # 163 Exp-G detections, 156 OSM TP at 100 m in the research run
DUP_PAIR = ("2660409251", "265889831")   # identical footprints, two acquisitions


def scene_dir(sid: str) -> str:
    for o in ORDERS:
        p = f"/mnt/drso-data/Order_{o}_PAN/{sid}"
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(sid)


# ============================================================ helpers that run under terratorch
def _footprint(sid: str):
    import fiona
    import geopandas as gpd
    from shapely.geometry import shape
    with fiona.open(scene_dir(sid) + "/BAND.shp") as f:
        g = shape(next(iter(f))["geometry"])
    return gpd.GeoSeries([g], crs=32642).to_crs(4326).iloc[0]       # .prj unreadable; raster CRS


def _request(work_dir: str, aoi_mode: str, specs: list[str]):
    """aoi_mode: 'cover' = envelope of all scene footprints + 0.01 deg; 'first' = first footprint."""
    from shapely.geometry import mapping
    from shapely.ops import unary_union
    items, fps = [], []
    for spec in specs:
        sid, _, path = spec.partition(",")
        fp = _footprint(sid); fps.append(fp)
        items.append({"scene_id": sid, "path": path or scene_dir(sid) + "/BAND.tif", "footprint": mapping(fp)})
    aoi = unary_union(fps).buffer(0.01).envelope if aoi_mode == "cover" else fps[0]
    wd = Path(work_dir); wd.mkdir(parents=True, exist_ok=True)
    req = {"layer_key": "wind_turbines", "aoi": mapping(aoi), "work_dir": str(wd),
           "out_gpkg": str(wd / "out.gpkg"), "result_json": str(wd / "result.json"),
           "imagery": {"cartosat3_pan": items}, "inputs": {"osm_wind_gpkg": OSM},
           "weights": {"checkpoint": CKPT, "hf_home": HF_HOME}, "gpu": {"device": 0, "vram_mb": 24000},
           "params": {"thresh": 160, "min_px": 15, "nms_m": 95, "osm_match_m": 60, "gsd_m": 0.45,
                      "mask_dir": MASKS}}
    (wd / "request.json").write_text(json.dumps(req, indent=1))
    print(wd / "request.json")


def _plugin_modules():
    import importlib.util
    sys.path.insert(0, str(PLUGIN_DIR)); sys.path.insert(0, str(PLUGIN_DIR.parent))
    spec = importlib.util.spec_from_file_location("wind_turbines_run_aoi", PLUGIN)
    ra = importlib.util.module_from_spec(spec); spec.loader.exec_module(ra)
    import turbine_post as TP
    return ra, TP


def _nms():
    """Cross-scene NMS + AOI clip on stored research outputs (no rasters read)."""
    import geopandas as gpd
    import numpy as np
    ra, TP = _plugin_modules()
    from shapely.ops import unary_union
    load = lambda s: gpd.read_file(f"{OUT_ALL}/{s}_expG_detections.gpkg")   # noqa: E731
    p = dict(ra.DEFAULTS)
    rep = {}
    # 1. the real identical-footprint pair (both stored outputs are empty: Exp-G's stage-0 gate)
    a, b = DUP_PAIR
    aoi = unary_union([_footprint(a), _footprint(b)]).buffer(0.01)
    f, _ = ra.postprocess([(a, load(a)), (b, load(b))], aoi, p, OSM, [])
    rep["dup_pair"] = {"in": [len(load(a)), len(load(b))], "out": len(f)}
    # 2. the same pair with a non-empty detection set: scene b re-observes scene a's turbines
    #    (TRUTH_SCENE's detections, jittered up to 7 m, half the areas larger in b)
    t = load(TRUTH_SCENE); rng = np.random.default_rng(0)
    t2 = t.copy(); t2["geometry"] = gpd.points_from_xy(t.geometry.x + rng.uniform(-5, 5, len(t)),
                                                         t.geometry.y + rng.uniform(-5, 5, len(t)), crs=t.crs)
    t2["area"] = t["area"] * np.where(np.arange(len(t)) % 2 == 0, 1.2, 0.8)
    aoi = _footprint(TRUTH_SCENE).buffer(0.01)
    f, epsg = ra.postprocess([(a, t), (b, t2)], aoi, p, OSM, [])
    rep["dup_synthetic"] = {"in": [len(t), len(t2)], "out": len(f), "epsg": epsg,
                            "n_views": sorted(set(f["n_views"].tolist())),
                            "from_b": int((f["scene_id"] == b).sum()),
                            "src_unique": bool(f["src_id"].is_unique)}
    # 3. real overlapping scenes 2660409331 / 2660409341 (3 shared turbines in the research run)
    s1, s2 = "2660409331", TRUTH_SCENE
    aoi = unary_union([_footprint(s1), _footprint(s2)]).buffer(0.01)
    d1, d2 = load(s1), load(s2)
    f, epsg = ra.postprocess([(s1, d1), (s2, d2)], aoi, p, OSM, [])
    fm = f.to_crs(epsg)
    gm = fm.geometry.values
    close = [(i, j) for i in range(len(fm)) for j in range(i + 1, len(fm))
             if fm.scene_id.iloc[i] != fm.scene_id.iloc[j] and gm[i].distance(gm[j]) < p["nms_m"]]
    allin = gpd.GeoDataFrame({"scene_id": [s1] * len(d1) + [s2] * len(d2),
                              "area": list(d1["area"]) + list(d2["area"])},
                             geometry=list(d1.geometry) + list(d2.geometry), crs=d1.crs).to_crs(epsg)
    pairs = 0
    for i in range(len(d1)):
        for j in range(len(d1), len(allin)):
            if allin.geometry.iloc[i].distance(allin.geometry.iloc[j]) < p["nms_m"]:
                pairs += 1
    rep["overlap_331_341"] = {"in": [len(d1), len(d2)], "cross_pairs": pairs, "out": len(f),
                              "cross_scene_within_nms_after": len(close),
                              "n_views2": int((f["n_views"] == 2).sum())}
    # 4. AOI clip: a half-scene AOI drops detections outside it
    fp = _footprint(TRUTH_SCENE); c = fp.centroid; minx, miny, maxx, maxy = fp.bounds
    from shapely.geometry import box
    half = box(minx, miny, c.x, maxy)
    f, _ = ra.postprocess([(TRUTH_SCENE, t)], half, p, OSM, [])
    inside = int(t.to_crs(4326).geometry.within(half).sum())
    rep["clip"] = {"in": len(t), "out": len(f), "inside": inside}
    print(json.dumps(rep))


def _osm():
    """OSM confirmation on synthetic points around real OSM turbines."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point
    ra, TP = _plugin_modules()
    o = TP.load_osm_wind(OSM)
    o = o.to_crs(32642).iloc[:4]
    pts, exp = [], []
    for k, g in enumerate(o.geometry):
        pts.append(Point(g.x + [30, 59, 61, 200][k], g.y)); exp.append([True, True, False, False][k])
    g0 = o.geometry.iloc[0]
    pts.append(Point(g0.x - 40, g0.y)); exp.append(False)      # 2nd detection near the same OSM point
    det = gpd.GeoDataFrame({"area": [5000, 5000, 5000, 5000, 100]}, geometry=pts, crs=32642)
    conf, oid = TP.osm_confirm(det, o, 60.0)
    # through postprocess: confidence + review flags, 4326 output
    aoi = gpd.GeoSeries([det.union_all().buffer(500).envelope], crs=32642).to_crs(4326).iloc[0]
    det["n_frag"] = 1
    f, _ = ra.postprocess([("s1", det)], aoi, dict(ra.DEFAULTS), OSM, [])
    w = []
    f2, _ = ra.postprocess([("s1", det)], aoi, dict(ra.DEFAULTS), None, w)
    print(json.dumps({"conf": conf, "expected": exp, "osm_ids": oid, "osm_has_id": "osm_id" in o.columns,
                      "crs": str(f.crs),
                      "rows": [[float(r.confidence), bool(r.osm_confirmed), bool(r.review), r["class"],
                                None if r["name"] is None or r["name"] != r["name"] else r["name"]]
                               for _, r in f.iterrows()],
                      "no_osm_conf": sorted(set(f2["confidence"].tolist())), "no_osm_warn": w}))


def _compare(out_gpkg: str, scene_gpkg: str, ref_gpkg: str):
    import geopandas as gpd
    import numpy as np
    out = gpd.read_file(out_gpkg, layer="features")
    sc = gpd.read_file(scene_gpkg)
    ref = gpd.read_file(ref_gpkg).to_crs(sc.crs)
    key = lambda g: sorted(zip(np.round(g.geometry.x, 2), np.round(g.geometry.y, 2), g["area"], g["n_frag"]))  # noqa: E731
    o = gpd.read_file(OSM).to_crs(32642)
    om = out.to_crs(32642)
    dist = [float(o.distance(g).min()) for g in om.geometry]
    conf = om["osm_confirmed"].astype(bool).tolist()
    print(json.dumps({"n_out": len(out), "n_scene": len(sc), "n_ref": len(ref), "scene_identical": key(sc) == key(ref),
                      "out_crs": str(out.crs), "columns": list(out.columns),
                      "classes": sorted(set(out["class"])), "conf_values": sorted(set(out["confidence"])),
                      "name_null": bool(out["name"].isna().all()), "src_unique": bool(out["src_id"].is_unique),
                      "n_confirmed": int(sum(conf)), "n_within60": int(sum(d <= 60 for d in dist)),
                      "n_within100": int(sum(d <= 100 for d in dist)),
                      "confirmed_le60": all(d <= 60.0 + 1e-6 for d, c in zip(dist, conf) if c),
                      "median_offset_m": float(np.median(dist)) if dist else None}))


def _coarse(work_dir: str):
    """A 1 m GeoTIFF (not 0.45 m) inside TRUTH_SCENE's footprint + a request for it."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box, mapping
    import geopandas as gpd
    wd = Path(work_dir); wd.mkdir(parents=True, exist_ok=True)
    tif = wd / "coarse" / "BAND.tif"; tif.parent.mkdir(exist_ok=True)
    x0, y0 = 655000.0, 2568000.0
    with rasterio.open(tif, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint16", crs="EPSG:32642",
                       transform=from_origin(x0, y0, 1.0, 1.0), nodata=0) as d:
        d.write(np.full((1, 256, 256), 500, np.uint16))
    fp = gpd.GeoSeries([box(x0, y0 - 256, x0 + 256, y0)], crs=32642).to_crs(4326).iloc[0]
    req = {"layer_key": "wind_turbines", "aoi": mapping(fp.buffer(0.001)), "work_dir": str(wd),
           "out_gpkg": str(wd / "out.gpkg"), "result_json": str(wd / "result.json"),
           "imagery": {"cartosat3_pan": [{"scene_id": "coarse1m", "path": str(tif), "footprint": mapping(fp)}]},
           "inputs": {"osm_wind_gpkg": OSM}, "weights": {"checkpoint": "/nonexistent.ckpt", "hf_home": "/nonexistent"},
           "gpu": {"device": 0}, "params": {}}
    (wd / "request.json").write_text(json.dumps(req, indent=1))
    print(wd / "request.json")


def _count(gpkg: str):
    import geopandas as gpd
    g = gpd.read_file(gpkg, layer="features")
    print(json.dumps({"n": len(g), "crs": str(g.crs), "columns": list(g.columns)}))


if __name__ == "__main__" and len(sys.argv) > 1:
    cmd = sys.argv[1]
    if cmd == "request":
        _request(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "nms":
        _nms()
    elif cmd == "osm":
        _osm()
    elif cmd == "coarse":
        _coarse(sys.argv[2])
    elif cmd == "count":
        _count(sys.argv[2])
    elif cmd == "compare":
        _compare(*sys.argv[2:5])
    sys.exit(0)


# ============================================================ pytest side
import pytest  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (os.path.isdir(MASKS) and os.path.isdir(OUT_ALL) and os.path.exists(OSM) and os.path.exists(PY)),
    reason="turbine research data / terratorch env not on this machine")


def _helper(*args, timeout=600):
    r = subprocess.run([PY, __file__, *args], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout.strip().splitlines()[-1]


def _plugin(req: Path, *extra, timeout=1800):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", NO_PROXY="")
    return subprocess.run([PY, str(PLUGIN), "--request", str(req), *extra], capture_output=True, text=True,
                          timeout=timeout, env=env)


@pytest.fixture(scope="module")
def truth_run(tmp_path_factory):
    wd = tmp_path_factory.mktemp("wt_truth")
    req = Path(_helper("request", str(wd), "cover", TRUTH_SCENE))
    r = _plugin(req)
    assert r.returncode == 0, r.stderr[-3000:]
    return wd, r


def test_expg_from_existing_mask_matches_research_output(truth_run):
    wd, r = truth_run
    assert not (wd / "masks").exists(), "mask step ran although a mask was supplied"
    assert "PROGRESS 1.0000" in r.stdout
    c = json.loads(_helper("compare", str(wd / "out.gpkg"), str(wd / f"{TRUTH_SCENE}_expG_detections.gpkg"),
                           f"{OUT_ALL}/{TRUTH_SCENE}_expG_detections.gpkg"))
    assert c["n_ref"] == 163
    assert c["n_scene"] == c["n_ref"] and c["scene_identical"], c       # same points, area, n_frag
    assert c["n_out"] == c["n_ref"]                                     # single scene, AOI covers it
    assert c["out_crs"] == "EPSG:4326"
    for col in ("class", "confidence", "name", "src_id", "scene_id", "area", "n_frag", "osm_confirmed"):
        assert col in c["columns"]
    assert c["classes"] == ["wind_turbine"] and c["name_null"] and c["src_unique"]
    assert set(c["conf_values"]) <= {0.4, 0.9}
    # OSM confirmation: exactly the detections with an OSM turbine within 60 m (one-to-one)
    assert c["confirmed_le60"] and c["n_confirmed"] == c["n_within60"] and c["n_confirmed"] > 90, c

    res = json.loads((wd / "result.json").read_text())
    assert res["status"] == "ok" and res["model"] == "turbine-dinov3-lora-round4"
    assert res["model_version"] == "epoch25 + Exp-G" and res["image_source"] == "Cartosat-3 PAN 0.45 m"
    assert res["image_dates"] == ["2025-12-27"]                          # BAND_META DateOfPass
    assert res["counts"]["features"] == 163 and res["counts"]["osm_confirmed"] == c["n_confirmed"]
    assert res["coverage"]["type"] in ("Polygon", "MultiPolygon") and res["utm_epsg"] == 32642


def test_cross_scene_nms_and_clip():
    rep = json.loads(_helper("nms"))
    assert rep["dup_pair"] == {"in": [0, 0], "out": 0}
    s = rep["dup_synthetic"]
    assert s["out"] == s["in"][0] == 163 and s["n_views"] == [2] and s["src_unique"]
    assert 0 < s["from_b"] < 163                                         # the larger observation wins
    o = rep["overlap_331_341"]
    assert o["cross_scene_within_nms_after"] == 0
    assert o["out"] == sum(o["in"]) - o["cross_pairs"] and o["n_views2"] == o["cross_pairs"] > 0, o
    assert rep["clip"]["out"] == rep["clip"]["inside"] and 0 < rep["clip"]["out"] < rep["clip"]["in"]


def test_osm_confirmation():
    rep = json.loads(_helper("osm"))
    assert rep["conf"] == rep["expected"]
    if rep["osm_has_id"]:
        assert all((i is not None) == c for i, c in zip(rep["osm_ids"], rep["conf"]))
    assert rep["crs"] == "EPSG:4326"
    for conf, ok, review, cls, name in rep["rows"]:
        assert conf == (0.9 if ok else 0.4) and review == (not ok) and cls == "wind_turbine" and name is None
    assert rep["no_osm_conf"] == [0.4] and rep["no_osm_warn"]


def test_check_two_scenes(tmp_path):
    req = Path(_helper("request", str(tmp_path / "ok"), "first", *DUP_PAIR))
    r = _plugin(req, "--check", timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rep = json.loads((tmp_path / "ok" / "check.json").read_text())
    assert rep["ok"] and rep["checkpoint"]["exists"] and rep["hf_home"]["model.safetensors"]
    assert rep["imports"]["custom_modules"] == "ok"
    assert sorted(rep["scenes_usable"]) == sorted(DUP_PAIR)
    for s in rep["scenes"]:
        assert s["crs"] == "EPSG:32642" and s["gsd_ok"] and s["intersects_aoi"] and s["blocks_past_eof"] == 0
        assert s["existing_mask"] and s["date"]
    assert not (tmp_path / "ok" / "out.gpkg").exists()                  # check mode writes no output


@pytest.mark.skipif(not os.path.exists(RAW141_TRUNCATED), reason="truncated raw 2660409141 not present")
def test_check_flags_truncated_scene(tmp_path):
    req = Path(_helper("request", str(tmp_path / "bad"), "first", "2660409141," + RAW141_TRUNCATED))
    r = _plugin(req, "--check", timeout=600)
    assert r.returncode == 1
    rep = json.loads((tmp_path / "bad" / "check.json").read_text())
    s = rep["scenes"][0]
    assert s["blocks_past_eof"] > 0 and not s["use"] and any("truncated" in p for p in rep["problems"])


def test_non_045_scene_skipped_with_warning(tmp_path):
    """A 1 m scene is skipped (warning, not a crash) before any model load; empty output, exit 0."""
    req = Path(_helper("coarse", str(tmp_path / "coarse")))
    r = _plugin(req, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    res = json.loads((tmp_path / "coarse" / "result.json").read_text())
    assert res["counts"]["features"] == 0 and res["coverage"] is None
    assert any("GSD" in w for w in res["warnings"])
    c = json.loads(_helper("count", str(tmp_path / "coarse" / "out.gpkg")))
    assert c["n"] == 0 and c["crs"] == "EPSG:4326" and "confidence" in c["columns"]
