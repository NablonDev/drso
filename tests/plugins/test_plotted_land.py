"""Plotted-land plugin tests on small real clips (Gujarat minus Kachchh) - no network, no GPU.

pytest lives in the beeline env, the plugin runs in the clay env, so this file is dual-use:
  * pytest (any env):         /home/vivek_jain/miniforge3/envs/beeline/bin/python -m pytest tests/plugins/test_plotted_land.py
  * fixture/inspection helper, run by the tests under the clay interpreter:
        <clay python> test_plotted_land.py fixture  <out_dir> <lon0> <lat0> <lon1> <lat1>
        <clay python> test_plotted_land.py statewide <lon0> <lat0> <lon1> <lat1>
        <clay python> test_plotted_land.py summary  <out.gpkg>
        <clay python> test_plotted_land.py wc       <out.tif> <lon0> <lat0> <lon1> <lat1> <class>
        <clay python> test_plotted_land.py lines    <out.gpkg> <lon> <lat>
Tests skip when the statewide data under /mnt/cog-data/guj is absent (e.g. on the prod machine).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN = REPO / "lulc_models" / "plotted_land" / "run_aoi.py"
CLAY = os.environ.get("PL_TEST_PY", "/home/vivek_jain/miniforge3/envs/clay/bin/python")
G = "/mnt/cog-data/guj"
VEC, PROB = G + "/vec", G + "/prob_u"
BLD = "/home/vivek_jain/guj_buildings/guj_buildings.gpkg"
STATEWIDE = G + "/pl/Gujarat_Plotted_Land_minus_Kachchh_byregion.gpkg"
# 2 x 2 km clip west of Ahmedabad; the statewide run has 7 clusters here (representative point inside)
CLIP = (72.4401, 22.9337, 72.4581, 22.9502)
O = 20037508.342789244; TM = 2 * O / 2 ** 18


# ============================================================ helpers that run under the clay interpreter
def _fixture(out_dir, lon0, lat0, lon1, lat1, margin=4500.0):
    """roads (4326 gpkg, layer features), prob tif list, buildings (layer features) for the clip + margin"""
    import geopandas as gpd, pandas as pd, shapely
    from pyproj import Transformer
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    t = Transformer.from_crs(4326, 3857, always_xy=True)
    x0, y0 = t.transform(lon0, lat0); x1, y1 = t.transform(lon1, lat1)
    x0 -= margin; y0 -= margin; x1 += margin; y1 += margin
    bl = re.compile(r"^(\d+)_(\d+)\.geojson$"); vec, prob = [], []
    for f in os.listdir(VEC):
        m = bl.match(f)
        if not m: continue
        bx, by = int(m.group(1)), int(m.group(2)); bx0 = -O + bx * TM; by1 = O - by * TM
        if bx0 < x1 and bx0 + 16 * TM > x0 and by1 > y0 and by1 - 16 * TM < y1:
            vec.append(VEC + "/" + f)
            u = PROB + "/u_%d_%d.tif" % (bx, by)
            if os.path.exists(u): prob.append(u)
    feats = []
    for f in vec: feats += json.load(open(f)).get("features", [])
    g = gpd.GeoDataFrame.from_features(feats, crs=3857) if feats else gpd.GeoDataFrame(geometry=[], crs=3857)
    g = g.clip(shapely.box(x0, y0, x1, y1))
    g = gpd.GeoDataFrame({"class": ["village_road"] * len(g)}, geometry=g.geometry.values, crs=3857).to_crs(4326)
    g.to_file(out / "roads.gpkg", layer="features", driver="GPKG")
    bb = tuple(gpd.GeoSeries([shapely.box(x0, y0, x1, y1)], crs=3857).to_crs(4326).total_bounds)
    B = pd.concat([gpd.read_file(BLD, layer=l, bbox=bb, engine="pyogrio", columns=[])[["geometry"]]
                   for l in ("google_open_buildings_v3", "microsoft_buildings_2026")], ignore_index=True)
    gpd.GeoDataFrame({"class": ["building"] * len(B)}, geometry=B.geometry.values, crs=4326).to_file(out / "buildings.gpkg", layer="features", driver="GPKG")
    print(json.dumps({"roads": len(g), "prob": prob, "buildings": len(B)}))


def _statewide(lon0, lat0, lon1, lat1):
    import geopandas as gpd, shapely
    aoi = shapely.box(lon0, lat0, lon1, lat1)
    C = gpd.read_file(STATEWIDE, layer="plotted_land_clusters", bbox=(lon0 - .02, lat0 - .02, lon1 + .02, lat1 + .02), engine="pyogrio")
    rp = C.to_crs(C.estimate_utm_crs()).geometry.representative_point().to_crs(4326)
    k = rp.within(aoi).values
    print(json.dumps({"clusters": int(k.sum()), "plots": int(C.n_plots[k].sum())}))


def _summary(path):
    import geopandas as gpd, pyogrio
    out = {}
    for l, gt in pyogrio.list_layers(path):
        d = gpd.read_file(path, layer=l)
        out[l] = {"n": len(d), "geom": gt, "crs": d.crs.to_epsg() if d.crs else None, "cols": [c for c in d.columns if c != "geometry"],
                  "classes": sorted(set(d["class"])) if "class" in d else None,
                  "src_ids": sorted(d["src_id"])[:3] if "src_id" in d else None,
                  "n_plots": int(d["n_plots"].sum()) if "n_plots" in d and len(d) else 0,
                  "built_null": int(d["built_pct"].isna().sum()) if "built_pct" in d else None}
    print(json.dumps(out))


def _wc(path, lon0, lat0, lon1, lat1, cls):
    """synthetic ESA WorldCover-like tile (EPSG:4326, uint8, 1/12000 deg ~ 10 m), one class everywhere"""
    import numpy as np, rasterio
    from rasterio.transform import from_origin
    r = 1 / 12000.0; w = int((lon1 - lon0) / r) + 1; h = int((lat1 - lat0) / r) + 1
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(lon0, lat1, r, r), nodata=0, compress="deflate") as d:
        d.write(np.full((1, h, w), int(cls), np.uint8))


def _salt(path, lon0, lat0, lon1, lat1):
    """synthetic salt-scape input: one salt_pan polygon over the whole box, with its coverage layer"""
    import geopandas as gpd
    from shapely.geometry import box
    b = box(lon0, lat0, lon1, lat1)
    gpd.GeoDataFrame({"class": ["salt_pan"], "layer": ["salt_pans"]}, geometry=[b], crs=4326).to_file(path, layer="features", driver="GPKG")
    gpd.GeoDataFrame({"id": [1]}, geometry=[b], crs=4326).to_file(path, layer="coverage", driver="GPKG")


def _lines(path, lon, lat):
    """three disjoint straight roads (no blocks can form) around lon/lat, EPSG:4326 layer features"""
    import geopandas as gpd
    from shapely.geometry import LineString
    d = 0.002
    gs = [LineString([(lon - d, lat), (lon + d, lat)]), LineString([(lon - d, lat + d), (lon - d / 4, lat + 2 * d)]),
          LineString([(lon + d, lat - d), (lon + 2 * d, lat - d)])]
    gpd.GeoDataFrame({"class": ["village_road"] * 3}, geometry=gs, crs=4326).to_file(path, layer="features", driver="GPKG")


# ============================================================ pytest side (stdlib only)
def clay(*args, timeout=900):
    r = subprocess.run([CLAY, __file__, *map(str, args)], capture_output=True, text=True, timeout=timeout)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else None


def need_data():
    import pytest
    for p in (CLAY, VEC, PROB, BLD, STATEWIDE, "/usr/bin/gdalbuildvrt"):
        if not os.path.exists(p):
            pytest.skip("test data/env missing: %s" % p)


def aoi_box(lon0, lat0, lon1, lat1):
    return {"type": "Polygon", "coordinates": [[[lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}


def make_request(wd, aoi, roads, prob, buildings=None, saltscape=None, params=None):
    wd = Path(wd); wd.mkdir(parents=True, exist_ok=True)
    req = {"layer_key": "plotted", "aoi": aoi, "work_dir": str(wd), "out_gpkg": str(wd / "out.gpkg"),
           "result_json": str(wd / "result.json"),
           "imagery": {},
           "inputs": {"village_roads_gpkg": str(roads), "village_roads_prob": list(prob),
                      **({"buildings_gpkg": str(buildings)} if buildings else {}),
                      **({"saltscape_gpkg": str(saltscape)} if saltscape else {})},
           "weights": {}, "gpu": None,
           "params": {"ideal": 1, "salt_threshold": 0.60, "prob_accept": 0.20, "roads_overlap_m": 3000, **(params or {})}}
    p = wd / "request.json"; p.write_text(json.dumps(req)); return p


def run_plugin(req_path, timeout=1800):
    env = dict(os.environ, OMP_NUM_THREADS="1", http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9",
               HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9", GDAL_HTTP_PROXY="127.0.0.1:9")
    r = subprocess.run([CLAY, str(PLUGIN), "--request", str(req_path)], capture_output=True, text=True, timeout=timeout, env=env)
    assert r.returncode == 0, "plugin failed:\n" + r.stdout[-2000:] + "\n" + r.stderr[-4000:]
    prog = [l for l in r.stdout.splitlines() if l.startswith("PROGRESS ")]
    assert prog and prog[-1].startswith("PROGRESS 1.0000"), r.stdout[-2000:]
    req = json.loads(Path(req_path).read_text())
    return json.loads(Path(req["result_json"]).read_text()), clay("summary", req["out_gpkg"]), r.stdout


_FIX = {}


def fixture_clip(tmp_root):
    if "clip" not in _FIX:
        d = Path(tmp_root) / "fixture"
        info = clay("fixture", d, *CLIP)
        _FIX["clip"] = (d, info)
    return _FIX["clip"]


def check_layers(s):
    assert set(s) >= {"features", "plots", "removed_saltworks"}
    f = s["features"]
    assert f["crs"] == 4326 and f["geom"] == "MultiPolygon"
    for c in ("class", "confidence", "name", "src_id", "n_plots", "area_ha", "built_pct", "built_class", "vacant_plots_pct"):
        assert c in f["cols"], c
    assert s["plots"]["crs"] == 4326
    for c in ("plot_id", "cluster_id", "area_m2", "built_pct", "built_class"):
        assert c in s["plots"]["cols"], c


def test_check_mode(tmp_path_factory):
    need_data()
    r = subprocess.run([CLAY, str(PLUGIN), "--check"], capture_output=True, text=True, timeout=120)
    rep = json.loads(r.stdout); assert r.returncode == 0 and rep["ok"], r.stdout
    tmp = tmp_path_factory.mktemp("check")
    d, info = fixture_clip(tmp_path_factory.getbasetemp())
    rq = make_request(tmp / "wd", aoi_box(*CLIP), d / "roads.gpkg", info["prob"], d / "buildings.gpkg")
    r = subprocess.run([CLAY, str(PLUGIN), "--check", "--request", str(rq)], capture_output=True, text=True, timeout=300)
    rep = json.loads(r.stdout); assert r.returncode == 0 and rep["ok"], r.stdout
    names = {c["name"]: c for c in rep["checks"]}
    assert names["village_roads_prob"]["ok"] and not names["saltscape_gpkg"]["ok"]
    assert not (tmp / "wd" / "out.gpkg").exists()                      # no processing
    # a missing roads file and a non-3857 prob raster must fail the check (synthetic 4326 raster)
    wc = tmp / "wc.tif"; clay("wc", wc, *CLIP, 40)
    rq = make_request(tmp / "wd2", aoi_box(*CLIP), tmp / "nope.gpkg", [str(wc)])
    r = subprocess.run([CLAY, str(PLUGIN), "--check", "--request", str(rq)], capture_output=True, text=True, timeout=300)
    rep = json.loads(r.stdout); names = {c["name"]: c for c in rep["checks"]}
    assert r.returncode == 1 and not rep["ok"] and not names["village_roads_gpkg"]["ok"] and not names["village_roads_prob"]["ok"]


def test_clip_matches_statewide(tmp_path_factory):
    need_data()
    d, info = fixture_clip(tmp_path_factory.getbasetemp())
    assert info["roads"] > 100 and info["prob"]
    ref = clay("statewide", *CLIP)
    rq = make_request(tmp_path_factory.mktemp("clip") / "wd", aoi_box(*CLIP), d / "roads.gpkg", info["prob"], d / "buildings.gpkg")
    t = time.time(); res, s, _ = run_plugin(rq); dt = time.time() - t
    check_layers(s)
    n = s["features"]["n"]
    print("plugin %d clusters / %d plots vs statewide %d / %d (%.0f s)" % (n, s["features"]["n_plots"], ref["clusters"], ref["plots"], dt))
    assert res["status"] == "ok" and res["model"] == "plotted-land-geometry" and res["model_version"] == "guj chord IDEAL=1"
    assert res["image_source"] == "derived from village roads" and res["coverage"]["type"] in ("Polygon", "MultiPolygon")
    assert res["counts"]["features"] == n and any("salt-works filter skipped" in w for w in res["warnings"])
    assert abs(n - ref["clusters"]) <= max(2, 0.3 * ref["clusters"]), (n, ref)
    assert s["features"]["classes"] == ["plotted_land"] and s["features"]["src_ids"][0].startswith("PLC-")
    assert s["features"]["built_null"] == 0


def test_saltfilter_synthetic_saltscape(tmp_path_factory):
    """salt pan over the whole area -> every ring is salt-scape -> all clusters move to removed_saltworks"""
    need_data()
    d, info = fixture_clip(tmp_path_factory.getbasetemp())
    tmp = tmp_path_factory.mktemp("salt")
    lon0, lat0, lon1, lat1 = CLIP
    sg = tmp / "saltscape.gpkg"; clay("salt", sg, lon0 - .03, lat0 - .03, lon1 + .03, lat1 + .03)
    rq = make_request(tmp / "wd", aoi_box(*CLIP), d / "roads.gpkg", info["prob"], d / "buildings.gpkg", saltscape=sg)
    res, s, _ = run_plugin(rq)
    check_layers(s)
    assert s["features"]["n"] == 0 and s["plots"]["n"] == 0 and s["removed_saltworks"]["n"] > 0
    assert not any("salt-works filter skipped" in w for w in res["warnings"])


def test_no_roads_aoi(tmp_path_factory):
    """AOI offshore (no roads, no prob tile) and AOI with only a few disjoint lines: valid empty layers, no crash"""
    need_data()
    d, info = fixture_clip(tmp_path_factory.getbasetemp())
    tmp = tmp_path_factory.mktemp("empty")
    rq = make_request(tmp / "sea", aoi_box(70.30, 20.30, 70.32, 20.32), d / "roads.gpkg", info["prob"], d / "buildings.gpkg")
    res, s, _ = run_plugin(rq)
    check_layers(s)
    assert s["features"]["n"] == 0 and s["plots"]["n"] == 0 and res["coverage"] is None and res["status"] == "ok"
    # lines that form no block, over a real prob tile: the full pipeline runs on empties
    lon0, lat0, lon1, lat1 = CLIP
    lp = tmp / "lines.gpkg"; clay("lines", lp, (lon0 + lon1) / 2, (lat0 + lat1) / 2)
    rq = make_request(tmp / "lines", aoi_box(*CLIP), lp, info["prob"])
    res, s, out = run_plugin(rq)
    check_layers(s)
    assert s["features"]["n"] == 0 and res["counts"]["roads_edges"] == 3 and "rectangles 0" in out
    assert any("no buildings layer" in w for w in res["warnings"])


def test_sigterm_exits_promptly(tmp_path_factory):
    need_data()
    d, info = fixture_clip(tmp_path_factory.getbasetemp())
    rq = make_request(tmp_path_factory.mktemp("term") / "wd", aoi_box(*CLIP), d / "roads.gpkg", info["prob"], d / "buildings.gpkg")
    p = subprocess.Popen([CLAY, str(PLUGIN), "--request", str(rq)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         start_new_session=True)
    for line in p.stdout:                                  # wait until the pipeline is under way
        if line.startswith("PROGRESS") and float(line.split()[1]) >= 0.10:
            break
    t = time.time(); os.killpg(p.pid, signal.SIGTERM); p.wait(timeout=60)
    assert p.returncode != 0 and time.time() - t < 30


if __name__ == "__main__":
    cmd, a = sys.argv[1], sys.argv[2:]
    if cmd == "fixture": _fixture(a[0], *map(float, a[1:5]))
    elif cmd == "statewide": _statewide(*map(float, a[:4]))
    elif cmd == "summary": _summary(a[0])
    elif cmd == "wc": _wc(a[0], *map(float, a[1:5]), int(a[5]))
    elif cmd == "salt": _salt(a[0], *map(float, a[1:5]))
    elif cmd == "lines": _lines(a[0], float(a[1]), float(a[2]))
    else: raise SystemExit("unknown command " + cmd)
