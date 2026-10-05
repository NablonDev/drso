"""SOLAR plugin tests: CPU post-processing on real Bhuj-Lakadia research outputs + synthetic S2.
No network, no GPU (model inference is not exercised; the end-to-end test feeds the research
probability raster through `run_aoi.py --prob`).

pytest lives in the beeline env, the plugin runs in the terratorch env:
    /home/vivek_jain/miniforge3/envs/beeline/bin/python -m pytest -q tests/plugins/test_solar.py
Tests that need the research files skip when they are absent (e.g. on the prod machine).
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
PLUG = REPO / "lulc_models" / "solar"
RUN = PLUG / "run_aoi.py"
PY = os.environ.get("SOLAR_TEST_PY", "/home/vivek_jain/miniforge3/envs/terratorch/bin/python")

D = Path("/mnt/drso-data/solar_lora/data")
PROB = D / "corridor_solar_prob.tif"
DET = D / "corridor_solar_filtered.gpkg"
CLASSIFIED = D / "corridor_solar_classified.gpkg"
SEAS = D / "FINAL_solar_seasonal.gpkg"
V2 = D / "FINAL_solar_v2.gpkg"
OSM = D / "osm_ref.gpkg"
OSM_LAYER = "osm_solar_arrays_aoi"
MBT = Path("/mnt/drso-data/wayback/bhuj_lakadia_L18_2026-08-05.mbtiles")
BUCKET = REPO / "data" / "bucket" / "models"
CKPT = BUCKET / "solar" / "epoch07.ckpt"
HF = BUCKET / "hf_home"
# 3014 x 3014 px window (col, row, w, h) of the research prob raster: 32 detections fully inside
WIN = (113013, 76023, 3014, 3014)

need = lambda *ps: pytest.mark.skipif(not all(Path(p).exists() for p in ps),  # noqa: E731
                                      reason="research files absent: " + ", ".join(map(str, ps)))
need_py = pytest.mark.skipif(not Path(PY).exists(), reason="terratorch env absent")


def _load(name):
    spec = importlib.util.spec_from_file_location("solarplug_" + name, PLUG / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


post = _load("solar_post")
s2 = _load("solar_s2")


# ------------------------------------------------------------------------- 08b polygonise
@need(PROB, DET)
def test_polygonize_window_matches_research_detections():
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window, bounds
    from shapely.geometry import box

    win = Window(*WIN)
    with rasterio.open(PROB) as ds:
        wb = box(*bounds(win, ds.transform))
    g4096 = post.polygonize_prob(PROB, 153, 5000.0, window=win, block=4096)
    g700 = post.polygonize_prob(PROB, 153, 5000.0, window=win, block=700)   # many seams
    # seams must not change the answer
    assert len(g700) == len(g4096)
    assert np.allclose(sorted(g700.merc_area_m2), sorted(g4096.merc_area_m2))
    assert g4096.is_valid.all()

    ref = gpd.read_file(DET, layer="detections")
    ref = ref[ref.within(wb)]
    ours = g4096[g4096.within(wb.buffer(-1.0))]            # drop pieces cut by the window edge
    assert len(ref) == 32 and len(ours) == len(ref)
    assert sorted(np.round(ours.merc_area_m2, 1)) == sorted(np.round(ref.merc_area_m2, 1))
    # geometry identical, not just areas (pair by area, compare shapes)
    o = ours.assign(k=np.round(ours.merc_area_m2, 1)).set_index("k").geometry
    r = ref.assign(k=np.round(ref.merc_area_m2, 1)).set_index("k").geometry
    sym = sum(o.loc[k].symmetric_difference(r.loc[k]).area for k in o.index)
    assert sym < 1.0


# ------------------------------------------------------------------------- 09 classify
@need(DET, OSM, CLASSIFIED)
def test_classify_against_osm_reproduces_research():
    import geopandas as gpd
    det = gpd.read_file(DET, layer="detections")
    osm = post.read_osm_solar(OSM, OSM_LAYER)
    g = post.classify_osm(det, osm, 5000.0)
    assert int(g["known"].sum()) == 51
    assert int((~g["known"]).sum()) == 529
    ref = gpd.read_file(CLASSIFIED, layer="DET_known_OSM")
    assert np.allclose(sorted(g.loc[g.known, "area_m2"]), sorted(ref["area_m2"]), atol=0.5)
    # absent OSM -> everything new
    assert not post.classify_osm(det, None, 5000.0)["known"].any()


@need(OSM)
def test_read_osm_layer_autoselect():
    # osm_ref has several layers; without a name every '*solar*' layer is used (plants + arrays)
    a = post.read_osm_solar(OSM)
    b = post.read_osm_solar(OSM, OSM_LAYER)
    assert len(a) == 22 and len(b) == 11 and str(a.crs).endswith("4326")


# ------------------------------------------------------------------------- 16 frozen band
def _seasonal_input():
    import geopandas as gpd
    import pandas as pd
    parts = []
    for lay, known in (("SOLAR_confirmed", True), ("SOLAR_new_verified", False),
                       ("SOLAR_rejected", False), ("SOLAR_excluded_below_20m", False)):
        g = gpd.read_file(SEAS, layer=lay)
        g["known"] = known
        g["research_layer"] = lay
        parts.append(g)
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    g["resolvable_20m"] = g["eroded_m2"] >= 4 * 400.0
    return g


@need(SEAS, V2)
def test_final_frozen_band_reproduces_final_v2():
    import geopandas as gpd
    g = _seasonal_input()
    g["cls"] = post.final_classes(g, True, (1159, 2351), 4, True)
    counts = g["cls"].value_counts().to_dict()
    want = {lay: len(gpd.read_file(V2, layer=lay)) for lay in
            ("SOLAR_confirmed", "SOLAR_new_verified", "SOLAR_rejected", "SOLAR_excluded_below_20m")}
    assert want == {"SOLAR_confirmed": 49, "SOLAR_new_verified": 65, "SOLAR_rejected": 365,
                    "SOLAR_excluded_below_20m": 99}
    assert counts == {"confirmed_OSM": 49, "new_verified_NIR": 65, "rejected_NIR": 365,
                      "excluded_below_20m": 99}
    # the same polygons, not just the same numbers
    ver = gpd.read_file(V2, layer="SOLAR_new_verified")
    assert sorted(np.round(g.loc[g.cls == "new_verified_NIR", "area_m2"], 2)) == \
        sorted(np.round(ver["area_m2"], 2))
    kept = g[g.cls.isin(post.KEEP)]
    assert len(kept) == len(gpd.read_file(V2, layer="SOLAR_FINAL_all")) == 114


@need(SEAS)
def test_final_without_s2_and_gate_switch():
    g = _seasonal_input()
    c = post.final_classes(g, False)
    assert set(c[g.known]) == {"confirmed_OSM"} and set(c[~g.known]) == {"new_unverified"}
    # a known polygon that fails the 20 m gate: dropped by default, kept with gate_known=False
    g2 = g.copy()
    i = g2.index[g2.known][0]
    g2.loc[i, "resolvable_20m"] = False
    assert post.final_classes(g2, True).loc[i] == "excluded_below_20m"
    assert post.final_classes(g2, True, gate_known=False).loc[i] == "confirmed_OSM"
    # resolvable new polygon without enough S2 observations -> unverified, not rejected
    j = g2.index[~g2.known & g2.resolvable_20m][0]
    g2.loc[j, "n_obs"] = 2
    assert post.final_classes(g2, True).loc[j] == "new_unverified"


# ------------------------------------------------------------------------- 14 S2 features
UTM = "EPSG:32642"
X0, Y0 = 600000.0, 2580000.0          # ~70.0 E, 23.3 N


def _band(path, value, res, inside, outside):
    import rasterio
    from rasterio.transform import from_origin
    n = int(1000 / res)
    a = np.full((n, n), outside, np.uint16)
    lo, hi = int(400 / res), int(600 / res)                 # 200 m square in the middle
    a[lo:hi, lo:hi] = inside if value is None else value
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="uint16",
                       crs=UTM, transform=from_origin(X0, Y0 + 1000, res, res), nodata=0) as d:
        d.write(a, 1)


def _scene(tmp, sid, date, cloud, vals, offset=0, broken=None, **extra):
    bands = {}
    for b, (v, res, out) in {"B04": (vals[0], 10, 2500), "B08": (vals[1], 10, 3000),
                             "B11": (vals[2], 20, 3500), "B12": (vals[3], 20, 3000)}.items():
        p = tmp / f"{sid}_{b}.tif"
        if b != broken:
            _band(p, None, res, v + offset, out + offset)
        bands[b] = str(p)
    return {"scene_id": sid, "date": date, "cloud": cloud, "bands": bands, **extra}


def _square_gdf():
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"known": [False, False]},
                         geometry=[box(X0 + 400, Y0 + 400, X0 + 600, Y0 + 600),      # 200 m panel
                                   box(X0 + 5000, Y0 + 5000, X0 + 5200, Y0 + 5200)],  # off-raster
                         crs=UTM).to_crs(4326)
    return g


def test_s2_features_from_local_bands(tmp_path):
    vals = (1000, 1800, 2000, 1600)
    items = [_scene(tmp_path, f"S2B_42QXL_2025{m:02d}15_0_L2A", f"2025-{m:02d}-15", 5.0, vals)
             for m in (1, 2, 3, 4, 5)]
    # same month + tile, cloudier: must be ignored in favour of the 5 % scene
    items.append(_scene(tmp_path, "S2A_42QXL_20250120_0_L2A", "2025-01-20", 12.0, (9000, 9000, 9000, 9000)))
    # over max_cloud: ignored
    items.append(_scene(tmp_path, "S2A_42QXL_20250620_0_L2A", "2025-06-20", 35.0, (9000, 9000, 9000, 9000)))
    # raw DN with the +1000 BOA offset still in: corrected back to the same values
    items.append(_scene(tmp_path, "S2A_42QXL_20250710_0_L2A", "2025-07-10", 3.0, vals, offset=1000,
                        boa_offset_applied=False))
    # selected (lowest cloud in its month) but a band file is missing: warning, scene skipped
    items.append(_scene(tmp_path, "S2A_42QXL_20250810_0_L2A", "2025-08-10", 1.0, vals, broken="B12"))
    # missing a band key entirely: warning at selection
    bad = _scene(tmp_path, "S2A_42QXL_20250910_0_L2A", "2025-09-10", 1.0, vals)
    del bad["bands"]["B11"]
    items.append(bad)

    g, used, warns = s2.seasonal_features(_square_gdf(), items, max_cloud=20.0)
    assert len(used) == 7                                    # months 1-5, 7, 8
    assert {u["date"] for u in used if u["date"].startswith("2025-01")} == {"2025-01-15"}
    r = g.iloc[0]
    assert r.n_obs == 6
    assert r.nir_med == pytest.approx(1800)
    assert r.ndvi_med == pytest.approx((1800 - 1000) / 2800)
    assert r.ndvi_std == pytest.approx(0.0)
    assert r.swir_ratio_med == pytest.approx(2000 / 1600)
    assert g.iloc[1].n_obs == 0 and np.isnan(g.iloc[1].nir_med)
    assert any("20250810" in w and "cannot open" in w for w in warns), warns
    assert any("20250910" in w and "missing bands" in w for w in warns), warns
    # and it passes the frozen band
    g["resolvable_20m"] = True
    assert post.final_classes(g, True).iloc[0] == "new_verified_NIR"


def test_resolvability_gate():
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 120, 120), box(1000, 0, 1040, 40), box(2000, 0, 2400, 30)],
                         crs="EPSG:7755")
    ok, area = s2.resolvable(g, 4, 20.0)
    assert list(ok) == [True, False, False]                 # 6x6 -> 4x4 ; 2x2 -> 0 ; 20x1 -> 0
    assert area[0] == pytest.approx(16 * 400)


def test_tile_key():
    assert s2.tile_key({"scene_id": "S2C_42QWM_20260909_0_L2A"}) == "42QWM"
    assert s2.tile_key({"scene_id": "x", "grid_code": "MGRS-43QBF"}) == "43QBF"
    assert s2.tile_key({"scene_id": "S2C_MSIL2A_20260909T055631_N0512_R091_T42QWM_20260909T084715"}) == "42QWM"


# ------------------------------------------------------------------------- coverage
@need(MBT)
def test_mbtiles_footprint_matches_tiles():
    import sqlite3
    fp = post.mbtiles_footprint_3857(str(MBT))
    con = sqlite3.connect(f"file:{MBT}?mode=ro", uri=True)
    n = con.execute("select count(*) from tiles where zoom_level=18").fetchone()[0]
    con.close()
    tile_m2 = (2 * post.R_MERC / 2 ** 18) ** 2
    assert fp.area == pytest.approx(n * tile_m2, rel=1e-9)
    sub = post.mbtiles_footprint_3857(str(MBT), (7812924, 2667356, 7814724, 2669156))
    assert 0 < sub.area < 0.01 * fp.area


# ------------------------------------------------------------------------- subprocess (terratorch env)
def _aoi_from_window(win):
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window, bounds
    from shapely.geometry import box, mapping
    with rasterio.open(PROB) as ds:
        b = bounds(Window(*win), ds.transform)
    return mapping(gpd.GeoSeries([box(*b)], crs=3857).to_crs(4326).iloc[0])


def _request(tmp, aoi, s2_items=None, osm=True, ckpt=CKPT, mbt=MBT):
    req = {"layer_key": "solar", "aoi": aoi, "work_dir": str(tmp), "out_gpkg": str(tmp / "out.gpkg"),
           "result_json": str(tmp / "result.json"),
           "imagery": {"esri_wayback_z18": [{"path": str(mbt), "release": "26334", "date": "2026-08-05"}],
                       "sentinel2_l2a": s2_items or []},
           "inputs": {"osm_solar_gpkg": str(OSM)} if osm else {},
           "weights": {"checkpoint": str(ckpt), "hf_home": str(HF)},
           "gpu": {"device": 0, "vram_mb": 14000},
           "params": {"threshold": 0.60, "min_area_m2": 5000, "nir_band": [1159, 2351], "min_obs": 4,
                      "min_pure_px": 4, "erode_m": 20, "osm_layer": OSM_LAYER}}
    p = tmp / "request.json"
    p.write_text(json.dumps(req))
    return p


def _env():
    # no network for the plugin: dead proxies, as the worker does
    e = dict(os.environ, HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
             http_proxy="http://127.0.0.1:9", https_proxy="http://127.0.0.1:9",
             GDAL_HTTP_PROXY="127.0.0.1:9", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
             CUDA_VISIBLE_DEVICES="")
    return e


@need_py
@need(MBT, PROB, CKPT, HF)
def test_check_mode(tmp_path):
    items = [_scene(tmp_path, "S2B_42QXL_20250115_0_L2A", "2025-01-15", 5.0, (1000, 1800, 2000, 1600))]
    p = _request(tmp_path, _aoi_from_window(WIN), s2_items=items)
    r = subprocess.run([PY, str(RUN), "--check", "--request", str(p)], capture_output=True, text=True,
                       env=_env(), timeout=300)
    rep = json.loads(r.stdout)
    assert r.returncode == 0 and rep["ok"], json.dumps(rep, indent=1)
    c = rep["checks"]
    assert c["custom_modules"]["ok"] and c["gdal:MBTiles_driver"]["ok"] and c["hf_cache"]["ok"]
    assert c["mbtiles[0]"]["driver"] == "MBTiles" and c["mbtiles[0]"]["bands"] == 4
    assert 2.0 < rep["aoi_overlap_km2"] < 4.0              # ~1.8 km square in Mercator ~ 2.7 km2 ground
    assert c["sentinel2_l2a"]["ok"] and c["osm_solar_gpkg"]["features"] == 11
    assert "gpu" not in c                                   # GPU is only probed with --check-gpu

    # broken inputs are reported, not raised
    p2 = _request(tmp_path, _aoi_from_window(WIN), ckpt=tmp_path / "nope.ckpt", mbt=tmp_path / "nope.mbtiles")
    r2 = subprocess.run([PY, str(RUN), "--check", "--request", str(p2)], capture_output=True, text=True,
                        env=_env(), timeout=300)
    rep2 = json.loads(r2.stdout)
    assert r2.returncode == 1 and not rep2["ok"]
    assert not rep2["checks"]["checkpoint"]["ok"] and not rep2["checks"]["mbtiles[0]"]["ok"]


@pytest.fixture(scope="module")
def known_window():
    """A window around an OSM-mapped site (classified research layer)."""
    if not (CLASSIFIED.exists() and PROB.exists()):
        pytest.skip("research files absent")
    import geopandas as gpd
    import rasterio
    from rasterio.windows import from_bounds
    k = gpd.read_file(CLASSIFIED, layer="DET_known_OSM").to_crs(3857)
    k = k.assign(a=k.area).sort_values("a")
    g = k.geometry.iloc[len(k) // 2]
    x0, y0, x1, y1 = g.buffer(300).bounds
    with rasterio.open(PROB) as ds:
        w = from_bounds(x0, y0, x1, y1, transform=ds.transform).round_offsets().round_lengths()
    return (int(w.col_off), int(w.row_off), int(w.width), int(w.height)), g


@need_py
@need(MBT, PROB, OSM)
def test_end_to_end_prob_mode(tmp_path, known_window):
    import geopandas as gpd
    win, site = known_window
    p = _request(tmp_path, _aoi_from_window(win))
    r = subprocess.run([PY, str(RUN), "--request", str(p), "--prob", str(PROB)], capture_output=True,
                       text=True, env=_env(), timeout=600)
    assert r.returncode == 0, r.stderr
    assert "PROGRESS 1.0000" in r.stdout
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["status"] == "ok" and res["model"] == "solar-dinov3-lora" and res["model_version"] == "v1 epoch07"
    assert res["image_source"] == "Esri Wayback z18 release 26334"       # no S2 supplied
    assert res["image_dates"] == ["2026-08-05"]
    assert any("Sentinel-2" in w for w in res["warnings"])
    assert res["counts"]["confirmed_OSM"] >= 1
    assert Path(res["artifacts"]["prob_raster"][0]).exists()
    f = gpd.read_file(tmp_path / "out.gpkg", layer="features")
    assert str(f.crs).endswith("4326") and len(f) == res["counts"]["features"]
    assert set(f["class"]) == {"solar_park"}
    assert set(f["confidence"]) <= {1.0, 0.5}
    assert f["src_id"].is_unique and f["name"].isna().all()
    for c in ("cls", "area_m2", "n_obs", "nir_med", "ndvi_med"):
        assert c in f.columns
    from shapely.geometry import shape
    aoi = shape(json.loads(p.read_text())["aoi"])
    assert f.geometry.within(aoi.buffer(1e-9)).all()
    assert shape(res["coverage"]).area == pytest.approx(aoi.area, rel=0.02)
    # the OSM site of the window is confirmed with confidence 1.0
    site4326 = gpd.GeoSeries([site], crs=3857).to_crs(4326).iloc[0]
    hit = f[f.intersects(site4326)]
    assert len(hit) and (hit["confidence"] == 1.0).all()


@need_py
@need(MBT, PROB)
def test_no_imagery_is_empty_ok(tmp_path):
    from shapely.geometry import box, mapping
    p = _request(tmp_path, mapping(box(80.0, 20.0, 80.01, 20.01)))   # far outside the MBTiles
    r = subprocess.run([PY, str(RUN), "--request", str(p)], capture_output=True, text=True,
                       env=_env(), timeout=300)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["counts"]["features"] == 0 and res["coverage"]["geometries"] == []
    import geopandas as gpd
    assert len(gpd.read_file(tmp_path / "out.gpkg", layer="features")) == 0


@need_py
@need(MBT, PROB)
def test_sigterm_exits_promptly(tmp_path):
    # large AOI so the run is still busy when the signal arrives
    big = (WIN[0] - 12000, WIN[1] - 12000, 30000, 30000)
    p = _request(tmp_path, _aoi_from_window(big), osm=False)
    proc = subprocess.Popen([PY, str(RUN), "--request", str(p), "--prob", str(PROB)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=_env(), start_new_session=True)
    t0 = time.time()
    for line in proc.stdout:
        if line.startswith("PROGRESS"):
            break
    time.sleep(1.0)
    os.killpg(proc.pid, signal.SIGTERM)
    rc = proc.wait(timeout=60)
    assert rc != 0 and time.time() - t0 < 90
    assert not (tmp_path / "result.json").exists()


def _flat_scene(tmp, sid, date, aoi, vals):
    """Constant-valued band files covering the AOI (UTM 42N, 10/20 m)."""
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    x0, y0, x1, y1 = gpd.GeoSeries([shape(aoi)], crs=4326).to_crs(UTM).total_bounds
    bands = {}
    for b, v, res in (("B04", vals[0], 10), ("B08", vals[1], 10), ("B11", vals[2], 20), ("B12", vals[3], 20)):
        w, h = int((x1 - x0) / res) + 20, int((y1 - y0) / res) + 20
        p = tmp / f"{sid}_{b}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint16", crs=UTM,
                           transform=from_origin(x0 - 10 * res, y1 + 10 * res, res, res), nodata=0) as d:
            d.write(np.full((h, w), v, np.uint16), 1)
        bands[b] = str(p)
    return {"scene_id": sid, "date": date, "cloud": 2.0, "bands": bands}


@need_py
@need(MBT, PROB, OSM)
def test_end_to_end_with_local_s2(tmp_path):
    import geopandas as gpd
    aoi = _aoi_from_window(WIN)                             # 37 detections, none on OSM
    # NIR 1800 everywhere: inside the frozen band -> every resolvable new detection verified
    items = [_flat_scene(tmp_path, f"S2B_42QXL_2026{m:02d}05_0_L2A", f"2026-{m:02d}-05", aoi,
                         (1000, 1800, 2000, 1600)) for m in (1, 2, 3, 4, 5)]
    p = _request(tmp_path, aoi, s2_items=items)
    r = subprocess.run([PY, str(RUN), "--request", str(p), "--prob", str(PROB)], capture_output=True,
                       text=True, env=_env(), timeout=600)
    assert r.returncode == 0, r.stderr
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["image_source"] == "Esri Wayback z18 release 26334 + Sentinel-2 L2A"
    assert res["image_dates"] == ["2026-01-05", "2026-02-05", "2026-03-05", "2026-04-05", "2026-05-05",
                                  "2026-08-05"]
    c = res["counts"]
    assert c["new_verified_NIR"] >= 20 and c["new_unverified"] == 0 and c["rejected_NIR"] == 0
    assert c["features"] == c["confirmed_OSM"] + c["new_verified_NIR"]
    f = gpd.read_file(tmp_path / "out.gpkg", layer="features")
    assert set(f["confidence"]) <= {1.0, 0.8}
    assert (f["n_obs"] == 5).all() and np.allclose(f["nir_med"], 1800)

    # NIR 3000: outside the band -> new ones rejected, OSM-confirmed kept
    items = [_flat_scene(tmp_path, f"S2B_42QXL_2026{m:02d}06_0_L2A", f"2026-{m:02d}-06", aoi,
                         (1000, 3000, 2000, 1600)) for m in (1, 2, 3, 4, 5)]
    p = _request(tmp_path, aoi, s2_items=items)
    r = subprocess.run([PY, str(RUN), "--request", str(p), "--prob", str(PROB)], capture_output=True,
                       text=True, env=_env(), timeout=600)
    assert r.returncode == 0, r.stderr
    c2 = json.loads((tmp_path / "result.json").read_text())["counts"]
    assert c2["new_verified_NIR"] == 0 and c2["rejected_NIR"] == c["new_verified_NIR"]
    assert c2["features"] == c2["confirmed_OSM"] == c["confirmed_OSM"]


@need(MBT)
def test_inference_windowing_with_dummy_model(tmp_path, monkeypatch):
    """Block/chip/AOI logic of solar_infer.run on the real MBTiles with a stand-in model on CPU
    (the real network is not run: no GPU in tests)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch = pytest.importorskip("torch")
    import geopandas as gpd
    import rasterio
    from rasterio.features import geometry_mask
    from shapely.geometry import Point
    infer = _load("solar_infer")

    class Dummy(torch.nn.Module):          # class-1 logit grows with the normalised red channel
        def forward(self, x):
            return torch.cat([torch.zeros_like(x[:, :1]), x[:, :1]], 1)

    aoi = gpd.GeoSeries([Point(70.19, 23.30).buffer(0.004)], crs=4326).to_crs(3857).iloc[0]
    out = tmp_path / "p.tif"
    st = infer.run(str(MBT), aoi, str(out), Dummy(), device="cpu", block=1024, tile=512, overlap=128,
                   batch_size=4)
    assert st["blocks_run"] >= 1
    with rasterio.open(out) as ds, rasterio.open(MBT) as src:
        a = ds.read(1)
        assert (ds.width, ds.height) == tuple(st["window"][2:])
        assert ds.transform == src.window_transform(rasterio.windows.Window(*st["window"]))
        outside = geometry_mask([aoi], out_shape=a.shape, transform=ds.transform)
        assert a[outside].max() == 0 and a[~outside].mean() > 10
        # the stand-in prob is a pure per-pixel function, so overlap averaging must return it exactly
        rgb = src.read(1, window=rasterio.windows.Window(*st["window"])).astype(np.float32)
    p = 1 / (1 + np.exp(-(rgb - 123.675) / 58.395))
    expect = (p * 255).astype(np.uint8)
    inside = ~outside
    assert np.abs(expect[inside].astype(int) - a[inside].astype(int)).max() <= 1
    # disjoint AOI -> nothing written
    far = gpd.GeoSeries([Point(80, 20).buffer(0.01)], crs=4326).to_crs(3857).iloc[0]
    assert infer.run(str(MBT), far, str(tmp_path / "q.tif"), Dummy(), device="cpu") == {}
