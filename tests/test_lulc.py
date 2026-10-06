"""LULC model wiring: runner, offline guard, imagery catalog, stored-output vs model-run paths.

Uses stand-in plugins (no real model, no GPU, no network) on a throw-away database.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import textwrap
import time

import numpy as np
import pytest

ADMIN = os.environ.get("BEELINE_TEST_DB_ADMIN", "postgresql://beeline@/postgres?host=/tmp&port=55432")
TEST_DB = "beeline_test_lulc"
os.environ["BEELINE_DB_URL"] = ADMIN.replace("/postgres?", f"/{TEST_DB}?")

import psycopg  # noqa: E402

with psycopg.connect(ADMIN, autocommit=True) as _c:
    _c.execute(f"drop database if exists {TEST_DB} with (force)")
    _c.execute(f"create database {TEST_DB}")

import geopandas as gpd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from shapely.geometry import LineString, Point, box, mapping  # noqa: E402

from beeline_api import db, imagery_store, jobs, lulc_runner, reference, settings, worker  # noqa: E402

AOI = box(69.80, 23.38, 69.90, 23.43)
PY = sys.executable


@pytest.fixture(scope="module", autouse=True)
def schema():
    db.init_schema()
    yield


def fake_model(tmp_path, body: str, name="fake", imagery=("esri_wayback_z18",), timeout_h=0.01):
    code = tmp_path / name
    code.mkdir(parents=True, exist_ok=True)
    (code / "run_aoi.py").write_text(textwrap.dedent(body))
    cfg = {"shared": {"hf_home": "models/hf_home"},
           "models": {name: {"layer": "solar", "code": str(code), "env": PY, "imagery": list(imagery),
                             "gpu": False, "timeout_h": timeout_h, "params": {"p": 1}}}}
    return cfg


PLUGIN_OK = """
import json, os, sys, urllib.request
sys.path.insert(0, os.environ["PYTHONPATH"])
import common
req = common.load_request()
common.progress(0.5, "half way")
# the worker must have closed every network route
blocked = False
try:
    urllib.request.urlopen("http://example.com", timeout=3)
except Exception:
    blocked = True
import geopandas as gpd
from shapely.geometry import shape
aoi = shape(req["aoi"])
g = gpd.GeoDataFrame({"class": ["solar_park"], "confidence": [0.8], "name": [None], "src_id": ["x1"],
                      "area_m2": [1234.5]}, geometry=[aoi.centroid.buffer(0.001)], crs=4326)
g.to_file(req["out_gpkg"], layer="features", driver="GPKG")
common.write_result(req, coverage=req["aoi"], image_source="test imagery", image_dates=["2026-08-05"],
                    model="fake-model", model_version="t1", counts={"features": 1},
                    warnings=[] if blocked else ["NETWORK WAS REACHABLE"],
                    artifacts={"env": {k: os.environ.get(k) for k in ("HTTPS_PROXY", "HF_HUB_OFFLINE")},
                               "imagery_keys": list(req["imagery"])})
"""


def test_runner_offline_and_outputs(tmp_path, monkeypatch):
    cfg = fake_model(tmp_path, PLUGIN_OK)
    monkeypatch.setattr(lulc_runner, "load_models", lambda: cfg)
    monkeypatch.setattr(settings, "REPO", settings.REPO)
    prog = []
    res = lulc_runner.run("fake", AOI, "TEST-0001", 1, {}, progress=lambda f, m: prog.append((f, m)))
    assert len(res.gdf) == 1 and res.gdf["class"][0] == "solar_park"
    assert res.gdf["attrs"][0]["area_m2"] == 1234.5
    assert res.meta["model"] == "fake-model" and res.meta["image_date"] == "2026-08-05"
    assert "NETWORK WAS REACHABLE" not in res.meta["warnings"]
    assert res.artifacts["env"]["HTTPS_PROXY"] == lulc_runner.DEAD_PROXY and res.artifacts["env"]["HF_HUB_OFFLINE"] == "1"
    assert res.artifacts["imagery_keys"] == ["esri_wayback_z18"]
    assert prog and prog[0][0] == 0.5
    req = json.loads((res.work_dir / "request.json").read_text())
    assert req["params"] == {"p": 1} and "BEELINE_ESRI_TOKEN" not in json.dumps(req)


def test_runner_failure_and_cancel(tmp_path, monkeypatch):
    cfg = fake_model(tmp_path, "import sys; print('boom detail'); sys.exit(3)\n")
    monkeypatch.setattr(lulc_runner, "load_models", lambda: cfg)
    with pytest.raises(lulc_runner.PluginError, match="exited with 3.*boom detail"):
        lulc_runner.run("fake", AOI, "TEST-0001", 2, {})
    cfg = fake_model(tmp_path, "import time\nwhile True: time.sleep(0.2)\n", name="slow")
    monkeypatch.setattr(lulc_runner, "load_models", lambda: cfg)
    t0 = time.time()
    with pytest.raises(jobs.Cancelled):
        lulc_runner.run("slow", AOI, "TEST-0001", 3, {}, cancelled=lambda: time.time() - t0 > 1)
    assert time.time() - t0 < 40
    cfg["models"]["slow"]["timeout_h"] = 2 / 3600
    with pytest.raises(lulc_runner.PluginError, match="timed out"):
        lulc_runner.run("slow", AOI, "TEST-0001", 4, {})


def test_preflight_reports_missing_pieces(monkeypatch):
    m = lulc_runner.model_cfg("solar")
    m = {**m, "python": "/nonexistent/python", "weights_abs": {"checkpoint": "/nonexistent/x.ckpt"}}
    probs = lulc_runner.preflight(m)
    assert any("environment" in p for p in probs) and any("weights" in p for p in probs)
    with pytest.raises(lulc_runner.PluginError):
        monkeypatch.setattr(lulc_runner, "preflight", lambda m: ["environment missing"])
        lulc_runner.run("solar", AOI, "TEST-0001", 5, {})


def _tiny_raster(path, bounds=(69.79, 23.37, 69.91, 23.44)):
    w, s, e, n = bounds
    with rasterio.open(path, "w", driver="GTiff", width=50, height=30, count=1, dtype="uint8", crs="EPSG:4326",
                       transform=from_origin(w, n, (e - w) / 50, (n - s) / 30)) as d:
        d.write(np.ones((1, 30, 50), np.uint8))


def _tiny_mbtiles(path, lon=69.85, lat=23.40):
    import mercantile
    t = mercantile.tile(lon, lat, 18)
    con = sqlite3.connect(path)
    con.executescript("create table metadata (name text, value text); create table tiles (zoom_level integer, "
                      "tile_column integer, tile_row integer, tile_data blob);")
    for dx in range(3):
        for dy in range(3):
            con.execute("insert into tiles values (18,?,?,?)", (t.x + dx, (2 ** 18 - 1) - (t.y + dy), b"jpg"))
    con.commit()
    con.close()


def test_catalog_register_and_query(tmp_path):
    r = tmp_path / "nisar.tif"
    _tiny_raster(r)
    imagery_store.register_raster("nisar_coh", r, key="t-coh")
    mb = tmp_path / "t.mbtiles"
    _tiny_mbtiles(mb)
    key = imagery_store.register_mbtiles(mb, "26334", "2026-08-05")
    got = imagery_store.for_aoi(AOI, products=("esri_wayback_z18", "nisar_coh", "sentinel2_l2a"))
    assert [i["item_key"] for i in got["nisar_coh"]] == ["t-coh"]
    assert got["esri_wayback_z18"][0]["release"] == "26334" and got["esri_wayback_z18"][0]["date"] == "2026-08-05"
    assert got["sentinel2_l2a"] == []
    cov = {c["product"]: c for c in imagery_store.coverage(AOI)}
    assert cov["nisar_coh"]["covered_pct"] == 100.0
    assert 0 < cov["esri_wayback_z18"]["covered_pct"] < 100     # 3x3 z18 tiles inside a big AOI
    far = box(80, 20, 80.1, 20.1)
    assert imagery_store.for_aoi(far)["nisar_coh"] == []
    with pytest.raises(ValueError):
        imagery_store.upsert("not_a_product", "x", AOI, "/x", "registered")


def test_acquire_rejects_unfetchable():
    with pytest.raises(ValueError, match="register local files"):
        imagery_store.acquire(["cartosat3_pan"], AOI, "TEST-0001")


# ---- extraction: stored output / model run / no imagery -------------------------------
@pytest.fixture()
def run_row():
    rid = f"LUL-T{int(time.time() * 1000) % 100000:05d}"
    db.exec_("""insert into runs (id, family, a_lon, a_lat, b_lon, b_lat, kv, circuit, band_km, status)
                values (%s, 'LUL-T', 69.80, 23.40, 69.95, 23.41, 400, 'dc', 1.0, 'draft')""", (rid,))
    return db.one("select * from runs where id=%s", (rid,))


class _Rep:
    def __init__(self, detail=None):
        self.job = {"id": 99, "detail": detail or {}}

    def cancelled(self):
        return False


def test_model_layer_served_from_stored_output(tmp_path, monkeypatch, run_row):
    from beeline_api import extraction, geo
    monkeypatch.setattr(settings, "REFERENCE_DIR", tmp_path / "ref")
    band, _ = geo.band_polygon((run_row["a_lon"], run_row["a_lat"]), (run_row["b_lon"], run_row["b_lat"]), 1.0, 500)
    g = gpd.GeoDataFrame({"class": ["solar_park"], "name": [None], "confidence": [1.0], "src_id": ["s1"],
                          "area_m2": [5000.0]}, geometry=[band.centroid.buffer(0.002)], crs=4326)
    reference.write_version("solar", g, "v-test", {"source": "stored test", "model": "solar-dinov3-lora",
                                                   "source_date": "2026-08-05"}, coverage=band.buffer(0.05))
    lay = {"key": "solar", "name": "Solar parks", "pipeline": "solar", "desc": "x", "default_confidence": 0.85}
    called = []
    monkeypatch.setattr(lulc_runner, "run", lambda *a, **k: called.append(1))
    gdf, cov, meta = extraction._lulc_model_layer(run_row, "solar", lay, band, lambda f: None, _Rep())
    assert len(gdf) == 1 and gdf["attrs"][0]["area_m2"] == 5000.0 and not called
    assert meta["source"].startswith("Stored output") and meta["lulc_stats"]["served_from"] == "stored"
    # forced: the model runs (stand-in), with catalog imagery present
    imagery_store.upsert("esri_wayback_z18", "force-test", band.buffer(0.05), "/x.mbtiles", "registered")

    class R:
        gdf = g.assign(attrs=[{"area_m2": 1.0}])[["class", "name", "confidence", "src_id", "attrs", "geometry"]]
        coverage = band
        meta = {"model": "m", "model_version": "v", "image_source": "i", "image_date": "2026-01-01", "warnings": [],
                "counts": {}}
        artifacts = {}
        work_dir = tmp_path
    monkeypatch.setattr(lulc_runner, "run", lambda *a, **k: R)
    monkeypatch.setattr(settings, "RUN_MODELS", True)        # dev.env switches model runs off
    gdf, cov, meta = extraction._lulc_model_layer(run_row, "solar", lay, band, lambda f: None,
                                                  _Rep({"force_models": True}))
    assert meta["lulc_stats"]["served_from"] == "model" and meta["model"] == "m"


def test_model_layer_without_imagery_is_not_run(tmp_path, monkeypatch, run_row):
    from beeline_api import extraction
    monkeypatch.setattr(settings, "REFERENCE_DIR", tmp_path / "ref2")
    band = box(85.0, 20.0, 85.1, 20.05)      # nothing stored, nothing in the catalog
    lay = {"key": "wind_turbines", "name": "Wind turbines", "pipeline": "wind_turbines", "desc": "x"}
    monkeypatch.setattr(lulc_runner, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not run")))
    gdf, cov, meta = extraction._lulc_model_layer(run_row, "wind_turbines", lay, band, lambda f: None, _Rep())
    assert len(gdf) == 0 and cov is None and meta["lulc_stats"]["served_from"] == "none"
    assert "cartosat3_pan" in meta["lulc_stats"]["warnings"][0]


def test_imagery_fetch_job_runs_acquirer(monkeypatch, run_row):
    from beeline_api import extraction
    seen = {}
    monkeypatch.setitem(imagery_store.ACQUIRERS, "sentinel1_rtc_vv",
                        lambda aoi, rid, progress=None, cancel=None: seen.update(rid=rid, area=aoi.area) or {"tiles": ["t"]})
    j = jobs.start(run_row["id"], "extract", {"imagery_fetch": ["sentinel1_rtc_vv"]})
    claimed = worker.claim()
    assert claimed["id"] == j["id"]
    worker.run_job(claimed)
    done = jobs.get(j["id"])
    assert done["status"] == "done", done["error"]
    assert seen["rid"] == run_row["id"] and done["detail"]["result"]["imagery"]["sentinel1_rtc_vv"] == {"tiles": ["t"]}


def test_models_yaml_points_at_installed_weights():
    cfg = lulc_runner.load_models()
    assert set(cfg["models"]) == {"buildings", "village_roads", "plotted_land", "solar", "substations", "wind_turbines"}
    for name in cfg["models"]:
        m = lulc_runner.model_cfg(name)
        for p in m["weights_abs"].values():
            assert os.path.exists(p), f"{name}: {p} not installed (scripts/install_models.py)"
