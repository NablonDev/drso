"""API integration tests on a throw-away database. Jobs run in-process through the worker code.

Needs: Postgres reachable via BEELINE_TEST_DB_ADMIN (default: dev cluster on /tmp:55432), the dev
reference store (BEELINE_REFERENCE_DIR), and a GPU or CPU torch. No network: imagery and DEM are
stubbed out.
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import pytest

ADMIN = os.environ.get("BEELINE_TEST_DB_ADMIN", "postgresql://beeline@/postgres?host=/tmp&port=55432")
TEST_DB = "beeline_test"
os.environ["BEELINE_DB_URL"] = ADMIN.replace("/postgres?", f"/{TEST_DB}?")

import psycopg  # noqa: E402

with psycopg.connect(ADMIN, autocommit=True) as _c:
    _c.execute(f"drop database if exists {TEST_DB} with (force)")
    _c.execute(f"create database {TEST_DB}")

from fastapi.testclient import TestClient  # noqa: E402

from beeline_api import app as appmod, db, imagery, jobs, solvejob, worker  # noqa: E402

A = {"lat": 23.4525, "lon": 69.5924, "name": "Bhujpool"}
B = {"lat": 23.3932, "lon": 70.5955, "name": "Shivlaka"}
SHORT_B = {"lat": 23.40, "lon": 69.95, "name": "Midway"}
IMG = {"esri": {"covered_pct": 100.0, "newest": "2026-03-03", "sources": ["Vantor LG03"]},
       "sentinel2": {"newest": "2026-10-01", "covered_pct": 100.0, "scenes": []}, "dem": {}, "covered": True}


@pytest.fixture(scope="module")
def client():
    imagery.check = lambda frame, band: IMG          # no network in tests
    solvejob.settings.DEM_DIR = None                 # no DEM fetch
    with TestClient(appmod.app) as c:
        yield c


def run_next_job():
    job = worker.claim()
    assert job is not None, "no queued job"
    worker.run_job(job)
    return jobs.get(job["id"])


def make_run(client, b=SHORT_B, band=5.0, kv=765) -> str:
    r = client.post("/api/runs", json={"a": A, "b": b, "kv": kv, "circuit": "dc", "band_km": band, "note": "test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------------------
def test_validation_errors(client):
    r = client.post("/api/runs", json={"a": A, "b": B, "kv": 500, "circuit": "dc", "band_km": 5})
    assert r.status_code == 422
    r = client.post("/api/runs", json={"a": A, "b": A, "kv": 765, "circuit": "dc", "band_km": 5})
    assert r.status_code == 400 and "1 km" in r.json()["detail"]
    r = client.post("/api/runs", json={"a": A, "b": B, "kv": 765, "circuit": "xx", "band_km": 5})
    assert r.status_code == 422
    r = client.post("/api/runs", json={"a": {"lat": 95, "lon": 70}, "b": B, "kv": 765, "circuit": "dc", "band_km": 5})
    assert r.status_code == 422


def test_band_and_validate(client):
    r = client.post("/api/geo/band", json={"a": A, "b": B, "band_km": 5}).json()
    assert 900 < r["area_km2"] < 1100 and r["beeline_km"] > 100


def test_full_flow(client):
    rid = make_run(client)
    assert client.get(f"/api/runs/{rid}").json()["status"] == "draft"
    # solve before extraction is refused with a clear message
    r = client.post(f"/api/runs/{rid}/solve")
    assert r.status_code == 400 and "Extract the layers first" in r.json()["detail"]
    # extraction starts; a second job of either kind is refused (single GPU, nothing queued)
    assert client.post(f"/api/runs/{rid}/extract", json={}).status_code == 202
    r = client.post(f"/api/runs/{rid}/extract", json={})
    assert r.status_code == 409 and r.json()["busy"] and "Nothing is queued" in r.json()["detail"]
    other = make_run(client)
    r = client.post(f"/api/runs/{other}/solve")
    assert r.status_code == 409 and "LULC extraction is running" in r.json()["detail"]
    j = run_next_job()
    assert j["status"] == "done", j["error"]
    run = client.get(f"/api/runs/{rid}").json()
    assert run["status"] == "review" and run["layers_ready"]
    L = client.get(f"/api/runs/{rid}/layers").json()
    keys = {l["key"]: l for l in L["layers"]}
    assert keys["buildings"]["feature_count"] > 0 and keys["clusters"]["feature_count"] > 0
    assert keys["buildings"]["model"] == "buildings_stub" and keys["buildings"]["image_date"] == "2026-03-03"
    assert keys["lulc"]["image_source"] == "Sentinel-2 L2A"
    assert keys["buildings"]["model_run_at"]
    assert any(f["kind"] == "empty" for f in L["flags"])
    # corridor locked once layers exist
    r = client.patch(f"/api/runs/{rid}", json={"band_km": 3})
    assert r.status_code == 409
    # vector tile with features, decodable, one source-layer per layer key
    import mapbox_vector_tile, mercantile
    tl = mercantile.tile(69.70, 23.43, 13)
    t = client.get(f"/api/tiles/{rid}/{tl.z}/{tl.x}/{tl.y}.pbf")
    assert t.status_code == 200 and len(t.content) > 100
    dec = mapbox_vector_tile.decode(t.content)
    assert "buildings" in dec and dec["buildings"]["features"][0]["properties"]["class"] == "building"
    fid = dec["buildings"]["features"][0]["properties"]["id"]
    ft = client.get(f"/api/runs/{rid}/features/{fid}").json()
    assert ft["layer_key"] == "buildings" and ft["model"] == "buildings_stub" and ft["chainage_km"]
    # solve
    assert client.post(f"/api/runs/{rid}/solve").status_code == 202
    j = run_next_job()
    assert j["status"] == "done", j["error"]
    s = client.get(f"/api/runs/{rid}/solve").json()
    assert s["feasible"] and len(s["result"]["routes"]) == 4
    r1 = s["result"]["routes"][0]
    assert r1["towers"][0]["type"] == "G" and r1["towers"][-1]["type"] == "G"
    assert r1["cost"]["total"] > 0 and r1["events"]
    for name in ("route1.dxf", "route1_shp.zip", "route1.kml", "route1_towers.csv", "report.pdf", "manifest.json",
                 f"{rid}_all.zip"):
        assert name in s["files"]
        f = client.get(f"/api/solves/{s['id']}/files/{name}")
        assert f.status_code == 200 and len(f.content) > 100
    assert client.get(f"/api/solves/{s['id']}/files/report.pdf").content[:4] == b"%PDF"
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/solves/{s['id']}/files/{rid}_all.zip").content))
    assert "layers.gpkg" in z.namelist() and "route4.dxf" in z.namelist()
    man = client.get(f"/api/solves/{s['id']}/files/manifest.json").json()
    assert man["run_id"] == rid and len(man["sources"]) >= 16
    assert client.get(f"/api/solves/{s['id']}/files/..%2Fmanifest.json").status_code == 404
    # copy & edit: layers carry over, rule override needs a reason, re-solve, compare
    cp = client.post(f"/api/runs/{rid}/copy", json={"note": "rev 2"}).json()
    rid2 = cp["id"]
    assert cp["rev"] == 2 and cp["status"] == "review" and cp["layers_ready"]
    r = client.put(f"/api/runs/{rid2}/rules/clusters", json={"enabled": True, "params": {"tower_setback_m": 80}})
    assert r.status_code == 400 and "reason" in r.json()["detail"].lower()
    r = client.put(f"/api/runs/{rid2}/rules/clusters", json={"enabled": True, "params": {"tower_setback_m": 80},
                                                              "reason": "client call"})
    assert r.status_code == 200 and r.json()["n_changed"] == 1
    assert client.post(f"/api/runs/{rid2}/solve").status_code == 202
    assert run_next_job()["status"] == "done"
    cmp = client.get(f"/api/runs/{rid2}/compare").json()
    assert cmp["parent"] == rid and cmp["rows"] and any("clusters" in c for c in cmp["changes"])
    man2 = client.get(f"/api/runs/{rid2}/solve").json()["manifest"]
    assert man2["overrides"][0]["reason"] == "client call" and man2["overrides"][0]["rev"] == 2


def test_rules_and_costs_validation(client):
    rid = make_run(client)
    r = client.put(f"/api/runs/{rid}/rules/canal", json={"enabled": True, "params": {"angle_deg": "45°"}, "reason": "x"})
    assert r.status_code == 400 and "must be one of" in r.json()["detail"]
    r = client.put(f"/api/runs/{rid}/rules/railway", json={"enabled": True, "params": {"angle_deg": [110, 70]}, "reason": "x"})
    assert r.status_code == 400
    r = client.put(f"/api/runs/{rid}/rules/nope", json={"enabled": True})
    assert r.status_code == 404
    r = client.put(f"/api/runs/{rid}/rules/canal", json={"enabled": False, "reason": "not on this corridor"})
    assert r.json()["n_off"] == 1
    assert client.delete(f"/api/runs/{rid}/rules").json()["n_off"] == 0
    rules = client.get(f"/api/runs/{rid}/rules").json()
    assert rules["n_rules"] == 30
    r = client.put(f"/api/runs/{rid}/costs", json={"tower": {"rate_per_tonne_rs": "abc"}})
    assert r.status_code == 400
    r = client.put(f"/api/runs/{rid}/costs", json={"land": {"agri_land_cr_ha": -1}})
    assert r.status_code == 400
    c0 = client.get(f"/api/runs/{rid}/costs").json()
    assert len(c0["categories"]) == 8 and not any(c["key"] == "delay" for c in c0["categories"])
    c1 = client.put(f"/api/runs/{rid}/costs", json={"tower": {"rate_per_tonne_rs": 60000}}).json()
    assert abs(c1["tonnage"][0]["cr_each"] * 2 - c0["tonnage"][0]["cr_each"]) <= 2e-3
    c2 = client.put(f"/api/runs/{rid}/costs", json={"tower_tonnage": {"A": 50}}).json()
    assert c2["tonnage"][0]["tonnes"] == 50 and c2["tonnage"][0]["changed"]
    # 400 kV resolves its own values
    r4 = make_run(client, kv=400)
    t4 = client.get(f"/api/runs/{r4}/costs").json()["tonnage"][0]["tonnes"]
    assert t4 != c0["tonnage"][0]["tonnes"]


def test_replace_and_add_sources(client, tmp_path):
    rid = make_run(client)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"KIND": "VIL", "NAME": "My village"},
         "geometry": {"type": "Polygon", "coordinates": [[[69.70, 23.43], [69.71, 23.43], [69.71, 23.44], [69.70, 23.44], [69.70, 23.43]]]}}]}
    p = tmp_path / "mine.geojson"
    p.write_text(json.dumps(fc))
    up = client.post("/api/uploads", files={"file": ("mine.geojson", p.read_bytes(), "application/geo+json")}).json()
    assert up["fields"] == ["KIND", "NAME"] and up["samples"]["KIND"] == ["VIL"]
    chk = client.post(f"/api/uploads/{up['id']}/mapping/check",
                      json={"layer_key": "clusters", "mapping": {"class_field": "KIND"}}).json()
    assert chk["unknown_values"] == ["VIL"]
    chk = client.post(f"/api/uploads/{up['id']}/mapping/check",
                      json={"layer_key": "clusters", "mapping": {"class_field": "KIND", "class_map": {"VIL": "builtup_cluster"}},
                            "remember_as": "my-gis-cell"}).json()
    assert chk["ok"] and not chk["unknown_values"]
    assert client.get("/api/mappings/my-gis-cell").json()["mapping"]["class_map"] == {"VIL": "builtup_cluster"}
    bad = client.put(f"/api/runs/{rid}/sources/clusters", json={"mode": "replace"})
    assert bad.status_code == 400
    ok = client.put(f"/api/runs/{rid}/sources/clusters",
                    json={"mode": "replace", "upload_id": up["id"],
                          "mapping": {"class_field": "KIND", "class_map": {"VIL": "builtup_cluster"}, "name_field": "NAME"}})
    assert ok.status_code == 200
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    L = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}
    assert L["clusters"]["feature_count"] == 1 and L["clusters"]["mode"] == "replace" and L["clusters"]["confidence"] == 1.0
    # switch buildings to "add": only buildings + clusters go stale and get re-extracted
    up2 = client.post("/api/uploads", files={"file": ("b.geojson", json.dumps(fc).encode(), "application/geo+json")}).json()
    client.put(f"/api/runs/{rid}/sources/buildings",
               json={"mode": "add", "upload_id": up2["id"], "mapping": {"const_class": "building"}})
    run = client.get(f"/api/runs/{rid}").json()
    assert set(run["layers_stale"]) == {"buildings", "clusters"} and not run["layers_ready"]
    client.post(f"/api/runs/{rid}/extract", json={})
    j = run_next_job()
    assert set((j["detail"]["result"]).keys()) == {"buildings", "clusters"}
    L = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}
    assert L["buildings"]["mode"] == "add" and "yours wins" in L["buildings"]["source"]


def test_cancel_and_stale_recovery(client):
    rid = make_run(client)
    j = client.post(f"/api/runs/{rid}/extract", json={}).json()
    c = client.post(f"/api/jobs/{j['id']}/cancel").json()
    assert c["status"] == "cancelled"
    assert client.get("/api/jobs/active").json()["job"] is None
    assert client.post(f"/api/jobs/{j['id']}/cancel").status_code == 409
    # an orphaned 'running' job (worker died) is failed on worker start and frees the slot
    db.exec_("insert into jobs (run_id, kind, status, started_at, heartbeat_at) values (%s,'extract','running', now(), now())", (rid,))
    assert client.post(f"/api/runs/{rid}/extract", json={}).status_code == 409
    worker.recover_stale(startup=True)
    assert client.get("/api/jobs/active").json()["job"] is None
    assert client.post(f"/api/runs/{rid}/extract", json={}).status_code == 202
    assert run_next_job()["status"] == "done"


def test_tiles_and_delete(client):
    assert client.get("/api/tiles/esri/19/1/1").status_code == 404     # above native zoom
    rid = make_run(client)
    assert client.delete(f"/api/runs/{rid}").status_code == 200
    assert client.get(f"/api/runs/{rid}").status_code == 404
    assert client.get("/api/runs/NOPE-0000").status_code == 404
