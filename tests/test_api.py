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

from beeline_api import app as appmod, aviation, cell_towers, db, imagery, jobs, osm_highways, osm_power, solvejob, worker  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "overpass_power_bhuj.json")
HW_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "overpass_highways_bhuj.json")   # 5 Oct 2026
LA_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "livingatlas_airports_kutch.json")  # 4 airports
OV_AERO = os.path.join(os.path.dirname(__file__), "fixtures", "overpass_aero_kutch.json")      # aerodromes + runways
AP_GPKG = os.path.join(os.path.dirname(__file__), "fixtures", "airports_kutch.gpkg")          # OSM aerodromes + runways
TS_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "ts_towers_bhuj.parquet")   # 871 sites, Bhuj–Bhachau

A = {"lat": 23.456596, "lon": 69.591253, "name": "Bhujpool"}   # gantry moved 9 Oct 2026
B = {"lat": 23.3932, "lon": 70.5955, "name": "Shivlaka"}
SHORT_B = {"lat": 23.40, "lon": 69.95, "name": "Midway"}
IMG = {"esri": {"covered_pct": 100.0, "newest": "2026-03-03", "sources": ["Vantor LG03"]},
       "sentinel2": {"newest": "2026-10-01", "covered_pct": 100.0, "scenes": []}, "dem": {}, "covered": True}


@pytest.fixture(scope="module")
def client():
    imagery.check = lambda frame, band: IMG          # no network in tests
    solvejob.settings.DEM_DIR = None                 # no DEM fetch
    osm_power._overpass = lambda q, timeout=300: json.load(open(FIXTURE))   # recorded Overpass answer
    osm_highways._overpass = lambda q, timeout=300: json.load(open(HW_FIXTURE))
    with TestClient(appmod.app) as c:
        osm_power.refresh_aoi((69.50, 23.25, 70.10, 23.60))                  # seed the OSM power store
        osm_highways.refresh_aoi((69.45, 23.15, 70.70, 23.70))               # seed the OSM highway store
        cell_towers.load_india(TS_FIXTURE)                                   # seed the cell tower store
        aviation._get = lambda params: json.load(open(LA_FIXTURE))            # recorded Living Atlas answer
        aviation._overpass = lambda q, timeout=180: json.load(open(OV_AERO))["runway" if '"runway"' in q else "aerodrome"]
        aviation.load_airports(AP_GPKG)                                      # seed the airport store
        aviation.load_runways(AP_GPKG)                                       # seed the runway store
        yield c


def run_next_job():
    job = worker.claim()
    assert job is not None, "no queued job"
    worker.run_job(job)
    return jobs.get(job["id"])


def make_run(client, b=SHORT_B, band=5.0, kv=765) -> str:
    r = client.post("/api/runs", json={"name": "Test corridor", "a": A, "b": b, "kv": kv, "circuit": "dc", "band_km": band, "note": "test"})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------------------------------------------------------------------------------
def test_validation_errors(client):
    r = client.post("/api/runs", json={"name": "Test corridor", "a": A, "b": B, "kv": 500, "circuit": "dc", "band_km": 5})
    assert r.status_code == 422
    r = client.post("/api/runs", json={"name": "   ", "a": A, "b": B, "kv": 765, "band_km": 25})
    assert r.status_code == 422                                   # corridor name is mandatory
    r = client.post("/api/runs", json={"name": "Bhuj – Lakadia", "a": A, "b": SHORT_B, "kv": 765, "band_km": 25})
    assert r.status_code == 201 and r.json()["id"].startswith("BHU-LAK-") and r.json()["circuit"] == "dc"
    assert r.json()["name"] == "Bhuj – Lakadia"
    assert client.delete(f"/api/runs/{r.json()['id']}").status_code == 200
    r = client.post("/api/runs", json={"name": "Test corridor", "a": A, "b": A, "kv": 765, "circuit": "dc", "band_km": 5})
    assert r.status_code == 400 and "1 km" in r.json()["detail"]
    r = client.post("/api/runs", json={"name": "Test corridor", "a": A, "b": B, "kv": 765, "circuit": "xx", "band_km": 5})
    assert r.status_code == 422
    r = client.post("/api/runs", json={"name": "Test corridor", "a": {"lat": 95, "lon": 70}, "b": B, "kv": 765, "circuit": "dc", "band_km": 5})
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
    # buildings: model layer served from the stored corridor output; lulc hidden (6 Oct 2026)
    assert keys["buildings"]["model"] == "corridor footprint delivery"
    assert keys["buildings"]["stats"]["lulc"]["served_from"] in ("stored", "stored_partial")   # test band leaves the corridor
    assert "lulc" not in keys and "power_lt" in keys and "reserve_forest" in keys
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
    assert ft["layer_key"] == "buildings" and ft["model"] == "corridor footprint delivery" and ft["chainage_km"]
    # solve. The test corridor crosses the Kachchh Desert Sanctuary; national parks and sanctuaries
    # are crossable with approval since 8 Oct 2026 (Rule_Table 69-72).
    assert client.post(f"/api/runs/{rid}/solve").status_code == 202
    j = run_next_job()
    assert j["status"] == "done", j["error"]
    s = client.get(f"/api/runs/{rid}/solve").json()
    assert s["feasible"] and len(s["result"]["routes"]) == 4, (s["result"].get("blocked") or {}).get("event")
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
    ov2 = next(o for o in man2["overrides"] if o["reason"] == "client call")
    assert ov2["rev"] == 2


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
    assert rules["n_rules"] == 48          # 8 Oct 2026 (+ approach section): + state highway, lake, pond, defence, monuments, salt pans (land + water), wetland
    r = client.put(f"/api/runs/{rid}/costs", json={"tower_price": {"A": "abc"}})
    assert r.status_code == 400
    r = client.put(f"/api/runs/{rid}/costs", json={"tower_price": {"B2": 1}})
    assert r.status_code == 400                       # B2 / C2 dropped
    r = client.put(f"/api/runs/{rid}/costs", json={"land": {"land_cr_ha": -1}})
    assert r.status_code == 400
    r = client.put(f"/api/runs/{rid}/costs", json={"land": {"agri_land_cr_ha": 1}})
    assert r.status_code == 400                       # single land rate now
    c0 = client.get(f"/api/runs/{rid}/costs").json()
    assert [c["key"] for c in c0["categories"]] == ["land", "forest", "line", "foundation", "buildings", "tower"]
    assert [t["type"] for t in c0["towers"]] == ["A", "B1", "C1", "D45", "D60"]
    c2 = client.put(f"/api/runs/{rid}/costs", json={"tower_price": {"A": 0.5}}).json()
    assert c2["towers"][0]["price_cr"] == 0.5 and c2["towers"][0]["changed"]
    c3 = client.put(f"/api/runs/{rid}/costs", json={"tower_price": {"A": c0["towers"][0]["default_price_cr"]}}).json()
    assert not c3["towers"][0]["changed"]
    # 400 kV resolves its own values
    r4 = make_run(client, kv=400)
    t4 = client.get(f"/api/runs/{r4}/costs").json()["towers"][0]["price_cr"]
    assert t4 != c0["towers"][0]["price_cr"]
    assert client.post("/api/runs", json={"name": "sc", "a": {"lat": 23, "lon": 70}, "b": {"lat": 23.1, "lon": 70.1},
                                          "kv": 400, "circuit": "sc", "band_km": 2}).status_code == 422


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


def test_voltage_parsing():
    p = osm_power.parse_voltages
    assert p("400000") == [400] and p("400000;220000") == [400, 220] and p("220 kV") == [220]
    assert p("66000") == [66] and p("33 kV") == [33] and p("230000") == [] and p(None) == [] and p("11000;132000") == [132]


def test_osm_power_refresh_from_ui(client):
    rid = make_run(client)
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    L = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}
    p = L["power"]
    assert p["status"] == "done" and p["feature_count"] > 0 and p["image_source"] == "OpenStreetMap"
    assert set(p["stats"]["by_voltage"]) <= {f"{k}_{v}" for k in ("line", "tower") for v in (765, 400, 220, 132)}
    # refresh: one extraction job for the power layer only; refused while anything runs
    j = client.post(f"/api/runs/{rid}/layers/power/refresh")
    assert j.status_code == 202, j.text
    assert client.post(f"/api/runs/{rid}/layers/power/refresh").status_code == 409
    assert client.post(f"/api/runs/{rid}/solve").status_code == 409
    done = run_next_job()
    assert done["status"] == "done" and list(done["detail"]["result"]) == ["power"]
    p2 = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}["power"]
    assert p2["layer_version_id"] != p["layer_version_id"] and "refreshed for this area" in p2["source"]
    assert p2["stats"]["refresh"]["load_id"]
    st = client.get("/api/osm/power").json()
    assert st["loads"][0]["run_id"] == rid and st["loads"][0]["kind"] == "aoi"
    # voltage reaches the map: tile property and feature popup
    import mapbox_vector_tile, mercantile
    found = None
    for lon in (69.62, 69.66, 69.70, 69.75, 69.80, 69.85):
        tl = mercantile.tile(lon, 23.43, 11)
        dec = mapbox_vector_tile.decode(client.get(f"/api/tiles/{rid}/{tl.z}/{tl.x}/{tl.y}.pbf").content)
        if "power" in dec and dec["power"]["features"]:
            found = dec["power"]["features"][0]
            break
    assert found and found["properties"]["voltage_kv"] in (765, 400, 220, 132)
    ft = client.get(f"/api/runs/{rid}/features/{found['properties']['id']}").json()
    assert ft["attrs"]["voltage_kv"] == found["properties"]["voltage_kv"]
    # a layer with no online source cannot be refreshed
    assert client.post(f"/api/runs/{rid}/layers/rail/refresh").status_code == 400


def test_cell_towers_layer_and_refresh(client):
    rid = make_run(client)
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    resp = client.get(f"/api/runs/{rid}/layers").json()
    t = {l["key"]: l for l in resp["layers"]}["telecom"]
    assert t["status"] == "done" and t["feature_count"] > 0 and t["image_source"] == "Tarang Sanchar"
    assert set(t["stats"]["by_structure"]) <= {"Ground", "Roof", "Wall"}
    assert sum(t["stats"]["by_structure"].values()) == t["feature_count"]
    assert "Cell tower" in t["rules"] and t["refresh_label"] == "Tarang Sanchar"
    # no machine source yet: the button is shown disabled, the endpoint refuses and starts nothing
    assert t["refreshable"] and t["refresh_available"] is False and "OTP" in t["refresh_note"]
    r = client.post(f"/api/runs/{rid}/layers/telecom/refresh")
    assert r.status_code == 503 and "not available" in r.json()["detail"]
    assert client.get("/api/jobs/active").json()["job"] is None
    st = client.get("/api/telecom/towers").json()
    assert st["refresh_available"] is False and st["loads"][0]["kind"] == "india"
    # popup carries the Tarang Sanchar attributes
    fid = db.one("""select f.id from features f join run_layers rl on rl.layer_version_id=f.layer_version_id
                    where rl.run_id=%s and rl.layer_key='telecom' limit 1""", (rid,))["id"]
    ft = client.get(f"/api/runs/{rid}/features/{fid}").json()
    assert ft["class"] == "cell_tower" and ft["attrs"]["structure"] in ("Ground", "Roof", "Wall") and ft["attrs"]["site_id"]

    # once a source is plugged in, the refresh path is the same as OSM power
    from shapely.geometry import shape
    c = shape(resp["band"]).centroid
    fetched = [{"site_id": "T-1", "structure": "ground", "lon": c.x, "lat": c.y, "operator": "X"},
               {"site_id": "T-2", "structure": "Roof", "lon": c.x + 0.001, "lat": c.y},
               {"site_id": "FAR", "structure": "Ground", "lon": 80.0, "lat": 20.0}]          # outside: dropped
    cell_towers.FETCHER = lambda bbox: (fetched, None)
    try:
        j = client.post(f"/api/runs/{rid}/layers/telecom/refresh")
        assert j.status_code == 202, j.text
        assert client.post(f"/api/runs/{rid}/layers/telecom/refresh").status_code == 409
        assert client.post(f"/api/runs/{rid}/solve").status_code == 409
        done = run_next_job()
        assert done["status"] == "done" and list(done["detail"]["result"]) == ["telecom"]
        t2 = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}["telecom"]
        assert t2["layer_version_id"] != t["layer_version_id"] and "refreshed for this area" in t2["source"]
        assert t2["feature_count"] == 2 and t2["stats"]["by_structure"] == {"Ground": 1, "Roof": 1}
        assert t2["stats"]["refresh"]["replaced"] >= t["feature_count"]
        st = client.get("/api/telecom/towers").json()
        assert st["loads"][0]["run_id"] == rid and st["loads"][0]["kind"] == "aoi" and st["loads"][0]["status"] == "done"
    finally:
        cell_towers.FETCHER = None
        cell_towers.load_india(TS_FIXTURE)        # restore the store for later tests


def test_highway_classification():
    c = osm_highways.classify
    assert c({"highway": "trunk", "ref": "NH27"}, [("nh", "NH")]) == [("nh", "NH 27", False)]
    assert c({"highway": "primary", "ref": "NH341;SH45"}) == [("nh", "NH 341", False), ("sh", "SH 45", False)]
    assert c({"highway": "trunk"}) == [("nh", None, True)]                      # untagged trunk = NH, inferred
    assert c({"highway": "primary"}) == [] and c({"highway": "trunk_link", "ref": "NH27"}) == []
    rel = osm_highways.relation_class({"route": "road", "network": "IN:NH", "ref": "341"})
    assert rel == ("nh", "NH 341") and c({"highway": "secondary"}, [rel]) == [("nh", "NH 341", False)]
    assert osm_highways.relation_class({"route": "road", "network": "IN:SH:GJ", "ref": "SH__"}) == ("sh", "SH")
    assert osm_highways.parse_refs("NE4") == {"nh": ["NE 4"]} and osm_highways.parse_refs("MDR 12") == {}


def test_osm_highways_refresh_from_ui(client):
    rid = make_run(client)
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    L = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}
    h = L["roads"]
    assert h["name"].startswith("Highways") and h["refresh_label"] == "OSM" and h["refresh_available"]
    assert h["status"] == "done" and h["feature_count"] > 0 and h["image_source"] == "OpenStreetMap"
    assert set(h["stats"]["by_class"]) <= {"nh", "sh", "nh_inferred"} and h["stats"]["by_class"].get("nh")
    assert "National highway" in h["rules"]
    j = client.post(f"/api/runs/{rid}/layers/roads/refresh")
    assert j.status_code == 202, j.text
    assert client.post(f"/api/runs/{rid}/layers/roads/refresh").status_code == 409
    assert client.post(f"/api/runs/{rid}/solve").status_code == 409
    done = run_next_job()
    assert done["status"] == "done" and list(done["detail"]["result"]) == ["roads"]
    h2 = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}["roads"]
    assert h2["layer_version_id"] != h["layer_version_id"] and "refreshed for this area" in h2["source"]
    assert h2["stats"]["refresh"]["load_id"] and h2["stats"]["refresh"]["replaced"] > 0
    st = client.get("/api/osm/highways").json()
    assert st["loads"][0]["run_id"] == rid and st["loads"][0]["kind"] == "aoi"
    # the popup names the highway
    fid = db.one("""select f.id from features f join run_layers rl on rl.layer_version_id=f.layer_version_id
                    where rl.run_id=%s and rl.layer_key='roads' and f.class='nh' and f.attrs->>'ref' is not null
                    limit 1""", (rid,))["id"]
    ft = client.get(f"/api/runs/{rid}/features/{fid}").json()
    assert ft["attrs"]["ref"].startswith(("NH ", "NE ")) and ft["attrs"]["road_class"] == "nh"


def test_overpass_retries_busy_mirror_before_moving_on(monkeypatch):
    calls = []

    class R:
        def __init__(self, code):
            self.status_code = code

        def json(self):
            return {"elements": []}

    answers = iter([504, 504, 200])
    monkeypatch.setattr(osm_power.httpx, "post", lambda url, **kw: calls.append(url) or R(next(answers)))
    monkeypatch.setattr(osm_power.time, "sleep", lambda s: None)
    assert osm_power.overpass_query("[out:json];") == {"elements": []}
    assert calls == [osm_power.OVERPASS_URLS[0]] * 3          # stayed on the first mirror


def test_runway_polygons_and_airport_filter():
    import geopandas as gpd
    from shapely.geometry import LineString, box as sbox
    g = gpd.GeoDataFrame({
        "osm_type": ["way"] * 4, "osm_id": [1, 2, 3, 4], "name": [None] * 4,
        "all_tags": ['{"aeroway": "runway", "width": "30", "ref": "05/23"}', '{"aeroway": "runway"}',
                     '{"aeroway": "runway", "disused": "yes"}', '{"aeroway": "runway"}']},
        geometry=[LineString([(69.60, 23.30), (69.62, 23.30)])] * 3 + [sbox(69.60, 23.30, 69.61, 23.3004)], crs=4326)
    rw = aviation.runway_polygons(g).set_index("src_id")
    assert list(rw.index) == ["way/1", "way/2", "way/4"]                      # disused dropped
    assert rw.loc["way/1", "width_source"] == "width tag" and rw.loc["way/2", "width_source"] == "default 45 m"
    assert rw.loc["way/4", "width_source"] == "polygon"
    a = rw.to_crs(32642).area
    length = 0.02 * 111320 * 0.9164                                          # 0.02° of longitude at 23.3° N
    assert abs(a["way/1"] / length - 30) < 1 and abs(a["way/2"] / length - 45) < 1.5
    assert aviation.parse_width("50m") == 50 and aviation.parse_width("1000") is None
    assert not aviation._is_airport("Khavda Abandoned Airstrip", '{"disused": "yes"}')
    assert not aviation._is_airport("Naharlagun Heliport", "{}") and aviation._is_airport("Bhuj Rudra Mata Air Force Base", "{}")


def test_aviation_layers_and_refresh_from_ui(client):
    st = client.get("/api/aviation").json()
    srcs = {r["source"]: r["n"] for r in st["airports"]}
    assert srcs["esri"] == 4 and srcs["osm"] >= 1                            # OSM air force stations not in Esri
    rid = make_run(client)
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    L = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}
    ap, rw = L["aviation"], L["approach"]
    assert ap["name"] == "Airports" and ap["refresh_label"] == "Living Atlas + OSM" and ap["refresh_available"]
    assert ap["status"] == "done" and ap["image_source"] == "Esri Living Atlas"
    # Bhuj airport is ~18 km south of A: outside the band, inside the NOC radius, so it is read whole
    assert ap["feature_count"] >= 1 and ap["stats"]["reach_m"] >= 20000
    assert "Airport — NOC radius" in ap["rules"]
    assert rw["name"] == "Airport approach path" and rw["refreshable"] and rw["refresh_label"] == "OSM"
    assert rw["status"] == "done"
    assert "Airport — no-go buffer and approach funnel" in rw["rules"] and rw["stats"]["reach_m"] >= 10000
    names = [r["name"] for r in db.all_("""select f.name from features f join run_layers rl on rl.layer_version_id=f.layer_version_id
                                           where rl.run_id=%s and rl.layer_key='aviation'""", (rid,))]
    assert "Bhuj Airport" in names
    # refresh: Living Atlas + OSM for the band grown by the NOC reach, one job, refused while busy
    j = client.post(f"/api/runs/{rid}/layers/aviation/refresh")
    assert j.status_code == 202, j.text
    assert client.post(f"/api/runs/{rid}/layers/aviation/refresh").status_code == 409
    assert client.post(f"/api/runs/{rid}/solve").status_code == 409
    done = run_next_job()
    assert done["status"] == "done" and list(done["detail"]["result"]) == ["aviation"]
    ap2 = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}["aviation"]
    assert ap2["layer_version_id"] != ap["layer_version_id"] and "refreshed for this area" in ap2["source"]
    ref = ap2["stats"]["refresh"]
    # Kutch: 4 Esri; OSM has 6 aerodromes — Khavda (disused) filtered, 4 overlap Esri, Naliya AFS kept
    assert ref["esri"] == 4 and ref["osm_fetched"] == 5 and ref["osm_not_in_esri"] == 1
    assert client.get("/api/aviation").json()["loads"][0]["run_id"] == rid
    osm_names = [r["name"] for r in db.all_("select name from aviation_airports where source='osm'")]
    assert "Naliya Air Force Station" in osm_names and "Khavda Abandoned Airstrip" not in osm_names
    # runways: their own refresh, from OSM
    j = client.post(f"/api/runs/{rid}/layers/approach/refresh")
    assert j.status_code == 202, j.text
    done = run_next_job()
    assert done["status"] == "done" and list(done["detail"]["result"]) == ["approach"]
    rw2 = {l["key"]: l for l in client.get(f"/api/runs/{rid}/layers").json()["layers"]}["approach"]
    assert rw2["layer_version_id"] != rw["layer_version_id"] and "refreshed for this area" in rw2["source"]
    assert rw2["stats"]["refresh"]["read"] == 11 and rw2["stats"]["refresh"]["kept"] == 10   # one disused
    assert client.get("/api/aviation").json()["loads"][0]["layer"] == "runways"
    assert db.one("select count(*) as n from aviation_runways")["n"] == 10                # replaced, not duplicated
    fid = db.one("""select f.id from features f join run_layers rl on rl.layer_version_id=f.layer_version_id
                    where rl.run_id=%s and rl.layer_key='aviation' and f.name='Bhuj Airport'""", (rid,))["id"]
    ft = client.get(f"/api/runs/{rid}/features/{fid}").json()
    assert ft["attrs"]["rundes"] == "05/23" and ft["attrs"]["source"] == "Esri Living Atlas"


def test_runways_read_beyond_the_band():
    from shapely.geometry import box as sbox
    near_bhuj = sbox(69.60, 23.31, 69.70, 23.33)                              # ~2 km north of the Bhuj runway
    assert len(aviation.read_runways(near_bhuj, 0)[0]) == 0
    g, m = aviation.read_runways(near_bhuj, 3000)
    assert len(g) >= 1 and m["india_loaded"]
    assert not g.geometry.iloc[0].within(near_bhuj)                         # whole runway, not clipped to the band


def test_normalise_flattens_z_and_keeps_fields():
    import geopandas as gpd
    import pandas as pd
    from beeline_api import reference
    from shapely.geometry import LineString
    g = gpd.GeoDataFrame({"name": ["MDPL", None], "uid": ["V-1", "V-2"], "owner": ["HPCL", "GAIL"], "junk": [1, 2]},
                         geometry=[LineString([(70.0, 23.4, 5.0), (70.1, 23.4, 6.0)]),
                                   LineString([(70.0, 23.5, 0.0), (70.1, 23.5, 0.0)])], crs=4326)
    n = reference.normalise(g, "pipelines", const_class="oil_pipeline", name_field="name", id_field="uid",
                            keep_fields=["owner"])
    assert not n.has_z.any() and list(n["class"]) == ["oil_pipeline"] * 2
    assert n["name"].iloc[0] == "MDPL" and pd.isna(n["name"].iloc[1])         # missing, not the string "nan"
    assert list(n["owner"]) == ["HPCL", "GAIL"] and "junk" not in n.columns


def test_hand_added_shapes(client, tmp_path):
    """Shapes drawn or loaded on the map (Vivek, 8 Oct 2026): KML / KMZ parse; temporary shapes join
    only their run's layer; permanent shapes also reach later runs; delete takes them out again."""
    rid = make_run(client)
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    kml = ("<?xml version='1.0' encoding='UTF-8'?><kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
           "<Placemark><name>Farm shed</name><Polygon><outerBoundaryIs><LinearRing><coordinates>"
           "69.700,23.430 69.701,23.430 69.701,23.431 69.700,23.431 69.700,23.430</coordinates></LinearRing>"
           "</outerBoundaryIs></Polygon></Placemark>"
           "<Placemark><name>Pole</name><Point><coordinates>69.702,23.432</coordinates></Point></Placemark></Document></kml>")
    fc = client.post("/api/geo/parse", files={"file": ("a.kml", kml.encode(), "application/vnd.google-earth.kml+xml")}).json()
    assert fc["count"] == 2 and fc["geometry_types"] == ["Point", "Polygon"]
    assert {f["properties"]["name"] for f in fc["features"]} == {"Farm shed", "Pole"}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("doc.kml", kml)
    assert client.post("/api/geo/parse", files={"file": ("a.kmz", buf.getvalue(), "application/vnd.google-earth.kmz")}).json()["count"] == 2
    assert client.post("/api/geo/parse", files={"file": ("a.txt", b"x", "text/plain")}).status_code == 400

    poly = [f for f in fc["features"] if f["geometry"]["type"] == "Polygon"]
    point = [f for f in fc["features"] if f["geometry"]["type"] == "Point"]
    # a point cannot go into a polygon layer; an unknown class is refused
    r = client.post(f"/api/runs/{rid}/shapes", json={"layer_key": "buildings", "name": "x", "scope": "temporary", "features": point})
    assert r.status_code == 400 and "polygon" in r.json()["detail"]
    r = client.post(f"/api/runs/{rid}/shapes", json={"layer_key": "buildings", "class": "nope", "name": "x", "scope": "temporary", "features": poly})
    assert r.status_code == 400

    def count(run_id, key):
        return next(l for l in client.get(f"/api/runs/{run_id}/layers").json()["layers"] if l["key"] == key)["feature_count"]

    n0 = count(rid, "buildings")
    r = client.post(f"/api/runs/{rid}/shapes", json={"layer_key": "buildings", "name": "Farm shed", "scope": "temporary", "features": poly})
    assert r.status_code == 201, r.text
    tmp_id = r.json()["added"][0]
    assert count(rid, "buildings") == n0 + 1
    run = client.get(f"/api/runs/{rid}").json()
    assert run["layers_ready"] and not run["layers_stale"]           # the run can be re-solved straight away
    r = client.post(f"/api/runs/{rid}/shapes", json={"layer_key": "reservoir", "name": "New pond", "scope": "permanent", "features": poly})
    assert r.status_code == 201, r.text
    perm_id = r.json()["added"][0]
    shapes = client.get(f"/api/runs/{rid}/shapes").json()["shapes"]
    assert {s["id"]: s["scope"] for s in shapes} == {tmp_id: "temporary", perm_id: "permanent"}

    # a later run over the same ground: the permanent shape comes with its extraction, the temporary one does not
    rid2 = make_run(client)
    client.post(f"/api/runs/{rid2}/extract", json={})
    assert run_next_job()["status"] == "done"
    assert [s["id"] for s in client.get(f"/api/runs/{rid2}/shapes").json()["shapes"]] == [perm_id]
    feats = client.get(f"/api/runs/{rid2}/layers").json()
    assert next(l for l in feats["layers"] if l["key"] == "reservoir")["feature_count"] >= 1

    # delete: the temporary shape leaves its run's layer
    assert client.delete(f"/api/runs/{rid}/shapes/{tmp_id}").status_code == 200
    assert count(rid, "buildings") == n0
    assert client.delete(f"/api/runs/{rid2}/shapes/{tmp_id}").status_code == 404
    assert client.delete(f"/api/runs/{rid}/shapes/{perm_id}").status_code == 200
    assert client.get(f"/api/runs/{rid}/shapes").json()["shapes"] == []


def test_quick_solve_skips_exports_until_asked(client):
    """Quick test solve (9 Oct 2026): routes only; the export set is made later from the same solve."""
    rid = make_run(client)
    client.post(f"/api/runs/{rid}/extract", json={})
    assert run_next_job()["status"] == "done"
    r = client.post(f"/api/runs/{rid}/solve", json={"exports": False})
    assert r.status_code == 202
    j = run_next_job()
    assert j["status"] == "done", j["error"]
    s = client.get(f"/api/runs/{rid}/solve").json()
    assert s["feasible"] and s["manifest"]["exports_skipped"] and s["files"] == []
    assert set(s["manifest"]["layer_version_ids"]) >= {"buildings", "clusters"}
    s2 = client.post(f"/api/solves/{s['id']}/exports").json()
    assert not s2["manifest"]["exports_skipped"]
    for name in ("route1.dxf", "route1.kml", "report.pdf", "manifest.json", "layers.gpkg", f"{rid}_all.zip"):
        assert name in s2["files"], name
    assert client.get(f"/api/solves/{s['id']}/files/report.pdf").content[:4] == b"%PDF"
    assert client.post("/api/solves/999999/exports").status_code == 404
