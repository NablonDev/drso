"""End-to-end API scenario against a running dev stack (API on BEELINE_API, default :8710)."""
import os, sys, time, json
import httpx

API = os.environ.get("BEELINE_API", "http://localhost:8710")
c = httpx.Client(base_url=API, timeout=120)


def wait(job_id, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        j = c.get(f"/api/jobs/{job_id}").json()
        if j["status"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(2)
    raise TimeoutError(job_id)


def ok(r, code=200):
    assert r.status_code == code, f"{r.request.method} {r.request.url} -> {r.status_code} {r.text[:500]}"
    return r.json()


A = {"lat": 23.4525, "lon": 69.5924, "name": "Bhujpool"}
B = {"lat": 23.3932, "lon": 70.5955, "name": "Shivlaka"}
v = ok(c.post("/api/geo/validate", json={"a": A, "b": B, "band_km": 5}))
print("validate", [x["text"] for x in v["checks"]])
band = ok(c.post("/api/geo/band", json={"a": A, "b": B, "band_km": 5}))
print("band", band["area_km2"], "km2, reuse", band["reuse"][:1])
run = ok(c.post("/api/runs", json={"a": A, "b": B, "kv": 765, "circuit": "dc", "band_km": 5, "note": "e2e"}), 201)
rid = run["id"]; print("run", rid)
img = ok(c.post(f"/api/runs/{rid}/imagery/check"))
print("imagery esri", img["esri"].get("covered_pct"), img["esri"].get("newest"), "s2", img["sentinel2"].get("newest"), img["sentinel2"].get("covered_pct"))
j = ok(c.post(f"/api/runs/{rid}/extract", json={}), 202)
busy = c.post(f"/api/runs/{rid}/solve")
assert busy.status_code == 409, busy.text
print("busy guard:", busy.json()["detail"])
busy2 = c.post(f"/api/runs/{rid}/extract", json={})
assert busy2.status_code == 409
j = wait(j["id"]); print("extract", j["status"], j["stage"], (j.get("error") or "")[:300])
assert j["status"] == "done"
L = ok(c.get(f"/api/runs/{rid}/layers"))
print("layers", [(l["key"], l.get("feature_count"), l.get("status")) for l in L["layers"]])
print("flags", [f["text"] for f in L["flags"]][:6])
t = c.get(f"/api/tiles/{rid}/12/2840/1757.pbf"); print("tile", t.status_code, len(t.content))
j = ok(c.post(f"/api/runs/{rid}/solve"), 202)
j = wait(j["id"]); print("solve", j["status"], j["stage"], (j.get("error") or "")[:600])
assert j["status"] == "done"
s = ok(c.get(f"/api/runs/{rid}/solve"))
print("feasible", s["feasible"], "files", s["files"])
for r in s["result"]["routes"]:
    print(" ", r["id"], r["name"], r["stats"]["length_km"], r["stats"]["towers"], round(r["cost"]["total"]/1e7, 1))
f = c.get(f"/api/solves/{s['id']}/files/route1.dxf"); print("dxf", f.status_code, len(f.content))
# copy + rule override + re-solve + compare
cp = ok(c.post(f"/api/runs/{rid}/copy", json={"note": "relax clusters"}), 201)
r2 = cp["id"]; print("copy", r2, cp["status"], cp["rev"])
bad = c.put(f"/api/runs/{r2}/rules/clusters", json={"enabled": True, "params": {"tower_setback_m": 60}})
assert bad.status_code == 400, bad.text
ok(c.put(f"/api/runs/{r2}/rules/clusters", json={"enabled": True, "params": {"tower_setback_m": 60, "conductor_buffer_m": 60}, "reason": "e2e test"}))
ok(c.put(f"/api/runs/{r2}/costs", json={"tower": {"rate_per_tonne_rs": 110000}}))
j = ok(c.post(f"/api/runs/{r2}/solve"), 202); j = wait(j["id"]); print("solve2", j["status"], (j.get("error") or "")[:300])
cmpr = ok(c.get(f"/api/runs/{r2}/compare")); print("compare", json.dumps(cmpr["rows"][0] if cmpr["rows"] else None)[:400], cmpr["changes"])
runs = ok(c.get("/api/runs")); print("runs", [(x["id"], x["status"], x["r1_length_km"]) for x in runs["runs"]][:4])
print("E2E OK")
