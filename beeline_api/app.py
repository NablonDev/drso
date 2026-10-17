"""Beeline route planner — HTTP API (FastAPI). Serves the React build too when present."""
from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from shapely.geometry import shape

from beeline_engine.config import (CIRCUITS, TOWER_TYPES, VOLTAGES, LineSpec, load_costs, load_layers, load_rules,
                                   pick, resolve_costs)
from beeline_engine.frame import Frame

from . import db, geo, imagery, jobs, layers, settings, uploads, user_shapes

log = logging.getLogger("beeline.api")
app = FastAPI(title="Beeline route planner", version="1.0")
app.add_middleware(GZipMiddleware, minimum_size=2048)


@app.on_event("startup")
def _startup():
    db.init_schema()


@app.exception_handler(jobs.Busy)
def _busy(_: Request, e: jobs.Busy):
    return JSONResponse(status_code=409, content={"detail": str(e), "busy": True, "job": _job_json(e.job)})


# ---- models ---------------------------------------------------------------------------
class Pt(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    name: str | None = None


def _routes_selected(v):
    """Routes to solve (Vivek, 13 Oct 2026): a non-empty subset of the engine's route ids, ordered; route 4
    (second least cost, kept apart from route 1) needs route 1."""
    ids = sorted({int(x) for x in (v or [])})
    if not ids:
        ids = [1]
    bad = [i for i in ids if i < 1 or i > 4]
    if bad:
        raise ValueError(f"unknown route id {bad[0]} (routes are 1–4)")
    if 4 in ids and 1 not in ids:
        raise ValueError("route 4 (second least cost) is solved apart from route 1 — select route 1 as well")
    return ids


class RunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)          # corridor name, shown on every page
    a: Pt
    b: Pt
    kv: int
    circuit: Literal["dc"] = "dc"                            # double circuit only (S/C, Q/C out of scope)
    band_km: float = Field(gt=0.4, le=50)      # Vivek, 13 Oct 2026: up to ± 50 km, default 25
    routes_wanted: int = Field(4, ge=1, le=4)
    routes_selected: list[int] | None = None                 # engine route ids; default route 1 only
    note: str | None = None
    reuse_from: str | None = None

    @field_validator("routes_selected")
    @classmethod
    def _routes(cls, v):
        return _routes_selected(v)

    @field_validator("kv")
    @classmethod
    def _kv(cls, v):
        if v not in VOLTAGES:
            raise ValueError(f"voltage must be one of {VOLTAGES}")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v):
        v = " ".join(v.split())
        if not v:
            raise ValueError("corridor name is required")
        return v


class RunPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=80)
    a: Pt | None = None
    b: Pt | None = None
    kv: int | None = None
    circuit: Literal["dc"] | None = None
    band_km: float | None = Field(None, gt=0.4, le=50)
    routes_wanted: int | None = Field(None, ge=1, le=4)
    routes_selected: list[int] | None = None
    note: str | None = None
    accept_partial: bool | None = None

    @field_validator("routes_selected")
    @classmethod
    def _routes(cls, v):
        return None if v is None else _routes_selected(v)


class SourceSet(BaseModel):
    mode: Literal["auto", "replace", "add"]
    upload_id: int | None = None
    mapping: dict | None = None


class RuleSet(BaseModel):
    enabled: bool = True
    params: dict[str, Any] = {}
    reason: str | None = None


class CopyReq(BaseModel):
    note: str | None = None


class MappingReq(BaseModel):
    layer_key: str
    mapping: dict
    remember_as: str | None = None


class BandReq(BaseModel):
    a: Pt
    b: Pt
    band_km: float = Field(gt=0.4, le=50)      # Vivek, 13 Oct 2026: up to ± 50 km, default 25


class ProcureReq(BaseModel):
    product: str
    charge_to: str | None = None
    note: str | None = None


class ExtractReq(BaseModel):
    layers: list[str] | None = None
    force_models: bool = False      # run the LULC models even where a stored output covers the band


class ImageryFetchReq(BaseModel):
    products: list[str]


# ---- helpers --------------------------------------------------------------------------
def _job_json(j: dict | None) -> dict | None:
    if not j:
        return None
    out = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in j.items()}
    out["eta_text"] = jobs.fmt_eta(j.get("eta_s")) if j.get("eta_s") else None
    return out


def _run(run_id: str) -> dict:
    r = db.one("select * from runs where id=%s", (run_id,))
    if not r:
        raise HTTPException(404, f"run {run_id} not found")
    return r


def _run_json(r: dict) -> dict:
    out = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
    out["a"] = {"lat": r["a_lat"], "lon": r["a_lon"], "name": r["a_name"]}
    out["b"] = {"lat": r["b_lat"], "lon": r["b_lon"], "name": r["b_name"]}
    return out


def _ensure_idle_for(run_id: str):
    a = jobs.active()
    if a and a["run_id"] == run_id:
        raise HTTPException(409, "This run has a job in progress; wait for it to finish or cancel it.")


def _family(name, a_name=None, b_name=None) -> str:
    """Run-id prefix: the first two words of the corridor name ("Bhuj–Lakadia" -> BHU-LAK), else the
    point names, else RUN."""
    def code(n):
        s = "".join(ch for ch in (n or "") if ch.isalpha()).upper()
        return (s[:3] or "XXX").ljust(3, "X")
    words = [w for w in re.split(r"[^A-Za-z]+", name or "") if w]
    if words:
        return f"{code(words[0])}-{code(words[1])}" if len(words) > 1 else code(words[0])
    return f"{code(a_name)}-{code(b_name)}" if (a_name or b_name) else "RUN"


# ---- health / config ------------------------------------------------------------------
@app.get("/api/health")
def health():
    ok_db = True
    try:
        db.one("select 1 as x")
    except Exception:
        ok_db = False
    w = db.one("select max(heartbeat_at) as hb from jobs") if ok_db else None
    du = shutil.disk_usage(settings.DATA)
    return {"ok": ok_db, "db": ok_db, "region": settings.REGION, "reference_dir": str(settings.REFERENCE_DIR),
            "disk_free_gb": round(du.free / 1e9, 1), "last_worker_heartbeat": w and w["hb"] and w["hb"].isoformat()}


@app.get("/api/config")
def config():
    return {"voltages": list(VOLTAGES), "circuits": list(CIRCUITS), "layers": load_layers()["layers"],
            "ruleset_version": load_rules()["version"], "costs_version": load_costs()["version"],
            "esri_max_zoom": settings.ESRI_MAX_ZOOM}


# ---- geo ------------------------------------------------------------------------------
@app.post("/api/geo/validate")
def geo_validate(req: BandReq):
    return geo.validate((req.a.lon, req.a.lat), (req.b.lon, req.b.lat))


@app.post("/api/geo/band")
def geo_band(req: BandReq, exclude: str | None = None):
    try:
        info = geo.band_info((req.a.lon, req.a.lat), (req.b.lon, req.b.lat), req.band_km)
    except ValueError as e:
        raise HTTPException(400, str(e))
    band = shape(info["band"])
    f = Frame((req.a.lon, req.a.lat), (req.b.lon, req.b.lat), req.band_km * 1000, 0, 100)
    reuse = []
    for r in db.all_("""select r.id, r.created_at, r.band_km, r.a_lon, r.a_lat, r.b_lon, r.b_lat, r.note from runs r
                        where exists (select 1 from run_layers rl where rl.run_id=r.id) order by r.created_at desc limit 50"""):
        if r["id"] == exclude:
            continue
        try:
            ob, _ = geo.band_polygon((r["a_lon"], r["a_lat"]), (r["b_lon"], r["b_lat"]), r["band_km"])
        except ValueError:
            continue
        inter = band.intersection(ob)
        if inter.is_empty:
            continue
        shared = geo.area_km2(inter, f)
        if shared < 1:
            continue
        reuse.append({"run_id": r["id"], "created_at": r["created_at"].isoformat(), "band_km": r["band_km"],
                      "shared_km2": round(shared, 1), "shared_pct": round(100 * shared / info["area_km2"], 1),
                      "new_km2": round(info["area_km2"] - shared, 1),
                      "full_cover": shared >= info["area_km2"] * 0.995})
    info["reuse"] = sorted(reuse, key=lambda x: -x["shared_km2"])[:5]
    return info


# ---- runs -----------------------------------------------------------------------------
@app.get("/api/runs")
def list_runs():
    rows = db.all_("""select r.*, s.result->'routes'->0->'stats'->>'length_km' as r1_length_km,
                             s.result->'routes'->0->'cost'->>'total' as r1_cost_rs, s.feasible
                      from runs r left join solves s on s.id = r.latest_solve_id order by r.created_at desc""")
    act = jobs.active()
    out = []
    for r in rows:
        j = _run_json(r)
        src = r["sources"] or {}
        j["custom_sources"] = sum(1 for v in src.values() if v.get("mode") in ("replace", "add"))
        j["job"] = _job_json(act) if act and act["run_id"] == r["id"] else None
        if not j["job"] and r["status"] in ("draft", "review"):
            lj = jobs.latest_for(r["id"])
            if lj and lj["status"] == "failed":
                j["last_error"] = (lj.get("error") or "").split("\n")[0]
        out.append(j)
    return {"runs": out, "active_job": _job_json(act)}


@app.post("/api/runs", status_code=201)
def create_run(req: RunCreate):
    try:
        Frame((req.a.lon, req.a.lat), (req.b.lon, req.b.lat), req.band_km * 1000, 0, 100)
    except ValueError as e:
        raise HTTPException(400, str(e))
    fam = _family(req.name, req.a.name, req.b.name)
    seq = db.one("select nextval('run_seq') as n")["n"]
    rid = f"{fam}-{seq:04d}"
    gv = geo.validate((req.a.lon, req.a.lat), (req.b.lon, req.b.lat))
    sel = req.routes_selected or [1]
    db.exec_("""insert into runs (id, family, rev, name, note, a_lon, a_lat, a_name, b_lon, b_lat, b_name, kv, circuit, band_km,
                                  routes_wanted, routes_selected, status, geo)
                values (%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'draft',%s)""",
             (rid, fam, req.name, req.note, req.a.lon, req.a.lat, req.a.name, req.b.lon, req.b.lat, req.b.name, req.kv,
              req.circuit, req.band_km, len(sel), db.J(sel), db.J(gv)))
    if req.reuse_from:
        _reuse(rid, req.reuse_from)
    return get_run(rid)


def _reuse(rid: str, src_id: str):
    """Carry the layers of an earlier run whose band covers this one."""
    new = _run(rid)
    old = _run(src_id)
    nb, f = geo.band_polygon((new["a_lon"], new["a_lat"]), (new["b_lon"], new["b_lat"]), new["band_km"])
    ob, _ = geo.band_polygon((old["a_lon"], old["a_lat"]), (old["b_lon"], old["b_lat"]), old["band_km"])
    # compared in the new band's metric frame: in degrees the long band edges bend, and a ±5 km band
    # stuck out of a ±25 km band on the same A-B by ~3 ha, so wider runs could never be reused
    if not f.geom_to_utm(ob).buffer(5.0).contains(f.geom_to_utm(nb)):
        raise HTTPException(400, f"{src_id} does not cover this band; only full-cover reuse is supported in v1")
    db.exec_("""insert into run_layers (run_id, layer_key, layer_version_id, carried, stale)
                select %s, layer_key, layer_version_id, true, false from run_layers where run_id=%s
                on conflict do nothing""", (rid, src_id))
    db.exec_("update runs set reuse_from=%s, sources=%s, status='review' where id=%s",
             (src_id, db.J(old["sources"] or {}), rid))


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    r = _run(run_id)
    j = _run_json(r)
    act = jobs.active()
    j["job"] = _job_json(act) if act and act["run_id"] == run_id else None
    j["last_job"] = _job_json(jobs.latest_for(run_id))
    j["last_extract"] = _job_json(jobs.latest_for(run_id, "extract"))
    rl = layers.run_layers(run_id)
    j["layers_ready"] = len(rl) > 0 and not any(x["stale"] for x in rl) and \
        {l["key"] for l in load_layers()["layers"] if not l.get("optional")} <= {x["layer_key"] for x in rl}
    j["layers_failed"] = [x["layer_key"] for x in rl if x["status"] == "failed"]
    j["layers_stale"] = [x["layer_key"] for x in rl if x["stale"]]
    j["parent"] = _run_json(_run(r["parent_id"])) if r["parent_id"] else None
    return j


@app.patch("/api/runs/{run_id}")
def patch_run(run_id: str, req: RunPatch):
    r = _run(run_id)
    _ensure_idle_for(run_id)
    corridor = any(v is not None for v in (req.a, req.b, req.kv, req.circuit, req.band_km))
    has_layers = db.one("select count(*) as n from run_layers where run_id=%s", (run_id,))["n"] > 0
    if corridor and has_layers:
        raise HTTPException(409, "The corridor is fixed once layers are extracted. Copy the run to change it.")
    sets, vals = [], []
    if req.a:
        sets += ["a_lat=%s", "a_lon=%s", "a_name=%s"]
        vals += [req.a.lat, req.a.lon, req.a.name]
    if req.b:
        sets += ["b_lat=%s", "b_lon=%s", "b_name=%s"]
        vals += [req.b.lat, req.b.lon, req.b.name]
    if req.name is not None:                     # renaming is allowed on a locked run
        nm = " ".join(req.name.split())
        if not nm:
            raise HTTPException(400, "corridor name is required")
        sets.append("name=%s")
        vals.append(nm)
    for k in ("kv", "circuit", "band_km", "routes_wanted", "note", "accept_partial"):
        v = getattr(req, k)
        if v is not None:
            if k == "kv" and v not in VOLTAGES:
                raise HTTPException(400, f"voltage must be one of {VOLTAGES}")
            sets.append(f"{k}=%s")
            vals.append(v)
    if req.routes_selected is not None:
        sets += ["routes_selected=%s", "routes_wanted=%s"]
        vals += [db.J(req.routes_selected), len(req.routes_selected)]
    if corridor:
        sets.append("imagery=null")
    if sets:
        db.exec_(f"update runs set {', '.join(sets)}, updated_at=now() where id=%s", (*vals, run_id))
    if corridor:
        nr = _run(run_id)
        try:
            Frame((nr["a_lon"], nr["a_lat"]), (nr["b_lon"], nr["b_lat"]), nr["band_km"] * 1000, 0, 100)
        except ValueError as e:
            raise HTTPException(400, str(e))
        db.exec_("update runs set geo=%s where id=%s",
                 (db.J(geo.validate((nr["a_lon"], nr["a_lat"]), (nr["b_lon"], nr["b_lat"]))), run_id))
    return get_run(run_id)


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: str):
    _run(run_id)
    _ensure_idle_for(run_id)
    if db.one("select count(*) as n from runs where parent_id=%s", (run_id,))["n"]:
        db.exec_("update runs set parent_id=null where parent_id=%s", (run_id,))
    db.exec_("delete from runs where id=%s", (run_id,))
    shutil.rmtree(settings.RUNS_DIR / run_id, ignore_errors=True)
    return {"deleted": run_id}


@app.post("/api/runs/{run_id}/copy", status_code=201)
def copy_run(run_id: str, req: CopyReq):
    r = _run(run_id)
    seq = db.one("select nextval('run_seq') as n")["n"]
    nid = f"{r['family']}-{seq:04d}"
    db.exec_("""insert into runs (id, family, rev, parent_id, name, note, a_lon, a_lat, a_name, b_lon, b_lat, b_name, kv,
                                  circuit, band_km, routes_wanted, routes_selected, sources, rule_overrides, cost_overrides, imagery,
                                  accept_partial, status, geo)
                select %s, family, rev+1, id, name, %s, a_lon, a_lat, a_name, b_lon, b_lat, b_name, kv, circuit, band_km,
                       routes_wanted, routes_selected, sources, rule_overrides, cost_overrides, imagery, accept_partial,
                       case when exists (select 1 from run_layers where run_id=%s) then 'review' else 'draft' end, geo
                from runs where id=%s""", (nid, req.note, run_id, run_id))
    db.exec_("""insert into run_layers (run_id, layer_key, layer_version_id, carried, stale)
                select %s, layer_key, layer_version_id, true, stale from run_layers where run_id=%s""", (nid, run_id))
    return get_run(nid)


# ---- imagery --------------------------------------------------------------------------
@app.post("/api/runs/{run_id}/imagery/check")
def imagery_check(run_id: str):
    r = _run(run_id)
    band, frame = geo.band_polygon((r["a_lon"], r["a_lat"]), (r["b_lon"], r["b_lat"]), r["band_km"])
    res = imagery.check(frame, band)
    db.exec_("update runs set imagery=%s, updated_at=now() where id=%s", (db.J(res), run_id))
    return res


@app.post("/api/runs/{run_id}/procurement", status_code=201)
def procurement(run_id: str, req: ProcureReq):
    r = _run(run_id)
    row = db.one("insert into procurement_requests (run_id, request) values (%s,%s) returning id, created_at",
                 (run_id, db.J({**req.model_dump(), "gap": (r.get("imagery") or {}).get("esri", {}).get("gap_geojson")})))
    return {"id": f"IMG-{row['created_at'].year}-{row['id']:04d}", "status": "stub",
            "message": "Recorded. Procurement is a stub in this release — nothing was sent to a vendor."}


# ---- sources / uploads ----------------------------------------------------------------
@app.put("/api/runs/{run_id}/sources/{layer_key}")
def set_source(run_id: str, layer_key: str, req: SourceSet):
    r = _run(run_id)
    _ensure_idle_for(run_id)
    reg = {l["key"]: l for l in load_layers()["layers"]}
    if layer_key not in reg:
        raise HTTPException(404, f"unknown layer {layer_key}")
    if req.mode in ("replace", "add"):
        if not req.upload_id:
            raise HTTPException(400, "attach a file for Replace / Add more")
        try:
            uploads.get(req.upload_id)
        except KeyError as e:
            raise HTTPException(404, str(e))
        chk = uploads.check_mapping(req.upload_id, layer_key, req.mapping or {})
        if not chk["ok"]:
            raise HTTPException(400, f"mapping targets not in the vocabulary: {chk['bad_targets']}")
    src = dict(r["sources"] or {})
    old = src.get(layer_key, {"mode": "auto"})
    new = req.model_dump(exclude_none=True)
    if req.mode == "auto":
        src.pop(layer_key, None)
    else:
        src[layer_key] = new
    db.exec_("update runs set sources=%s, updated_at=now() where id=%s", (db.J(src), run_id))
    if old != new:
        keys = [layer_key] + (["clusters"] if layer_key == "buildings" else [])
        db.exec_("update run_layers set stale=true where run_id=%s and layer_key = any(%s)", (run_id, keys))
    return get_run(run_id)


@app.post("/api/uploads", status_code=201)
def upload(file: UploadFile = File(...)):
    try:
        return uploads.save(file.filename or "upload", file.file)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"could not read the file: {e}")


@app.get("/api/uploads/{uid}")
def upload_info(uid: int, layer: str | None = None):
    try:
        up = uploads.get(uid)
    except KeyError as e:
        raise HTTPException(404, str(e))
    try:
        info = uploads.inspect(Path(up["path"]), layer)
    except Exception as e:
        raise HTTPException(400, f"could not read layer {layer!r}: {e}")
    return {"id": uid, "filename": up["filename"], "created_at": up["created_at"].isoformat(), **info}


@app.post("/api/uploads/{uid}/mapping/check")
def mapping_check(uid: int, req: MappingReq):
    try:
        res = uploads.check_mapping(uid, req.layer_key, req.mapping)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if req.remember_as and res["ok"]:
        uploads.remember_mapping(req.remember_as, req.layer_key, req.mapping)
    return res


@app.get("/api/mappings/{source_key}")
def mapping_recall(source_key: str):
    return {"mapping": uploads.recall_mapping(source_key)}


# ---- rules / costs --------------------------------------------------------------------
def _display_rules(r: dict) -> dict:
    spec = (r["kv"], r["circuit"])
    rules = load_rules()
    ov = r["rule_overrides"] or {}
    groups = []
    for g in rules["groups"]:
        items = []
        for rule in g["rules"]:
            o = ov.get(rule["id"], {})
            params = []
            for p in rule["params"]:
                d = pick(p["value"], *spec)
                v = (o.get("params") or {}).get(p["key"], d)
                params.append({"key": p["key"], "label": p["label"], "type": p["type"], "unit": p.get("unit"),
                               "options": p.get("options"), "default": d, "value": v, "changed": v != d,
                               "best_guess": bool(p.get("best_guess"))})
            items.append({"id": rule["id"], "name": rule["name"], "note": rule.get("note"), "layer": rule.get("layer"), "classes": rule.get("classes"),
                          "params": params, "enabled": o.get("enabled", True), "reason": o.get("reason"),
                          "rev": o.get("rev"), "n_changed": sum(1 for p in params if p["changed"])})
        groups.append({"key": g["key"], "title": g["title"], "desc": g["desc"], "statutory": bool(g.get("statutory")),
                       "rules": items})
    n_rules = sum(len(g["rules"]) for g in groups)
    return {"version": rules["version"], "groups": groups, "n_rules": n_rules,
            "n_params": sum(len(x["params"]) for g in groups for x in g["rules"]),
            "n_changed": sum(x["n_changed"] for g in groups for x in g["rules"]),
            "n_off": sum(1 for g in groups for x in g["rules"] if not x["enabled"])}


@app.get("/api/runs/{run_id}/rules")
def get_rules(run_id: str):
    return _display_rules(_run(run_id))


@app.put("/api/runs/{run_id}/rules/{rule_id}")
def set_rule(run_id: str, rule_id: str, req: RuleSet):
    r = _run(run_id)
    _ensure_idle_for(run_id)
    disp = _display_rules(r)
    rule = next((x for g in disp["groups"] for x in g["rules"] if x["id"] == rule_id), None)
    if not rule:
        raise HTTPException(404, f"unknown rule {rule_id}")
    defaults = {p["key"]: p for p in rule["params"]}
    params = {}
    for k, v in req.params.items():
        if k not in defaults:
            raise HTTPException(400, f"unknown parameter {k}")
        p = defaults[k]
        if p["type"] == "num":
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{p['label']} must be a number")
            if v < 0:
                raise HTTPException(400, f"{p['label']} cannot be negative")
            if v == int(v):
                v = int(v) if isinstance(p["default"], int) else v
        elif p["type"] == "rng":
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise HTTPException(400, f"{p['label']} needs two values")
            try:
                v = [float(v[0]), float(v[1])]
            except (TypeError, ValueError):
                raise HTTPException(400, f"{p['label']} needs two numbers")
            if v[0] > v[1]:
                raise HTTPException(400, f"{p['label']}: the lower bound is above the upper bound")
        elif p["type"] == "sel":
            if p.get("options") and v not in p["options"]:
                raise HTTPException(400, f"{p['label']} must be one of {p['options']}")
        if v != p["default"]:
            params[k] = v
    changed = bool(params) or not req.enabled
    if changed and not (req.reason or "").strip():
        raise HTTPException(400, "A reason is required — it is tagged as an AESL override in the report.")
    ov = dict(r["rule_overrides"] or {})
    if changed:
        prev = ov.get(rule_id, {})
        ov[rule_id] = {"enabled": req.enabled, "params": params, "reason": req.reason.strip(),
                       "rev": prev.get("rev") if prev.get("params") == params and prev.get("enabled", True) == req.enabled
                       else r["rev"]}
    else:
        ov.pop(rule_id, None)
    db.exec_("update runs set rule_overrides=%s, updated_at=now() where id=%s", (db.J(ov), run_id))
    return _display_rules(_run(run_id))


@app.delete("/api/runs/{run_id}/rules")
def reset_rules(run_id: str):
    _run(run_id)
    _ensure_idle_for(run_id)
    db.exec_("update runs set rule_overrides='{}'::jsonb, updated_at=now() where id=%s", (run_id,))
    return _display_rules(_run(run_id))


def _display_costs(r: dict) -> dict:
    spec = LineSpec(r["kv"], r["circuit"])
    cfg = load_costs()
    ov = r["cost_overrides"] or {}
    resolved = resolve_costs(spec, ov)
    cats = []
    for c in cfg["categories"]:
        items = []
        for it in c["items"]:
            d = pick(it["value"], spec.kv, spec.circuit)
            v = (ov.get(c["key"]) or {}).get(it["key"], d)
            items.append({"key": it["key"], "label": it["label"], "unit": it["unit"], "default": d, "value": v,
                          "changed": v != d, "best_guess": bool(it.get("best_guess"))})
        cats.append({"key": c["key"], "title": c["title"], "desc": c["desc"], "status": c["status"], "items": items})
    bands = {p["key"]: p["value"] for g in load_rules()["groups"] for rr in g["rules"] if rr["id"] == "tower_type_bands"
             for p in rr["params"]}
    base = resolve_costs(spec, {k: v for k, v in ov.items() if k != "tower_price"})["towers"]
    lo = 0
    towers = []
    for t in TOWER_TYPES:
        hi = bands[t]
        p, d = resolved["towers"][t]["price_rs"] / 1e7, base[t]["price_rs"] / 1e7
        towers.append({"type": t, "deviation": f"{lo:g}–{hi:g}°", "price_cr": round(p, 4), "default_price_cr": round(d, 4),
                       "fdn_l": round(resolved["towers"][t]["fdn_rs"] / 1e5, 2),
                       "area_m2": round(resolved["towers"][t]["area_m2"], 2), "changed": abs(p - d) > 1e-9})
        lo = hi
    return {"version": cfg["version"], "categories": cats, "towers": towers,
            "line_cr_km": round(resolved["line_rs_per_m"] * 1000 / 1e7, 4),
            "n_tcc": sum(1 for c in cats if c["status"] == "tcc")}


@app.get("/api/runs/{run_id}/costs")
def get_costs(run_id: str):
    return _display_costs(_run(run_id))


@app.put("/api/runs/{run_id}/costs")
def set_costs(run_id: str, body: dict[str, dict[str, Any]]):
    r = _run(run_id)
    _ensure_idle_for(run_id)
    disp = _display_costs(r)
    defaults = {c["key"]: {i["key"]: i["default"] for i in c["items"]} for c in disp["categories"]}
    tdefault = {t["type"]: t["default_price_cr"] for t in disp["towers"]}
    ov = {k: dict(v) for k, v in (r["cost_overrides"] or {}).items()}
    for cat, items in body.items():
        if cat == "tower_price":
            for t, v in items.items():
                if t not in tdefault:
                    raise HTTPException(400, f"unknown tower type {t}")
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"tower price {t} must be a number")
                if v <= 0:
                    raise HTTPException(400, "tower price must be positive")
                ov.setdefault(cat, {})
                if abs(v - tdefault[t]) < 1e-9:
                    ov[cat].pop(t, None)
                else:
                    ov[cat][t] = v
            continue
        if cat not in defaults:
            raise HTTPException(400, f"unknown cost category {cat}")
        for k, v in items.items():
            if k not in defaults[cat]:
                raise HTTPException(400, f"unknown cost item {cat}.{k}")
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{cat}.{k} must be a number")
            if v < 0:
                raise HTTPException(400, f"{cat}.{k} cannot be negative")
            ov.setdefault(cat, {})
            if abs(v - float(defaults[cat][k])) < 1e-12:
                ov[cat].pop(k, None)
            else:
                ov[cat][k] = v
    ov = {k: v for k, v in ov.items() if v}
    db.exec_("update runs set cost_overrides=%s, updated_at=now() where id=%s", (db.J(ov), run_id))
    return _display_costs(_run(run_id))


# ---- jobs -----------------------------------------------------------------------------
@app.post("/api/runs/{run_id}/extract", status_code=202)
def start_extract(run_id: str, req: ExtractReq | None = None):
    r = _run(run_id)
    if req and req.layers:
        unknown = set(req.layers) - {l["key"] for l in load_layers()["layers"]}
        if unknown:
            raise HTTPException(400, f"unknown layers {sorted(unknown)}")
    src = r["sources"] or {}
    for k, s in src.items():
        if s.get("mode") in ("replace", "add") and not s.get("upload_id"):
            raise HTTPException(400, f"layer {k} is set to {s['mode']} but has no file attached")
    layers_req = req.layers if req else None
    force = bool(req and req.force_models)
    if force and not layers_req:
        layers_req = [l["key"] for l in load_layers()["layers"] if l["producer"] == "lulc_model"]
    j = jobs.start(run_id, "extract", {"requested": layers_req, "force_models": force})
    return _job_json(j)


@app.get("/api/runs/{run_id}/imagery/catalog")
def imagery_catalog(run_id: str):
    """What imagery this machine holds for the run's band, per product (the models use only this)."""
    from . import imagery_store
    r = _run(run_id)
    band, _ = geo.band_polygon((r["a_lon"], r["a_lat"]), (r["b_lon"], r["b_lat"]), r["band_km"], margin_m=500)
    cov = imagery_store.coverage(band)
    items = imagery_store.for_aoi(band)
    needs: dict[str, list[str]] = {}
    from . import lulc_runner
    for name, m in lulc_runner.load_models()["models"].items():
        for p in m.get("imagery") or []:
            needs.setdefault(p, []).append(m["layer"])
    for c in cov:
        c["fetchable"] = c["product"] in imagery_store.FETCHABLE
        c["used_by"] = needs.get(c["product"], [])
        c["sample"] = [{k: v for k, v in it.items() if k != "footprint"} for it in items.get(c["product"], [])[:5]]
    return {"products": cov, "credentials": {"esri_token": bool(settings.ESRI_TOKEN),
                                            "planetary_computer_key": bool(settings.PC_SUBSCRIPTION_KEY)}}


@app.post("/api/runs/{run_id}/imagery/fetch", status_code=202)
def imagery_fetch(run_id: str, req: ImageryFetchReq):
    from . import imagery_store
    _run(run_id)
    bad = [p for p in req.products if p not in imagery_store.FETCHABLE]
    if bad or not req.products:
        raise HTTPException(400, f"cannot be fetched online: {bad or 'nothing selected'} — Cartosat PAN and NISAR are "
                                 "registered from files on this machine")
    j = jobs.start(run_id, "extract", {"imagery_fetch": req.products})
    return _job_json(j)


@app.get("/api/models")
def models_status():
    """LULC model plugins: environment, code and weights present (no model is loaded)."""
    from . import lulc_runner
    out = []
    for name in lulc_runner.load_models()["models"]:
        m = lulc_runner.model_cfg(name)
        out.append({"name": name, "layer": m["layer"], "imagery": m.get("imagery") or [], "gpu": bool(m.get("gpu")),
                    "needs_layers": m.get("needs_layers") or [], "problems": lulc_runner.preflight(m),
                    "params": m.get("params") or {}})
    return {"models": out}


# ---- shapes added on the map (Vivek, 8 Oct 2026) -------------------------------------------
class ShapeAdd(BaseModel):
    layer_key: str
    class_: str | None = Field(None, alias="class")
    name: str = Field(min_length=1, max_length=120)
    scope: Literal["temporary", "permanent"]
    features: list[dict] = Field(min_length=1, max_length=user_shapes.MAX_FEATURES)


@app.post("/api/geo/parse")
def parse_map_file(file: UploadFile = File(...)):
    """A KML / KMZ / GeoJSON / GeoPackage / zipped shapefile -> GeoJSON features in WGS84 (nothing is stored)."""
    try:
        return user_shapes.parse_file(file.filename or "upload", file.file)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(400, f"could not read the file: {e}")


@app.get("/api/runs/{run_id}/shapes")
def list_shapes(run_id: str):
    r = _run(run_id)
    return {"shapes": user_shapes.list_for_run(r)}


@app.post("/api/runs/{run_id}/shapes", status_code=201)
def add_shapes(run_id: str, req: ShapeAdd):
    r = _run(run_id)
    _ensure_idle_for(run_id)
    reg = {l["key"]: l for l in load_layers()["layers"]}
    lay = reg.get(req.layer_key)
    if lay is None:
        raise HTTPException(404, f"unknown layer {req.layer_key}")
    cls = req.class_ or lay["classes"][0]
    if cls not in lay["classes"]:
        raise HTTPException(400, f"class {cls} is not one of {lay['name']}'s classes: {', '.join(lay['classes'])}")
    if req.layer_key not in {x["layer_key"] for x in layers.run_layers(run_id)}:
        raise HTTPException(400, f"{lay['name']} has not been extracted for this run yet; extract the layers first")
    try:
        ids = user_shapes.add(r, req.layer_key, cls, " ".join(req.name.split()), req.scope, req.features)
    except (ValueError, KeyError, TypeError) as e:
        raise HTTPException(400, f"shape not added: {e}")
    from shapely.geometry import shape as _shape
    user_shapes.rebuild_run_layer(r, req.layer_key, changed=[_shape(f["geometry"]) for f in req.features])
    db.exec_("update runs set updated_at=now() where id=%s", (run_id,))
    return {"added": ids, "layer_key": req.layer_key, "shapes": user_shapes.list_for_run(r)}


@app.delete("/api/runs/{run_id}/shapes/{shape_id}")
def delete_shape(run_id: str, shape_id: int):
    r = _run(run_id)
    _ensure_idle_for(run_id)
    visible = {s["id"]: s for s in user_shapes.list_for_run(r)}
    if shape_id not in visible:
        raise HTTPException(404, f"shape {shape_id} is not on this run")
    row = user_shapes.delete(shape_id)
    if row:
        from shapely.geometry import shape as _shape
        user_shapes.rebuild_run_layer(r, row["layer_key"], changed=[_shape(visible[shape_id]["geometry"])])
    db.exec_("update runs set updated_at=now() where id=%s", (run_id,))
    return {"deleted": shape_id, "layer_key": visible[shape_id]["layer_key"], "shapes": user_shapes.list_for_run(r)}


class SolveReq(BaseModel):
    exports: bool = True            # False: quick test solve — routes only, exports made on request later


@app.post("/api/runs/{run_id}/solve", status_code=202)
def start_solve(run_id: str, req: SolveReq | None = None):
    r = _run(run_id)
    rl = layers.run_layers(run_id)
    # a layer added to the registry after a run was extracted (optional: true) does not block its solve —
    # the engine treats it as empty (14 Oct 2026: railway platforms)
    reg = [l["key"] for l in load_layers()["layers"] if not l.get("optional")]
    have = {x["layer_key"]: x for x in rl}
    missing = [k for k in reg if k not in have]
    stale = [k for k, x in have.items() if x["stale"]]
    if missing or stale:
        a = jobs.active()
        if a:
            raise jobs.Busy(a)
        raise HTTPException(400, "Extract the layers first: " + ", ".join(missing + stale) + " not ready for this revision.")
    j = jobs.start(run_id, "solve", {"exports": True if req is None else bool(req.exports)})
    return _job_json(j)


@app.get("/api/jobs/active")
def job_active():
    return {"job": _job_json(jobs.active())}


@app.get("/api/jobs/{job_id}")
def job_get(job_id: int):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return _job_json(j)


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: int):
    j = jobs.request_cancel(job_id)
    if not j:
        raise HTTPException(409, "job is not running")
    return _job_json(j)


# ---- layers ---------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/layers")
def run_layers(run_id: str):
    r = _run(run_id)
    reg = load_layers()["layers"]
    have = {x["layer_key"]: x for x in layers.run_layers(run_id)}
    rules_by_layer: dict[str, list[str]] = {}
    for g in load_rules()["groups"]:
        for rr in g["rules"]:
            for lk in ([rr["layer"]] if rr.get("layer") else []) + (rr.get("also_layers") or []):
                rules_by_layer.setdefault(lk, []).append(rr["name"])
    band, frame = geo.band_polygon((r["a_lon"], r["a_lat"]), (r["b_lon"], r["b_lat"]), r["band_km"])
    band_km2 = geo.area_km2(band, frame)
    out, flags = [], []
    src = r["sources"] or {}
    for l in reg:
        x = have.get(l["key"])
        item = {"key": l["key"], "name": l["name"], "desc": l.get("desc"), "producer": l["producer"],
                "configured_producer": l["producer"], "refreshable": bool(l.get("refreshable")),
                "refresh_label": l.get("refresh_label"), **_refresh_state(l),
                "minutes_per_km2": l.get("minutes_per_km2"), "rules": rules_by_layer.get(l["key"], []),
                "source_setting": src.get(l["key"], {"mode": "auto"}), "extracted": x is not None}
        if x:
            cov_pct = None
            if x["coverage_geojson"]:
                cov = shape(x["coverage_geojson"])
                cov_pct = round(100 * geo.area_km2(band.intersection(cov), frame) / band_km2, 1)
            item.update({k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in x.items()
                         if k not in ("coverage_geojson",)})
            item["extracted_producer"] = x["producer"]
            item["coverage_pct"] = cov_pct
            if x["status"] == "failed":
                flags.append({"layer": l["key"], "kind": "failed", "text": f"{l['name']}: extraction failed — {x['error']}"})
            elif x["feature_count"] == 0:
                flags.append({"layer": l["key"], "kind": "empty",
                              "text": f"{l['name']}: nothing found in the band" +
                              (f" — {len(item['rules'])} rule(s) will not fire." if item["rules"] else ".")})
            elif x["confidence"] is not None and x["confidence"] < 0.8:
                flags.append({"layer": l["key"], "kind": "low_conf",
                              "text": f"{l['name']}: confidence {x['confidence']:.2f} — below 0.80; check it against the imagery."})
            if cov_pct is not None and cov_pct < 99.5 and x["status"] != "failed":
                flags.append({"layer": l["key"], "kind": "coverage",
                              "text": f"{l['name']}: data covers {cov_pct:.0f}% of the band — rules on it do not fire elsewhere."})
            if x["stale"]:
                flags.append({"layer": l["key"], "kind": "stale", "text": f"{l['name']}: source changed — extract again before solving."})
        out.append(item)
    tot = db.one("""select coalesce(sum(lv.feature_count),0) as n from run_layers rl join layer_versions lv on lv.id=rl.layer_version_id
                    where rl.run_id=%s""", (run_id,))["n"]
    return {"layers": out, "flags": flags, "band_km2": round(band_km2, 1), "features_total": int(tot),
            "band": band.__geo_interface__}


@app.post("/api/runs/{run_id}/layers/{layer_key}/refresh", status_code=202)
def refresh_layer(run_id: str, layer_key: str):
    """Re-pull the layer's source for this run's area, update the shared store, re-extract the layer."""
    _run(run_id)
    reg = {l["key"]: l for l in load_layers()["layers"]}
    if layer_key not in reg:
        raise HTTPException(404, f"unknown layer {layer_key}")
    if not reg[layer_key].get("refreshable"):
        raise HTTPException(400, f"{reg[layer_key]['name']} has no online source to refresh from")
    lay = reg[layer_key]
    label = lay.get("refresh_label") or "its source"
    st = _refresh_state(lay)
    if not st["refresh_available"]:
        raise HTTPException(503, f"{lay['name']}: refresh from {label} is not available — {st['refresh_note']}")
    src = (_run(run_id)["sources"] or {}).get(layer_key, {"mode": "auto"})
    if src.get("mode") == "replace":
        raise HTTPException(400, f"{lay['name']} uses your file (Replace); switch it to Auto-fetch or Add more to refresh from {label}")
    j = jobs.start(run_id, "extract", {"requested": [layer_key], "refresh": True})
    return _job_json(j)


def _refresh_state(lay: dict) -> dict:
    """Whether a refreshable layer's online source can be reached by this deployment."""
    if not lay.get("refreshable"):
        return {"refresh_available": False, "refresh_note": None}
    if lay["producer"] == "tarangsanchar":
        from . import cell_towers
        ok = cell_towers.refresh_available()
        return {"refresh_available": ok, "refresh_note": None if ok else cell_towers.REFRESH_PENDING}
    return {"refresh_available": True, "refresh_note": None}


@app.get("/api/osm/power")
def osm_power_status():
    from . import osm_power
    st = osm_power.status()
    for l in st["loads"]:
        for k in ("osm_timestamp", "started_at", "finished_at"):
            if l.get(k) is not None:
                l[k] = l[k].isoformat()
    return st


@app.get("/api/osm/highways")
def osm_highways_status():
    from . import osm_highways
    st = osm_highways.status()
    for l in st["loads"]:
        for k in ("osm_timestamp", "started_at", "finished_at"):
            if l.get(k) is not None:
                l[k] = l[k].isoformat()
    return st


@app.get("/api/aviation")
def aviation_status():
    from . import aviation
    st = aviation.status()
    for l in st["loads"]:
        for k in ("data_date", "started_at", "finished_at"):
            if l.get(k) is not None:
                l[k] = l[k].isoformat()
    return st


@app.get("/api/telecom/towers")
def cell_towers_status():
    from . import cell_towers
    st = cell_towers.status()
    for l in st["loads"]:
        for k in ("data_date", "started_at", "finished_at"):
            if l.get(k) is not None:
                l[k] = l[k].isoformat()
    return st


@app.get("/api/runs/{run_id}/features/{fid}")
def feature(run_id: str, fid: int):
    r = _run(run_id)
    f = db.one("""select f.id, f.class, f.name, f.confidence, f.origin, f.src_id, f.attrs, ST_AsGeoJSON(f.geom, 6)::jsonb as geom,
                         ST_Area(f.geom::geography) as area_m2, ST_Length(f.geom::geography) as length_m,
                         lv.layer_key, lv.mode, lv.source, lv.model, lv.model_version, lv.image_source, lv.image_date,
                         lv.model_run_at, lv.confidence as layer_conf, lv.file_name
                  from features f join layer_versions lv on lv.id=f.layer_version_id
                  join run_layers rl on rl.layer_version_id=lv.id and rl.run_id=%s where f.id=%s""", (run_id, fid))
    if not f:
        raise HTTPException(404, "feature not found in this run")
    g = shape(f["geom"])
    fr = Frame((r["a_lon"], r["a_lat"]), (r["b_lon"], r["b_lat"]), r["band_km"] * 1000, 0, 100)
    fg = fr.geom_utm_to_frame(fr.geom_to_utm(g))
    f["chainage_km"] = [round(max(0, fg.bounds[0]) / 1000, 2), round(min(fr.length_m, fg.bounds[2]) / 1000, 2)]
    return {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in f.items()}


class NearestPoint(BaseModel):
    layer: str
    lon: float
    lat: float
    within_m: float = 3000


class NearestRequest(BaseModel):
    points: list[NearestPoint]


@app.post("/api/runs/{run_id}/nearest")
def nearest(run_id: str, body: NearestRequest):
    """For each point, the closest feature of that run layer within within_m: the feature, the closest
    point on it and the distance. The route map ties each bend to the feature it reacts to (15 Oct 2026).
    Null where nothing is within reach."""
    _run(run_id)
    out = []
    with db.conn() as c:
        for q in body.points[:400]:
            # the point inlined as a constant so the GiST index serves both the box filter and the KNN order
            deg = q.within_m / 111320.0 * 1.2
            pt = "ST_SetSRID(ST_MakePoint(%s, %s), 4326)"
            r = c.execute(f"""
              select f.id, f.name, f.class, ST_X(ST_ClosestPoint(f.geom, {pt})) as lon, ST_Y(ST_ClosestPoint(f.geom, {pt})) as lat,
                     ST_Distance(f.geom::geography, {pt}::geography) as distance_m
              from features f join run_layers rl on rl.layer_version_id = f.layer_version_id
              where rl.run_id = %s and rl.layer_key = %s and f.geom && ST_Expand({pt}, %s)
              order by f.geom <-> {pt} limit 1""",
                          (q.lon, q.lat, q.lon, q.lat, q.lon, q.lat, run_id, q.layer, q.lon, q.lat, deg, q.lon, q.lat)).fetchone()
            if r and float(r["distance_m"]) > q.within_m:
                r = None
            out.append(None if not r else {"id": r["id"], "name": r["name"], "class": r["class"], "lon": r["lon"], "lat": r["lat"],
                                           "distance_m": round(float(r["distance_m"]), 1)})
    return {"results": out}


class BufferItem(BaseModel):
    kind: str
    layer: str
    buffer_m: float
    classes: list[str] | None = None
    ids: list[int] | None = None
    near_m: float = 1000
    limit: int = 400


class BuffersRequest(BaseModel):
    route: list[list[float]]
    items: list[BufferItem]


@app.post("/api/runs/{run_id}/buffers")
def buffers(run_id: str, body: BuffersRequest):
    """Features of a run layer near the route, each buffered by buffer_m (0 = the feature itself), as GeoJSON:
    the setback a bend kept outside, the parallel-run band along a road or canal, a no-crossing zone
    (15 Oct 2026). One item per rule; features within near_m of the route line, at most `limit` each."""
    _run(run_id)
    if len(body.route) < 2:
        raise HTTPException(400, "route needs at least two points")
    line = json.dumps({"type": "LineString", "coordinates": [[float(x), float(y)] for x, y in body.route]})
    feats = []
    with db.conn() as c:
        for it in body.items[:30]:
            # the corridor round the route, cut into small pieces so the spatial index does the work (17 Oct 2026:
            # 0.7 s instead of 22 s for village roads within 150 m of a 112 km route)
            rows = c.execute("""
              with c as (select ST_Subdivide(ST_Buffer(ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326)::geography, %s)::geometry, 32) as g),
                   hit as (select distinct f.id from features f
                           join run_layers rl on rl.layer_version_id = f.layer_version_id
                           join c on f.geom && c.g and ST_Intersects(f.geom, c.g)
                           where rl.run_id = %s and rl.layer_key = %s
                             and (%s::text[] is null or f.class = any(%s::text[]))
                             and (%s::bigint[] is null or f.id = any(%s::bigint[]))
                           limit %s)
              select f.id, f.name, f.class,
                     ST_AsGeoJSON(ST_SimplifyPreserveTopology(
                       case when %s > 0 then ST_Buffer(f.geom::geography, %s)::geometry else f.geom end, 0.00003), 6)::json as geom
              from hit join features f on f.id = hit.id""",
                             (line, it.near_m, run_id, it.layer, it.classes, it.classes, it.ids, it.ids,
                              max(1, min(it.limit, 2000)), it.buffer_m, it.buffer_m)).fetchall()
            for r in rows:
                feats.append({"type": "Feature", "geometry": r["geom"],
                              "properties": {"kind": it.kind, "layer": it.layer, "id": r["id"], "name": r["name"], "class": r["class"], "buffer_m": it.buffer_m}})
    return {"type": "FeatureCollection", "features": feats}


class WithinRequest(BaseModel):
    layer: str
    radius_m: float
    points: list[Pt]
    classes: list[str] | None = None


@app.post("/api/runs/{run_id}/within")
def within(run_id: str, body: WithinRequest):
    """Which of the points lie within radius_m of a feature of that run layer (optionally of the given
    classes): one hit per point-feature pair. Set-based on the geography index, so a whole route's towers
    against the airport NOC radius takes well under a second (15 Oct 2026)."""
    _run(run_id)
    if not body.points:
        return {"hits": []}
    with db.conn() as c:
        rows = c.execute("""
          with p as (select ord, ST_SetSRID(ST_MakePoint(lon, lat), 4326) as g
                     from unnest(%s::float[], %s::float[]) with ordinality as u(lon, lat, ord)),
               f as (select f.id, f.name, f.class, f.geom from features f
                     join run_layers rl on rl.layer_version_id = f.layer_version_id
                     where rl.run_id = %s and rl.layer_key = %s and (%s::text[] is null or f.class = any(%s::text[])))
          select p.ord - 1 as i, f.id, f.name, f.class
          from p join f on ST_DWithin(p.g::geography, f.geom::geography, %s)
          order by p.ord""", ([q.lon for q in body.points], [q.lat for q in body.points], run_id, body.layer,
                              body.classes, body.classes, body.radius_m)).fetchall()
    return {"hits": [{"i": r["i"], "id": r["id"], "name": r["name"], "class": r["class"]} for r in rows]}


_MVT_NATIVE: bool | None = None   # PostGIS built with protobuf-c? decided on first tile
# Left out of tiles below z12: too many small features to see or to encode at overview scale (a z9
# tile of a ±25 km band held 79k village-road segments: 50 s, 2.8 MB). The map style hides them
# below z12 too (web/src/components/map/style.ts DENSE_MINZOOM).
DENSE_LAYERS = ("buildings", "village_roads")


def _tile_native(run_id, z, x, y, skip_dense) -> bytes:
    with db.conn() as c:
        rows = c.execute("""
          with env as (select ST_TileEnvelope(%s, %s, %s) as e),
               lv as (select layer_key, layer_version_id from run_layers where run_id=%s
                      and (not %s or layer_key <> all(%s)))
          select q.layer_key, ST_AsMVT(q, q.layer_key, 4096, 'geom', 'id') as tile from (
            select lv.layer_key, f.id, f.class, f.name, f.confidence, f.origin,
                   (f.attrs->>'voltage_kv')::int as voltage_kv,
                   ST_AsMVTGeom(ST_Transform(f.geom, 3857), env.e, 4096, 64, true) as geom
            from features f join lv on f.layer_version_id = lv.layer_version_id, env
            where f.geom && ST_Transform(env.e, 4326)
          ) q where q.geom is not null group by q.layer_key""", (z, x, y, run_id, skip_dense, list(DENSE_LAYERS))).fetchall()
    return b"".join(bytes(r["tile"]) for r in rows if r["tile"])


def _tile_python(run_id, z, x, y, skip_dense) -> bytes:
    """Same tile, encoded in Python (PostGIS builds without protobuf-c, e.g. conda-forge)."""
    import mapbox_vector_tile
    import shapely
    with db.conn() as c:
        rows = c.execute("""
          with env as (select ST_TileEnvelope(%s, %s, %s) as e),
               lv as (select layer_key, layer_version_id from run_layers where run_id=%s
                      and (not %s or layer_key <> all(%s)))
          select lv.layer_key, f.id, f.class, f.name, f.confidence, f.origin,
                 (f.attrs->>'voltage_kv')::int as voltage_kv,
                 -- ST_Simplify drops rings below a tile pixel (preserveCollapsed false); the topology-
                 -- preserving variant kept every ring of the pixel-traced solar parks: 8.3 MB, 17 s per z9 tile
                 ST_AsBinary(ST_Simplify(ST_ClipByBox2D(ST_Transform(f.geom, 3857),
                     ST_Expand(env.e, (ST_XMax(env.e) - ST_XMin(env.e)) * 64 / 4096)),
                     (ST_XMax(env.e) - ST_XMin(env.e)) / 4096, false)) as wkb,
                 ST_XMin(env.e) as x0, ST_YMin(env.e) as y0, ST_XMax(env.e) as x1, ST_YMax(env.e) as y1
          from features f join lv on f.layer_version_id = lv.layer_version_id, env
          where f.geom && ST_Transform(env.e, 4326)""", (z, x, y, run_id, skip_dense, list(DENSE_LAYERS))).fetchall()
    if not rows:
        return b""
    bounds = (rows[0]["x0"], rows[0]["y0"], rows[0]["x1"], rows[0]["y1"])
    layers_: dict[str, list] = {}
    for r in rows:
        if r["wkb"] is None:
            continue
        g = shapely.from_wkb(bytes(r["wkb"]))
        if g.is_empty:
            continue
        layers_.setdefault(r["layer_key"], []).append({
            "geometry": g, "id": r["id"],
            "properties": {k: v for k, v in (("id", r["id"]), ("class", r["class"]), ("name", r["name"]),
                                             ("confidence", r["confidence"]), ("origin", r["origin"]),
                                             ("voltage_kv", r["voltage_kv"])) if v is not None}})
    return mapbox_vector_tile.encode(
        [{"name": k, "features": v} for k, v in layers_.items()],
        default_options={"quantize_bounds": bounds, "extents": 4096, "y_coord_down": False})


@app.get("/api/tiles/{run_id}/{z}/{x}/{y}.pbf")
def vector_tile(run_id: str, z: int, x: int, y: int):
    global _MVT_NATIVE
    if z < 6 or z > 16:
        return Response(b"", media_type="application/x-protobuf")
    skip_dense = z < 12
    data = None
    if _MVT_NATIVE is not False:
        try:
            data = _tile_native(run_id, z, x, y, skip_dense)
            if data:
                _MVT_NATIVE = True
        except Exception as e:
            if "protobuf" not in str(e).lower():
                raise
            _MVT_NATIVE = False
            log.warning("PostGIS lacks protobuf-c; encoding vector tiles in Python")
    if data is None:
        data = _tile_python(run_id, z, x, y, skip_dense)
    return Response(data, media_type="application/x-protobuf", headers={"Cache-Control": "no-cache"})


@app.get("/api/tiles/esri/{z}/{x}/{y}")
def esri_tile(z: int, x: int, y: int):
    data, ctype = imagery.tile(z, x, y)
    if data is None:
        raise HTTPException(404, ctype)
    return Response(data, media_type=ctype, headers={"Cache-Control": "public, max-age=604800"})


# ---- terrain (10 Oct 2026): hillshade / 3D tiles, cursor elevation, route profile -------------
@app.get("/api/tiles/dem/{z}/{x}/{y}.png")
def dem_tile(z: int, x: int, y: int):
    from . import terrain
    data = terrain.tile(z, x, y)
    if data is None:
        raise HTTPException(404, "no DEM tile on disk here")
    return Response(data, media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/dem/elevation")
def dem_elevation(lon: float, lat: float):
    from . import terrain
    return {"lon": lon, "lat": lat, "elevation_m": terrain.elevation(lon, lat)}


@app.get("/api/runs/{run_id}/dem")
def dem_coverage(run_id: str):
    r = _run(run_id)
    from . import terrain
    from beeline_engine.frame import Frame
    fr = Frame((r["a_lon"], r["a_lat"]), (r["b_lon"], r["b_lat"]), r["band_km"] * 1000, 500, 100)
    return terrain.covers(fr.band_polygon_lonlat(margin=True).bounds)


@app.get("/api/solves/{sid}/routes/{route_id}/profile")
def route_profile(sid: int, route_id: int):
    s = db.one("select result, run_id from solves where id=%s", (sid,))
    if not s:
        raise HTTPException(404, "solve not found")
    route = next((r for r in (s["result"] or {}).get("routes", []) if r["id"] == route_id), None)
    if route is None:
        raise HTTPException(404, "route not in this solve")
    run = db.one("select kv from runs where id=%s", (s["run_id"],))
    from . import terrain
    return terrain.profile(route, int(run["kv"]))


# ---- manual tower moves (10 Oct 2026) ---------------------------------------------------
class TowerCheck(BaseModel):
    route_id: int
    tower_no: int
    lon: float
    lat: float


class TowerMove(BaseModel):
    tower_no: int
    lon: float
    lat: float


class TowerMoves(BaseModel):
    route_id: int
    moves: list[TowerMove] = Field(min_length=1, max_length=200)
    note: str | None = None


@app.get("/api/solves/{sid}/edit/status")
def edit_status(sid: int, prepare: bool = False):
    """Whether the solve's rules are loaded for live checks; prepare=true starts loading them."""
    from . import towers_edit
    st = towers_edit.status(sid)
    if prepare and not st["ready"] and not st.get("building"):
        import threading
        threading.Thread(target=towers_edit.context, args=(sid,), daemon=True).start()
        st = {"ready": False, "building": True}
    return st


@app.get("/api/solves/{sid}/routes/{route_id}/towers/{tower_no}/why")
def tower_why(sid: int, route_id: int, tower_no: int):
    """Why the line bends at this tower: the straight alternative tested with the solve's own rules (see
    towers_edit.EditContext.explain). The rule checker is loaded on first use (a few minutes per solve);
    until then {"preparing": true} and the caller polls."""
    import threading
    from . import towers_edit
    if not db.one("select id from solves where id=%s", (sid,)):
        raise HTTPException(404, "solve not found")
    ctx = towers_edit.context(sid, wait=False)
    if ctx is None:
        if not towers_edit.status(sid).get("building"):
            threading.Thread(target=towers_edit.context, args=(sid,), daemon=True).start()
        return {"preparing": True}
    try:
        return ctx.explain(route_id, tower_no)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/solves/{sid}/edit/check")
def edit_check(sid: int, req: TowerCheck):
    from . import towers_edit
    ctx = towers_edit.context(sid, wait=False)
    if ctx is None:
        return {"ok": None, "preparing": True}
    try:
        return ctx.check(req.route_id, req.tower_no, req.lon, req.lat)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/solves/{sid}/nogo/{z}/{x}/{y}.png")
def nogo_tile(sid: int, z: int, x: int, y: int):
    from . import towers_edit
    ctx = towers_edit.context(sid, wait=False)
    if ctx is None:
        raise HTTPException(503, "rules still loading")
    return Response(ctx.nogo_tile(z, x, y), media_type="image/png", headers={"Cache-Control": "no-cache"})


@app.post("/api/solves/{sid}/edit/apply", status_code=201)
def edit_apply(sid: int, req: TowerMoves):
    """Apply hand moves: a new solve version with the route's moved towers and re-typed neighbours."""
    from . import towers_edit
    s = db.one("select run_id from solves where id=%s", (sid,))
    if not s:
        raise HTTPException(404, "solve not found")
    a = jobs.active()
    if a and a["run_id"] == s["run_id"]:
        raise jobs.Busy(a)
    ctx = towers_edit.context(sid)
    try:
        out = ctx.apply(req.route_id, [m.model_dump() for m in req.moves], req.note)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"solve_id": out["solve_id"], "solve": _solve_json(db.one("select * from solves where id=%s", (out["solve_id"],)))}


# ---- results --------------------------------------------------------------------------
def _solve_json(s: dict) -> dict:
    files, sizes = [], {}
    if s.get("out_dir") and Path(s["out_dir"]).exists():
        for p in sorted(Path(s["out_dir"]).iterdir()):
            if p.is_file():
                files.append(p.name)
                sizes[p.name] = p.stat().st_size
    return {"file_sizes": sizes, "id": s["id"], "run_id": s["run_id"], "created_at": s["created_at"].isoformat(), "feasible": s["feasible"],
            "result": s["result"], "manifest": s["manifest"], "files": files}


@app.get("/api/runs/{run_id}/solve")
def latest_solve(run_id: str):
    r = _run(run_id)
    if not r["latest_solve_id"]:
        raise HTTPException(404, "this run has not been solved yet")
    return _solve_json(db.one("select * from solves where id=%s", (r["latest_solve_id"],)))


@app.get("/api/solves/{sid}")
def get_solve(sid: int):
    s = db.one("select * from solves where id=%s", (sid,))
    if not s:
        raise HTTPException(404, "solve not found")
    return _solve_json(s)


@app.post("/api/solves/{sid}/exports")
def make_exports(sid: int):
    """Write the export set for a solve run with exports skipped."""
    s = db.one("select id, run_id from solves where id=%s", (sid,))
    if not s:
        raise HTTPException(404, "solve not found")
    a = jobs.active()
    if a and a["run_id"] == s["run_id"]:
        raise jobs.Busy(a)
    from .solvejob import exports_for_solve
    exports_for_solve(sid)
    return _solve_json(db.one("select * from solves where id=%s", (sid,)))


@app.get("/api/solves/{sid}/files/{name}")
def solve_file(sid: int, name: str):
    s = db.one("select out_dir, run_id from solves where id=%s", (sid,))
    if not s or not s["out_dir"]:
        raise HTTPException(404, "solve not found")
    p = (Path(s["out_dir"]) / name).resolve()
    if p.parent != Path(s["out_dir"]).resolve() or not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p, filename=f"{s['run_id']}_{name}" if not name.startswith(s["run_id"]) else name)


@app.get("/api/runs/{run_id}/compare")
def compare(run_id: str):
    r = _run(run_id)
    if not r["parent_id"]:
        return {"parent": None}
    p = _run(r["parent_id"])
    if not (r["latest_solve_id"] and p["latest_solve_id"]):
        return {"parent": p["id"], "rows": None, "changes": _changes(p, r)}
    a = db.one("select result from solves where id=%s", (p["latest_solve_id"],))["result"]
    b = db.one("select result from solves where id=%s", (r["latest_solve_id"],))["result"]
    rows = []
    for rb in b.get("routes", []):
        ra = next((x for x in a.get("routes", []) if x["id"] == rb["id"]), None)
        if not ra:
            continue
        def row(label, fa, fb, unit, nd=1):
            return {"label": label, "prev": fa, "now": fb, "delta": None if fa is None or fb is None else round(fb - fa, nd),
                    "unit": unit}
        rows.append({"route": rb["id"], "name": rb["name"], "rows": [
            row("Length", ra["stats"]["length_km"], rb["stats"]["length_km"], "km", 2),
            row("Towers", ra["stats"]["towers"], rb["stats"]["towers"], "", 0),
            row("Angle points", ra["stats"]["angle_points"], rb["stats"]["angle_points"], "", 0),
            row("Forest taken", ra["stats"]["forest_ha"], rb["stats"]["forest_ha"], "ha", 2),
            row("Total cost", round(ra["cost"]["total"] / 1e7, 2), round(rb["cost"]["total"] / 1e7, 2), "₹ Cr", 2)]})
    return {"parent": p["id"], "parent_rev": p["rev"], "rev": r["rev"], "rows": rows, "changes": _changes(p, r),
            "parent_routes": [{"id": x["id"], "geometry": x["geometry"]} for x in a.get("routes", [])]}


def _changes(p: dict, r: dict) -> list[str]:
    out = []
    reg = {l["key"]: l["name"] for l in load_layers()["layers"]}
    ps, rs = p["sources"] or {}, r["sources"] or {}
    for k in sorted(set(ps) | set(rs)):
        if ps.get(k) != rs.get(k):
            out.append(f"{reg.get(k, k)}: {ps.get(k, {}).get('mode', 'auto')} → {rs.get(k, {}).get('mode', 'auto')}")
    po, ro = p["rule_overrides"] or {}, r["rule_overrides"] or {}
    for k in sorted(set(po) | set(ro)):
        if po.get(k) != ro.get(k):
            out.append(f"Rule {k}: {'changed' if k in ro else 'reset to default'}" + (f" — {ro[k].get('reason')}" if k in ro else ""))
    if (p["cost_overrides"] or {}) != (r["cost_overrides"] or {}):
        out.append("Cost rates changed")
    return out


# ---- static SPA -----------------------------------------------------------------------
DIST = Path(__file__).resolve().parent.parent / "web" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "not found")
        f = DIST / path
        if path and f.is_file():
            return FileResponse(f)
        # the shell must be revalidated on every load, or a browser keeps an old bundle after a rebuild
        # (hashed assets under /assets stay cacheable)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
