"""Step 2 — solve the route on the reviewed layers of a run."""
from __future__ import annotations

import logging
import shutil
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import pyogrio
import shapely

from beeline_engine.config import load_engine, load_layers, load_rules, load_costs, rule_catalogue
from beeline_engine.pipeline import Cancelled as EngineCancelled
from beeline_engine.pipeline import LayerInput, SolveInput, run_solve

from . import db, exports, layers, settings
from .jobs import Cancelled, Reporter

log = logging.getLogger(__name__)


def _git_rev() -> str | None:
    import os
    if os.environ.get("BEELINE_COMMIT"):
        return os.environ["BEELINE_COMMIT"]
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=settings.REPO, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def build_manifest(run: dict, rls: list[dict], solve_result: dict | None) -> dict:
    cat = {r["id"]: r for r in rule_catalogue(run["rule_overrides"])}
    overrides = []
    for rid, ov in (run["rule_overrides"] or {}).items():
        r = cat.get(rid)
        if not r:
            continue
        if ov.get("enabled") is False:
            overrides.append({"rule": r["name"], "rule_id": rid, "param": "enabled", "default": True, "value": False,
                              "reason": ov.get("reason"), "rev": ov.get("rev"), "statutory": r["group"] == "lock"})
        for k, v in (ov.get("params") or {}).items():
            p = next((x for x in r["params"] if x["key"] == k), None)
            if p is None:
                continue
            overrides.append({"rule": r["name"], "rule_id": rid, "param": p["label"], "key": k, "default": p["default"],
                              "value": v, "reason": ov.get("reason"), "rev": ov.get("rev"),
                              "statutory": r["group"] == "lock"})
    costs = load_costs()
    unsigned = [c["title"] for c in costs["categories"] if c["status"] == "tcc"]
    notices = []
    if unsigned:
        notices.append("Draft rates — awaiting AESL TCC sign-off: " + ", ".join(unsigned) + ".")
    if any(o["statutory"] for o in overrides):
        notices.append("A statutory minimum was changed in this run; see Overrides.")
    if run.get("accept_partial"):
        notices.append("Solved over partial high-resolution imagery by the drafter's choice.")
    return {
        "run_id": run["id"], "rev": run["rev"], "parent": run.get("parent_id"),
        "ruleset_version": load_rules()["version"], "costs_version": costs["version"],
        "layers_version": load_layers()["version"], "engine_commit": _git_rev(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corridor": {"a": {"lon": run["a_lon"], "lat": run["a_lat"]}, "a_name": run.get("a_name"),
                     "b": {"lon": run["b_lon"], "lat": run["b_lat"]},
                     "b_name": run.get("b_name"), "band_km": run["band_km"]},
        "line": {"kv": run["kv"], "circuit": run["circuit"], "routes_wanted": run["routes_wanted"],
                 **((solve_result or {}).get("line") or {})},
        "sources": [{"layer": r["layer_key"], "mode": r["mode"], "producer": r["producer"], "source": r["source"],
                     "file": r["file_name"], "model": r["model"], "model_version": r["model_version"],
                     "image_source": r["image_source"], "image_date": r["image_date"],
                     "model_run_at": r["model_run_at"], "features": r["feature_count"], "confidence": r["confidence"],
                     "extracted_in": r["extracted_in"], "carried": r["carried"]} for r in rls],
        "rule_overrides": run["rule_overrides"], "cost_overrides": run["cost_overrides"],
        "overrides": overrides, "notices": notices,
        "engine": load_engine(), "imagery": run.get("imagery"),
        "timings": (solve_result or {}).get("timings"),
    }


def run_solve_job(job: dict, rep: Reporter):
    run = db.one("select * from runs where id=%s", (job["run_id"],))
    db.exec_("update runs set status='solving', updated_at=now() where id=%s", (run["id"],))
    rls = layers.run_layers(run["id"])
    reg = {l["key"]: l for l in load_layers()["layers"]}
    stale = [r["layer_key"] for r in rls if r["stale"]]
    missing = [k for k in reg if k not in {r["layer_key"] for r in rls}]
    if stale or missing:
        raise ValueError(f"layers not extracted for this revision: {', '.join(stale + missing)}")
    rep(0.01, "Loading layers", force=True)
    t_load = time.time()
    inputs = {}
    for r in rls:
        rep.check()
        gdf = layers.read_version(r["layer_version_id"]) if r["status"] == "done" else None
        cov = None
        if r["coverage_geojson"]:
            cov = shapely.geometry.shape(r["coverage_geojson"])
        inputs[r["layer_key"]] = LayerInput(r["layer_key"], gdf, r["mode"], r["source"] or "", r["confidence"], cov,
                                            r["model"])
    inp = SolveInput(a=(run["a_lon"], run["a_lat"]), b=(run["b_lon"], run["b_lat"]), band_km=run["band_km"],
                     kv=run["kv"], circuit=run["circuit"], layers=inputs, a_name=run.get("a_name") or "A",
                     b_name=run.get("b_name") or "B", routes_wanted=run["routes_wanted"],
                     rule_overrides={k: {kk: vv for kk, vv in v.items()} for k, v in (run["rule_overrides"] or {}).items()},
                     cost_overrides=run["cost_overrides"] or {}, dem_dir=settings.DEM_DIR,
                     cache_dir=settings.DATA / "cache" / "rasters")
    want_exports = bool((job.get("detail") or {}).get("exports", True))
    load_layers_s = round(time.time() - t_load, 2)

    def prog(f, stage):
        rep(0.03 + 0.9 * f, stage)

    try:
        result = run_solve(inp, progress=prog, cancel=rep.cancelled)
    except EngineCancelled:
        raise Cancelled()
    rep(0.94, "Writing exports" if want_exports else "Saving the routes (exports skipped)", force=True)
    t_save = time.time()
    manifest = build_manifest(run, rls, result)
    manifest["timings"] = {**(manifest.get("timings") or {}), "load_layers_s": load_layers_s}
    # the layer versions this solve used, so exports made later read the same data
    manifest["layer_version_ids"] = {r["layer_key"]: r["layer_version_id"] for r in rls}
    manifest["exports_skipped"] = not want_exports
    row = db.one("""insert into solves (run_id, job_id, feasible, result, manifest) values (%s,%s,%s,%s,%s)
                    returning id, created_at""", (run["id"], job["id"], result["feasible"], db.J(result), db.J(manifest)))
    out_dir = settings.RUNS_DIR / run["id"] / f"solve_{row['id']}"
    if out_dir.exists():
        # a new solve id owns this folder; anything here is left over (e.g. a reset database)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    if want_exports:
        solve = {"id": row["id"], "created_at": row["created_at"].isoformat(), "result": result}
        files = write_exports(run, solve, manifest, rls, out_dir)
    manifest["timings"]["save_exports_s"] = round(time.time() - t_save, 2)
    db.exec_("update solves set out_dir=%s, manifest=%s where id=%s", (str(out_dir), db.J(manifest), row["id"]))
    status = "complete" if result["feasible"] else "blocked"
    db.exec_("update runs set status=%s, latest_solve_id=%s, updated_at=now() where id=%s", (status, row["id"], run["id"]))
    rep(1.0, "Routes ready" if result["feasible"] else "No viable route", detail={"solve_id": row["id"]}, force=True)
    return {"solve_id": row["id"], "files": list(files)}


def write_exports(run: dict, solve: dict, manifest: dict, rls: list[dict], out_dir: Path) -> list:
    """layers.gpkg (extracted layers, source and confidence per feature) + the export set."""
    lg = out_dir / "layers.gpkg"
    lg.unlink(missing_ok=True)
    for r in rls:
        g = layers.read_version(r["layer_version_id"])
        if len(g):
            g["layer_source"] = r["source"]
            g["layer_mode"] = r["mode"]
            g["model"] = r["model"]
            g["image_date"] = r["image_date"]
            pyogrio.write_dataframe(g, lg, layer=r["layer_key"], driver="GPKG")
    return list(exports.build_all(run, solve, manifest, out_dir))


def exports_for_solve(sid: int) -> list:
    """Exports for a solve made with exports skipped (quick test solve). Uses the layer versions the
    solve recorded; the run's current layers when the solve predates that record."""
    s = db.one("select * from solves where id=%s", (sid,))
    if not s:
        raise ValueError("solve not found")
    run = db.one("select * from runs where id=%s", (s["run_id"],))
    rls = layers.run_layers(run["id"])
    used = (s["manifest"] or {}).get("layer_version_ids") or {}
    for r in rls:
        if r["layer_key"] in used:
            r["layer_version_id"] = used[r["layer_key"]]
    out_dir = Path(s["out_dir"]) if s["out_dir"] else settings.RUNS_DIR / run["id"] / f"solve_{sid}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(s["manifest"] or {})
    manifest["exports_skipped"] = False
    solve = {"id": sid, "created_at": s["created_at"].isoformat(), "result": s["result"]}
    files = write_exports(run, solve, manifest, rls, out_dir)
    db.exec_("update solves set out_dir=%s, manifest=%s where id=%s", (str(out_dir), db.J(manifest), sid))
    return files
