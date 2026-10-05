"""Step 1 — extract the layers of a run.

Per layer, by the drafter's choice on the Data sources screen:
  auto     reference store (cached layers) or a LULC model plugin (stub in v1), or derived
  replace  the drafter's file, mapped to the Beeline vocabulary; the model is skipped
  add      auto + the drafter's file; where they overlap the drafter's features win

A layer that fails is recorded as failed and the others are kept; a retry redoes only the
failed (or explicitly requested) layers. Layers carried from a previous revision are not
re-extracted unless their source changed.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone

import geopandas as gpd
import pandas as pd

from beeline_engine.config import LineSpec, load_layers, resolve_rules

from . import db, imagery, layers, models, reference, settings, uploads
from .derive import derive_clusters
from .geo import band_polygon
from .jobs import Cancelled, Reporter

log = logging.getLogger(__name__)


def layers_to_extract(run: dict, requested: list[str] | None = None) -> list[str]:
    reg = [l["key"] for l in load_layers()["layers"]]
    if requested:
        return [k for k in reg if k in requested or (k == "clusters" and "buildings" in requested)]
    have = {r["layer_key"]: r for r in layers.run_layers(run["id"])}
    todo = []
    for k in reg:
        r = have.get(k)
        if r is None or r["stale"] or r["status"] == "failed":
            todo.append(k)
    if "buildings" in todo and "clusters" not in todo:
        todo.append("clusters")
    return [k for k in reg if k in todo]


def run_extraction(job: dict, rep: Reporter):
    run = db.one("select * from runs where id=%s", (job["run_id"],))
    db.exec_("update runs set status='extracting', updated_at=now() where id=%s", (run["id"],))
    band, frame = band_polygon((run["a_lon"], run["a_lat"]), (run["b_lon"], run["b_lat"]), run["band_km"], margin_m=500)
    reg = {l["key"]: l for l in load_layers()["layers"]}
    todo = layers_to_extract(run, (job.get("detail") or {}).get("layers"))
    area = frame.geom_to_utm(band).area / 1e6
    weights = {k: max(0.2, reg[k].get("minutes_per_km2", 0.01) * area) for k in todo}
    total_w = sum(weights.values()) or 1.0
    done_w = 0.0
    results = {}

    # imagery snapshot for provenance (and fetch, when real models need pixels)
    rep(0.01, "Checking imagery over the band", force=True)
    img = run.get("imagery")
    if not img:
        try:
            img = imagery.check(frame, band)
            db.exec_("update runs set imagery=%s where id=%s", (db.J(img), run["id"]))
        except Exception as e:
            img = {"error": str(e)}
    if settings.FETCH_IMAGERY_ON_EXTRACT:
        rep(0.02, "Fetching imagery into the bucket", force=True)
        imagery.prefetch_esri(band, progress=lambda f: rep(0.02 + 0.1 * f, "Fetching Esri tiles"), cancel=rep.cancelled)
        s2 = (img or {}).get("sentinel2", {}).get("scenes", [])
        imagery.fetch_sentinel2_visual(s2, band, settings.IMAGERY_DIR / "sentinel2" / run["id"])

    spec = LineSpec(run["kv"], run["circuit"])
    rules = resolve_rules(spec, run["rule_overrides"])
    sources = run["sources"] or {}

    for k in todo:
        rep.check()
        lay = reg[k]
        src = sources.get(k, {"mode": "auto"})
        mode = src.get("mode", "auto")
        w = weights[k]

        def sub(f, _k=k, _w=w):
            rep(0.03 + 0.95 * (done_w + _w * f) / total_w, f"{reg[_k]['name']}",
                detail={"layers": {_k: {"status": "running", "progress": round(f, 2)}}})

        sub(0.0)
        t0 = time.time()
        try:
            gdf, meta, cov = _extract_one(run, k, lay, mode, src, band, img, rules, sub, rep)
            lv = layers.create_version(run["id"], k, mode, lay["producer"], meta, gdf, cov)
            layers.link(run["id"], k, lv)
            results[k] = {"status": "done", "features": 0 if gdf is None else len(gdf), "seconds": round(time.time() - t0, 1)}
        except Cancelled:
            raise
        except Exception as e:
            log.exception("layer %s failed", k)
            lv = layers.create_version(run["id"], k, mode, lay["producer"], {"source": lay.get("desc")}, None, None,
                                       status="failed", error=f"{type(e).__name__}: {e}")
            layers.link(run["id"], k, lv)
            results[k] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        done_w += w
        rep(0.03 + 0.95 * done_w / total_w, reg[k]["name"], detail={"layers": {k: results[k]}}, force=True)

    failed = [k for k, r in results.items() if r["status"] == "failed"]
    db.exec_("update runs set status='review', updated_at=now() where id=%s", (run["id"],))
    rep(1.0, "Layers ready for review" if not failed else f"{len(failed)} layer(s) failed — retry them from the review screen",
        detail={"failed": failed}, force=True)
    return results


def _extract_one(run, k, lay, mode, src, band, img, rules, progress, rep):
    now = datetime.now(timezone.utc).isoformat()
    meta = {"source": lay.get("desc"), "model_run_at": now}
    auto_gdf, auto_cov = None, None
    if mode in ("auto", "add"):
        if lay["producer"] == "derived":
            bl = _current_layer(run["id"], "buildings")
            p = rules["clusters"].params
            auto_gdf = derive_clusters(bl, float(p.get("max_spacing_m", 50)), float(p.get("min_area_m2", 9000)),
                                       lay.get("default_confidence", 0.91))
            bmeta = db.one("""select lv.image_source, lv.image_date, lv.mode from run_layers rl join layer_versions lv
                              on lv.id=rl.layer_version_id where rl.run_id=%s and rl.layer_key='buildings'""", (run["id"],))
            bcov = db.one("""select ST_AsBinary(lv.coverage) as c from run_layers rl join layer_versions lv
                             on lv.id=rl.layer_version_id where rl.run_id=%s and rl.layer_key='buildings'""", (run["id"],))
            import shapely
            auto_cov = shapely.from_wkb(bytes(bcov["c"])) if bcov and bcov["c"] else None
            meta.update(source=f"Derived from buildings at {p.get('max_spacing_m', 50)} m / {p.get('min_area_m2', 9000)} m²",
                        model="cluster-derive", model_version="1.0", confidence=lay.get("default_confidence"),
                        image_source=(bmeta or {}).get("image_source"), image_date=(bmeta or {}).get("image_date"))
        elif lay["producer"] == "model":
            plugin = models.get(lay["model"])
            ctx = models.ModelContext(layer_key=k, band=band, imagery=img or {}, progress=progress, cancelled=rep.cancelled)
            out = plugin.run(ctx)
            auto_gdf, auto_cov = out.gdf, out.coverage
            meta.update(model=out.model, model_version=out.model_version, image_source=out.image_source,
                        image_date=out.image_date, model_run_at=out.run_at, confidence=out.confidence,
                        source=lay.get("desc"))
        else:
            ok, why = reference.ensure_coverage(k, band)
            gdf, m = reference.read_clip(k, band)
            if gdf is None:
                raise FileNotFoundError(f"'{lay['name']}' is not in the reference store ({why})")
            auto_gdf, auto_cov = gdf, reference.coverage_of(m)
            meta.update(source=(m or {}).get("source") or lay.get("desc"), image_date=(m or {}).get("source_date"),
                        model=(m or {}).get("model"), confidence=(m or {}).get("confidence"),
                        model_version=(m or {}).get("version"))
        if auto_cov is not None:
            auto_cov = auto_cov.intersection(band)
        if auto_gdf is not None:
            auto_gdf = auto_gdf.copy()
            auto_gdf["origin"] = "auto"
            if meta.get("confidence") is None:
                meta["confidence"] = lay.get("default_confidence")
    if mode == "auto":
        return auto_gdf, meta, auto_cov
    up_id = src.get("upload_id")
    if not up_id:
        raise ValueError(f"'{lay['name']}' is set to {mode} but no file is attached")
    user = uploads.read_mapped(int(up_id), k, src.get("mapping") or {}, band)
    user["origin"] = "user"
    upr = uploads.get(int(up_id))
    if mode == "replace":
        meta = {"source": f"Your file — {upr['filename']}", "file_name": upr["filename"], "confidence": 1.0,
                "model_run_at": now, "model": None}
        return user, meta, band
    # add: drafter's features win where the two overlap
    if auto_gdf is not None and len(auto_gdf) and len(user):
        hit = gpd.sjoin(auto_gdf[["geometry"]], user[["geometry"]], how="inner", predicate="intersects").index.unique()
        auto_gdf = auto_gdf.drop(index=hit)
    merged = gpd.GeoDataFrame(pd.concat([auto_gdf, user], ignore_index=True), crs=4326) if auto_gdf is not None else user
    meta["source"] = f"{meta.get('source')} + your file {upr['filename']} (yours wins)"
    meta["file_name"] = upr["filename"]
    return merged, meta, auto_cov.union(band) if auto_cov is not None else band


def _current_layer(run_id: str, key: str) -> gpd.GeoDataFrame | None:
    r = db.one("select layer_version_id from run_layers where run_id=%s and layer_key=%s", (run_id, key))
    return layers.read_version(r["layer_version_id"]) if r else None
