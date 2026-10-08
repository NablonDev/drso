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

from pathlib import Path

import geopandas as gpd
import pyogrio
import pandas as pd

from beeline_engine.config import LineSpec, load_layers, resolve_rules

from . import aviation, cell_towers, db, imagery, imagery_store, layers, lulc_runner, models, osm_highways, osm_power, reference, settings, uploads, user_shapes
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


def run_imagery_fetch(job: dict, rep: Reporter):
    """Licensed online acquisition into the imagery catalog (the only place imagery is fetched)."""
    run = db.one("select * from runs where id=%s", (job["run_id"],))
    band, _ = band_polygon((run["a_lon"], run["a_lat"]), (run["b_lon"], run["b_lat"]), run["band_km"], margin_m=500)
    products = (job.get("detail") or {}).get("imagery_fetch") or []
    rep(0.01, "Fetching imagery under the operator licences", force=True)
    out = imagery_store.acquire(products, band, run["id"], progress=lambda f, msg: rep(0.02 + 0.96 * f, msg),
                                cancel=rep.cancelled)
    rep.check()
    rep(1.0, "Imagery stored on this machine", detail={"imagery": out}, force=True)
    return {"imagery": out}


def run_extraction(job: dict, rep: Reporter):
    if (job.get("detail") or {}).get("imagery_fetch"):
        return run_imagery_fetch(job, rep)
    run = db.one("select * from runs where id=%s", (job["run_id"],))
    db.exec_("update runs set status='extracting', updated_at=now() where id=%s", (run["id"],))
    band, frame = band_polygon((run["a_lon"], run["a_lat"]), (run["b_lon"], run["b_lat"]), run["band_km"], margin_m=500)
    reg = {l["key"]: l for l in load_layers()["layers"]}
    todo = layers_to_extract(run, (job.get("detail") or {}).get("requested"))
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
            gdf = user_shapes.merge(run, k, gdf, band)          # hand-added shapes (temporary for this run, permanent)
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
        elif lay["producer"] == "osm":
            kvs = tuple(lay.get("voltages") or osm_power.VOLTAGES)     # power: HT, power_lt: LT
            if _refresh_requested(rep):
                minx, miny, maxx, maxy = band.bounds
                res = osm_power.refresh_aoi((minx, miny, maxx, maxy), run_id=run["id"],
                                            progress=lambda f, st: progress(0.8 * f), voltages=kvs)
                meta["refresh"] = res
            auto_gdf, om = osm_power.read_band(band, kvs)
            auto_cov = None
            if not om["india_loaded"] and not om["last_refresh"]:
                raise FileNotFoundError("OSM power store is empty — load India (python -m beeline_api.osm_power "
                                        "load-india) or press Refresh from OSM for this run")
            meta.update(source=om["source"], image_source="OpenStreetMap", image_date=om["osm_timestamp"],
                        model=None, model_version=None, confidence=lay.get("default_confidence"))
            meta["stats_extra"] = {"by_voltage": om["counts"]}
        elif lay["producer"] == "osm_highways":
            if _refresh_requested(rep):
                minx, miny, maxx, maxy = band.bounds
                res = osm_highways.refresh_aoi((minx, miny, maxx, maxy), run_id=run["id"],
                                               progress=lambda f, st: progress(0.8 * f))
                meta["refresh"] = res
            auto_gdf, hm = osm_highways.read_band(band)
            auto_cov = None
            if not hm["india_loaded"] and not hm["last_refresh"]:
                raise FileNotFoundError("OSM highway store is empty — load India (python -m beeline_api.osm_highways "
                                        "load-india) or press Refresh from OSM for this run")
            meta.update(source=hm["source"], image_source="OpenStreetMap", image_date=hm["osm_timestamp"],
                        model=None, model_version=None, confidence=lay.get("default_confidence"))
            meta["stats_extra"] = {"by_class": hm["counts"]}
        elif lay["producer"] in ("esri_airports", "osm_runways"):
            is_ap = lay["producer"] == "esri_airports"
            rid, pk, scale = ("airport_noc", "radius_km", 1000.0) if is_ap else ("airport_prohibition", "distance_m", 1.0)
            reach = max(float(lay.get("reach_m", 0)), float((rules[rid].params if rid in rules else {}).get(pk, 0)) * scale)
            if _refresh_requested(rep):
                res = (aviation.refresh_airports if is_ap else aviation.refresh_runways)(
                    tuple(aviation._reach(band, reach).bounds), run_id=run["id"], progress=lambda f, st: progress(0.8 * f))
                meta["refresh"] = res
            auto_gdf, am = (aviation.read_airports if is_ap else aviation.read_runways)(band, reach)
            auto_cov = None
            if not am["india_loaded"] and not am.get("last_refresh"):
                raise FileNotFoundError(f"{lay['name']} store is empty — load India (python -m beeline_api.aviation "
                                        f"{'load-airports' if is_ap else 'load-runways --gpkg airports.gpkg'})")
            meta.update(source=am["source"], image_source="Esri Living Atlas" if is_ap else "OpenStreetMap",
                        image_date=am["data_date"], model=None, model_version=None,
                        confidence=lay.get("default_confidence"))
            meta["stats_extra"] = {"by_source" if is_ap else "by_width": am["counts"], "reach_m": reach}
        elif lay["producer"] == "tarangsanchar":
            if _refresh_requested(rep):
                minx, miny, maxx, maxy = band.bounds
                res = cell_towers.refresh_aoi((minx, miny, maxx, maxy), run_id=run["id"],
                                              progress=lambda f, st: progress(0.8 * f))
                meta["refresh"] = res
            auto_gdf, tm = cell_towers.read_band(band)
            auto_cov = None
            if not tm["india_loaded"] and not tm["last_refresh"]:
                raise FileNotFoundError("Cell tower store is empty — load India (python -m beeline_api.cell_towers "
                                        "load-india)")
            meta.update(source=tm["source"], image_source="Tarang Sanchar", image_date=tm["data_date"],
                        model=None, model_version=None, confidence=lay.get("default_confidence"))
            meta["stats_extra"] = {"by_structure": tm["counts"]}
        elif lay["producer"] == "lulc_model":
            auto_gdf, auto_cov, m2 = _lulc_model_layer(run, k, lay, band, progress, rep)
            meta.update(m2)
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


def _lulc_model_layer(run, k, lay, band, progress, rep):
    """Stored output when it covers the band (unless forced); otherwise run the model plugin on
    the imagery held on this machine. No imagery -> empty layer with no coverage (reported)."""
    force = bool((rep.job.get("detail") or {}).get("force_models"))
    stored, sm = reference.read_clip(k, band)
    scov = reference.coverage_of(sm)
    s_frac = 0.0 if scov is None else scov.intersection(band).area / band.area
    if stored is not None and s_frac >= 0.995 and not force:
        return stored, scov, {"source": f"Stored output — {(sm or {}).get('source')}",
                              "image_source": (sm or {}).get("image_source"),
                              "image_date": (sm or {}).get("source_date"), "model": (sm or {}).get("model"),
                              "model_version": (sm or {}).get("model_version") or (sm or {}).get("version"),
                              "confidence": (sm or {}).get("confidence"),
                              "lulc_stats": {"served_from": "stored", "stored_version": (sm or {}).get("version")}}
    name = lay["pipeline"]
    m = lulc_runner.model_cfg(name)
    primary = (m.get("imagery") or [None])[0]
    have = imagery_store.coverage(band)
    pc = next((c for c in have if c["product"] == primary), {"covered_pct": 0})
    if primary and pc["covered_pct"] < 0.5:
        warn = (f"no {primary} imagery on this machine for this band — fetch it on the Images screen; "
                f"the model was not run")
        if stored is not None and len(stored):
            return stored, scov, {"source": f"Stored output (partial, {100 * s_frac:.0f}% of band) — {(sm or {}).get('source')}",
                                  "model": (sm or {}).get("model"), "image_date": (sm or {}).get("source_date"),
                                  "lulc_stats": {"served_from": "stored_partial", "warnings": [warn]}}
        return _empty_gdf(), None, {"source": f"{lay['name']}: not run — {warn}", "model": m.get("name"),
                                    "lulc_stats": {"served_from": "none", "warnings": [warn]}}
    missing = [k for k in (m.get("required_layers") or [])
               if (g := _current_layer(run["id"], k)) is None or not len(g)]
    if missing:
        warn = f"no {', '.join(missing)} for this band (input to this model); the model was not run"
        if stored is not None and len(stored):
            return stored, scov, {"source": f"Stored output (partial, {100 * s_frac:.0f}% of band) — {(sm or {}).get('source')}",
                                  "model": (sm or {}).get("model"), "image_date": (sm or {}).get("source_date"),
                                  "lulc_stats": {"served_from": "stored_partial", "warnings": [warn]}}
        return _empty_gdf(), None, {"source": f"{lay['name']}: not run — {warn}", "model": m.get("name"),
                                    "lulc_stats": {"served_from": "none", "warnings": [warn]}}
    if not settings.RUN_MODELS:
        warn = "model runs are switched off on this machine (BEELINE_RUN_MODELS=0); stored output only"
        if stored is not None and len(stored):
            return stored, scov, {"source": f"Stored output (partial, {100 * s_frac:.0f}% of band) — {(sm or {}).get('source')}",
                                  "model": (sm or {}).get("model"), "image_date": (sm or {}).get("source_date"),
                                  "lulc_stats": {"served_from": "stored_partial", "warnings": [warn]}}
        return _empty_gdf(), None, {"source": f"{lay['name']}: not run — {warn}", "model": m.get("name"),
                                    "lulc_stats": {"served_from": "none", "warnings": [warn]}}
    inputs = _plugin_inputs(run, name, m, band)
    res = lulc_runner.run(name, band, run["id"], rep.job["id"], inputs,
                          progress=lambda f, msg: progress(f), cancelled=rep.cancelled)
    return res.gdf, res.coverage, {
        "source": f"{lay.get('desc')} — model run {res.meta.get('model')} {res.meta.get('model_version') or ''}".strip(),
        "model": res.meta.get("model"), "model_version": res.meta.get("model_version"),
        "image_source": res.meta.get("image_source"), "image_date": res.meta.get("image_date"),
        "confidence": lay.get("default_confidence"),
        "lulc_stats": {"served_from": "model", "warnings": res.meta.get("warnings"), "counts": res.meta.get("counts"),
                        "artifacts": res.artifacts, "work_dir": str(res.work_dir)}}


def _empty_gdf():
    return gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": []}, geometry=[], crs=4326)


def _layer_to_gpkg(run_id: str, key: str, dst: Path) -> str | None:
    g = _current_layer(run_id, key)
    if g is None or not len(g):
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    g = g.drop(columns=[c for c in ("attrs",) if c in g])
    pyogrio.write_dataframe(g, dst, layer="features", driver="GPKG")
    return str(dst)


def _saltscape_gpkg(spec: dict, band, dst: Path) -> str | None:
    """Salt-scape polygons for the plotted-land salt-works filter, from the reference store: {layer:
    classes or None for all}. Read 1 km past the band (the filter's ring reaches that far). Layer
    `features` + layer `coverage` (union of the source layers' coverage)."""
    from shapely.ops import unary_union
    area = band.buffer(0.011)
    parts, covs = [], []
    for key, classes in spec.items():
        g, meta = reference.read_clip(key, area)
        if g is None:
            continue
        cv = reference.coverage_of(meta)
        if cv is not None:
            covs.append(cv.intersection(area))
        if classes:
            g = g[g["class"].isin(classes)]
        g = g[g.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(g):
            parts.append(g[["class", "geometry"]].assign(layer=key))
    if not parts and not covs:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    feats = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326) if parts else \
        gpd.GeoDataFrame({"class": [], "layer": []}, geometry=[], crs=4326)
    pyogrio.write_dataframe(feats, dst, layer="features", driver="GPKG", promote_to_multi=True)
    if covs:
        pyogrio.write_dataframe(gpd.GeoDataFrame({"id": [1]}, geometry=[unary_union(covs)], crs=4326), dst,
                                layer="coverage", driver="GPKG", append=True, promote_to_multi=True)
    return str(dst)


def _plugin_inputs(run: dict, name: str, m: dict, band) -> dict:
    """Upstream layers of this run + local reference inputs the plugin may use."""
    d = settings.RUNS_DIR / run["id"] / "lulc" / "_inputs"
    ins: dict = {}
    if name == "plotted_land":
        ins["village_roads_gpkg"] = _layer_to_gpkg(run["id"], "village_roads", d / "village_roads.gpkg")
        ins["buildings_gpkg"] = _layer_to_gpkg(run["id"], "buildings", d / "buildings.gpkg")
        r = db.one("""select lv.stats from run_layers rl join layer_versions lv on lv.id=rl.layer_version_id
                      where rl.run_id=%s and rl.layer_key='village_roads'""", (run["id"],))
        lulc = ((r or {}).get("stats") or {}).get("lulc") or {}
        ins["village_roads_prob"] = (lulc.get("artifacts") or {}).get("prob_raster") or []
        ins["saltscape_gpkg"] = _saltscape_gpkg(m.get("saltscape") or {}, band, d / "saltscape.gpkg")
    ref = settings.DATA / "reference_inputs"
    for key, fname in (("osm_solar_gpkg", "osm_solar_india.gpkg"), ("osm_wind_gpkg", "osm_wind.gpkg"),
                       ("osm_substations_gpkg", "osm_substations.gpkg")):
        if (ref / fname).exists():
            ins[key] = str(ref / fname)
    return ins


def _refresh_requested(rep) -> bool:
    d = rep.job.get("detail") or {}
    return bool(d.get("refresh") or d.get("refresh_osm"))      # refresh_osm: jobs queued before the rename


def _current_layer(run_id: str, key: str) -> gpd.GeoDataFrame | None:
    r = db.one("select layer_version_id from run_layers where run_id=%s and layer_key=%s", (run_id, key))
    return layers.read_version(r["layer_version_id"]) if r else None
