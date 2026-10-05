#!/usr/bin/env python
"""Load the existing LULC model outputs into the reference store as stored outputs.

Extraction serves a model layer from here when the stored coverage spans the run's band, so
Bhuj–Lakadia and Gujarat-minus-Kachchh runs do not re-run the models. Sources (read-only):

  village_roads  statewide /mnt/cog-data/guj/vec/*.geojson (Gujarat minus Kachchh)
                 + corridor /mnt/cog-data/roaddet/AOI_centrelines_width_buf_edge.gpkg
  plotted        statewide /mnt/cog-data/guj/pl/Gujarat_Plotted_Land_minus_Kachchh_byregion.gpkg
  solar          corridor  /mnt/drso-data/solar_lora/data/FINAL_solar_v2.gpkg  (SOLAR_FINAL_all)
  substations    corridor  /mnt/cog-data/vivek/stageA/AESL_SUBSTATIONS_FINAL.gpkg
  wind_turbines  corridor  per-scene Exp-G diagnostics, OSM-confirmed (TP) detections only

    python scripts/seed_lulc_outputs.py [--only village_roads plotted ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import mercantile  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from beeline_api import reference  # noqa: E402

VERSION = "lulc-outputs-2026-10-05"
CORRIDOR_AOI = "/mnt/drso-data/wayback/aoi.geojson"


def corridor_aoi():
    return unary_union(list(gpd.read_file(CORRIDOR_AOI).to_crs(4326).geometry))


def _std(gdf, cls, name=None, conf=None, src_prefix="", keep=()):
    out = gpd.GeoDataFrame({
        "class": cls, "name": name if name is not None else None,
        "confidence": conf if conf is not None else None,
        "src_id": [f"{src_prefix}{i}" for i in range(len(gdf))],
        **{k: gdf[k].values for k in keep if k in gdf}}, geometry=gdf.geometry.values, crs=gdf.crs)
    return out.to_crs(4326)


# ---- village roads -----------------------------------------------------------------------
KEEP_ROADS = ("surface_width_m", "width_p90_m", "lane_tier", "bridged", "length_m")


def _read_block(p: str):
    try:
        g = pyogrio.read_dataframe(p)
    except Exception:
        return None
    if g.crs is None:
        g = g.set_crs(3857)
    # skeleton centrelines carry a vertex per pixel; 0.5 m keeps the shape (setbacks are 15 m+)
    g["geometry"] = g.geometry.simplify(0.5)
    return g


def block_box(tag: str):
    bx, by = map(int, tag.split("_"))
    w = mercantile.bounds(bx, by, 18).west
    n = mercantile.bounds(bx, by, 18).north
    e = mercantile.bounds(bx + 15, by + 15, 18).east
    s = mercantile.bounds(bx + 15, by + 15, 18).south
    return box(w, s, e, n)


def seed_roads():
    files = sorted(Path("/mnt/cog-data/guj/vec").glob("*.geojson"))
    print(f"village_roads: {len(files)} statewide blocks", flush=True)
    parts, cov = [], []
    with ProcessPoolExecutor(8) as ex:
        for i, (f, g) in enumerate(zip(files, ex.map(_read_block, map(str, files), chunksize=64))):
            cov.append(block_box(f.stem))
            if g is not None and len(g):
                parts.append(g)
            if i % 2000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    state = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=3857)
    corr = pyogrio.read_dataframe("/mnt/cog-data/roaddet/AOI_centrelines_width_buf_edge.gpkg")
    corr["geometry"] = corr.geometry.simplify(0.5)
    a = _std(state, "village_road", src_prefix="guj-", keep=KEEP_ROADS)
    b = _std(corr, "village_road", src_prefix="bl-", keep=KEEP_ROADS)
    allg = gpd.GeoDataFrame(pd.concat([a, b], ignore_index=True), crs=4326)
    coverage = unary_union(cov + [corridor_aoi()]).simplify(0.0005)
    reference.write_version("village_roads", allg, VERSION, {
        "source": "Village roads model output — statewide Gujarat minus Kachchh (Sep 2026) + Bhuj–Lakadia corridor (edge-fixed)",
        "image_source": "Esri Wayback z18 release 26334 (redirects to older releases where 26334 has no tile)",
        "source_date": "2026-08-05", "model": "microsoft-roaddetections-res-unet-6", "model_version": "HF_GG08_60",
        "confidence": 0.80}, coverage=coverage)
    print(f"village_roads: {len(allg):,} edges", flush=True)


# ---- plotted land ------------------------------------------------------------------------
def seed_plotted():
    f = "/mnt/cog-data/guj/pl/Gujarat_Plotted_Land_minus_Kachchh_byregion.gpkg"
    g = pyogrio.read_dataframe(f, layer="plotted_land_clusters")
    out = _std(g, "plotted_land", src_prefix="", keep=("cluster_id", "n_plots", "area_ha", "built_pct", "built_class",
                                                      "vacant_plots_pct"))
    out["src_id"] = g["cluster_id"].astype(str).values
    files = sorted(Path("/mnt/cog-data/guj/vec").glob("*.geojson"))
    coverage = unary_union([block_box(p.stem) for p in files]).simplify(0.0005)
    reference.write_version("plotted", out, VERSION, {
        "source": "Plotted land (plot-layout geometry over the statewide road run) — Gujarat minus Kachchh, by-region run",
        "image_source": "derived from village roads (Esri Wayback z18) + Google/MS buildings + ESA WorldCover 2021",
        "source_date": "2026-09-30", "model": "plotted-land-geometry", "model_version": "guj chord IDEAL=1",
        "confidence": 0.61}, coverage=coverage)
    print(f"plotted: {len(out):,} clusters", flush=True)


def seed_plotted_corridor(work_dir: str):
    """Add the Bhuj–Lakadia plotted-land run (plugin output in work_dir) to the statewide version."""
    import json
    from shapely.geometry import shape
    cur_v = reference.current_version("plotted")
    state, sm = reference.read_clip("plotted", box(60, 5, 100, 40), version=cur_v)
    state = state[state["src_id"].astype(str).str.startswith("GJ-")] if len(state) else state
    if "attrs" in state:      # read_clip packs extra columns; unpack for writing
        ex = pd.DataFrame(list(state["attrs"]))
        state = gpd.GeoDataFrame(pd.concat([state.drop(columns="attrs").reset_index(drop=True), ex], axis=1),
                                 geometry="geometry", crs=4326)
    c = pyogrio.read_dataframe(Path(work_dir) / "out.gpkg", layer="features")
    res = json.loads((Path(work_dir) / "result.json").read_text())
    c["src_id"] = "BL-" + c["src_id"].astype(str)
    keep = ["class", "name", "confidence", "src_id", "n_plots", "area_ha", "built_pct", "built_class",
            "vacant_plots_pct", "geometry"]
    c = c[[k for k in keep if k in c]]
    allg = gpd.GeoDataFrame(pd.concat([state, c], ignore_index=True), crs=4326)
    cov = unary_union([reference.coverage_of(sm), shape(res["coverage"])])
    reference.write_version("plotted", allg, VERSION + "-bl", {
        "source": "Plotted land — statewide Gujarat minus Kachchh (Sep 2026) + Bhuj–Lakadia corridor run "
                  "(5 Oct 2026, edge-fixed corridor roads, Google+MS buildings, WorldCover 2021 salt filter)",
        "image_source": "derived from village roads (Esri Wayback z18) + Google/MS buildings + ESA WorldCover 2021",
        "source_date": "2026-09-30", "model": "plotted-land-geometry", "model_version": "guj chord IDEAL=1",
        "confidence": 0.61, "corridor_warnings": res.get("warnings")}, coverage=cov)
    print(f"plotted: {len(state):,} statewide + {len(c):,} corridor clusters", flush=True)


# ---- solar -------------------------------------------------------------------------------
def seed_solar():
    g = pyogrio.read_dataframe("/mnt/drso-data/solar_lora/data/FINAL_solar_v2.gpkg", layer="SOLAR_FINAL_all")
    conf = np.where(g["cls"].astype(str).str.contains("OSM"), 1.0, 0.8)
    out = _std(g, "solar_park", conf=conf, src_prefix="sol-", keep=("cls", "area_m2", "n_obs", "nir_med", "ndvi_med"))
    reference.write_version("solar", out, VERSION, {
        "source": "Solar park model output — Bhuj–Lakadia corridor (FINAL_solar_v2: 49 OSM-confirmed + 65 NIR-verified)",
        "image_source": "Esri Wayback z18 release 26334 + Sentinel-2 L2A (Oct 2025 – Jun 2026)",
        "source_date": "2026-08-05", "model": "solar-dinov3-lora", "model_version": "v1 epoch07", "confidence": 0.85},
        coverage=corridor_aoi())
    print(f"solar: {len(out)} parks", flush=True)


# ---- substations -------------------------------------------------------------------------
def seed_substations():
    f = "/mnt/cog-data/vivek/stageA/AESL_SUBSTATIONS_FINAL.gpkg"
    a = pyogrio.read_dataframe(f, layer="model_confirms_osm")
    b = pyogrio.read_dataframe(f, layer="model_new_detections")
    osm = pyogrio.read_dataframe(f, layer="osm_substations").to_crs(4326)
    g = gpd.GeoDataFrame(pd.concat([a, b], ignore_index=True), crs=a.crs).to_crs(4326)
    conf = np.where(g["source"].astype(str).str.contains("osm", case=False), 0.9, 0.75)
    out = _std(g, "substation", conf=conf, src_prefix="ss-", keep=("source", "area_ha", "coh", "vv_db", "minwidth"))
    # OSM name where the detection confirms an OSM substation
    j = gpd.sjoin(out[["geometry"]], osm[["name", "geometry"]], how="left", predicate="intersects")
    names = j.groupby(level=0)["name"].first()
    out["name"] = names.reindex(out.index).values
    cov = unary_union(list(pyogrio.read_dataframe(f, layer="aoi_corridor").to_crs(4326).geometry))
    reference.write_version("substations", out, VERSION, {
        "source": "Substation model output — Bhuj–Lakadia corridor (AESL_SUBSTATIONS_FINAL: stage B + NISAR/S1 gate + manual QA)",
        "image_source": "Esri Wayback z18 release 26334 + NISAR coherence + Sentinel-1 RTC",
        "source_date": "2026-08-05", "model": "substation-dinov3-lora-stageB", "model_version": "epoch22",
        "confidence": 0.85}, coverage=cov)
    print(f"substations: {len(out)}", flush=True)


# ---- wind turbines -----------------------------------------------------------------------
def seed_turbines():
    from beeline_api import imagery_store  # noqa: F401  (catalog holds the PAN footprints)
    from beeline_api import db
    d = Path("/mnt/drso-data/devesh/turbine_expg/out_all")
    pts, scenes = [], []
    for p in sorted(d.glob("*_expG_diagnostic.gpkg")):
        sid = p.name.split("_")[0]
        g = pyogrio.read_dataframe(p)
        if "status" not in g:
            continue
        tp = g[g["status"] == "TP"].copy()
        scenes.append(sid)
        if len(tp):
            tp["scene_id"] = sid
            pts.append(tp.to_crs(4326))
    g = gpd.GeoDataFrame(pd.concat(pts, ignore_index=True), crs=4326)
    # Detection points sit 50-100 m from the tower, offset along the shadow (plugin survey,
    # 5 Oct 2026). These detections are OSM-confirmed, so place each on its OSM turbine.
    osm = gpd.read_file("/home/vivek_jain/seggpt-local/turbines_work/generator_source_wind_32642.gpkg").to_crs(32642)
    gu = g.to_crs(32642)
    j = gpd.sjoin_nearest(gu[["geometry"]], osm[["osm_id", "geometry"]] if "osm_id" in osm else osm[["geometry"]],
                          how="left", max_distance=100, distance_col="osm_dist")
    j = j[~j.index.duplicated(keep="first")]
    snapped = []
    for i, row in j.iterrows():
        if row.get("index_right") == row.get("index_right") and row.get("osm_dist") == row.get("osm_dist"):
            snapped.append(osm.geometry.loc[int(row["index_right"])])
        else:
            snapped.append(gu.geometry.loc[i])
    g["osm_dist_m"] = j["osm_dist"].values
    g["osm_id"] = j["osm_id"].values if "osm_id" in j else None
    g = gpd.GeoDataFrame(g.drop(columns="geometry"), geometry=gpd.GeoSeries(snapped, crs=32642).to_crs(4326).values, crs=4326)
    # cross-scene duplicates (identical-footprint scenes): keep one point per 95 m
    utm = g.to_crs(32642)
    keep, taken = [], np.zeros(len(utm), bool)
    xy = np.c_[utm.geometry.x, utm.geometry.y]
    for i in range(len(utm)):
        if taken[i]:
            continue
        d2 = ((xy - xy[i]) ** 2).sum(1)
        taken |= d2 <= 95 ** 2
        keep.append(i)
    g = g.iloc[keep].reset_index(drop=True)
    out = _std(g, "wind_turbine", conf=np.full(len(g), 0.9), src_prefix="wt-",
               keep=("scene_id", "area", "n_frag", "osm_id", "osm_dist_m"))
    out["osm_confirmed"] = True
    rows = db.all_("select ST_AsText(footprint) as w from imagery_items where product='cartosat3_pan' and item_key = any(%s)",
                   (scenes,))
    from shapely import wkt
    cov = unary_union([wkt.loads(r["w"]) for r in rows]) if rows else corridor_aoi()
    reference.write_version("wind_turbines", out, VERSION, {
        "source": "Wind turbine model output — Bhuj–Lakadia PAN scenes, OSM-confirmed detections only, placed on the "
                  "matched OSM turbine (unconfirmed detections withheld until the curated corridor final is loaded)",
        "image_source": "Cartosat-3 PAN 0.45 m", "source_date": None, "model": "turbine-dinov3-lora-round4",
        "model_version": "epoch25 + Exp-G", "confidence": 0.9}, coverage=cov)
    print(f"wind_turbines: {len(out)} (OSM-confirmed, deduplicated) from {len(scenes)} scenes", flush=True)


SEEDERS = {"solar": seed_solar, "substations": seed_substations, "wind_turbines": seed_turbines,
           "plotted": seed_plotted, "village_roads": seed_roads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--plotted-corridor", help="work_dir of a plotted_land plugin run on Bhuj–Lakadia")
    a = ap.parse_args()
    if a.plotted_corridor:
        seed_plotted_corridor(a.plotted_corridor)
        return
    for k, fn in SEEDERS.items():
        if a.only and k not in a.only:
            continue
        fn()


if __name__ == "__main__":
    main()
