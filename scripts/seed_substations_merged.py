#!/usr/bin/env python
"""Replace the stored substation output with the statewide merged file (Vivek, 13 Oct 2026).

    python scripts/seed_substations_merged.py /mnt/cog-data/devesh/sub_merge/Gujarat_Substations_FINAL_merged.gpkg

The file holds every detected / OSM-matched substation polygon in Gujarat (7,420) with the model's
confidence, review flag, voltage and source. Coverage: the extent of the features (statewide), so every
run's band is served from here and the substation model is not re-run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402

from beeline_api import reference  # noqa: E402

VERSION = "substations-merged-2026-10-13"
KEEP = ("source", "area_ha", "voltage", "matches_osm", "review", "district", "region", "osm_id", "minwidth", "vv_db", "coh")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpkg")
    a = ap.parse_args()
    g = pyogrio.read_dataframe(a.gpkg).to_crs(4326)
    g["geometry"] = shapely.force_2d(shapely.make_valid(g.geometry.values))
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    conf = g["confidence"].astype(float).clip(0.05, 1.0).values if "confidence" in g else np.full(len(g), 0.85)
    name = g["name"].where(g["name"].astype(str).str.strip().ne("") & g["name"].notna(), None) if "name" in g else None
    out = gpd.GeoDataFrame({"class": "substation", "name": name, "confidence": conf,
                            "src_id": [f"ssm-{i}" for i in range(len(g))],
                            **{k: g[k].values for k in KEEP if k in g}}, geometry=g.geometry.values, crs=4326)
    out = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    cov = shapely.box(*out.total_bounds)
    path = reference.write_version("substations", out, VERSION, {
        "source": "Substation model output — Gujarat merged (Gujarat_Substations_FINAL_merged.gpkg, 13 Oct 2026: stage B + S1 / NISAR gate + OSM match + review)",
        "image_source": "Esri Wayback z18 + NISAR coherence + Sentinel-1 RTC",
        "source_date": "2026-10-13", "model": "substation-dinov3-lora-stageB", "model_version": "merged-2026-10-13",
        "confidence": 0.85}, coverage=cov)
    print(f"substations: {len(out)} (matches_osm {int(out['matches_osm'].sum()) if 'matches_osm' in out else '?'}) -> {path}", flush=True)


if __name__ == "__main__":
    main()
