#!/usr/bin/env python
"""Ingest one statewide gpkg into the reference store.

Example:
  python scripts/ingest_reference.py forest "LULC output/Forest Areas/gj_forest_delivery.gpkg" \
      --const-class reserve_forest --name-field NAME --version 2026-10-05 \
      --source "Forest Dept / SOI" --source-date 2026-09-01 --coverage-file gujarat_boundary.geojson

Class mapping: either --const-class, or --class-field with --class-map 'SRC=beeline,SRC2=beeline'.
Source values already in the Beeline vocabulary map to themselves; anything else falls back to
the layer's first class and is reported.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from beeline_api import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("key")
    ap.add_argument("path")
    ap.add_argument("--layer")
    ap.add_argument("--class-field")
    ap.add_argument("--class-map", default="")
    ap.add_argument("--const-class")
    ap.add_argument("--name-field")
    ap.add_argument("--conf-field")
    ap.add_argument("--id-field")
    ap.add_argument("--version", required=True)
    ap.add_argument("--source", required=True, help="human description of the source")
    ap.add_argument("--source-date", help="survey / imagery / publication date (YYYY-MM-DD)")
    ap.add_argument("--model", help="model name+version if a model produced the file")
    ap.add_argument("--model-run-date")
    ap.add_argument("--confidence", type=float, help="layer-level confidence")
    ap.add_argument("--coverage-file", help="polygon file of where the data is valid (default: data bbox)")
    a = ap.parse_args()

    gdf = pyogrio.read_dataframe(a.path, layer=a.layer)
    cmap = dict(kv.split("=", 1) for kv in a.class_map.split(",") if kv)
    norm = reference.normalise(gdf, a.key, class_field=a.class_field, class_map=cmap, const_class=a.const_class,
                               name_field=a.name_field, conf_field=a.conf_field, id_field=a.id_field)
    cov = None
    if a.coverage_file:
        c = gpd.read_file(a.coverage_file).to_crs(4326)
        cov = unary_union(list(c.geometry))
    prov = {"source": a.source, "source_file": str(Path(a.path).name), "source_date": a.source_date,
            "model": a.model, "model_run_date": a.model_run_date, "confidence": a.confidence}
    out = reference.write_version(a.key, norm, a.version, prov, coverage=cov)
    print(f"{a.key}: {len(norm)} features -> {out}")
    if norm.attrs.get("unknown_classes"):
        print(f"  unmapped source classes (fell back to the default class): {norm.attrs['unknown_classes']}")


if __name__ == "__main__":
    main()
