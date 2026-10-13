#!/usr/bin/env python
"""Ingest the second forest package (Vivek, 13 Oct 2026): the five layers of 6 Oct updated, plus the
eco-sensitive zones as a new reference layer.

    python scripts/ingest_forest_5cat.py "<LULC output>/transfers/Forests/gj_forest_5cat.gpkg"

Package layers -> Beeline layers
  reserve_forest            -> reserve_forest       Reserved Forest, IFA 1927 s.20
  forest_protected_area     -> forest_in_pa         Reserved Forest inside a NP / WLS
  national_park             -> national_park        WLPA 1972 s.35
  wls_conservation_reserve  -> wildlife_sanctuary   WLPA 1972 s.18/26A
  social_forest             -> social_forest        FSI 2023 cover within 100 m of road / rail / canal
  eco_sensitive_zone        -> eco_sensitive_zone   ESZ notifications (new, No-Go like the parks and sanctuaries)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402

from beeline_api import reference  # noqa: E402

VERSION = "forest-5cat-2026-10-13"
KEEP = ["designation", "statute", "section", "ifa_section", "wlpa_section", "notification_ref", "notification_status",
        "protected_area_name", "pa_class", "area_ha", "chosen_source", "positional_accuracy_m", "licence",
        "density_class", "sf_class", "nearest_corridor", "district", "taluka", "village", "esz_total_km2", "is_overlay"]
# (package layer, beeline key, class, id prefix, layer confidence)
MAP = [
    ("reserve_forest", "reserve_forest", "reserve_forest", "rf", 0.90),
    ("forest_protected_area", "forest_in_pa", "reserve_forest_in_pa", "rfpa", 0.90),
    ("national_park", "national_park", "national_park", "np", 0.95),
    ("wls_conservation_reserve", "wildlife_sanctuary", "wildlife_sanctuary", "wls", 0.95),
    ("social_forest", "social_forest", "social_forest", "sf", 0.70),
    ("eco_sensitive_zone", "eco_sensitive_zone", "eco_sensitive_zone", "esz", 0.90),
]


def _layer(gpkg, layer, cls, prefix):
    g = pyogrio.read_dataframe(gpkg, layer=layer)
    g["geometry"] = shapely.force_2d(shapely.make_valid(g.geometry.values))
    g = g[~g.geometry.is_empty & g.geometry.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])]
    g["geometry"] = [shapely.unary_union([p for p in getattr(x, "geoms", [x]) if p.geom_type in ("Polygon", "MultiPolygon")])
                     for x in g.geometry.values]
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    name = g["name"].where(g["name"].astype(str).str.strip().ne(""), None) if "name" in g else None
    if name is None and "protected_area_name" in g:
        name = g["protected_area_name"].where(g["protected_area_name"].astype(str).str.strip().ne(""), None)
    return gpd.GeoDataFrame({"class": cls, "name": name, "confidence": None,
                             "src_id": [f"{prefix}-{i}" for i in range(len(g))],
                             "source_confidence": g.get("confidence"),
                             **{k: g[k].values for k in KEEP if k in g}},
                            geometry=g.geometry.values, crs=g.crs).to_crs(4326)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", help="gj_forest_5cat.gpkg")
    a = ap.parse_args()
    src = Path(a.package)
    for layer, key, cls, prefix, conf in MAP:
        g = _layer(src, layer, cls, prefix)
        prov = {"source": f"Gujarat forest package 2 ({src.name}, layer {layer})",
                "source_date": "2026-10-13", "confidence": conf,
                "note": {"social_forest": "derived: FSI 2023 forest cover within 100 m of road / rail / canal, outside PAs",
                         "eco_sensitive_zone": "ESZ notifications round the parks and sanctuaries (the collar outside the PA)"}
                .get(key, "legal boundaries (BharatMaps RFA / GatiShakti / State Forest Dept)")}
        out = reference.write_version(key, g, VERSION, prov, coverage=shapely.box(*g.total_bounds))
        print(f"{key:20s} {len(g):7d}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
