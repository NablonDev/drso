#!/usr/bin/env python
"""Ingest the forest package into five reference layers (Vivek, 6 Oct 2026).

    python scripts/ingest_forest.py "<LULC output>/transfers/Forests/gj_forest_delivery.gpkg"

Package layers -> Beeline layers (no generic forest / protected layer)
  reserve_forest            -> reserve_forest      Reserved Forest, IFA 1927 s.20 (39,711)
  forest_protected_area     -> forest_in_pa        Reserved Forest inside a NP / WLS (2,586)
  national_park             -> national_park       WLPA 1972 s.35 (4)
  wls_conservation_reserve  -> wildlife_sanctuary  WLPA 1972 s.18/26A (24; no conservation reserve)
  social_forest             -> social_forest       FSI 2023 cover within 100 m of road / rail / canal (23,101)
Not ingested: _supplementary_pa_fill (OSM gap-fills, ~1 km accuracy; the package keeps them out of its
headline layers) and _gap_protected_forest_IFA_s29 (empty).
Kept attributes: designation, statute, section, area_ha, chosen_source, positional_accuracy_m, licence,
plus density_class / sf_class / nearest_corridor for social forest. The package's text confidence
(legal / derived) is kept as `source_confidence`.
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

VERSION = "forest-delivery-2026-10-05"
KEEP = ["designation", "statute", "section", "area_ha", "chosen_source", "positional_accuracy_m", "licence",
        "density_class", "sf_class", "nearest_corridor"]
# (package layer, beeline key, class, id prefix, layer confidence)
MAP = [
    ("reserve_forest", "reserve_forest", "reserve_forest", "rf", 0.90),
    ("forest_protected_area", "forest_in_pa", "reserve_forest_in_pa", "rfpa", 0.90),
    ("national_park", "national_park", "national_park", "np", 0.95),
    ("wls_conservation_reserve", "wildlife_sanctuary", "wildlife_sanctuary", "wls", 0.95),
    ("social_forest", "social_forest", "social_forest", "sf", 0.70),
]


def _layer(gpkg, layer, cls, prefix):
    g = pyogrio.read_dataframe(gpkg, layer=layer)
    g["geometry"] = shapely.force_2d(shapely.make_valid(g.geometry.values))
    g = g[~g.geometry.is_empty & g.geometry.geom_type.isin(["Polygon", "MultiPolygon", "GeometryCollection"])]
    g["geometry"] = [shapely.unary_union([p for p in getattr(x, "geoms", [x]) if p.geom_type in ("Polygon", "MultiPolygon")])
                     for x in g.geometry.values]
    g = g[~g.geometry.is_empty].reset_index(drop=True)
    name = g["name"].where(g["name"].astype(str).str.strip().ne(""), None) if "name" in g else None
    return gpd.GeoDataFrame({"class": cls, "name": name, "confidence": None,
                             "src_id": [f"{prefix}-{i}" for i in range(len(g))],
                             "source_confidence": g.get("confidence"),
                             **{k: g[k].values for k in KEEP if k in g}},
                            geometry=g.geometry.values, crs=g.crs).to_crs(4326)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", help="gj_forest_delivery.gpkg")
    a = ap.parse_args()
    src = Path(a.package)
    for layer, key, cls, prefix, conf in MAP:
        g = _layer(src, layer, cls, prefix)
        prov = {"source": f"Gujarat forest delivery ({src.name}, layer {layer})",
                "source_date": "2026-10-05", "confidence": conf,
                "note": "legal boundaries (BharatMaps RFA / GatiShakti), ~30 m positional accuracy"
                        if key != "social_forest" else
                        "derived: FSI 2023 forest cover within 100 m of road / rail / canal, outside PAs; ~46 m"}
        out = reference.write_version(key, g, VERSION, prov, coverage=shapely.box(*g.total_bounds))
        print(f"{key:20s} {len(g):7d}  -> {out}", flush=True)


if __name__ == "__main__":
    main()
