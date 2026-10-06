#!/usr/bin/env python
"""Ingest the water-features package into eight reference layers (Vivek, 6 Oct 2026).

    python scripts/ingest_water_features.py "<LULC output>/transfers/water_features/gujarat_water_bodies_by_type.zip"

Package layers -> Beeline layers
  tanks_ponds, aquaculture                         -> pond        (pond, aquaculture_pond)
  rivers_streams                                   -> river
  lakes_ponds                                      -> lake
  canals (polygons) + canals_gatishakti (lines)    -> canal
  reservoirs_dams                                  -> reservoir
  (none in the package)                            -> nalla       (empty layer, filled later)
  other_wetland, waterlogged                       -> wet_land    (wetland, waterlogged)
  everything else                                  -> other_water (creek, beach, coral_reef, lagoon,
       coastal_waters, tidal_saline_flats, salt_pans, mud_flat, salt_marsh, salt_pan, mangrove,
       mangroves, detected_water_2025_26, marine_25km  detected_water, marine_buffer)
`all_water_bodies` is the union of the type layers and is not ingested separately.
Kept attributes: source, detail, legal_status, coord_precision_m, nwia_code, area_km2 / length_km.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from beeline_api import reference  # noqa: E402

VERSION = "water-features-2026-08-31"
KEEP = ["source", "detail", "legal_status", "coord_precision_m", "nwia_code", "area_km2", "length_km"]
DETAIL_CLASS = {
    "Coastal-Natural-Creek": "creek", "waterway area": "creek", "Coastal-Natural-Sand/Beach": "beach",
    "Coastal-Natural-Coral reef": "coral_reef", "Coastal-Natural-Lagoon": "lagoon",
    "Coastal-Natural-Inter-tidal mud flat": "mud_flat", "Coastal-Natural-Salt marsh": "salt_marsh",
}


def _read(gpkg, layer):
    g = pyogrio.read_dataframe(gpkg, layer=layer)
    g["geometry"] = shapely.force_2d(shapely.make_valid(g.geometry.values))
    return g[~g.geometry.is_empty]


def _std(g, cls, prefix):
    out = gpd.GeoDataFrame({"class": cls, "name": g["name"].where(g["name"].astype(str).str.strip().isin(["", ".", "None", "nan"]) == False, None)  # noqa: E712
                            if "name" in g else None, "confidence": None,
                            "src_id": [f"{prefix}-{i}" for i in range(len(g))],
                            **{k: g[k].values for k in KEEP if k in g}}, geometry=g.geometry.values, crs=g.crs)
    return out.to_crs(4326)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package", help="zip or gpkg")
    a = ap.parse_args()
    src = Path(a.package)
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp(prefix="water_features_"))
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        gpkg = next(tmp.glob("**/*.gpkg"))
    else:
        gpkg = src
    try:
        L = {name: _read(gpkg, name) for name, _ in pyogrio.list_layers(gpkg) if name != "all_water_bodies"}
        layers = {
            "pond": pd.concat([_std(L["tanks_ponds"], "pond", "pond"), _std(L["aquaculture"], "aquaculture_pond", "aqua")]),
            "river": _std(L["rivers_streams"], "river", "river"),
            "lake": _std(L["lakes_ponds"], "lake", "lake"),
            "canal": pd.concat([_std(L["canals"], "canal", "canal"),
                                _std(L["canals_gatishakti"].assign(name=None), "canal", "canal-gs")]),
            "reservoir": _std(L["reservoirs_dams"], "reservoir", "res"),
            "wet_land": pd.concat([_std(L["other_wetland"], "wetland", "wl"), _std(L["waterlogged"], "waterlogged", "wlog")]),
        }
        other = []
        for lyr, default in (("coastal_waters", None), ("tidal_saline_flats", None), ("salt_pans", "salt_pan"),
                             ("mangroves", "mangrove"), ("detected_water_2025_26", "detected_water"),
                             ("marine_25km", "marine_buffer")):
            g = L[lyr]
            cls = g["detail"].map(DETAIL_CLASS).fillna(default or ("salt_marsh" if lyr == "tidal_saline_flats" else "creek")) \
                if default is None else default
            other.append(_std(g, cls, lyr))
        layers["other_water"] = pd.concat(other)
        layers["nalla"] = gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": []}, geometry=[], crs=4326)
        allg = pd.concat([v for v in layers.values() if len(v)])
        coverage = shapely.box(*allg.total_bounds)
        for key, g in layers.items():
            g = gpd.GeoDataFrame(g.reset_index(drop=True), geometry="geometry", crs=4326)
            prov = {"source": f"Gujarat water bodies by type ({src.name}): NWIA 2019 + OSM + GatiShakti"
                              + (" — no source in the package yet" if key == "nalla" else ""),
                    "source_date": "2026-08-31", "confidence": 0.85,
                    "note": "inventory, not notification — legal_status per feature"}
            out = reference.write_version(key, g, VERSION, prov, coverage=coverage)
            cls = dict(g["class"].value_counts()) if len(g) else {}
            print(f"{key:12s} {len(g):7d}  {cls}  -> {out}")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
