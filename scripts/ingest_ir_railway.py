#!/usr/bin/env python
"""Ingest the Indian Railways package (Vivek, 14 Oct 2026): ir_raw.gpkg → two reference layers.

    python scripts/ingest_ir_railway.py "<LULC output>/transfers/railway/ir_raw.gpkg"

  ir_track          → rail             (class railway; replaces the OSM dev extract as the current version)
  ir_platform_poly  → rail_platforms   (class platform, plus class platform_crossing_zone: each platform's
                                        minimum rectangle extended 1,500 m beyond both ends along its long side
                                        — the line may not cross there)
  ir_stations_pt is not ingested (station points; the platforms carry the geometry the rules need).
Coverage: India (the package is national); runs read their band only.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402
import shapely  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402

from beeline_api import reference  # noqa: E402

VERSION = "ir-railway-2026-10-14"
EXTEND_M = 1500.0


def _zone(rect: Polygon, extend: float) -> Polygon:
    """The rectangle stretched by `extend` beyond both short ends (along its long side)."""
    c = list(rect.exterior.coords)[:4]
    e = [(c[k], c[(k + 1) % 4]) for k in range(4)]
    L = [math.dist(a, b) for a, b in e]
    k = 0 if L[0] >= L[1] else 1                      # e[k], e[k+2] are the long sides
    (a, b), (cc, d) = e[k], e[k + 2]
    ux, uy = (b[0] - a[0]) / L[k], (b[1] - a[1]) / L[k]
    a2, b2 = (a[0] - ux * extend, a[1] - uy * extend), (b[0] + ux * extend, b[1] + uy * extend)
    c2, d2 = (cc[0] + ux * extend, cc[1] + uy * extend), (d[0] - ux * extend, d[1] - uy * extend)   # cc→d runs opposite
    return Polygon([a2, b2, c2, d2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    a = ap.parse_args()
    src = Path(a.package)
    trk = pyogrio.read_dataframe(src, layer="ir_track").to_crs(4326)
    trk["geometry"] = shapely.force_2d(trk.geometry.values)
    trk = trk[~trk.geometry.is_empty].reset_index(drop=True)
    rail = gpd.GeoDataFrame({"class": "railway", "name": trk.get("tmssection"), "confidence": 0.95,
                             "src_id": [f"irt-{v}" for v in trk["objectid"]],
                             "gauge": trk.get("gauge"), "line": trk.get("line"), "railway_zone": trk.get("railway"),
                             "division": trk.get("division")}, geometry=trk.geometry.values, crs=4326)
    reference.write_version("rail", rail, VERSION, {
        "source": f"Indian Railways track ({src.name}, layer ir_track)", "source_date": "2026-10-14", "confidence": 0.95,
        "note": "national track centrelines by TMS section; gauge BG/MG/NG"}, coverage=shapely.box(68, 6, 98, 36))
    print("rail", len(rail), flush=True)
    pl = pyogrio.read_dataframe(src, layer="ir_platform_poly")
    pl["geometry"] = shapely.force_2d(shapely.make_valid(pl.geometry.values))
    pl = pl[~pl.geometry.is_empty].reset_index(drop=True)
    utm = pl.to_crs(32643)                              # metric for the rectangles (zone 43 covers most of India; good enough for 1.5 km)
    zones = [_zone(g.minimum_rotated_rectangle, EXTEND_M) for g in utm.geometry]
    zg = gpd.GeoSeries(zones, crs=32643).to_crs(4326)
    name = pl["asset_detl"] if "asset_detl" in pl else None
    plat = gpd.GeoDataFrame({"class": "platform", "name": name, "confidence": 0.95,
                             "src_id": [f"irp-{v}" for v in pl["objectid"]], "area_sqm": pl.get("area_sqm"),
                             "district": pl.get("district")}, geometry=pl.geometry.to_crs(4326).values, crs=4326)
    zone = gpd.GeoDataFrame({"class": "platform_crossing_zone", "name": name, "confidence": 0.95,
                             "src_id": [f"irz-{v}" for v in pl["objectid"]], "area_sqm": None, "district": pl.get("district")},
                            geometry=zg.values, crs=4326)
    allp = gpd.GeoDataFrame(__import__("pandas").concat([plat, zone], ignore_index=True), crs=4326)
    reference.write_version("rail_platforms", allp, VERSION, {
        "source": f"Indian Railways platforms ({src.name}, layer ir_platform_poly)", "source_date": "2026-10-14",
        "confidence": 0.95, "note": "platform polygons + crossing-exclusion zones (platform rectangle + 1,500 m each end)"},
        coverage=shapely.box(68, 6, 98, 36))
    print("rail_platforms", len(plat), "platforms +", len(zone), "zones", flush=True)


if __name__ == "__main__":
    main()
