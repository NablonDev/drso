#!/usr/bin/env python
"""Build the DEV reference store for the Bhuj–Lakadia AOI (testing only, never production).

Writes into BEELINE_REFERENCE_DIR (set it to data/bucket/reference_dev). Sources: local files
on the dev VM plus OpenStreetMap (Overpass). Every layer's meta says "DEV".
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
assert "reference_dev" in os.environ.get("BEELINE_REFERENCE_DIR", ""), "set BEELINE_REFERENCE_DIR=.../reference_dev"

import geopandas as gpd  # noqa: E402
import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
from rasterio import features as rfeatures  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from rasterio.windows import from_bounds  # noqa: E402
from shapely.geometry import LineString, Polygon, box, shape  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

from beeline_api import reference  # noqa: E402

AOI = box(69.578, 23.169, 70.611, 23.681)
VIVEK = Path("/mnt/cog-data/vivek")
LULC_OUT = VIVEK / "production_route" / "LULC output"
VERSION = "dev-bhuj-lakadia-20261005"
UA = {"User-Agent": "beeline-route-planner-dev/1.0"}
S, W, N, E = AOI.bounds[1], AOI.bounds[0], AOI.bounds[3], AOI.bounds[2]


def overpass(q: str) -> list[dict]:
    body = f"[out:json][timeout:180];({q});out geom;"
    for attempt in range(4):
        try:
            r = httpx.post("https://overpass-api.de/api/interpreter", data={"data": body}, headers=UA, timeout=240)
            if r.status_code == 200:
                return r.json()["elements"]
            print("  overpass", r.status_code, "retrying")
        except httpx.HTTPError as e:
            print("  overpass error", e)
        time.sleep(15 * (attempt + 1))
    raise RuntimeError("overpass failed")


AREA_KEYS = {"landuse", "natural", "leisure", "boundary", "building", "power", "military", "amenity", "aeroway"}


def osm_gdf(q: str, polygons: bool | None = None) -> gpd.GeoDataFrame:
    rows = []
    for el in overpass(q):
        tags = el.get("tags", {})
        geom = None
        if el["type"] == "node":
            from shapely.geometry import Point
            geom = Point(el["lon"], el["lat"])
        elif el["type"] == "way" and "geometry" in el:
            cs = [(p["lon"], p["lat"]) for p in el["geometry"]]
            closed = len(cs) > 3 and cs[0] == cs[-1]
            want_poly = polygons if polygons is not None else (closed and bool(AREA_KEYS & set(tags)))
            geom = Polygon(cs) if (want_poly and closed) else LineString(cs) if len(cs) > 1 else None
        elif el["type"] == "relation":
            lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
                     for m in el.get("members", []) if m.get("type") == "way" and m.get("geometry") and m.get("role") != "inner"]
            if polygons is False:
                geom = unary_union(lines) if lines else None
            else:
                polys = list(polygonize(unary_union(lines))) if lines else []
                geom = unary_union(polys) if polys else None
        if geom is None or geom.is_empty:
            continue
        rows.append({"tags": tags, "name": tags.get("name:en") or tags.get("name"), "osm_id": f"{el['type']}/{el['id']}",
                     "geometry": geom})
    if not rows:
        return gpd.GeoDataFrame({"tags": [], "name": [], "osm_id": []}, geometry=[], crs=4326)
    return gpd.GeoDataFrame(rows, crs=4326)


def put(key, gdf, source, cls_series=None, const=None, conf=None, source_date="2026-10-05", model=None):
    gdf = gdf.copy()
    if cls_series is not None:
        gdf["cls"] = cls_series
    norm = reference.normalise(gdf, key, class_field="cls" if cls_series is not None else None,
                               const_class=const, name_field="name" if "name" in gdf else None,
                               id_field="osm_id" if "osm_id" in gdf else None)
    norm = norm[norm.intersects(AOI)]
    reference.write_version(key, norm, VERSION, {"source": f"DEV — {source}", "source_date": source_date,
                                                 "model": model, "confidence": conf, "dev": True}, coverage=AOI)
    print(f"{key:12s} {len(norm):7d}  {source}")


def bbox():
    return f"({S},{W},{N},{E})"


def main():
    b = bbox()
    skip = set(os.environ.get("SKIP", "").split(","))
    # --- LULC from the Esri 10 m annual land cover (2025), vectorised at 30 m -------------
    if "lulc" not in skip:
      with rasterio.open(VIVEK / "buildup_detection/gujarat_lulc_2025_10m.tif") as src:
        aoi_m = gpd.GeoSeries([AOI], crs=4326).to_crs(src.crs).iloc[0]
        win = from_bounds(*aoi_m.bounds, transform=src.transform)
        out_shape = (int(win.height // 3), int(win.width // 3))
        arr = src.read(1, window=win, out_shape=out_shape, resampling=Resampling.mode)
        tr = rasterio.windows.transform(win, src.transform) * rasterio.Affine.scale(win.width / out_shape[1], win.height / out_shape[0])
        arr = rfeatures.sieve(arr, 12)
        esri = {1: "water_body", 2: "orchard", 4: "marsh", 5: "agriculture", 7: "built_up", 8: "barren", 11: "shrub"}
        shapes = [(shape(g), esri.get(int(v), "unclassified")) for g, v in rfeatures.shapes(arr, transform=tr) if v != 0]
        lulc = gpd.GeoDataFrame({"cls": [c for _, c in shapes]}, geometry=[g for g, _ in shapes], crs=src.crs).to_crs(4326)
      put("lulc", lulc, "Esri 10 m Land Cover 2025 (vectorised 30 m; trees mapped to orchard)", cls_series=lulc["cls"],
          conf=0.85, source_date="2025-12-31", model="esri-lulc-2025 (proxy for lulc model)")

    # --- buildings: Microsoft 2026 footprints ---------------------------------------------
    if "buildings" not in skip:
      bl = pyogrio.read_dataframe(VIVEK / "buildup_detection/gujarat_4layer_FINAL_corridor.gpkg",
                                layer="microsoft_buildings_2026", bbox=AOI.bounds)
      put("buildings", bl, "Microsoft building footprints 2026", const="building", conf=0.94,
        source_date="2026-03-03", model="ms-buildings-2026 (proxy for footprint model)")

    # --- power: OSM-derived HV + LV lines and towers --------------------------------------
    if "power" not in skip:
      pw = osm_gdf(f'way["power"~"^(line|minor_line)$"]{bbox()};node["power"="tower"]{bbox()};', polygons=False)
      pw["cls"] = pw.geom_type.map(lambda t: "power_tower" if t == "Point" else "power_line")
      put("power", pw, "OSM power=line/minor_line + power=tower", cls_series=pw["cls"], conf=0.85)

    # --- airports -------------------------------------------------------------------------
    ap = osm_gdf(f'way["aeroway"="aerodrome"]{bbox()};relation["aeroway"="aerodrome"]{bbox()};', polygons=True)
    ap = ap[ap.geom_type.isin(["Polygon", "MultiPolygon"])]
    put("aviation", ap, "OSM aeroway=aerodrome", const="airport", conf=1.0)

    # --- OSM layers ------------------------------------------------------------------------
    rd = osm_gdf(f'way["highway"~"^(motorway|trunk|primary|secondary)$"]{b};', polygons=False)
    def road_cls(t):
        ref = (t.get("ref") or "").upper()
        if "NH" in ref or t.get("highway") in ("motorway", "trunk"):
            return "nh"
        if "SH" in ref or t.get("highway") == "primary":
            return "sh"
        return "mdr"
    put("roads", rd, "OSM highways (trunk/NH → nh, primary/SH → sh, secondary → mdr)",
        cls_series=rd["tags"].map(road_cls), conf=0.95)
    rl = osm_gdf(f'way["railway"="rail"]{b};', polygons=False)
    put("rail", rl, "OSM railway=rail", const="railway", conf=0.96)
    wl = osm_gdf(f'way["waterway"~"^(river|canal|stream|dam)$"]{b};'
                 f'way["natural"="water"]{b};relation["natural"="water"]{b};', polygons=None)
    def water_cls(t):
        ww = t.get("waterway")
        if ww == "river":
            return "river"
        if ww == "canal":
            return "canal"
        if ww == "stream":
            return "nalla"
        if ww == "dam":
            return "dam"
        if t.get("water") in ("river",):
            return "river"
        if t.get("water") in ("canal",):
            return "canal"
        return "pond"
    put("water", wl, "OSM waterways and water polygons", cls_series=wl["tags"].map(water_cls), conf=0.9)
    # forest, protected areas and wetlands: real statewide layers from the forest delivery
    # (scripts/ingest_forest.py); no OSM stand-ins (6 Oct 2026)
    pl = osm_gdf(f'way["man_made"="pipeline"]{b};', polygons=False)
    put("pipelines", pl, "OSM man_made=pipeline", cls_series=pl["tags"].map(
        lambda t: "oil_pipeline" if t.get("substance") in ("oil", "crude") else "gas_pipeline"), conf=0.8)
    df = osm_gdf(f'way["landuse"="military"]{b};relation["landuse"="military"]{b};', polygons=True)
    df = df[df.geom_type.isin(["Polygon", "MultiPolygon"])]
    put("defence", df, "OSM landuse=military", const="defence_land", conf=0.9)
    mo = osm_gdf(f'node["historic"~"monument|memorial|fort"]{b};way["historic"~"monument|fort"]{b};', polygons=True)
    put("monuments", mo, "OSM historic monuments (NOT the ASI list)", const="monument", conf=0.7)
    ss = osm_gdf(f'way["power"="substation"]{b};relation["power"="substation"]{b};', polygons=True)
    ss = ss[ss.geom_type.isin(["Polygon", "MultiPolygon"])]
    put("substations", ss, "OSM power=substation", const="substation", conf=0.9)
    so = osm_gdf(f'way["power"="plant"]["plant:source"="solar"]{b};relation["power"="plant"]["plant:source"="solar"]{b};'
                 f'way["generator:source"="solar"]["generator:type"="solar_photovoltaic_panel"]{b};', polygons=True)
    so = so[so.geom_type.isin(["Polygon", "MultiPolygon"])]
    put("solar", so, "OSM solar plants", const="solar_park", conf=0.85)
    # layers with no dev source: write empty versions so the run shows LAYER_EMPTY honestly
    for key, cls in [("plotted", "plotted_land")]:
        put(key, gpd.GeoDataFrame({"name": []}, geometry=[], crs=4326), "no dev source", const=cls, conf=None)


if __name__ == "__main__":
    main()
