"""Derived layers. Built-up clusters = buildings grouped at the rule's spacing and area."""
from __future__ import annotations

import geopandas as gpd
from shapely.ops import unary_union

from beeline_engine.frame import utm_epsg


def derive_clusters(buildings: gpd.GeoDataFrame | None, max_spacing_m: float = 50.0,
                    min_area_m2: float = 9000.0, confidence: float = 0.91) -> gpd.GeoDataFrame:
    """Buildings closer than max_spacing_m join one cluster; clusters below min_area_m2 drop.
    Closing (buffer out by half the spacing, back in) keeps the outline tight to the buildings."""
    empty = gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": []}, geometry=[], crs=4326)
    if buildings is None or len(buildings) == 0:
        return empty
    c = buildings.geometry.union_all().centroid
    epsg = utm_epsg(c.x, c.y)
    b = buildings.to_crs(epsg)
    h = max_spacing_m / 2.0
    merged = unary_union(list(b.geometry.buffer(h, resolution=4))).buffer(-h + 0.01, resolution=4)
    parts = [p for p in getattr(merged, "geoms", [merged]) if p.area >= min_area_m2]
    if not parts:
        return empty
    n = len(parts)
    g = gpd.GeoDataFrame({"class": ["builtup_cluster"] * n, "name": [None] * n, "confidence": [confidence] * n,
                          "src_id": [f"cluster-{i + 1}" for i in range(n)]}, geometry=parts, crs=epsg)
    return g.to_crs(4326)
