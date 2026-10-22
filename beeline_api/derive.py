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
    # UTM zone from the bounding-box centre (21 Oct 2026: a union of every footprint just for its centroid took
    # over 40 min on a 230 km × ±25 km band)
    minx, miny, maxx, maxy = buildings.total_bounds
    epsg = utm_epsg((minx + maxx) / 2.0, (miny + maxy) / 2.0)
    b = buildings.to_crs(epsg)
    h = max_spacing_m / 2.0
    parts = []
    for merged in _merge_touching(list(b.geometry.buffer(h, resolution=4))):
        closed = merged.buffer(-h + 0.01, resolution=4)
        parts.extend(p for p in getattr(closed, "geoms", [closed]) if p.area >= min_area_m2)
    if not parts:
        return empty
    n = len(parts)
    g = gpd.GeoDataFrame({"class": ["builtup_cluster"] * n, "name": [None] * n, "confidence": [confidence] * n,
                          "src_id": [f"cluster-{i + 1}" for i in range(n)]}, geometry=parts, crs=epsg)
    return g.to_crs(4326)


def _merge_touching(geoms):
    """Union of the geometries, one group of touching ones at a time (same result as one union_all; each union stays
    small). Groups = connected components of the STRtree intersects graph."""
    import numpy as np
    import shapely
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if not geoms:
        return []
    arr = np.asarray(geoms, dtype=object)
    i, j = shapely.STRtree(arr).query(arr, predicate="intersects")
    n = len(arr)
    _, lab = connected_components(coo_matrix((np.ones(len(i), bool), (i, j)), shape=(n, n)), directed=False)
    order = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[order])) + 1
    return [shapely.union_all(arr[g]) if len(g) > 1 else arr[g[0]] for g in np.split(order, cuts)]
