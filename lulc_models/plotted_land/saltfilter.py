"""Remove salt-works clusters: 1 km ring (cluster buffer 1000 m minus 50 m) covered by salt-scape
polygons >= threshold (0.60). Built % is NOT used: salt heaps are detected as buildings by Google/MS.

Origin: /mnt/cog-data/guj/pl/code/saltfilter_guj.py, which measured WorldCover water + wetland + mangrove
+ bare (80/90/95/60) in the ring. Changed 6 Oct 2026 (Vivek): the salt-scape now comes from the delivered
vector layers (salt_pans soil layer + the salt-pan / mud-flat / salt-marsh / mangrove classes of
other_water), passed by the request as one gpkg; no WorldCover. The ring fraction is taken over the part
of the ring the salt-scape layers cover (as the original left WorldCover nodata out)."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import shapely
from shapely.strtree import STRtree


def ring_saltscape(C: gpd.GeoDataFrame, salt: gpd.GeoDataFrame, cov):
    """fraction of salt-scape area in each cluster's 50-1000 m ring (within the layers' coverage), and
    fraction of the ring the coverage reaches. C, salt and cov in the same metric CRS."""
    if not len(C):
        return np.zeros(0), np.zeros(0)
    ring = C.geometry.buffer(1000).difference(C.geometry.buffer(50)).values
    geoms = [g for g in salt.geometry.values if g is not None and not g.is_empty]
    tree = STRtree(geoms) if geoms else None
    sc = np.zeros(len(C)); have = np.zeros(len(C))
    for i, r in enumerate(ring):
        rc = r if cov is None else r.intersection(cov)
        if rc.is_empty or r.area <= 0:
            continue
        have[i] = rc.area / r.area
        if tree is None:
            continue
        hit = tree.query(rc, predicate="intersects")
        if len(hit):
            sc[i] = shapely.union_all([geoms[j] for j in hit]).intersection(rc).area / rc.area
    return sc, have


def saltfilter(C: gpd.GeoDataFrame, Pl: gpd.GeoDataFrame, salt: gpd.GeoDataFrame, cov, thr: float = 0.60, log=print):
    """-> kept clusters, kept plots, removed clusters (with ring_saltscape), mean ring coverage fraction"""
    sc, have = ring_saltscape(C, salt, cov)
    C = C.copy(); C["ring_saltscape"] = sc.round(3)
    is_salt = (C.ring_saltscape >= thr).values
    bad = set(C.cluster_id[is_salt])
    keepC = C[~is_salt].drop(columns="ring_saltscape"); keepP = Pl[~Pl.cluster_id.isin(bad)]
    log("saltfilter: clusters %d -> %d (removed %d, %.0f ha) | plots %d -> %d" % (
        len(C), len(keepC), is_salt.sum(), C[is_salt].area_ha.sum() if len(C) else 0, len(Pl), len(keepP)))
    return keepC, keepP, C[is_salt], (float(have.mean()) if len(have) else 1.0)
