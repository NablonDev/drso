"""Remove salt-works clusters (copied from /mnt/cog-data/guj/pl/code/saltfilter_guj.py): 1 km ring
(cluster buffer 1000 m minus 50 m) WorldCover water+wetland+mangrove+bare (80/90/95/60) >= threshold (0.60).
Built % is NOT used: salt heaps are detected as buildings by Google/MS.

Changes: WorldCover comes only from local paths given by the request (never /vsicurl; several tiles are
mosaicked with a VRT); the threshold is a parameter; each ring is read and rasterised in its own window
(same result, bounded memory for long bands); WorldCover nodata (0, outside the supplied tiles) is left out of
the ring fraction -- inside the tiles the result equals the original, which had full coverage."""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

SALT = (80, 90, 95, 60)


def ring_saltscape(C: gpd.GeoDataFrame, wc_path: str):
    """fraction of salt-scape classes in each cluster's 50-1000 m ring, and fraction of ring pixels with data"""
    if not len(C):
        return np.zeros(0), np.zeros(0)
    ring = gpd.GeoSeries(C.geometry.buffer(1000).difference(C.geometry.buffer(50)).values, crs=C.crs).to_crs(4326)
    sc = np.zeros(len(C)); have = np.zeros(len(C))
    with rasterio.open(wc_path) as s:
        for i, g in enumerate(ring.geometry):
            w = from_bounds(*g.bounds, transform=s.transform).round_offsets().round_lengths()
            if w.width < 1 or w.height < 1:
                continue
            a = s.read(1, window=w, boundless=True, fill_value=0); tr = s.window_transform(w)
            m = rasterize([(g, 1)], out_shape=a.shape, transform=tr, fill=0, dtype="uint8").astype(bool)
            v = a[m]; valid = v != 0
            have[i] = valid.mean() if len(v) else 0
            sc[i] = np.isin(v[valid], SALT).mean() if valid.any() else 0
    return sc, have


def saltfilter(C: gpd.GeoDataFrame, Pl: gpd.GeoDataFrame, wc_path: str, thr: float = 0.60, log=print):
    """-> kept clusters, kept plots, removed clusters (with ring_saltscape), mean ring data fraction"""
    sc, have = ring_saltscape(C, wc_path)
    C = C.copy(); C["ring_saltscape"] = sc.round(3)
    salt = (C.ring_saltscape >= thr).values
    bad = set(C.cluster_id[salt])
    keepC = C[~salt].drop(columns="ring_saltscape"); keepP = Pl[~Pl.cluster_id.isin(bad)]
    log("saltfilter: clusters %d -> %d (removed %d, %.0f ha) | plots %d -> %d" % (
        len(C), len(keepC), salt.sum(), C[salt].area_ha.sum() if len(C) else 0, len(Pl), len(keepP)))
    return keepC, keepP, C[salt], (float(have.mean()) if len(have) else 1.0)
