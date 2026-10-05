"""Admit strict near-identical triplets as clusters (copied from /mnt/cog-data/guj/pl/code/add_triplets.py):
3 unclustered rectangles, each rect>=0.88, edges <=3 m apart, azimuth within 3 deg, area / long side / short side
within +-15% (|log ratio|<=0.15), median plot <=10,000 m2.
Changes: a function on in-memory frames; empty rectangles/clusters no longer crash (orig L19 max() of empty)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely import STRtree


def add_triplets(R: gpd.GeoDataFrame, C: gpd.GeoDataFrame, log=print):
    R = R.copy()
    un = np.where(R.cluster.values < 0)[0] if len(R) else np.zeros(0, int)
    if len(un) < 3:
        log("triplets admitted: 0 | clusters %d -> %d" % (len(C), len(C))); return R, C
    geom = R.geometry.values; mrr = shapely.oriented_envelope(geom); c = shapely.get_coordinates(mrr).reshape(-1, 5, 2)[:, :4]
    e1 = c[:, 1] - c[:, 0]; e2 = c[:, 2] - c[:, 1]; l1 = np.hypot(*e1.T); l2 = np.hypot(*e2.T); L = np.maximum(l1, l2); S = np.minimum(l1, l2)
    lv = np.where((l1 >= l2)[:, None], e1, e2); az = np.degrees(np.arctan2(lv[:, 1], lv[:, 0])) % 90
    g = geom[un]
    pl, pr = STRtree(g).query(g, predicate="dwithin", distance=3.0); m = pl < pr; pl, pr = pl[m], pr[m]
    A = R.area_m2.values[un]; a_ = az[un]; Lu = L[un]; Su = S[un]; rc = R.rect.values[un]
    dz = np.abs(a_[pl] - a_[pr]); dz = np.minimum(dz, 90 - dz); q = lambda v: np.abs(np.log(v[pl] / v[pr])) <= 0.15
    ok = (dz <= 3) & q(A) & q(Su) & q(Lu) & (rc[pl] >= 0.88) & (rc[pr] >= 0.88)
    n, lab = connected_components(coo_matrix((np.ones(ok.sum()), (pl[ok], pr[ok])), shape=(len(g), len(g))), directed=False); cnt = np.bincount(lab)
    mx = [int(v) for v in (C.cluster.max() if len(C) else None, R.cluster.max()) if v is not None and not pd.isna(v)]
    nid = max(mx + [-1]) + 1; rows = []; geoms = []
    for k in np.where(cnt == 3)[0]:
        ix = un[lab == k]
        if np.median(R.area_m2.values[ix]) > 10000: continue
        R.loc[ix, "cluster"] = nid
        cl = shapely.buffer(shapely.buffer(shapely.union_all(mrr[ix]), 8.0, quad_segs=2), -8.0, quad_segs=2); a = R.area_m2.values[ix]
        rows.append(dict(cluster=nid, n_plots=3, plots_ha=a.sum() / 1e4, foot_ha=cl.area / 1e4, fill_ratio=a.sum() / cl.area)); geoms.append(cl); nid += 1
    if rows:
        C2 = gpd.GeoDataFrame(pd.concat([C, gpd.GeoDataFrame(rows, geometry=geoms, crs=C.crs)], ignore_index=True), crs=C.crs)
    else:
        C2 = C
    log("triplets admitted: %d | clusters %d -> %d" % (len(rows), len(C), len(C2)))
    return R, C2
