"""rectangles + clusters -> building-footprint coverage -> client deliverable
(copied from /mnt/cog-data/guj/pl/code/pl_deliver_guj.py).

Changes: footprints come from ONE layer of the run's buildings gpkg (layer `features`, else the first layer)
instead of the two hard-coded google_open_buildings_v3 / microsoft_buildings_2026 layers; they are unioned
exactly as before, so overlapping sources are not double counted.  Empty clusters give empty frames
(orig L8 total_bounds of an empty frame).  buildings_gpkg=None -> built_pct / built_class null.
Cluster ids: <prefix><nnnn> (default PLC-), ordered west->east, north->south on a 100 m grid as before."""
from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import shapely
from shapely import STRtree

BINS = [-0.01, 5, 20, 40, 100.01]
LAB = ["Vacant (<5% built)", "Sparsely built (5-20%)", "Partly built (20-40%)", "Built-up (>=40%)"]
CLUSTER_OUT = ["cluster_id", "n_plots", "area_ha", "plots_area_ha", "built_pct", "built_class", "vacant_plots",
               "vacant_plots_pct", "plot_built_pct_median"]
PLOT_OUT = ["plot_id", "cluster_id", "area_m2", "built_pct", "built_class"]


def _cat(p):
    s = pd.cut(p, BINS, labels=LAB, right=False).astype(object)
    return s.where(pd.notna(p), None)


def empty_frames(crs):
    C = gpd.GeoDataFrame({c: pd.Series(dtype="object" if c in ("cluster_id", "built_class") else "float64") for c in CLUSTER_OUT},
                         geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    P = gpd.GeoDataFrame({c: pd.Series(dtype="object" if c in ("plot_id", "cluster_id", "built_class") else "float64") for c in PLOT_OUT},
                         geometry=gpd.GeoSeries([], crs=crs), crs=crs)
    return C, P


def buildings_layer(path):
    names = [l for l, _ in pyogrio.list_layers(path)]
    return "features" if "features" in names else names[0]


def deliver(R: gpd.GeoDataFrame, CL: gpd.GeoDataFrame, buildings_gpkg: str | None, id_prefix="PLC-", log=print):
    crs = CL.crs
    R = R[R.cluster >= 0].reset_index(drop=True)
    if not len(CL):
        return empty_frames(crs)
    CL = CL.reset_index(drop=True).copy()
    if buildings_gpkg:
        bb = tuple(gpd.GeoSeries([shapely.box(*CL.total_bounds)], crs=crs).buffer(500).to_crs(4326).total_bounds)
        FP = gpd.read_file(buildings_gpkg, layer=buildings_layer(buildings_gpkg), bbox=bb, engine="pyogrio", columns=[])
        FP = FP[FP.geometry.notna()].to_crs(crs)
        FP = shapely.make_valid(FP.geometry.values); FP = FP[np.unique(STRtree(FP).query(CL.geometry.values, predicate="intersects")[1])]
        U = shapely.get_parts(shapely.union_all(FP))
        log("deliver: %d footprints touching %d clusters" % (len(FP), len(CL)))

        def cov(geoms):
            gi, pi = STRtree(U).query(geoms, predicate="intersects")
            a = shapely.area(shapely.intersection(geoms[gi], U[pi])); return np.minimum(np.bincount(gi, weights=a, minlength=len(geoms)) / shapely.area(geoms), 1.0)
        R["built_pct"] = (100 * cov(R.geometry.values)).round(1); CL["built_pct"] = (100 * cov(CL.geometry.values)).round(1)
    else:
        R["built_pct"] = np.nan; CL["built_pct"] = np.nan
    c = CL.geometry.representative_point(); CL = CL.assign(_x=c.x.round(-2), _y=-c.y.round(-2)).sort_values(["_x", "_y"]).reset_index(drop=True)
    CL["cluster_id"] = [id_prefix + "%04d" % (i + 1) for i in range(len(CL))]; R["cluster_id"] = R.cluster.map(dict(zip(CL.cluster, CL.cluster_id)))
    p = R.geometry.representative_point(); R = R.assign(_x=p.x, _y=-p.y).sort_values(["cluster_id", "_x", "_y"]).reset_index(drop=True)
    R["plot_id"] = R.cluster_id + "-" + (R.groupby("cluster_id").cumcount() + 1).map("{:03d}".format)
    R["area_m2"] = R.geometry.area.round(0); R["built_class"] = _cat(R.built_pct)
    P = R[PLOT_OUT + ["geometry"]]
    v = P.groupby("cluster_id").agg(n_plots=("plot_id", "size"), plots_area_ha=("area_m2", lambda a: a.sum() / 1e4),
                                    vacant_plots=("built_class", lambda s: (s == LAB[0]).sum()), plot_built_pct_median=("built_pct", "median"))
    C = CL[["cluster_id", "built_pct", "geometry"]].merge(v, on="cluster_id")
    C["area_ha"] = (C.geometry.area / 1e4).round(2); C["plots_area_ha"] = C.plots_area_ha.round(2)
    C["vacant_plots_pct"] = (100 * C.vacant_plots / C.n_plots).round(1); C["built_class"] = _cat(C.built_pct)
    if not buildings_gpkg:
        C["vacant_plots"] = np.nan; C["vacant_plots_pct"] = np.nan
    C = gpd.GeoDataFrame(C[CLUSTER_OUT + ["geometry"]], crs=crs)
    assert P.plot_id.is_unique and C.cluster_id.is_unique
    return C, gpd.GeoDataFrame(P, crs=crs)
