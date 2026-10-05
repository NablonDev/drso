#!/usr/bin/env python3
"""Attach NISAR coherence to every detection and score the filter cascade.

Scored on SUBSTATION RECALL (how many OSM substations survive), not on
detection count. Those differ: a substation can be hit by two detections, so
dropping one costs no recall. The tower work was burned by exactly this class
of metric confusion, so the number reported here is the one that matters.

chain_len from the tower cascade is deliberately absent -- it encodes
collinearity along a transmission line, which substations do not have.
"""
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely import affinity

COH = "/mnt/cog-data/nisar/out_coh/NISAR_coh_HH_min3.tif"
REV = "/mnt/cog-data/vivek/stageA/STAGEB_review.gpkg"
OUT = "/mnt/cog-data/vivek/stageA/STAGEB_review_nisar.gpkg"
OSMF = "/mnt/drso-data/solar_lora/data/osm_ref.gpkg"
DX, DY = 15.0, 5.0


def coh_of(ds, gdf):
    g = gdf.to_crs(ds.crs)
    out = []
    for geom in g.geometry:
        gs = affinity.translate(geom, xoff=DX, yoff=DY)
        try:
            a, _ = mask(ds, [gs.__geo_interface__], crop=True, filled=False)
            v = a.compressed() if hasattr(a, "compressed") else a.ravel()
            v = v[np.isfinite(v)]
            v = v[(v > 0) & (v <= 1)]
            out.append(float(np.median(v)) if v.size else np.nan)
        except Exception:
            out.append(np.nan)
    return np.array(out)


ds = rasterio.open(COH)
tp = gpd.read_file(REV, layer="detected_TP")
fp = gpd.read_file(REV, layer="detected_FP")
tp["coh"] = coh_of(ds, tp)
fp["coh"] = coh_of(ds, fp)
ds.close()

det = gpd.GeoDataFrame(
    __import__("pandas").concat([tp, fp], ignore_index=True), crs=tp.crs)

osm = gpd.read_file(OSMF, layer="osm_substations_aoi").to_crs(4326)
corr = gpd.read_file(OSMF, layer="aoi_corridor").to_crs(4326).geometry.union_all()
osm["in_corr"] = osm.intersects(corr)
n_corr = int(osm["in_corr"].sum())


def recall_of(sub):
    if not len(sub):
        return 0
    sidx = sub.sindex
    hit = np.zeros(len(osm), bool)
    for i, geom in enumerate(osm.geometry):
        c = sub.iloc[list(sidx.intersection(geom.bounds))]
        hit[i] = bool(len(c) and c.intersects(geom).any())
    return int((hit & osm["in_corr"].values).sum())


print("baseline p>=0.50 >=0.05ha : %d det | recall %d/%d"
      % (len(det), recall_of(det), n_corr))
print("\n%-34s %5s %5s %8s %10s" % ("filter", "det", "FP", "recall", "FP/100km2"))
RULES = [
    ("none", lambda d: d.index == d.index),
    ("ha > 0.15", lambda d: d.ha > 0.15),
    ("coh > 0.25  (tower rule)", lambda d: d.coh > 0.25),
    ("coh > 0.30", lambda d: d.coh > 0.30),
    ("coh > 0.40", lambda d: d.coh > 0.40),
    ("coh > 0.30 AND ha > 0.15", lambda d: (d.coh > 0.30) & (d.ha > 0.15)),
    ("coh > 0.40 AND ha > 0.15", lambda d: (d.coh > 0.40) & (d.ha > 0.15)),
]
for name, fn in RULES:
    s = det[fn(det).fillna(False) if hasattr(fn(det), "fillna") else fn(det)]
    r = recall_of(s)
    nfp = int((~s.matches_osm).sum())
    print("%-34s %5d %5d %6d/%d %10.1f"
          % (name, len(s), nfp, r, n_corr, nfp / 49.09))

det["keep_coh030"] = (det.coh > 0.30).fillna(False)
det[det.matches_osm].to_file(OUT, layer="detected_TP", driver="GPKG")
det[~det.matches_osm].to_file(OUT, layer="detected_FP", driver="GPKG")
gpd.GeoDataFrame(geometry=[corr], crs=4326).to_file(OUT, layer="aoi_corridor", driver="GPKG")
for lay in ("osm_found", "osm_missed"):
    gpd.read_file(REV, layer=lay).to_file(OUT, layer=lay, driver="GPKG")
print("\nwrote %s  (detections now carry `coh`)" % OUT)
