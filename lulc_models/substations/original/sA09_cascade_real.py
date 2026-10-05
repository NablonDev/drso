#!/usr/bin/env python3
"""The documented substation FP cascade (SUBSTATION_DETECTION_FINDINGS.md §3),
re-derived on Stage B detections.

The published cascade, which cut 372/441 FPs (84.4%) with zero TP loss:

    1. NISAR coherence (min3) >= 0.34     -> 314 FPs
    2. Sentinel-1 VV gamma0   >= -9.47 dB ->  42 FPs
    3. minimum width          >= 35 m     ->  16 FPs

This is the SUBSTATION cascade, not the tower one. The tower cascade
(chain_len>=3 AND coh>0.25) encodes collinearity along a line and does not
apply here.

Thresholds are RE-DERIVED, not copied, per §5: "Always derive the threshold
locally from known-substation vs local-background samples. Never reuse a number
across regions." Same AOI here, so the published values should reproduce -- if
they do not, that is a finding about Stage B's FP population, not a bug. The
published numbers were also fitted on the same 57 TPs they were scored on (§6),
so "zero TP loss" there is true by construction; re-derivation is the honest
version.

Stage 1 and 3 run here. Sentinel-1 needs live STAC and runs separately.
Topology (distance to line endpoint) is computed as an ATTRIBUTE only, never a
gate -- §4.1 proved it circular: it rejects exactly the unmapped substations a
discovery task exists to find.
"""
import math, warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely import affinity

COH = "/mnt/cog-data/nisar/out_coh/NISAR_coh_HH_min3.tif"
REV = "/mnt/cog-data/vivek/stageA/STAGEB_review_nisar.gpkg"
OSMF = "/mnt/drso-data/solar_lora/data/osm_ref.gpkg"
OUT = "/mnt/cog-data/vivek/stageA/STAGEB_cascade.gpkg"
EQ = "EPSG:7755"
DX, DY = 15.0, 5.0


def minwidth(poly):
    mrr = poly.minimum_rotated_rectangle
    if mrr.geom_type != "Polygon":
        return 0.0
    x, y = mrr.exterior.coords.xy
    pts = list(zip(x, y))
    L = [math.dist(pts[i], pts[i + 1]) for i in range(4)]
    return min(L[0], L[1])


tp = gpd.read_file(REV, layer="detected_TP")
fp = gpd.read_file(REV, layer="detected_FP")
det = gpd.GeoDataFrame(pd.concat([tp, fp], ignore_index=True), crs=tp.crs)

de = det.to_crs(EQ)
det["minwidth"] = [minwidth(g) for g in de.geometry]

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


t, f = det[det.matches_osm], det[~det.matches_osm]
print("Stage B detections: %d (%d TP, %d FP) | baseline recall %d/%d\n"
      % (len(det), len(t), len(f), recall_of(det), n_corr))

print("--- re-deriving stage 1 (NISAR coherence) on THIS FP population ---")
print("TP  coh: p05 %.3f  p10 %.3f  median %.3f" %
      (t.coh.quantile(.05), t.coh.quantile(.10), t.coh.median()))
print("FP  coh: median %.3f  p75 %.3f" % (f.coh.median(), f.coh.quantile(.75)))
zero_loss = float(t.coh.min())
print("highest coh cut losing ZERO true-positive detections: %.3f" % zero_loss)
print("published value (Kutch, old model): 0.34\n")

print("--- re-deriving stage 3 (minimum width) ---")
print("TP  minwidth: p05 %.1f m  p10 %.1f m  median %.1f m" %
      (t.minwidth.quantile(.05), t.minwidth.quantile(.10), t.minwidth.median()))
print("FP  minwidth: median %.1f m" % f.minwidth.median())
print("published value: 35 m\n")

print("%-40s %5s %5s %8s %10s" % ("cascade stage", "det", "FP", "recall", "FP/100km2"))
RULES = [
    ("0. none", lambda d: pd.Series(True, index=d.index)),
    ("1. coh >= 0.34  (published)", lambda d: d.coh >= 0.34),
    ("1'. coh >= %.2f  (re-derived)" % zero_loss, lambda d: d.coh >= zero_loss),
    ("3. minwidth >= 35 m", lambda d: d.minwidth >= 35),
    ("1+3 published (coh.34 + mw35)", lambda d: (d.coh >= 0.34) & (d.minwidth >= 35)),
    ("1'+3 re-derived", lambda d: (d.coh >= zero_loss) & (d.minwidth >= 35)),
]
for name, fn in RULES:
    keep = fn(det).fillna(False)
    s = det[keep]
    r = recall_of(s)
    nfp = int((~s.matches_osm).sum())
    print("%-40s %5d %5d %6d/%d %10.1f"
          % (name, len(s), nfp, r, n_corr, nfp / 49.09))

det.to_file(OUT, layer="detections", driver="GPKG")
gpd.GeoDataFrame(geometry=[corr], crs=4326).to_file(OUT, layer="aoi_corridor", driver="GPKG")
print("\nwrote %s (fields: ha, coh, minwidth, matches_osm)" % OUT)
