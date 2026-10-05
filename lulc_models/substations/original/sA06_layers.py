#!/usr/bin/env python3
"""QGIS review layers at the chosen operating point: p>=0.50, >=0.05 ha.

Reuses sA04_score.polygonise verbatim. Three earlier export attempts tried to
be clever about producing every threshold at once and each was slower than the
straightforward one -- the scorer's version is measured at ~6 min for a full
pass and is already the code the reported metrics came from, so the layers and
the numbers cannot drift apart.

Four layers, which is what visual review actually needs:
  detected_TP    detections touching a known OSM substation
  detected_FP    detections touching none -- candidates, not necessarily errors
  osm_found      OSM substations the model hit
  osm_missed     OSM substations it did not, flagged for whether imagery exists

matches_osm is a real BOOLEAN. A categorised renderer built on 1/0 against a
boolean field renders nothing, silently.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/mnt/cog-data/vivek/stageA/scripts")

import geopandas as gpd
import numpy as np
from sA04_score import polygonise, OSMF

OUT = "/mnt/cog-data/vivek/stageA/STAGEB_review.gpkg"
SRC = "/mnt/cog-data/vivek/stageA/corridor_sub_prob_stageB.tif"
P, MIN_HA = 0.50, 0.05

osm = gpd.read_file(OSMF, layer="osm_substations_aoi").to_crs(4326)
corr = gpd.read_file(OSMF, layer="aoi_corridor").to_crs(4326).geometry.union_all()
osm["in_corridor"] = osm.intersects(corr)
osm["has_imagery"] = osm["in_corridor"]

print("polygonising at p>=%.2f, >=%.2f ha ..." % (P, MIN_HA), flush=True)
det = polygonise(SRC, P, MIN_HA)
dj = det.to_crs(4326).reset_index(drop=True)
print("detections: %d" % len(dj), flush=True)

oidx = osm.sindex
m = np.zeros(len(dj), dtype=bool)
for j, geom in enumerate(dj.geometry):
    cand = osm.iloc[list(oidx.intersection(geom.bounds))]
    m[j] = bool(len(cand) and cand.intersects(geom).any())
dj["matches_osm"] = m

didx = dj.sindex
hit = np.zeros(len(osm), dtype=bool)
for i, geom in enumerate(osm.geometry):
    cand = dj.iloc[list(didx.intersection(geom.bounds))]
    hit[i] = bool(len(cand) and cand.intersects(geom).any())
osm["detected"] = hit

gpd.GeoDataFrame(geometry=[corr], crs=4326).to_file(OUT, layer="aoi_corridor", driver="GPKG")
dj[m][["ha", "matches_osm", "geometry"]].to_file(OUT, layer="detected_TP", driver="GPKG")
dj[~m][["ha", "matches_osm", "geometry"]].to_file(OUT, layer="detected_FP", driver="GPKG")
cols = ["osm_id", "name", "voltage", "in_corridor", "has_imagery", "detected", "geometry"]
osm[hit][cols].to_file(OUT, layer="osm_found", driver="GPKG")
osm[~hit][cols].to_file(OUT, layer="osm_missed", driver="GPKG")

print("\n=== p>=%.2f, >=%.2f ha ===" % (P, MIN_HA))
print("detected_TP  %4d   (%.2f km2)" % (int(m.sum()), dj[m].ha.sum() / 100))
print("detected_FP  %4d   (%.2f km2)" % (int((~m).sum()), dj[~m].ha.sum() / 100))
print("osm_found    %4d / %d" % (int(hit.sum()), len(osm)))
print("osm_missed   %4d   of which %d have NO imagery (outside corridor)"
      % (int((~hit).sum()), int(((~hit) & ~osm["has_imagery"]).sum())))
print("recall: %d/78 bbox | %d/%d in-corridor"
      % (hit.sum(), (hit & osm.in_corridor.values).sum(), int(osm.in_corridor.sum())))
print("\nwrote %s" % OUT)
