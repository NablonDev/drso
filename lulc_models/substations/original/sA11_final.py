#!/usr/bin/env python3
"""Final AESL substation deliverable for the Bhuj-Lakadia corridor.

Contents, kept as SEPARATE layers so provenance is never ambiguous:

  osm_substations       what OSM already records in the corridor. Unmodified.
                        Carries has_imagery, because 12 of the 78 rows in the
                        source layer sit outside the corridor polygon and have
                        no pixels -- they are not model failures and must not
                        be presented as such.

  model_new_detections  detections that survive the full chain AND are not on
                        an OSM substation. These are the deliverable's actual
                        contribution: candidate substations OSM does not have.

  model_confirms_osm    detections that land on an existing OSM substation.
                        Independent confirmation, not new information.

  osm_not_detected      OSM substations the model missed, flagged for whether
                        imagery exists at all.

Chain applied, every threshold re-derived on THIS detector rather than copied
(SUBSTATION_DETECTION_FINDINGS.md sec.5 -- never reuse a threshold across
regions or detectors):

  p >= 0.50, area >= 0.05 ha, NISAR coh_min3 >= 0.337,
  Sentinel-1 VV gamma0 >= -10.10 dB, minus 69 manual rejections.

minimum-width >= 35 m from the published cascade is deliberately NOT applied:
it costs 7 substations here because Stage B recovers the sub-0.5 ha class the
old model could not see, which is the same bias sec.2 identifies as the root
cause of the original recall gap.
"""
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd

SRC = "/mnt/cog-data/vivek/stageA/STAGEB_cascade_s1.gpkg"
REJ = "/mnt/cog-data/vivek/stageA/manual_rejects.gpkg"
OSMF = "/mnt/drso-data/solar_lora/data/osm_ref.gpkg"
OUT = "/mnt/cog-data/vivek/stageA/AESL_SUBSTATIONS_FINAL.gpkg"
EQ = "EPSG:7755"
COH0, VV0 = 0.337, -10.10

det = gpd.read_file(SRC, layer="detections").to_crs(4326)
rej = gpd.read_file(REJ).to_crs(4326)
osm = gpd.read_file(OSMF, layer="osm_substations_aoi").to_crs(4326)
corr = gpd.read_file(OSMF, layer="aoi_corridor").to_crs(4326).geometry.union_all()
osm["in_corridor"] = osm.intersects(corr)
osm["has_imagery"] = osm["in_corridor"]
n_corr = int(osm["in_corridor"].sum())
print("detections %d | manual rejects %d | OSM %d (%d in corridor)"
      % (len(det), len(rej), len(osm), n_corr))

passes = ((det.coh >= COH0) & (det.vv_db >= VV0)).fillna(False)
print("after SAR cascade: %d" % int(passes.sum()))

# a reject point inside a detection removes it
sidx = det.sindex
killed = np.zeros(len(det), bool)
for pt in rej.geometry:
    for i in sidx.intersection(pt.bounds):
        if det.geometry.iloc[i].contains(pt) or det.geometry.iloc[i].intersects(pt):
            killed[i] = True
print("manually rejected: %d detections (from %d points)"
      % (int(killed.sum()), len(rej)))
unmatched = int((~rej.geometry.apply(
    lambda pt: any(det.geometry.iloc[i].intersects(pt)
                   for i in sidx.intersection(pt.bounds)))).sum())
if unmatched:
    print("  NOTE: %d reject points fell inside no detection (ignored)" % unmatched)

keep = passes & ~killed
final = det[keep].copy()
print("final kept: %d\n" % len(final))

# recall of the delivered set
fidx = final.sindex
hit = np.zeros(len(osm), bool)
for i, geom in enumerate(osm.geometry):
    c = final.iloc[list(fidx.intersection(geom.bounds))]
    hit[i] = bool(len(c) and c.intersects(geom).any())
osm["detected_by_model"] = hit
rec = int((hit & osm["in_corridor"].values).sum())
print("recall of delivered set: %d/%d in-corridor OSM substations (%.1f%%)"
      % (rec, n_corr, 100 * rec / n_corr))

final["area_ha"] = final.to_crs(EQ).area / 1e4
final["source"] = np.where(final.matches_osm, "model_confirms_osm", "model_new")
cols = ["source", "area_ha", "coh", "vv_db", "minwidth", "geometry"]
new = final[~final.matches_osm][cols]
conf = final[final.matches_osm][cols]

ocols = ["osm_id", "name", "voltage", "in_corridor", "has_imagery",
         "detected_by_model", "geometry"]
osm[ocols].to_file(OUT, layer="osm_substations", driver="GPKG")
new.to_file(OUT, layer="model_new_detections", driver="GPKG")
conf.to_file(OUT, layer="model_confirms_osm", driver="GPKG")
osm[~hit][ocols].to_file(OUT, layer="osm_not_detected", driver="GPKG")
gpd.GeoDataFrame(geometry=[corr], crs=4326).to_file(OUT, layer="aoi_corridor", driver="GPKG")

print("\n--- layers ---")
print("osm_substations        %4d  (%d in corridor, %d with no imagery)"
      % (len(osm), n_corr, int((~osm.has_imagery).sum())))
print("model_confirms_osm     %4d" % len(conf))
print("model_new_detections   %4d  <- the new contribution" % len(new))
print("osm_not_detected       %4d  (%d of them have no imagery)"
      % (int((~hit).sum()), int(((~hit) & ~osm.has_imagery).sum())))
print("\nwrote %s" % OUT)
