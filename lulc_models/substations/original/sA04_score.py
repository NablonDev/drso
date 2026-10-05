#!/usr/bin/env python3
"""Corridor scoring on the CORRECTED denominator, with a threshold sweep.

Two defects in v3_score.py are fixed here.

  DENOMINATOR. v3_score.py counted recall against all 78 rows of
  osm_substations_aoi. That layer was populated from the BOUNDING BOX of the
  corridor quadrilateral, so 12 of its rows sit outside the corridor entirely.
  Imagery was fetched clipped to the polygon, so those 12 have no pixels at
  all -- and because the probability rasters are uint8 with nodata=0, "no
  imagery" and "confident background" are the same value. Every model was
  therefore penalised for 12 sites it was never shown. Verified: all 12
  zero-probability sites are outside the corridor, and no in-corridor site
  reads zero.

  FIXED OPERATING POINT. p>=0.60 / >=0.10 ha is one point on a curve nobody has
  measured. Two known misses are detections killed by that rule rather than
  detection failures. This sweeps instead of assuming.

Recall is reported on three denominators so the correction is auditable rather
than asserted: the old 78, the 66 with imagery, and the 64 strictly within.
"""
import argparse, warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from scipy import ndimage

OSMF = "/mnt/drso-data/solar_lora/data/osm_ref.gpkg"
EQ = "EPSG:7755"
MODELS = {
    "OLD_national": "/mnt/drso-data/substation_nat/data/corridor_sub_prob.tif",
    "v1_hand80": "/mnt/drso-data/substation_nat/data/corridor_sub_prob_hand.tif",
    "v2_devesh75": "/mnt/drso-data/substation_nat/data/corridor_sub_prob_v2.tif",
    "v3_combined151": "/mnt/drso-data/substation_nat/data/corridor_sub_prob_v3.tif",
    "STAGE_B": "/mnt/cog-data/vivek/stageA/corridor_sub_prob_stageB.tif",
}


def polygonise(path, thresh, min_ha):
    ds = rasterio.open(path)
    cut = int(round(thresh * 255))
    px = abs(ds.transform.a * ds.transform.e)
    min_px = int(min_ha * 1e4 / px)
    polys = []
    H, W, STEP, OV = ds.height, ds.width, 8192, 256
    for y in range(0, H, STEP - OV):
        for x in range(0, W, STEP - OV):
            h, w = min(STEP, H - y), min(STEP, W - x)
            if h < 8 or w < 8:
                continue
            win = rasterio.windows.Window(x, y, w, h)
            a = ds.read(1, window=win)
            m = a >= cut
            if not m.any():
                continue
            lab, n = ndimage.label(m)
            if n == 0:
                continue
            sizes = np.bincount(lab.ravel())
            lut = np.zeros(sizes.size, dtype=bool)
            lut[1:] = sizes[1:] >= min_px
            keep = lut[lab]
            if not keep.any():
                continue
            tf = ds.window_transform(win)
            for geom, v in shapes(keep.astype(np.uint8), mask=keep, transform=tf):
                if v == 1:
                    polys.append(geom)
    from shapely.geometry import shape
    g = gpd.GeoDataFrame(geometry=[shape(p) for p in polys], crs=ds.crs)
    ds.close()
    if not len(g):
        return g
    g = g.dissolve().explode(index_parts=False).reset_index(drop=True)
    g["ha"] = g.to_crs(EQ).area / 1e4
    return g[g["ha"] >= min_ha].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="STAGE_B")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    osm = gpd.read_file(OSMF, layer="osm_substations_aoi").to_crs(4326)
    corr = gpd.read_file(OSMF, layer="aoi_corridor").to_crs(4326).geometry.union_all()
    within = osm.within(corr)
    touch = osm.intersects(corr)
    print("denominators:  bbox(old) %d | imaged %d | strictly-within %d"
          % (len(osm), int(touch.sum()), int(within.sum())))

    grid = ([(0.30, 0.05), (0.40, 0.05), (0.50, 0.05),
             (0.50, 0.10), (0.60, 0.05), (0.60, 0.10), (0.70, 0.10)]
            if a.sweep else [(0.60, 0.10)])

    rows = []
    for name in [m.strip() for m in a.models.split(",")]:
        path = MODELS[name]
        for th, mh in grid:
            det = polygonise(path, th, mh)
            if not len(det):
                rows.append(dict(model=name, p=th, min_ha=mh, det=0,
                                 r78=0, r66=0, r64=0, tp=0, fp=0, km2=0.0))
                continue
            dj = det.to_crs(4326)
            sidx = dj.sindex
            hit = np.zeros(len(osm), bool)
            for i, geom in enumerate(osm.geometry):
                cand = dj.iloc[list(sidx.intersection(geom.bounds))]
                hit[i] = bool(len(cand) and cand.intersects(geom).any())
            oidx = osm.sindex
            isTP = np.zeros(len(dj), bool)
            for j, geom in enumerate(dj.geometry):
                cand = osm.iloc[list(oidx.intersection(geom.bounds))]
                isTP[j] = bool(len(cand) and cand.intersects(geom).any())
            rows.append(dict(
                model=name, p=th, min_ha=mh, det=len(dj),
                r78=hit.sum(), r66=(hit & touch.values).sum(),
                r64=(hit & within.values).sum(),
                tp=int(isTP.sum()), fp=int((~isTP).sum()),
                km2=float(det["ha"].sum() / 100.0)))
            print("[%s] p>=%.2f >=%.2fha -> %d det | recall %d/78 %d/66 %d/64 | FP %d"
                  % (name, th, mh, len(dj), rows[-1]["r78"], rows[-1]["r66"],
                     rows[-1]["r64"], rows[-1]["fp"]), flush=True)

    d = pd.DataFrame(rows)
    d["rec78"] = 100 * d.r78 / len(osm)
    d["rec66"] = 100 * d.r66 / int(touch.sum())
    d["rec64"] = 100 * d.r64 / int(within.sum())
    d["fp_per_100km2"] = d.fp / 49.09
    print("\n%-15s %5s %6s %6s %8s %8s %8s %7s %9s"
          % ("model", "p", "minha", "det", "rec/78", "rec/66", "rec/64", "FP", "FP/100km2"))
    for _, r in d.iterrows():
        print("%-15s %5.2f %6.2f %6d %7.1f%% %7.1f%% %7.1f%% %7d %9.1f"
              % (r.model, r.p, r.min_ha, r.det, r.rec78, r.rec66, r.rec64,
                 r.fp, r.fp_per_100km2))
    d.to_csv("/mnt/cog-data/vivek/stageA/score_corrected.csv", index=False)
    print("\nwrote /mnt/cog-data/vivek/stageA/score_corrected.csv")


if __name__ == "__main__":
    main()
