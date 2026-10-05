#!/usr/bin/env python3
"""Can NISAR coherence separate Stage B's true positives from its false ones?

The tower work (NISAR_SAR_TOWERS.md) built a cascade:

    keep = chain_len >= 3  AND  coh_min3 > 0.25

Only the second test can transfer. chain_len is collinearity along a power
line; substations are not collinear, so that half is inapplicable here. What
might carry over is coherence: a switchyard is a large stable metal structure,
and the tower work measured coh_min3 0.641 at a 765 kV tower against 0.25
background, decaying to far-field within 90 m.

Two reasons this could still fail, both stated up front so the result is not
oversold either way:
  - That 0.641 is for 765 kV lattice. The corridor is 79% <=66 kV, and at 66 kV
    the paired test gave AUC 0.567 -- no usable signal. A switchyard is far more
    metal than a 66 kV pole, so it is worth testing, but it is not implied.
  - Coherence is high on ANY stable bright surface. Buildings, rock and salt
    were 88% of false positives in the tower work. Those are exactly what a
    substation detector also confuses.

Controls the doc requires and this script runs:
  - LAYOVER OFFSET dx=+15.0 dy=+5.0 m applied before sampling (§4). Skipping it
    halved enrichment in the tower work.
  - A NULL control: the same statistic at random points inside the corridor. If
    TP coherence is not clearly above that, there is nothing here.
  - AUC, not a threshold picked after seeing the data.
"""
import warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

COH = "/mnt/cog-data/nisar/out_coh/NISAR_coh_HH_min3.tif"
REV = "/mnt/cog-data/vivek/stageA/STAGEB_review.gpkg"
DX, DY = 15.0, 5.0            # radar layover, measured over 993 towers
SEED = 42


def stats(ds, gdf, label):
    """Median/p75 coherence inside each polygon, layover-shifted."""
    g = gdf.to_crs(ds.crs)
    from shapely import affinity
    out = []
    for geom in g.geometry:
        gs = affinity.translate(geom, xoff=DX, yoff=DY)
        try:
            a, _ = mask(ds, [gs.__geo_interface__], crop=True, filled=False)
        except Exception:
            continue
        v = a.compressed() if hasattr(a, "compressed") else a.ravel()
        v = v[np.isfinite(v)]
        v = v[(v > 0) & (v <= 1)]
        if v.size:
            out.append(float(np.median(v)))
    out = np.array(out)
    if out.size:
        print("%-22s n=%3d  median %.3f  p25 %.3f  p75 %.3f  frac>0.25 %.2f"
              % (label, out.size, np.median(out), np.percentile(out, 25),
                 np.percentile(out, 75), (out > 0.25).mean()))
    else:
        print("%-22s n=0 (no valid pixels -- raster may not cover these)" % label)
    return out


def auc(pos, neg):
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = allv.argsort().argsort().astype(float) + 1
    rp = r[:len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


ds = rasterio.open(COH)
print("coherence raster: %s  %dx%d" % (ds.crs, ds.width, ds.height))
b = ds.bounds
print("bounds: %.0f %.0f %.0f %.0f\n" % (b.left, b.bottom, b.right, b.top))

tp = gpd.read_file(REV, layer="detected_TP")
fp = gpd.read_file(REV, layer="detected_FP")
corr = gpd.read_file(REV, layer="aoi_corridor")

# does the NISAR clip even cover the corridor?
cc = corr.to_crs(ds.crs).geometry.union_all()
from shapely.geometry import box as sbox
cov = cc.intersection(sbox(*ds.bounds)).area / cc.area
print("NISAR clip covers %.1f%% of the AOI corridor\n" % (100 * cov))

v_tp = stats(ds, tp, "TP (matches OSM)")
v_fp = stats(ds, fp, "FP (not in OSM)")
v_fp_big = stats(ds, fp[fp.ha > 0.15], "FP  ha>0.15")

# NULL control: random points in the corridor, buffered to the median TP size
rng = np.random.default_rng(SEED)
minx, miny, maxx, maxy = cc.bounds
pts, tries = [], 0
med_r = float(np.sqrt(np.median(tp.to_crs(ds.crs).area) / np.pi))
while len(pts) < 300 and tries < 20000:
    tries += 1
    from shapely.geometry import Point
    p_ = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
    if cc.contains(p_):
        pts.append(p_.buffer(med_r))
null = gpd.GeoDataFrame(geometry=pts, crs=ds.crs)
v_null = stats(ds, null, "NULL random corridor")

print("\n--- AUC (1.0 = perfect separation, 0.5 = none) ---")
print("TP vs FP          : %.3f" % auc(v_tp, v_fp))
print("TP vs FP(ha>0.15) : %.3f" % auc(v_tp, v_fp_big))
print("TP vs NULL        : %.3f   <- if this is ~0.5 there is no signal at all"
      % auc(v_tp, v_null))

print("\n--- what the tower filter coh>0.25 would do here ---")
for t in (0.25, 0.30, 0.40, 0.50):
    kt = (v_tp > t).sum(); kf = (v_fp > t).sum()
    print("  coh>%.2f : keeps %d/%d TP (%.0f%%) and %d/%d FP (%.0f%%)"
          % (t, kt, len(v_tp), 100 * kt / max(len(v_tp), 1),
             kf, len(v_fp), 100 * kf / max(len(v_fp), 1)))
ds.close()
