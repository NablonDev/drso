#!/usr/bin/env python3
"""Stage 2 of the documented cascade: Sentinel-1 VV gamma0 on Stage B detections.

Fetch logic is lifted from s1_fp_filter.py so the measurement is comparable to
the published one: sentinel-1-rtc from Planetary Computer, one scene per
relative orbit per month so passes are geometrically comparable, mean linear
gamma0 inside each polygon, converted to dB after stacking.

The published cut was VV >= -9.47 dB. It is RE-DERIVED here, per §5 of the
findings doc, which is explicit that thresholds must never be reused across
regions -- and, by the same logic, should be re-checked when the detector
changes, because the FP population changes with it.
"""
import json, urllib.parse, urllib.request, warnings
warnings.filterwarnings("ignore")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, from_bounds

SRC = "/mnt/cog-data/vivek/stageA/STAGEB_cascade.gpkg"
OUT = "/mnt/cog-data/vivek/stageA/STAGEB_cascade_s1.gpkg"
OSMF = "/mnt/drso-data/solar_lora/data/osm_ref.gpkg"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SIGN = "https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
AOI = {"type": "Polygon", "coordinates": [[[69.5784, 23.1691], [70.6109, 23.1691],
        [70.6109, 23.6809], [69.5784, 23.6809], [69.5784, 23.1691]]]}
NSCENES = 8


def sign(h):
    try:
        return json.load(urllib.request.urlopen(
            SIGN + urllib.parse.quote(h, safe=""), timeout=60))["href"]
    except Exception:
        return h


d = gpd.read_file(SRC, layer="detections")
print("detections: %d" % len(d), flush=True)

body = {"collections": ["sentinel-1-rtc"], "intersects": AOI,
        "datetime": "2025-10-01T00:00:00Z/2026-09-01T00:00:00Z", "limit": 200}
req = urllib.request.Request(STAC, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
feats = json.load(urllib.request.urlopen(req, timeout=120)).get("features", [])
print("RTC scenes found: %d" % len(feats), flush=True)

byorb = {}
for f in feats:
    k = (f["properties"].get("sat:relative_orbit"), f["properties"]["datetime"][:7])
    byorb.setdefault(k, f)
picked = list(byorb.values())[:NSCENES]
print("using %d scenes" % len(picked), flush=True)

stack = []
for n, f in enumerate(picked, 1):
    href = sign(f["assets"]["vv"]["href"])
    vals = np.full(len(d), np.nan)
    try:
        ds = rasterio.open(href)
    except Exception as e:
        print("  scene %d open failed: %r" % (n, e), flush=True)
        stack.append(vals); continue
    g = d.to_crs(ds.crs)
    wb = ds.bounds
    for i, geom in enumerate(g.geometry):
        b = geom.bounds
        if b[2] < wb[0] or b[0] > wb[2] or b[3] < wb[1] or b[1] > wb[3]:
            continue
        try:
            w = from_bounds(*b, transform=ds.transform).round_offsets().round_lengths()
            w = w.intersection(Window(0, 0, ds.width, ds.height))
            if w.width < 1 or w.height < 1:
                continue
            a = ds.read(1, window=w).astype("float32")
            m = rasterize([(geom, 1)], out_shape=a.shape,
                          transform=ds.window_transform(w), fill=0, dtype="uint8").astype(bool)
            m &= np.isfinite(a) & (a > 0)
            if m.sum():
                vals[i] = float(a[m].mean())
        except Exception:
            pass
    ds.close()
    stack.append(vals)
    print("  scene %d/%d %s coverage %.0f%%"
          % (n, len(picked), f["properties"]["datetime"][:10],
             100 * np.isfinite(vals).mean()), flush=True)

A = np.vstack(stack)
with np.errstate(divide="ignore", invalid="ignore"):
    db = 10 * np.log10(A)
d["vv_db"] = np.nanmean(db, axis=0)
d["vv_std"] = np.nanstd(db, axis=0)

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


t, f_ = d[d.matches_osm], d[~d.matches_osm]
print("\nTP vv_db: p05 %.2f median %.2f | FP vv_db: median %.2f"
      % (t.vv_db.quantile(.05), t.vv_db.median(), f_.vv_db.median()))
zl = float(np.nanmin(t.vv_db))
print("highest VV cut losing zero TP detections: %.2f dB  (published -9.47)" % zl)

COH0 = 0.337
print("\n%-44s %5s %5s %8s %10s" % ("cascade", "det", "FP", "recall", "FP/100km2"))
import pandas as pd
RULES = [
    ("0. none", lambda x: pd.Series(True, index=x.index)),
    ("1. coh>=0.337", lambda x: x.coh >= COH0),
    ("1+2 published VV>=-9.47", lambda x: (x.coh >= COH0) & (x.vv_db >= -9.47)),
    ("1+2 re-derived VV>=%.2f" % zl, lambda x: (x.coh >= COH0) & (x.vv_db >= zl)),
]
for name, fn in RULES:
    s = d[fn(d).fillna(False)]
    print("%-44s %5d %5d %6d/%d %10.1f"
          % (name, len(s), int((~s.matches_osm).sum()), recall_of(s), n_corr,
             int((~s.matches_osm).sum()) / 49.09))

d.to_file(OUT, layer="detections", driver="GPKG")
gpd.GeoDataFrame(geometry=[corr], crs=4326).to_file(OUT, layer="aoi_corridor", driver="GPKG")
print("\nwrote %s" % OUT)
