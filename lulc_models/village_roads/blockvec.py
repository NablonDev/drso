"""Halo vectorisation of one block (from guj/code/guj_vec.py :: one()).

Same algorithm and parameters as the statewide run: the 3x3 neighbouring u_ COGs form a
(CORE + 2*HV)^2 window, the graph is built/pruned/bridged over the whole window and only the
block core is kept, so adjacent blocks agree near their shared edge provided their neighbours
were final when they were vectorised (the scheduler in run_aoi.py guarantees that; a missing
neighbour reads as zero). Width comes from the native n_ COGs; if they are absent the width
fields are null. No TensorFlow here (runs in spawned CPU workers).
"""
from __future__ import annotations

import json
import math
import os
import warnings

import numpy as np

from aoi_grid import NT, O, RES, TM, block_lat

warnings.filterwarnings("ignore")

TIERS = [(3.0, "track"), (4.6, "single_lane"), (6.5, "intermediate_lane"), (9.0, "two_lane"),
         (16.0, "two_lane_paved_shoulder"), (1e9, "multi_lane")]
CORE = NT * 256
P = {"threshold": 0.2, "width_threshold": 0.5, "max_gap_m": 250.0, "halo_px": 512, "min_px": 500}


def init(params: dict | None = None):
    """Worker initialiser: set the vectoriser parameters for this process."""
    import vectorize as V
    if params:
        for k in P:
            if params.get(k) is not None:
                P[k] = type(P[k])(params[k])
    V.MAX_GAP_M = float(P["max_gap_m"])


def tier(w):
    if w is None:
        return None
    for hi, n in TIERS:
        if w < hi:
            return n
    return "multi_lane"


def window(d, pre, bx, by, hv):
    import rasterio
    n = CORE + 2 * hv; out = np.zeros((n, n), np.float32)
    if not d or not os.path.isdir(d):
        return out
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            f = "%s/%s_%d_%d.tif" % (d, pre, bx + dx * NT, by + dy * NT)
            if not os.path.exists(f):
                continue
            oy = hv + dy * CORE; ox = hv + dx * CORE
            r0, r1 = max(0, oy), min(n, oy + CORE); c0, c1 = max(0, ox), min(n, ox + CORE)
            if r0 >= r1 or c0 >= c1:
                continue
            with rasterio.open(f) as r:
                a = r.read(1, window=((r0 - oy, r1 - oy), (c0 - ox, c1 - ox)))
            out[r0:r1, c0:c1] = a.astype(np.float32) / 255.0
    return out


def vec_block(bx, by, u_dir, n_dir, out_dir):
    """Vectorise block (bx,by) -> out_dir/<bx>_<by>.geojson (EPSG:3857). Returns (tag, n, bridges)."""
    import scipy.ndimage as ndi
    from skimage.morphology import remove_small_holes, remove_small_objects
    import vectorize as V
    from vec_clip import clip_polyline

    tag = "%d_%d" % (bx, by); out = os.path.join(out_dir, tag + ".geojson")
    if os.path.exists(out):
        with open(out) as f:
            return tag, len(json.load(f)["features"]), 0
    HV = int(P["halo_px"])
    px_m = RES * math.cos(math.radians(block_lat(by)))
    prob = window(u_dir, "u", bx, by, HV)
    feats = []; nb = 0
    if (prob[HV:HV + CORE, HV:HV + CORE] >= P["threshold"]).sum() >= P["min_px"]:
        mask, skel, G = V.build(prob, px_m); G = V.prune(G, px_m)
        G, nb = V.bridge(G, prob, px_m); G = V.collapse(G); G = V.prune(G, px_m, rounds=2)
        nprob = window(n_dir, "n", bx, by, HV); m = nprob >= P["width_threshold"]
        m = remove_small_holes(m, connectivity=1, area_threshold=200)
        m = remove_small_objects(m, connectivity=1, min_size=250)
        dt = ndi.distance_transform_edt(m); n = CORE + 2 * HV
        x0 = -O + bx * TM; y0 = O - by * TM
        lo, hi = HV - 0.5, HV + CORE - 0.5
        for u, v, d in G.edges(data=True):
            pts = d.get("pts")
            if pts is None or len(pts) < 2:
                continue
            for part in clip_polyline(pts.astype(float), lo, hi):
                L = float(np.linalg.norm(np.diff(part, axis=0), axis=1).sum())
                if L < 1.0:
                    continue
                rr = np.clip(np.round(part[:, 0]).astype(int), 0, n - 1)
                cc = np.clip(np.round(part[:, 1]).astype(int), 0, n - 1)
                vals = dt[rr, cc]; vals = vals[vals > 0]; w_m = w90 = None
                if vals.size >= 3:
                    w_m = round(float(2 * np.median(vals) * px_m), 2)
                    w90 = round(float(2 * np.percentile(vals, 90) * px_m), 2)
                xy = [(x0 + (c - HV + 0.5) * RES, y0 - (r - HV + 0.5) * RES) for r, c in part]
                feats.append({"type": "Feature", "properties": {
                    "block": tag, "bridged": int(bool(d.get("bridged"))), "surface_width_m": w_m,
                    "width_p90_m": w90, "lane_tier": tier(w_m), "length_m": round(L * px_m, 1)},
                    "geometry": {"type": "LineString",
                                 "coordinates": [[float(x), float(y)] for x, y in xy]}})
    fc = {"type": "FeatureCollection",
          "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}},
          "features": feats}
    with open(out + ".t", "w") as f:
        json.dump(fc, f)
    os.replace(out + ".t", out)
    return tag, len(feats), nb


def vec_job(args):
    bx, by, u_dir, n_dir, out_dir = args
    return vec_block(bx, by, u_dir, n_dir, out_dir)
