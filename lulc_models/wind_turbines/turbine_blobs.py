"""Raw blob extraction from a round4 turbine mask (one pass over the scene raster).

Copied from /mnt/drso-data/devesh/turbine_expg/extract_blobs.py. The algorithm (strip reads,
8-connected labelling, moments, strip-overlap dedupe) is unchanged. Changes:
- thresh / min_px are arguments (were THRESH / MINPX env vars read at import);
- the mask path is an argument (was MASKDIR env or a Windows share);
- the cache is written where the caller says (was blob_cache/ next to the script);
- a cancel hook between strips.
"""
import json
import math
import os

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import ndimage

STRIP = 6000
OV = 400


def extract(mask_path, thresh=160, min_px=15, cache_path=None, tile=None, on_strip=None):
    """Return dict(tile, gsd, blobs, transform, crs, bounds) with blobs as
    (cy, cx, area_px, major_px, minor_px, angle_deg) tuples in pixel units."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            d = json.load(f)
        if d.get("thresh") == thresh and d.get("min_px") == min_px and d.get("mask") == os.path.abspath(mask_path):
            return d
    src = rasterio.open(mask_path); gsd = abs(src.transform.a); H, W = src.height, src.width
    out = []; r0 = 0
    try:
        while r0 < H:
            h = min(STRIP + OV, H - r0)
            arr = src.read(1, window=Window(0, r0, W, h))
            b = arr >= thresh
            del arr
            if b.any():
                lab, n = ndimage.label(b, structure=np.ones((3, 3), int))
                if n:
                    idx = np.arange(1, n + 1)
                    sizes = ndimage.sum(b, lab, idx)          # cheap area pass first
                    keep = idx[sizes >= min_px]
                    if len(keep):
                        objs = ndimage.find_objects(lab)
                        for li in keep:
                            sl = objs[li - 1]
                            ys, xs = np.nonzero(lab[sl] == li)
                            ys = ys + sl[0].start + r0; xs = xs + sl[1].start
                            cy, cx = ys.mean(), xs.mean()
                            yy = ys - cy; xx = xs - cx
                            cov = np.cov(np.vstack([yy, xx])) if len(ys) > 2 else np.eye(2)
                            ev, evec = np.linalg.eigh(np.clip(cov, -1e12, 1e12))
                            ev = np.clip(ev, 1e-9, None)
                            out.append((float(cy), float(cx), int(len(ys)),
                                        float(4 * math.sqrt(ev[1])), float(4 * math.sqrt(ev[0])),
                                        float(math.degrees(math.atan2(evec[1, 1], evec[0, 1])) % 180.0)))
                del lab
            del b
            r0 += STRIP
            if on_strip:
                on_strip(min(1.0, r0 / H))
        # strip-overlap dedupe
        out.sort(key=lambda z: (round(z[0], 1), round(z[1], 1), z[2]))
        ded = []
        for b in out:
            if ded and abs(b[0] - ded[-1][0]) < 1 and abs(b[1] - ded[-1][1]) < 1 and b[2] == ded[-1][2]:
                continue
            ded.append(b)
        tr = src.transform; crs = src.crs.to_wkt() if src.crs else None; bnds = list(src.bounds)
    finally:
        src.close()
    d = dict(tile=tile, gsd=gsd, blobs=ded, transform=list(tr)[:6], crs=crs, bounds=bnds,
             thresh=thresh, min_px=min_px, mask=os.path.abspath(mask_path))
    if cache_path:
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, cache_path)
    return d
