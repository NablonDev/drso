"""On-disk cache for the rule compiler's layer rasters (8 Oct 2026).

Feature masks, distance transforms and crossing-angle rasters depend only on the corridor frame
and the layer's geometry — not on rule values — so a solve after a rule or cost change reuses them.
Keys: frame signature + a fingerprint of the layer's geometry (WKB hash), so a re-extraction or a
hand-added shape gives a new key on its own. Stored losslessly, so a cached solve is bit-identical:
  mask   bit-packed bool
  dsq    the distance transform as squared pixel distances (uint32; the float field is rebuilt
         with the same sqrt the transform uses)
  ang    nearest-segment direction, float32, + its squared pixel distance as above
The cache is size-capped; the least recently used files go first.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np


def sq_pixels(edt) -> np.ndarray:
    """Exact squared pixel distances from scipy's float64 EDT output (sqrt of an integer)."""
    return np.rint(np.square(edt)).astype(np.uint32)


def dist_from_sq(sq, res_m: float) -> np.ndarray:
    """The float32 metre field exactly as _Rasterizer builds it from the EDT."""
    return np.sqrt(sq.astype(np.float64)).astype(np.float32) * res_m


class RasterCache:
    def __init__(self, root: str | Path, frame, max_bytes: int = 6 * 1024 ** 3):
        self.root = Path(root)
        self.max_bytes = max_bytes
        f = frame
        sig = repr((round(f.A[0], 2), round(f.A[1], 2), round(f.B[0], 2), round(f.B[1], 2), f.epsg,
                    round(f.half_width_m, 2), round(f.margin_m, 2), f.res_m, f.width, f.height))
        self.dir = self.root / hashlib.sha1(sig.encode()).hexdigest()[:16]
        self.hits = 0
        self.misses = 0

    def _path(self, kind: str, fp: str) -> Path:
        return self.dir / f"{kind}_{fp}.npz"

    def _load(self, kind: str, fp: str):
        p = self._path(kind, fp)
        if not p.exists():
            self.misses += 1
            return None
        try:
            with np.load(p) as z:
                data = {k: z[k] for k in z.files}
            os.utime(p, None)                                   # LRU touch
            self.hits += 1
            return data
        except Exception:
            self.misses += 1
            return None

    def _save(self, kind: str, fp: str, **arrays) -> None:
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            p = self._path(kind, fp)
            tmp = p.with_suffix(".tmp.npz")
            np.savez(tmp, **arrays)
            os.replace(tmp, p)
            self._evict()
        except Exception:
            pass                                                # a cache failure never fails a solve

    def _evict(self) -> None:
        files = sorted(self.root.glob("*/*.npz"), key=lambda q: q.stat().st_mtime)
        total = sum(q.stat().st_size for q in files)
        while files and total > self.max_bytes:
            q = files.pop(0)
            total -= q.stat().st_size
            q.unlink(missing_ok=True)

    # ---- typed accessors ------------------------------------------------------------------
    def get_mask(self, fp: str, shape):
        d = self._load("mask", fp)
        if d is None:
            return None, False
        if int(d["empty"]):
            return None, True
        return np.unpackbits(d["bits"], count=int(np.prod(shape))).reshape(shape).astype(bool), True

    def put_mask(self, fp: str, m) -> None:
        if m is None:
            self._save("mask", fp, empty=np.array(1))
        else:
            self._save("mask", fp, empty=np.array(0), bits=np.packbits(m.ravel()))

    def get_dist(self, fp: str):
        """-> (squared pixel distances uint32 | None, found)"""
        d = self._load("dsq", fp)
        if d is None:
            return None, False
        if int(d["empty"]):
            return None, True
        return d["sq"], True

    def put_dist(self, fp: str, sq) -> None:
        if sq is None:
            self._save("dsq", fp, empty=np.array(1))
        else:
            self._save("dsq", fp, empty=np.array(0), sq=sq)

    def get_angle(self, fp: str):
        """-> (angle float32 | None, squared pixel distances uint32 | None, found)"""
        d = self._load("ang", fp)
        if d is None:
            return None, None, False
        if int(d["empty"]):
            return None, None, True
        return d["ang"], d["sq"], True

    def put_angle(self, fp: str, ang, sq) -> None:
        if ang is None:
            self._save("ang", fp, empty=np.array(1))
        else:
            self._save("ang", fp, empty=np.array(0), ang=ang, sq=sq)


def geometry_fingerprint(gdf, classes) -> str:
    """Stable hash of a layer's geometry (+ the class filter) for cache keys."""
    import shapely
    h = hashlib.sha1()
    h.update(repr(tuple(classes or ())).encode())
    if gdf is None or not len(gdf):
        h.update(b"empty")
        return h.hexdigest()[:24]
    wkb = shapely.to_wkb(np.asarray(gdf.geometry.values), hex=False)
    for b in wkb:
        h.update(b if b is not None else b"-")
    h.update(str(gdf.crs).encode())
    return h.hexdigest()[:24]


def stamp() -> float:
    return time.time()
