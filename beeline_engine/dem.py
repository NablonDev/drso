"""Copernicus GLO-30 DEM: fetch the 1°x1° COG tiles covering a bbox into the bucket and sample them.

Source: s3://copernicus-dem-30m (public, HTTPS). Tiles are cached; a tile that does not exist
upstream (open sea) is remembered as missing so it is not re-requested.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.merge import merge

log = logging.getLogger(__name__)
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"
SOURCE = "Copernicus GLO-30 DEM (30 m)"


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def ensure_tiles(bbox: tuple[float, float, float, float], cache_dir: Path, timeout: float = 120) -> list[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bbox
    out = []
    for lat in range(math.floor(miny), math.floor(maxy) + 1):
        for lon in range(math.floor(minx), math.floor(maxx) + 1):
            name = tile_name(lat, lon)
            path = cache_dir / f"{name}.tif"
            missing = cache_dir / f"{name}.missing"
            if path.exists():
                out.append(path)
                continue
            if missing.exists():
                continue
            url = f"{BASE}/{name}/{name}.tif"
            tmp = path.with_suffix(".part")
            with httpx.stream("GET", url, timeout=timeout, follow_redirects=True) as r:
                if r.status_code == 404:
                    missing.touch()
                    continue
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
            tmp.rename(path)
            log.info("fetched DEM tile %s", name)
            out.append(path)
    return out


class DemSampler:
    """Samples heights (m) at lon/lat points from the cached tiles of one bbox."""

    def __init__(self, tiles: list[Path], bbox):
        self.ok = bool(tiles)
        if not self.ok:
            return
        srcs = [rasterio.open(t) for t in tiles]
        pad = 0.01
        arr, tr = merge(srcs, bounds=(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad))
        for s in srcs:
            s.close()
        self.arr = arr[0].astype(np.float32)
        self.tr = tr
        self.inv = ~tr

    def sample(self, lon, lat) -> np.ndarray:
        lon = np.asarray(lon, float)
        lat = np.asarray(lat, float)
        if not self.ok:
            return np.full(lon.shape, np.nan)
        col, row = self.inv * (lon, lat)
        # bilinear
        c0 = np.floor(col - 0.5).astype(int)
        r0 = np.floor(row - 0.5).astype(int)
        fc = col - 0.5 - c0
        fr = row - 0.5 - r0
        h, w = self.arr.shape
        c0 = np.clip(c0, 0, w - 2)
        r0 = np.clip(r0, 0, h - 2)
        a = self.arr
        z = (a[r0, c0] * (1 - fc) * (1 - fr) + a[r0, c0 + 1] * fc * (1 - fr)
             + a[r0 + 1, c0] * (1 - fc) * fr + a[r0 + 1, c0 + 1] * fc * fr)
        return z
