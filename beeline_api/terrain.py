"""Ground height on the map from the Copernicus GLO-30 tiles the solver already uses (10 Oct 2026).

- Terrarium-encoded RGB tiles for MapLibre (`raster-dem`): hillshade and optional 3D terrain.
- Elevation at a point (fallback for the cursor readout when 3D terrain is off).
- A route's longitudinal profile: ground every ~20 m along the alignment, towers, conductor sag,
  clearance requirement.
Only tiles already on disk are read; nothing is fetched here.
"""
from __future__ import annotations

import io
import math
from functools import lru_cache
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.transform import from_bounds

from beeline_engine import dem as demmod

from . import settings

TILE_PX = 256
MIN_Z, MAX_Z = 5, 15
CACHE = settings.DATA / "cache" / "terrain"


def _dem_dir() -> Path:
    # read-only: the tiles the solver fetched; when DEM fetching is switched off (DEM_DIR None, as in
    # the API tests) the default on-disk location is still read
    return Path(settings.DEM_DIR) if settings.DEM_DIR else settings.DATA / "dem" / "copernicus"


def _tifs() -> list[Path]:
    d = _dem_dir()
    return sorted(d.glob("*.tif")) if d.exists() else []


@lru_cache(maxsize=1)
def _index() -> list[tuple[tuple[float, float, float, float], Path]]:
    """(lon/lat bounds, path) of every DEM tile on disk."""
    out = []
    for p in _tifs():
        with rasterio.open(p) as src:
            b = src.bounds
            out.append(((b.left, b.bottom, b.right, b.top), p))
    return out


def tiles_for(bounds: tuple[float, float, float, float]) -> list[Path]:
    w, s, e, n = bounds
    return [p for (l, b, r, t), p in _index() if not (r <= w or l >= e or t <= s or b >= n)]


def covers(bbox: tuple[float, float, float, float]) -> dict:
    """Which 1° DEM tiles a lon/lat bbox needs and which exist on disk."""
    minx, miny, maxx, maxy = bbox
    need, have = [], []
    for lat in range(math.floor(miny), math.floor(maxy) + 1):
        for lon in range(math.floor(minx), math.floor(maxx) + 1):
            name = demmod.tile_name(lat, lon)
            need.append(name)
            if (_dem_dir() / f"{name}.tif").exists():
                have.append(name)
    return {"needed": need, "present": have, "complete": len(have) == len(need), "source": demmod.SOURCE}


def _encode_terrarium(z: np.ndarray) -> bytes:
    v = np.nan_to_num(z, nan=0.0).astype(np.float64) + 32768.0
    v = np.clip(v, 0, 65535.999)
    r = np.floor(v / 256.0)
    g = np.floor(v - r * 256.0)
    b = np.floor((v - np.floor(v)) * 256.0)
    rgb = np.dstack([r, g, b]).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def tile(z: int, x: int, y: int) -> bytes | None:
    """Terrarium PNG for a web-mercator tile, from the DEM tiles on disk (None where none cover it)."""
    if z < MIN_Z or z > MAX_Z:
        return None
    p = CACHE / str(z) / str(x) / f"{y}.png"
    if p.exists():
        return p.read_bytes()
    ll = mercantile.bounds(x, y, z)
    srcs = tiles_for((ll.west, ll.south, ll.east, ll.north))
    if not srcs:
        return None
    xy = mercantile.xy_bounds(x, y, z)
    dst = np.full((TILE_PX, TILE_PX), np.nan, np.float32)
    dst_tr = from_bounds(xy.left, xy.bottom, xy.right, xy.top, TILE_PX, TILE_PX)
    for sp in srcs:
        with rasterio.open(sp) as src:
            part = np.full((TILE_PX, TILE_PX), np.nan, np.float32)
            reproject(rasterio.band(src, 1), part, dst_transform=dst_tr, dst_crs="EPSG:3857",
                      resampling=Resampling.bilinear, dst_nodata=np.nan, src_nodata=src.nodata)
            m = np.isfinite(part)
            dst[m] = part[m]
    data = _encode_terrarium(dst)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    except OSError:
        pass
    return data


def elevation(lon: float, lat: float) -> float | None:
    srcs = tiles_for((lon, lat, lon, lat))
    if not srcs:
        return None
    s = demmod.DemSampler(srcs, (lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002))
    z = float(s.sample(np.array([lon]), np.array([lat]))[0])
    return None if not math.isfinite(z) else round(z, 1)


def profile(route: dict, kv: int, step_m: float = 20.0) -> dict:
    """Longitudinal section of a solved route (lon/lat towers in the solve result): ground every
    step_m along the alignment, towers with attachment heights, per-span sag and the clearance line."""
    from pyproj import Geod
    from beeline_engine.config import load_engine
    eng = load_engine()["spotting"]
    att = float(eng["lowest_attachment_m"][kv])
    clear = float(eng["ground_clearance_m"][kv])
    sag_d = float(eng["sag_at_design_span_m"][kv])
    sag_ref = float((eng.get("sag_reference_span_m") or {}).get(kv, 370))
    T = route["towers"]
    g = Geod(ellps="WGS84")
    lons = np.array([t["lon"] for t in T]); lats = np.array([t["lat"] for t in T])
    bbox = (lons.min() - 0.01, lats.min() - 0.01, lons.max() + 0.01, lats.max() + 0.01)
    srcs = tiles_for(bbox)
    sampler = demmod.DemSampler(srcs, bbox) if srcs else None
    chain = [0.0]
    for i in range(1, len(T)):
        chain.append(chain[-1] + g.inv(lons[i - 1], lats[i - 1], lons[i], lats[i])[2])
    ground_x, ground_z = [], []
    for i in range(1, len(T)):
        L = chain[i] - chain[i - 1]
        n = max(2, int(L // step_m) + 1)
        ts = np.linspace(0, 1, n, endpoint=(i == len(T) - 1))
        lo = lons[i - 1] + ts * (lons[i] - lons[i - 1]); la = lats[i - 1] + ts * (lats[i] - lats[i - 1])
        zz = sampler.sample(lo, la) if sampler else np.full(len(ts), np.nan)
        ground_x.extend((chain[i - 1] + ts * L).tolist()); ground_z.extend([None if not math.isfinite(v) else round(float(v), 1) for v in zz])
    towers = []
    for i, t in enumerate(T):
        gz = t.get("ground_z")
        ext = float(t.get("extension_m") or 0.0)
        attach = None if gz is None else round(gz + (0.0 if t["type"] == "G" else att + ext), 1)
        towers.append({"no": t["no"], "type": t["type"], "chainage_m": round(chain[i], 1), "ground_z": gz,
                       "extension_m": ext, "attach_z": attach, "deflection_deg": t.get("deflection_deg"),
                       "violations": t.get("violations", []), "lon": t["lon"], "lat": t["lat"]})
    spans = []
    for s in route["spans"]:
        a, b = towers[s["a"]], towers[s["b"]]
        L = s["length_m"]
        spans.append({"a": s["a"], "b": s["b"], "length_m": round(L, 1), "sag_m": round(sag_d * (L / sag_ref) ** 2, 2),
                      "min_clearance_m": s.get("min_clearance_m"), "crossings": [c.get("rule") for c in s.get("crossings", [])],
                      "chainage_a": a["chainage_m"], "chainage_b": b["chainage_m"], "attach_a": a["attach_z"], "attach_b": b["attach_z"]})
    return {"kv": kv, "clearance_m": clear, "attachment_m": att, "sag_reference_span_m": sag_ref, "step_m": step_m,
            "dem": demmod.SOURCE if sampler else None, "length_m": round(chain[-1], 1),
            "ground": {"x": [round(v, 1) for v in ground_x], "z": ground_z}, "towers": towers, "spans": spans}
