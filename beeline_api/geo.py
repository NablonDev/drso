"""Corridor geo-validation: India state/district lookup, band polygon, area, extraction estimate."""
from __future__ import annotations

import json
import logging
import threading

import geopandas as gpd
import httpx
from shapely.geometry import Point, shape

from beeline_engine.config import load_layers
from beeline_engine.frame import Frame

from . import settings

log = logging.getLogger(__name__)
ADMIN_DIR = settings.DATA / "admin"
SOURCES = {
    "adm1": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM1/geoBoundaries-IND-ADM1_simplified.geojson",
    "adm2": "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM2/geoBoundaries-IND-ADM2_simplified.geojson",
}
_lock = threading.Lock()
_cache: dict[str, gpd.GeoDataFrame] = {}


def ensure_admin() -> None:
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    for k, url in SOURCES.items():
        p = ADMIN_DIR / f"india_{k}.geojson"
        if p.exists() and p.stat().st_size > 1000:
            continue
        r = httpx.get(url, follow_redirects=True, timeout=180, headers={"User-Agent": settings.HTTP_UA})
        r.raise_for_status()
        p.write_bytes(r.content)


def _admin(level: str) -> gpd.GeoDataFrame | None:
    with _lock:
        if level not in _cache:
            p = ADMIN_DIR / f"india_{level}.geojson"
            if not p.exists():
                try:
                    ensure_admin()
                except Exception as e:  # offline: geo-validation degrades to "unknown"
                    log.warning("admin boundaries unavailable: %s", e)
                    return None
            g = gpd.read_file(p).to_crs(4326)
            g["geometry"] = g.geometry.make_valid()
            _cache[level] = g
        return _cache[level]


def lookup(lon: float, lat: float) -> dict:
    pt = Point(lon, lat)
    out = {"in_india": None, "state": None, "district": None}
    a1 = _admin("adm1")
    if a1 is None:
        return out
    hit = a1[a1.contains(pt)]
    out["in_india"] = bool(len(hit))
    if len(hit):
        out["state"] = hit.iloc[0]["shapeName"]
    a2 = _admin("adm2")
    if a2 is not None:
        h2 = a2[a2.contains(pt)]
        if len(h2):
            out["district"] = h2.iloc[0]["shapeName"]
    return out


def validate(a: tuple[float, float], b: tuple[float, float]) -> dict:
    checks = []
    ok_coords = all(-180 <= x <= 180 for x in (a[0], b[0])) and all(-90 <= y <= 90 for y in (a[1], b[1]))
    checks.append({"ok": ok_coords, "text": "Both coordinates are valid decimal degrees, WGS 84"})
    la, lb = lookup(*a), lookup(*b)
    in_india = la["in_india"] and lb["in_india"]
    checks.append({"ok": bool(in_india) if la["in_india"] is not None else None,
                   "text": "Both points fall inside India" if la["in_india"] is not None else "State boundaries unavailable"})
    checks.append({"ok": la["state"] is not None, "text": f"Start state — {la['state'] or 'outside India'}",
                   "sub": f"{la['district'] or '—'} district · point A"})
    checks.append({"ok": lb["state"] is not None, "text": f"End state — {lb['state'] or 'outside India'}",
                   "sub": f"{lb['district'] or '—'} district · point B"})
    single = la["state"] is not None and la["state"] == lb["state"]
    dist = None
    try:
        f = Frame(a, b, 1000, 0, 100)
        dist = f.length_m / 1000
    except ValueError as e:
        checks.append({"ok": False, "text": str(e)})
    if la["state"] and lb["state"]:
        checks.append({"ok": True, "text": "Corridor stays within one state" if single else
                       f"Corridor crosses states: {la['state']} → {lb['state']}"})
    import unicodedata
    norm = lambda v: unicodedata.normalize("NFKD", v or "").encode("ascii", "ignore").decode().lower().strip()
    region_ok = norm(la["state"]) == norm(settings.REGION) and norm(lb["state"]) == norm(settings.REGION)
    return {"a": la, "b": lb, "checks": checks, "single_state": single, "beeline_km": dist,
            "region_supported": region_ok,
            "ok": ok_coords and bool(in_india) and dist is not None}


def band_info(a, b, band_km: float) -> dict:
    f = Frame(tuple(a), tuple(b), band_km * 1000.0, 0.0, 100.0)
    band = f.band_polygon_lonlat()
    area = f.geom_to_utm(band).area / 1e6
    mins = 0.0
    for l in load_layers()["layers"]:
        mins += l.get("minutes_per_km2", 0) * area
    return {"band": json.loads(gpd.GeoSeries([band], crs=4326).to_json())["features"][0]["geometry"],
            "area_km2": round(area, 1), "beeline_km": round(f.length_m / 1000, 2),
            "extraction_minutes_full": round(mins, 1)}


def band_polygon(a, b, band_km: float, margin_m: float = 0.0):
    f = Frame(tuple(a), tuple(b), band_km * 1000.0, margin_m, 100.0)
    return f.band_polygon_lonlat(margin=margin_m > 0), f


def area_km2(geom_lonlat, frame: Frame) -> float:
    return frame.geom_to_utm(geom_lonlat).area / 1e6


def to_shape(geojson):
    return shape(geojson)
