"""Imagery over a band: what exists online, what is held locally, and the Esri tile proxy.

- Esri World Imagery: capture date and resolution sampled with the MapServer identify call on a
  grid over the band (the service's citation layer). Tiles are proxied and cached in the bucket.
- Sentinel-2 L2A: Element84 Earth Search STAC (AWS open data), least-cloudy recent scenes.
- DEM: Copernicus GLO-30 tiles (beeline_engine.dem).

With BEELINE_FETCH_IMAGERY=1 the extraction step also downloads the Sentinel-2 scenes and the
Esri z18 tiles of the band into the bucket (needed once real models are plugged in).
"""
from __future__ import annotations

import concurrent.futures as cf
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import mercantile
import numpy as np
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

from beeline_engine import dem as demmod
from beeline_engine.frame import Frame

from . import settings

log = logging.getLogger(__name__)
ESRI_IDENTIFY = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/identify"
STAC = "https://earth-search.aws.element84.com/v1/search"
HR_MAX_RES_M = 0.6          # "high resolution" for the buildings / lines models (50 cm class)
HR_MAX_AGE_MONTHS = 18      # older captures count as a gap (prototype rule)


def _client():
    return httpx.Client(timeout=30, headers={"User-Agent": settings.HTTP_UA}, follow_redirects=True)


def esri_identify(lon: float, lat: float, client: httpx.Client) -> dict | None:
    d = 0.01
    params = {"geometry": f"{lon},{lat}", "geometryType": "esriGeometryPoint", "sr": 4326, "layers": "all:0",
              "tolerance": 1, "mapExtent": f"{lon - d},{lat - d},{lon + d},{lat + d}", "imageDisplay": "400,400,96",
              "returnGeometry": "false", "f": "json"}
    for _ in range(3):
        try:
            r = client.get(ESRI_IDENTIFY, params=params)
            r.raise_for_status()
            res = [x for x in r.json().get("results", []) if x.get("layerId") == 0]
            if not res:
                return None
            a = res[0]["attributes"]
            ds = a.get("DATE (YYYYMMDD)")
            dt = datetime.strptime(ds, "%Y%m%d").date() if ds and ds.isdigit() else None
            return {"date": dt.isoformat() if dt else None, "resolution_m": float(a.get("RESOLUTION (M)") or 0) or None,
                    "source": a.get("SOURCE"), "sensor": a.get("DESCRIPTION"), "release": a.get("ReleaseName")}
        except (httpx.HTTPError, ValueError, KeyError):
            continue
    return None


def esri_coverage(frame: Frame, spacing_m: float = 3000.0) -> dict:
    """Sample the Esri citation on a grid over the band; classify each sample cell."""
    xs = np.arange(spacing_m / 2, frame.length_m, spacing_m)
    ys = np.arange(-frame.half_width_m + spacing_m / 2, frame.half_width_m, spacing_m)
    if len(ys) == 0:
        ys = np.array([0.0])
    pts = [(x, y) for x in xs for y in ys]
    lon, lat = frame.frame_to_lonlat([p[0] for p in pts], [p[1] for p in pts])
    today = date.today()
    cutoff = today - timedelta(days=int(HR_MAX_AGE_MONTHS * 30.4))
    with _client() as client, cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda ll: esri_identify(ll[0], ll[1], client), zip(lon, lat)))
    cells, gap_cells = [], []
    dates = []
    for (x, y), r in zip(pts, res):
        cell = box(x - spacing_m / 2, y - spacing_m / 2, x + spacing_m / 2, y + spacing_m / 2)
        ok = bool(r and r["resolution_m"] and r["resolution_m"] <= HR_MAX_RES_M and r["date"]
                  and date.fromisoformat(r["date"]) >= cutoff)
        (cells if ok else gap_cells).append(cell)
        if r and r["date"]:
            dates.append(r["date"])
    band = box(0, -frame.half_width_m, frame.length_m, frame.half_width_m)
    gap = unary_union(gap_cells).intersection(band) if gap_cells else None
    gap_area = gap.area / 1e6 if gap is not None and not gap.is_empty else 0.0
    total = band.area / 1e6
    ranges = []
    if gap is not None and not gap.is_empty:
        for g in sorted(getattr(gap, "geoms", [gap]), key=lambda g: g.bounds[0]):
            ranges.append([round(g.bounds[0] / 1000, 1), round(g.bounds[2] / 1000, 1)])
    good = [r for r in res if r]
    res_m = sorted({r["resolution_m"] for r in good if r["resolution_m"]})
    gap_lonlat = None
    if gap is not None and not gap.is_empty:
        from shapely import affinity
        g = affinity.rotate(gap, frame.bearing_rad, origin=(0, 0), use_radians=True)
        g = affinity.translate(g, frame.A[0], frame.A[1])
        gap_lonlat = frame.geom_to_lonlat(g)
    return {
        "product": "Esri World Imagery", "samples": len(pts),
        "dates": sorted(set(dates)), "newest": max(dates) if dates else None, "oldest": min(dates) if dates else None,
        "resolutions_m": res_m, "sources": sorted({f"{r['source']} {r['sensor'] or ''}".strip() for r in good if r.get("source")}),
        "covered_pct": round(100 * (1 - gap_area / total), 1) if total else 0,
        "gap_km2": round(gap_area, 1), "gap_ranges_km": ranges,
        "gap_geojson": gap_lonlat.__geo_interface__ if gap_lonlat is not None else None,
        "rule": f"≤ {HR_MAX_RES_M} m and captured within {HR_MAX_AGE_MONTHS} months",
    }


def sentinel2(band_lonlat, days: int = 90, max_cloud: float = 20.0) -> dict:
    end = date.today()
    start = end - timedelta(days=days)
    body = {"collections": ["sentinel-2-l2a"], "intersects": band_lonlat.__geo_interface__,
            "datetime": f"{start.isoformat()}T00:00:00Z/{end.isoformat()}T23:59:59Z",
            "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": 100,
            "fields": {"include": ["id", "properties.datetime", "properties.eo:cloud_cover", "geometry", "assets.visual.href"]}}
    with _client() as c:
        r = c.post(STAC, json=body)
        r.raise_for_status()
        feats = r.json().get("features", [])
    feats.sort(key=lambda f: (f["properties"]["datetime"][:10], -f["properties"].get("eo:cloud_cover", 100)), reverse=True)
    covered = None
    chosen = []
    for f in feats:
        g = shape(f["geometry"])
        if covered is not None and covered.buffer(1e-6).contains(band_lonlat):
            break
        if covered is None or not covered.contains(g.intersection(band_lonlat)):
            chosen.append(f)
            covered = g if covered is None else covered.union(g)
    cov_pct = 0.0
    if covered is not None:
        cov_pct = 100 * band_lonlat.intersection(covered).area / band_lonlat.area
    scenes = [{"id": f["id"], "date": f["properties"]["datetime"][:10], "cloud": round(f["properties"].get("eo:cloud_cover", 0), 1),
               "visual": f.get("assets", {}).get("visual", {}).get("href")} for f in chosen]
    return {"product": "Sentinel-2 L2A", "resolution_m": 10, "scenes": scenes, "covered_pct": round(cov_pct, 1),
            "newest": scenes[0]["date"] if scenes else None,
            "cloud": round(max((s["cloud"] for s in scenes), default=0), 1)}


def dem_status(band_lonlat) -> dict:
    minx, miny, maxx, maxy = band_lonlat.bounds
    names = [demmod.tile_name(la, lo) for la in range(math.floor(miny), math.floor(maxy) + 1)
             for lo in range(math.floor(minx), math.floor(maxx) + 1)]
    held = [n for n in names if (settings.DEM_DIR / f"{n}.tif").exists()]
    return {"product": "Copernicus GLO-30 DEM", "resolution_m": 30, "tiles": names, "held": held,
            "covered_pct": 100.0, "note": "fetched on demand into the bucket"}


def check(frame: Frame, band_lonlat) -> dict:
    out = {"checked_at": datetime.utcnow().isoformat() + "Z"}
    try:
        out["esri"] = esri_coverage(frame)
    except Exception as e:
        out["esri"] = {"product": "Esri World Imagery", "error": f"{type(e).__name__}: {e}"}
    try:
        out["sentinel2"] = sentinel2(band_lonlat)
    except Exception as e:
        out["sentinel2"] = {"product": "Sentinel-2 L2A", "error": f"{type(e).__name__}: {e}"}
    out["dem"] = dem_status(band_lonlat)
    esri = out.get("esri", {})
    out["covered"] = bool(esri.get("covered_pct", 0) >= 99.5)
    return out


# ---- Esri tile proxy with on-disk cache -------------------------------------------------
_http = None


def tile(z: int, x: int, y: int) -> tuple[bytes | None, str]:
    """(bytes, content_type) from cache or upstream; (None, reason) when it should 404."""
    if z > settings.ESRI_MAX_ZOOM:
        return None, "above native zoom"
    p = settings.TILE_CACHE_DIR / str(z) / str(x) / f"{y}.jpg"
    if p.exists():
        return p.read_bytes(), "image/jpeg"
    global _http
    if _http is None:
        _http = httpx.Client(timeout=20, headers={"User-Agent": settings.HTTP_UA}, follow_redirects=True,
                             limits=httpx.Limits(max_connections=32))
    url = settings.ESRI_TILE_URL.format(z=z, x=x, y=y)
    r = _http.get(url)
    if r.status_code != 200 or not r.content:
        return None, f"upstream {r.status_code}"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.rename(p)
    return r.content, r.headers.get("content-type", "image/jpeg")


def prefetch_esri(band_lonlat, zoom: int = 18, progress=None, cancel=None) -> int:
    """Download every z-level tile over the band into the cache (used when real models run)."""
    minx, miny, maxx, maxy = band_lonlat.bounds
    tiles = [t for t in mercantile.tiles(minx, miny, maxx, maxy, zoom)
             if band_lonlat.intersects(box(*mercantile.bounds(t)))]
    n = 0
    with cf.ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(lambda t: tile(t.z, t.x, t.y), tiles)):
            n += 1
            if progress and i % 500 == 0:
                progress(i / max(1, len(tiles)))
            if cancel and cancel():
                break
    return n


def fetch_sentinel2_visual(scenes: list[dict], band_lonlat, out_dir: Path) -> list[Path]:
    """Clip each chosen scene's visual COG to the band and store it in the bucket."""
    import rasterio
    from rasterio.mask import mask as rmask
    from rasterio.warp import transform_geom
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in scenes:
        if not s.get("visual"):
            continue
        dst = out_dir / f"{s['id']}_visual_band.tif"
        if dst.exists():
            paths.append(dst)
            continue
        with rasterio.open(s["visual"]) as src:
            g = transform_geom("EPSG:4326", src.crs, band_lonlat.__geo_interface__)
            try:
                arr, tr = rmask(src, [g], crop=True)
            except ValueError:
                continue
            prof = src.profile.copy()
            prof.update(height=arr.shape[1], width=arr.shape[2], transform=tr, driver="GTiff", compress="deflate", tiled=True)
            with rasterio.open(dst, "w", **prof) as d:
                d.write(arr)
        paths.append(dst)
    return paths
