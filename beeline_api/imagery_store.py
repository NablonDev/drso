"""Imagery catalog and acquisition.

LULC models never fetch imagery (Vivek, 5 Oct 2026). Imagery reaches the machine in two ways,
both recorded in the catalog table `imagery_items` with footprint, dates and origin:

  acquire  — online fetch under the operator's licences (Images screen → "Fetch imagery"):
             esri_wayback_z18 (MBTiles), sentinel2_l2a (B04/B08/B11/B12 clipped GeoTIFFs),
             sentinel1_rtc_vv (Planetary Computer, signed), worldcover (ESA, 10 m)
  register — files already on the machine: Cartosat-3 PAN scene folders, MBTiles, NISAR
             coherence or any raster (`python -m beeline_api.imagery_store register-...`)

Model plugins receive only catalog items intersecting their AOI (`for_aoi`).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import math
import re
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx
import mercantile
import shapely
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union

from . import db, settings

log = logging.getLogger(__name__)
PRODUCTS = ("esri_wayback_z18", "sentinel2_l2a", "sentinel1_rtc_vv", "cartosat3_pan", "worldcover", "nisar_coh")
ROOT = settings.IMAGERY_DIR

SCHEMA = """
create table if not exists imagery_items (
  id bigserial primary key,
  product text not null,
  item_key text not null,
  footprint geometry(MultiPolygon, 4326) not null,
  acquired date,
  acquired_end date,
  resolution_m double precision,
  path text not null,
  files jsonb not null default '{}'::jsonb,
  meta jsonb not null default '{}'::jsonb,
  origin text not null check (origin in ('fetched', 'registered')),
  licence text,
  run_id text,
  created_at timestamptz not null default now(),
  unique (product, item_key)
);
create index if not exists imagery_items_fp on imagery_items using gist (footprint);
"""


def init_schema():
    with db.conn() as c:
        with c.transaction():
            c.execute("select pg_advisory_xact_lock(724003)")
            c.execute(SCHEMA)


def _multi(g):
    g = shapely.make_valid(g)
    if g.geom_type == "Polygon":
        return shapely.MultiPolygon([g])
    if g.geom_type == "MultiPolygon":
        return g
    polys = [p for p in getattr(g, "geoms", []) if p.geom_type in ("Polygon", "MultiPolygon")]
    return _multi(unary_union(polys)) if polys else None


def upsert(product: str, item_key: str, footprint, path: str, origin: str, acquired=None, acquired_end=None,
           resolution_m=None, files=None, meta=None, licence=None, run_id=None) -> int:
    if product not in PRODUCTS:
        raise ValueError(f"unknown imagery product {product}")
    init_schema()
    fp = _multi(footprint)
    if fp is None or fp.is_empty:
        raise ValueError(f"{item_key}: empty footprint")
    row = db.one("""insert into imagery_items (product, item_key, footprint, acquired, acquired_end, resolution_m, path,
                        files, meta, origin, licence, run_id)
                    values (%s,%s,ST_GeomFromText(%s,4326),%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    on conflict (product, item_key) do update set footprint=excluded.footprint, acquired=excluded.acquired,
                        acquired_end=excluded.acquired_end, resolution_m=excluded.resolution_m, path=excluded.path,
                        files=excluded.files, meta=excluded.meta, origin=excluded.origin, licence=excluded.licence
                    returning id""",
                 (product, item_key, fp.wkt, acquired, acquired_end, resolution_m, str(path), db.J(files or {}),
                  db.J(meta or {}), origin, licence, run_id))
    return row["id"]


def for_aoi(aoi_lonlat, products=PRODUCTS) -> dict[str, list[dict]]:
    """Catalog items intersecting the AOI, shaped as a plugin request's `imagery` block."""
    init_schema()
    rows = db.all_("""select product, item_key, acquired, acquired_end, resolution_m, path, files, meta,
                             ST_AsGeoJSON(footprint, 6)::jsonb as footprint
                      from imagery_items where product = any(%s)
                        and ST_Intersects(footprint, ST_GeomFromText(%s, 4326))
                      order by product, acquired desc nulls last""", (list(products), aoi_lonlat.wkt))
    out: dict[str, list[dict]] = {p: [] for p in products}
    for r in rows:
        it = {"item_key": r["item_key"], "path": r["path"], "footprint": r["footprint"],
              "date": r["acquired"].isoformat() if r["acquired"] else None, **(r["meta"] or {})}
        if r["product"] == "sentinel2_l2a":
            it.update(scene_id=r["item_key"], bands=r["files"])
        elif r["product"] in ("sentinel1_rtc_vv", "cartosat3_pan"):
            it["scene_id"] = r["item_key"]
        out[r["product"]].append(it)
    return out


def coverage(aoi_lonlat) -> list[dict]:
    """Per product: share of the AOI covered, item count, newest/oldest date."""
    init_schema()
    rows = db.all_("""with a as (select ST_GeomFromText(%s, 4326) as g)
                      select p.product,
                             count(i.id) as n,
                             max(i.acquired) as newest, min(i.acquired) as oldest,
                             coalesce(ST_Area(ST_Intersection(ST_Union(i.footprint), (select g from a))::geography)
                                      / ST_Area((select g from a)::geography), 0) as frac
                      from unnest(%s::text[]) as p(product)
                      left join imagery_items i on i.product = p.product
                           and ST_Intersects(i.footprint, (select g from a))
                      group by p.product order by p.product""", (aoi_lonlat.wkt, list(PRODUCTS)))
    return [{"product": r["product"], "items": r["n"], "covered_pct": round(100 * float(r["frac"] or 0), 1),
             "newest": r["newest"].isoformat() if r["newest"] else None,
             "oldest": r["oldest"].isoformat() if r["oldest"] else None} for r in rows]


# ---- registration of files already on the machine --------------------------------------
def _raster_footprint(path: Path):
    import rasterio
    from rasterio.warp import transform_bounds
    with rasterio.open(path) as r:
        b = transform_bounds(r.crs, "EPSG:4326", *r.bounds)
        res = abs(r.res[0]) if r.crs and r.crs.is_projected else None
    return box(*b), res


def _pan_date(folder: Path) -> date | None:
    for name in ("BAND_META.txt", "BAND_META.xml"):
        p = folder / name
        if not p.exists():
            continue
        t = p.read_text(errors="ignore")
        m = re.search(r"(?:DateOfPass|Acquisition_?Date|SceneDate|ImagingDate)\W*[:=>]\s*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4}|[0-9]{4}-[0-9]{2}-[0-9]{2})", t, re.I)
        if m:
            s = m.group(1)
            for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(s, fmt).date()
                except ValueError:
                    pass
    return None


def register_pan(root: Path, licence: str | None = None, overrides: dict[str, str] | None = None) -> list[str]:
    """Every <root>/**/<scene>/BAND.tif becomes a cartosat3_pan item (footprint from BAND.shp or raster)."""
    import geopandas as gpd
    out = []
    overrides = overrides or {}
    for band in sorted(Path(root).rglob("BAND.tif")):
        sid = band.parent.name
        path = Path(overrides.get(sid, band))
        shp = band.parent / "BAND.shp"
        fp = None
        if shp.exists():
            try:
                fp = unary_union(list(gpd.read_file(shp).to_crs(4326).geometry))
            except Exception:
                fp = None
        rfp, res = _raster_footprint(path)
        upsert("cartosat3_pan", sid, fp or rfp, str(path), "registered", acquired=_pan_date(band.parent),
               resolution_m=res, meta={"order": band.parent.parent.name, "sensor": "Cartosat-3 PAN"}, licence=licence)
        out.append(sid)
    return out


def mbtiles_footprint(path: Path, zoom: int = 18):
    """Union of 64x64-tile cells that hold tiles at `zoom` (cheap, from the tiles index)."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute("""select tile_column / 64 as cx, tile_row / 64 as cy from tiles where zoom_level=?
                          group by 1, 2""", (zoom,)).fetchall()
    meta = dict(con.execute("select name, value from metadata").fetchall()) if con.execute(
        "select count(*) from sqlite_master where name='metadata'").fetchone()[0] else {}
    con.close()
    n = 2 ** zoom
    cells = []
    for cx, cy in rows:
        x0, x1 = cx * 64, cx * 64 + 64
        # MBTiles rows are TMS (origin bottom-left)
        y_tms0, y_tms1 = cy * 64, cy * 64 + 64
        y0, y1 = n - y_tms1, n - y_tms0
        w = mercantile.bounds(x0, y0, zoom).west
        nn = mercantile.bounds(x0, y0, zoom).north
        e = mercantile.bounds(x1 - 1, y1 - 1, zoom).east
        s = mercantile.bounds(x1 - 1, y1 - 1, zoom).south
        cells.append(box(w, s, e, nn))
    return unary_union(cells) if cells else None, meta


def register_mbtiles(path: Path, release: str, acquired: str | None, licence: str | None = None) -> str:
    fp, meta = mbtiles_footprint(Path(path))
    if fp is None:
        raise ValueError(f"{path}: no z18 tiles")
    key = f"{release}:{Path(path).stem}"
    upsert("esri_wayback_z18", key, fp, str(path), "registered", acquired=acquired, resolution_m=0.597,
           meta={"release": release, "zoom": 18, "name": meta.get("name")}, licence=licence)
    return key


def register_raster(product: str, path: Path, acquired: str | None = None, key: str | None = None,
                    licence: str | None = None, meta: dict | None = None) -> str:
    fp, res = _raster_footprint(Path(path))
    key = key or Path(path).stem
    upsert(product, key, fp, str(path), "registered", acquired=acquired, resolution_m=res, meta=meta or {},
           licence=licence)
    return key


# ---- licensed acquisition (online) ------------------------------------------------------
def _client():
    return httpx.Client(timeout=60, headers={"User-Agent": settings.HTTP_UA}, follow_redirects=True,
                        limits=httpx.Limits(max_connections=32))


def acquire_wayback(aoi_lonlat, run_id: str, release: str | None = None, buffer_m: float = 2500,
                    progress=None, cancel=None) -> dict:
    """z18 tiles over AOI+buffer into one MBTiles (TMS rows), following release redirects."""
    release = release or settings.WAYBACK_RELEASE
    lat = aoi_lonlat.centroid.y
    buf = aoi_lonlat.buffer(buffer_m / (111320 * max(0.2, math.cos(math.radians(lat)))))
    tiles = [t for t in mercantile.tiles(*buf.bounds, 18) if buf.intersects(box(*mercantile.bounds(t)))]
    out = ROOT / "esri_wayback" / f"{run_id}_{release}.mbtiles"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.executescript("""create table metadata (name text, value text);
                         create table tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);""")
    con.executemany("insert into metadata values (?,?)", [("name", f"{run_id} Wayback {release}"), ("format", "jpg"),
                                                          ("minzoom", "18"), ("maxzoom", "18"),
                                                          ("bounds", ",".join(f"{v:.6f}" for v in buf.bounds))])
    tok = settings.ESRI_TOKEN
    url = settings.WAYBACK_URL
    releases: dict[str, int] = {}
    failed = 0

    def get(t):
        u = url.format(release=release, z=t.z, x=t.x, y=t.y) + (f"?token={tok}" if tok else "")
        for _ in range(3):
            try:
                r = c.get(u)
                if r.status_code == 200 and r.content:
                    m = re.search(r"/tile/(\d+)/", str(r.url))
                    return t, r.content, m.group(1) if m else release
                if r.status_code == 404:
                    return t, None, None
            except httpx.HTTPError:
                continue
        return t, None, "error"

    with _client() as c, cf.ThreadPoolExecutor(16) as ex:
        for i, (t, data, rel) in enumerate(ex.map(get, tiles)):
            if data is not None:
                con.execute("insert into tiles values (18,?,?,?)", (t.x, (2 ** 18 - 1) - t.y, data))
                releases[rel] = releases.get(rel, 0) + 1
            elif rel == "error":
                failed += 1
            if i % 2000 == 0:
                con.commit()
                if progress:
                    progress(i / max(1, len(tiles)), f"Esri Wayback tiles {i:,}/{len(tiles):,}")
                if cancel and cancel():
                    break
    con.execute("create unique index tile_index on tiles (zoom_level, tile_column, tile_row)")
    con.commit()
    con.close()
    tmp.rename(out)
    key = register_mbtiles(out, release, settings.WAYBACK_RELEASE_DATE, licence=settings.LICENCE_ESRI)
    db.exec_("update imagery_items set origin='fetched', run_id=%s, meta = meta || %s where product='esri_wayback_z18' and item_key=%s",
             (run_id, db.J({"releases": releases, "failed_tiles": failed, "tiles": len(tiles)}), key))
    return {"item": key, "tiles": len(tiles), "failed": failed, "releases": releases}


def _window_to_file(href: str, aoi_lonlat, dst: Path) -> bool:
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(href) as src:
        b = transform_bounds("EPSG:4326", src.crs, *aoi_lonlat.bounds)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            return False
        arr = src.read(window=win)
        prof = src.profile.copy()
        prof.update(width=win.width, height=win.height, transform=src.window_transform(win), driver="GTiff",
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dst, "w", **prof) as d:
        d.write(arr)
    return True


def acquire_sentinel2(aoi_lonlat, run_id: str, months: int = 12, max_cloud: float = 20.0, progress=None,
                      cancel=None) -> dict:
    """Lowest-cloud L2A scene per MGRS tile per month; B04/B08/B11/B12 clipped to the AOI."""
    windows = []
    first = date.today().replace(day=1)
    for k in range(months):
        y, m = first.year, first.month - k
        while m <= 0:
            y, m = y - 1, m + 12
        m_start = date(y, m, 1)
        m_end = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)) if k else date.today()
        windows.append((m_start, m_end))
    got = []
    with _client() as c:
        for k, (m_start, m_end) in enumerate(windows):
            if cancel and cancel():
                break
            body = {"collections": ["sentinel-2-l2a"], "intersects": mapping(aoi_lonlat),
                    "datetime": f"{m_start}T00:00:00Z/{m_end}T23:59:59Z",
                    "query": {"eo:cloud_cover": {"lt": max_cloud}}, "limit": 200}
            r = c.post(settings.STAC_URL, json=body)
            r.raise_for_status()
            best: dict[str, dict] = {}
            for f in r.json().get("features", []):
                tile = f["properties"].get("grid:code") or f["properties"].get("s2:mgrs_tile") or f["id"].split("_")[1]
                if tile not in best or f["properties"]["eo:cloud_cover"] < best[tile]["properties"]["eo:cloud_cover"]:
                    best[tile] = f
            for f in best.values():
                files = {}
                for band, asset in (("B04", "red"), ("B08", "nir"), ("B11", "swir16"), ("B12", "swir22")):
                    href = f["assets"].get(asset, {}).get("href")
                    if not href:
                        continue
                    dst = ROOT / "sentinel2" / run_id / f["id"] / f"{band}.tif"
                    if dst.exists() or _window_to_file(href, aoi_lonlat, dst):
                        files[band] = str(dst)
                if len(files) == 4:
                    d = f["properties"]["datetime"][:10]
                    upsert("sentinel2_l2a", f["id"], shape(f["geometry"]).intersection(aoi_lonlat.buffer(0.01)),
                           str(ROOT / "sentinel2" / run_id / f["id"]), "fetched", acquired=d, resolution_m=10,
                           files=files, meta={"cloud": f["properties"]["eo:cloud_cover"],
                                              # the solar NIR band was calibrated on offset-applied values
                                              "boa_offset_applied": f["properties"].get("earthsearch:boa_offset_applied", True),
                                              "processing_baseline": f["properties"].get("s2:processing_baseline")},
                           licence="Copernicus Sentinel data (open)", run_id=run_id)
                    got.append(f["id"])
            if progress:
                progress((k + 1) / months, f"Sentinel-2 month {m_start:%b %Y}: {len(best)} tiles")
    return {"scenes": len(got)}


def _pc_sign(c: httpx.Client, href: str, collection: str, cache: dict) -> str:
    if collection not in cache:
        h = {"Ocp-Apim-Subscription-Key": settings.PC_SUBSCRIPTION_KEY} if settings.PC_SUBSCRIPTION_KEY else {}
        r = c.get(f"https://planetarycomputer.microsoft.com/api/sas/v1/token/{collection}", headers=h)
        r.raise_for_status()
        cache[collection] = r.json()["token"]
    return f"{href}?{cache[collection]}"


def acquire_sentinel1(aoi_lonlat, run_id: str, n_scenes: int = 8, months: int = 11, progress=None,
                      cancel=None) -> dict:
    """Sentinel-1 RTC VV (Planetary Computer), clipped to the AOI; one scene per orbit-month."""
    end = date.today()
    start = end - timedelta(days=int(months * 30.4))
    body = {"collections": ["sentinel-1-rtc"], "intersects": mapping(aoi_lonlat),
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200}
    got = []
    with _client() as c:
        r = c.post("https://planetarycomputer.microsoft.com/api/stac/v1/search", json=body)
        r.raise_for_status()
        feats = sorted(r.json().get("features", []), key=lambda f: f["properties"]["datetime"], reverse=True)
        seen, tok = set(), {}
        for f in feats:
            if len(got) >= n_scenes or (cancel and cancel()):
                break
            key = (f["properties"].get("sat:relative_orbit"), f["properties"]["datetime"][:7])
            if key in seen or "vv" not in f["assets"]:
                continue
            seen.add(key)
            dst = ROOT / "sentinel1" / run_id / f"{f['id']}_vv.tif"
            if dst.exists() or _window_to_file(_pc_sign(c, f["assets"]["vv"]["href"], "sentinel-1-rtc", tok), aoi_lonlat, dst):
                upsert("sentinel1_rtc_vv", f["id"], shape(f["geometry"]).intersection(aoi_lonlat.buffer(0.01)), str(dst),
                       "fetched", acquired=f["properties"]["datetime"][:10], resolution_m=10,
                       meta={"orbit": f["properties"].get("sat:relative_orbit")},
                       licence="Copernicus Sentinel data (open), via Microsoft Planetary Computer", run_id=run_id)
                got.append(f["id"])
            if progress:
                progress(len(got) / n_scenes, f"Sentinel-1 scenes {len(got)}/{n_scenes}")
    return {"scenes": len(got)}


def acquire_worldcover(aoi_lonlat, run_id: str, progress=None, cancel=None) -> dict:
    """ESA WorldCover 2021 v200 3x3° tiles intersecting the AOI, clipped to the AOI."""
    minx, miny, maxx, maxy = aoi_lonlat.bounds
    got = []
    for lat in range(int(math.floor(miny / 3) * 3), int(math.floor(maxy / 3) * 3) + 1, 3):
        for lon in range(int(math.floor(minx / 3) * 3), int(math.floor(maxx / 3) * 3) + 1, 3):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            name = f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map"
            href = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{name}.tif"
            dst = ROOT / "worldcover" / run_id / f"{name}.tif"
            try:
                ok = dst.exists() or _window_to_file(href, aoi_lonlat, dst)
            except Exception as e:
                log.warning("WorldCover %s: %s", name, e)
                ok = False
            if ok:
                key = f"{run_id}:{name}"
                register_raster("worldcover", dst, acquired="2021-12-31", key=key,
                                licence="ESA WorldCover 2021 (CC BY 4.0)", meta={"year": 2021})
                db.exec_("update imagery_items set origin='fetched', run_id=%s where product='worldcover' and item_key=%s",
                         (run_id, key))
                got.append(name)
    if progress:
        progress(1.0, f"WorldCover tiles {len(got)}")
    return {"tiles": got}


ACQUIRERS = {"esri_wayback_z18": acquire_wayback, "sentinel2_l2a": acquire_sentinel2,
             "sentinel1_rtc_vv": acquire_sentinel1, "worldcover": acquire_worldcover}
FETCHABLE = tuple(ACQUIRERS)


def acquire(products: list[str], aoi_lonlat, run_id: str, progress=None, cancel=None) -> dict:
    bad = [p for p in products if p not in ACQUIRERS]
    if bad:
        raise ValueError(f"cannot be fetched online: {bad} (register local files instead)")
    out = {}
    for i, p in enumerate(products):
        def sub(f, msg, _i=i):
            if progress:
                progress((_i + f) / len(products), msg)
        out[p] = ACQUIRERS[p](aoi_lonlat, run_id, progress=sub, cancel=cancel)
    return out


# ---- CLI -------------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(prog="python -m beeline_api.imagery_store")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("register-pan", help="register Cartosat-3 PAN scene folders (…/<scene>/BAND.tif)")
    p.add_argument("root", nargs="+")
    p.add_argument("--override", action="append", default=[], help="SCENE=PATH to a replacement BAND.tif")
    p.add_argument("--licence")
    p = sub.add_parser("register-mbtiles")
    p.add_argument("path")
    p.add_argument("--release", required=True)
    p.add_argument("--date")
    p.add_argument("--licence")
    p = sub.add_parser("register-raster")
    p.add_argument("product", choices=PRODUCTS)
    p.add_argument("path")
    p.add_argument("--date")
    p.add_argument("--key")
    p.add_argument("--licence")
    sub.add_parser("list")
    a = ap.parse_args()
    db.init_schema()
    init_schema()
    if a.cmd == "register-pan":
        ov = dict(x.split("=", 1) for x in a.override)
        for r in a.root:
            print(register_pan(Path(r), a.licence, ov))
    elif a.cmd == "register-mbtiles":
        print(register_mbtiles(Path(a.path), a.release, a.date, a.licence))
    elif a.cmd == "register-raster":
        print(register_raster(a.product, Path(a.path), a.date, a.key, a.licence))
    else:
        for r in db.all_("select product, item_key, acquired, origin, path from imagery_items order by product, item_key"):
            print(r["product"], r["item_key"], r["acquired"], r["origin"], r["path"])


if __name__ == "__main__":
    main()
