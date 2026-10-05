"""Airports and runways for India — the two aviation layers.

  airports  layer `aviation` ("Airports", class airport). Esri India Living Atlas
            "Airport_Boundary" (AAI-attributed boundaries, 269 airports on 5 Oct 2026), plus the
            OSM aeroway=aerodrome polygons that no Esri polygon overlaps. The delivered
            airport_boundary_merged.gpkg stacked both without de-duplication (255 of its 370 OSM
            polygons sit on an Esri one), so it is not loaded as is.
  runways   layer `approach` ("Airport approach path", class runway). OSM aeroway=runway from
            the delivered airports.gpkg: polygons kept, centrelines widened by their width tag
            (else 45 m, the code-4 runway width) with flat ends. Disused / abandoned runways are
            dropped. The airport-prohibition rule measures from these polygons.

Store: PostGIS tables `aviation_airports`, `aviation_runways`; every load is logged in
`aviation_loads` (layer, kind, data date, AOI, run).

  India load   python -m beeline_api.aviation load-airports [--osm-gpkg airports.gpkg]
               python -m beeline_api.aviation load-runways --gpkg airports.gpkg
  AOI refresh  refresh_airports(bbox) — Living Atlas envelope query + Overpass aeroway=aerodrome;
               replaces the airports of that bbox (both sources) in one transaction, OSM ones
               again only where no Esri polygon overlaps. UI: "Refresh from Living Atlas + OSM".
               refresh_runways(bbox) — Overpass aeroway=runway; replaces the runways of that bbox.
               UI: "Refresh from OSM" on the approach path layer.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import httpx
import pandas as pd
import shapely
from shapely.geometry import box, shape
from shapely.ops import unary_union

from . import db, settings

log = logging.getLogger(__name__)
LIVING_ATLAS_URL = "https://livingatlas.esri.in/server1/rest/services/India/Airport_Boundary/MapServer/0"
ESRI_FIELDS = ("airport", "other", "state", "amsl", "rundes", "dim", "opeown", "scdul", "source")
DEFAULT_RUNWAY_WIDTH_M = 45.0
PAGE = 1000

SCHEMA = """
create table if not exists aviation_airports (
  id bigserial primary key,
  source text not null check (source in ('esri', 'osm')),
  src_id text not null,
  name text,
  tags jsonb not null default '{}'::jsonb,
  geom geometry(MultiPolygon, 4326) not null,
  load_id bigint
);
create index if not exists aviation_airports_geom on aviation_airports using gist (geom);
create index if not exists aviation_airports_src on aviation_airports (source, src_id);

create table if not exists aviation_runways (
  id bigserial primary key,
  src_id text not null,
  name text,
  width_m double precision,
  width_source text,
  tags jsonb not null default '{}'::jsonb,
  geom geometry(MultiPolygon, 4326) not null,
  load_id bigint
);
create index if not exists aviation_runways_geom on aviation_runways using gist (geom);

create table if not exists aviation_loads (
  id bigserial primary key,
  layer text not null check (layer in ('airports', 'runways')),
  kind text not null check (kind in ('india', 'aoi')),
  source text not null,
  data_date timestamptz,
  aoi geometry(Polygon, 4326),
  run_id text,
  status text not null default 'running',
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  counts jsonb not null default '{}'::jsonb,
  error text
);
"""


def init_schema():
    with db.conn() as c:
        with c.transaction():
            c.execute("select pg_advisory_xact_lock(724005)")
            c.execute(SCHEMA)


def _multi(g):
    g = shapely.make_valid(g)
    if g.geom_type == "GeometryCollection":
        g = unary_union([p for p in g.geoms if p.geom_type in ("Polygon", "MultiPolygon")])
    if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
        return None
    return g if g.geom_type == "MultiPolygon" else shapely.MultiPolygon([g])


def _clean(v):
    if v is None or (isinstance(v, float) and v != v):
        return None
    return v.item() if hasattr(v, "item") else v


def _wkb(g):
    return shapely.to_wkb(shapely.set_srid(g, 4326), hex=True, include_srid=True)


# ---- Esri Living Atlas ----------------------------------------------------------------
def _get(params: dict) -> dict:
    last = None
    for attempt in range(4):
        try:
            r = httpx.get(f"{LIVING_ATLAS_URL}/query", params=params, timeout=120,
                          headers={"User-Agent": settings.HTTP_UA})
            if r.status_code == 200:
                d = r.json()
                if "error" not in d:
                    return d
                last = f"error {d['error'].get('code')}: {d['error'].get('message')}"
            else:
                last = f"HTTP {r.status_code}"
        except (httpx.HTTPError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Esri Living Atlas unavailable ({last})")


def fetch_esri(bbox: tuple[float, float, float, float] | None = None) -> list[dict]:
    """All Living Atlas airports (or those intersecting bbox) as [{src_id, name, tags, geom}]."""
    base = {"where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson", "returnGeometry": "true",
            "orderByFields": "objectid"}
    if bbox:
        base.update(geometry=",".join(str(v) for v in bbox), geometryType="esriGeometryEnvelope", inSR=4326,
                    spatialRel="esriSpatialRelIntersects")
    out, offset = [], 0
    while True:
        d = _get({**base, "resultOffset": offset, "resultRecordCount": PAGE})
        feats = d.get("features", [])
        for f in feats:
            p = f.get("properties") or {}
            g = _multi(shape(f["geometry"])) if f.get("geometry") else None
            if g is None:
                continue
            tags = {("boundary_source" if k == "source" else k): (v.strip() if isinstance(v, str) else v)
                    for k in ESRI_FIELDS if (v := _clean(p.get(k))) not in (None, "")}   # Esri "source" = AAI report
            out.append({"src_id": str(p.get("objectid")), "name": p.get("airport") or p.get("other"),
                        "tags": tags, "geom": g})
        if len(feats) < PAGE and not d.get("exceededTransferLimit"):
            break
        offset += len(feats)
        if not feats:
            break
    return out


# ---- airports ---------------------------------------------------------------------------
_NOT_AIRPORT = re.compile(r"heliport|helipad|seaplane", re.I)


def _is_airport(name, all_tags) -> bool:
    """Live airports only: disused / abandoned aerodromes, heliports and seaplane ports are not kept."""
    t = json.loads(all_tags) if isinstance(all_tags, str) else (all_tags or {})
    if t.get("disused") == "yes" or t.get("abandoned") == "yes" or "disused:aeroway" in t or "abandoned:aeroway" in t:
        return False
    return not _NOT_AIRPORT.search(str(name or "")) and t.get("aerodrome:type") != "heliport"


# ---- OpenStreetMap (Overpass) ----------------------------------------------------------
def _overpass(query: str, timeout: int = 180) -> dict:
    from . import osm_power
    return osm_power.overpass_query(query, timeout)


def _osm_query(key: str, value: str, bbox) -> str:
    minx, miny, maxx, maxy = bbox
    bb = f"{miny},{minx},{maxy},{maxx}"
    return f'[out:json][timeout:180];(way["{key}"="{value}"]({bb});relation["{key}"="{value}"]({bb}););out geom;'


def _element_geom(el: dict):
    """Overpass `out geom` element -> shapely: closed way = Polygon (as the delivered gpkg does),
    open way = LineString, multipolygon relation = (Multi)Polygon from its outer / inner members."""
    if el["type"] == "way":
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry") or [] if p]
        if len(pts) < 2:
            return None
        if len(pts) >= 4 and pts[0] == pts[-1]:
            return shapely.Polygon(pts)
        return shapely.LineString(pts)
    if el["type"] == "relation":
        rings = {"outer": [], "inner": []}
        for m in el.get("members", []):
            pts = [(p["lon"], p["lat"]) for p in m.get("geometry") or [] if p]
            if m.get("type") == "way" and len(pts) >= 2:
                rings["inner" if m.get("role") == "inner" else "outer"].append(shapely.LineString(pts))
        outer = unary_union(list(shapely.polygonize(rings["outer"]).geoms)) if rings["outer"] else None
        if outer is None or outer.is_empty:
            return None
        if rings["inner"]:
            outer = outer.difference(unary_union(list(shapely.polygonize(rings["inner"]).geoms)))
        return outer
    return None


def fetch_osm_aerodromes(bbox) -> tuple[list[dict], datetime]:
    """Live OSM airports intersecting bbox, filtered like the India load."""
    d = _overpass(_osm_query("aeroway", "aerodrome", bbox))
    out = []
    for el in d.get("elements", []):
        t = el.get("tags", {})
        g = _element_geom(el)
        g = _multi(g) if g is not None and g.geom_type in ("Polygon", "MultiPolygon") else None
        if g is None or not _is_airport(t.get("name"), t):
            continue
        tags = {"iata": t.get("iata"), "icao": t.get("icao"), "operator": t.get("operator"), "name_en": t.get("name:en")}
        out.append({"src_id": f"{el['type']}/{el['id']}", "name": t.get("name"),
                    "tags": {k: v for k, v in tags.items() if v}, "geom": g})
    return out, _osm_ts(d)


def fetch_osm_runways(bbox) -> tuple[gpd.GeoDataFrame, datetime]:
    """Live OSM runways intersecting bbox, in the delivered gpkg's shape (for runway_polygons)."""
    d = _overpass(_osm_query("aeroway", "runway", bbox))
    rows = []
    for el in d.get("elements", []):
        g = _element_geom(el)
        if g is not None and not g.is_empty:
            t = el.get("tags", {})
            rows.append({"osm_type": el["type"], "osm_id": el["id"], "name": t.get("name"), "all_tags": json.dumps(t),
                         "geometry": g})
    g = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326) if rows else gpd.GeoDataFrame(
        {"osm_type": [], "osm_id": [], "name": [], "all_tags": []}, geometry=[], crs=4326)
    return g, _osm_ts(d)


def _osm_ts(d: dict) -> datetime:
    ts = (d.get("osm3s") or {}).get("timestamp_osm_base")
    return datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)


def _osm_aerodromes(gpkg: Path) -> list[dict]:
    g = gpd.read_file(gpkg, layer="aeroway_aerodrome").to_crs(4326)
    out = []
    for r in g.itertuples(index=False):
        geom = _multi(r.geometry) if r.geometry is not None else None
        if geom is None or not _is_airport(r.name, getattr(r, "all_tags", None)):
            continue
        tags = {k: _clean(getattr(r, k, None)) for k in ("iata", "icao", "operator", "name_en")}
        out.append({"src_id": f"{r.osm_type}/{r.osm_id}", "name": _clean(r.name), "tags": {k: v for k, v in tags.items() if v},
                    "geom": geom})
    return out


def _not_covered(osm: list[dict], esri: list[dict]) -> list[dict]:
    if not osm or not esri:
        return osm
    tree = shapely.STRtree([e["geom"] for e in esri])
    hit = tree.query([o["geom"] for o in osm], predicate="intersects")
    covered = set(hit[0].tolist())
    return [o for i, o in enumerate(osm) if i not in covered]


def _copy_airports(cur, rows, source, load_id):
    with cur.copy("copy aviation_airports (source, src_id, name, tags, geom, load_id) from stdin") as cp:
        for r in rows:
            cp.write_row((source, r["src_id"], r["name"], db.dumps(r["tags"]), _wkb(r["geom"]), load_id))


def load_airports(osm_gpkg: Path | None = None, progress=None) -> dict:
    init_schema()
    src = "Esri India Living Atlas Airport_Boundary" + (f" + OSM aerodromes not in it ({Path(osm_gpkg).name})"
                                                        if osm_gpkg else "")
    now = datetime.now(timezone.utc)
    lid = db.one("""insert into aviation_loads (layer, kind, source, data_date) values ('airports', 'india', %s, %s)
                    returning id""", (src, now))["id"]
    try:
        esri = fetch_esri()
        osm = _not_covered(_osm_aerodromes(Path(osm_gpkg)), esri) if osm_gpkg else []
        counts = {"esri": len(esri), "osm_not_in_esri": len(osm)}
        with db.conn() as c:
            with c.transaction():
                c.execute("truncate aviation_airports")
                with c.cursor() as cur:
                    _copy_airports(cur, esri, "esri", lid)
                    _copy_airports(cur, osm, "osm", lid)
                c.execute("update aviation_loads set status='done', finished_at=now(), counts=%s where id=%s",
                          (db.J(counts), lid))
                c.execute("""update aviation_loads set status='superseded'
                             where layer='airports' and kind='aoi' and status='done' and id < %s""", (lid,))
        return {"load_id": lid, **counts}
    except Exception as e:
        db.exec_("update aviation_loads set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}", lid))
        raise


def refresh_airports(bbox: tuple[float, float, float, float], run_id: str | None = None, progress=None) -> dict:
    """Re-pull the airports intersecting bbox (minx, miny, maxx, maxy) from both sources — Esri
    Living Atlas and OSM — and replace them. Both are fetched before anything is written, so a
    failure on either leaves the store as it was."""
    init_schema()
    aoi = box(*bbox)
    lid = db.one("""insert into aviation_loads (layer, kind, source, aoi, run_id)
                    values ('airports', 'aoi', 'Esri India Living Atlas Airport_Boundary + OSM aeroway=aerodrome',
                            ST_GeomFromText(%s, 4326), %s) returning id""", (aoi.wkt, run_id))["id"]
    try:
        if progress:
            progress(0.1, "Querying Esri Living Atlas")
        esri = fetch_esri(bbox)
        if progress:
            progress(0.4, "Querying OpenStreetMap (Overpass)")
        osm, osm_ts = fetch_osm_aerodromes(bbox)
        if progress:
            progress(0.7, "Replacing airports in the store")
        with db.conn() as c:
            with c.transaction():
                before = {r["source"]: r["n"] for r in c.execute(
                    """select source, count(*) as n from aviation_airports
                       where ST_Intersects(geom, ST_GeomFromText(%s, 4326)) group by 1""", (aoi.wkt,)).fetchall()}
                c.execute("""delete from aviation_airports where
                             (source='esri' and (ST_Intersects(geom, ST_GeomFromText(%s, 4326)) or src_id = any(%s)))
                             or (source='osm' and (ST_Intersects(geom, ST_GeomFromText(%s, 4326)) or src_id = any(%s)))""",
                          (aoi.wkt, [e["src_id"] for e in esri], aoi.wkt, [o["src_id"] for o in osm]))
                with c.cursor() as cur:
                    _copy_airports(cur, esri, "esri", lid)
                    _copy_airports(cur, osm, "osm", lid)
                # OSM is kept only where no Esri polygon (refreshed or not) overlaps it
                dup = c.execute("""delete from aviation_airports o using aviation_airports e
                                   where o.source='osm' and o.load_id=%s and e.source='esri'
                                   and ST_Intersects(o.geom, e.geom)""", (lid,)).rowcount
                counts = {"esri": len(esri), "osm_fetched": len(osm), "osm_not_in_esri": len(osm) - int(dup),
                          "replaced_esri": int(before.get("esri", 0)), "replaced_osm": int(before.get("osm", 0))}
                c.execute("""update aviation_loads set status='done', finished_at=now(), counts=%s, data_date=%s
                             where id=%s""", (db.J(counts), osm_ts, lid))
        return {"load_id": lid, "data_date": osm_ts.isoformat(), **counts}
    except Exception as e:
        db.exec_("update aviation_loads set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}", lid))
        raise


# ---- runways ----------------------------------------------------------------------------
_W = re.compile(r"(\d+(?:\.\d+)?)")


def parse_width(v) -> float | None:
    m = _W.search(str(v)) if v not in (None, "") else None
    w = float(m.group(1)) if m else None
    return w if w is not None and 5 <= w <= 150 else None


def runway_polygons(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """OSM aeroway=runway features -> polygons with width_m / width_source; disused dropped."""
    tags = g["all_tags"].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
    dead = tags.map(lambda t: t.get("disused") == "yes" or t.get("abandoned") == "yes" or "disused:aeroway" in t
                    or "abandoned:aeroway" in t)
    g = g[~dead].copy()
    tags = tags[~dead]
    g["width_tag"] = tags.map(lambda t: parse_width(t.get("width"))).astype(object)   # None, not NaN
    g["tags"] = tags.map(lambda t: {k: t[k] for k in ("ref", "surface", "length", "width", "name", "icao", "iata")
                                    if k in t})
    out = []
    for zone, part in g.groupby(((g.geometry.centroid.x + 180) // 6 + 1).astype(int)):
        utm = part.to_crs(32600 + int(zone))
        for r, geom in zip(part.itertuples(index=False), utm.geometry):
            if geom is None or geom.is_empty:
                continue
            wt = r.width_tag if isinstance(r.width_tag, float) and r.width_tag == r.width_tag else None
            if geom.geom_type in ("Polygon", "MultiPolygon"):
                poly, w, ws = geom, wt, "polygon"
            else:
                w = wt or DEFAULT_RUNWAY_WIDTH_M
                ws = "width tag" if wt else f"default {DEFAULT_RUNWAY_WIDTH_M:g} m"
                poly = geom.buffer(w / 2.0, cap_style="flat")
            out.append({"src_id": f"{r.osm_type}/{r.osm_id}", "name": r.tags.get("ref") or _clean(r.name),
                        "width_m": w, "width_source": ws, "tags": r.tags,
                        "geometry": gpd.GeoSeries([poly], crs=utm.crs).to_crs(4326).iloc[0]})
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326) if out else gpd.GeoDataFrame(
        {"src_id": [], "name": [], "width_m": [], "width_source": [], "tags": []}, geometry=[], crs=4326)


def load_runways(gpkg: Path, progress=None) -> dict:
    init_schema()
    raw = gpd.read_file(gpkg, layer="aeroway_runway").to_crs(4326)
    rw = runway_polygons(raw)
    counts = {"read": len(raw), "kept": len(rw), "dropped_disused": len(raw) - len(rw),
              "from_polygon": int((rw["width_source"] == "polygon").sum()),
              "width_tag": int((rw["width_source"] == "width tag").sum()),
              "default_width": int(rw["width_source"].str.startswith("default").sum())}
    lid = db.one("""insert into aviation_loads (layer, kind, source, data_date) values ('runways', 'india', %s, %s)
                    returning id""", (f"OSM aeroway=runway ({Path(gpkg).name})", datetime.fromtimestamp(Path(gpkg).stat().st_mtime, timezone.utc)))["id"]
    with db.conn() as c:
        with c.transaction():
            c.execute("truncate aviation_runways")
            with c.cursor() as cur:
                _copy_runways(cur, rw, lid)
            c.execute("update aviation_loads set status='done', finished_at=now(), counts=%s where id=%s",
                      (db.J(counts), lid))
            c.execute("""update aviation_loads set status='superseded'
                         where layer='runways' and kind='aoi' and status='done' and id < %s""", (lid,))
    return {"load_id": lid, **counts}


def _copy_runways(cur, rw: gpd.GeoDataFrame, lid: int):
    with cur.copy("copy aviation_runways (src_id, name, width_m, width_source, tags, geom, load_id) from stdin") as cp:
        for r in rw.itertuples(index=False):
            g = _multi(r.geometry)
            if g is not None:
                cp.write_row((r.src_id, _clean(r.name), _clean(r.width_m), r.width_source, db.dumps(r.tags), _wkb(g), lid))


def refresh_runways(bbox: tuple[float, float, float, float], run_id: str | None = None, progress=None) -> dict:
    """Re-pull the OSM runways intersecting bbox (minx, miny, maxx, maxy) and replace them."""
    init_schema()
    aoi = box(*bbox)
    lid = db.one("""insert into aviation_loads (layer, kind, source, aoi, run_id)
                    values ('runways', 'aoi', 'OpenStreetMap Overpass aeroway=runway', ST_GeomFromText(%s, 4326), %s)
                    returning id""", (aoi.wkt, run_id))["id"]
    try:
        if progress:
            progress(0.1, "Querying OpenStreetMap (Overpass)")
        raw, ts = fetch_osm_runways(bbox)
        rw = runway_polygons(raw)
        if progress:
            progress(0.7, "Replacing runways in the store")
        with db.conn() as c:
            with c.transaction():
                before = c.execute("""select count(*) as n from aviation_runways
                                      where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (aoi.wkt,)).fetchone()["n"]
                c.execute("""delete from aviation_runways
                             where ST_Intersects(geom, ST_GeomFromText(%s, 4326)) or src_id = any(%s)""",
                          (aoi.wkt, list(rw["src_id"])))
                with c.cursor() as cur:
                    _copy_runways(cur, rw, lid)
                counts = {"read": len(raw), "kept": len(rw), "dropped_disused": len(raw) - len(rw), "replaced": int(before)}
                c.execute("""update aviation_loads set status='done', finished_at=now(), counts=%s, data_date=%s
                             where id=%s""", (db.J(counts), ts, lid))
        return {"load_id": lid, "data_date": ts.isoformat(), **counts}
    except Exception as e:
        db.exec_("update aviation_loads set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}", lid))
        raise


# ---- reading ----------------------------------------------------------------------------
def _reach(band_lonlat, reach_m: float):
    """The band grown by reach_m (metres), in lon/lat: features beyond the band that a rule reaches."""
    if reach_m <= 0:
        return band_lonlat
    s = gpd.GeoSeries([band_lonlat], crs=4326)
    utm = s.to_crs(s.estimate_utm_crs())
    return utm.buffer(reach_m).to_crs(4326).iloc[0]


def _loads(c, layer, wkt):
    return c.execute("""select kind, source, data_date, finished_at from aviation_loads
                        where layer=%s and status='done' and (kind='india' or ST_Intersects(aoi, ST_GeomFromText(%s, 4326)))
                        order by finished_at desc""", (layer, wkt)).fetchall()


def read_airports(band_lonlat, reach_m: float = 0.0) -> tuple[gpd.GeoDataFrame, dict]:
    """Airports within reach_m of the band, whole (not clipped: the NOC radius runs from the airport)."""
    init_schema()
    wkt = _reach(band_lonlat, reach_m).wkt
    with db.conn() as c:
        rows = c.execute("""select source, src_id, name, tags, ST_AsBinary(geom) as wkb from aviation_airports
                            where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (wkt,)).fetchall()
        loads = _loads(c, "airports", band_lonlat.wkt)
    gdf = gpd.GeoDataFrame({
        "class": ["airport"] * len(rows),
        "name": [r["name"] for r in rows],
        "confidence": [None] * len(rows),
        "src_id": [f"{r['source']}:{r['src_id']}" for r in rows],
        "attrs": [{"source": "Esri Living Atlas" if r["source"] == "esri" else "OpenStreetMap", **(r["tags"] or {})}
                  for r in rows],
    }, geometry=list(shapely.from_wkb([bytes(r["wkb"]) for r in rows])) if rows else [], crs=4326)
    india = next((l for l in loads if l["kind"] == "india"), None)
    aoi = next((l for l in loads if l["kind"] == "aoi"), None)
    src = "Esri India Living Atlas airport boundaries + OSM airports not in it"
    if aoi:
        src += f" · refreshed for this area {aoi['finished_at']:%d %b %Y %H:%M} UTC"
    elif india:
        src += f" · India load {india['finished_at']:%d %b %Y}"
    newest = max([l["data_date"] for l in loads if l["data_date"]], default=None)
    counts = {"esri": sum(r["source"] == "esri" for r in rows), "osm": sum(r["source"] == "osm" for r in rows)}
    return gdf, {"source": src, "data_date": newest.isoformat() if newest else None, "india_loaded": india is not None,
                 "last_refresh": aoi["finished_at"].isoformat() if aoi else None, "counts": counts}


def read_runways(band_lonlat, reach_m: float = 0.0) -> tuple[gpd.GeoDataFrame, dict]:
    """Runway polygons within reach_m of the band, whole."""
    init_schema()
    wkt = _reach(band_lonlat, reach_m).wkt
    with db.conn() as c:
        rows = c.execute("""select src_id, name, width_m, width_source, tags, ST_AsBinary(geom) as wkb from aviation_runways
                            where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (wkt,)).fetchall()
        loads = _loads(c, "runways", band_lonlat.wkt)
    gdf = gpd.GeoDataFrame({
        "class": ["runway"] * len(rows),
        "name": [f"Runway {r['name']}" if r["name"] else "Runway" for r in rows],
        "confidence": [None] * len(rows),
        "src_id": [f"osm:{r['src_id']}" for r in rows],
        "attrs": [{**(r["tags"] or {}), "width_m": r["width_m"], "width_source": r["width_source"]} for r in rows],
    }, geometry=list(shapely.from_wkb([bytes(r["wkb"]) for r in rows])) if rows else [], crs=4326)
    india = next((l for l in loads if l["kind"] == "india"), None)
    aoi = next((l for l in loads if l["kind"] == "aoi"), None)
    src = "OpenStreetMap aeroway=runway, centrelines widened to runway width"
    if aoi:
        src += f" · refreshed for this area {aoi['finished_at']:%d %b %Y %H:%M} UTC"
    elif india:
        src += f" · {india['source']}"
    newest = max([l["data_date"] for l in loads if l["data_date"]], default=None)
    counts: dict[str, int] = {}
    for r in rows:
        k = "default_width" if (r["width_source"] or "").startswith("default") else (r["width_source"] or "unknown").replace(" ", "_")
        counts[k] = counts.get(k, 0) + 1
    return gdf, {"source": src, "data_date": newest.isoformat() if newest else None,
                 "india_loaded": india is not None, "last_refresh": aoi["finished_at"].isoformat() if aoi else None,
                 "counts": counts}


def status() -> dict:
    init_schema()
    return {
        "airports": db.all_("select source, count(*) as n from aviation_airports group by 1 order by 1"),
        "runways": db.all_("select width_source, count(*) as n from aviation_runways group by 1 order by 1"),
        "loads": db.all_("""select id, layer, kind, source, data_date, run_id, status, started_at, finished_at, counts, error,
                                   ST_AsGeoJSON(aoi)::jsonb as aoi from aviation_loads order by id desc limit 20"""),
    }


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    la = sub.add_parser("load-airports")
    la.add_argument("--osm-gpkg", help="airports.gpkg (aeroway_aerodrome layer) to add OSM airports missing from Esri")
    lr = sub.add_parser("load-runways")
    lr.add_argument("--gpkg", required=True, help="airports.gpkg (aeroway_runway layer)")
    for name in ("refresh-airports", "refresh-runways"):
        sub.add_parser(name).add_argument("bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    sub.add_parser("status")
    a = ap.parse_args()
    db.init_schema()
    if a.cmd == "load-airports":
        print(load_airports(Path(a.osm_gpkg) if a.osm_gpkg else None))
    elif a.cmd == "load-runways":
        print(load_runways(Path(a.gpkg)))
    elif a.cmd == "refresh-airports":
        print(refresh_airports(tuple(a.bbox)))
    elif a.cmd == "refresh-runways":
        print(refresh_runways(tuple(a.bbox)))
    else:
        print(json.dumps(status(), indent=1, default=str))


if __name__ == "__main__":
    main()
