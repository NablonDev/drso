"""OpenStreetMap transmission lines and towers for India, stored by voltage.

Only 765, 400, 220 and 132 kV are kept (Vivek, 5 Oct 2026). Objects: power=line ways and
power=tower nodes. A tower carries the voltage(s) of the line(s) it supports — OSM rarely tags
voltage on towers. A line or tower with several voltages (e.g. "400000;220000") is stored once
per voltage.

Store: PostGIS table `osm_power`, list-partitioned by voltage_kv (osm_power_765 / _400 / _220 /
_132). Every load is logged in `osm_power_loads` with its OSM data timestamp.

  India load   python -m beeline_api.osm_power load-india [--pbf PATH]   (Geofabrik extract)
  AOI refresh  refresh_aoi(bbox)  — Overpass; replaces the lines and towers of that bbox in one
               transaction. Triggered from the UI ("Refresh from OSM") as an extraction job.
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import httpx
import shapely
from shapely.geometry import LineString, Point, box

from . import db, settings

log = logging.getLogger(__name__)
VOLTAGES = (765, 400, 220, 132)
KEEP_TAGS = ("name", "ref", "operator", "owner", "circuits", "cables", "voltage", "frequency", "line", "location",
             "design", "material", "structure", "start_date", "wires")
GEOFABRIK_URL = "https://download.geofabrik.de/asia/india-latest.osm.pbf"
GEOFABRIK_STATE = "https://download.geofabrik.de/asia/india-updates/state.txt"
OVERPASS_URLS = settings.OVERPASS_URLS
QUICK_RETRIES = 3          # same-mirror retries on a busy answer (429/502/503/504)
OSM_DIR = settings.DATA / "osm"

SCHEMA = """
create table if not exists osm_power (
  id bigserial,
  voltage_kv int not null,
  kind text not null check (kind in ('line', 'tower')),
  osm_type text not null,
  osm_id bigint not null,
  tags jsonb not null default '{}'::jsonb,
  geom geometry(Geometry, 4326) not null,
  load_id bigint,
  primary key (voltage_kv, id)
) partition by list (voltage_kv);
create table if not exists osm_power_765 partition of osm_power for values in (765);
create table if not exists osm_power_400 partition of osm_power for values in (400);
create table if not exists osm_power_220 partition of osm_power for values in (220);
create table if not exists osm_power_132 partition of osm_power for values in (132);
create index if not exists osm_power_geom on osm_power using gist (geom);
create index if not exists osm_power_osm on osm_power (osm_type, osm_id);

create table if not exists osm_power_loads (
  id bigserial primary key,
  kind text not null check (kind in ('india', 'aoi')),
  source text not null,
  osm_timestamp timestamptz,
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
            c.execute("select pg_advisory_xact_lock(724002)")
            c.execute(SCHEMA)


# ---- tag parsing ----------------------------------------------------------------------
_NUM = re.compile(r"(\d+(?:\.\d+)?)\s*(k?v)?", re.I)


def parse_voltages(tag: str | None) -> list[int]:
    """'400000;220000' -> [400, 220]; '400 kV' -> [400]; anything outside the four levels dropped."""
    if not tag:
        return []
    out = []
    for part in re.split(r"[;,/]", tag):
        m = _NUM.search(part.strip())
        if not m:
            continue
        v = float(m.group(1))
        unit = (m.group(2) or "").lower()
        kv = v if unit == "kv" or (not unit and v < 1000) else v / 1000.0
        k = int(round(kv))
        if k in VOLTAGES and abs(kv - k) < 0.5 and k not in out:
            out.append(k)
    return out


def _is_line(tags: dict) -> bool:
    return tags.get("power") == "line" and tags.get("line") not in ("busbar", "bay")


def _keep(tags: dict) -> dict:
    return {k: tags[k] for k in KEEP_TAGS if k in tags}


# ---- writing ----------------------------------------------------------------------------
def _copy_rows(cur, rows, load_id):
    with cur.copy("copy osm_power (voltage_kv, kind, osm_type, osm_id, tags, geom, load_id) from stdin") as cp:
        for kv, kind, otype, oid, tags, geom in rows:
            cp.write_row((kv, kind, otype, oid, db.dumps(tags),
                          shapely.to_wkb(shapely.set_srid(geom, 4326), hex=True, include_srid=True), load_id))


def _rows_from(lines: list, towers: dict) -> tuple[list, dict]:
    """lines: [(osm_id, tags, coords, node_ids)], towers: {node_id: (lon, lat, tags)}."""
    rows, counts = [], {}
    tower_kv: dict[int, set] = {}
    for oid, tags, coords, nids in lines:
        kvs = parse_voltages(tags.get("voltage"))
        if not kvs or len(coords) < 2:
            continue
        g = LineString(coords)
        for kv in kvs:
            rows.append((kv, "line", "way", oid, _keep(tags), g))
            counts[f"line_{kv}"] = counts.get(f"line_{kv}", 0) + 1
        for n in nids:
            if n in towers:
                tower_kv.setdefault(n, set()).update(kvs)
    for nid, kvs in tower_kv.items():
        lon, lat, tags = towers[nid]
        p = Point(lon, lat)
        for kv in sorted(kvs, reverse=True):
            rows.append((kv, "tower", "node", nid, _keep(tags), p))
            counts[f"tower_{kv}"] = counts.get(f"tower_{kv}", 0) + 1
    return rows, counts


# ---- India load (Geofabrik) -------------------------------------------------------------
def _osm_timestamp_from_state() -> datetime | None:
    try:
        r = httpx.get(GEOFABRIK_STATE, timeout=20, headers={"User-Agent": settings.HTTP_UA})
        m = re.search(r"timestamp=(\S+)", r.text)
        return datetime.fromisoformat(m.group(1).replace("\\:", ":").replace("Z", "+00:00")) if m else None
    except Exception:
        return None


def _pbf_timestamp(pbf: Path) -> datetime | None:
    try:
        out = subprocess.run(["osmium", "fileinfo", "-g", "header.option.osmosis_replication_timestamp", str(pbf)],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return datetime.fromisoformat(out.replace("Z", "+00:00")) if out else None
    except Exception:
        return None


def ensure_india_pbf(pbf: Path | None = None) -> Path:
    """The Geofabrik India extract in the bucket (downloaded once, shared by every OSM store)."""
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    pbf = Path(pbf) if pbf else OSM_DIR / "india-latest.osm.pbf"
    if not pbf.exists():
        log.info("downloading %s", GEOFABRIK_URL)
        tmp = pbf.with_suffix(".part")
        with httpx.stream("GET", GEOFABRIK_URL, timeout=600, follow_redirects=True,
                          headers={"User-Agent": settings.HTTP_UA}) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 22):
                    f.write(chunk)
        tmp.rename(pbf)
    return pbf


def india_timestamp(pbf: Path) -> datetime | None:
    return _pbf_timestamp(pbf) or _osm_timestamp_from_state()


def load_india(pbf: Path | None = None, progress=None) -> dict:
    import osmium
    init_schema()
    pbf = ensure_india_pbf(pbf)
    ts = india_timestamp(pbf)
    filtered = OSM_DIR / "india-power.osm.pbf"
    subprocess.run(["osmium", "tags-filter", str(pbf), "w/power=line", "n/power=tower", "-o", str(filtered),
                    "--overwrite"], check=True)
    load = db.one("insert into osm_power_loads (kind, source, osm_timestamp) values ('india', %s, %s) returning id",
                  (f"Geofabrik India extract ({pbf.name})", ts))
    lid = load["id"]

    towers: dict[int, tuple] = {}
    lines: list = []

    class H(osmium.SimpleHandler):
        def node(self, n):
            if n.tags.get("power") == "tower":
                towers[n.id] = (n.location.lon, n.location.lat, dict(n.tags))

        def way(self, w):
            tags = dict(w.tags)
            if not _is_line(tags):
                return
            try:
                coords = [(nd.lon, nd.lat) for nd in w.nodes]
            except osmium.InvalidLocationError:
                coords = [(nd.lon, nd.lat) for nd in w.nodes if nd.location.valid()]
            lines.append((w.id, tags, coords, [nd.ref for nd in w.nodes]))

    H().apply_file(str(filtered), locations=True)
    rows, counts = _rows_from(lines, towers)
    counts["lines_no_voltage_tag"] = sum(1 for l in lines if not l[1].get("voltage"))
    counts["lines_other_voltage"] = sum(1 for l in lines if l[1].get("voltage") and not parse_voltages(l[1]["voltage"]))
    counts["lines_read"] = len(lines)
    counts["towers_read"] = len(towers)
    with db.conn() as c:
        with c.transaction():
            for kv in VOLTAGES:
                c.execute(f"truncate osm_power_{kv}")
            with c.cursor() as cur:
                _copy_rows(cur, rows, lid)
            c.execute("""update osm_power_loads set status='done', finished_at=now(), counts=%s where id=%s""",
                      (db.J(counts), lid))
            # the India load replaced every area, so earlier AOI refreshes no longer describe the data
            c.execute("""update osm_power_loads set status='superseded' where kind='aoi' and status='done' and id < %s""",
                      (lid,))
    return {"load_id": lid, "osm_timestamp": ts.isoformat() if ts else None, **counts}


# ---- AOI refresh (Overpass) -------------------------------------------------------------
def _overpass(query: str, timeout: int = 300) -> dict:
    return overpass_query(query, timeout)


def overpass_query(query: str, timeout: int = 300) -> dict:
    """POST an Overpass QL query, trying each mirror, three rounds with back-off.

    overpass-api.de balances over several backends and one can answer 504 within seconds while
    the next call succeeds (~1 in 6, 5 Oct 2026), so a busy answer is retried on the same mirror
    a few times before moving to a slower one.
    """
    last = None
    for attempt in range(3):
        for url in OVERPASS_URLS:
            for quick in range(QUICK_RETRIES + 1):
                try:
                    r = httpx.post(url, data={"data": query}, timeout=timeout + 60,
                                   headers={"User-Agent": settings.HTTP_UA})
                    if r.status_code == 200:
                        return r.json()
                    last = f"{url} -> HTTP {r.status_code}"
                    if r.status_code not in (429, 502, 503, 504):
                        break
                except (httpx.HTTPError, ValueError) as e:
                    last = f"{url} -> {type(e).__name__}: {e}"
                    break
                time.sleep(2 * (quick + 1))
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"Overpass unavailable ({last})")


def refresh_aoi(bbox: tuple[float, float, float, float], run_id: str | None = None, progress=None) -> dict:
    """Re-pull lines and towers intersecting bbox (minx, miny, maxx, maxy) and replace them."""
    init_schema()
    minx, miny, maxx, maxy = bbox
    aoi = box(minx, miny, maxx, maxy)
    load = db.one("""insert into osm_power_loads (kind, source, aoi, run_id)
                     values ('aoi', 'OpenStreetMap Overpass', ST_GeomFromText(%s, 4326), %s) returning id""",
                  (aoi.wkt, run_id))
    lid = load["id"]
    try:
        if progress:
            progress(0.1, "Querying OpenStreetMap (Overpass)")
        q = (f'[out:json][timeout:300];(way["power"="line"]({miny},{minx},{maxy},{maxx}););'
             f'(._;>;);out body;')
        data = _overpass(q)
        ts = data.get("osm3s", {}).get("timestamp_osm_base")
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)
        nodes, towers, lines = {}, {}, []
        for el in data.get("elements", []):
            if el["type"] == "node":
                nodes[el["id"]] = (el["lon"], el["lat"])
                if el.get("tags", {}).get("power") == "tower":
                    towers[el["id"]] = (el["lon"], el["lat"], el["tags"])
        for el in data.get("elements", []):
            if el["type"] == "way" and _is_line(el.get("tags", {})):
                coords = [nodes[n] for n in el["nodes"] if n in nodes]
                lines.append((el["id"], el["tags"], coords, el["nodes"]))
        rows, counts = _rows_from(lines, towers)
        # towers outside the bbox stay as they are (not deleted, so not re-inserted)
        rows = [r for r in rows if r[1] == "line" or aoi.contains(r[5])]
        counts = {}
        for kv, kind, *_ in rows:
            counts[f"{kind}_{kv}"] = counts.get(f"{kind}_{kv}", 0) + 1
        line_ids = [r[3] for r in rows if r[1] == "line"]
        if progress:
            progress(0.7, "Replacing lines and towers in the store")
        with db.conn() as c:
            with c.transaction():
                before = c.execute("""select count(*) as n from osm_power
                                      where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (aoi.wkt,)).fetchone()["n"]
                c.execute("""delete from osm_power where ST_Intersects(geom, ST_GeomFromText(%s, 4326))
                             or (kind='line' and osm_id = any(%s))""", (aoi.wkt, line_ids))
                with c.cursor() as cur:
                    _copy_rows(cur, rows, lid)
                counts["replaced"] = int(before)
                c.execute("""update osm_power_loads set status='done', finished_at=now(), counts=%s, osm_timestamp=%s
                             where id=%s""", (db.J(counts), ts, lid))
        return {"load_id": lid, "osm_timestamp": ts.isoformat(), **counts}
    except Exception as e:
        db.exec_("update osm_power_loads set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}", lid))
        raise


# ---- reading ----------------------------------------------------------------------------
def read_band(band_lonlat) -> tuple[gpd.GeoDataFrame, dict]:
    """Lines and towers intersecting the band, as layer features (class power_line / power_tower)."""
    init_schema()
    wkt = band_lonlat.wkt
    with db.conn() as c:
        rows = c.execute("""select voltage_kv, kind, osm_type, osm_id, tags, ST_AsBinary(geom) as wkb
                            from osm_power where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (wkt,)).fetchall()
        loads = c.execute("""select id, kind, source, osm_timestamp, finished_at from osm_power_loads
                             where status='done' and (kind='india' or ST_Intersects(aoi, ST_GeomFromText(%s, 4326)))
                             order by finished_at desc""", (wkt,)).fetchall()
    geoms = shapely.from_wkb([bytes(r["wkb"]) for r in rows]) if rows else []
    gdf = gpd.GeoDataFrame({
        "class": ["power_line" if r["kind"] == "line" else "power_tower" for r in rows],
        "name": [(r["tags"] or {}).get("name") or (f"{r['voltage_kv']} kV line" if r["kind"] == "line" else None) for r in rows],
        "confidence": [None] * len(rows),
        "src_id": [f"osm:{r['osm_type']}/{r['osm_id']}" for r in rows],
        "attrs": [{"voltage_kv": r["voltage_kv"], **(r["tags"] or {})} for r in rows],
    }, geometry=list(geoms), crs=4326)
    if len(gdf):
        is_line = gdf["class"] == "power_line"
        gdf.loc[is_line, "geometry"] = gdf.loc[is_line].geometry.intersection(band_lonlat)
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    india = next((l for l in loads if l["kind"] == "india"), None)
    aoi = next((l for l in loads if l["kind"] == "aoi"), None)
    newest = max([l["osm_timestamp"] for l in loads if l["osm_timestamp"]], default=None)
    src = "OpenStreetMap power=line / power=tower, 765/400/220/132 kV"
    if india:
        src += f" · India extract {india['osm_timestamp']:%d %b %Y}" if india["osm_timestamp"] else " · India extract"
    if aoi:
        src += f" · refreshed for this area {aoi['finished_at']:%d %b %Y %H:%M} UTC"
    counts = {}
    for r in rows:
        counts[f"{r['kind']}_{r['voltage_kv']}"] = counts.get(f"{r['kind']}_{r['voltage_kv']}", 0) + 1
    return gdf, {"source": src, "osm_timestamp": newest.isoformat() if newest else None,
                 "india_loaded": india is not None, "last_refresh": aoi["finished_at"].isoformat() if aoi else None,
                 "counts": counts}


def status() -> dict:
    init_schema()
    counts = db.all_("""select voltage_kv, kind, count(*) as n from osm_power group by 1, 2 order by 1 desc, 2""")
    loads = db.all_("""select id, kind, source, osm_timestamp, run_id, status, started_at, finished_at, counts, error,
                              ST_AsGeoJSON(aoi)::jsonb as aoi from osm_power_loads order by id desc limit 20""")
    return {"counts": counts, "loads": loads, "voltages": list(VOLTAGES)}


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("load-india")
    li.add_argument("--pbf")
    ra = sub.add_parser("refresh")
    ra.add_argument("bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    sub.add_parser("status")
    a = ap.parse_args()
    db.init_schema()
    if a.cmd == "load-india":
        print(load_india(Path(a.pbf) if a.pbf else None))
    elif a.cmd == "refresh":
        print(refresh_aoi(tuple(a.bbox)))
    else:
        import json
        print(json.dumps(status(), indent=1, default=str))


if __name__ == "__main__":
    main()
