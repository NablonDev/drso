"""OpenStreetMap national and state highways for India, stored by class (nh / sh).

This is the source of the `roads` layer ("Highways — NH, SH"; Vivek, 5 Oct 2026). It replaced
the cached Geosadak + OSM gpkg, so MDR is no longer carried (no rule uses it).

A way is classed from, in order:
  ref        "NH341", "NH 48", "NE4" -> nh;  "SH42", "SH__" -> sh  ("NH341;SH45" is both)
  relations  route=road with network IN:NH / IN:NE -> nh, IN:SH:<state> -> sh
  inference  highway=motorway / trunk with neither -> nh, marked inferred (OSM India tags NHs
             as trunk; a missed NH breaks a crossing rule, a spurious one costs a tension tower)
A way in both classes is stored once per class. Link roads (*_link) are not kept.

Store: PostGIS table `osm_highways`, list-partitioned by road_class (osm_highways_nh / _sh).
Every load is logged in `osm_highway_loads` with its OSM data timestamp.

  India load   python -m beeline_api.osm_highways load-india [--pbf PATH]   (Geofabrik extract,
               shared with osm_power)
  AOI refresh  refresh_aoi(bbox) — Overpass; replaces the highways of that bbox in one
               transaction. Triggered from the UI ("Refresh from OSM") as an extraction job.
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import shapely
from shapely.geometry import LineString, box

from . import db, osm_power

log = logging.getLogger(__name__)
CLASSES = ("nh", "sh")
HIGHWAYS = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified")
INFER_NH = ("motorway", "trunk")
KEEP_TAGS = ("name", "ref", "highway", "lanes", "surface", "maxspeed", "oneway", "bridge", "tunnel", "layer",
             "operator", "toll", "int_ref", "old_ref")

SCHEMA = """
create table if not exists osm_highways (
  id bigserial,
  road_class text not null,
  osm_id bigint not null,
  ref text,
  highway text,
  inferred boolean not null default false,
  tags jsonb not null default '{}'::jsonb,
  geom geometry(LineString, 4326) not null,
  load_id bigint,
  primary key (road_class, id)
) partition by list (road_class);
create table if not exists osm_highways_nh partition of osm_highways for values in ('nh');
create table if not exists osm_highways_sh partition of osm_highways for values in ('sh');
create index if not exists osm_highways_geom on osm_highways using gist (geom);
create index if not exists osm_highways_osm on osm_highways (osm_id);

create table if not exists osm_highway_loads (
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
            c.execute("select pg_advisory_xact_lock(724004)")
            c.execute(SCHEMA)


# ---- classification -----------------------------------------------------------------------
_REF = re.compile(r"^\s*(NH|NE|SH)\s*[-.]?\s*([0-9A-Z_]*)\s*$", re.I)


def parse_refs(ref: str | None) -> dict[str, list[str]]:
    """'NH341;SH45' -> {'nh': ['NH 341'], 'sh': ['SH 45']}; 'NE4' -> {'nh': ['NE 4']}; others dropped."""
    out: dict[str, list[str]] = {}
    for part in re.split(r"[;,/]", ref or ""):
        m = _REF.match(part)
        if not m:
            continue
        pre, num = m.group(1).upper(), m.group(2).upper().strip("_")
        r = f"{pre} {num}" if num else pre
        cls = "sh" if pre == "SH" else "nh"
        if r not in out.setdefault(cls, []):
            out[cls].append(r)
    return out


def relation_class(tags: dict) -> tuple[str, str] | None:
    """route=road relation -> (class, ref) from its network, e.g. IN:NH + '341' -> ('nh', 'NH 341')."""
    if tags.get("route") != "road":
        return None
    net = (tags.get("network") or "").upper()
    if net in ("IN:NH", "IN:NE"):
        cls, pre = "nh", net[-2:]
    elif net.startswith("IN:SH"):
        cls, pre = "sh", "SH"
    else:
        return None
    ref = (tags.get("ref") or "").strip()
    got = parse_refs(ref).get(cls)
    if got:
        return cls, got[0]
    num = ref.upper().strip("_ ")
    return cls, f"{pre} {num}" if num else pre


def classify(tags: dict, rels: list[tuple[str, str]] = ()) -> list[tuple[str, str | None, bool]]:
    """Way tags + its road relations -> [(class, ref, inferred)]; [] when not an NH / SH."""
    hw = tags.get("highway")
    if hw not in HIGHWAYS:
        return []
    refs = parse_refs(tags.get("ref"))
    for cls, r in rels:
        if r not in refs.setdefault(cls, []):
            refs[cls].append(r)
    for cls, rs in refs.items():                 # a numbered ref makes the bare "NH" / "SH" redundant
        if any(" " in r for r in rs):
            refs[cls] = [r for r in rs if " " in r]
    out = [(cls, "; ".join(refs[cls]), False) for cls in CLASSES if refs.get(cls)]
    if not out and hw in INFER_NH:
        out = [("nh", None, True)]
    return out


def _keep(tags: dict) -> dict:
    return {k: tags[k] for k in KEEP_TAGS if k in tags}


def _rows_from(ways: list, way_rels: dict) -> tuple[list, dict]:
    """ways: [(osm_id, tags, coords)] -> store rows and counts by class."""
    rows, counts = [], {}
    for oid, tags, coords in ways:
        if len(coords) < 2:
            continue
        hits = classify(tags, way_rels.get(oid, []))
        if not hits:
            continue
        g = LineString(coords)
        for cls, ref, inferred in hits:
            rows.append((cls, oid, ref, tags.get("highway"), inferred, _keep(tags), g))
            k = f"{cls}_inferred" if inferred else cls
            counts[k] = counts.get(k, 0) + 1
    return rows, counts


def _copy_rows(cur, rows, load_id):
    with cur.copy("copy osm_highways (road_class, osm_id, ref, highway, inferred, tags, geom, load_id) from stdin") as cp:
        for cls, oid, ref, hw, inf, tags, geom in rows:
            cp.write_row((cls, oid, ref, hw, inf, db.dumps(tags),
                          shapely.to_wkb(shapely.set_srid(geom, 4326), hex=True, include_srid=True), load_id))


# ---- India load (Geofabrik) -------------------------------------------------------------
def load_india(pbf: Path | None = None, progress=None) -> dict:
    import osmium
    init_schema()
    pbf = osm_power.ensure_india_pbf(pbf)
    ts = osm_power.india_timestamp(pbf)
    filtered = osm_power.OSM_DIR / "india-highways.osm.pbf"
    subprocess.run(["osmium", "tags-filter", str(pbf), f"w/highway={','.join(HIGHWAYS)}", "r/route=road",
                    "-o", str(filtered), "--overwrite"], check=True)
    load = db.one("insert into osm_highway_loads (kind, source, osm_timestamp) values ('india', %s, %s) returning id",
                  (f"Geofabrik India extract ({pbf.name})", ts))
    lid = load["id"]

    way_rels: dict[int, list] = {}
    ways: list = []

    class Rels(osmium.SimpleHandler):           # pass 1: relations come last in the file
        def relation(self, r):
            rc = relation_class(dict(r.tags))
            if rc:
                for m in r.members:
                    if m.type == "w":
                        way_rels.setdefault(m.ref, []).append(rc)

    class Ways(osmium.SimpleHandler):
        def way(self, w):
            tags = dict(w.tags)
            if tags.get("highway") not in HIGHWAYS:
                return
            try:
                coords = [(nd.lon, nd.lat) for nd in w.nodes]
            except osmium.InvalidLocationError:
                coords = [(nd.lon, nd.lat) for nd in w.nodes if nd.location.valid()]
            ways.append((w.id, tags, coords))

    Rels().apply_file(str(filtered))
    Ways().apply_file(str(filtered), locations=True)
    rows, counts = _rows_from(ways, way_rels)
    counts["ways_read"] = len(ways)
    counts["trunk_motorway_ways"] = sum(1 for w in ways if w[1].get("highway") in INFER_NH)
    with db.conn() as c:
        with c.transaction():
            for cls in CLASSES:
                c.execute(f"truncate osm_highways_{cls}")
            with c.cursor() as cur:
                _copy_rows(cur, rows, lid)
            c.execute("update osm_highway_loads set status='done', finished_at=now(), counts=%s where id=%s",
                      (db.J(counts), lid))
            # the India load replaced every area, so earlier AOI refreshes no longer describe the data
            c.execute("update osm_highway_loads set status='superseded' where kind='aoi' and status='done' and id < %s",
                      (lid,))
    return {"load_id": lid, "osm_timestamp": ts.isoformat() if ts else None, **counts}


# ---- AOI refresh (Overpass) -------------------------------------------------------------
def _overpass(query: str, timeout: int = 300) -> dict:
    return osm_power.overpass_query(query, timeout)


def overpass_ql(bbox) -> str:
    minx, miny, maxx, maxy = bbox
    bb = f"{miny},{minx},{maxy},{maxx}"
    hw = "|".join(HIGHWAYS)
    # untagged primary..unclassified ways can only be NH/SH through a ref or a relation, so they
    # are not fetched without one; the relation's members are taken inside the bbox only
    return (f'[out:json][timeout:300];'
            f'(way["highway"~"^({"|".join(INFER_NH)})$"]({bb});'
            f'way["highway"~"^({hw})$"]["ref"]({bb});)->.a;'
            f'rel["route"="road"]["network"~"^IN:(NH|NE|SH)"]({bb})->.r;'
            f'way(r.r)({bb})["highway"~"^({hw})$"]->.b;'
            f'(.a;.b;)->.w;.w out geom;.r out body;')


def refresh_aoi(bbox: tuple[float, float, float, float], run_id: str | None = None, progress=None) -> dict:
    """Re-pull NH / SH ways intersecting bbox (minx, miny, maxx, maxy) and replace them."""
    init_schema()
    minx, miny, maxx, maxy = bbox
    aoi = box(minx, miny, maxx, maxy)
    load = db.one("""insert into osm_highway_loads (kind, source, aoi, run_id)
                     values ('aoi', 'OpenStreetMap Overpass', ST_GeomFromText(%s, 4326), %s) returning id""",
                  (aoi.wkt, run_id))
    lid = load["id"]
    try:
        if progress:
            progress(0.1, "Querying OpenStreetMap (Overpass)")
        data = _overpass(overpass_ql(bbox))
        ts = data.get("osm3s", {}).get("timestamp_osm_base")
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else datetime.now(timezone.utc)
        way_rels: dict[int, list] = {}
        ways = []
        for el in data.get("elements", []):
            if el["type"] == "relation":
                rc = relation_class(el.get("tags", {}))
                if rc:
                    for m in el.get("members", []):
                        if m["type"] == "way":
                            way_rels.setdefault(m["ref"], []).append(rc)
        for el in data.get("elements", []):
            if el["type"] == "way":
                coords = [(p["lon"], p["lat"]) for p in el.get("geometry", []) if p]
                ways.append((el["id"], el.get("tags", {}), coords))
        rows, counts = _rows_from(ways, way_rels)
        ids = list({r[1] for r in rows})
        if progress:
            progress(0.7, "Replacing highways in the store")
        with db.conn() as c:
            with c.transaction():
                before = c.execute("""select count(*) as n from osm_highways
                                      where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (aoi.wkt,)).fetchone()["n"]
                c.execute("""delete from osm_highways where ST_Intersects(geom, ST_GeomFromText(%s, 4326))
                             or osm_id = any(%s)""", (aoi.wkt, ids))
                with c.cursor() as cur:
                    _copy_rows(cur, rows, lid)
                counts["replaced"] = int(before)
                c.execute("""update osm_highway_loads set status='done', finished_at=now(), counts=%s, osm_timestamp=%s
                             where id=%s""", (db.J(counts), ts, lid))
        return {"load_id": lid, "osm_timestamp": ts.isoformat(), **counts}
    except Exception as e:
        db.exec_("update osm_highway_loads set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}", lid))
        raise


# ---- reading ----------------------------------------------------------------------------
def read_band(band_lonlat) -> tuple[gpd.GeoDataFrame, dict]:
    """NH / SH ways intersecting the band, clipped to it, as layer features (class nh / sh)."""
    init_schema()
    wkt = band_lonlat.wkt
    with db.conn() as c:
        rows = c.execute("""select road_class, osm_id, ref, highway, inferred, tags, ST_AsBinary(geom) as wkb
                            from osm_highways where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (wkt,)).fetchall()
        loads = c.execute("""select id, kind, source, osm_timestamp, finished_at from osm_highway_loads
                             where status='done' and (kind='india' or ST_Intersects(aoi, ST_GeomFromText(%s, 4326)))
                             order by finished_at desc""", (wkt,)).fetchall()
    geoms = shapely.from_wkb([bytes(r["wkb"]) for r in rows]) if rows else []
    gdf = gpd.GeoDataFrame({
        "class": [r["road_class"] for r in rows],
        "name": [r["ref"] or (r["tags"] or {}).get("name") or ("National highway (inferred)" if r["inferred"] else None)
                 for r in rows],
        "confidence": [None] * len(rows),
        "src_id": [f"osm:way/{r['osm_id']}" for r in rows],
        "attrs": [{**(r["tags"] or {}), "road_class": r["road_class"], "ref": r["ref"], "highway": r["highway"],
                   **({"inferred": "highway=" + r["highway"] + ", no NH/SH ref"} if r["inferred"] else {})}
                  for r in rows],
    }, geometry=list(geoms), crs=4326)
    if len(gdf):
        gdf["geometry"] = gdf.geometry.intersection(band_lonlat)
        gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    india = next((l for l in loads if l["kind"] == "india"), None)
    aoi = next((l for l in loads if l["kind"] == "aoi"), None)
    newest = max([l["osm_timestamp"] for l in loads if l["osm_timestamp"]], default=None)
    src = "OpenStreetMap national and state highways (ref, IN:NH / IN:SH relations; untagged trunk = NH)"
    if india:
        src += f" · India extract {india['osm_timestamp']:%d %b %Y}" if india["osm_timestamp"] else " · India extract"
    if aoi:
        src += f" · refreshed for this area {aoi['finished_at']:%d %b %Y %H:%M} UTC"
    counts: dict[str, int] = {}
    for cls, inf in zip(gdf["class"], gdf["attrs"]):
        k = f"{cls}_inferred" if "inferred" in inf else cls
        counts[k] = counts.get(k, 0) + 1
    return gdf, {"source": src, "osm_timestamp": newest.isoformat() if newest else None,
                 "india_loaded": india is not None, "last_refresh": aoi["finished_at"].isoformat() if aoi else None,
                 "counts": counts}


def status() -> dict:
    init_schema()
    counts = db.all_("""select road_class, inferred, count(*) as n from osm_highways group by 1, 2 order by 1, 2""")
    loads = db.all_("""select id, kind, source, osm_timestamp, run_id, status, started_at, finished_at, counts, error,
                              ST_AsGeoJSON(aoi)::jsonb as aoi from osm_highway_loads order by id desc limit 20""")
    return {"counts": counts, "loads": loads, "classes": list(CLASSES)}


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
