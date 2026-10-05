"""Mobile (cell) towers for India from DoT's Tarang Sanchar portal.

One row per tower site: Tarang Sanchar site_id, structure type (Ground / Roof / Wall), point.
No split by type (Vivek, 5 Oct 2026) — the type is kept as an attribute for the popup.

Store: PostGIS table `cell_towers`. Every load is logged in `cell_tower_loads` with the date of
the data it carries.

  India load   python -m beeline_api.cell_towers load-india [--parquet PATH]
               The 2022 Tarang Sanchar scrape mirrored by ramSeraph/indian_communications
               (665,077 sites). The portal itself offers no bulk download.
  AOI refresh  refresh_aoi(bbox) — replaces the towers of that bbox in one transaction.
               Triggered from the UI ("Refresh from Tarang Sanchar") as an extraction job.
               PENDING: the portal's tower search sits behind name + email + phone + CAPTCHA +
               an emailed OTP, and answers Indian IPs only; the NIC mirror needs a token. Until a
               legitimate machine source exists (PM GatiShakti Public, an NIC token), FETCHER is
               None and the refresh endpoint refuses with REFRESH_PENDING. Plug a fetcher in by
               setting FETCHER = fn(bbox) -> (rows, data_date) with rows of
               {"site_id", "structure", "lon", "lat", **extra}; nothing else changes.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import geopandas as gpd
import httpx
import shapely
from shapely.geometry import Point, box

from . import db, settings

log = logging.getLogger(__name__)
STRUCTURES = ("Ground", "Roof", "Wall")
INDIA_URL = ("https://github.com/ramSeraph/indian_communications/releases/download/cell-towers/"
             "TS_Celltower_locations.parquet")
INDIA_SOURCE = "Tarang Sanchar (DoT) 2022 scrape, mirrored by ramSeraph/indian_communications"
INDIA_DATA_DATE = date(2022, 1, 1)   # the mirror says "data from 2022"; no finer date is published
TS_DIR = settings.DATA / "telecom"

REFRESH_PENDING = ("Tarang Sanchar has no machine-readable tower search: the portal needs name, email, "
                   "phone, CAPTCHA and an emailed OTP, and answers Indian IPs only. AOI refresh is off "
                   "until a legitimate source (PM GatiShakti Public or an NIC token) is set up.")
FETCHER: Callable[[tuple], tuple[list[dict], date | datetime | None]] | None = None

SCHEMA = """
create table if not exists cell_towers (
  id bigserial primary key,
  site_id text,
  structure text,
  tags jsonb not null default '{}'::jsonb,
  geom geometry(Point, 4326) not null,
  load_id bigint
);
create index if not exists cell_towers_geom on cell_towers using gist (geom);
create index if not exists cell_towers_site on cell_towers (site_id);

create table if not exists cell_tower_loads (
  id bigserial primary key,
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
            c.execute("select pg_advisory_xact_lock(724003)")
            c.execute(SCHEMA)


def refresh_available() -> bool:
    return FETCHER is not None


def _structure(v) -> str | None:
    s = str(v).strip().title() if v is not None else ""
    return s if s in STRUCTURES else (s or None)


def _ts(d) -> datetime | None:
    if d is None:
        return None
    if isinstance(d, datetime):
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _counts(structs) -> dict:
    out: dict[str, int] = {}
    for s in structs:
        k = s or "unknown"
        out[k] = out.get(k, 0) + 1
    return out


# ---- writing ----------------------------------------------------------------------------
def _copy_rows(cur, rows, load_id):
    """rows: [(site_id, structure, tags, Point)]."""
    with cur.copy("copy cell_towers (site_id, structure, tags, geom, load_id) from stdin") as cp:
        for sid, st, tags, geom in rows:
            cp.write_row((sid, st, db.dumps(tags),
                          shapely.to_wkb(shapely.set_srid(geom, 4326), hex=True, include_srid=True), load_id))


# ---- India load (mirrored Tarang Sanchar scrape) ----------------------------------------
def load_india(parquet: Path | None = None, progress=None) -> dict:
    init_schema()
    TS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(parquet) if parquet else TS_DIR / "TS_Celltower_locations.parquet"
    if not path.exists():
        log.info("downloading %s", INDIA_URL)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", INDIA_URL, timeout=600, follow_redirects=True,
                          headers={"User-Agent": settings.HTTP_UA}) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 22):
                    f.write(chunk)
        tmp.rename(path)
    g = gpd.read_parquet(path)
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty & (g.geometry.geom_type == "Point")]
    extra = [c for c in g.columns if c not in ("site_id", "type", "geometry", "valid")]
    rows = [(None if sid is None else str(sid), _structure(t),
             {k: (v.item() if hasattr(v, "item") else v) for k, v in zip(extra, vals) if v is not None} if extra else {}, p)
            for sid, t, p, *vals in zip(g["site_id"], g["type"], g.geometry, *(g[c] for c in extra))]
    counts = _counts(r[1] for r in rows)
    counts["towers_read"] = len(rows)
    load = db.one("insert into cell_tower_loads (kind, source, data_date) values ('india', %s, %s) returning id",
                  (f"{INDIA_SOURCE} ({path.name})", _ts(INDIA_DATA_DATE)))
    lid = load["id"]
    with db.conn() as c:
        with c.transaction():
            c.execute("truncate cell_towers")
            with c.cursor() as cur:
                _copy_rows(cur, rows, lid)
            c.execute("update cell_tower_loads set status='done', finished_at=now(), counts=%s where id=%s",
                      (db.J(counts), lid))
            # the India load replaced every area, so earlier AOI refreshes no longer describe the data
            c.execute("update cell_tower_loads set status='superseded' where kind='aoi' and status='done' and id < %s",
                      (lid,))
    return {"load_id": lid, "data_date": INDIA_DATA_DATE.isoformat(), **counts}


# ---- AOI refresh ------------------------------------------------------------------------
class RefreshUnavailable(RuntimeError):
    pass


def refresh_aoi(bbox: tuple[float, float, float, float], run_id: str | None = None, progress=None) -> dict:
    """Re-pull the towers inside bbox (minx, miny, maxx, maxy) and replace them."""
    if FETCHER is None:
        raise RefreshUnavailable(REFRESH_PENDING)
    init_schema()
    minx, miny, maxx, maxy = bbox
    aoi = box(minx, miny, maxx, maxy)
    load = db.one("""insert into cell_tower_loads (kind, source, aoi, run_id)
                     values ('aoi', 'Tarang Sanchar', ST_GeomFromText(%s, 4326), %s) returning id""",
                  (aoi.wkt, run_id))
    lid = load["id"]
    try:
        if progress:
            progress(0.1, "Querying Tarang Sanchar")
        got, data_date = FETCHER(bbox)
        ts = _ts(data_date) or datetime.now(timezone.utc)
        rows = []
        for t in got:
            p = Point(float(t["lon"]), float(t["lat"]))
            if not aoi.contains(p):        # towers outside the bbox stay as they are
                continue
            tags = {k: v for k, v in t.items() if k not in ("site_id", "structure", "lon", "lat") and v is not None}
            rows.append((None if t.get("site_id") is None else str(t["site_id"]), _structure(t.get("structure")),
                         tags, p))
        counts = _counts(r[1] for r in rows)
        if progress:
            progress(0.7, "Replacing towers in the store")
        with db.conn() as c:
            with c.transaction():
                before = c.execute("""select count(*) as n from cell_towers
                                      where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (aoi.wkt,)).fetchone()["n"]
                c.execute("delete from cell_towers where ST_Intersects(geom, ST_GeomFromText(%s, 4326))", (aoi.wkt,))
                with c.cursor() as cur:
                    _copy_rows(cur, rows, lid)
                counts["replaced"] = int(before)
                c.execute("""update cell_tower_loads set status='done', finished_at=now(), counts=%s, data_date=%s
                             where id=%s""", (db.J(counts), ts, lid))
        return {"load_id": lid, "data_date": ts.isoformat(), **counts}
    except Exception as e:
        db.exec_("update cell_tower_loads set status='failed', finished_at=now(), error=%s where id=%s",
                 (f"{type(e).__name__}: {e}", lid))
        raise


# ---- reading ----------------------------------------------------------------------------
def read_band(band_lonlat) -> tuple[gpd.GeoDataFrame, dict]:
    """Towers inside the band, as layer features (class cell_tower)."""
    init_schema()
    wkt = band_lonlat.wkt
    with db.conn() as c:
        rows = c.execute("""select site_id, structure, tags, ST_AsBinary(geom) as wkb
                            from cell_towers where ST_Intersects(geom, ST_GeomFromText(%s, 4326))""", (wkt,)).fetchall()
        loads = c.execute("""select id, kind, source, data_date, finished_at from cell_tower_loads
                             where status='done' and (kind='india' or ST_Intersects(aoi, ST_GeomFromText(%s, 4326)))
                             order by finished_at desc""", (wkt,)).fetchall()
    geoms = shapely.from_wkb([bytes(r["wkb"]) for r in rows]) if rows else []
    gdf = gpd.GeoDataFrame({
        "class": ["cell_tower"] * len(rows),
        "name": [f"{r['structure']} tower" if r["structure"] else "Cell tower" for r in rows],
        "confidence": [None] * len(rows),
        "src_id": [f"ts:{r['site_id']}" if r["site_id"] else None for r in rows],
        "attrs": [{"site_id": r["site_id"], "structure": r["structure"], **(r["tags"] or {})} for r in rows],
    }, geometry=list(geoms), crs=4326)
    india = next((l for l in loads if l["kind"] == "india"), None)
    aoi = next((l for l in loads if l["kind"] == "aoi"), None)
    newest = max([l["data_date"] for l in loads if l["data_date"]], default=None)
    src = "Tarang Sanchar (DoT) mobile tower sites"
    if india:
        src += f" · India load, data {india['data_date']:%Y}" if india["data_date"] else " · India load"
    if aoi:
        src += f" · refreshed for this area {aoi['finished_at']:%d %b %Y %H:%M} UTC"
    return gdf, {"source": src, "data_date": newest.isoformat() if newest else None,
                 "india_loaded": india is not None, "last_refresh": aoi["finished_at"].isoformat() if aoi else None,
                 "counts": _counts(r["structure"] for r in rows)}


def status() -> dict:
    init_schema()
    counts = db.all_("select structure, count(*) as n from cell_towers group by 1 order by 2 desc")
    loads = db.all_("""select id, kind, source, data_date, run_id, status, started_at, finished_at, counts, error,
                              ST_AsGeoJSON(aoi)::jsonb as aoi from cell_tower_loads order by id desc limit 20""")
    return {"counts": counts, "loads": loads, "refresh_available": refresh_available(),
            "refresh_note": None if refresh_available() else REFRESH_PENDING}


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    li = sub.add_parser("load-india")
    li.add_argument("--parquet")
    ra = sub.add_parser("refresh")
    ra.add_argument("bbox", nargs=4, type=float, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    sub.add_parser("status")
    a = ap.parse_args()
    db.init_schema()
    if a.cmd == "load-india":
        print(load_india(Path(a.parquet) if a.parquet else None))
    elif a.cmd == "refresh":
        print(refresh_aoi(tuple(a.bbox)))
    else:
        import json
        print(json.dumps(status(), indent=1, default=str))


if __name__ == "__main__":
    main()
