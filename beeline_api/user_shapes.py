"""Shapes added by hand on the map (drawn, or loaded from a KML / KMZ / GeoJSON / shapefile / GPKG).

Vivek, 8 Oct 2026:
  temporary  used only in the run it was added to
  permanent  part of that layer's data from now on: added to the run it was added in and to every
             later extraction of the layer whose band it touches

A shape becomes an ordinary feature of its layer (origin "drawn", src_id "shape:<id>"), so the rules
treat it like any other feature. Adding or deleting one rebuilds the run's current layer version in
place of a re-extraction, so the run can be re-solved straight away.
"""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import mapping, shape
from shapely.validation import make_valid

from beeline_engine.config import LineSpec, load_layers, resolve_rules

from . import db, layers
from .derive import derive_clusters
from .geo import band_polygon, run_margin_m

SCHEMA = """
create table if not exists user_shapes (
  id bigserial primary key,
  scope text not null check (scope in ('temporary', 'permanent')),
  run_id text references runs(id) on delete cascade,      -- temporary shapes only
  added_in_run text,                                      -- where it was added (kept after the run is deleted)
  layer_key text not null,
  class text not null,
  name text not null,
  attrs jsonb not null default '{}'::jsonb,
  geom geometry(Geometry, 4326) not null,
  created_at timestamptz not null default now(),
  deleted_at timestamptz
);
create index if not exists user_shapes_geom on user_shapes using gist (geom);
create index if not exists user_shapes_layer on user_shapes (layer_key) where deleted_at is null;
"""

KIND_GEOMS = {"polygon": {"Polygon", "MultiPolygon"}, "line": {"LineString", "MultiLineString"},
              "point": {"Point", "MultiPoint"}}
FILE_TYPES = {".kml", ".kmz", ".geojson", ".json", ".gpkg", ".zip", ".shp"}
MAX_FEATURES = 5000


def init_schema():
    with db.conn() as c:
        with c.transaction():
            c.execute("select pg_advisory_xact_lock(724009)")
            c.execute(SCHEMA)


def accepts(layer: dict, geom_type: str) -> bool:
    kind = layer.get("kind", "any")
    return kind == "any" or geom_type in KIND_GEOMS.get(kind, set())


# ---- file parsing ---------------------------------------------------------------------------
def parse_file(filename: str, fileobj) -> dict:
    """Any supported map file -> GeoJSON FeatureCollection in WGS84 (all layers / KML folders)."""
    ext = Path(filename).suffix.lower()
    if ext not in FILE_TYPES:
        raise ValueError(f"unsupported file type {ext or '(none)'}; use KML, KMZ, GeoJSON, GeoPackage or a zipped shapefile")
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        src = d / Path(filename).name
        src.write_bytes(fileobj.read())
        path = src
        if ext in (".kmz", ".zip"):
            with zipfile.ZipFile(src) as z:
                for m in z.namelist():
                    if ".." in m or m.startswith("/"):
                        raise ValueError("the archive contains unsafe paths")
                z.extractall(d / "x")
            found = [p for pat in ("*.kml", "*.shp", "*.gpkg", "*.geojson") for p in (d / "x").rglob(pat)]
            if not found:
                raise ValueError("the archive holds no KML, shapefile, GeoPackage or GeoJSON")
            path = found[0]
        frames = []
        for lyr in [l[0] for l in pyogrio.list_layers(path)]:
            g = pyogrio.read_dataframe(path, layer=lyr)
            if len(g):
                g["_layer"] = lyr
                frames.append(g)
        if not frames:
            raise ValueError("the file has no features")
        g = pd.concat(frames, ignore_index=True)
        g = gpd.GeoDataFrame(g, geometry="geometry", crs=frames[0].crs)
    if g.crs is None:
        g = g.set_crs(4326)
    g = g.to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    g = g.explode(index_parts=False)                      # GeometryCollections from KML into parts
    g = g[g.geometry.geom_type.isin(["Point", "LineString", "Polygon", "MultiPoint", "MultiLineString", "MultiPolygon"])]
    if len(g) > MAX_FEATURES:
        raise ValueError(f"the file has {len(g):,} features; the limit is {MAX_FEATURES:,}")
    name_col = next((c for c in ("Name", "name", "NAME") if c in g.columns), None)
    feats = []
    for i, (_, r) in enumerate(g.iterrows()):
        geom = shapely.force_2d(r.geometry)
        nm = r[name_col] if name_col and isinstance(r[name_col], str) and r[name_col].strip() else None
        feats.append({"type": "Feature", "properties": {"name": nm, "source_layer": r["_layer"]}, "geometry": mapping(geom)})
    return {"type": "FeatureCollection", "features": feats, "count": len(feats),
            "geometry_types": sorted({f["geometry"]["type"] for f in feats})}


# ---- storage --------------------------------------------------------------------------------
def add(run: dict, layer_key: str, cls: str, name: str, scope: str, features: list[dict]) -> list[int]:
    reg = {l["key"]: l for l in load_layers()["layers"]}
    lay = reg[layer_key]
    ids = []
    geoms = []
    for f in features:
        g = shape(f["geometry"])
        if not g.is_valid:
            g = make_valid(g)
        g = shapely.force_2d(g)
        if g.is_empty:
            raise ValueError("an empty shape cannot be added")
        if not accepts(lay, g.geom_type):
            raise ValueError(f"{lay['name']} takes {lay.get('kind')} shapes, not {g.geom_type}")
        geoms.append((g, (f.get("properties") or {}).get("name")))
    for i, (g, own) in enumerate(geoms):
        nm = name if len(geoms) == 1 else f"{name} — {own or i + 1}"
        row = db.one("""insert into user_shapes (scope, run_id, added_in_run, layer_key, class, name, attrs, geom)
                        values (%s, %s, %s, %s, %s, %s, %s, ST_GeomFromText(%s, 4326)) returning id""",
                     (scope, run["id"] if scope == "temporary" else None, run["id"], layer_key, cls, nm,
                      db.J({"source_name": own}), g.wkt))
        ids.append(row["id"])
    return ids


def delete(shape_id: int) -> dict | None:
    row = db.one("update user_shapes set deleted_at=now() where id=%s and deleted_at is null returning id, layer_key, scope",
                 (shape_id,))
    return row


def _band(run: dict):
    band, _ = band_polygon((run["a_lon"], run["a_lat"]), (run["b_lon"], run["b_lat"]), run["band_km"], margin_m=run_margin_m())
    return band


def list_for_run(run: dict) -> list[dict]:
    """Temporary shapes of this run and permanent shapes that touch its band."""
    band = _band(run)
    rows = db.all_("""select id, scope, added_in_run, layer_key, class, name, created_at, ST_AsGeoJSON(geom, 7)::jsonb as geometry,
                             ST_GeometryType(geom) as gtype
                      from user_shapes where deleted_at is null
                        and ((scope='temporary' and run_id=%s)
                             or (scope='permanent' and ST_Intersects(geom, ST_GeomFromText(%s, 4326))))
                      order by created_at desc, id desc""", (run["id"], band.wkt))
    reg = {l["key"]: l["name"] for l in load_layers()["layers"]}
    return [{**r, "layer_name": reg.get(r["layer_key"], r["layer_key"]), "created_at": r["created_at"].isoformat(),
             "gtype": r["gtype"].replace("ST_", "")} for r in rows]


def features_for(run: dict, layer_key: str, band=None) -> gpd.GeoDataFrame:
    """The run's active shapes of one layer as layer features, clipped to the band."""
    band = band if band is not None else _band(run)
    rows = db.all_("""select id, class, name, ST_AsBinary(geom) as wkb from user_shapes
                      where deleted_at is null and layer_key=%s
                        and ((scope='temporary' and run_id=%s)
                             or (scope='permanent' and ST_Intersects(geom, ST_GeomFromText(%s, 4326))))""",
                   (layer_key, run["id"], band.wkt))
    if not rows:
        return gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": [], "origin": [], "attrs": []},
                                geometry=[], crs=4326)
    geoms = shapely.from_wkb([bytes(r["wkb"]) for r in rows])
    clipped = [g if g.geom_type in ("Point", "MultiPoint") else g.intersection(band) for g in geoms]
    g = gpd.GeoDataFrame({"class": [r["class"] for r in rows], "name": [r["name"] for r in rows],
                          "confidence": [1.0] * len(rows), "src_id": [f"shape:{r['id']}" for r in rows],
                          "origin": ["drawn"] * len(rows), "attrs": [{"hand_added": True} for _ in rows]},
                         geometry=clipped, crs=4326)
    return g[~g.geometry.is_empty].reset_index(drop=True)


def merge(run: dict, layer_key: str, gdf: gpd.GeoDataFrame | None, band) -> gpd.GeoDataFrame | None:
    """Extraction hook: the layer's output plus the run's active shapes."""
    extra = features_for(run, layer_key, band)
    if not len(extra):
        return gdf
    if gdf is None or not len(gdf):
        return extra
    base = gdf[~gdf.get("src_id", pd.Series([""] * len(gdf), index=gdf.index)).astype(str).str.startswith("shape:")]
    return gpd.GeoDataFrame(pd.concat([base, extra], ignore_index=True), geometry="geometry", crs=4326)


def _new_version_like(run: dict, cur: dict, layer_key: str, source: str) -> int:
    meta = {"source": source, "file_name": cur["file_name"], "model": cur["model"], "model_version": cur["model_version"],
            "image_source": cur["image_source"], "image_date": cur["image_date"], "model_run_at": cur["model_run_at"],
            "confidence": cur["confidence"], "stats_extra": {k: v for k, v in (cur["stats"] or {}).items() if k != "classes"}}
    cov = shape(cur["coverage_geojson"]) if cur["coverage_geojson"] else None
    return layers.create_version(run["id"], layer_key, cur["mode"], cur["producer"], meta, None, cov)


def _copy_features(src_lv: int, dst_lv: int, drop_shapes: bool, drop_ids: list[int] | None = None):
    """Server-side copy of a version's features (a run layer can hold ~100k buildings)."""
    db.exec_("""insert into features (layer_version_id, class, name, confidence, src_id, origin, attrs, geom)
                select %s, class, name, confidence, src_id, origin, attrs, geom from features
                where layer_version_id=%s
                  and (not %s or coalesce(src_id, '') not like 'shape:%%')
                  and not (id = any(%s))""", (dst_lv, src_lv, drop_shapes, drop_ids or []))


def _finish_version(lv: int, extra: gpd.GeoDataFrame | None):
    if extra is not None and len(extra):
        layers.write_features(lv, extra)          # also refreshes feature_count
    else:
        db.exec_("update layer_versions set feature_count=(select count(*) from features where layer_version_id=%s) where id=%s",
                 (lv, lv))
    db.exec_("""update layer_versions set stats = jsonb_set(coalesce(stats, '{}'::jsonb), '{classes}',
                  coalesce((select jsonb_object_agg(class, n) from (select class, count(*) n from features
                            where layer_version_id=%s group by class) c), '{}'::jsonb)) where id=%s""", (lv, lv))


def rebuild_run_layer(run: dict, layer_key: str, changed=None) -> int | None:
    """New version of the run's layer: current features without hand-added shapes + the active
    shapes. `changed` (WGS84 geometries of the shapes added or removed) lets a buildings change
    re-derive the built-up clusters around it only. Returns the new version id."""
    cur = next((r for r in layers.run_layers(run["id"]) if r["layer_key"] == layer_key), None)
    if cur is None:
        return None
    extra = features_for(run, layer_key)
    base_src = (cur["source"] or "").split(" + hand-added shapes")[0]
    lv = _new_version_like(run, cur, layer_key, base_src + (f" + hand-added shapes ({len(extra)})" if len(extra) else ""))
    _copy_features(cur["layer_version_id"], lv, drop_shapes=True)
    _finish_version(lv, extra)
    layers.link(run["id"], layer_key, lv)
    if layer_key == "buildings" and changed:
        _rebuild_clusters_near(run, lv, changed)
    return lv


def _rebuild_clusters_near(run: dict, buildings_lv: int, changed: list):
    """Re-derive built-up clusters only where the buildings changed: the clusters within the rule's
    spacing of a changed shape are replaced by clusters derived from the buildings around them."""
    from beeline_engine.frame import utm_epsg
    cur = next((r for r in layers.run_layers(run["id"]) if r["layer_key"] == "clusters"), None)
    if cur is None:
        return
    rules = resolve_rules(LineSpec(run["kv"], run["circuit"]), run["rule_overrides"] or {})
    p = rules["clusters"].params
    spacing, min_area = float(p.get("max_spacing_m", 50)), float(p.get("min_area_m2", 9000))
    reg = {l["key"]: l for l in load_layers()["layers"]}
    area = shapely.union_all([shapely.force_2d(g) for g in changed])
    epsg = utm_epsg(area.centroid.x, area.centroid.y)

    def buf_ll(g, d):
        return gpd.GeoSeries([g], crs=4326).to_crs(epsg).buffer(d).to_crs(4326).iloc[0]

    near = buf_ll(area, spacing)
    hit = db.all_("""select id, ST_AsBinary(geom) as wkb from features where layer_version_id=%s
                     and coalesce(src_id, '') not like 'shape:%%' and ST_Intersects(geom, ST_GeomFromText(%s, 4326))""",
                  (cur["layer_version_id"], near.wkt))
    affected = list(shapely.from_wkb([bytes(r["wkb"]) for r in hit])) if hit else []
    core = shapely.union_all([near, *affected])
    region = buf_ll(core, spacing)
    rows = db.all_("""select class, ST_AsBinary(geom) as wkb from features
                      where layer_version_id=%s and ST_Intersects(geom, ST_GeomFromText(%s, 4326))""",
                   (buildings_lv, region.wkt))
    bl = gpd.GeoDataFrame({"class": [r["class"] for r in rows]},
                          geometry=list(shapely.from_wkb([bytes(r["wkb"]) for r in rows])) if rows else [], crs=4326)
    local = derive_clusters(bl, spacing, min_area, reg["clusters"].get("default_confidence", 0.91))
    if len(local):
        local = local[local.intersects(core)].copy()
        local["origin"] = "auto"
    lv = _new_version_like(run, cur, "clusters", cur["source"] or "")
    _copy_features(cur["layer_version_id"], lv, drop_shapes=False, drop_ids=[r["id"] for r in hit])
    _finish_version(lv, local)
    layers.link(run["id"], "clusters", lv)
