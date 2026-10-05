"""Layer versions and their features in PostGIS."""
from __future__ import annotations

import io

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import MultiPolygon, Polygon, mapping

from . import db


def _multi(geom):
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    if not polys:
        return None
    return shapely.unary_union(polys) if len(polys) > 1 else _multi(polys[0])


def create_version(run_id: str, layer_key: str, mode: str, producer: str, meta: dict, gdf: gpd.GeoDataFrame | None,
                   coverage=None, status: str = "done", error: str | None = None) -> int:
    cov = _multi(coverage)
    cov_wkt = cov.wkt if cov is not None else None
    stats = {}
    if gdf is not None and len(gdf):
        stats["classes"] = {str(k): int(v) for k, v in gdf["class"].value_counts().items()}
    if meta.get("stats_extra"):
        stats["by_voltage"] = meta["stats_extra"]
    if meta.get("refresh"):
        stats["refresh"] = meta["refresh"]
    row = db.one("""insert into layer_versions (run_id, layer_key, mode, producer, source, file_name, model, model_version,
                      image_source, image_date, model_run_at, confidence, feature_count, coverage, status, error, stats)
                    values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                            case when %s::text is null then null else ST_Multi(ST_GeomFromText(%s, 4326)) end,
                            %s,%s,%s) returning id""",
                 (run_id, layer_key, mode, producer, meta.get("source"), meta.get("file_name"), meta.get("model"),
                  meta.get("model_version"), meta.get("image_source"), meta.get("image_date"), meta.get("model_run_at"),
                  meta.get("confidence"), 0 if gdf is None else int(len(gdf)), cov_wkt, cov_wkt, status, error,
                  db.J(stats)))
    lv = row["id"]
    if gdf is not None and len(gdf):
        write_features(lv, gdf)
    return lv


def write_features(lv: int, gdf: gpd.GeoDataFrame):
    g = gdf.to_crs(4326)
    geoms = shapely.set_srid(np.asarray(g.geometry.values), 4326)
    wkb = shapely.to_wkb(geoms, hex=True, include_srid=True)
    conf = g["confidence"] if "confidence" in g else pd.Series([None] * len(g))
    origin = g["origin"] if "origin" in g else pd.Series(["auto"] * len(g))
    attrs = g["attrs"] if "attrs" in g else pd.Series([None] * len(g))
    with db.conn() as c:
        with c.cursor() as cur:
            with cur.copy("copy features (layer_version_id, class, name, confidence, src_id, origin, attrs, geom) from stdin") as cp:
                for i in range(len(g)):
                    cf = conf.iloc[i]
                    cp.write_row((lv, g["class"].iloc[i], _s(g["name"].iloc[i] if "name" in g else None),
                                  None if cf is None or (isinstance(cf, float) and np.isnan(cf)) else float(cf),
                                  _s(g["src_id"].iloc[i] if "src_id" in g else None), origin.iloc[i],
                                  db.dumps(attrs.iloc[i]) if isinstance(attrs.iloc[i], dict) else None, wkb[i]))
    db.exec_("update layer_versions set feature_count=(select count(*) from features where layer_version_id=%s) where id=%s",
             (lv, lv))


def _s(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    v = str(v)
    return None if v in ("None", "nan") else v


def link(run_id: str, layer_key: str, lv: int, carried: bool = False):
    db.exec_("""insert into run_layers (run_id, layer_key, layer_version_id, carried, stale) values (%s,%s,%s,%s,false)
                on conflict (run_id, layer_key) do update set layer_version_id=excluded.layer_version_id,
                carried=excluded.carried, stale=false""", (run_id, layer_key, lv, carried))


def read_version(lv: int) -> gpd.GeoDataFrame:
    with db.conn() as c:
        rows = c.execute("""select class, name, confidence, src_id, origin, attrs, ST_AsBinary(geom) as wkb
                            from features where layer_version_id=%s""", (lv,)).fetchall()
    if not rows:
        return gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": [], "origin": [], "attrs": []},
                                geometry=[], crs=4326)
    geoms = shapely.from_wkb([bytes(r["wkb"]) for r in rows])
    return gpd.GeoDataFrame({"class": [r["class"] for r in rows], "name": [r["name"] for r in rows],
                             "confidence": [r["confidence"] for r in rows], "src_id": [r["src_id"] for r in rows],
                             "origin": [r["origin"] for r in rows], "attrs": [r["attrs"] for r in rows]},
                            geometry=geoms, crs=4326)


def version(lv: int) -> dict | None:
    return db.one("""select lv.*, ST_AsGeoJSON(coverage)::jsonb as coverage_geojson from layer_versions lv where id=%s""", (lv,))


def run_layers(run_id: str) -> list[dict]:
    return db.all_("""select rl.layer_key, rl.carried, rl.stale, lv.id as layer_version_id, lv.mode, lv.producer, lv.source,
                             lv.file_name, lv.model, lv.model_version, lv.image_source, lv.image_date, lv.model_run_at,
                             lv.confidence, lv.feature_count, lv.status, lv.error, lv.stats, lv.created_at,
                             lv.run_id as extracted_in,
                             ST_AsGeoJSON(lv.coverage, 6)::jsonb as coverage_geojson
                      from run_layers rl join layer_versions lv on lv.id = rl.layer_version_id
                      where rl.run_id=%s order by rl.layer_key""", (run_id,))
