"""User files for Replace / Add more: inspect, map attributes to the Beeline vocabulary, read."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pyogrio

from beeline_engine.config import load_layers

from . import db, reference, settings

ALLOWED = {".gpkg", ".geojson", ".json", ".zip", ".kml", ".shp"}


def save(filename: str, fileobj) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED:
        raise ValueError(f"unsupported file type {ext}; use GeoPackage, GeoJSON, KML or a zipped shapefile")
    row = db.one("insert into uploads (filename, path) values (%s, '') returning id", (filename,))
    uid = row["id"]
    d = settings.UPLOADS_DIR / str(uid)
    d.mkdir(parents=True, exist_ok=True)
    dst = d / Path(filename).name
    with open(dst, "wb") as f:
        shutil.copyfileobj(fileobj, f)
    path = dst
    if ext == ".zip":
        with zipfile.ZipFile(dst) as z:
            for m in z.namelist():
                if ".." in m or m.startswith("/"):
                    raise ValueError("zip contains unsafe paths")
            z.extractall(d / "unzipped")
        shps = list((d / "unzipped").rglob("*.shp")) + list((d / "unzipped").rglob("*.gpkg")) + \
            list((d / "unzipped").rglob("*.geojson"))
        if not shps:
            raise ValueError("zip holds no .shp, .gpkg or .geojson")
        path = shps[0]
    info = inspect(path)
    db.exec_("update uploads set path=%s, layers=%s, info=%s where id=%s",
             (str(path), db.J(info["layers"]), db.J(info), uid))
    return {"id": uid, "filename": filename, **info}


def inspect(path: Path, layer: str | None = None) -> dict:
    layers = [l[0] for l in pyogrio.list_layers(path)]
    lyr = layer or layers[0]
    meta = pyogrio.read_info(path, layer=lyr)
    df = pyogrio.read_dataframe(path, layer=lyr, max_features=2000)
    samples = {}
    for c in meta["fields"]:
        vals = df[c].dropna().astype(str)
        samples[c] = sorted(vals.unique().tolist())[:40]
    return {"layers": layers, "layer": lyr, "fields": list(meta["fields"]), "feature_count": int(meta["features"]),
            "crs": meta["crs"], "geometry_type": meta["geometry_type"], "samples": samples,
            "bounds": list(meta.get("total_bounds") or [])}


def get(uid: int) -> dict:
    r = db.one("select * from uploads where id=%s", (uid,))
    if not r:
        raise KeyError(f"upload {uid} not found")
    return r


def check_mapping(uid: int, layer_key: str, mapping: dict) -> dict:
    """mapping: {layer?, class_field?, class_map{src: beeline}, const_class?, name_field?, conf_field?}"""
    up = get(uid)
    reg = {l["key"]: l for l in load_layers()["layers"]}
    if layer_key not in reg:
        raise KeyError(f"unknown layer {layer_key}")
    vocab = reg[layer_key]["classes"]
    info = inspect(Path(up["path"]), mapping.get("layer"))
    unknown = []
    if mapping.get("class_field"):
        cm = mapping.get("class_map") or {}
        vals = info["samples"].get(mapping["class_field"], [])
        unknown = [v for v in vals if cm.get(v, v) not in vocab]
    bad_targets = [v for v in (mapping.get("class_map") or {}).values() if v not in vocab]
    return {"vocabulary": vocab, "unknown_values": unknown, "bad_targets": bad_targets,
            "fallback_class": vocab[0], "ok": not bad_targets}


def read_mapped(uid: int, layer_key: str, mapping: dict, band_lonlat) -> gpd.GeoDataFrame:
    up = get(uid)
    gdf = pyogrio.read_dataframe(up["path"], layer=mapping.get("layer"))
    norm = reference.normalise(gdf, layer_key, class_field=mapping.get("class_field"),
                               class_map=mapping.get("class_map") or {}, const_class=mapping.get("const_class"),
                               name_field=mapping.get("name_field"), conf_field=mapping.get("conf_field"))
    norm = norm[norm.intersects(band_lonlat)].copy()
    lines_polys = norm.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])
    if lines_polys.any():
        norm.loc[lines_polys, "geometry"] = norm.loc[lines_polys].geometry.intersection(band_lonlat)
    norm = norm[~norm.geometry.is_empty]
    norm["confidence"] = norm["confidence"].fillna(1.0)
    return norm.reset_index(drop=True)


def remember_mapping(source_key: str, layer_key: str, mapping: dict):
    db.exec_("""insert into attr_mappings (source_key, layer_key, mapping) values (%s,%s,%s)
                on conflict (source_key) do update set layer_key=excluded.layer_key, mapping=excluded.mapping, updated_at=now()""",
             (source_key, layer_key, db.J(mapping)))


def recall_mapping(source_key: str) -> dict | None:
    r = db.one("select * from attr_mappings where source_key=%s", (source_key,))
    return r["mapping"] if r else None
