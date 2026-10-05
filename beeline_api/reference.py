"""Reference layer store (the "cached" statewide layers in the local bucket).

Layout:  <REFERENCE_DIR>/<region>/<layer_key>/<version>/data.gpkg   (layer "features")
                                                         /meta.json
         <REFERENCE_DIR>/<region>/<layer_key>/CURRENT                (text: active version)

Every stored feature carries: class (Beeline vocabulary), name, confidence, src_id.
meta.json carries the provenance: source description, source/imagery date, model + model run
date when a model produced it, and the coverage polygon (where the data is valid).

When a band is not covered, `ensure_coverage` runs the per-layer fetch hook configured in
config/reference_sources.yaml (empty by default) and otherwise reports the gap.
"""
from __future__ import annotations

import json
import logging
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
import yaml
from shapely import make_valid
from shapely.geometry import box, mapping, shape

from beeline_engine.config import CONFIG_DIR, load_layers

from . import settings

log = logging.getLogger(__name__)


def _layer_dir(key: str, region: str | None = None) -> Path:
    return settings.REFERENCE_DIR / (region or settings.REGION) / key


def current_version(key: str, region: str | None = None) -> str | None:
    p = _layer_dir(key, region) / "CURRENT"
    return p.read_text().strip() if p.exists() else None


def meta(key: str, version: str | None = None, region: str | None = None) -> dict | None:
    version = version or current_version(key, region)
    if not version:
        return None
    p = _layer_dir(key, region) / version / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def normalise(gdf: gpd.GeoDataFrame, key: str, class_field: str | None = None, class_map: dict | None = None,
              const_class: str | None = None, name_field: str | None = None, conf_field: str | None = None,
              id_field: str | None = None, keep_fields: list[str] | None = None) -> gpd.GeoDataFrame:
    """Source features -> the store's columns (class, name, confidence, src_id, geometry), 2D.
    keep_fields: source columns carried along as extra columns (read_clip packs them into attrs)."""
    reg = {l["key"]: l for l in load_layers()["layers"]}[key]
    vocab = reg["classes"]
    if gdf.crs is None:
        raise ValueError("source has no CRS")
    gdf = gdf.to_crs(4326)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["geometry"] = shapely.force_2d(gdf.geometry.apply(lambda g: g if g.is_valid else make_valid(g)).values)
    if const_class:
        cls = pd.Series(const_class, index=gdf.index)
    elif class_field:
        raw = gdf[class_field].astype(str)
        cls = raw.map(lambda v: (class_map or {}).get(v, v if v in vocab else None))
    else:
        cls = pd.Series(vocab[0] if len(vocab) == 1 else None, index=gdf.index)
    unknown = sorted(set(gdf.loc[cls.isna(), class_field].astype(str))) if class_field else []
    cls = cls.fillna(vocab[0])
    bad = set(cls) - set(vocab)
    if bad:
        raise ValueError(f"classes {bad} not in the {key} vocabulary {vocab}")
    out = gpd.GeoDataFrame({
        "class": cls.values,
        "name": gdf[name_field].map(lambda v: None if v is None or v != v or str(v).strip() == "" else str(v)).values
        if name_field else None,
        "confidence": pd.to_numeric(gdf[conf_field], errors="coerce").values if conf_field else None,
        "src_id": gdf[id_field].astype(str).values if id_field else gdf.index.astype(str),
    }, geometry=gdf.geometry.values, crs=4326)
    for f in keep_fields or []:
        if f in gdf.columns and f not in out.columns:
            out[f] = gdf[f].values
    out.attrs["unknown_classes"] = unknown
    return out


def write_version(key: str, gdf: gpd.GeoDataFrame, version: str, provenance: dict, coverage=None,
                  region: str | None = None, make_current: bool = True) -> Path:
    d = _layer_dir(key, region) / version
    d.mkdir(parents=True, exist_ok=True)
    path = d / "data.gpkg"
    if path.exists():
        path.unlink()
    if len(gdf):
        pyogrio.write_dataframe(gdf, path, layer="features", driver="GPKG", promote_to_multi=False)
    if coverage is None:
        coverage = box(*gdf.total_bounds) if len(gdf) else None
    m = {
        "key": key, "version": version, "region": region or settings.REGION,
        "feature_count": int(len(gdf)),
        "classes": {str(k): int(v) for k, v in gdf["class"].value_counts().items()} if len(gdf) else {},
        "coverage": mapping(coverage) if coverage is not None else None,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        **provenance,
    }
    (d / "meta.json").write_text(json.dumps(m, indent=2, default=str))
    if make_current:
        (_layer_dir(key, region) / "CURRENT").write_text(version)
    return path


def read_clip(key: str, band_lonlat, version: str | None = None, region: str | None = None):
    """(GeoDataFrame clipped to the band, meta) or (None, None) when the layer is not in the store."""
    version = version or current_version(key, region)
    if not version:
        return None, None
    d = _layer_dir(key, region) / version
    m = json.loads((d / "meta.json").read_text())
    path = d / "data.gpkg"
    if not path.exists():
        return _empty(), m
    gdf = pyogrio.read_dataframe(path, layer="features", bbox=tuple(band_lonlat.bounds))
    if len(gdf):
        gdf = gdf[gdf.intersects(band_lonlat)]
        poly = gdf.geom_type.isin(["Polygon", "MultiPolygon", "LineString", "MultiLineString"])
        if poly.any():
            gdf.loc[poly, "geometry"] = gdf.loc[poly].geometry.intersection(band_lonlat)
        gdf = gdf[~gdf.geometry.is_empty]
    # columns beyond the standard ones (e.g. road width, turbine scene) travel as attrs
    std = {"class", "name", "confidence", "src_id", "geometry", "attrs"}
    extra = [c for c in gdf.columns if c not in std]
    if extra:
        gdf = gdf.copy()
        gdf["attrs"] = [{c: (None if (v is None or v != v) else (v.item() if hasattr(v, "item") else v))
                         for c, v in zip(extra, row)} for row in gdf[extra].itertuples(index=False, name=None)]
        gdf = gdf.drop(columns=extra)
    return gdf.reset_index(drop=True), m


def coverage_of(m: dict | None):
    if not m or not m.get("coverage"):
        return None
    return shape(m["coverage"])


def _empty() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": []}, geometry=[], crs=4326)


def _hooks() -> dict:
    p = CONFIG_DIR / "reference_sources.yaml"
    if not p.exists():
        return {}
    return (yaml.safe_load(p.read_text()) or {}).get("fetch_hooks", {}) or {}


def ensure_coverage(key: str, band_lonlat) -> tuple[bool, str]:
    """Make sure the store covers the band. Runs the configured fetch hook if not."""
    m = meta(key)
    cov = coverage_of(m)
    if cov is not None and cov.buffer(1e-6).contains(band_lonlat):
        return True, "covered"
    hook = _hooks().get(key)
    if not hook:
        return False, "not covered and no fetch hook configured" if m else "layer not in the reference store"
    minx, miny, maxx, maxy = band_lonlat.bounds
    cmd = hook.format(minx=minx, miny=miny, maxx=maxx, maxy=maxy, key=key,
                      region=settings.REGION, reference_dir=settings.REFERENCE_DIR)
    log.info("running fetch hook for %s: %s", key, cmd)
    r = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=6 * 3600)
    if r.returncode != 0:
        return False, f"fetch hook failed ({r.returncode}): {r.stderr[-400:]}"
    cov = coverage_of(meta(key))
    ok = cov is not None and cov.buffer(1e-6).contains(band_lonlat)
    return ok, "fetched" if ok else "fetch hook ran but the band is still not covered"


def store_status() -> list[dict]:
    out = []
    for l in load_layers()["layers"]:
        m = meta(l["key"])
        out.append({"key": l["key"], "name": l["name"], "version": m["version"] if m else None,
                    "feature_count": m["feature_count"] if m else 0,
                    "source": m.get("source") if m else None, "source_date": m.get("source_date") if m else None,
                    "has_coverage": bool(m and m.get("coverage"))})
    return out
