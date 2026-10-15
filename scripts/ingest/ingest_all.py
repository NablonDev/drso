#!/usr/bin/env python
"""Ingest every reference layer listed in a manifest (see config/reference_manifest.example.yaml).

A layer may list several files; they are concatenated into one version. Coverage defaults to the
union of the files' bounding boxes unless `coverage_file` is given (per layer or top level).

Per-entry options beyond the reference.normalise arguments:
  line_width_m   lines become polygons of this width (flat ends), for layers whose rule needs an
                 area (bridge decks from centrelines). A number, or {field: <column>, map: {value:
                 width}, default: width} to pick the width per feature.
  const_fields   {column: value} stamped on every feature and carried to attrs (e.g. provider,
                 licence) — used to mark licence-restricted sources so they can be found and removed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import yaml  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from beeline_api import reference  # noqa: E402


def widen_lines(gdf: gpd.GeoDataFrame, spec) -> gpd.GeoDataFrame:
    """Line features -> polygons `width` metres wide (flat caps), buffered in their own UTM zone."""
    gdf = gdf.to_crs(4326)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if isinstance(spec, dict):
        col, wmap, dflt = spec.get("field"), spec.get("map") or {}, float(spec.get("default", 7))
        width = gdf[col].map(lambda v: float(wmap.get(v, dflt))) if col in gdf else pd.Series(dflt, index=gdf.index)
    else:
        width = pd.Series(float(spec), index=gdf.index)
    is_line = gdf.geom_type.isin(["LineString", "MultiLineString"])
    b = gdf.geometry.bounds
    zone = (((b.minx + b.maxx) / 2 + 180) // 6 + 1).astype(int)          # UTM zone from the bbox centre
    out = gdf.geometry.copy()
    for z in sorted(zone[is_line].unique()):
        idx = gdf.index[is_line & (zone == z)]
        utm = gdf.loc[idx].geometry.to_crs(32600 + int(z))
        buf = gpd.GeoSeries([g.buffer(w / 2.0, cap_style="flat") for g, w in zip(utm, width.loc[idx])],
                            index=idx, crs=utm.crs).to_crs(4326)
        out.loc[idx] = buf
    gdf["geometry"] = out
    gdf["width_m"] = width.where(is_line)
    return gdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--only", nargs="*", help="layer keys to ingest")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-restricted", action="store_true",
                    help="skip entries marked licence_restricted: true (for a client delivery)")
    a = ap.parse_args()
    m = yaml.safe_load(open(a.manifest))
    root = Path(m["root"])
    for key, entries in m["layers"].items():
        if a.only and key not in a.only:
            continue
        entries = entries if isinstance(entries, list) else [entries]
        parts, covs, unknown, sources = [], [], set(), []
        for e in entries:
            if a.no_restricted and e.get("licence_restricted"):
                print(f"{key}: skipped licence-restricted {e.get('layer') or e['path']}")
                continue
            p = root / e["path"]
            if not p.exists():
                print(f"{key}: MISSING {p}")
                continue
            gdf = pyogrio.read_dataframe(p, layer=e.get("layer"))
            keep = list(e.get("keep_fields") or [])
            if e.get("line_width_m"):
                gdf = widen_lines(gdf, e["line_width_m"])
                keep.append("width_m")
            for k, v in (e.get("const_fields") or {}).items():
                gdf[k] = v
                keep.append(k)
            norm = reference.normalise(gdf, key, class_field=e.get("class_field"), class_map=e.get("class_map") or {},
                                       const_class=e.get("const_class"), name_field=e.get("name_field"),
                                       conf_field=e.get("conf_field"), id_field=e.get("id_field"),
                                       keep_fields=keep)
            unknown |= set(norm.attrs.get("unknown_classes", []))
            parts.append(norm)
            covs.append(box(*norm.total_bounds) if len(norm) else None)
            sources.append(e.get("source") or p.name)
            print(f"{key}: {p.name} -> {len(norm)} features {dict(norm['class'].value_counts())}")
        if not parts:
            continue
        allg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
        cov_file = (entries[0].get("coverage_file") or m.get("coverage_file"))
        cov = unary_union(list(gpd.read_file(cov_file).to_crs(4326).geometry)) if cov_file else \
            unary_union([c for c in covs if c is not None])
        if unknown:
            print(f"  {key}: unmapped source classes fell back to the default: {sorted(unknown)}")
        if a.dry_run:
            continue
        prov = {"source": " + ".join(sources), "source_date": entries[0].get("source_date"),
                "model": entries[0].get("model"), "confidence": entries[0].get("confidence")}
        out = reference.write_version(key, allg, m["version"], prov, coverage=cov)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
