#!/usr/bin/env python
"""Ingest every reference layer listed in a manifest (see config/reference_manifest.example.yaml).

A layer may list several files; they are concatenated into one version. Coverage defaults to the
union of the files' bounding boxes unless `coverage_file` is given (per layer or top level).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
import yaml  # noqa: E402
from shapely.geometry import box  # noqa: E402
from shapely.ops import unary_union  # noqa: E402

from beeline_api import reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--only", nargs="*", help="layer keys to ingest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    m = yaml.safe_load(open(a.manifest))
    root = Path(m["root"])
    for key, entries in m["layers"].items():
        if a.only and key not in a.only:
            continue
        entries = entries if isinstance(entries, list) else [entries]
        parts, covs, unknown, sources = [], [], set(), []
        for e in entries:
            p = root / e["path"]
            if not p.exists():
                print(f"{key}: MISSING {p}")
                continue
            gdf = pyogrio.read_dataframe(p, layer=e.get("layer"))
            norm = reference.normalise(gdf, key, class_field=e.get("class_field"), class_map=e.get("class_map") or {},
                                       const_class=e.get("const_class"), name_field=e.get("name_field"),
                                       conf_field=e.get("conf_field"), id_field=e.get("id_field"),
                                       keep_fields=e.get("keep_fields"))
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
