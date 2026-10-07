#!/usr/bin/env python
"""Ingest the 7 Oct 2026 water-feature deliveries into the reference store (Vivek, 7 Oct 2026).

  river      gujarat_rivers_canals_final.gpkg, "Rivers - Conflated", except descr = Reservoir
  reservoir  the current reservoir version + the "Reservoir" polygons of that layer
  canal      gujarat_rivers_canals_final.gpkg, "Canals - Conflated" (centrelines)
  nalla      sisdp_gujarat_drainage.zip, Phase II 1:10k drainage (2016-19), category Stream only
             (Phase I is 95 % the same lines on the Bhuj corridor; its rivers / canals come from
             the conflated file instead)

Usage: python scripts/ingest_water.py [--version water-2026-10-07]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402
from shapely.geometry import shape  # noqa: E402

from beeline_api import reference  # noqa: E402

SRC = ROOT / "LULC output" / "transfers" / "water_features"
RC = SRC / "gujarat_rivers_canals_final.gpkg"
DRAIN_ZIP = SRC / "sisdp_gujarat_drainage.zip"
PHASE2 = "Phase II 1:10k drainage - Gujarat (2016-19)"


def coverage_of(key):
    m = reference.meta(key)
    return shape(m["coverage"]) if m and m.get("coverage") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="water-2026-10-07")
    a = ap.parse_args()
    v = a.version
    prov = {"source_file": RC.name, "source_date": "2026-10-07", "model": None, "model_run_date": None, "confidence": None}

    rivers = pyogrio.read_dataframe(RC, layer="Rivers - Conflated")
    is_res = rivers["descr"].astype(str) == "Reservoir"
    keep = ["source_agency", "source_layer", "descr", "nwia_class", "legal_status", "area_km2"]

    cov = coverage_of("river")
    n = reference.normalise(rivers[~is_res], "river", const_class="river", name_field="name", keep_fields=keep)
    reference.write_version("river", n, v, {**prov, "source": "Rivers - Conflated (GatiShakti UGI refined + SAC/ISRO NWIA 2019); "
                                             "Reservoir polygons moved to the Reservoir layer"}, coverage=cov)
    print(f"river: {len(n)} features")

    # reservoir: the current version plus the conflated layer's reservoir polygons
    cov = coverage_of("reservoir")
    cur_v = reference.current_version("reservoir")
    cur_path = reference._layer_dir("reservoir", None) / cur_v / "data.gpkg" if cur_v else None
    cur = pyogrio.read_dataframe(cur_path, layer="features") if cur_path and cur_path.exists() else None
    add = reference.normalise(rivers[is_res], "reservoir", const_class="reservoir", name_field="name", keep_fields=keep)
    add["src_id"] = "rivers-conflated:" + add["src_id"].astype(str)
    res = add if cur is None else pd.concat([cur, add], ignore_index=True)
    res = res.set_crs(4326, allow_override=True)
    reference.write_version("reservoir", res, v, {**prov, "source": f"Reservoir {cur_v} + {len(add)} 'Reservoir' polygons "
                                                  "from Rivers - Conflated (GatiShakti UGI)"}, coverage=cov)
    print(f"reservoir: {0 if cur is None else len(cur)} existing + {len(add)} added = {len(res)}")

    cov = coverage_of("canal")
    canals = pyogrio.read_dataframe(RC, layer="Canals - Conflated")
    n = reference.normalise(canals, "canal", const_class="canal", name_field="name",
                            keep_fields=["canal_class", "source_agency", "legal_status", "length_km"])
    reference.write_version("canal", n, v, {**prov, "source": "Canals - Conflated (GatiShakti UGI, refined to <=0.5 m/px)"},
                            coverage=cov)
    print(f"canal: {len(n)} features")

    cov = coverage_of("nalla")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(DRAIN_ZIP) as z:
            z.extract("sisdp_gujarat_drainage.gpkg", td)
        d = pyogrio.read_dataframe(Path(td) / "sisdp_gujarat_drainage.gpkg", layer=PHASE2, where="category = 'Stream'")
    n = reference.normalise(d, "nalla", const_class="nalla", id_field="src_id", keep_fields=["dr_code", "length_km"])
    reference.write_version("nalla", n, v, {**prov, "source_file": DRAIN_ZIP.name, "source_date": "2019-12-31",
                                            "source": "NRSC/ISRO Bhuvan SISDP Phase II 1:10k drainage (2016-19), streams"},
                            coverage=cov)
    print(f"nalla: {len(n)} features")


if __name__ == "__main__":
    main()
