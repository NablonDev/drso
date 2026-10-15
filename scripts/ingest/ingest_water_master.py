#!/usr/bin/env python
"""Ingest gujarat_water_master.zip (8 Oct 2026, Vivek: overwrite the existing water layers).

  river        Rivers, except wb_subtype "Reservoir" (those join the reservoir layer, 7 Oct decision)
  reservoir    Reservoirs + the Rivers "Reservoir" polygons
  canal        Canals
  nalla        Nallas (order 3-4)
  lake         Lakes
  pond         Ponds-Tanks (footprints) + Ponds-Tanks (points) + NWIA aquaculture ponds (Wetlands)
  wet_land     Wetlands: inland waterlogged, riverine wetland, other
  other_water  Wetlands: coastal creek, mud flat, mangrove, salt marsh, salt pan, beach, coral reef,
               lagoon + the CRZ layer's marine 25 km screening buffer
  crz          CRZ zones (CZMP 2019): crz_ia, crz_ia_mangrove_buffer, crz_ib, crz_ii, crz_iii_ndz,
               crz_iiia, crz_iiib, crz_iva, crz_ivb
  dams         Dams - lines + Dams - points
Not used: Hydraulic structures (no matching layer). Coverage of every layer: the Gujarat boundary.

Usage: python scripts/ingest/ingest_water_master.py <path to gujarat_water_master.gpkg or .zip> [--version ...]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import pyogrio  # noqa: E402

from beeline_api import reference  # noqa: E402

WET_CODES = {   # NWIA 2019 wettcode → class, for rows whose subtype is the bare code
    "1101": ("wet_land", "wetland"), "1105": ("wet_land", "waterlogged"), "1203": ("wet_land", "waterlogged"),
    "1204": ("other_water", "salt_pan"), "1205": ("pond", "aquaculture_pond"), "9999": ("wet_land", "wetland"),
    "2101": ("other_water", "lagoon"), "2102": ("other_water", "creek"), "2103": ("other_water", "beach"),
    "2104": ("other_water", "mud_flat"), "2105": ("other_water", "salt_marsh"), "2106": ("other_water", "mangrove"),
    "2107": ("other_water", "coral_reef"), "2201": ("other_water", "salt_pan"), "2202": ("pond", "aquaculture_pond"),
}


def wet_class(sub: str):
    s = (sub or "").strip()
    if s in WET_CODES:
        return WET_CODES[s]
    low = s.lower()
    if "aquaculture" in low:
        return ("pond", "aquaculture_pond")
    if "salt pan" in low:
        return ("other_water", "salt_pan")
    if "waterlogged" in low:
        return ("wet_land", "waterlogged")
    for key, cls in (("creek", "creek"), ("mud flat", "mud_flat"), ("tidal / saline flat", "mud_flat"),
                     ("mangrove", "mangrove"), ("salt marsh", "salt_marsh"), ("beach", "beach"),
                     ("coral", "coral_reef"), ("lagoon", "lagoon")):
        if key in low:
            return ("other_water", cls)
    return ("wet_land", "wetland")            # riverine wetland, "Other", "other wetland"


def crz_class(sub: str) -> str | None:
    s = (sub or "").upper().replace(" ", "")
    if "MARINE25KM" in s:
        return None                            # → other_water marine_buffer
    if "MANGROVEBUFFER" in s:
        return "crz_ia_mangrove_buffer"
    if s.startswith("CRZ-IVA"):
        return "crz_iva"
    if s.startswith("CRZ-IVB"):
        return "crz_ivb"
    if s.startswith("CRZ-IIIA"):
        return "crz_iiia"
    if s.startswith("CRZ-IIIB") or "200TO500" in s:
        return "crz_iiib"
    if s.startswith("CRZ-III"):
        return "crz_iii_ndz"
    if s.startswith("CRZ-II"):
        return "crz_ii"
    if s.startswith("CRZ-IB"):
        return "crz_ib"
    if s.startswith("CRZ-IA") or "ECOLOGICALLYSENSITIVE" in s:
        return "crz_ia"
    return "crz_ia"


def norm(key, g, cls_series, name_col="name", keep=("wb_subtype", "geom_source", "precision_class", "licence_status")):
    g = g.copy()
    g["_cls"] = cls_series.values
    n = reference.normalise(g, key, class_field="_cls", name_field=name_col if name_col in g else None,
                            keep_fields=[k for k in keep if k in g.columns])
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--version", default="water-master-2026-10-08")
    a = ap.parse_args()
    src = Path(a.path)
    with tempfile.TemporaryDirectory() as td:
        if src.suffix == ".zip":
            with zipfile.ZipFile(src) as z:
                z.extract("gujarat_water_master.gpkg", td)
            src = Path(td) / "gujarat_water_master.gpkg"
        L = lambda name: pyogrio.read_dataframe(src, layer=name)          # noqa: E731
        cov = L("Gujarat boundary").geometry.union_all()
        v = a.version
        prov = {"source_file": "gujarat_water_master.gpkg", "source_date": "2026-10-08", "model": None,
                "model_run_date": None, "confidence": None}

        def write(key, frames, source):
            frames = [f for f in frames if f is not None and len(f)]
            g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=4326) if frames \
                else gpd.GeoDataFrame({"class": []}, geometry=[], crs=4326)
            reference.write_version(key, g, v, {**prov, "source": source}, coverage=cov)
            print(f"{key}: {len(g)} features {g['class'].value_counts().to_dict() if len(g) else {}}")

        rivers = L("Rivers")
        is_res = rivers["wb_subtype"].astype(str) == "Reservoir"
        write("river", [norm("river", rivers[~is_res], pd.Series("river", index=rivers[~is_res].index))],
              "Water master (8 Oct 2026): Rivers — GatiShakti re-harvested + NWIA 2019 riverine + CWC")
        res = L("Reservoirs")
        write("reservoir", [norm("reservoir", res, pd.Series("reservoir", index=res.index)),
                            norm("reservoir", rivers[is_res], pd.Series("reservoir", index=rivers[is_res].index))],
              "Water master (8 Oct 2026): Reservoirs (OSM, CWC, GatiShakti, max observed extent) + GatiShakti river 'Reservoir' polygons")
        can = L("Canals")
        write("canal", [norm("canal", can, pd.Series("canal", index=can.index))],
              "Water master (8 Oct 2026): Canals — GatiShakti re-harvested")
        nal = L("Nallas (order 3-4)")
        write("nalla", [norm("nalla", nal, pd.Series("nalla", index=nal.index))],
              "Water master (8 Oct 2026): Nallas, Strahler order 3-4 — SISDP 1:10k + WRIS 1:50k")
        lak = L("Lakes")
        write("lake", [norm("lake", lak, pd.Series("lake", index=lak.index))],
              "Water master (8 Oct 2026): Lakes — OSM, WRIS, NWIA 2019, GatiShakti")
        wet = L("Wetlands")
        wc = wet["wb_subtype"].astype(str).map(wet_class)
        wkey = wc.map(lambda t: t[0])
        wcls = wc.map(lambda t: t[1])
        pf = L("Ponds-Tanks (footprints)")
        pp = L("Ponds-Tanks (points)")
        aq = wet[wkey == "pond"]
        write("pond", [norm("pond", pf, pd.Series("pond", index=pf.index)),
                       norm("pond", pp, pd.Series("pond", index=pp.index)),
                       norm("pond", aq, wcls[wkey == "pond"])],
              "Water master (8 Oct 2026): Ponds-Tanks footprints (GatiShakti) + Jal Dharohar census points + NWIA aquaculture ponds")
        write("wet_land", [norm("wet_land", wet[wkey == "wet_land"], wcls[wkey == "wet_land"])],
              "Water master (8 Oct 2026): Wetlands (NWIA 2019) — inland waterlogged, riverine and other wetland")
        crz = L("CRZ")
        cc = crz["wb_subtype"].astype(str).map(crz_class)
        marine = crz[cc.isna()]
        write("other_water", [norm("other_water", wet[wkey == "other_water"], wcls[wkey == "other_water"]),
                              norm("other_water", marine, pd.Series("marine_buffer", index=marine.index))],
              "Water master (8 Oct 2026): Wetlands (NWIA 2019) coastal classes + marine 25 km screening buffer")
        write("crz", [norm("crz", crz[cc.notna()], cc[cc.notna()])],
              "Water master (8 Oct 2026): CRZ zones, CZMP 2019 (GCZMA signed sheets, digitised)")
        dl, dp = L("Dams - lines"), L("Dams - points")
        write("dams", [norm("dams", dl, pd.Series("dam", index=dl.index)), norm("dams", dp, pd.Series("dam", index=dp.index))],
              "Water master (8 Oct 2026): Dams — OSM dam axes + NRLD/NDSA register points")


if __name__ == "__main__":
    main()
