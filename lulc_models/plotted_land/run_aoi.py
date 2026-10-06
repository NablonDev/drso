"""Plotted-land plugin (Beeline LULC contract, lulc_models/CONTRACT.md).

  AOI -> village roads clipped to AOI + roads_overlap_m -> road-probability VRT from the given tifs
      -> UTM zone of the AOI centroid -> geometry pipeline (chord variant, IDEAL) -> strict triplets
      -> core trim (cluster representative point inside the AOI) -> building coverage (run buildings layer)
      -> salt-works filter (inputs.saltscape_gpkg: delivered salt pans / mud flats / salt marsh / mangrove;
         skipped with a warning when none is supplied). No WorldCover since 6 Oct 2026.
      -> out.gpkg: `features` (clusters, EPSG:4326), `plots`, `removed_saltworks`

Same per-region steps as the statewide run /mnt/cog-data/guj/pl/code/guj_pl.py one() + guj_pl_byregion.py.

usage: run_aoi.py --request request.json
       run_aoi.py --check [--request request.json]     (environment / input report as JSON, no processing)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GDAL_CACHEMAX", "2048")

import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # lulc_models/common.py
sys.path.insert(0, str(HERE))

import common  # noqa: E402

MODEL = "plotted-land-geometry"
MODEL_VERSION = "guj chord IDEAL=1"
IMAGE_SOURCE = "derived from village roads"
DEFAULTS = {"ideal": 1, "salt_threshold": 0.60, "prob_accept": 0.20, "roads_overlap_m": 3000,
            "id_prefix": "PLC-", "debug": False}
FEATURE_ATTRS = ["n_plots", "area_ha", "built_pct", "built_class", "vacant_plots_pct"]


def gdalbuildvrt_bin():
    return shutil.which("gdalbuildvrt") or ("/usr/bin/gdalbuildvrt" if os.access("/usr/bin/gdalbuildvrt", os.X_OK) else None)


def is_remote(p: str) -> bool:
    return str(p).startswith("/vsi") or "://" in str(p)


def utm_epsg(lon: float, lat: float) -> int:
    zone = min(60, int((lon + 180.0) // 6.0) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def first_layer(path, prefer="features"):
    import pyogrio
    names = [l for l, _ in pyogrio.list_layers(path)]
    return prefer if prefer in names else names[0]


def build_vrt(paths, vrt, list_file):
    Path(list_file).write_text("\n".join(paths) + "\n")
    subprocess.run([gdalbuildvrt_bin(), "-q", "-input_file_list", list_file, vrt], check=True)
    return vrt


# ------------------------------------------------------------------------------------------- check
def check(req: dict | None) -> dict:
    checks = []
    def add(name, ok, detail="", required=True):
        checks.append({"name": name, "ok": bool(ok), "required": required, "detail": detail})
    mods = ["numpy", "pandas", "geopandas", "shapely", "rasterio", "pyproj", "networkx", "scipy", "pyogrio",
            "pipeline", "triplets", "deliver", "saltfilter"]
    bad = []
    for m in mods:
        try:
            __import__(m)
        except Exception as e:  # noqa: BLE001
            bad.append("%s: %s" % (m, e))
    add("imports", not bad, "; ".join(bad) or "%d modules" % len(mods))
    b = gdalbuildvrt_bin(); add("gdalbuildvrt", b is not None, b or "not found on PATH or /usr/bin")
    if req is not None:
        import pyogrio, rasterio, shapely.geometry
        try:
            shapely.geometry.shape(req["aoi"]); add("aoi", True, "")
        except Exception as e:  # noqa: BLE001
            add("aoi", False, str(e))
        inp = req.get("inputs") or {}
        rg = inp.get("village_roads_gpkg")
        try:
            i = pyogrio.read_info(rg, layer=first_layer(rg)); add("village_roads_gpkg", True, "%s: %d features, %s" % (rg, i["features"], i["crs"]))
        except Exception as e:  # noqa: BLE001
            add("village_roads_gpkg", False, "%s: %s" % (rg, e))
        tifs = inp.get("village_roads_prob") or []
        if isinstance(tifs, str):
            tifs = [tifs]
        bad = []
        for t in tifs:
            try:
                if is_remote(t):
                    raise ValueError("remote path")
                with rasterio.open(t) as s:
                    if s.crs is None or s.crs.to_epsg() != 3857 or s.dtypes[0] != "uint8":
                        raise ValueError("crs %s dtype %s (need EPSG:3857 Byte)" % (s.crs, s.dtypes[0]))
            except Exception as e:  # noqa: BLE001
                bad.append("%s: %s" % (t, e))
        add("village_roads_prob", len(tifs) > 0 and not bad,
            ("%d tifs, %d bad: %s" % (len(tifs), len(bad), "; ".join(bad[:5]))) if tifs else "no tifs given")
        bg = inp.get("buildings_gpkg")
        if bg:
            try:
                i = pyogrio.read_info(bg, layer=first_layer(bg)); add("buildings_gpkg", True, "%s: %d features" % (bg, i["features"]), required=False)
            except Exception as e:  # noqa: BLE001
                add("buildings_gpkg", False, "%s: %s" % (bg, e), required=False)
        else:
            add("buildings_gpkg", False, "not given: built_pct will be null", required=False)
        sg = inp.get("saltscape_gpkg")
        if sg:
            try:
                i = pyogrio.read_info(sg, layer="features")
                add("saltscape_gpkg", True, "%s: %d features" % (sg, i["features"]), required=False)
            except Exception as e:  # noqa: BLE001
                add("saltscape_gpkg", False, "%s: %s" % (sg, e), required=False)
        else:
            add("saltscape_gpkg", False, "not given: salt filter will be skipped", required=False)
    return {"ok": all(c["ok"] for c in checks if c["required"]), "checks": checks}


# --------------------------------------------------------------------------------------------- run
def to_multi(gs):
    import shapely
    from shapely.geometry import MultiPolygon, Polygon
    return gs.apply(lambda g: MultiPolygon([g]) if isinstance(g, Polygon) else g) if len(gs) else gs


def write_layer(gdf, path, layer, geometry_type):
    gdf.to_file(path, layer=layer, driver="GPKG", engine="pyogrio", geometry_type=geometry_type)


def run(req: dict):
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import pyogrio
    import rasterio
    import shapely
    from shapely.geometry import shape, box, mapping
    import pipeline, triplets, deliver as deliver_mod, saltfilter as salt_mod

    P = dict(DEFAULTS); P.update(req.get("params") or {})
    wd = Path(req["work_dir"]); out = req["out_gpkg"]; warnings = []
    inp = req.get("inputs") or {}
    log = common.log
    common.progress(0.0, "start")

    aoi = shapely.make_valid(shape(req["aoi"]))
    c = aoi.centroid; crs = utm_epsg(c.x, c.y)
    aoi_utm = gpd.GeoSeries([aoi], crs=4326).to_crs(crs).iloc[0]
    clip_utm = aoi_utm.buffer(float(P["roads_overlap_m"]))
    clip_4326 = gpd.GeoSeries([clip_utm], crs=crs).to_crs(4326).iloc[0]
    clip_3857 = gpd.GeoSeries([clip_utm], crs=crs).to_crs(3857).iloc[0]
    log("AOI %.2f km2, UTM EPSG:%d, roads overlap %.0f m" % (aoi_utm.area / 1e6, crs, P["roads_overlap_m"]))

    # ---- roads (EPSG:4326 from the roads plugin -> 3857 as in the statewide inputs)
    rg = inp.get("village_roads_gpkg")
    if not rg or not os.path.exists(rg):
        raise SystemExit("inputs.village_roads_gpkg missing: %s" % rg)
    roads = gpd.read_file(rg, layer=first_layer(rg), bbox=tuple(clip_4326.bounds), engine="pyogrio", columns=[])
    roads_info = pyogrio.read_info(rg, layer=first_layer(rg), force_total_bounds=True)
    roads = roads[roads.geometry.notna()]
    roads = roads.set_crs(4326, allow_override=False) if roads.crs is None else roads
    roads = roads.to_crs(3857).explode(index_parts=False)
    roads = roads[roads.geom_type == "LineString"]
    roads = roads.clip(clip_3857).explode(index_parts=False)
    roads = gpd.GeoDataFrame(geometry=roads[roads.geom_type == "LineString"].geometry.values, crs=3857)
    log("roads in AOI + overlap: %d edges" % len(roads))
    common.progress(0.05, "roads %d" % len(roads))

    # ---- probability VRT from the tifs that touch the clip region
    tifs = inp.get("village_roads_prob") or []
    if isinstance(tifs, str):
        tifs = [tifs]
    foot = []; use = []
    for t in tifs:
        if is_remote(t):
            raise SystemExit("remote prob raster not allowed: %s" % t)
        with rasterio.open(t) as s:
            if s.crs is None or s.crs.to_epsg() != 3857 or s.dtypes[0] != "uint8":
                raise SystemExit("prob raster %s is %s %s, need EPSG:3857 Byte" % (t, s.crs, s.dtypes[0]))
            bx = box(*s.bounds)
        foot.append(bx)
        if bx.intersects(clip_3857):
            use.append(t)
    ext = []
    if foot:
        ext.append(gpd.GeoSeries([shapely.union_all(foot)], crs=3857).to_crs(4326).iloc[0])
    tb = roads_info.get("total_bounds")
    if roads_info.get("features") and tb is not None and np.all(np.isfinite(tb)):
        ext.append(box(*tb))
    cov = aoi.intersection(shapely.union_all(ext)) if ext else None
    coverage = mapping(cov) if cov is not None and not cov.is_empty else None

    R = C = None
    if not use:
        warnings.append("no road-probability raster touches the AOI: plotted land not computed")
        log(warnings[-1])
    elif not len(roads):
        log("no roads in the AOI: empty output")
    else:
        vrt = build_vrt(use, str(wd / "prob.vrt"), str(wd / "prob_list.txt"))
        common.progress(0.08, "prob VRT %d tifs" % len(use))
        prog = lambda f, m: common.progress(0.10 + 0.65 * f, m)
        R, C = pipeline.run(roads, vrt, crs, ideal=int(P["ideal"]), p_accept=float(P["prob_accept"]), log=log,
                            progress=prog, debug_dir=str(wd / "debug") if P.get("debug") else None)
        common.progress(0.76, "triplets")
        R, C = triplets.add_triplets(R, C, log=log)
        # core trim: keep clusters whose representative point lies in the AOI
        if len(C):
            rp = C.geometry.representative_point()
            core = rp.within(aoi_utm).values
            C = C[core].copy(); R = R[R.cluster.isin(C.cluster)].copy()
        log("clusters in AOI core: %d" % len(C))
    if R is None:
        R, C = pipeline.empty_result(crs)

    common.progress(0.80, "building coverage")
    bg = inp.get("buildings_gpkg")
    if bg and not os.path.exists(bg):
        warnings.append("buildings_gpkg %s not found: built_pct is null" % bg); bg = None
    elif not bg:
        warnings.append("no buildings layer supplied: built_pct is null")
    CC, PP = deliver_mod.deliver(R, C, bg, id_prefix=str(P["id_prefix"]), log=log)

    common.progress(0.90, "salt-works filter")
    removed = CC.iloc[0:0].assign(ring_saltscape=pd.Series(dtype="float64"))
    sg = inp.get("saltscape_gpkg")
    if not sg or not os.path.exists(sg):
        warnings.append("no salt-scape layers supplied: salt-works filter skipped")
        log(warnings[-1])
    elif len(CC):
        import shapely
        salt = gpd.read_file(sg, layer="features", engine="pyogrio").to_crs(crs)
        cv = gpd.read_file(sg, layer="coverage", engine="pyogrio") if "coverage" in [l for l, _ in pyogrio.list_layers(sg)] else None
        cov = shapely.union_all(cv.to_crs(crs).geometry.values) if cv is not None and len(cv) else None
        CC, PP, removed, have = salt_mod.saltfilter(CC, PP, salt, cov, thr=float(P["salt_threshold"]), log=log)
        if have < 0.99:
            warnings.append("salt-scape layers cover only %.0f%% of the salt-filter rings" % (100 * have))

    # ---- outputs, EPSG:4326
    common.progress(0.96, "writing")
    if os.path.exists(out):
        os.remove(out)
    F = CC.to_crs(4326)
    feats = gpd.GeoDataFrame({"class": ["plotted_land"] * len(F), "confidence": pd.Series([None] * len(F), dtype="float64"),
                              "name": pd.Series([None] * len(F), dtype="object"), "src_id": F.cluster_id.values,
                              **{k: F[k].values for k in FEATURE_ATTRS}},
                             geometry=to_multi(F.geometry).values, crs=4326)
    write_layer(feats, out, "features", "MultiPolygon")
    write_layer(PP.to_crs(4326), out, "plots", "Polygon")
    rem = removed.to_crs(4326); rem = gpd.GeoDataFrame(rem.drop(columns="geometry"), geometry=to_multi(rem.geometry).values, crs=4326)
    write_layer(rem, out, "removed_saltworks", "MultiPolygon")
    counts = {"features": int(len(feats)), "plots": int(len(PP)), "removed_saltworks": int(len(rem)),
              "rectangles": int(len(R)), "roads_edges": int(len(roads)), "utm_epsg": int(crs)}
    common.write_result(req, coverage=coverage, image_source=IMAGE_SOURCE, image_dates=[], model=MODEL,
                        model_version=MODEL_VERSION, counts=counts, warnings=warnings, artifacts={})
    common.progress(1.0, "done: %d clusters, %d plots" % (len(feats), len(PP)))
    return counts


def main():
    if "--check" in sys.argv:
        req = None
        if "--request" in sys.argv:
            req = json.loads(Path(sys.argv[sys.argv.index("--request") + 1]).read_text())
        rep = check(req)
        print(json.dumps(rep, indent=1))
        sys.exit(0 if rep["ok"] else 1)
    req = common.load_request()          # installs the SIGTERM -> SystemExit("cancelled") handler
    try:
        run(req)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        print("plotted_land failed: %r" % e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
