"""`run_aoi.py --request r.json --check`: readiness report without loading weights or touching
the GPU. Verifies imports, the checkpoint file, the Hugging Face cache for the DINOv3 backbone,
and that every registered imagery file opens and meets the AOI."""
from __future__ import annotations

import glob
import importlib
import os
from pathlib import Path

HF_REPO_DEFAULT = "facebook/dinov3-vitl16-pretrain-lvd1689m"


def _add(checks, name, ok, detail="", level=None):
    checks.append({"name": name, "ok": bool(ok), "level": level or ("ok" if ok else "fail"),
                   "detail": detail})


def hf_snapshot(hf_home: str, repo: str):
    """Return the snapshot dir holding config.json + *.safetensors for `repo`, or None."""
    if not hf_home:
        return None
    key = "models--" + repo.replace("/", "--")
    for hub in (Path(hf_home) / "hub", Path(hf_home)):
        for snap in sorted(glob.glob(str(hub / key / "snapshots" / "*"))):
            p = Path(snap)
            if (p / "config.json").exists() and (list(p.glob("*.safetensors")) or
                                                 list(p.glob("*.bin"))):
                return p
    return None


def run_check(req: dict, resolve, aoi4326) -> dict:
    checks = []
    # imports (no weights, no CUDA initialisation)
    for mod in ("numpy", "scipy", "rasterio", "geopandas", "shapely", "pyogrio", "pyproj",
                "torch", "transformers", "terratorch", "terratorch.tasks", "custom_modules"):
        try:
            importlib.import_module(mod)
            _add(checks, "import " + mod, True)
        except Exception as e:  # noqa: BLE001
            _add(checks, "import " + mod, False, repr(e)[:300])
    repo = HF_REPO_DEFAULT
    try:
        import custom_modules.terratorch_dinov3_hf_backbone as hb
        repo = hb.DEFAULT_CHECKPOINT
    except Exception:  # noqa: BLE001
        pass

    w = req.get("weights") or {}
    ck = resolve(w.get("checkpoint"))
    if ck and Path(ck).is_file():
        with open(ck, "rb") as f:
            magic = f.read(4)
        size = Path(ck).stat().st_size
        _add(checks, "checkpoint", magic == b"PK\x03\x04" and size > 1e8,
             "%s (%.0f MB, %s)" % (ck, size / 2**20,
                                    "torch zip" if magic == b"PK\x03\x04" else "unexpected header"))
    else:
        _add(checks, "checkpoint", False, "missing: %s" % ck)
    hf = resolve(w.get("hf_home"))
    snap = hf_snapshot(hf, repo)
    _add(checks, "hf_cache " + repo, snap is not None, str(snap or "not found under %s" % hf))

    from shapely.geometry import box, shape
    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform_bounds
    import sub_mbtiles

    img = req.get("imagery") or {}
    aoi3857 = gpd.GeoSeries([aoi4326], crs=4326).to_crs(3857).iloc[0]
    wb = img.get("esri_wayback_z18") or []
    if not wb:
        _add(checks, "esri_wayback_z18", False, "no MBTiles registered for the AOI")
    for e in wb:
        p = resolve(e.get("path"))
        try:
            with rasterio.open(p) as ds:
                crs_ok = ds.crs is not None and ds.crs.to_epsg() == 3857 and ds.count >= 3
                info = "%dx%d bands=%d %s" % (ds.width, ds.height, ds.count, ds.crs)
            fp, n = sub_mbtiles.footprint_3857(p, aoi3857.bounds)
            inter = fp.intersection(aoi3857).area / 1e6 if n else 0.0
            _add(checks, "mbtiles %s" % Path(p).name, crs_ok and inter > 0,
                 "%s; %d tiles in AOI bbox; data ∩ AOI = %.2f km2 (mercator)" % (info, n, inter))
        except Exception as ex:  # noqa: BLE001
            _add(checks, "mbtiles %s" % p, False, repr(ex)[:300])

    def raster_meets(kind, entries, required):
        if not entries:
            _add(checks, kind, not required, "none registered for the AOI",
                 level="warn" if not required else "fail")
            return
        for e in entries:
            p = resolve(e.get("path"))
            try:
                with rasterio.open(p) as ds:
                    b = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
                    meets = box(*b).intersects(aoi4326)
                    if e.get("footprint"):
                        fp = e["footprint"]
                        fp = shape(fp.get("geometry", fp) if fp.get("type") == "Feature" else fp)
                        meets = meets and fp.intersects(aoi4326)
                    _add(checks, "%s %s" % (kind, Path(p).name), meets,
                         "%s %s res=%s %s" % (e.get("scene_id") or "", e.get("date") or "",
                                              tuple(round(r, 2) for r in ds.res), ds.crs))
            except Exception as ex:  # noqa: BLE001
                _add(checks, "%s %s" % (kind, p), False, repr(ex)[:300])

    # S1 is optional for the run (detections become low-confidence review items without it)
    raster_meets("sentinel1_rtc_vv", img.get("sentinel1_rtc_vv") or [], required=False)
    raster_meets("nisar_coh", img.get("nisar_coh") or [], required=False)

    osm = resolve((req.get("inputs") or {}).get("osm_substations_gpkg"))
    if osm:
        try:
            import pyogrio
            layers = [l[0] for l in pyogrio.list_layers(osm)]
            _add(checks, "osm_substations_gpkg", True, "%s layers=%s" % (osm, layers))
        except Exception as ex:  # noqa: BLE001
            _add(checks, "osm_substations_gpkg", False, repr(ex)[:300])
    else:
        _add(checks, "osm_substations_gpkg", True, "not given: no OSM confirmation / names",
             level="warn")

    ok = not any(c["level"] == "fail" for c in checks)
    return {"plugin": "substations", "ok": ok, "checks": checks}
