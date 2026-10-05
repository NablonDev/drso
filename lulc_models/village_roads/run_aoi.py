"""VILLAGE ROADS plugin: Microsoft RoadDetections res-unet-6 (HF_GG08_60) on Esri Wayback z18.

    <roaddet venv>/bin/python lulc_models/village_roads/run_aoi.py --request request.json
    <roaddet venv>/bin/python lulc_models/village_roads/run_aoi.py --check --request request.json

Steps (contract: lulc_models/CONTRACT.md):
  1. AOI -> z18 tile mask, dilated by params.buffer_m, -> 16x16-tile blocks on the global grid.
  2. Tile presence scan over every MBTiles in imagery.esri_wayback_z18; blocks with no tile
     in their core are dropped ("no imagery") and stay out of the coverage.
  3. GPU: per block, (16+2*2)^2 tile mosaic -> 4-scale union + native probability COGs.
     Blocks whose core tiles are all missing/black after decoding are skipped.
  4. CPU (spawned workers, overlapped with 3): a block is vectorised once it is inferred and
     none of its 8 neighbours is still pending. Neighbours that will never be inferred (out of
     plan, no imagery, failed) count as final, so nothing ever waits forever.
  5. Merge, reproject to EPSG:4326 and clip to the AOI -> out.gpkg layer `features`.
  6. result.json: coverage = valid imagery tiles of inferred blocks, clipped to the AOI.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
import sysconfig
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))           # lulc_models/common.py

DEFAULTS = {"scales": [1.0, 1.5, 2.01, 3.0], "threshold": 0.2, "width_threshold": 0.5,
            "buffer_m": 2400, "max_gap_m": 250.0, "halo_px": 512, "min_px": 500,
            "batch_size": 4, "vec_workers": 4, "keep_native": False, "max_failed_frac": 0.05}
MODEL = "microsoft-roaddetections-res-unet-6"
GDAL_TOOLS = ["gdal_translate", "ogr2ogr", "ogrinfo"]


# ---- environment ----------------------------------------------------------------------------
def nvidia_lib_dirs():
    sp = sysconfig.get_paths()["purelib"]
    return sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))


def ensure_cuda_libs():
    """The dynamic loader reads LD_LIBRARY_PATH only at process start, so put this
    interpreter's pip-installed CUDA libs on it and re-exec once (same PID / process group)."""
    if os.environ.get("VILLAGE_ROADS_REEXEC") == "1":
        return
    dirs = nvidia_lib_dirs()
    cur = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    if not dirs or all(d in cur for d in dirs):
        return
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(dirs + cur), VILLAGE_ROADS_REEXEC="1")
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


def params_of(req):
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (req.get("params") or {}).items() if v is not None})
    return p


def resolve(path):
    return str(Path(path).expanduser().resolve()) if path else path


# ---- --check --------------------------------------------------------------------------------
def check(req_path) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never touch the GPU here
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    rep = {"plugin": "village_roads", "python": sys.version.split()[0], "executable": sys.executable,
           "packages": {}, "nvidia_lib_dirs": len(nvidia_lib_dirs()), "weights": {}, "mbtiles": [],
           "gdal_tools": {}, "errors": []}
    import importlib
    for m in ["numpy", "scipy", "PIL", "rasterio", "skimage", "networkx", "sknw", "tensorflow", "tf_keras"]:
        try:
            mod = importlib.import_module(m)
            rep["packages"][m] = getattr(mod, "__version__", "ok")
        except Exception as e:
            rep["packages"][m] = None
            rep["errors"].append("import %s: %s" % (m, str(e).splitlines()[0] if str(e) else type(e).__name__))
    try:
        req = json.loads(Path(req_path).read_text())
    except Exception as e:
        rep["errors"].append("request: %s" % e); print(json.dumps(rep, indent=1)); return 1
    ck = resolve((req.get("weights") or {}).get("checkpoint"))
    w = {"path": ck, "exists": bool(ck) and os.path.isfile(ck)}
    if w["exists"]:
        try:
            with open(ck, "rb") as f:
                head = f.read(2)
            w["size_mb"] = round(os.path.getsize(ck) / 2**20, 1)
            w["readable"] = True
            w["pickle"] = head[:1] == b"\x80"
            if not w["pickle"]:
                rep["errors"].append("weights do not look like a pickle: %s" % ck)
        except Exception as e:
            w["readable"] = False; rep["errors"].append("weights unreadable: %s" % e)
    else:
        rep["errors"].append("weights missing: %s" % ck)
    rep["weights"] = w
    plan = None
    try:
        from aoi_grid import plan_blocks
        plan = plan_blocks(req["aoi"], 0)
        rep["aoi_tiles"] = int(plan.aoi_mask.sum())
    except Exception as e:
        rep["errors"].append("aoi: %s" % e)
    items = (req.get("imagery") or {}).get("esri_wayback_z18") or []
    if not items:
        rep["errors"].append("no esri_wayback_z18 imagery in the request")
    from mbtiles import Source
    for i, it in enumerate(items):
        e = {"path": it.get("path")}
        try:
            s = Source(it, i)
            e.update(opens=True, release=s.release, date=s.date, format=s.meta.get("format"),
                     scheme="xyz" if s.xyz else "tms")
            e["z18_tiles"] = s.con.execute("select count(*) from tiles where zoom_level=18").fetchone()[0]
            if plan is not None:
                n = 0
                for c, y in s.tiles(plan.c0, plan.c0 + plan.w - 1, plan.r0, plan.r0 + plan.h - 1, False):
                    n += bool(plan.aoi_mask[y - plan.r0, c - plan.c0])
                e["z18_tiles_in_aoi"] = n
                if n == 0:
                    rep["errors"].append("no z18 tiles intersect the AOI: %s" % it.get("path"))
            s.close()
        except Exception as ex:
            e["opens"] = False; rep["errors"].append("mbtiles %s: %s" % (it.get("path"), ex))
        rep["mbtiles"].append(e)
    for t in GDAL_TOOLS:
        rep["gdal_tools"][t] = shutil.which(t)
        if not rep["gdal_tools"][t]:
            rep["errors"].append("GDAL tool missing: %s" % t)
    if all(rep["gdal_tools"].values()):
        rep["ogr_clip_geos"] = _ogr_clip_selftest(rep)
    rep["ok"] = not rep["errors"]
    print(json.dumps(rep, indent=1))
    return 0 if rep["ok"] else 1


def _ogr_clip_selftest(rep):
    import tempfile
    from outputs import ogr_clip, write_aoi, write_json
    with tempfile.TemporaryDirectory() as d:
        try:
            write_aoi(d + "/a.geojson", {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]})
            write_json(d + "/l.geojson", {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                       "geometry": {"type": "LineString", "coordinates": [[-1, 0.5], [2, 0.5]]}}]})
            f = ogr_clip(d + "/l.geojson", d + "/a.geojson", d + "/o.geojson", s_srs="EPSG:4326")
            xs = [c[0] for c in f[0]["geometry"]["coordinates"]]
            ok = abs(min(xs)) < 1e-9 and abs(max(xs) - 1) < 1e-9
        except Exception as e:
            ok = False; rep["errors"].append("ogr2ogr -clipdst self-test failed: %s" % e)
            return ok
    if not ok:
        rep["errors"].append("ogr2ogr -clipdst returned a wrong geometry (GEOS missing?)")
    return ok


# ---- run ------------------------------------------------------------------------------------
class Scheduler:
    """Tracks block state and releases blocks for vectorisation under the 8-neighbour rule."""

    def __init__(self, blocks):
        self.state = {b: "pending" for b in blocks}       # pending | done | skip | failed
        self.queued = set()

    def final(self, b):
        return self.state.get(b, "absent") != "pending"

    def ready(self, around):
        from aoi_grid import neighbours
        out = []
        for b in around:
            if self.state.get(b) == "done" and b not in self.queued \
                    and all(self.final(n) for n in neighbours(*b)):
                self.queued.add(b); out.append(b)
        return out


def run(req) -> None:
    import numpy as np
    import common
    import aoi_grid as G
    import blockvec
    from mbtiles import TileSources
    import outputs as OUT

    t0 = time.time()
    p = params_of(req)
    wd = Path(req["work_dir"]).resolve()
    d_u = wd / "artifacts" / "prob_u"; d_n = wd / "prob_n"; d_v = wd / "vec"; d_b = wd / "blocks"
    for d in (d_u, d_n, d_v, d_b):
        d.mkdir(parents=True, exist_ok=True)
    warnings = []
    ck = resolve((req.get("weights") or {}).get("checkpoint"))
    model_version = Path(ck).stem if ck else "HF_GG08_60"
    aoi_geom = G.aoi_geometry(req["aoi"])
    aoi_path = str(wd / "aoi_4326.geojson"); OUT.write_aoi(aoi_path, aoi_geom)

    def finish(rows, cov, counts, srcs_used=(), artifacts=()):
        w = OUT.write_gpkg(req["out_gpkg"], rows)
        if w:
            warnings.append(w)
        rel = sorted({s.release for s in srcs_used if s.release})
        dates = sorted({s.date for s in srcs_used if s.date})
        src_txt = ("Esri Wayback z18 release %s" % rel[0] if len(rel) == 1 else
                   "Esri Wayback z18 releases %s" % ", ".join(rel) if rel else
                   "Esri Wayback z18" if srcs_used else None)
        counts["features"] = len(rows)
        counts["coverage_km2"] = round(OUT.multipolygon_area_km2(cov), 2)
        counts["elapsed_s"] = round(time.time() - t0, 1)
        common.write_result(req, coverage=cov, image_source=src_txt, image_dates=dates, model=MODEL,
                            model_version=model_version, counts=counts, warnings=warnings,
                            artifacts={"prob_raster": list(artifacts)} if artifacts else {})
        common.progress(1.0, "done: %d features" % len(rows))

    ld = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    common.log("interpreter %s, CUDA lib dirs on LD_LIBRARY_PATH: %d/%d"
               % (sys.executable, sum(d in ld for d in nvidia_lib_dirs()), len(nvidia_lib_dirs())))
    # 1. plan
    common.progress(0.0, "planning blocks")
    plan = G.plan_blocks(req["aoi"], float(p["buffer_m"]))
    planned = plan.blocks()
    common.log("plan: grid c0=%d r0=%d %dx%d tiles, %d blocks (buffer %.0f m)"
               % (plan.c0, plan.r0, plan.w, plan.h, len(planned), float(p["buffer_m"])))
    items = (req.get("imagery") or {}).get("esri_wayback_z18") or []
    counts = {"blocks_planned": len(planned)}
    if not items:
        warnings.append("no Esri Wayback z18 imagery for the AOI: empty output")
        return finish([], OUT.EMPTY_COVERAGE, counts)

    # 2. presence
    src = TileSources(items)
    warnings += ["imagery source unusable: %s" % e for e in src.errors]
    if not src.sources:
        return finish([], OUT.EMPTY_COVERAGE, counts)
    pres = src.presence(plan)
    cand = [b for b in planned if pres[plan.block_slice(*b)].any()]
    counts["blocks_no_imagery"] = len(planned) - len(cand)
    if counts["blocks_no_imagery"]:
        warnings.append("%d of %d blocks have no imagery tile and were not processed"
                        % (counts["blocks_no_imagery"], len(planned)))
    common.log("imagery: %d sources, %d of %d blocks have tiles" % (len(src.sources), len(cand), len(planned)))
    common.progress(0.02, "%d blocks to infer" % len(cand))
    if not cand:
        return finish([], OUT.EMPTY_COVERAGE, counts)

    # 3+4. inference with overlapped vectorisation
    import multiprocessing as mp
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    sched = Scheduler(cand)
    valid = np.zeros((plan.h, plan.w), bool)
    used = {}
    nfail = 0; nskip = 0; nbr = 0; vdone = 0; inferred = 0
    futures = {}
    pool = ProcessPoolExecutor(max_workers=max(1, int(p["vec_workers"])), mp_context=mp.get_context("spawn"),
                               initializer=blockvec.init, initargs=(p,))
    model = None
    nat_alive = set()
    vec_ok = set()

    def submit(bl):
        for b in bl:
            futures[pool.submit(blockvec.vec_job, (b[0], b[1], str(d_u), str(d_n), str(d_v)))] = b

    def prune_native(b):
        if p["keep_native"]:
            return
        for nb in [b] + G.neighbours(*b):
            if nb not in nat_alive:
                continue
            if all(sched.state.get(x) != "pending" and (sched.state.get(x) != "done" or x in vec_ok)
                   for x in [nb] + G.neighbours(*nb)):
                f = d_n / ("n_%d_%d.tif" % nb)
                if f.exists():
                    f.unlink()
                nat_alive.discard(nb)

    def collect():
        nonlocal nbr, vdone
        for f in [f for f in list(futures) if f.done()]:
            b = futures.pop(f)
            tag, k, nb = f.result()          # a vectorisation error is fatal (deterministic code)
            nbr += nb; vdone += 1; vec_ok.add(b); prune_native(b)

    try:
        for i, b in enumerate(cand):
            tag = "%d_%d" % b
            fu = d_u / ("u_%s.tif" % tag); fv = d_b / ("%s.npz" % tag)
            if fu.exists() and fv.exists():                 # resume
                z = np.load(fv); v = z["valid"]; uz = json.loads(str(z["used"]))
                st = "done" if v.any() else "skip"
            else:
                mos, v, uz = src.mosaic(*b)
                st = "done" if v.any() else "skip"
                if st == "done":
                    if model is None:
                        common.progress(0.02, "loading model")
                        vram = common.gpu_env(req)
                        from blockinfer import load_model
                        model = load_model(ck, vram)
                    try:
                        from blockinfer import predict_block, write_block
                        un, na = predict_block(model, mos, scales=[float(s) for s in p["scales"]],
                                               batch_size=int(p["batch_size"]))
                        write_block(b[0], b[1], un, na, str(d_u), str(d_n))
                    except SystemExit:
                        raise
                    except Exception as e:
                        nfail += 1; st = "failed"
                        common.log("block %s FAILED: %r" % (tag, e))
                        if nfail > max(3, float(p["max_failed_frac"]) * len(cand)):
                            raise RuntimeError("too many failed blocks (%d), last: %r" % (nfail, e))
                del mos
                if st == "done":
                    np.savez(fv, valid=v, used=json.dumps({str(k): n for k, n in uz.items()}))
            if st == "skip":
                nskip += 1
            if st == "done":
                inferred += 1; nat_alive.add(b)
                valid[plan.block_slice(*b)] |= v
                for k, n in uz.items():
                    used[int(k)] = used.get(int(k), 0) + n
            sched.state[b] = st
            submit(sched.ready([b] + G.neighbours(*b)))
            collect()
            if (i + 1) % 5 == 0 or i + 1 == len(cand):
                el = time.time() - t0
                common.progress(0.03 + 0.82 * (i + 1) / len(cand),
                                "inferred %d/%d blocks, vectorised %d" % (i + 1, len(cand), vdone))
                common.log("[%d/%d] %s  %.1fs/block  vectorised %d  failed %d"
                           % (i + 1, len(cand), tag, el / (i + 1), vdone, nfail))
        if model is not None:
            del model
        submit(sched.ready(list(sched.state)))              # all final now
        n_left = len(futures)
        while futures:
            wait(list(futures), timeout=5, return_when=FIRST_COMPLETED)
            collect()
            common.progress(0.85 + 0.07 * (1 - len(futures) / max(n_left, 1)),
                            "vectorised %d/%d" % (vdone, inferred))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        for pr in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                pr.terminate()
            except Exception:
                pass
        src.close()
    if nskip:
        warnings.append("%d blocks had only missing/black tiles and were skipped" % nskip)
    if nfail:
        warnings.append("%d blocks failed inference and are excluded from coverage" % nfail)

    # 5. merge + clip
    common.progress(0.93, "merging and clipping")
    done = [b for b, s in sched.state.items() if s == "done"]
    feats = OUT.merge_blocks(str(d_v), done)
    clipped = OUT.clip_lines(feats, aoi_path, str(wd))
    rows = OUT.feature_rows(clipped)
    common.progress(0.97, "coverage")
    cov = OUT.coverage(valid, plan.transform, aoi_path, str(wd))
    counts.update(blocks_with_imagery=len(cand), blocks_inferred=inferred, blocks_skipped_black=nskip,
                  blocks_failed=nfail, edges_unclipped=len(feats), bridges=nbr,
                  length_km=round(sum(r[1]["length_m"] or 0 for r in rows) / 1000, 2))
    srcs_used = [s for s in src.sources if used.get(s.idx)]
    artifacts = sorted(str(d_u / ("u_%d_%d.tif" % b)) for b in done)
    finish(rows, cov, counts, srcs_used, artifacts)


def main():
    if "--check" in sys.argv:
        if "--request" not in sys.argv:
            raise SystemExit("usage: run_aoi.py --check --request request.json")
        sys.exit(check(sys.argv[sys.argv.index("--request") + 1]))
    ensure_cuda_libs()
    import common
    req = common.load_request()
    try:
        run(req)
    except SystemExit:
        raise
    except BaseException as e:
        import traceback
        traceback.print_exc()
        print("village_roads failed: %r" % e, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
