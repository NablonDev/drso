"""Village-roads plugin tests - no network, no GPU, no TensorFlow inference.

pytest lives in the beeline env, the plugin runs in the roaddet venv, so this file is dual-use:
  * pytest:   /home/vivek_jain/miniforge3/envs/beeline/bin/python -m pytest tests/plugins/test_village_roads.py
  * helpers, run by the tests under the roaddet venv interpreter:
        <venv python> test_village_roads.py vec      <u_dir> <out_dir> <bx> <by>
        <venv python> test_village_roads.py fake_run <request.json> [sleep_s]
    fake_run drives run_aoi.main() end to end with the model replaced by a synthetic predictor
    (a '+' shaped road through every block centre, zeroed where the imagery is missing), so
    planning, MBTiles reading, scheduling, vectorisation, clipping and outputs are real.
Data: corridor MBTiles /mnt/drso-data/wayback/bhuj_lakadia_L18_2026-08-05.mbtiles and the
statewide outputs under /mnt/cog-data/guj (prob_u, vec). Tests skip when they are absent.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO / "lulc_models" / "village_roads"
RUN = PLUGIN_DIR / "run_aoi.py"
VENV_PY = os.environ.get("VR_TEST_PY", os.path.join(os.environ.get("BEELINE_ROADDET_VENV",
                                                                   "/mnt/cog-data/roaddet/venv"), "bin", "python"))
MB = "/mnt/drso-data/wayback/bhuj_lakadia_L18_2026-08-05.mbtiles"
WEIGHTS = "/mnt/cog-data/roaddet/HF_GG08_60.pkl"
GUJ = "/mnt/cog-data/guj"
PARITY_BLOCK = (184480, 114032)       # all 8 neighbours present in guj/prob_u; 102 edges in guj/vec
CORNER_FULL = (69.697265625, 23.463246331550348)    # block corner inside the corridor imagery
CORNER_EDGE = (69.58740234374999, 23.463246331550348)  # west footprint edge: 2 of 4 blocks empty
HALF = 0.015                                        # AOI half-size (deg) -> 2x2 blocks, buffer 0


# ============================================================ helpers run under the venv interpreter
def _helper_vec(u_dir, out_dir, bx, by):
    sys.path.insert(0, str(PLUGIN_DIR))
    import blockvec
    blockvec.init({"max_gap_m": 250.0})
    print(json.dumps(blockvec.vec_block(int(bx), int(by), u_dir, None, out_dir)))


def _helper_fake_run(req_path, sleep_s="0"):
    os.environ["VILLAGE_ROADS_REEXEC"] = "1"          # no re-exec into the real entry point
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(PLUGIN_DIR))
    import numpy as np
    import blockinfer
    delay = float(sleep_s)

    def fake_load_model(checkpoint, vram_mb=None):
        assert os.path.isfile(checkpoint)
        return "fake-model"

    def fake_predict(model, mos, scales=None, batch_size=4, halo=2):
        if delay:
            time.sleep(delay)
        core = mos[halo * 256:(halo + 16) * 256, halo * 256:(halo + 16) * 256]
        p = np.zeros((4096, 4096), np.float32)
        p[2040:2056, :] = 0.95
        p[:, 2040:2056] = 0.95
        p *= (core.max(axis=2) > 0)
        return p, p.copy()

    blockinfer.load_model = fake_load_model
    blockinfer.predict_block = fake_predict
    sys.argv = [str(RUN), "--request", req_path]
    import runpy
    try:
        # run as __main__ so spawned vectorisation workers re-import run_aoi.py (as in production),
        # not this test file
        runpy.run_path(str(RUN), run_name="__main__")
    finally:
        assert "tensorflow" not in sys.modules, "TensorFlow must not be imported by the fake run"


def _helper_predict_shapes(out_dir, bx, by):
    """predict_block/write_block with infer.predict_array stubbed (CPU only, no model)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(PLUGIN_DIR))
    import numpy as np
    import infer
    import blockinfer
    calls = []

    def stub(model, rgb, batch_size=4):
        calls.append(rgb.shape)
        return rgb.mean(axis=0).astype(np.float32)
    infer.predict_array = stub
    rng = np.random.default_rng(0)
    mos = rng.integers(0, 255, (20 * 256, 20 * 256, 3), dtype=np.uint8)
    un, na = blockinfer.predict_block(None, mos, scales=[1.0, 1.5, 2.01, 3.0])
    ref = mos[512:512 + 4096, 512:512 + 4096].astype(np.float32).mean(axis=2) / 255.0
    assert un.shape == na.shape == (4096, 4096)
    assert np.allclose(na, ref, atol=1e-6) and (un >= na - 1e-6).all()
    assert calls == [(3, 5120, 5120), (3, 3413, 3413), (3, 2547, 2547), (3, 1706, 1706)], calls
    blockinfer.write_block(int(bx), int(by), un, na, out_dir, out_dir)
    print("ok")


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    {"vec": _helper_vec, "fake_run": _helper_fake_run, "predict_shapes": _helper_predict_shapes}[mode](*args)
    sys.exit(0)


# ============================================================ tests (pytest, beeline env)
import pytest  # noqa: E402

sys.path.insert(0, str(PLUGIN_DIR))
import aoi_grid as G  # noqa: E402

need_venv = pytest.mark.skipif(not os.path.exists(VENV_PY), reason="roaddet venv absent")
need_mb = pytest.mark.skipif(not os.path.exists(MB), reason="corridor MBTiles absent")
need_guj = pytest.mark.skipif(not os.path.isdir(GUJ + "/prob_u"), reason="statewide outputs absent")
need_gdal = pytest.mark.skipif(not shutil.which("ogr2ogr"), reason="GDAL CLI absent")


def box(lon, lat, h=HALF):
    return {"type": "Polygon", "coordinates": [[[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h],
                                                [lon - h, lat + h], [lon - h, lat - h]]]}


def request(tmp, aoi, items=None, buffer_m=0, weights=WEIGHTS, **params):
    Path(tmp).mkdir(parents=True, exist_ok=True)
    wd = Path(tmp) / "wd"
    req = {"layer_key": "village_roads", "aoi": aoi, "work_dir": str(wd), "out_gpkg": str(wd / "out.gpkg"),
           "result_json": str(wd / "result.json"),
           "imagery": {"esri_wayback_z18": items if items is not None else
                       [{"path": MB, "release": "26334", "date": "2026-08-05"}]},
           "inputs": {}, "weights": {"checkpoint": weights}, "gpu": {"device": 0, "vram_mb": 2000},
           "params": dict({"buffer_m": buffer_m, "vec_workers": 4}, **params)}
    p = Path(tmp) / "request.json"
    p.write_text(json.dumps(req))
    return p, req


def fake_run(req_path, sleep_s=0, timeout=900):
    return subprocess.run([VENV_PY, __file__, "fake_run", str(req_path), str(sleep_s)], capture_output=True,
                          text=True, timeout=timeout, cwd=str(REPO))


def progress_values(stdout):
    return [float(m.group(1)) for m in re.finditer(r"^PROGRESS (\S+) ", stdout, re.M)]


def gpkg_rows(path):
    con = sqlite3.connect(path)
    cols = [r[1] for r in con.execute("PRAGMA table_info(features)")]
    rows = [dict(zip(cols, r)) for r in con.execute("SELECT * FROM features")]
    types = {r[1]: r[2] for r in con.execute("PRAGMA table_info(features)")}
    con.close()
    return rows, types


def ogr_features(path):
    out = Path(path).with_suffix(".check.geojson")
    subprocess.run(["ogr2ogr", "-f", "GeoJSON", str(out), str(path), "features"], check=True,
                   capture_output=True)
    return json.loads(out.read_text())["features"]


def ogr_layer_info(path):
    r = subprocess.run(["ogrinfo", "-json", "-so", str(path), "features"], check=True, capture_output=True, text=True)
    return json.loads(r.stdout)["layers"][0]


# ---- AOI -> blocks --------------------------------------------------------------------------
def test_plan_alignment_and_statewide_tags():
    bx, by = PARITY_BLOCK
    # a small AOI inside one statewide block resolves to exactly that block tag
    x0, y1 = -G.O + (bx + 5) * G.TM, G.O - (by + 5) * G.TM
    lon0, lat1 = G.x3857_to_lonlat(x0, y1); lon1, lat0 = G.x3857_to_lonlat(x0 + 3 * G.TM, y1 - 3 * G.TM)
    aoi = {"type": "Polygon", "coordinates": [[[float(lon0), float(lat0)], [float(lon1), float(lat0)],
                                               [float(lon1), float(lat1)], [float(lon0), float(lat1)],
                                               [float(lon0), float(lat0)]]]}
    p = G.plan_blocks(aoi, 0)
    assert p.c0 % 16 == 0 and p.r0 % 16 == 0 and p.h % 16 == 0 and p.w % 16 == 0
    assert p.blocks() == [PARITY_BLOCK]
    assert int(p.aoi_mask.sum()) in (9, 16)          # 3x3 tiles (4x4 if the edges touch)
    p2 = G.plan_blocks(aoi, 2400)
    assert all(b[0] % 16 == 0 and b[1] % 16 == 0 for b in p2.blocks())
    assert PARITY_BLOCK in p2.blocks() and len(p2.blocks()) > 1


def test_buffer_is_conservative_and_tight():
    lon, lat, buf = 70.0, 23.4, 2400.0
    p = G.plan_blocks(box(lon, lat, 0.005), buf)
    k = 1 / math.cos(math.radians(lat + 0.005))
    (bx0, by0), (bx1, by1) = [G.lonlat_to_3857(lon + s * 0.005, lat + s * 0.005) for s in (-1, 1)]
    bx0, by0, bx1, by1 = float(bx0), float(by0), float(bx1), float(by1)
    # every point within buffer_m (ground) of the AOI box falls in a kept tile
    for t in range(720):
        a = math.radians(t / 2)
        for frac in (0.999, 0.5):
            d = buf * frac * k                       # mercator metres
            cx = min(max(bx0, bx0 + (bx1 - bx0) * (0.5 + 0.5 * math.cos(a))), bx1)
            cy = min(max(by0, by0 + (by1 - by0) * (0.5 + 0.5 * math.sin(a))), by1)
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            c, r = int((x + G.O) // G.TM), int((G.O - y) // G.TM)
            assert p.buf_mask[r - p.r0, c - p.c0], (t, frac)
    # tiles whose centre is more than buffer + 2.5 tiles from the box are not kept
    rr, cc = __import__("numpy").nonzero(p.buf_mask)
    xs = -G.O + (cc + p.c0 + 0.5) * G.TM; ys = G.O - (rr + p.r0 + 0.5) * G.TM
    dx = __import__("numpy").maximum(0, __import__("numpy").maximum(bx0 - xs, xs - bx1))
    dy = __import__("numpy").maximum(0, __import__("numpy").maximum(by0 - ys, ys - by1))
    assert float(__import__("numpy").hypot(dx, dy).max()) <= buf * k + 2.5 * G.TM


def test_multipolygon_and_feature_aoi():
    a, b = box(70.0, 23.4, 0.003), box(70.2, 23.4, 0.003)
    mp = {"type": "MultiPolygon", "coordinates": [a["coordinates"], b["coordinates"]]}
    fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": a},
                                                     {"type": "Feature", "properties": {}, "geometry": b}]}
    assert G.plan_blocks(mp, 0).blocks() == G.plan_blocks(fc, 0).blocks()
    assert len(G.plan_blocks(mp, 0).blocks()) >= 2


# ---- MBTiles reading ------------------------------------------------------------------------
def _tile_rgb(path, c, y):
    import io
    import numpy as np
    from PIL import Image
    con = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    data = con.execute("select tile_data from tiles where zoom_level=18 and tile_column=? and tile_row=?",
                       (c, 2 ** 18 - 1 - y)).fetchone()[0]
    con.close()
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB")), data


def _mini_mbtiles(path, tiles, date):
    con = sqlite3.connect(path)
    con.execute("create table metadata (name text, value text)")
    con.execute("create table tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob)")
    con.executemany("insert into metadata values (?,?)", [("format", "jpg"), ("description", "test release 99999 (%s)" % date)])
    con.executemany("insert into tiles values (18,?,?,?)", [(c, 2 ** 18 - 1 - y, d) for c, y, d in tiles])
    con.commit(); con.close()


@need_mb
def test_mbtiles_mosaic_matches_source_tiles():
    from mbtiles import TileSources
    p = G.plan_blocks(box(*CORNER_FULL, 0.002), 0)
    src = TileSources([{"path": MB, "release": "26334", "date": "2026-08-05"}])
    pres = src.presence(p)
    assert pres[p.aoi_mask].all()
    bx, by = p.blocks()[0]
    mos, valid, used = src.mosaic(bx, by)
    assert mos.shape == (20 * 256, 20 * 256, 3) and valid.shape == (16, 16)
    assert valid.all() and used == {0: 256}
    for dx, dy in [(0, 0), (7, 11), (-2, -2), (17, 3)]:            # core and halo tiles
        ref, _ = _tile_rgb(MB, bx + dx, by + dy)
        got = mos[(dy + 2) * 256:(dy + 3) * 256, (dx + 2) * 256:(dx + 3) * 256]
        assert (got == ref).all(), (dx, dy)
    assert src.sources[0].release == "26334" and src.sources[0].date == "2026-08-05"
    src.close()


@need_mb
def test_mbtiles_black_and_missing_tiles_fall_through(tmp_path):
    import io
    import numpy as np
    from PIL import Image
    from mbtiles import TileSources
    p = G.plan_blocks(box(*CORNER_EDGE, 0.002), 0)
    blocks = p.blocks()
    one = TileSources([{"path": MB, "date": "2026-08-05"}])
    counts = {b: int(one.mosaic(*b)[1].sum()) for b in blocks}
    partial = [b for b, n in counts.items() if 0 < n < 256]
    empty = [b for b, n in counts.items() if n == 0]
    assert partial and empty
    bx, by = partial[0]
    mos1, v1, _ = one.mosaic(bx, by)
    hole = [(c, r) for r in range(16) for c in range(16) if not v1[r, c]][0]
    have = [(c, r) for r in range(16) for c in range(16) if v1[r, c]][0]
    _, real = _tile_rgb(MB, bx + have[0], by + have[1])
    blk = io.BytesIO(); Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(blk, "JPEG")
    newer = tmp_path / "newer.mbtiles"
    # newer source: a black tile over a valid one (must fall through) + a real tile in a hole
    _mini_mbtiles(newer, [(bx + have[0], by + have[1], blk.getvalue()),
                          (bx + hole[0], by + hole[1], real)], "2027-01-01")
    two = TileSources([{"path": MB, "date": "2026-08-05"}, {"path": str(newer)}])
    assert [s.idx for s in two.sources] == [1, 0]                    # newest first
    assert two.sources[0].date == "2027-01-01" and two.sources[0].release == "99999"
    mos2, v2, used = two.mosaic(bx, by)
    assert int(v2.sum()) == int(v1.sum()) + 1 and used == {0: int(v1.sum()), 1: 1}
    sl = lambda m, t: m[(t[1] + 2) * 256:(t[1] + 3) * 256, (t[0] + 2) * 256:(t[0] + 3) * 256]
    assert (sl(mos2, have) == sl(mos1, have)).all()                  # black tile ignored
    assert sl(mos2, hole).max() > 0
    one.close(); two.close()


# ---- vectorisation + merge + clip on existing statewide probability blocks ------------------
@pytest.fixture(scope="module")
def parity_vec(tmp_path_factory):
    if not (os.path.exists(VENV_PY) and os.path.isdir(GUJ + "/prob_u")):
        pytest.skip("venv or statewide outputs absent")
    d = tmp_path_factory.mktemp("vec")
    u, v = d / "u", d / "v"; u.mkdir(); v.mkdir()
    bx, by = PARITY_BLOCK
    for b in [PARITY_BLOCK] + G.neighbours(bx, by):
        shutil.copy("%s/prob_u/u_%d_%d.tif" % (GUJ, *b), u)
    r = subprocess.run([VENV_PY, __file__, "vec", str(u), str(v), str(bx), str(by)], capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return d


@need_venv
@need_guj
def test_vectorisation_reproduces_statewide_block(parity_vec):
    tag = "%d_%d" % PARITY_BLOCK
    mine = json.loads((parity_vec / "v" / (tag + ".geojson")).read_text())["features"]
    ref = json.loads(Path("%s/vec/%s.geojson" % (GUJ, tag)).read_text())["features"]
    assert len(mine) == len(ref) > 50
    assert all(a["geometry"] == b["geometry"] for a, b in zip(mine, ref))
    assert all(a["properties"]["length_m"] == b["properties"]["length_m"] for a, b in zip(mine, ref))
    assert all(a["properties"]["bridged"] == b["properties"]["bridged"] for a, b in zip(mine, ref))
    # prob_n was pruned from the statewide run, so widths are null here (measured there)
    assert all(a["properties"]["surface_width_m"] is None and a["properties"]["lane_tier"] is None for a in mine)


@need_venv
@need_guj
@need_gdal
def test_merge_clip_and_gpkg_schema(parity_vec, tmp_path):
    import numpy as np
    import outputs as OUT
    bx, by = PARITY_BLOCK
    east = (bx + 16, by)
    shutil.copy("%s/vec/%d_%d.geojson" % (GUJ, *east), parity_vec / "v")
    feats = OUT.merge_blocks(str(parity_vec / "v"), [east, PARITY_BLOCK])
    assert [f["properties"]["src_id"] for f in feats][:2] == ["vr_%d_%d_0" % PARITY_BLOCK, "vr_%d_%d_1" % PARITY_BLOCK]
    # AOI: triangle across both blocks, so edges are cut
    x0, y0 = -G.O + (bx + 4) * G.TM, G.O - (by + 2) * G.TM
    x1, y1 = -G.O + (bx + 28) * G.TM, G.O - (by + 14) * G.TM
    lo, la = G.x3857_to_lonlat([x0, x1, x0, x0], [y0, y1, y1, y0])
    tri = {"type": "Polygon", "coordinates": [[[float(a), float(b)] for a, b in zip(lo, la)]]}
    aoi_path = str(tmp_path / "aoi.geojson"); OUT.write_aoi(aoi_path, tri)
    clipped = OUT.clip_lines(feats, aoi_path, str(tmp_path))
    assert 0 < len(clipped) < 2 * len(feats)
    ids = [p["src_id"] for _, p in clipped]
    assert len(set(ids)) == len(ids)
    rows = OUT.feature_rows(clipped)
    gp = tmp_path / "out.gpkg"
    assert OUT.write_gpkg(gp, rows) is None                          # spatial index created too
    # inside the triangle (barycentric test with a small tolerance)
    A, B, C = [np.array(v) for v in tri["coordinates"][0][:3]]
    def inside(pt, eps=1e-9):
        v0, v1, v2 = C - A, B - A, np.array(pt) - A
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = d00 * d11 - d01 * d01; v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
        return v >= -eps and w >= -eps and v + w <= 1 + 1e-7
    ogr = ogr_features(gp)
    assert len(ogr) == len(rows)
    assert all(inside(c) for f in ogr for c in f["geometry"]["coordinates"])
    assert {f["properties"]["class"] for f in ogr} == {"village_road"}
    assert all(f["properties"]["confidence"] is None and f["properties"]["name"] is None for f in ogr)
    # length_m is the ground length of the clipped piece
    for f in ogr[:50]:
        assert abs(f["properties"]["length_m"] - OUT.ground_length_m(f["geometry"]["coordinates"])) < 0.11
    info = ogr_layer_info(gp)
    assert info["geometryFields"][0]["type"] == "LineString"
    assert "4326" in json.dumps(info["geometryFields"][0]["coordinateSystem"])
    _, types = gpkg_rows(gp)
    for n, t in OUT.FIELDS:
        assert types[n] == t, (n, types[n])


@need_venv
@need_guj
def test_predict_block_shapes_and_georef_match_statewide(tmp_path):
    bx, by = PARITY_BLOCK
    r = subprocess.run([VENV_PY, __file__, "predict_shapes", str(tmp_path), str(bx), str(by)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
    info = lambda f: json.loads(subprocess.run(["gdalinfo", "-json", f], capture_output=True, text=True,
                                               check=True).stdout)
    mine = info(str(tmp_path / ("u_%d_%d.tif" % PARITY_BLOCK)))
    ref = info("%s/prob_u/u_%d_%d.tif" % (GUJ, *PARITY_BLOCK))
    assert mine["geoTransform"] == ref["geoTransform"] and mine["size"] == ref["size"]
    assert mine["bands"][0]["type"] == ref["bands"][0]["type"] == "Byte"
    assert mine["metadata"]["IMAGE_STRUCTURE"]["LAYOUT"] == "COG"
    assert (tmp_path / ("n_%d_%d.tif" % PARITY_BLOCK)).exists()


@need_gdal
def test_empty_gpkg_keeps_schema(tmp_path):
    import outputs as OUT
    gp = tmp_path / "empty.gpkg"
    OUT.write_gpkg(gp, [])
    info = ogr_layer_info(gp)
    assert info["featureCount"] == 0
    assert {f["name"] for f in info["fields"]} >= {"class", "confidence", "name", "src_id", "surface_width_m",
                                                   "width_p90_m", "lane_tier", "bridged", "length_m"}


# ---- end to end with a synthetic predictor ------------------------------------------------
def _check_common(req, r):
    assert r.returncode == 0, (r.stdout[-3000:], r.stderr[-3000:])
    pv = progress_values(r.stdout)
    assert pv and pv[-1] == 1.0 and all(b >= a for a, b in zip(pv, pv[1:]))
    res = json.loads(Path(req["result_json"]).read_text())
    assert res["status"] == "ok" and res["model"] == "microsoft-roaddetections-res-unet-6"
    assert res["model_version"] == "HF_GG08_60"
    return res


@need_venv
@need_mb
@need_gdal
def test_fake_run_full_imagery(tmp_path):
    import outputs as OUT
    rp, req = request(tmp_path, box(*CORNER_FULL))
    r = fake_run(rp)
    res = _check_common(req, r)
    assert res["image_source"] == "Esri Wayback z18 release 26334" and res["image_dates"] == ["2026-08-05"]
    c = res["counts"]
    assert c["blocks_planned"] == 4 and c["blocks_inferred"] == 4 and c["blocks_no_imagery"] == 0
    pr = res["artifacts"]["prob_raster"]
    assert len(pr) == 4 and all(os.path.exists(f) and "/artifacts/prob_u/u_" in f for f in pr)
    gi = json.loads(subprocess.run(["gdalinfo", "-json", pr[0]], capture_output=True, text=True, check=True).stdout)
    assert gi["size"] == [4096, 4096] and gi["bands"][0]["type"] == "Byte" and "3857" in json.dumps(gi["coordinateSystem"])
    # coverage == the AOI (full imagery)
    aoi_km2 = OUT.multipolygon_area_km2({"coordinates": [box(*CORNER_FULL)["coordinates"]]})
    assert abs(c["coverage_km2"] - aoi_km2) / aoi_km2 < 0.01
    # '+' through each block centre: 2 horizontal + 2 vertical lines across the AOI
    w_km = 2 * HALF * 111.32 * math.cos(math.radians(CORNER_FULL[1])); h_km = 2 * HALF * 110.57
    assert abs(c["length_km"] - 2 * (w_km + h_km)) / (2 * (w_km + h_km)) < 0.08
    rows, _ = gpkg_rows(req["out_gpkg"])
    assert len(rows) == c["features"] > 0 and {x["class"] for x in rows} == {"village_road"}
    assert len({x["src_id"] for x in rows}) == len(rows)
    assert all(x["surface_width_m"] is not None and 8.0 < x["surface_width_m"] < 11.0 for x in rows
               if x["length_m"] > 100)                              # 16 px band * 0.548 m = 8.8 m
    # native rasters are pruned once every user is vectorised
    assert not list((Path(req["work_dir"]) / "prob_n").glob("n_*.tif"))


@need_venv
@need_mb
@need_gdal
def test_fake_run_imagery_edge_never_waits(tmp_path):
    rp, req = request(tmp_path, box(*CORNER_EDGE))
    t = time.time()
    r = fake_run(rp, timeout=600)
    res = _check_common(req, r)
    c = res["counts"]
    assert c["blocks_planned"] == 4 and c["blocks_no_imagery"] == 2 and c["blocks_inferred"] == 2
    assert any("no imagery" in w for w in res["warnings"])
    aoi_km2 = (2 * HALF * 111.32 * math.cos(math.radians(CORNER_EDGE[1]))) * (2 * HALF * 110.57)
    assert 0.05 * aoi_km2 < c["coverage_km2"] < 0.5 * aoi_km2         # 2 partial blocks of 4
    assert c["features"] > 0
    cov = res["coverage"]
    assert cov["type"] == "MultiPolygon" and cov["coordinates"]
    assert time.time() - t < 600


@need_venv
@need_gdal
def test_fake_run_no_imagery_is_empty_ok(tmp_path):
    rp, req = request(tmp_path, box(72.5, 23.0, 0.005))            # far from the corridor MBTiles
    r = fake_run(rp, timeout=300)
    res = _check_common(req, r)
    assert res["counts"]["features"] == 0 and res["coverage"] == {"type": "MultiPolygon", "coordinates": []}
    assert ogr_layer_info(req["out_gpkg"])["featureCount"] == 0
    rp2, req2 = request(tmp_path / "none", box(72.5, 23.0, 0.005), items=[])
    res2 = _check_common(req2, fake_run(rp2, timeout=300))
    assert res2["coverage"] == {"type": "MultiPolygon", "coordinates": []} and res2["image_source"] is None


@need_venv
@need_mb
def test_sigterm_stops_process_group(tmp_path):
    rp, req = request(tmp_path, box(*CORNER_FULL), )
    # kill while the spawned vectorisation workers are busy (all 4 blocks inferred, jobs queued)
    pr = subprocess.Popen([VENV_PY, __file__, "fake_run", str(rp), "1"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, start_new_session=True, cwd=str(REPO))
    t = time.time()
    for line in pr.stdout:
        if line.startswith("PROGRESS") and "inferred 4/4" in line:
            break
        assert time.time() - t < 120
    time.sleep(3)
    kids = subprocess.run(["pgrep", "-s", str(pr.pid)], capture_output=True, text=True).stdout.split()
    assert len(kids) >= 2                                            # parent + workers
    os.killpg(pr.pid, signal.SIGTERM)
    t = time.time()
    pr.wait(timeout=30)
    assert pr.returncode != 0 and time.time() - t < 30
    time.sleep(2)
    with pytest.raises(ProcessLookupError):                         # no orphaned workers left
        os.killpg(pr.pid, 0)
    assert not Path(req["result_json"]).exists()


# ---- --check --------------------------------------------------------------------------------
@need_venv
@need_mb
def test_check_mode_ok(tmp_path):
    rp, _ = request(tmp_path, box(*CORNER_FULL))
    r = subprocess.run([VENV_PY, str(RUN), "--check", "--request", str(rp)], capture_output=True, text=True,
                       timeout=300, env=dict(os.environ, HTTPS_PROXY="http://127.0.0.1:9", HTTP_PROXY="http://127.0.0.1:9"))
    rep = json.loads(r.stdout)
    assert r.returncode == 0 and rep["ok"], rep["errors"]
    assert rep["packages"]["tensorflow"] and rep["packages"]["tf_keras"] and rep["packages"]["sknw"]
    assert rep["weights"]["exists"] and rep["weights"]["readable"] and rep["weights"]["pickle"]
    assert rep["mbtiles"][0]["z18_tiles_in_aoi"] == rep["aoi_tiles"] > 0
    assert rep["ogr_clip_geos"] is True and all(rep["gdal_tools"].values())
    assert not (tmp_path / "wd").exists()                            # check has no side effects


@need_venv
def test_check_mode_reports_problems(tmp_path):
    rp, _ = request(tmp_path, box(72.5, 23.0, 0.005), weights=str(tmp_path / "nope.pkl"),
                    items=[{"path": str(tmp_path / "missing.mbtiles")}] + ([{"path": MB}] if os.path.exists(MB) else []))
    r = subprocess.run([VENV_PY, str(RUN), "--check", "--request", str(rp)], capture_output=True, text=True, timeout=300)
    rep = json.loads(r.stdout)
    assert r.returncode == 1 and not rep["ok"]
    err = " ".join(rep["errors"])
    assert "weights missing" in err and "missing.mbtiles" in err
    if os.path.exists(MB):
        assert "no z18 tiles intersect the AOI" in err
