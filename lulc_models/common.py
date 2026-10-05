"""Shared plumbing for LULC plugins (standard library only — runs in any model env)."""
from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from pathlib import Path

_t0 = time.time()


def load_request() -> dict:
    if "--request" not in sys.argv:
        raise SystemExit("usage: run_aoi.py --request request.json")
    p = Path(sys.argv[sys.argv.index("--request") + 1])
    req = json.loads(p.read_text())
    Path(req["work_dir"]).mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(SystemExit("cancelled")))
    return req


def progress(frac: float, msg: str = ""):
    print(f"PROGRESS {max(0.0, min(1.0, frac)):.4f} {msg}", flush=True)


def log(msg: str):
    print(f"[{time.time() - _t0:7.1f}s] {msg}", flush=True)


def write_result(req: dict, **kw):
    out = {"status": "ok", "coverage": None, "image_source": None, "image_dates": [], "model": None,
           "model_version": None, "counts": {}, "warnings": [], "artifacts": {}}
    out.update(kw)
    Path(req["result_json"]).write_text(json.dumps(out, indent=1, default=str))


def gpu_env(req: dict):
    """Pin the device and expose the VRAM cap to the framework."""
    g = req.get("gpu") or {}
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(g.get("device", 0)))
    return int(g.get("vram_mb") or 12000)


def torch_vram_cap(vram_mb: int):
    import torch
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2**20
        torch.cuda.set_per_process_memory_fraction(min(0.95, vram_mb / total))


# ---- web-mercator tile maths (z18 imagery) ---------------------------------------------
def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lr = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds_3857(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    R = 6378137.0 * math.pi
    size = 2 * R / 2 ** z
    minx = -R + x * size
    maxy = R - y * size
    return minx, maxy - size, minx + size, maxy
