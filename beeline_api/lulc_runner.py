"""Run a LULC model plugin (lulc_models/<name>/run_aoi.py) for one run's band.

The plugin is started in its own environment and process group with every network route
closed (proxies → dead port, Hugging Face offline). It gets only catalog imagery intersecting
the AOI and the run's upstream layers. See lulc_models/CONTRACT.md.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pyogrio
import yaml
from shapely.geometry import mapping, shape

from beeline_engine.config import CONFIG_DIR

from . import imagery_store, settings

log = logging.getLogger(__name__)
DEAD_PROXY = "http://127.0.0.1:9"


class PluginError(RuntimeError):
    pass


@dataclass
class PluginResult:
    gdf: gpd.GeoDataFrame
    coverage: object | None
    meta: dict
    artifacts: dict
    work_dir: Path


def load_models() -> dict:
    return yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())


def _expand(s: str) -> str:
    return (s.replace("${ENV_ROOT}", settings.MODEL_ENV_ROOT)
             .replace("${ROADDET_VENV}", settings.ROADDET_VENV))


def model_cfg(name: str) -> dict:
    cfg = load_models()
    if name not in cfg["models"]:
        raise PluginError(f"model '{name}' is not in config/models.yaml")
    m = dict(cfg["models"][name])
    m["name"] = name
    m["python"] = _expand(m["env"])
    m["code_dir"] = settings.REPO / m["code"]
    m["hf_home"] = str(settings.DATA / cfg["shared"]["hf_home"])
    m["weights_abs"] = {k: str(settings.DATA / v) for k, v in (m.get("weights") or {}).items()}
    return m


def offline_env(m: dict) -> dict:
    # the model environment brings its own GDAL/PROJ (wheels): never inherit the worker's
    drop = {"PROJ_DATA", "PROJ_LIB", "GDAL_DATA", "GDAL_DRIVER_PATH", "LD_LIBRARY_PATH", "PYTHONHOME",
            "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV"}
    env = {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy") and k not in drop}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
              "GDAL_HTTP_PROXY", "FTP_PROXY"):
        env[k] = DEAD_PROXY
    env.update(NO_PROXY="", no_proxy="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HOME=m["hf_home"], GDAL_HTTP_TIMEOUT="5", CPL_VSIL_CURL_ALLOWED_EXTENSIONS="none",
               PYTHONUNBUFFERED="1", PYTHONPATH=str(settings.REPO / "lulc_models"),
               CUDA_VISIBLE_DEVICES=str(settings.MODEL_GPU_DEVICE), OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"))
    # PATH: the model env's bin, then the system tools (gdal-bin), as in the worker image. The worker's
    # own conda env must not lend its GDAL CLIs: they look for PROJ data the plugin env does not set.
    env["PATH"] = ":".join([str(Path(m["python"]).parent), "/usr/local/bin", "/usr/bin", "/bin"])
    env.pop("BEELINE_ESRI_TOKEN", None)      # credentials never reach a model process
    env.pop("BEELINE_PC_SUBSCRIPTION_KEY", None)
    return env


def preflight(m: dict) -> list[str]:
    """Problems that stop the plugin from starting (shown to the drafter as the failure reason)."""
    probs = []
    if not Path(m["python"]).exists():
        probs.append(f"model environment not installed: {m['python']}")
    if not (m["code_dir"] / "run_aoi.py").exists():
        probs.append(f"plugin code missing: {m['code_dir']}/run_aoi.py")
    for k, p in m["weights_abs"].items():
        if not Path(p).exists():
            probs.append(f"weights missing ({k}): {p} — run scripts/install_models.py")
    return probs


def build_request(m: dict, aoi, work_dir: Path, inputs: dict, params: dict | None = None) -> dict:
    img = imagery_store.for_aoi(aoi, products=tuple(m.get("imagery") or ()))
    return {
        "layer_key": m["layer"], "aoi": mapping(aoi), "work_dir": str(work_dir),
        "out_gpkg": str(work_dir / "out.gpkg"), "result_json": str(work_dir / "result.json"),
        "imagery": img, "inputs": inputs,
        "weights": {**m["weights_abs"], "hf_home": m["hf_home"]},
        "gpu": {"device": settings.MODEL_GPU_DEVICE, "vram_mb": m.get("vram_mb", 12000)} if m.get("gpu") else None,
        "params": {**(m.get("params") or {}), **(params or {})},
    }


def run(name: str, aoi, run_id: str, job_id: int, inputs: dict, progress=None, cancelled=None,
        params: dict | None = None, check_only: bool = False) -> PluginResult | dict:
    m = model_cfg(name)
    probs = preflight(m)
    if probs:
        raise PluginError("; ".join(probs))
    work = settings.RUNS_DIR / run_id / "lulc" / name / f"job{job_id}"
    if work.exists() and not check_only:
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    inputs = {k: (str(Path(v).resolve()) if isinstance(v, str) and v and not v.startswith("/") else
                  [str(Path(x).resolve()) for x in v] if isinstance(v, list) else v) for k, v in inputs.items()}
    req = build_request(m, aoi, work, inputs, params)
    (work / "request.json").write_text(json.dumps(req, indent=1, default=str))
    cmd = [m["python"], str(m["code_dir"] / "run_aoi.py")] + (["--check"] if check_only else []) + \
        ["--request", str(work / "request.json")]
    logf = open(work / "plugin.log", "w")
    proc = subprocess.Popen(cmd, cwd=str(m["code_dir"]), env=offline_env(m), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True, bufsize=1)
    deadline = time.time() + float(m.get("timeout_h", 6)) * 3600
    killed = {"why": None}

    def watchdog():
        while proc.poll() is None:
            if cancelled and cancelled():
                killed["why"] = "cancelled"
            elif time.time() > deadline:
                killed["why"] = f"timed out after {m.get('timeout_h', 6)} h"
            if killed["why"]:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    for _ in range(30):
                        if proc.poll() is not None:
                            return
                        time.sleep(1)
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                return
            time.sleep(2)

    threading.Thread(target=watchdog, daemon=True).start()
    tail: list[str] = []
    for line in proc.stdout:
        logf.write(line)
        logf.flush()
        line = line.rstrip("\n")
        if line.startswith("PROGRESS "):
            parts = line.split(" ", 2)
            try:
                if progress:
                    progress(float(parts[1]), parts[2] if len(parts) > 2 else "")
            except ValueError:
                pass
        else:
            tail = (tail + [line])[-40:]
    rc = proc.wait()
    logf.close()
    if killed["why"] == "cancelled":
        from .jobs import Cancelled
        raise Cancelled()
    if killed["why"]:
        raise PluginError(f"{name}: {killed['why']}")
    if rc != 0:
        raise PluginError(f"{name} exited with {rc}: " + " | ".join(tail[-8:]))
    if check_only:
        try:
            return json.loads(next(l for l in reversed(tail) if l.strip().startswith("{")))
        except (StopIteration, json.JSONDecodeError):
            return {"ok": True, "log": tail[-20:]}
    res = json.loads((work / "result.json").read_text())
    if res.get("status") != "ok":
        raise PluginError(f"{name}: {res.get('status')} — {res.get('error') or ''}")
    out = work / "out.gpkg"
    gdf = pyogrio.read_dataframe(out, layer="features") if out.exists() else gpd.GeoDataFrame(
        {"class": [], "name": [], "confidence": [], "src_id": []}, geometry=[], crs=4326)
    gdf = gdf.to_crs(4326) if gdf.crs else gdf.set_crs(4326)
    for col in ("class", "name", "confidence", "src_id"):
        if col not in gdf:
            gdf[col] = None
    extra = [c for c in gdf.columns if c not in ("class", "name", "confidence", "src_id", "geometry")]
    gdf["attrs"] = [{c: (None if v != v else v) for c, v in zip(extra, row)} if extra else None
                    for row in gdf[extra].itertuples(index=False, name=None)] if extra else None
    gdf = gdf[["class", "name", "confidence", "src_id", "attrs", "geometry"]]
    cov = shape(res["coverage"]) if res.get("coverage") else None
    meta = {"model": res.get("model"), "model_version": res.get("model_version"),
            "image_source": res.get("image_source"),
            "image_date": max(res.get("image_dates") or [None], key=lambda d: d or ""),
            "warnings": res.get("warnings") or [], "counts": res.get("counts") or {}}
    return PluginResult(gdf=gdf, coverage=cov, meta=meta, artifacts=res.get("artifacts") or {}, work_dir=work)
