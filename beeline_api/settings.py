"""Runtime settings from the environment (Docker sets these; dev defaults point into ./data)."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("BEELINE_DATA", REPO / "data" / "bucket"))
DB_URL = os.environ.get("BEELINE_DB_URL", "postgresql+psycopg://beeline@/beeline?host=/tmp&port=55432")
REGION = os.environ.get("BEELINE_REGION", "gujarat")
# Reference layer store; production and dev data never mix (dev builder writes to reference_dev)
REFERENCE_DIR = Path(os.environ.get("BEELINE_REFERENCE_DIR", DATA / "reference"))
IMAGERY_DIR = DATA / "imagery"
DEM_DIR = DATA / "dem" / "copernicus"
RUNS_DIR = DATA / "runs"
UPLOADS_DIR = DATA / "uploads"
TILE_CACHE_DIR = IMAGERY_DIR / "esri" / "tilecache"
ESRI_TILE_URL = os.environ.get(
    "BEELINE_ESRI_TILE_URL",
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}")
ESRI_MAX_ZOOM = int(os.environ.get("BEELINE_ESRI_MAX_ZOOM", "18"))
FETCH_IMAGERY_ON_EXTRACT = os.environ.get("BEELINE_FETCH_IMAGERY", "0") == "1"
HTTP_UA = "beeline-route-planner/1.0"
OVERPASS_URLS = [u.strip() for u in os.environ.get(
    "BEELINE_OVERPASS_URLS",
    "https://overpass-api.de/api/interpreter,https://overpass.kumi.systems/api/interpreter").split(",") if u.strip()]

for d in (DATA, REFERENCE_DIR, IMAGERY_DIR, DEM_DIR, RUNS_DIR, UPLOADS_DIR, TILE_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- licensed imagery acquisition (credentials come from the environment / deploy/.env) ----
ESRI_TOKEN = os.environ.get("BEELINE_ESRI_TOKEN", "")
WAYBACK_URL = os.environ.get(
    "BEELINE_WAYBACK_URL",
    "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/{release}/{z}/{y}/{x}")
WAYBACK_RELEASE = os.environ.get("BEELINE_WAYBACK_RELEASE", "26334")
WAYBACK_RELEASE_DATE = os.environ.get("BEELINE_WAYBACK_RELEASE_DATE", "2026-08-05")
LICENCE_ESRI = os.environ.get("BEELINE_LICENCE_ESRI", "Esri World Imagery Wayback — operator licence")
STAC_URL = os.environ.get("BEELINE_STAC_URL", "https://earth-search.aws.element84.com/v1/search")
PC_SUBSCRIPTION_KEY = os.environ.get("BEELINE_PC_SUBSCRIPTION_KEY", "")
# ---- LULC model plugins ----
MODEL_ENV_ROOT = os.environ.get("BEELINE_MODEL_ENV_ROOT", "/home/vivek_jain/miniforge3/envs")
ROADDET_VENV = os.environ.get("BEELINE_ROADDET_VENV", "/mnt/cog-data/roaddet/venv")
MODELS_DIR = DATA / "models"
MODEL_GPU_DEVICE = int(os.environ.get("BEELINE_MODEL_GPU", "0"))
# Kill switch: 0 = never start a model plugin on this machine (stored outputs still served)
RUN_MODELS = os.environ.get("BEELINE_RUN_MODELS", "1") == "1"
