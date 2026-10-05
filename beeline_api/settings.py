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

for d in (DATA, REFERENCE_DIR, IMAGERY_DIR, DEM_DIR, RUNS_DIR, UPLOADS_DIR, TILE_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)
