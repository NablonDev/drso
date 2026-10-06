"""LULC model plugins.

Contract — a plugin turns imagery over a band into one layer:

    class MyModel(ModelPlugin):
        name = "buildings_v5"; version = "5.0.1"; layer_key = "buildings"; imagery = "esri"
        def run(self, ctx: ModelContext) -> ModelOutput: ...

ModelContext carries the band polygon (lon/lat), the imagery snapshot of the run (products,
capture dates, local file paths when fetched), a progress callback and a cancel check.
ModelOutput carries the features (GeoDataFrame, EPSG:4326, columns class/name/confidence/src_id),
the coverage polygon (where the model actually had imagery), the image source and date it read,
and a layer-level confidence.

v1 ships stubs: each one returns the layer from the reference store and stamps the provenance
the real model would (imagery product + capture date, model name/version, run time). Register a
real model by adding it to REGISTRY under the name used in config/layers.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import geopandas as gpd

from . import reference


@dataclass
class ModelContext:
    layer_key: str
    band: object                       # shapely Polygon lon/lat
    imagery: dict                      # run imagery snapshot (see imagery.check)
    progress: Callable[[float], None] = lambda f: None
    cancelled: Callable[[], bool] = lambda: False
    params: dict = field(default_factory=dict)


@dataclass
class ModelOutput:
    gdf: gpd.GeoDataFrame
    coverage: object | None
    image_source: str | None
    image_date: str | None
    model: str
    model_version: str
    confidence: float | None
    run_at: str
    notes: str = ""


class ModelPlugin:
    name = "base"
    version = "0"
    layer_key = ""
    imagery = "esri"

    def run(self, ctx: ModelContext) -> ModelOutput:  # pragma: no cover - interface
        raise NotImplementedError


def _imagery_provenance(imagery: dict, product: str) -> tuple[str | None, str | None]:
    if product == "esri":
        e = (imagery or {}).get("esri") or {}
        src = "Esri World Imagery" + (f" ({', '.join(e.get('sources', []))})" if e.get("sources") else "")
        return src, e.get("newest")
    if product == "sentinel2":
        s = (imagery or {}).get("sentinel2") or {}
        return "Sentinel-2 L2A", s.get("newest")
    return None, None


class ReferenceStub(ModelPlugin):
    """Stand-in for a model: serves the reference-store layer with model-style provenance."""

    def __init__(self, name: str, layer_key: str, imagery: str):
        self.name = name
        self.layer_key = layer_key
        self.imagery = imagery
        self.version = "stub-1.0"

    def run(self, ctx: ModelContext) -> ModelOutput:
        ctx.progress(0.1)
        gdf, meta = reference.read_clip(self.layer_key, ctx.band)
        ctx.progress(0.9)
        if gdf is None:
            raise FileNotFoundError(f"no reference data for layer '{self.layer_key}' (stub model has nothing to serve)")
        src, dt = _imagery_provenance(ctx.imagery, self.imagery)
        cov = reference.coverage_of(meta)
        return ModelOutput(gdf=gdf, coverage=cov, image_source=src, image_date=dt or (meta or {}).get("source_date"),
                           model=self.name, model_version=self.version,
                           confidence=(meta or {}).get("confidence"),
                           run_at=datetime.now(timezone.utc).isoformat(),
                           notes=f"stub: served from reference store version {(meta or {}).get('version')}")


REGISTRY: dict[str, ModelPlugin] = {
    "lulc_stub": ReferenceStub("lulc_stub", "lulc", "sentinel2"),
    "buildings_stub": ReferenceStub("buildings_stub", "buildings", "esri"),
    "water_stub": ReferenceStub("water_stub", "water", "sentinel2"),
    "power_stub": ReferenceStub("power_stub", "power", "esri"),
    "plotted_stub": ReferenceStub("plotted_stub", "plotted", "esri"),
}


def get(name: str) -> ModelPlugin:
    if name not in REGISTRY:
        raise KeyError(f"model plugin '{name}' is not registered")
    return REGISTRY[name]
