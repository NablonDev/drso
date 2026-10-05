# LULC model plugin contract

A plugin is a self-contained program under `lulc_models/<name>/` run by the worker as a
subprocess, in its own Python environment (configured in `config/models.yaml`).

**No network.** The worker starts every plugin with all HTTP(S)/GDAL proxies pointed at a dead
port and `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. A plugin reads imagery only from the
paths it is given; imagery acquisition (licensed, online) is a separate step
(`beeline_api/acquire.py`) that fills the imagery catalog beforehand.

## Invocation

    <env python> lulc_models/<name>/run_aoi.py --request <work_dir>/request.json

## request.json

```json
{
  "layer_key": "solar",
  "aoi": {"type": "Polygon", "coordinates": [...]},          // EPSG:4326, the run band (+margin)
  "work_dir": "/data/bucket/runs/<run>/lulc/solar/<job>",   // scratch, deleted on success
  "out_gpkg": "<work_dir>/out.gpkg",
  "result_json": "<work_dir>/result.json",
  "imagery": {                                              // only what the catalog holds for the AOI
    "esri_wayback_z18": [{"path": ".../x.mbtiles", "release": "26334", "date": "2026-08-05",
                          "footprint": {...}}],
    "sentinel2_l2a":    [{"scene_id": "...", "date": "2026-03-04", "cloud": 3.1,
                          "bands": {"B04": "...tif", "B08": "...", "B11": "...", "B12": "..."},
                          "footprint": {...}}],
    "sentinel1_rtc_vv": [{"scene_id": "...", "date": "...", "path": "...tif", "footprint": {...}}],
    "cartosat3_pan":    [{"scene_id": "2660409141", "path": ".../BAND.tif", "footprint": {...}}],
    "worldcover":       [{"path": "...tif", "year": 2021, "footprint": {...}}],
    "nisar_coh":        [{"path": "...tif", "footprint": {...}}]
  },
  "inputs": {                                               // upstream layers of the same run
    "village_roads_gpkg": "...", "village_roads_prob": ["...tif"],
    "buildings_gpkg": "...", "osm_solar_gpkg": "...", "osm_wind_gpkg": "...",
    "osm_substations_gpkg": "..."
  },
  "weights": {"checkpoint": "...", "hf_home": "...", "...": "..."},
  "gpu": {"device": 0, "vram_mb": 12000},
  "params": {}                                              // config/models.yaml defaults + overrides
}
```

## Outputs

- `out_gpkg`, layer `features`, **EPSG:4326**, clipped to the AOI. Required columns: `class`
  (a value of the layer's vocabulary in `config/layers.yaml`), `confidence` (0–1 or null),
  `name` (or null), `src_id` (stable within the run). Extra columns are kept as attributes.
- `result_json`:
  ```json
  {"status": "ok",
   "coverage": {...},              // GeoJSON (4326) where the model actually had imagery and ran
   "image_source": "Esri Wayback z18 release 26334",
   "image_dates": ["2026-08-05"],
   "model": "solar-dinov3-lora", "model_version": "v1 epoch07",
   "counts": {"features": 114}, "warnings": ["..."],
   "artifacts": {"prob_raster": ["...tif"]}}       // kept files for downstream layers
  ```
- Progress: lines on stdout `PROGRESS <0..1> <message>`. Anything else is logged.
- Exit code 0 on success (including "no imagery → empty output, empty coverage");
  non-zero on failure, with the reason on stderr.
- Cancellation: the worker sends SIGTERM to the process group; exit promptly.

`lulc_models/common.py` implements the request/progress/result plumbing; plugins import it by
path (it has no dependencies beyond the standard library).
