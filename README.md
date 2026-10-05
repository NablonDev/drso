# Beeline — transmission route planner

Two-step flow: **extract layers** (LULC models + reference data, hours when real models run) →
drafter reviews → **solve the route** (minutes, re-runnable). Four alignments per solve, spotted
towers, ₹ cost, a typed decision log, DXF/SHP/KML/CSV/PDF exports and a reproducible manifest.

```
web/              React + TypeScript + MapLibre UI (served by the API from web/dist)
beeline_api/      FastAPI app, GPU worker, extraction, solve job, exports, imagery, uploads
beeline_engine/   route solver (pure Python: NumPy/SciPy/Shapely + PyTorch on GPU)
config/           layers.yaml, rules.yaml (31 rules), costs.yaml (8 categories), engine.yaml
deploy/           Dockerfile, docker-compose.yml, environment.yml, .env.example
scripts/          ingest_reference.py, ingest_all.py, build_dev_reference.py, benchmark.py, dev_up.sh
tests/            test_engine.py (synthetic, incl. brute-force optimality), test_api.py, e2e_api.py
```

## Production deployment (cloud L4 VM, VPN, no TLS, no auth)

1. **Host prerequisites** — NVIDIA driver ≥ 550, Docker Engine + compose plugin, NVIDIA container
   toolkit (`nvidia-ctk runtime configure --runtime=docker && systemctl restart docker`).
   Check: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
2. **Code** — `git clone <repo> /opt/beeline && cd /opt/beeline/deploy && cp .env.example .env`;
   set `BEELINE_DB_PASSWORD`, the bucket and pgdata paths, and the port.
3. **Start** — `docker compose up -d --build` (first build ≈ 10–15 min: conda env + CUDA torch).
   `docker compose ps` should show db healthy, api healthy, worker running.
   `curl localhost:8080/api/health` → `{"ok": true, ...}`.
4. **Reference layers** — copy the statewide gpkgs onto the VM, fill
   `config/reference_manifest.yaml` from the example, then:
   ```
   docker compose run --rm -v /srv/beeline/incoming:/srv/beeline/incoming api \
       python scripts/ingest_all.py config/reference_manifest.yaml --dry-run   # check classes
   docker compose run --rm -v /srv/beeline/incoming:/srv/beeline/incoming api \
       python scripts/ingest_all.py config/reference_manifest.yaml
   ```
   Layers land in `<bucket>/reference/gujarat/<layer>/<version>/` with provenance and coverage.
   Fetch hooks for AOIs that are not covered go in `config/reference_sources.yaml`.
5. **OSM power lines and towers (all India)** — 765 / 400 / 220 / 132 kV only, stored by voltage
   (PostGIS table `osm_power`, one partition per voltage). Load once (downloads the ~1.7 GB Geofabrik
   India extract into the bucket, ~5 min):
   ```
   docker compose run --rm worker python -m beeline_api.osm_power load-india
   docker compose run --rm worker python -m beeline_api.osm_power status
   ```
   "Refresh from OSM" on the power layer (Review the layers) re-pulls the run's area from Overpass,
   replaces it in the store in one transaction, and re-extracts the power layer. It runs as an
   extraction job, so it is refused while a solve runs. Every load is logged with its OSM timestamp,
   and the layer's provenance says which extract / refresh it came from. Lines with no voltage tag
   or another voltage are not stored (India extract of 3 Oct 2026: 2,547 untagged, 11,787 other).
6. **Cell towers (all India, Tarang Sanchar)** — one PostGIS table `cell_towers` (site id,
   structure type Ground / Roof / Wall). Load once (downloads the 20 MB parquet of the 2022
   Tarang Sanchar scrape mirrored by ramSeraph/indian_communications, 665,077 sites, ~20 s):
   ```
   docker compose run --rm worker python -m beeline_api.cell_towers load-india
   docker compose run --rm worker python -m beeline_api.cell_towers status
   ```
   The data is from 2022 and ~81 % of DoT's Mar-2025 tower count; treat it as a floor, not a
   census. Licence: CC0 on the curation only, the route is listed as "not-so-open" — clear it
   before a client deliverable. "Refresh from Tarang Sanchar" is shown on the review screen but
   disabled: the portal's tower search needs name + email + phone + CAPTCHA + emailed OTP and
   answers Indian IPs only, so there is no machine source yet. When one is secured (PM GatiShakti
   Public, an NIC token), set `cell_towers.FETCHER`; the refresh job, store replace, provenance
   and UI already work (covered by `tests/test_api.py::test_cell_towers_layer_and_refresh`).
   Rule 31 "Cell tower" keeps new towers 50 m from tower sites — a placeholder pending AESL.
7. **Highways — NH and SH (all India, OSM)** — the `roads` layer. PostGIS table `osm_highways`,
   one partition per class (nh / sh). Uses the same Geofabrik India extract as the power store
   (downloaded once):
   ```
   docker compose run --rm worker python -m beeline_api.osm_highways load-india
   docker compose run --rm worker python -m beeline_api.osm_highways status
   ```
   A way is NH when its `ref` reads NH… / NE…, or it belongs to a route relation with network
   IN:NH / IN:NE; SH when its ref reads SH… or its relation is IN:SH:<state>. A motorway / trunk
   with neither is counted as NH and marked *inferred* in the popup and the legend. A way that is
   both is stored once per class. `*_link` ramps are not kept. MDR is no longer carried (it had
   no rule). "Refresh from OSM" on the Highways layer re-pulls the run's area from Overpass,
   replaces it in the store in one transaction and re-extracts the layer — same job, lock and
   provenance as the power layer.
8. **Open** `http://<vm>:8080/` over the VPN.

Upgrades: `git pull && docker compose up -d --build`. The worker marks any job it was running as
failed on restart, so restart only when `GET /api/jobs/active` is `{"job": null}`.

Logs: `docker compose logs -f api worker`. Data: Postgres in `BEELINE_PGDATA`; imagery tiles,
DEM, run exports and uploads in `BEELINE_BUCKET`. No backups are configured (by decision).

Verified on the dev VM (5 Oct 2026): fresh-database boot of db + api (4 processes) + GPU worker,
`tests/e2e_api.py` end to end against the containers (imagery checks, DEM fetch, extraction, 4-route
solve on GPU, exports, copy, re-solve, compare), native PostGIS vector tiles, SPA deep links.

## Behaviour worth knowing

- **One GPU, one job.** Extraction and solve never queue behind each other: a second job of either
  kind is refused with HTTP 409 and a message naming the running job. Enforced by a partial unique
  index, so concurrent clicks cannot both start.
- **Layer provenance.** Every layer version records source, mode (auto / replace / add), model and
  version, imagery product and capture date, model run time, confidence and the coverage polygon.
  Outside the coverage polygon a layer is "no data", reported as `NO_COVERAGE`, not "nothing found".
- **LULC models are stubs** (`beeline_api/models.py`): they serve the reference-store layer and stamp
  model-style provenance. Register a real model in `REGISTRY` under the name used in `layers.yaml`.
- **Imagery.** Esri World Imagery capture dates come from the MapServer identify service, sampled on a
  3 km grid; ≤ 0.6 m and ≤ 18 months counts as covered. Sentinel-2 from Earth Search STAC. DEM is
  Copernicus GLO-30 (30 m), fetched on demand. `BEELINE_FETCH_IMAGERY=1` also downloads Esri z18
  tiles and clipped Sentinel-2 scenes into the bucket during extraction (for real models). The map's
  Esri tiles go through the API's caching proxy; the browser never calls Esri directly.
- **Procurement** is a stub (records the request, sends nothing).

## Route solver (v1, deliberately simple)

1. Rule compile on a 20 m raster in a frame rotated along the bee-line: tower-forbidden,
   angle-tower-forbidden and conductor-forbidden masks from distance transforms; ₹/m² corridor cost;
   crossing-direction rasters; foundation multiplier by soil.
2. One pylon-graph pass: tower sites every 50 m, spans 100 m – 1.125 × design span, state =
   (site, incoming span), tower type and cost from the deflection band, layer-wise DP on GPU,
   ordered by chainage. Exact for that graph (checked against brute-force Dijkstra in the tests).
   River-only long spans up to 1 km. A and B are gantries; the first/last towers are dead-ends,
   the terminal yard is carved out of the substation rule.
3. Four routes = the same solve with one extra internal penalty each (angle towers / lateral offset /
   sensitive land); a near-duplicate route is re-solved with the earlier corridor priced up.
   Reported ₹ is always the true cost.
4. Spotting: string-pulling to angle points, equal spans on each leg, exact vector checks, small
   nudges where the grid and the exact check disagree, section points, forced crossing tower types,
   DEM clearance with a +18 m extension.
5. Decision log: typed events + templates (no LLM), with source and confidence on every event.

Benchmark (`scripts/benchmark.py`, 150 km × ±7.5 km, 4 routes, GPU capped at 24 GB): 43 s on the dev
A100, 1.9 GB peak GPU memory. Expect ~2–3 min on an L4 (memory-bandwidth bound).

Interpretations to confirm with AESL (all configurable): A/B are gantries and the terminal yard is
carved out of the substation rule; new towers keep the corridor half-width from existing conductors
(the 100 m rule applies to existing towers); minimum span = the rule's 100 m crossing minimum;
river-only long spans up to 1 km, with crossing towers raised by the height the sag needs;
"Protected areas and ESZ = No-Go" added as rule 30. Plotted land, monuments, defence land and
SH/MDR roads have no rule among the 29 yet, so they do not move the route.

Not in v1 (by decision — add when run time allows): pass-2 refinement, corridor-guided detours
around blocks larger than the band, backwards (non-monotone) detours, joint routing + tower height,
the n±1 deviation limit next to power-line crossings (reported as a warning, not enforced).

## Development

```
source dev.env                    # conda env 'beeline', PROJ/GDAL data, dev reference store
./scripts/dev_up.sh               # Postgres (:55432), API (:8710), worker
python -m pytest -q tests/test_engine.py tests/test_api.py
python tests/e2e_api.py           # against the running stack
cd web && npm run dev             # UI dev server on :5173 proxying /api
```
The dev reference store (`data/bucket/reference_dev`) is built by `scripts/build_dev_reference.py`
from local files + OSM for the Bhuj–Lakadia AOI only. It is test data, never production.
