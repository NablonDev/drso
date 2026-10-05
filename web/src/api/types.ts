// Shapes of the Beeline API responses, written from web/fixtures and beeline_api/app.py.

export type RunStatus = "draft" | "extracting" | "review" | "solving" | "complete" | "blocked" | "failed";
export type Circuit = "sc" | "dc" | "qc";
export type SourceMode = "auto" | "replace" | "add";
export type LonLat = [number, number];

export interface Pt {
  lat: number;
  lon: number;
  name?: string | null;
}

export interface Job {
  id: number;
  run_id: string;
  kind: "extract" | "solve";
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  progress: number | null;
  stage: string | null;
  message: string | null;
  eta_s: number | null;
  eta_text: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  heartbeat_at: string | null;
  cancel_requested: boolean;
  error: string | null;
  detail: JobDetail | null;
}

export interface JobLayerStatus {
  status: "running" | "done" | "failed";
  progress?: number;
  features?: number;
  seconds?: number;
  error?: string;
}

export interface JobDetail {
  layers?: Record<string, JobLayerStatus> | string[] | null;
  failed?: string[];
  solve_id?: number;
  result?: { files?: string[]; solve_id?: number } & Record<string, unknown>;
  [k: string]: unknown;
}

export interface GeoCheck {
  ok: boolean;
  text: string;
  sub?: string;
}

export interface GeoPoint {
  in_india: boolean;
  state: string | null;
  district: string | null;
}

export interface GeoValidate {
  a: GeoPoint;
  b: GeoPoint;
  checks: GeoCheck[];
  single_state: boolean;
  beeline_km: number | null;
  region_supported: boolean;
  ok: boolean;
}

export interface ReuseCandidate {
  run_id: string;
  created_at: string;
  band_km: number;
  shared_km2: number;
  shared_pct: number;
  new_km2: number;
  full_cover: boolean;
}

export interface BandInfo {
  band: GeoJSON.Polygon;
  area_km2: number;
  beeline_km: number;
  extraction_minutes_full: number;
  reuse: ReuseCandidate[];
}

export interface SourceSetting {
  mode: SourceMode;
  upload_id?: number | null;
  mapping?: Mapping | null;
}

export interface Imagery {
  covered: boolean;
  checked_at?: string;
  error?: string;
  esri?: {
    product: string;
    rule: string;
    dates: string[];
    newest: string | null;
    oldest: string | null;
    gap_km2: number;
    samples: number;
    sources: string[];
    covered_pct: number;
    gap_geojson: GeoJSON.Geometry | null;
    gap_ranges_km: [number, number][];
    resolutions_m: number[];
  };
  sentinel2?: {
    product: string;
    cloud: number | null;
    newest: string | null;
    scenes: { id: string; date: string; cloud: number; visual?: string }[];
    covered_pct: number;
    resolution_m: number;
  };
  dem?: {
    product: string;
    held: string[];
    tiles: string[];
    note?: string;
    covered_pct: number;
    resolution_m: number;
  };
}

export interface RuleOverride {
  rev: number | null;
  params: Record<string, unknown>;
  reason: string | null;
  enabled: boolean;
}

export interface Run {
  id: string;
  family: string;
  rev: number;
  parent_id: string | null;
  note: string | null;
  created_at: string;
  updated_at: string;
  a_lon: number;
  a_lat: number;
  a_name: string | null;
  b_lon: number;
  b_lat: number;
  b_name: string | null;
  kv: number;
  circuit: Circuit;
  band_km: number;
  routes_wanted: number;
  sources: Record<string, SourceSetting> | null;
  rule_overrides: Record<string, RuleOverride> | null;
  cost_overrides: Record<string, Record<string, number>> | null;
  imagery: Imagery | null;
  accept_partial: boolean | null;
  reuse_from: string | null;
  status: RunStatus;
  latest_solve_id: number | null;
  geo: GeoValidate | null;
  a: Pt;
  b: Pt;
  job: Job | null;
}

export interface RunListItem extends Run {
  r1_length_km: string | null;
  r1_cost_rs: string | null;
  feasible: boolean | null;
  custom_sources: number;
  last_error?: string;
}

export interface RunDetail extends Run {
  last_job: Job | null;
  last_extract: Job | null;
  layers_ready: boolean;
  layers_failed: string[];
  layers_stale: string[];
  parent: Run | null;
}

export interface RunsResponse {
  runs: RunListItem[];
  active_job: Job | null;
}

export interface LayerConfig {
  key: string;
  name: string;
  kind: "polygon" | "line" | "point" | "any";
  producer: "model" | "derived" | "reference" | "osm" | "tarangsanchar";
  model?: string;
  imagery?: string;
  derived_from?: string;
  desc: string;
  minutes_per_km2: number;
  default_confidence: number;
  classes: string[];
}

export interface AppConfig {
  voltages: number[];
  circuits: Circuit[];
  layers: LayerConfig[];
  ruleset_version: string;
  costs_version: string;
  esri_max_zoom: number;
}

export interface Health {
  ok: boolean;
  db: boolean;
  region: string;
  reference_dir: string;
  disk_free_gb: number;
  last_worker_heartbeat: string | null;
}

export interface RunLayer {
  key: string;
  name: string;
  desc: string | null;
  producer: string;
  configured_producer?: string;
  extracted_producer?: string;
  /** the layer has an online source (OSM, Tarang Sanchar) that "Refresh" re-pulls for the run's area */
  refreshable?: boolean;
  /** source name for the button: "Refresh from {refresh_label}" */
  refresh_label?: string | null;
  /** false when this deployment cannot reach the source; refresh_note says why */
  refresh_available?: boolean;
  refresh_note?: string | null;
  minutes_per_km2: number | null;
  rules: string[];
  source_setting: SourceSetting;
  extracted: boolean;
  // present once extracted
  layer_key?: string;
  carried?: boolean;
  stale?: boolean;
  layer_version_id?: number;
  mode?: SourceMode;
  source?: string | null;
  file_name?: string | null;
  model?: string | null;
  model_version?: string | null;
  image_source?: string | null;
  image_date?: string | null;
  model_run_at?: string | null;
  confidence?: number | null;
  feature_count?: number;
  status?: "done" | "failed";
  error?: string | null;
  stats?: {
    classes?: Record<string, number>;
    /** OSM power: counts keyed "line_765", "tower_400" … */
    by_voltage?: Record<string, number>;
    /** cell towers: counts keyed by structure type "Ground", "Roof", "Wall" */
    by_structure?: Record<string, number>;
    /** OSM highways: counts keyed "nh", "sh", "nh_inferred" */
    by_class?: Record<string, number>;
    refresh?: { load_id: number; osm_timestamp?: string; data_date?: string; replaced?: number } & Record<string, number | string>;
    /** LULC model layers: where the features came from */
    lulc?: {
      served_from: "stored" | "stored_partial" | "model" | "none";
      stored_version?: string;
      warnings?: string[] | null;
      counts?: Record<string, number>;
    };
  } | null;
  created_at?: string;
  extracted_in?: string;
  coverage_pct?: number | null;
}

export interface LayerFlag {
  layer: string;
  kind: "failed" | "empty" | "low_conf" | "coverage" | "stale";
  text: string;
}

export interface LayersResponse {
  layers: RunLayer[];
  flags: LayerFlag[];
  band_km2: number;
  features_total: number;
  band: GeoJSON.Polygon;
}

export interface Feature {
  id: number;
  class: string | null;
  name: string | null;
  confidence: number | null;
  origin: "auto" | "user" | null;
  src_id: string | null;
  geom: GeoJSON.Geometry;
  area_m2: number | null;
  length_m: number | null;
  layer_key: string;
  mode: SourceMode;
  source: string | null;
  model: string | null;
  model_version: string | null;
  image_source: string | null;
  image_date: string | null;
  model_run_at: string | null;
  layer_conf: number | null;
  file_name: string | null;
  chainage_km: [number, number];
  attrs?: Record<string, string | number> | null;
}

export interface RuleParam {
  key: string;
  label: string;
  type: "num" | "rng" | "sel";
  unit: string | null;
  options: string[] | null;
  default: number | string | [number, number];
  value: number | string | [number, number];
  changed: boolean;
  best_guess: boolean;
}

export interface Rule {
  id: string;
  name: string;
  note: string | null;
  layer: string | null;
  params: RuleParam[];
  enabled: boolean;
  reason: string | null;
  rev: number | null;
  n_changed: number;
}

export interface RuleGroup {
  key: string;
  title: string;
  desc: string;
  statutory: boolean;
  rules: Rule[];
}

export interface RulesResponse {
  version: string;
  groups: RuleGroup[];
  n_rules: number;
  n_params: number;
  n_changed: number;
  n_off: number;
}

export interface CostItem {
  key: string;
  label: string;
  unit: string;
  default: number;
  value: number;
  changed: boolean;
  best_guess: boolean;
}

export interface CostCategory {
  key: string;
  title: string;
  desc: string;
  status: "tcc" | "sourced";
  items: CostItem[];
}

export interface TonnageRow {
  type: string;
  deviation: string;
  tonnes: number;
  default_tonnes: number;
  cr_each: number;
  changed: boolean;
}

export interface CostsResponse {
  version: string;
  categories: CostCategory[];
  tonnage: TonnageRow[];
  rate_per_tonne_rs: number;
  n_tcc: number;
  tonnage_best_guess: boolean;
}

export interface EventSource {
  layer: string;
  layer_name: string;
  mode: SourceMode;
  confidence: number | null;
  source: string | null;
  feature?: string | null;
}

export interface RouteEvent {
  type: string;
  chainage_km: number | null;
  text: string;
  severity: "info" | "warning" | "error" | string;
  source: EventSource | null;
  data?: Record<string, unknown>;
}

export interface Tower {
  no: number;
  lon: number;
  lat: number;
  type: string;
  deflection_deg: number;
  turn: string;
  angle_point: boolean;
  chainage_km: number;
  span_ahead_m: number | null;
  reasons: string[];
  violations: string[];
  warnings: string[];
  extension_m: number;
  ground_z: number | null;
  offset_m?: number;
}

export interface Span {
  a: number;
  b: number;
  length_m: number;
  crossings: unknown[];
  min_clearance_m: number | null;
}

export interface RouteCost {
  total: number;
  towers: number;
  foundation: number;
  conductor: number;
  land_row: number;
  crops_trees: number;
  forest: number;
  crossings: number;
  environment: number;
}

export interface RouteStats {
  length_km: number;
  beeline_km: number;
  vs_beeline_pct: number;
  towers: number;
  angle_points: number;
  angle_towers?: number;
  suspension_towers: number;
  tension_towers: number;
  tower_mix: Record<string, number>;
  forest_ha: number;
  agri_ha: number;
  corridor_ha: number;
  crossings: Record<string, number>;
  crossings_total: number;
  violations: number;
  warnings: number;
  max_lateral_m: number;
  areas_ha?: Record<string, number>;
}

export interface Route {
  id: number;
  key: string;
  name: string;
  feasible: boolean;
  geometry: LonLat[];
  towers: Tower[];
  spans: Span[];
  cost: RouteCost;
  stats: RouteStats;
  events: RouteEvent[];
}

export interface Blocker {
  rule_id: string;
  rule: string;
  statutory: boolean;
  feature: string;
  from_km: number;
  to_km: number;
  band_cover_pct: number;
  source?: EventSource | null;
}

export interface SolveResult {
  feasible: boolean;
  routes: Route[];
  events_global: RouteEvent[];
  blocked?: { reach_km: number; blockers: Blocker[]; event: RouteEvent };
  line?: { kv: number; circuit: string; design_span_m: number; half_width_m: number; span_max_m: number; span_min_m: number };
  frame?: { length_km: number; half_width_km: number };
  rules_fired?: Record<string, number>;
  timings?: { total_s: number };
}

export interface ManifestOverride {
  key: string;
  rev: number | null;
  rule: string;
  param: string;
  value: unknown;
  reason: string | null;
  default: unknown;
  rule_id: string;
  statutory: boolean;
}

export interface ManifestSource {
  file: string | null;
  mode: SourceMode;
  layer: string;
  model: string | null;
  source: string | null;
  carried: boolean;
  features: number;
  producer: string;
  confidence: number | null;
  image_date: string | null;
  extracted_in: string | null;
  image_source: string | null;
  model_run_at: string | null;
  model_version: string | null;
}

export interface Manifest {
  rev: number;
  run_id: string;
  parent: string | null;
  overrides: ManifestOverride[];
  sources: ManifestSource[];
  notices: string[];
  ruleset_version: string;
  costs_version: string;
  engine_commit?: string;
  layers_version?: string;
  created_at: string;
  corridor: { a: LonLat; b: LonLat; a_name: string; b_name: string; band_km: number };
  line: Record<string, unknown>;
  [k: string]: unknown;
}

export interface Solve {
  id: number;
  run_id: string;
  created_at: string;
  feasible: boolean;
  result: SolveResult;
  manifest: Manifest;
  files: string[];
}

export interface CompareRow {
  label: string;
  prev: number | null;
  now: number | null;
  delta: number | null;
  unit: string;
}

export interface CompareResponse {
  parent: string | null;
  parent_rev?: number;
  rev?: number;
  rows?: { route: number; name: string; rows: CompareRow[] }[] | null;
  changes?: string[];
  parent_routes?: { id: number; geometry: LonLat[] }[];
}

export interface UploadInfo {
  id: number;
  filename: string;
  layers: string[];
  layer: string;
  fields: string[];
  feature_count: number;
  crs: string | null;
  geometry_type: string | null;
  samples: Record<string, string[]>;
  bounds: number[];
}

export interface Mapping {
  layer?: string | null;
  class_field?: string | null;
  class_map?: Record<string, string>;
  const_class?: string | null;
  name_field?: string | null;
  conf_field?: string | null;
}

export interface MappingCheck {
  vocabulary: string[];
  unknown_values: string[];
  bad_targets: string[];
  fallback_class: string;
  ok: boolean;
}

export interface ProcurementResponse {
  id: string;
  status: string;
  message: string;
}

export interface ImageryProduct {
  product: string;
  items: number;
  covered_pct: number;
  newest: string | null;
  oldest: string | null;
  fetchable: boolean;
  used_by: string[];
  sample: Record<string, unknown>[];
}

export interface ImageryCatalog {
  products: ImageryProduct[];
  credentials: { esri_token: boolean; planetary_computer_key: boolean };
}

export interface ModelStatus {
  name: string;
  layer: string;
  imagery: string[];
  gpu: boolean;
  needs_layers: string[];
  problems: string[];
  params: Record<string, unknown>;
}
