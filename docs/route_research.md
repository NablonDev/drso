# Route solver — algorithm research and recommendation (28 Sep 2026)

Screen 9 "Solve the route" in the Beeline flow. Target: ≤30 min on one L4 machine. Objective: minimise total ₹ cost with tower cost as the dominant term; all LULC rules as hard constraints.

## Confirmed by Vivek, 28 Sep
- **No whole-path constraints** (no caps on forest ha, crossing counts, etc.) → graph solve is exact; no Lagrangian/resource-constrained search needed.
- **Largest instance: 150 km × 15 km band.**

## Recommendation (rev 4)

Hierarchical **pylon-graph** solver, angle-aware, with DP tower spotting. Pass 1 covers the **whole 15 km band**, so no corridor-narrowing risk.

0. **Feasibility check (seconds)** — connected components on the tower-allowed mask with span-length connectivity. If start/end disconnected inside the band, stop and report the blocking feature and chainage range; switch to corridor-guided mode (see "Massive no-go").
1. **Rule compile (GPU, once per revision, cached)** — 57 rules → tower-forbidden mask, conductor-forbidden mask (kept separate: wetland/river/road span-over but no tower), per-class distance transforms for setbacks, crossing features with orientation, ₹ site-cost raster, ₹/m² strip-cost raster at 10 m. 22.5 M cells.
2. **Crossing gateways** — every crossable line feature (rail, NH/SH, pipeline, power, telecom, canal) gets candidate crossing nodes every ~50 m inside the band; each carries the angle window (60–90°), max span, forced tower type both sides, rejected up-front if either landing is tower-forbidden. Parallel-run rules as lateral buffers.
3. **Pass 1 — pylon graph, 50 m sites, full band** — vertices = legal tower sites (+ buffered no-go polygon corners as extra angle-point candidates); edges = spans within [Smin, Smax] that pass conductor mask + gateway rules + quick DEM-profile sag check; state = (site, incoming heading); tower type/cost from deflection angle (IS 802 bands, max 60°); edge cost = conductor + **oriented strip integral of 2×CHW RoW cost**; vertex cost = tower + foundation (soil, slope) + 100% land. **DAG ordered by chainage along the beeline** → layer-wise DP on GPU; Bellman-Ford fallback if a detour breaks monotonicity. Gives R1. R2–R4 by re-running with cells near prior routes penalised (LION k-diverse method).
4. **Pass 2 — 20 m sites in a 500 m band around each route**, same graph model, deflection change limited to ±60°.
5. **Tower spotting DP (seconds)** — along each alignment on DEM: height/extension, catenary clearance, weight span, crossing clearances/tower types, section length (per-km cost in graph + repair pass). Re-rank R1–R4 on spotted cost.

Coarse raster LCP (cell, heading) kept as a UI preview, and as the corridor axis in corridor-guided mode.

### Sizing, 150 km × 15 km, L4 24 GB (spans 250–450 m, 400 kV class)

| Stage | Size | Est. time |
|---|---|---|
| 1 Rule compile | 22.5 M cells × ~20 classes | 1–2 min (cached) |
| 3 Pass 1 | 900k sites × ~88 fwd successors = 79 M edges, ~7 B transitions, 630 MB | 1–3 min |
| R2–R4 | 3 re-runs of pass 1 | 3–9 min |
| 4 Pass 2 | 187k sites × ~190 successors (±60°) = ~100 M edges, ~20 B transitions, 0.8 GB | 2–5 min per route |
| 5 Spotting | 400–500 towers | seconds |

Worst case ~20–25 min for 4 routes. Levers: pass 1 at 75 m; 3 routes; run the four pass-2 solves concurrently on GPU. 765 kV spans (450–550 m) add ~50% to pass 2, still in budget. Rules-only revision = solve-only re-run (~24 min figure in the UI doc).

### Constraint → mechanism

| Constraint | Mechanism |
|---|---|
| Tower no-go (buildings, wetland 50 m, monument 100 m, airport 1.5 km approach) | Distance transform → tower mask; nodes removed. Hard. |
| Conductor no-go (built-up, approach-surface height cap) | Conductor mask; edges removed. Hard. |
| Setbacks / distance from feature | Per-class distance transforms, tower and conductor separately (pipeline 30 m tower; river spans over). |
| RoW strip cost | Oriented box integral per span, not a line integral. Forest = last preference, priced not banned. |
| Line-feature crossings | Gateway nodes with angle window, span limit, forced tower type; parallel-run lateral buffer. |
| Elevation | Slope class → foundation cost / no-go; DEM profile under each candidate span for quick clearance in pass 1/2; full clearance + height in spotting. |
| Section length | Not exact in a graph: per-km cost + repair pass after spotting. |

### More compute — what it buys
Exact solver already; more time = precision, not a new method. Ranked: (1) **joint routing + spotting with tower height as a third state dimension** (5–10× pass 2) — the one real tower-cost gain, phase 2; (2) whole-region search — only for blocked bands; (3) Pareto front cost/forest/clearance-risk — only if AESL asks; (4) Monte Carlo over LULC confidence; (5) 10 m pass 2 — marginal.

### Massive no-go (e.g. Wild Ass Sanctuary, ~4,950 km², between Lakadia and Halvad)
Sanctuary + 10 km default ESZ blocks the whole 15 km band for 40–60 km of chainage; the beeline DAG has no path. Handling: (a) feasibility check fails fast and names the blocker; (b) **corridor-guided mode** — 50 m raster Dijkstra over a wide extent (200×200 km = 16 M cells, seconds) finds the detour topology (south via Surajbari–Maliya–Morbi); chainage along that corridor becomes the DAG axis; pylon graph solved in a band around it; (c) **rule mode toggle** for WLS: hard no-go vs priced with NBWL clearance + delay cost (open AESL item "avoid ESZ or carry NBWL clearance") — produce both routes in the R1–R4 table. Also: Kutch is GIB range (Ranjitsinh order — diverters/undergrounding, still to verify on sci.gov.in); Little Rann saline/seasonally flooded — foundation-cost layer decides the through-route cost. Reference: Adani WRSS-XXI ESIA kept Lakadia–Bhuj 765 kV and Bhachau LILO 8–24 km clear of the sanctuary.

### Existing lines in the band
Not a compliance oracle: rules moved (CEA 2023, RoW 2025, airport 20 km, Forest 2023, MoP 2024), ground moved (built-up, ESZ, wetlands), different endpoints/voltage. Uses, in value order:
1. **Validation set** — replay existing lines through the rule compile; violations under new rules are expected, violations under old rules flag LULC extraction errors or as-built drift. Regression test for screen 8.
2. **Parallel-run preference as a cost** — discount on strip cost and Delay & time inside ~100–300 m either side, outside the rank-5 parallel buffer. AESL sets the discount; zero disables.
3. **Corridor axis** in corridor-guided mode when a line already goes round the blocker.
4. Evidence for the `PARALLEL_RUN` decision-log event.
Must not: copy alignment; share crossings (own gateway checks); skip the line as a rank-5 constraint. Risk: corridor consolidation = common-mode failure (cyclones in Kutch) — check CEA planning manual reliability guidance; ask AESL for a parallel-length cap or penalty.

## Methods assessed

| Method | Tower cost modelled | Verdict |
|---|---|---|
| Raster LCP (Dijkstra/A*) — EPRI-GTC practice | No (₹/m only, free turns) | UI preview + corridor-guided axis |
| Any-angle / visibility graph over buffered no-go polygons | No, but minimal turns | Adopted for angle-point candidates |
| LCP + vertex siting + spotting (Queiroz 2019) | Yes, decoupled | Sub-optimal, ~5 min/40 km |
| Improved A* w/ turn penalty (Chen 2026) | No — turn angle proxy | Rejected as solver; see note |
| Pylon graph + line graph (Piveteau 2017; Wiedemann & Adjiashvili 2021, LION) | Yes, exact | Core of the solver |
| Tower spotting DP (Medeiros 2021) | Yes, on fixed alignment | Final stage |
| GIS + GA (Qin et al., Array 17, 2023) | No — cell-cost sum only | Rejected; see note |
| Weighted-region continuous SP (Mitchell–Papadimitriou) | No | Exact but impractical at 150 km |
| MILP over sparse angle points | Yes | Redundant with DAG DP |
| Multi-objective Pareto search | — | Only if AESL wants "least forest" vs "least cost" routes |
| GA/PSO/ACO/ABC, RRT*, RL/learned heuristics | Varies | Rejected: no optimality/auditability |

### Note — Chen, Ma, Guo, Zhou, Pan & Sun (2026), Sci. Rep. 16:21867, doi 10.1038/s41598-026-53156-5

Suzhou Electric Power Design Institute. 220 kV, ~30 km planned line in Suzhou (flat farmland); solved on a **5 km subset** only. UAV oblique photogrammetry (DJI M300, RTK) → Pix4D → grid 5–30 m. Step cost g = w_d·distance + w_t·|Δz| + w_l·crossing penalty + w_c·turn angle over three nodes; h = 3D Euclidean. Weights empirical. Obstacles soft. C++; i9-13900K + RTX 4090; **no runtime reported**.

Results: proposed 5,482 m / 1.9 m / 6 crossings / 51.8° total turn; Dijkstra 5,341 / 2.4 / 9 / 72.6°; RRT 5,576 / 2.2 / 6 / 96.8°.

Issues: (1) turn cost without heading in state breaks optimality; (2) h unweighted vs weighted g — admissibility unstated; (3) distance-only A* (5,318 m) ≠ Dijkstra (5,341 m) on same graph; (4) total turn angle ≠ angle-tower count/type; (5) no spans, towers, crossing-angle rule or hard no-go. Takeaway: corner penalty right in spirit; done properly as (site, heading) state and exact tower-type cost.

### Note — Qin, Li, Ding, Zhao & Meng (2023), Array 17:100266, doi 10.1016/j.array.2022.100266

Yunnan University. 500 kV, ~150 km, Jiantang → Tai'an. AHP-weighted 0–5 raster at 150 m; avoidance = cost 9999 (soft). Chromosome = turning points on perpendiculars to the beeline; pop 200, 100 gens. ArcGIS C# add-in. No runtime.

Table 5: built 147.9 km / 251,790 / 25 avoidance crossings; **GA 143.1 km / 31,736 / 3; Dijkstra 147.9 km / 11,429 / 1**. Dijkstra beats GA; the "GA superior" conclusion is not supported by the paper's own numbers. Takeaways: perpendicular-section encoding = our chainage DAG; no tower/span/angle cost; soft no-go; 150 m too coarse.

## Open inputs
Voltage/tower family and span limits (assumed 250–450 m); DEM source (need ≤10 m; SRTM 30 m inadequate); max section length rule; TCC-signed tower/foundation rates; MoP 2024 compensation in or out; WLS mode (avoid vs NBWL clearance); parallel-run discount and any parallel-length cap.

## Next
Engine spec: data contracts per stage, exact cost functions, GPU kernel layout for layer-wise DP, decision-log event schema (TURN_TO_AVOID, CROSSING, PARALLEL_RUN, LAYER_EMPTY, LOW_CONF, NO_IMAGERY, BAND_BLOCKED).