#!/usr/bin/env python
"""Benchmark the solve on a synthetic 150 km x ±7.5 km corridor with GPU memory capped at 24 GB (L4)."""
import sys, time, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, geopandas as gpd, torch
from shapely.geometry import Point, LineString, box
from beeline_engine.frame import Frame
from beeline_engine.pipeline import run_solve, SolveInput, LayerInput

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 24e9 / total))
A, B = (71.0, 22.5), (72.45, 22.62)
f = Frame(A, B, 7500, 500, 20)
rng = np.random.default_rng(7)
def fg(g):
    from shapely import affinity
    g = affinity.rotate(g, f.bearing_rad, origin=(0, 0), use_radians=True)
    return f.geom_to_lonlat(affinity.translate(g, f.A[0], f.A[1]))
L = f.length_m
villages = [fg(Point(rng.uniform(2000, L - 2000), rng.uniform(-7000, 7000)).buffer(rng.uniform(150, 600))) for _ in range(400)]
bldg = [fg(Point(rng.uniform(0, L), rng.uniform(-7500, 7500)).buffer(8)) for _ in range(20000)]
lines = []
for _ in range(12):
    x = rng.uniform(5000, L - 5000); a = math.radians(rng.uniform(40, 140))
    lines.append(fg(LineString([(x - 9000 * math.cos(a), -9000 * math.sin(a)), (x + 9000 * math.cos(a), 9000 * math.sin(a))])))
towers = [fg(Point(p)) for l in lines for p in [ (0,0) ]][:0]
rivers = []
for _ in range(4):
    x = rng.uniform(10000, L - 10000)
    rivers.append(fg(box(x - 120, -9000, x + 120, 9000))); rivers.append(fg(LineString([(x, -9000), (x, 9000)])))
def g(geoms, cls):
    return gpd.GeoDataFrame({"class": [cls] * len(geoms), "name": [None] * len(geoms), "confidence": [0.9] * len(geoms)}, geometry=geoms, crs=4326)
layers = {"clusters": g(villages, "builtup_cluster"), "buildings": g(bldg, "building"), "power": g(lines, "power_line"),
          "water": g(rivers, "river")}
li = {k: LayerInput(k, v, "auto", "bench", 0.9, None) for k, v in layers.items()}
t = time.time()
res = run_solve(SolveInput(A, B, 7.5, 765, "dc", li, routes_wanted=4, dem_dir=None))
wall = time.time() - t
print(f"beeline {f.length_m/1000:.1f} km, band ±7.5 km; feasible={res['feasible']}; wall {wall:.1f}s; timings {res['timings']}")
for r in res.get("routes", []):
    print(r["id"], r["name"], r["stats"]["length_km"], r["stats"]["towers"], r["timings"])
if torch.cuda.is_available():
    print("peak GPU MB", round(torch.cuda.max_memory_allocated() / 1e6))
