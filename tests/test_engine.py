"""Engine tests on synthetic corridors (no database, no network). Run: pytest tests/test_engine.py"""
from __future__ import annotations

import heapq
import math

import geopandas as gpd
import numpy as np
import pytest
import torch
from shapely.geometry import LineString, Point, box

from beeline_engine.compile import compile_rules
from beeline_engine.config import CIRCUITS, VOLTAGES, LineSpec, resolve_costs, resolve_rules
from beeline_engine.frame import Frame
from beeline_engine.pipeline import LayerInput, SolveInput, run_solve
from beeline_engine.solver import (SolverParams, edge_costs, gantry_seeds, make_grid, prepare_strip, site_costs, solve,
                                   tower_type_cost, transition_matrix, _Rasters)

A = (72.00, 22.30)
B = (72.20, 22.32)   # ~20.7 km


def frame(a=A, b=B, hw=2000, res=20):
    return Frame(a, b, hw, 500, res)


def gdf(geoms, cls, name=None):
    return gpd.GeoDataFrame({"class": [cls] * len(geoms), "name": [name] * len(geoms), "confidence": [0.9] * len(geoms)},
                            geometry=geoms, crs=4326)


def frame_geom(f: Frame, g):
    """frame-coordinate shapely geom -> lon/lat"""
    from shapely import affinity
    g = affinity.rotate(g, f.bearing_rad, origin=(0, 0), use_radians=True)
    g = affinity.translate(g, f.A[0], f.A[1])
    return f.geom_to_lonlat(g)


def solve_input(layers: dict, a=A, b=B, band_km=2.0, kv=765, circuit="dc", overrides=None, routes=1):
    li = {k: LayerInput(k, v, "auto", "test", 0.9, None) for k, v in layers.items()}
    return SolveInput(a=a, b=b, band_km=band_km, kv=kv, circuit=circuit, layers=li, routes_wanted=routes,
                      rule_overrides=overrides or {}, dem_dir=None)


def towers_frame(f: Frame, route):
    xs = [t["x_utm"] for t in route["towers"]]
    ys = [t["y_utm"] for t in route["towers"]]
    fx, fy = f.utm_to_frame(np.array(xs), np.array(ys))
    return list(zip(fx, fy))


# ---------------------------------------------------------------------------------------
def test_config_resolves_every_voltage_and_circuit():
    for kv in VOLTAGES:
        for c in CIRCUITS:
            spec = LineSpec(kv, c)
            r = resolve_rules(spec)
            k = resolve_costs(spec)
            assert r["corridor_half_width"].params["half_width_m"] > 0
            assert r["design_span"].params["plain_m"] > 0
            assert all(v > 0 for v in k["tonnage"].values())
            assert k["n_circuits"] == {"sc": 1, "dc": 2, "qc": 4}[c]


def test_frame_roundtrip():
    f = frame()
    x, y = f.frame_to_utm([0, f.length_m, 1000], [0, 0, 300])
    fx, fy = f.utm_to_frame(x, y)
    assert np.allclose(fx, [0, f.length_m, 1000], atol=1e-6)
    assert np.allclose(fy, [0, 0, 300], atol=1e-6)
    lon, lat = f.frame_to_lonlat(0, 0)
    assert abs(lon - A[0]) < 1e-7 and abs(lat - A[1]) < 1e-7


def test_open_ground_is_straight():
    res = run_solve(solve_input({}))
    assert res["feasible"]
    r = res["routes"][0]
    assert r["stats"]["vs_beeline_pct"] < 0.5
    assert r["stats"]["angle_points"] == 0
    # tension towers only at the dead ends and the 5 km section points
    assert r["stats"]["tension_towers"] <= 2 + math.ceil(r["stats"]["length_km"] / 5)
    spans = [s["length_m"] for s in r["spans"][1:-1]]
    assert max(spans) <= 450 + 1e-6
    total = sum(v for k, v in r["cost"].items() if k != "total")
    assert abs(total - r["cost"]["total"]) < 1.0


def test_avoids_village_cluster_with_buffers():
    f = frame()
    village = frame_geom(f, Point(f.length_m / 2, 0).buffer(600))
    res = run_solve(solve_input({"clusters": gdf([village], "builtup_cluster", "Testpur")}))
    assert res["feasible"]
    r = res["routes"][0]
    vf = Point(f.length_m / 2, 0).buffer(600)
    for x, y in towers_frame(f, r):
        assert vf.distance(Point(x, y)) >= 120 - 0.5          # tower buffer 120 m
    line = LineString(towers_frame(f, r))
    assert line.distance(vf) >= 120 - 0.5                     # conductor buffer 120 m
    assert any(e["type"] == "BEELINE_BLOCKER" for e in res["events_global"])
    assert any(e["type"] == "TURN_TO_AVOID" and "Testpur" in e["text"] for e in r["events"])


def test_rule_switched_off_lets_route_go_straight():
    f = frame()
    village = frame_geom(f, Point(f.length_m / 2, 0).buffer(600))
    res = run_solve(solve_input({"clusters": gdf([village], "builtup_cluster")},
                                overrides={"clusters": {"enabled": False}}))
    assert res["feasible"]
    assert res["routes"][0]["stats"]["vs_beeline_pct"] < 0.5
    assert any(e["type"] == "RULE_OFF" for e in res["events_global"])


def test_railway_crossing_angle_and_span():
    f = frame()
    # railway crossing the band at 30° to the bee-line — the route must turn to cross at >= 70°
    c = f.length_m / 2
    rail = frame_geom(f, LineString([(c - 4000 * math.cos(math.radians(30)), -4000 * math.sin(math.radians(30))),
                                     (c + 4000 * math.cos(math.radians(30)), 4000 * math.sin(math.radians(30)))]))
    res = run_solve(solve_input({"rail": gdf([rail], "railway", "Test line")}))
    assert res["feasible"]
    r = res["routes"][0]
    xs = [cr for s in r["spans"] for cr in s["crossings"] if cr["rule"] == "railway"]
    assert len(xs) == 1
    assert xs[0]["angle_deg"] >= 70 - 1e-6
    assert xs[0]["span_m"] <= 300 + 1e-6                 # min(300 m, 80 % of 400 m)
    a, b = next((s["a"], s["b"]) for s in r["spans"] if s["crossings"])
    types = {t["no"]: t["type"] for t in r["towers"]}
    assert types[a] in ("D45", "D60") and types[b] in ("D45", "D60")   # crossing tower type D


def test_wide_river_needs_long_span():
    f = frame()
    c = f.length_m / 2
    river = frame_geom(f, box(c - 175, -3000, c + 175, 3000))     # 350 m of water + 150 m banks = 650 m
    line = frame_geom(f, LineString([(c, -3000), (c, 3000)]))
    res = run_solve(solve_input({"river": gdf([river, line], "river", "Test river")}))
    assert res["feasible"]
    r = res["routes"][0]
    long_spans = [s for s in r["spans"] if s["length_m"] > 450]
    assert long_spans and all(any(cr["rule"] == "river" for cr in s["crossings"]) for s in long_spans)
    for x, y in towers_frame(f, r)[1:-1]:
        assert not (c - 175 - 150 + 0.5 < x < c + 175 + 150 - 0.5)


def test_crossings_take_angle_towers_both_sides():
    """Railway, national-highway and river crossings: both towers of the crossing span are angle towers
    (at least B1, the cheapest one that fits the deflection) (Vivek, 6 Oct 2026)."""
    f = frame()
    c = f.length_m / 2
    stream = frame_geom(f, LineString([(c, -3000), (c, 3000)]))          # minor river: a line, no width
    nh = frame_geom(f, LineString([(c - 4000, -3000), (c - 4000, 3000)]))
    res = run_solve(solve_input({"river": gdf([stream], "river", "Test stream"), "roads": gdf([nh], "nh", "NH 41")}))
    assert res["feasible"]
    r = res["routes"][0]
    types = [t["type"] for t in r["towers"]]
    hits = [(i, cr["rule"]) for i, s in enumerate(r["spans"]) for cr in s["crossings"] if cr["rule"] in ("river", "national_highway")]
    assert {h[1] for h in hits} == {"river", "national_highway"}
    for i, rule in hits:
        assert types[i] != "A" and types[i + 1] != "A", (rule, types[i], types[i + 1])


def test_no_stretch_over_5_km_without_an_angle_tower():
    """Section point interval (plain 5 km): angle towers never more than 5 km apart, even on a straight line."""
    res = run_solve(solve_input({}))
    r = res["routes"][0]
    cum = [0.0]
    for s in r["spans"]:
        cum.append(cum[-1] + s["length_m"])
    idx = [i for i, t in enumerate(r["towers"]) if t["type"] != "A"]
    assert idx[0] == 0 and idx[-1] == len(r["towers"]) - 1
    assert max(cum[b] - cum[a] for a, b in zip(idx, idx[1:])) <= 5000.0 + 1e-6


class _FakeDem:
    """DEM stand-in: height (m) as a function of frame x along A->B."""
    ok = True

    def __init__(self, f: Frame, zfun):
        self.f, self.zfun = f, zfun

    def sample(self, lon, lat):
        ux, uy = self.f.lonlat_to_utm(np.asarray(lon, float), np.asarray(lat, float))
        fx, _ = self.f.utm_to_frame(ux, uy)
        return self.zfun(np.asarray(fx, float)).astype(float)


def _solve_on_dem(monkeypatch, zfun, layers=None, overrides=None):
    from beeline_engine import pipeline as pl
    holder = {}

    def fake_sampler(tiles, bb):
        holder["dem"] = _FakeDem(frame(), zfun)
        return holder["dem"]
    monkeypatch.setattr(pl.demmod, "ensure_tiles", lambda bb, d: [])
    monkeypatch.setattr(pl.demmod, "DemSampler", fake_sampler)
    inp = solve_input(layers or {}, overrides=overrides)
    inp.dem_dir = "unused"
    return run_solve(inp)


def test_ridge_gets_a_tower_instead_of_a_short_clearance(monkeypatch):
    """A 70 m knife-edge ridge mid-span: +18 m extensions cannot clear it, so a tower goes on it."""
    f = frame()
    c = f.length_m / 2 + 225.0          # between two grid towers (~450 m spans on the flat)
    # hilly span limit lifted so only the clearance re-spotting can put a tower on the ridge
    res = _solve_on_dem(monkeypatch, lambda x: np.clip(70.0 - np.abs(x - c) * 0.9, 0, None),
                        overrides={"design_span": {"params": {"hilly_m": 500}}})
    r = res["routes"][0]
    assert all((s["min_clearance_m"] is None or s["min_clearance_m"] >= 15.0 - 1e-6) for s in r["spans"]), \
        min(s["min_clearance_m"] for s in r["spans"] if s["min_clearance_m"] is not None)
    added = [t for t in r["towers"] if "clearance" in t["reasons"]]
    assert added and any(abs(t["chainage_km"] * 1000 - c) < 120 for t in added)
    assert any(e["type"] == "CLEARANCE_TOWER" for e in r["events"])


def test_hilly_ground_takes_hilly_spans_and_sections(monkeypatch):
    """Rolling hills steeper than 10°: spans at most the hilly design span (300 m at 765 kV) and
    tension towers at most 3 km apart there; flat ground keeps 5 km."""
    f = frame()
    hills = lambda x: np.where(x > f.length_m / 2, 40.0 * np.abs(np.sin(x / 150.0)), 0.0)    # ~15° slopes
    res = _solve_on_dem(monkeypatch, hills)
    r = res["routes"][0]
    cum = [0.0]
    for s in r["spans"]:
        cum.append(cum[-1] + s["length_m"])
    half = f.length_m / 2
    hilly_spans = [s for i, s in enumerate(r["spans"][1:-1], 1) if cum[i] > half + 500]
    assert hilly_spans and max(s["length_m"] for s in hilly_spans) <= 300.0 + 1e-6
    idx = [i for i, t in enumerate(r["towers"]) if t["type"] != "A"]
    gaps = [(cum[a], cum[b] - cum[a]) for a, b in zip(idx, idx[1:])]
    assert max(g for start, g in gaps if start > half + 500) <= 3000.0 + 1e-6


def test_band_fully_blocked_is_reported():
    f = frame()
    c = f.length_m / 2
    wall = frame_geom(f, box(c - 1500, -6000, c + 1500, 6000))
    res = run_solve(solve_input({"wildlife_sanctuary": gdf([wall], "wildlife_sanctuary", "Test WLS")}))
    assert not res["feasible"]
    b = res["blocked"]
    assert b["blockers"] and b["blockers"][0]["rule_id"] == "protected_areas"
    assert b["blockers"][0]["feature"] == "Test WLS"
    assert abs(b["reach_km"] - (c - 1500) / 1000) < 1.0


def test_terminal_yard_is_carved_out():
    f = frame()
    yard = frame_geom(f, box(-350, -350, 350, 350))          # A sits in the middle of a 700 m yard
    res = run_solve(solve_input({"substations": gdf([yard], "substation", "Start S/S")}))
    assert res["feasible"]
    r = res["routes"][0]
    first = towers_frame(f, r)[1]
    assert not box(-350, -350, 350, 350).contains(Point(first))


def test_four_routes_are_distinct_objectives():
    f = frame()
    village = frame_geom(f, Point(f.length_m * 0.4, 300).buffer(500))
    res = run_solve(solve_input({"clusters": gdf([village], "builtup_cluster")}, routes=4))
    assert res["feasible"] and len(res["routes"]) == 4
    keys = [r["key"] for r in res["routes"]]
    assert keys == ["least_cost", "fewest_angles", "closest_beeline", "least_sensitive"]
    costs = [r["cost"]["total"] for r in res["routes"]]
    assert costs[0] <= min(costs) * 1.02    # route 1 is (near) the cheapest true cost


@pytest.mark.parametrize("kv,circuit", [(400, "dc"), (220, "sc"), (132, "qc")])
def test_other_voltages_solve(kv, circuit):
    res = run_solve(solve_input({}, kv=kv, circuit=circuit))
    assert res["feasible"]
    r = res["routes"][0]
    design = res["line"]["design_span_m"]
    assert max(s["length_m"] for s in r["spans"][1:-1]) <= design * 1.125 + 1e-6


def test_dp_is_optimal_against_dijkstra():
    """Brute-force Dijkstra over the same (site, heading) graph gives the same optimum."""
    f = Frame(A, (72.04, 22.302), 400, 500, 20)
    blob = frame_geom(f, Point(f.length_m / 2, 0).buffer(150))
    spec = LineSpec(220, "sc")
    rules = resolve_rules(spec)
    costs = resolve_costs(spec)
    comp = compile_rules(f, {"clusters": gdf([blob], "builtup_cluster")}, rules, costs, 220, 17.5, 150, 350)
    tc = tower_type_cost(costs, 220)
    bands = {k: rules["tower_type_bands"].params[k] for k in ("A", "B1", "B2", "C1", "C2", "D45", "D60")}
    p = SolverParams(50, 100, 393.75, 60, bands, tc)
    out = solve(comp, p)
    assert out.feasible
    g = make_grid(comp, p)
    S, angle_only = site_costs(comp, g, p)
    T, TA = transition_matrix(g, p)
    strip = prepare_strip(comp, p)
    seedA = gantry_seeds(comp, g, p, S, strip, False, tc["D60"])
    seedB = gantry_seeds(comp, g, p, S, strip, True, tc["D60"])
    ras = _Rasters(comp, p, torch.device("cuda" if torch.cuda.is_available() else "cpu"), strip)
    ras.set_river_cols(comp, g, p)
    n_s = int(math.ceil(p.span_max_m / 10)) + 1
    E = edge_costs(comp, g, p, ras, 0, g.nc, n_s).cpu().numpy()
    K = len(g.offsets)
    # Dijkstra over states (c, r, k_in); start states come from gantry seeds
    INF = float("inf")
    dist = {}
    pq = []
    for c in range(g.nc + 1):
        for r in range(g.nr):
            if np.isfinite(seedA[c, r]):
                # tower placed as first dead-end; leaving with any heading k
                for k in range(K):
                    dc, dr = g.offsets[k]
                    c2, r2 = c + dc, r + dr
                    if c >= g.nc or c2 > g.nc or not (0 <= r2 < g.nr) or not np.isfinite(E[c, r, k]):
                        continue
                    d = seedA[c, r] + E[c, r, k]
                    if d < dist.get((c2, r2, k), INF):
                        dist[(c2, r2, k)] = d
                        heapq.heappush(pq, (d, c2, r2, k))
    best = INF
    done = set()
    while pq:
        d, c, r, k = heapq.heappop(pq)
        if (c, r, k) in done:
            continue
        done.add((c, r, k))
        if np.isfinite(seedB[c, r]):
            best = min(best, d + seedB[c, r])
        if c >= g.nc or not np.isfinite(S[c, r]):
            continue
        Trow = TA[k] if angle_only[c, r] else T[k]
        for k2 in range(K):
            if not np.isfinite(Trow[k2]):
                continue
            dc, dr = g.offsets[k2]
            c2, r2 = c + dc, r + dr
            if c2 > g.nc or not (0 <= r2 < g.nr) or not np.isfinite(E[c, r, k2]):
                continue
            nd = d + Trow[k2] + S[c, r] + E[c, r, k2]
            if nd < dist.get((c2, r2, k2), INF):
                dist[(c2, r2, k2)] = nd
                heapq.heappush(pq, (nd, c2, r2, k2))
    assert math.isfinite(best)
    assert abs(best - out.cost_rs) / best < 1e-4, (best, out.cost_rs)


def test_airport_prohibition_reaches_from_outside_the_frame():
    """A runway beyond the raster still forbids towers within the prohibition distance."""
    f = frame()
    x = f.length_m / 2
    out_top = f.half_width_m + f.margin_m + 500                       # 500 m beyond the raster edge
    rw = frame_geom(f, box(x - 1000, out_top, x + 1000, out_top + 45))
    rules = resolve_rules(LineSpec(765, "dc"))
    costs = resolve_costs(LineSpec(765, "dc"))
    comp = compile_rules(f, {"approach": gdf([rw], "runway")}, rules, costs, 765, 30, 150, 400)
    d = float(rules["airport_prohibition"].params["distance_m"])
    col, row = f.frame_to_pixel(x, out_top - d + 300)                # inside the frame, inside the distance
    assert comp.tower_forbidden[int(row), int(col)] and comp.conductor_forbidden[int(row), int(col)]
    col, row = f.frame_to_pixel(x, out_top - d - 300)                # inside the frame, beyond the distance
    assert not comp.tower_forbidden[int(row), int(col)]
    assert comp.rules_fired.get("airport_prohibition")
