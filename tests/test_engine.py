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
from beeline_engine.config import CIRCUITS, TOWER_TYPES, VOLTAGES, LineSpec, resolve_costs, resolve_rules
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
            assert set(k["towers"]) == set(TOWER_TYPES)
            prices = [k["towers"][t]["price_rs"] for t in TOWER_TYPES]
            assert all(p > 0 for p in prices) and prices == sorted(prices)     # dearer with deviation
            assert k["line_rs_per_m"] > 0
    assert CIRCUITS == ("dc",)
    # Route Cost Parameters 3: 765 kV D/C A tower ₹0.4765 Cr, line 2.349 ₹Cr/km
    k = resolve_costs(LineSpec(765, "dc"))
    assert abs(k["towers"]["A"]["price_rs"] - 0.4765e7) < 1 and abs(k["line_rs_per_m"] * 1000 / 1e7 - 2.349) < 1e-9
    k = resolve_costs(LineSpec(765, "dc"), {"tower_price": {"A": 0.5}, "crops": {"x": 1}})
    assert abs(k["towers"]["A"]["price_rs"] - 0.5e7) < 1 and "crops" not in k


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
    """300 m of water + 150 m bank buffers each side: a single span of 600 m or more, longer than the
    370 m normal span. TSD: one span ≤ 720 m (60 % of the 1,200 m adjacent-span allowance) and the
    wind span ≤ 400 m, so the crossing towers' other spans are short and the towers are tension type."""
    f = frame()
    c = f.length_m / 2
    river = frame_geom(f, box(c - 150, -3000, c + 150, 3000))
    line = frame_geom(f, LineString([(c, -3000), (c, 3000)]))
    res = run_solve(solve_input({"river": gdf([river, line], "river", "Test river")}))
    assert res["feasible"]
    r = res["routes"][0]
    S, T = r["spans"], r["towers"]
    long_ix = [i for i, s in enumerate(S) if s["length_m"] > 370]
    assert long_ix and all(any(cr["rule"] == "river" for cr in S[i]["crossings"]) for i in long_ix)
    for i in long_ix:
        assert S[i]["length_m"] <= 720 + 1e-6
        assert (S[i - 1]["length_m"] + S[i]["length_m"]) / 2 <= 400 + 1e-6
        assert (S[i]["length_m"] + S[i + 1]["length_m"]) / 2 <= 400 + 1e-6
        assert T[i]["type"] != "A" and T[i + 1]["type"] != "A"
    for x, y in towers_frame(f, r)[1:-1]:
        assert not (c - 150 - 150 + 0.5 < x < c + 150 + 150 - 0.5)


def test_crossings_take_angle_towers_both_sides():
    """National-highway crossings: both towers of the crossing span are angle towers (at least B1).
    Minor river crossings may use A, B or C (Rule_Table 38, 8 Oct 2026), so no tension is forced there."""
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
        if rule == "national_highway":
            assert types[i] != "A" and types[i + 1] != "A", (rule, types[i], types[i + 1])
        else:
            assert not any(x.startswith("crossing:river") for x in r["towers"][i]["reasons"] if types[i] == "A")


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
    # the search itself (terrain in the span cost) or the spotting puts a tower at the crest
    assert any(abs(t["chainage_km"] * 1000 - c) < 120 for t in r["towers"])
    assert not any("ground_clearance" in (t["warnings"] or []) for t in r["towers"])


def test_search_takes_the_cheaper_of_hill_and_detour(monkeypatch):
    """A steep 120 m hill on the bee-line, flat ground to either side: the search prices the terrain
    (clearance, extensions, tension towers where the ground rises > 10 m between towers) and chooses
    between climbing and going round on cost (TSD figures, 8 Oct 2026)."""
    f = frame()
    c = f.length_m / 2

    class HillDem(_FakeDem):
        def sample(self, lon, lat):
            ux, uy = self.f.lonlat_to_utm(np.asarray(lon, float), np.asarray(lat, float))
            fx, fy = self.f.utm_to_frame(ux, uy)
            d = np.hypot((np.asarray(fx) - c) / 2500.0, np.asarray(fy) / 700.0)
            return (120.0 * np.clip(1 - d, 0, None) * (1 + 0.5 * np.sin(np.asarray(fx) / 90.0))).astype(float)
    from beeline_engine import pipeline as pl
    from shapely import affinity
    monkeypatch.setattr(pl.demmod, "ensure_tiles", lambda bb, d: [])
    monkeypatch.setattr(pl.demmod, "DemSampler", lambda tiles, bb: HillDem(frame(), None))

    def solve(layers):
        inp = solve_input(layers)
        inp.dem_dir = "unused"
        return run_solve(inp)["routes"][0]
    free = solve({})
    blob = frame_geom(f, affinity.scale(Point(c, 0).buffer(1), 2500 * 0.85, 700 * 0.85))
    detour = solve({"clusters": gdf([blob], "builtup_cluster")})
    assert max(t["ground_z"] or 0 for t in detour["towers"]) < 20          # the forced route really goes round
    # TSD clearance (18.15 m, 32.6 m attachment, extensions ≤ +6 m on A/B/C) makes the undulating hill
    # dear: the free search goes round too, within the search's estimate of the forced detour
    assert max(t["ground_z"] or 0 for t in free["towers"]) < 20
    assert free["cost"]["total"] <= detour["cost"]["total"] * 1.015, (free["cost"]["total"] / 1e7, detour["cost"]["total"] / 1e7)


def test_hilly_ground_takes_tension_towers_and_longer_spans(monkeypatch):
    """Hilly = the ground rises more than 10 m between adjacent towers (Route_Rules 3, 8 Oct 2026):
    both towers of such a span are tension towers (B1 at least) and tension towers are at most 10 spans
    / 3 km apart there. A steady 6 % grade over the second half (about 22 m per 370 m span) is hilly; the
    flat first half is not. No span exceeds the 370 m TSD normal span."""
    f = frame()
    half = f.length_m / 2
    grade = lambda x: np.where(x > half, (x - half) * 0.06, 0.0)
    res = _solve_on_dem(monkeypatch, grade)
    r = res["routes"][0]
    inner = r["spans"][1:-1]
    assert max(s["length_m"] for s in inner) <= 370.0 + 1e-6
    T = r["towers"]
    up = [t for t in T if t["chainage_km"] * 1000 > half + 400 and t["type"] != "G"]
    assert up and all(t["type"] != "A" for t in up), [t["type"] for t in up]
    assert any("hilly" in t["reasons"] for t in up)
    flat = [t for t in T if t["chainage_km"] * 1000 < half - 400 and t["type"] not in ("G",)]
    assert any(t["type"] == "A" for t in flat)


def test_design_span_is_the_hard_maximum_and_editable():
    """Rule 'Design span' caps every span in the search and the spotting (gantry slack spans and
    river crossings aside); changing it on the Rules screen changes the cap."""
    for override, cap in ((None, 400.0), ({"design_span": {"params": {"plain_m": 350}}}, 350.0)):
        r = run_solve(solve_input({}, overrides=override))["routes"][0]
        inner = r["spans"][1:-1]
        assert max(s["length_m"] for s in inner) <= cap + 1e-6, (cap, max(s["length_m"] for s in inner))


def _existing_400(f, c, ys):
    from shapely.geometry import Point as P
    line = frame_geom(f, LineString([(c, -2000), (c, 2000)]))
    towers = [frame_geom(f, P(c, y)) for y in ys]
    return gpd.GeoDataFrame({"class": ["power_line"] + ["power_tower"] * len(towers),
                             "name": ["Existing 400 kV"] + [None] * len(towers),
                             "attrs": [{"voltage_kv": 400}] * (1 + len(towers))},
                            geometry=[line] + towers, crs=4326)


def test_existing_line_is_not_crossed_within_80m_of_a_tower():
    """Existing 400 kV line across the corridor, towers every 400 m (Vivek, 8 Oct 2026): the wire may
    be crossed anywhere except within 80 m of an existing tower; the mid-span is no longer required."""
    f = frame()
    c = f.length_m / 2 + 37.0
    ex = LineString([(c, -2000), (c, 2000)])
    # (1) a tower 30 m from the bee-line: the crossing keeps 80 m from it
    ys = np.arange(-1800.0, 1801.0, 400.0) + 30.0
    r = run_solve(solve_input({"power": _existing_400(f, c, ys)}))["routes"][0]
    x = LineString(towers_frame(f, r)).intersection(ex)
    assert x.geom_type == "Point"
    d = min(abs(x.y - y) for y in ys)
    assert d >= 80.0 - 1.0, d
    # (2) the nearest tower 110 m from the bee-line: outside the old mid-span window (160-240 m) but
    # more than 80 m from the tower, so the line crosses where it runs, close to the bee-line
    ys = np.arange(-1800.0, 1801.0, 400.0) + 110.0
    r = run_solve(solve_input({"power": _existing_400(f, c, ys)}))["routes"][0]
    x = LineString(towers_frame(f, r)).intersection(ex)
    assert x.geom_type == "Point"
    d = min(abs(x.y - y) for y in ys)
    assert d >= 80.0 - 1.0 and d < 160.0, d


def test_power_line_crossing_window_and_tower_types():
    """Rule_Table 9: lines of 66 kV and above are crossed at 70–110°; TSD B: D towers either side of
    a 400 kV+ line, B/C/D for 132–220 kV; 33 kV lines at any angle with suspension towers allowed."""
    from shapely.geometry import Point as P
    f = frame()
    c = f.length_m / 2
    for kv, layer, rule in ((220, "power", "power_line"), (765, "power", "power_line"), (33, "power_lt", "power_line_lt")):
        # 20° to the bee-line and right across the ±2 km band, so the route cannot go round it
        line = frame_geom(f, LineString([(c - 7000, -2548), (c + 7000, 2548)]))
        pts = [frame_geom(f, P(c - 7000 + t * 14000, -2548 + t * 5096)) for t in np.linspace(0, 1, 38)]
        g = gpd.GeoDataFrame({"class": ["power_line"] + ["power_tower"] * len(pts), "name": [f"{kv} kV"] + [None] * len(pts),
                              "attrs": [{"voltage_kv": kv}] * (1 + len(pts))}, geometry=[line] + pts, crs=4326)
        res = run_solve(solve_input({layer: g}))
        assert res["feasible"], kv
        r = res["routes"][0]
        types = [t["type"] for t in r["towers"]]
        cr = [(i, x) for i, s in enumerate(r["spans"]) for x in s["crossings"] if x["rule"] == rule]
        assert cr, kv
        if kv >= 66:
            assert all(x["angle_deg"] >= 70 - 1e-6 for _, x in cr), (kv, [x["angle_deg"] for _, x in cr])
        else:
            assert any(x["angle_deg"] < 60 for _, x in cr), (kv, [x["angle_deg"] for _, x in cr])
        for i, _ in cr:
            pair = (types[i], types[i + 1])
            if kv >= 400:
                assert all(t in ("D45", "D60") for t in pair), (kv, pair)
            elif kv >= 132:
                assert all(t != "A" for t in pair), (kv, pair)


def test_band_fully_blocked_is_reported():
    """A solar park across the whole band (No-Go, not relaxable): no route, and the blocker is named."""
    f = frame()
    c = f.length_m / 2
    wall = frame_geom(f, box(c - 1500, -6000, c + 1500, 6000))
    res = run_solve(solve_input({"solar": gdf([wall], "solar_park", "Test solar park")}))
    assert not res["feasible"]
    b = res["blocked"]
    assert b["blockers"] and b["blockers"][0]["rule_id"] == "solar"
    assert b["blockers"][0]["feature"] == "Test solar park"


def test_terminal_yard_is_carved_out():
    f = frame()
    yard = frame_geom(f, box(-350, -350, 350, 350))          # A sits in the middle of a 700 m yard
    res = run_solve(solve_input({"substations": gdf([yard], "substation", "Start S/S")}))
    assert res["feasible"]
    r = res["routes"][0]
    first = towers_frame(f, r)[1]
    assert not box(-350, -350, 350, 350).contains(Point(first))


def test_four_routes_are_distinct_objectives():
    """Route_Rules 5: least cost, minimum deviation from the bee-line, relaxed route, second least cost."""
    f = frame()
    village = frame_geom(f, Point(f.length_m * 0.4, 300).buffer(500))
    res = run_solve(solve_input({"clusters": gdf([village], "builtup_cluster")}, routes=4))
    assert res["feasible"] and len(res["routes"]) == 4
    keys = [r["key"] for r in res["routes"]]
    assert keys == ["least_cost", "min_deviation", "relaxed", "second_least_cost"]
    costs = {r["key"]: r["cost"]["total"] for r in res["routes"]}
    assert costs["least_cost"] <= costs["min_deviation"] * 1.02 and costs["least_cost"] <= costs["second_least_cost"] * 1.02
    r1, r4 = (LineString(towers_frame(f, r)) for r in (res["routes"][0], res["routes"][3]))
    assert r4.intersection(r1.buffer(150)).length / r4.length < 0.9      # the second least cost runs elsewhere


@pytest.mark.parametrize("kv,circuit", [(400, "dc"), (220, "dc"), (132, "dc")])
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
    spec = LineSpec(220, "dc")
    rules = resolve_rules(spec)
    costs = resolve_costs(spec)
    comp = compile_rules(f, {"clusters": gdf([blob], "builtup_cluster")}, rules, costs, 220, 17.5, 150, 350)
    tc = tower_type_cost(costs, 220)
    bands = {k: rules["tower_type_bands"].params[k] for k in TOWER_TYPES}
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


def test_airport_no_go_buffer_and_funnel():
    """Rule_Table 14-15: No-Go within 6 km of the airport boundary, 10 km inside the approach funnel;
    an airport beyond the raster still reaches in."""
    f = frame()
    x = f.length_m / 2
    top = f.half_width_m + f.margin_m                                 # raster top edge (frame y)
    rules = resolve_rules(LineSpec(765, "dc"))
    costs = resolve_costs(LineSpec(765, "dc"))
    # boundary 5.5 km beyond the raster top: the 6 km buffer reaches 500 m into the raster
    b1 = top + 5500
    airport = frame_geom(f, box(x - 1500, b1, x + 1500, b1 + 2000))
    comp = compile_rules(f, {"aviation": gdf([airport], "airport")}, rules, costs, 765, 33.5, 150, 370)
    col, row = f.frame_to_pixel(x, top - 200)                        # 5.7 km from the boundary
    assert comp.tower_forbidden[int(row), int(col)] and comp.conductor_forbidden[int(row), int(col)]
    col, row = f.frame_to_pixel(x, top - 1500)                       # 7.0 km
    assert not comp.tower_forbidden[int(row), int(col)]
    assert comp.rules_fired.get("airport_prohibition")
    # boundary 8 km beyond the raster top, runway pointing at the corridor: only the approach funnel
    # (10 km) reaches in
    b2 = top + 8000
    airport = frame_geom(f, box(x - 1500, b2, x + 1500, b2 + 2500))
    rw = frame_geom(f, box(x - 22, b2 + 200, x + 22, b2 + 2300))
    comp = compile_rules(f, {"aviation": gdf([airport], "airport"), "approach": gdf([rw], "runway")},
                         rules, costs, 765, 33.5, 150, 370)
    col, row = f.frame_to_pixel(x, top - 200)                        # on the runway axis, 8.2 km out
    assert comp.conductor_forbidden[int(row), int(col)]
    col, row = f.frame_to_pixel(x + 3000, top - 200)                 # 3 km to the side: outside the funnel
    assert not comp.conductor_forbidden[int(row), int(col)]


def test_nalla_keeps_towers_40m_away_and_is_spanned():
    """Nallas (Rule_Table 81-82, Vivek 8 Oct 2026): no tower within 40 m of the centreline; the
    conductor crosses them."""
    f = frame()
    xs = [f.length_m * k / 8 for k in range(1, 8)]
    lines = [frame_geom(f, LineString([(x, -1500), (x + 40, 1500)])) for x in xs]
    r = run_solve(solve_input({"nalla": gdf(lines, "nalla")}))["routes"][0]
    pts = towers_frame(f, r)
    fl = [LineString([(x, -1500), (x + 40, 1500)]) for x in xs]
    assert min(Point(p).distance(l) for p in pts[1:-1] for l in fl) >= 40.0 - 0.5
    route = LineString(pts)
    assert all(route.intersects(l) for l in fl)

# ---- AESL Tower Spotting Data + Route_Rules.xlsx (8 Oct 2026) ---------------------------------
def test_tsd_span_tables_hold_on_a_solved_route():
    """765 kV: no span over the 370 m normal span; every tower's wind span (mean of its two spans)
    ≤ 400 m; every angle tower's span pair fits its type's adjacent-span allowance; tension towers at
    most 15 spans apart."""
    from beeline_engine.config import load_engine
    from beeline_engine.solver import span_sum_limit
    f = frame()
    village = frame_geom(f, Point(f.length_m * 0.4, 0).buffer(700))
    r = run_solve(solve_input({"clusters": gdf([village], "builtup_cluster")}))["routes"][0]
    tsd = load_engine()["tsd"]["765"]
    T, S = r["towers"], r["spans"]
    assert max(s["length_m"] for s in S[1:-1]) <= 370.0 + 1e-6
    for i in range(2, len(T) - 2):
        L1, L2 = S[i - 1]["length_m"], S[i]["length_m"]
        assert (L1 + L2) / 2 <= 400.0 + 1e-6
        lim = span_sum_limit(tsd["span_sum"], T[i]["type"], T[i]["deflection_deg"])
        assert L1 + L2 <= lim + 1e-6 and max(L1, L2) <= 0.6 * lim + 1e-6, (i, T[i]["type"], T[i]["deflection_deg"], L1, L2)
    idx = [i for i, t in enumerate(T) if t["type"] != "A"]
    assert max(b - a for a, b in zip(idx, idx[1:])) <= 15


def test_tsd_weight_span_and_extension_steps():
    """Weight span: a suspension tower in a dip (uplift below the A minimum of 200 m) becomes a tension
    type whose limit holds. Extensions: the smallest step of the type; A/B/C stop at +6 m, a bigger
    need makes the tower a D with +18 m."""
    from beeline_engine.config import load_engine
    from beeline_engine.spotting import SpotParams, Span, Tower, _raise, _tsd_checks
    tsd = load_engine()["tsd"]["765"]
    sp = SpotParams(design_span_m=370, span_max_m=370, min_span_m=100, max_dev_deg=60, type_bands={},
                    section_km=5, parallel_angle_deg=30, tower_cost_rs={}, ground_clearance_m=18.15,
                    lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150, tsd=tsd)
    zs = [0, 0, 20, 0, 20, 0, 0]                       # the middle tower sits 20 m below both neighbours
    T = [Tower(i, i * 350.0, 0.0, "G" if i in (0, 6) else "A", 0.0, "-", False, ground_z=z) for i, z in enumerate(zs)]
    T[1].reasons.append("dead_end"); T[5].reasons.append("dead_end")
    S = [Span(i, i + 1, 350.0) for i in range(6)]
    _tsd_checks(T, S, sp)
    assert T[3].type in ("B1", "D45") and "weight_span" in T[3].reasons
    t = Tower(9, 0, 0, "A", 0, "-", False)
    _raise(t, 4.0, sp, "extension"); assert (t.type, t.extension_m) == ("A", 4.5)
    t = Tower(9, 0, 0, "B1", 0, "-", False)
    _raise(t, 2.0, sp, "extension"); assert (t.type, t.extension_m) == ("B1", 3.0)
    t = Tower(9, 0, 0, "A", 0, "-", False)
    _raise(t, 10.0, sp, "extension"); assert (t.type, t.extension_m) == ("D45", 18.0)


def test_relaxed_route_crosses_a_last_resort_water_body():
    """A pond across the whole band (Rule_Table 41-42: No-Go with 30 m buffers, crossing as last
    resort): the normal routes find nothing; the relaxed route crosses it and carries 6 months'
    approval."""
    f = frame()
    c = f.length_m / 2
    pond = frame_geom(f, box(c - 40, -6000, c + 40, 6000))
    res = run_solve(solve_input({"pond": gdf([pond], "pond", "Test pond")}, routes=4))
    assert res["feasible"]
    keys = [r["key"] for r in res["routes"]]
    assert keys == ["relaxed"], keys
    r = res["routes"][0]
    assert any(e["type"] == "APPROVAL" and "Pond" in e["text"] for e in r["events"])
    assert r["stats"]["approval_months"] >= 6
    assert any(e["type"] == "ROUTE_NOT_FOUND" for e in res["events_global"])


def test_bridges_are_crossed_with_d_towers_and_flyover_extension():
    """Bridges and flyovers (Rule_Table 25-26): crossable at 80–90° with D towers; a flyover crossing
    gives both crossing towers the D tower's +18 m step."""
    f = frame()
    c = f.length_m / 2
    fly = frame_geom(f, box(c - 10, -6000, c + 10, 6000))
    r = run_solve(solve_input({"bridges": gdf([fly], "flyover", "Test flyover")}))["routes"][0]
    hits = [(i, x) for i, s in enumerate(r["spans"]) for x in s["crossings"] if x["rule"] == "bridges"]
    assert hits
    for i, x in hits:
        for t in (r["towers"][i], r["towers"][i + 1]):
            assert t["type"] in ("D45", "D60") and t["extension_m"] >= 18.0, (t["type"], t["extension_m"])
        assert x["angle_deg"] >= 80 - 1e-6


def test_buildings_keep_every_tower_100m_away():
    """Rule_Table 17-18: the 100 m tower buffer from buildings applies to every tower, suspension
    included."""
    f = frame()
    houses = [frame_geom(f, box(x - 6, -6, x + 6, 6)) for x in (f.length_m * 0.3, f.length_m * 0.6)]
    r = run_solve(solve_input({"buildings": gdf(houses, "building")}))["routes"][0]
    hb = [box(x - 6, -6, x + 6, 6) for x in (f.length_m * 0.3, f.length_m * 0.6)]
    for p in towers_frame(f, r)[1:-1]:
        assert min(Point(p).distance(h) for h in hb) >= 100.0 - 0.5


def test_dead_end_towers_turn_at_most_45_degrees():
    """TSD note J: the dead-end towers next to the gantries turn at most 15° line side + 30° slack
    side. A building block right ahead of A forces the line out sideways; the first tower's turn stays
    within 45°."""
    f = frame()
    block = frame_geom(f, box(250, -150, 900, 150))
    r = run_solve(solve_input({"clusters": gdf([block], "builtup_cluster")}))["routes"][0]
    T = r["towers"]
    assert T[1]["deflection_deg"] <= 45 + 0.5 and T[-2]["deflection_deg"] <= 45 + 0.5, (T[1]["deflection_deg"], T[-2]["deflection_deg"])
