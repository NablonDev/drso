"""Engine tests on synthetic corridors (no database, no network). Run: pytest tests/test_engine.py"""
from __future__ import annotations

import heapq
import math

import geopandas as gpd
import numpy as np
import pandas as pd
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
    assert xs[0]["span_m"] <= 600 + 1e-6                 # Vivek, 13 Oct 2026: other crossings up to 600 m (was 300)
    a, b = next((s["a"], s["b"]) for s in r["spans"] if s["crossings"])
    types = {t["no"]: t["type"] for t in r["towers"]}
    assert types[a] in ("D45", "D60") and types[b] in ("D45", "D60")   # crossing tower type D


def test_wide_river_needs_long_span():
    """240 m of water + 150 m bank buffers each side: a single span of 540 m or more, longer than the
    400 m span cap. Crossing spans go to 600 m (Vivek, 13 Oct 2026; was 720) and the TSD wind span is
    ≤ 400 m, so the crossing towers' other spans are short and the towers are tension type."""
    f = frame()
    c = f.length_m / 2
    river = frame_geom(f, box(c - 120, -3000, c + 120, 3000))
    line = frame_geom(f, LineString([(c, -3000), (c, 3000)]))
    res = run_solve(solve_input({"river": gdf([river, line], "river", "Test river")}))
    assert res["feasible"]
    r = res["routes"][0]
    S, T = r["spans"], r["towers"]
    long_ix = [i for i, s in enumerate(S) if s["length_m"] > 400]       # 765 kV span cap 400 m (9 Oct 2026)
    assert long_ix and all(any(cr["rule"] == "river" for cr in S[i]["crossings"]) for i in long_ix)
    for i in long_ix:
        assert S[i]["length_m"] <= 600 + 1e-6
        assert (S[i - 1]["length_m"] + S[i]["length_m"]) / 2 <= 400 + 1e-6
        assert (S[i]["length_m"] + S[i + 1]["length_m"]) / 2 <= 400 + 1e-6
        assert T[i]["type"] != "A" and T[i + 1]["type"] != "A"
    for x, y in towers_frame(f, r)[1:-1]:
        assert not (c - 120 - 150 + 0.5 < x < c + 120 + 150 - 0.5)


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
    # 2 %: since closest-step extensions (9 Oct 2026) the search skirts nearer the hill's fringe and the
    # spotting may add a tower there (measured 1.52 %)
    assert free["cost"]["total"] <= detour["cost"]["total"] * 1.02, (free["cost"]["total"] / 1e7, detour["cost"]["total"] / 1e7)


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
    assert max(s["length_m"] for s in inner) <= 400.0 + 1e-6            # span cap 400 m (9 Oct 2026)
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
    """Existing 400 kV line across the corridor, towers every 400 m: the wire may be crossed anywhere
    except next to an existing tower — 20 % of its span, 80 m here (Vivek, 8 and 10 Oct 2026); the
    mid-span is no longer required."""
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
    # (2) the nearest tower 110 m from the bee-line: the crossing keeps 80 m from it. Since 14 Oct 2026 the
    # 25 m refinement and the vertex moves shift the line to the preferred null point (200 m from the towers)
    # when that costs nothing, as on this empty band — so only the 80 m ban is asserted
    ys = np.arange(-1800.0, 1801.0, 400.0) + 110.0
    r = run_solve(solve_input({"power": _existing_400(f, c, ys)}))["routes"][0]
    x = LineString(towers_frame(f, r)).intersection(ex)
    assert x.geom_type == "Point"
    d = min(abs(x.y - y) for y in ys)
    assert d >= 80.0 - 1.0, d


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
    # A sits in a 700 m yard, 100 m inside its east fence (the gantry span keeps to 250 m, 9 Oct 2026)
    yard = frame_geom(f, box(-600, -350, 100, 350))
    res = run_solve(solve_input({"substations": gdf([yard], "substation", "Start S/S")}))
    assert res["feasible"]
    r = res["routes"][0]
    first = towers_frame(f, r)[1]
    assert not box(-600, -350, 100, 350).contains(Point(first))


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
    E = edge_costs(comp, g, p, ras, 0, g.nc, n_s)[0].cpu().numpy()
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
    assert max(s["length_m"] for s in S[1:-1]) <= 400.0 + 1e-6          # span cap 400 m (9 Oct 2026)
    for i in range(2, len(T) - 2):
        L1, L2 = S[i - 1]["length_m"], S[i]["length_m"]
        assert (L1 + L2) / 2 <= 400.0 + 1e-6
        lim = span_sum_limit(tsd["span_sum"], T[i]["type"], T[i]["deflection_deg"])
        assert L1 + L2 <= lim + 1e-6 and max(L1, L2) <= 0.6 * lim + 1e-6, (i, T[i]["type"], T[i]["deflection_deg"], L1, L2)
    idx = [i for i, t in enumerate(T) if t["type"] != "A"]
    assert max(b - a for a, b in zip(idx, idx[1:])) <= 15


def test_tsd_weight_span_and_extension_steps():
    """Weight span: a suspension tower in a dip (uplift below the A minimum of 200 m) is raised by a body
    extension or made a stronger type, whichever is cheaper while the limit holds. Extensions: the
    step closest to the need (9 Oct 2026): A/B/C reach +6 m, which serves needs up to 12 m; above
    that the tower becomes a D with +18 m."""
    from beeline_engine.config import load_engine
    from beeline_engine.spotting import SpotParams, Span, Tower, _raise, _tsd_checks
    tsd = load_engine()["tsd"]["765"]
    sp = SpotParams(design_span_m=370, span_max_m=370, min_span_m=100, max_dev_deg=60, type_bands={},
                    section_km=5, parallel_angle_deg=30,
                    tower_cost_rs={"A": 4.765e6, "B1": 6.886e6, "C1": 7.803e6, "D45": 9.754e6, "D60": 11.085e6},
                    ground_clearance_m=18.15,
                    lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150, tsd=tsd)
    zs = [0, 0, 20, 0, 20, 0, 0]                       # the middle tower sits 20 m below both neighbours
    T = [Tower(i, i * 350.0, 0.0, "G" if i in (0, 6) else "A", 0.0, "-", False, ground_z=z) for i, z in enumerate(zs)]
    T[1].reasons.append("dead_end"); T[5].reasons.append("dead_end")
    S = [Span(i, i + 1, 350.0) for i in range(6)]
    _tsd_checks(T, S, sp)
    # the dip tower: an A raised +4.5 m (weight span ≥ 200 m) is cheaper than a B1 or D45
    assert (T[3].type, T[3].extension_m) == ("A", 4.5) and "weight_span_extension" in T[3].reasons
    for typ, need, want in (("A", 4.0, ("A", 4.5)), ("B1", 2.0, ("B1", 3.0)), ("A", 1.0, ("A", 0.0)),
                            ("A", 10.0, ("A", 6.0)), ("A", 12.5, ("D45", 18.0)), ("D45", 23.0, ("D45", 18.0)),
                            ("D45", 26.0, ("D45", 33.0))):
        t = Tower(9, 0, 0, typ, 0, "-", False)
        _raise(t, need, sp, "extension")
        assert (t.type, t.extension_m) == want, (typ, need, t.type, t.extension_m)


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
    """Rule_Table 17-18: the tower buffer from single buildings applies to every tower, suspension
    included — the corridor half-width, 33.5 m at 765 kV, since 13 Oct 2026 (was 100 m)."""
    f = frame()
    houses = [frame_geom(f, box(x - 6, -6, x + 6, 6)) for x in (f.length_m * 0.3, f.length_m * 0.6)]
    r = run_solve(solve_input({"buildings": gdf(houses, "building")}))["routes"][0]
    hb = [box(x - 6, -6, x + 6, 6) for x in (f.length_m * 0.3, f.length_m * 0.6)]
    for p in towers_frame(f, r)[1:-1]:
        assert min(Point(p).distance(h) for h in hb) >= 33.5 - 0.5


def test_dead_end_towers_turn_at_most_45_degrees():
    """TSD note J: the dead-end towers next to the gantries turn at most 15° line side + 30° slack
    side. A building block right ahead of A forces the line out sideways; the first tower's turn stays
    within 45°."""
    f = frame()
    block = frame_geom(f, box(250, -150, 900, 150))
    r = run_solve(solve_input({"clusters": gdf([block], "builtup_cluster")}))["routes"][0]
    T = r["towers"]
    assert T[1]["deflection_deg"] <= 45 + 0.5 and T[-2]["deflection_deg"] <= 45 + 0.5, (T[1]["deflection_deg"], T[-2]["deflection_deg"])


def test_approach_section_spans_and_free_go_zone():
    """CEA Schedule VII approach section (Vivek, 8 Oct 2026): within 2 km of each gantry spans keep to
    250 m; the 150 m gantry go zone is free of every rule — a pond right on the gantry point does not
    stop the line."""
    f = frame()
    pond_a = frame_geom(f, Point(0, 0).buffer(120))
    pond_b = frame_geom(f, Point(f.length_m, 0).buffer(120))
    r = run_solve(solve_input({"pond": gdf([pond_a, pond_b], "pond", "Gantry pond")}))
    assert r["feasible"]
    rt = r["routes"][0]
    pts = towers_frame(f, rt)
    for (x0, y0), (x1, y1), s in zip(pts[1:-2], pts[2:-1], rt["spans"][1:-1]):
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        if math.hypot(mx, my) <= 2000 or math.hypot(mx - f.length_m, my) <= 2000:
            assert s["length_m"] <= 400 + 1e-6, s["length_m"]      # 13 Oct 2026: no separate approach span limit
    # the gantry spans keep the approach span too (Vivek, 9 Oct 2026)
    assert 100 - 1e-6 <= rt["spans"][0]["length_m"] <= 400 + 1e-6 and 100 - 1e-6 <= rt["spans"][-1]["length_m"] <= 400 + 1e-6


# ---- speed-ups (9 Oct 2026): vectorised frame transform, raster cache, parallel routes ------------
def test_vectorised_frame_transform_matches_per_geometry():
    f = frame()
    import shapely
    geoms = [Point(f.A[0] + 120.5, f.A[1] - 40.0), LineString([(f.A[0], f.A[1]), (f.A[0] + 900, f.A[1] + 333)]),
             box(f.A[0] + 10, f.A[1] + 10, f.A[0] + 510, f.A[1] + 260), None]
    vec = f.geoms_utm_to_frame(np.array(geoms, dtype=object))
    for g, v in zip(geoms, vec):
        if g is None:
            assert v is None
            continue
        one = f.geom_utm_to_frame(g)
        assert np.allclose(shapely.get_coordinates(one), shapely.get_coordinates(v), atol=1e-6)


def _speed_layers(f):
    c = f.length_m / 2
    rail = frame_geom(f, LineString([(c - 4000 * math.cos(math.radians(30)), -4000 * math.sin(math.radians(30))),
                                     (c + 4000 * math.cos(math.radians(30)), 4000 * math.sin(math.radians(30)))]))
    village = frame_geom(f, Point(f.length_m * 0.3, 300).buffer(500))
    return {"rail": gdf([rail], "railway", "Test line"), "clusters": gdf([village], "builtup_cluster"),
            "power": _existing_400(f, f.length_m * 0.7, [-1000, -600, -200, 200, 600, 1000])}


def test_raster_cache_is_lossless(tmp_path):
    """A compile from the on-disk raster cache is bit-identical to a fresh one; a changed layer
    misses the cache."""
    from beeline_engine.rastercache import RasterCache
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    args = (rules, costs, 765, 33.5, 150, 370)
    fresh = compile_rules(f, _speed_layers(f), *args)
    c1 = RasterCache(tmp_path, f)
    first = compile_rules(f, _speed_layers(f), *args, raster_cache=c1)
    assert c1.misses > 0          # (a hit is possible here: two derived keys with the same geometry share an entry)
    c2 = RasterCache(tmp_path, f)
    second = compile_rules(f, _speed_layers(f), *args, raster_cache=c2)
    assert c2.hits > 0 and c2.misses == 0
    for comp in (first, second):
        for attr in ("tower_forbidden", "angle_forbidden", "conductor_forbidden", "strip_density", "site_cost"):
            assert np.array_equal(getattr(fresh, attr), getattr(comp, attr), equal_nan=True), attr
        for a, b in zip(fresh.crossings, comp.crossings):
            assert a.rule_id == b.rule_id and np.array_equal(a.angle, b.angle, equal_nan=True)
        for a, b in zip(fresh.parallels, comp.parallels):
            assert np.array_equal(a.angle, b.angle, equal_nan=True)
    moved = _speed_layers(f)
    moved["clusters"] = gdf([frame_geom(f, Point(f.length_m * 0.3, -300).buffer(500))], "builtup_cluster")
    c3 = RasterCache(tmp_path, f)
    compile_rules(f, moved, *args, raster_cache=c3)
    assert c3.misses > 0


def test_parallel_routes_match_one_at_a_time(monkeypatch):
    """Routes searched together give exactly the routes of the one-at-a-time loop."""
    from beeline_engine import pipeline
    f = frame()
    orig = pipeline.load_engine
    out = {}
    for w in (1, 3):
        def le(_w=w):
            e = orig()
            e["parallel_routes"] = {"max_workers": _w, "gpu_gb_per_route": 0.01}
            return e
        monkeypatch.setattr(pipeline, "load_engine", le)
        out[w] = run_solve(solve_input(_speed_layers(f), routes=4))
    if torch.cuda.is_available():
        assert out[3]["timings"]["route_workers"] == 3
    assert [r["key"] for r in out[1]["routes"]] == [r["key"] for r in out[3]["routes"]]
    for a, b in zip(out[1]["routes"], out[3]["routes"]):
        assert [(t["x_utm"], t["y_utm"], t["type"]) for t in a["towers"]] == \
               [(t["x_utm"], t["y_utm"], t["type"]) for t in b["towers"]], a["key"]
        assert a["cost"]["total"] == b["cost"]["total"]
        assert [e["type"] for e in a["events"]] == [e["type"] for e in b["events"]]
    assert [e["type"] for e in out[1]["events_global"]] == [e["type"] for e in out[3]["events_global"]]


def test_go_zone_ignores_setbacks_but_not_overlap():
    """Go zone, 500 m round each gantry (Vivek, 9 Oct 2026): no setback / water rule — a tower may
    stand in a pond — but a tower base may not overlap a substation or an existing tower."""
    from beeline_engine.spotting import Checker, SpotParams
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    pond = frame_geom(f, Point(300, 0).buffer(60))
    sub = frame_geom(f, box(250, 150, 330, 230))                       # 260 m from A: not the terminal yard
    pt = frame_geom(f, Point(300, -200))
    tw = gpd.GeoDataFrame({"class": ["power_tower"], "name": [None], "attrs": [{"voltage_kv": 220}]},
                          geometry=[pt], crs=4326)
    layers = {"pond": gdf([pond], "pond"), "substations": gdf([sub], "substation"), "power": tw}
    comp = compile_rules(f, layers, rules, costs, 765, 33.5, 500, 370, terminal_yard_radius_m=150)
    hd = math.sqrt(784) / math.sqrt(2)

    def forbidden(x, y):
        col, row = f.frame_to_pixel(x, y)
        return bool(comp.tower_forbidden[int(row), int(col)])
    assert not forbidden(300, 0)                                       # in the pond, inside the go zone
    assert forbidden(290, 190) and forbidden(300, 150 - hd + 5)        # on / next to the substation
    assert not forbidden(300, 150 - hd - 25)
    assert forbidden(300, -200 + 2 * hd - 5) and not forbidden(300, -200 + 2 * hd + 25)   # existing tower
    # outside the go zone the pond rule holds as before
    pond2 = frame_geom(f, Point(1500, 0).buffer(60))
    comp2 = compile_rules(f, {"pond": gdf([pond2], "pond")}, rules, costs, 765, 33.5, 500, 370)
    col, row = f.frame_to_pixel(1500, 0)
    assert comp2.tower_forbidden[int(row), int(col)]
    # the spotting check agrees
    sp = SpotParams(design_span_m=370, span_max_m=370, min_span_m=100, max_dev_deg=60, type_bands={},
                    section_km=5, parallel_angle_deg=10, tower_cost_rs={}, ground_clearance_m=18,
                    lowest_attachment_m=32, sag_at_design_m=14, go_radius_m=500)
    ck = Checker(comp, sp)
    assert ck.tower_ok(300, 0, False) and not ck.tower_ok(290, 190, False) and not ck.tower_ok(300, -200, False)


def test_go_zone_keeps_crossings_of_220kv_and_above():
    """Go zone (Vivek, 9 Oct 2026): crossings of existing lines from 220 kV up keep their rules
    (angle window, crossing towers, no crossing near an existing tower); lower voltages are waived."""
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)

    def line(x, kv):
        g = frame_geom(f, LineString([(x, -1500), (x, 1500)]))
        t = [frame_geom(f, Point(x, y)) for y in (-200, 200)]
        return gpd.GeoDataFrame({"class": ["power_line", "power_tower", "power_tower"], "name": [f"{kv} kV"] + [None] * 2,
                                 "attrs": [{"voltage_kv": kv}] * 3}, geometry=[g] + t, crs=4326)
    power = gpd.GeoDataFrame(pd.concat([line(300, 400), line(400, 132)], ignore_index=True), crs=4326)
    comp = compile_rules(f, {"power": power}, rules, costs, 765, 33.5, 500, 370, terminal_yard_radius_m=150)

    def angle_at(x, y):
        col, row = f.frame_to_pixel(x, y)
        vals = [cr.angle[int(row), int(col)] for cr in comp.crossings if cr.rule_id == "power_line"]
        return any(np.isfinite(v) for v in vals)
    assert angle_at(300, 0)                 # 400 kV inside the go zone: still a governed crossing
    assert not angle_at(400, 0)             # 132 kV inside the go zone: waived
    # Vivek, 14 Oct 2026: inside the go zone a >= 220 kV crossing keeps only its angle window — the
    # no-crossing stretch next to their towers is waived there
    def cond_near(x, y):
        col, row = f.frame_to_pixel(x, y)
        return bool(comp.conductor_forbidden[int(row) - 1:int(row) + 2, int(col) - 1:int(col) + 2].any())
    assert not cond_near(300, 230) and not cond_near(400, 230)
    go = [cr for cr in comp.crossings if cr.rule_id == "power_line" and cr.name.endswith("(go zone)")]
    assert go and all(cr.side_m is None and cr.max_dev_deg is None and cr.window is not None for cr in go)


# ---- learn from existing lines; protected areas No-Go; proximity pull (9 Oct 2026) ----------------
def _tower_at(f, x, y, kv=220):
    return gpd.GeoDataFrame({"class": ["power_tower"], "name": [None], "attrs": [{"voltage_kv": kv}]},
                            geometry=[frame_geom(f, Point(x, y))], crs=4326)


def test_existing_tower_opens_the_pond_it_stands_in_only_when_blocked():
    f = frame()
    c = f.length_m / 2
    pond = frame_geom(f, box(c, -2600, c + 300, 2600))                 # closes the whole band
    blocked = run_solve(solve_input({"pond": gdf([pond], "pond", "Big tank")}))
    assert not blocked["feasible"]
    res = run_solve(solve_input({"pond": gdf([pond], "pond", "Big tank"), "power": _tower_at(f, c + 150, 1500)}))
    assert res["feasible"]
    r = res["routes"][0]
    assert r["events"][0]["type"] == "PRECEDENT" and "pond" in r["events"][0]["text"]
    assert any("Eased by precedent" in a["item"] for a in r["stats"]["approval_items"])
    # an open band: the precedent is not used even with the tower there
    pond2 = frame_geom(f, box(c, 500, c + 300, 2600))
    r2 = run_solve(solve_input({"pond": gdf([pond2], "pond"), "power": _tower_at(f, c + 150, 1500)}))["routes"][0]
    assert all(e["type"] != "PRECEDENT" for e in r2["events"]) and not r2["stats"].get("precedent_items")


def test_sanctuary_is_no_go_unless_existing_towers_stand_in_it():
    f = frame()
    c = f.length_m / 2
    ws = frame_geom(f, box(c, -2600, c + 800, 2600))
    layers = {"wildlife_sanctuary": gdf([ws], "wildlife_sanctuary", "Test Sanctuary")}
    assert not run_solve(solve_input(layers))["feasible"]
    res = run_solve(solve_input({**layers, "power": _tower_at(f, c + 400, -1500)}))
    assert res["feasible"]
    st = res["routes"][0]["stats"]
    assert st["approval_months"] >= 6 and any("forest diversion" in a["item"] for a in st["approval_items"])


def test_proximity_pulls_the_search_but_not_the_cost(monkeypatch):
    """Existing 400 kV line 1.5 km off the bee-line: with the 20 % search discount within 1 km the
    route runs near it; the reported cost is the true cost (never below the no-discount route's)."""
    f = frame()
    ex = gpd.GeoDataFrame({"class": ["power_line"], "name": ["Existing 400"], "attrs": [{"voltage_kv": 400}]},
                          geometry=[frame_geom(f, LineString([(-1000, 1500), (f.length_m + 1000, 1500)]))], crs=4326)
    on = run_solve(solve_input({"power": ex}))["routes"][0]
    off = run_solve(solve_input({"power": ex}, overrides={"existing_line_proximity": {"enabled": False}}))["routes"][0]
    near = lambda r: sum(1 for x, y in towers_frame(f, r) if abs(y - 1500) <= 1000) / len(r["towers"])
    assert near(on) > 0.5 and near(off) < 0.2, (near(on), near(off))
    assert on["cost"]["total"] >= off["cost"]["total"] * 0.999


def test_uneven_spans_when_equal_spacing_is_blocked():
    """A 700 m straight leg with a village road across its middle (30 m tower setback): two equal
    350 m spans would put the tower on the road; uneven spacing (9 Oct 2026) keeps two spans with the
    tower clear of it, each span within the 400 m cap."""
    from beeline_engine.spotting import Checker, SpotParams, _spot_leg
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    x0 = 3000.0
    road = frame_geom(f, LineString([(x0 + 350, -2000), (x0 + 350, 2000)]))
    comp = compile_rules(f, {"village_roads": gdf([road], "village_road")}, rules, costs, 765, 33.5, 150, 400)
    sp = SpotParams(design_span_m=400, span_max_m=400, min_span_m=100, max_dev_deg=60, type_bands={},
                    section_km=5, parallel_angle_deg=30, tower_cost_rs={}, ground_clearance_m=18.15,
                    lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150)
    chk = Checker(comp, sp)
    pts = _spot_leg(chk, (x0, 0.0), (x0 + 700, 0.0), sp)
    assert pts is not None and len(pts) == 1, pts
    x = pts[0][0] - x0
    assert abs(x - 350) >= 30 - 1e-6 and max(x, 700 - x) <= 400 + 1e-6, x


def test_angle_span_fix_never_leaves_a_span_below_the_minimum():
    """_angle_span_fix inserts a tower near an angle tower whose adjacent spans exceed the TSD table;
    the remainder of the split span must stay ≥ the minimum span (bug fixed 9 Oct 2026: 11 m / 24 m)."""
    from beeline_engine.config import load_engine
    from beeline_engine.spotting import Checker, SpotParams, _angle_span_fix
    f = frame()
    spec = LineSpec(765, "dc")
    comp = compile_rules(f, {}, resolve_rules(spec), resolve_costs(spec), 765, 33.5, 150, 400)
    tsd = load_engine()["tsd"]["765"]
    sp = SpotParams(design_span_m=400, span_max_m=400, min_span_m=100, max_dev_deg=60, type_bands={"A": 2, "B1": 15, "C1": 30, "D45": 45, "D60": 60},
                    section_km=5, parallel_angle_deg=30, tower_cost_rs={}, ground_clearance_m=18.15,
                    lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150, tsd=tsd)
    chk = Checker(comp, sp)
    # a 58° angle tower (D60: span sum ≤ 268 m) between a 115 m span and a 390 m span
    x0 = 3000.0
    p0, p1 = (x0, 0.0), (x0 + 115.0, 0.0)
    th = math.radians(58)
    p2 = (p1[0] + 390 * math.cos(th), p1[1] + 390 * math.sin(th))
    p3 = (p2[0] + 300 * math.cos(th), p2[1] + 300 * math.sin(th))
    pts = _angle_span_fix(chk, [(x0 - 300, 0.0), p0, p1, p2, p3, (p3[0] + 300, p3[1])], sp)
    spans = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts[:-1], pts[1:])]
    assert min(spans[1:-1]) >= 100 - 1e-6, [round(x) for x in spans]


def test_new_tower_keeps_the_tower_setback_from_existing_conductors():
    """Vivek, 13 Oct 2026: a new tower keeps the tower height (765 kV: 72.5 m, no buffer) from an existing
    line's conductors and towers, so a falling tower cannot reach the other line."""
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    c = f.length_m / 2
    ex = _existing_400(f, c, [-1900, 1900])                 # towers far from the bee-line
    comp = compile_rules(f, {"power": ex}, rules, costs, 765, 33.5, 150, 400)

    def forbidden(x, y):
        col, row = f.frame_to_pixel(x, y)
        return bool(comp.tower_forbidden[int(row), int(col)])
    assert forbidden(c + 40, 0) and forbidden(c - 75, 0)       # 40 / 75 m from the wire (80 m since 14 Oct 2026)
    assert not forbidden(c + 100, 0) and not forbidden(c - 115, 0)


def test_crossing_ban_is_a_share_of_the_existing_span():
    """Vivek, 13 Oct 2026: a crossing is allowed within ± 20 % of the span from its null point, i.e. not
    within 30 % of the span of either tower — 120 m on a 400 m span, 30 m on a 66 kV line with poles
    every 100 m (10 Oct: 20 % next to the towers)."""
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    c = f.length_m / 2
    g = frame_geom(f, LineString([(c, -1500), (c, 1500)]))
    poles = [frame_geom(f, Point(c, y)) for y in np.arange(-1500, 1501, 100)]
    lt = gpd.GeoDataFrame({"class": ["power_line"] + ["power_tower"] * len(poles), "name": ["66 kV"] + [None] * len(poles),
                           "attrs": [{"voltage_kv": 66}] * (1 + len(poles))}, geometry=[g] + poles, crs=4326)
    comp = compile_rules(f, {"power_lt": lt}, rules, costs, 765, 33.5, 150, 400)
    bans = [vr for vr in comp.conductor_nogo if vr.rule_id == "power_line_lt" and vr.d == 0]
    assert bans, "no crossing bans compiled"
    hit = lambda y: any(vr.hits(Point(c, y).buffer(1.0)) for vr in bans)     # the real check is span-line vs piece
    assert hit(10) and hit(-15) and hit(25)     # within 30 m of the poles at 0 / 0
    assert not hit(50) and not hit(-50)         # mid-span of a 100 m pole span: crossable
    # the 400 kV helper: towers every 400 m → 80 m
    comp2 = compile_rules(f, {"power": _existing_400(f, c, list(np.arange(-1800, 1801, 400.0)))}, rules, costs, 765, 33.5, 150, 400)
    bans2 = [vr for vr in comp2.conductor_nogo if vr.rule_id == "power_line" and vr.d == 0]
    hit2 = lambda y: any(vr.hits(Point(c, y).buffer(1.0)) for vr in bans2)
    assert hit2(-1800 + 60) and hit2(-1800 + 100) and not hit2(-1800 + 150)


def test_marsh_allows_towers_on_piles():
    """Vivek, 10 Oct 2026: towers may stand on soil-map marshy land; the foundation costs 1.85× (piles)."""
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    marsh = frame_geom(f, box(f.length_m / 2 - 600, -2600, f.length_m / 2 + 600, 2600))
    comp = compile_rules(f, {"marshy_land": gdf([marsh], "marshy_land")}, rules, costs, 765, 33.5, 150, 400)
    col, row = f.frame_to_pixel(f.length_m / 2, 0)
    assert not comp.tower_forbidden[int(row), int(col)] and not comp.conductor_forbidden[int(row), int(col)]
    assert abs(comp.soil_mult[int(row), int(col)] - 1.85) < 1e-6


def test_canal_buffers_start_at_its_edge():
    """Vivek, 10 Oct 2026: canal lines are centrelines of a ~30 m wide canal; the 30 m tower and
    parallel-conductor buffers start 15 m out, i.e. 45 m from the centreline."""
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    c = f.length_m / 2
    canal = frame_geom(f, LineString([(c, -2500), (c, 2500)]))
    comp = compile_rules(f, {"canal": gdf([canal], "canal", "Test canal")}, rules, costs, 765, 33.5, 150, 400)
    tower = [vr for vr in comp.tower_nogo if vr.rule_id == "canal"]
    assert tower and abs(tower[0].d - 45.0) < 1e-6, [vr.d for vr in tower]
    assert any(vr.hits(Point(c + 40, 0)) for vr in tower) and not any(vr.hits(Point(c + 50, 0)) for vr in tower)
    par = [pr for pr in comp.parallels if pr.rule_id == "canal"]
    col, row = f.frame_to_pixel(c + 40, 0)
    assert par and np.isfinite(par[0].angle[int(row), int(col)])         # within 45 m: parallel check applies
    col, row = f.frame_to_pixel(c + 70, 0)
    assert not np.isfinite(par[0].angle[int(row), int(col)])


def test_crossing_span_slack_is_removed_by_spotting():
    """10 Oct 2026: the search may exceed a crossing-span limit by half a site spacing; the spotting
    slides the two crossing towers along the span so the exact limit (350 m at a power line) holds."""
    from beeline_engine.spotting import Checker, SpotParams, _crossing_span_fix
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    c = f.length_m / 2
    comp = compile_rules(f, {"power": _existing_400(f, c, list(np.arange(-1800, 1801, 400.0)) )}, rules, costs, 765, 33.5, 150, 400)
    sp = SpotParams(design_span_m=400, span_max_m=400, min_span_m=100, max_dev_deg=60, type_bands={},
                    section_km=5, parallel_angle_deg=30, tower_cost_rs={}, ground_clearance_m=18.15,
                    lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150)
    chk = Checker(comp, sp)
    y = 0.0                                             # mid-span of the existing line (towers at ±200): crossable
    pts = [(c - 900, y), (c - 540, y), (c - 185, y), (c + 185, y), (c + 525, y), (c + 865, y)]   # 370 m crossing span, neighbours ≤ 400
    out = _crossing_span_fix(chk, pts, sp)
    L = math.hypot(out[3][0] - out[2][0], out[3][1] - out[2][1])
    assert L <= 350 + 1e-6, L                                   # crossing span limit 350 m (10 Oct 2026)
    assert chk.tower_ok(*out[2], True) and chk.tower_ok(*out[3], True)
    assert chk.span_crossings(out[2], out[3])[0]
    assert all(math.hypot(b[0] - a[0], b[1] - a[1]) >= 100 for a, b in zip(out[:-1], out[1:]))


def test_grid_tower_in_a_setback_is_snapped_to_the_cheapest_legal_spot():
    """10 Oct 2026: the search may use a site whose legal cell lies within 25 m; the spotting moves such a
    tower to a legal point within the nudge radius that keeps both spans and all turns legal — since
    14 Oct 2026 the cheapest by exact local cost: next to a village road crossing at 30°, a 14 m sideways
    step would clear its 30 m setback but kinks the line (B1); a 24 m slide along the line keeps an A."""
    from beeline_engine.config import TOWER_TYPES
    from beeline_engine.spotting import Checker, SpotParams, _snap_to_legal, _deflection
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    c = f.length_m / 2
    k = math.cos(math.radians(30)), math.sin(math.radians(30))
    road = frame_geom(f, LineString([(c - 1500 * k[0], -1500 * k[1]), (c + 1500 * k[0], 1500 * k[1])]))
    comp = compile_rules(f, {"village_roads": gdf([road], "village_road")}, rules, costs, 765, 33.5, 150, 400)
    prices = {"A": 2.0e7, "B1": 2.6e7, "C1": 3.0e7, "D45": 3.6e7, "D60": 4.0e7}
    sp = SpotParams(design_span_m=400, span_max_m=400, min_span_m=100, max_dev_deg=60,
                    type_bands={"A": 2, "B1": 15, "C1": 30, "D45": 45, "D60": 60},
                    section_km=5, parallel_angle_deg=30, tower_cost_rs={t: prices.get(t, 4.0e7) for t in TOWER_TYPES},
                    ground_clearance_m=18.15, lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150,
                    nudge_radius_m=30)
    chk = Checker(comp, sp)
    bad = (c - 36.0, 0.0)                                                  # 18 m from the road: inside its 30 m setback
    assert not chk.tower_ok(*bad, True)
    pts = [(c - 1086, 0.0), (c - 736, 0.0), (c - 386, 0.0), bad, (c + 314, 0.0), (c + 664, 0.0), (c + 1014, 0.0)]
    out = _snap_to_legal(chk, pts, sp)
    assert out[3] != bad and chk.tower_ok(*out[3], True)
    assert math.hypot(out[3][0] - bad[0], out[3][1] - bad[1]) <= 35 + 1e-6
    assert _deflection(out[2], out[3], out[4])[0] <= 2.0 + 1e-6        # still a suspension tower: it slid along the line
    assert out[0] == pts[0] and out[6] == pts[6]


def test_forest_cap_prices_forest_up_until_the_route_stays_under_it():
    """Forest rule `max_area_ha` (Vivek, 10 Oct 2026): a 200 m deep reserve-forest strip closes the band
    except for a 300 m gap at its north edge. Uncapped, the least-cost route goes straight through
    (1.3 ha); capped at 1 ha it takes the gap, the event names the multiplier, and the forest cost is
    charged at the true prices."""
    f = frame()
    c = f.length_m / 2
    rf = frame_geom(f, box(c, -2600, c + 200, 1700))
    layers = {"reserve_forest": gdf([rf], "reserve_forest", "Test RF")}
    free = run_solve(solve_input(layers, overrides={"forest": {"params": {"max_area_ha": 0}}}))["routes"][0]
    assert free["stats"]["forest_ha"] > 1.0, free["stats"]["forest_ha"]
    assert all(e["type"] != "FOREST_CAP" for e in free["events"])
    res = run_solve(solve_input(layers, overrides={"forest": {"params": {"max_area_ha": 1}}}))
    assert res["feasible"]
    r = res["routes"][0]
    assert r["stats"]["forest_ha"] <= 1.0, r["stats"]["forest_ha"]
    ev = [e for e in r["events"] if e["type"] == "FOREST_CAP"]
    assert ev and "capped at 1 ha" in ev[0]["text"] and "×" in ev[0]["text"], [e["text"] for e in r["events"]][:3]
    # true prices: forest ₹ per ha charged is the ordinary rate, not the search multiplier
    per_ha = r["cost"]["forest"] / max(r["stats"]["forest_ha"], 1e-6) if r["stats"]["forest_ha"] else 0.0
    free_per_ha = free["cost"]["forest"] / free["stats"]["forest_ha"]
    assert per_ha <= free_per_ha * 1.05
    assert r["cost"]["total"] > free["cost"]["total"]


def test_forest_priced_copy_matches_a_recompile():
    """The cap search's forest-priced compile is a shallow copy with the forest cells scaled (11 Oct 2026);
    it must equal a full compile with the forest cost items multiplied."""
    from beeline_engine.pipeline import _forest_priced
    f = frame()
    c = f.length_m / 2
    rf = frame_geom(f, box(c, -800, c + 200, 800))
    layers = {"reserve_forest": gdf([rf], "reserve_forest", "RF")}
    gdfs = {k: v for k, v in layers.items()}
    spec = LineSpec(765, "dc")
    rules = resolve_rules(spec)
    base = compile_rules(f, gdfs, rules, resolve_costs(spec), 765, 33.5)
    scaled = {"forest": {k: v * 10 for k, v in {"npv_l_ha": 8.61, "ca_l_ha": 1.1, "plans_l_ha": 1.6, "rf_land_cr_ha": 0.48}.items()}}
    full = compile_rules(f, gdfs, rules, resolve_costs(spec, scaled), 765, 33.5)
    cheap = _forest_priced(base, 10.0)
    fm = base.cost_class == 4
    assert fm.any()
    assert np.allclose(cheap.strip_density, full.strip_density, rtol=1e-4)
    assert np.allclose(cheap.land_rate_m2, full.land_rate_m2, rtol=1e-4)
    assert np.array_equal(cheap.tower_forbidden, base.tower_forbidden)
    assert cheap.costs["forest"]["npv_l_ha"] == full.costs["forest"]["npv_l_ha"]


def _line_with_towers(f, x, span=400.0, kv=400):
    """An existing line across the whole band at chainage x with mapped towers every `span` m."""
    line = frame_geom(f, LineString([(x, -2600), (x, 2600)]))
    tw = [frame_geom(f, Point(x, y)) for y in np.arange(-2400, 2401, span)]
    n = 1 + len(tw)
    return gpd.GeoDataFrame({"class": ["power_line"] + ["power_tower"] * len(tw), "name": [f"{kv} kV line"] * n,
                             "confidence": [0.9] * n, "attrs": [{"voltage_kv": kv}] * n}, geometry=[line] + tw, crs=4326)


def test_crossing_towers_sit_in_the_side_window():
    """Vivek, 13 Oct 2026: the two crossing towers stand 1.2-1.7 × our tower height from the crossed wire
    (87-123 m at 765 kV) and turn at most 45°; the crossing is inside the allowed band of the crossed span."""
    f = frame()
    c = f.length_m / 2
    res = run_solve(solve_input({"power": _line_with_towers(f, c)}))
    assert res["feasible"]
    r = res["routes"][0]
    xs = [(s, x) for s in r["spans"] for x in s["crossings"] if x["rule"] == "power_line"]
    assert len(xs) == 1
    s, x = xs[0]
    assert x["side_window_m"] == [87, 123]
    assert 87 - 0.1 <= x["side_a_m"] <= 123 + 0.1 and 87 - 0.1 <= x["side_b_m"] <= 123 + 0.1, x
    assert x.get("violation") is None
    assert 174 <= s["length_m"] <= 246 + 40       # sides are measured to the wire's raster band (± 1.5 cells)
    for i in (s["a"], s["b"]):
        assert r["towers"][i]["deflection_deg"] <= 45 + 1e-6
    assert r["stats"]["violations"] == 0


def test_preferred_crossing_is_taken_when_it_is_free():
    """With the bee-line through a null point, the search crosses there at 90° and nothing is relaxed."""
    f = frame()
    c = f.length_m / 2
    # towers at ±200, ±600 … so the bee-line (y = 0) passes the null point of a span
    line = frame_geom(f, LineString([(c, -2600), (c, 2600)]))
    tw = [frame_geom(f, Point(c, y)) for y in np.arange(-2200, 2201, 400)]
    n = 1 + len(tw)
    g = gpd.GeoDataFrame({"class": ["power_line"] + ["power_tower"] * len(tw), "name": ["400 kV line"] * n,
                          "confidence": [0.9] * n, "attrs": [{"voltage_kv": 400}] * n}, geometry=[line] + tw, crs=4326)
    res = run_solve(solve_input({"power": g}))
    r = res["routes"][0]
    x = [x for s in r["spans"] for x in s["crossings"] if x["rule"] == "power_line"][0]
    assert x["angle_preferred"] and x["point_preferred"], x
    assert r["stats"]["crossings_relaxed"] == 0
    assert all(e["type"] != "CROSSING_RELAXED" for e in r["events"])


def test_relaxed_crossing_is_reported():
    """A crossing away from the null point is allowed at the penalty and named in the events."""
    f = frame()
    c = f.length_m / 2
    res = run_solve(solve_input({"power": _line_with_towers(f, c)}))
    r = res["routes"][0]
    x = [x for s in r["spans"] for x in s["crossings"] if x["rule"] == "power_line"][0]
    if not x["point_preferred"] or not x["angle_preferred"]:
        assert r["stats"]["crossings_relaxed"] >= 1
        ev = [e for e in r["events"] if e["type"] == "CROSSING_RELAXED"]
        assert ev and "allowed because" in ev[0]["text"]


def test_other_crossings_may_span_600m_and_turn_at_most_45():
    """Other crossings (rail, roads, rivers …): no null point, span up to 600 m, 45° at the crossing towers."""
    f = frame()
    c = f.length_m / 2
    rail = frame_geom(f, LineString([(c, -2600), (c, 2600)]))
    res = run_solve(solve_input({"rail": gdf([rail], "railway", "Test line")}))
    assert res["feasible"]
    r = res["routes"][0]
    xs = [(s, x) for s in r["spans"] for x in s["crossings"] if x["rule"] == "railway"]
    assert xs
    s, x = xs[0]
    assert s["length_m"] <= 600 + 1e-3 and x.get("violation") is None
    assert "side_window_m" not in x and x.get("point_preferred") is None
    for i in (s["a"], s["b"]):
        assert r["towers"][i]["deflection_deg"] <= 45 + 1e-6


# ---- angle-point moves (Vivek, 14 Oct 2026: reduce angle towers under the rules) --------------------------

def _empty_band_checker():
    """A compile with one far-away building (a non-empty layer set) and spot params with the 765 kV prices."""
    from beeline_engine.config import TOWER_TYPES
    from beeline_engine.spotting import Checker, SpotParams
    f = frame()
    spec = LineSpec(765, "dc")
    rules, costs = resolve_rules(spec), resolve_costs(spec)
    far = frame_geom(f, box(100, 2000, 120, 2020))
    comp = compile_rules(f, {"buildings": gdf([far], "building")}, rules, costs, 765, 33.5, 150, 400)
    prices = {"A": 0.4765e7, "B1": 0.6886e7, "C1": 0.7803e7, "D45": 0.9754e7, "D60": 1.1085e7}
    sp = SpotParams(design_span_m=400, span_max_m=400, min_span_m=100, max_dev_deg=60,
                    type_bands={"A": 2, "B1": 15, "C1": 30, "D45": 45, "D60": 60},
                    section_km=5, parallel_angle_deg=30, tower_cost_rs={t: prices.get(t, 1.1085e7) for t in TOWER_TYPES},
                    ground_clearance_m=18.15, lowest_attachment_m=32.6, sag_at_design_m=14.278, go_radius_m=150)
    return f, Checker(comp, sp), sp


def _polyline_towers(chk, sp, vertices, heading_out_deg):
    """Gantry + dead-end stand-ins, the vertices with spotted legs between them, and a straight tail."""
    from beeline_engine.spotting import _spot_leg
    a0 = vertices[0]
    pts = [(a0[0] - 400, a0[1]), (a0[0] - 200, a0[1])]
    for a, b in zip(vertices[:-1], vertices[1:]):
        pts += [a] + list(_spot_leg(chk, a, b, sp))
    end = vertices[-1]
    th = math.radians(heading_out_deg)
    pts += [end, (end[0] + 200 * math.cos(th), end[1] + 200 * math.sin(th)), (end[0] + 400 * math.cos(th), end[1] + 400 * math.sin(th))]
    return pts


def _vertices(P):
    from beeline_engine.spotting import _deflection
    return [(k, round(_deflection(P[k - 1], P[k], P[k + 1])[0], 1)) for k in range(1, len(P) - 1) if _deflection(P[k - 1], P[k], P[k + 1])[0] > 2]


def test_merge_bends_joins_two_small_bends_into_one():
    """Idea 2: two 8° same-direction bends 300 m apart (B1 + B1) become one bend of ≤ 16° at the tangent
    intersection (one tower fewer, ₹0.6+ Cr saved) on an empty band."""
    from beeline_engine.spotting import _merge_bends
    f, chk, sp = _empty_band_checker()
    c = f.length_m / 2
    th = math.radians(8)
    v1 = (c - 1000, -300.0)
    v2 = (v1[0] + 300 * math.cos(th), v1[1] + 300 * math.sin(th))
    th2 = math.radians(16)
    end = (v2[0] + 1850 * math.cos(th2), v2[1] + 1850 * math.sin(th2))
    pts = _polyline_towers(chk, sp, [(c - 2850, -300.0), v1, v2, end], 16)
    assert len(_vertices(pts)) == 2
    out, saved, moves = _merge_bends(chk, pts, sp)
    vt = _vertices(out)
    assert moves == 1 and len(vt) == 1 and 12 <= vt[0][1] <= 17 and saved > 0.5e7, (vt, saved)


def test_polish_trims_a_bend_to_the_next_type_band():
    """Idea 3: a 17° bend (C1) is moved inward along its bisector until it is ≤ 15° (B1), the neighbours
    staying suspension towers."""
    from beeline_engine.spotting import _cost_polish
    f, chk, sp = _empty_band_checker()
    c = f.length_m / 2
    v = (c - 150, -300.0)
    th = math.radians(17)
    end = (v[0] + 1850 * math.cos(th), v[1] + 1850 * math.sin(th))
    pts = _polyline_towers(chk, sp, [(c - 2000, -300.0), v, end], 17)
    out, saved, moves = _cost_polish(chk, pts, sp)
    vt = _vertices(out)
    assert moves >= 1 and len(vt) == 1 and vt[0][1] <= 15.0, vt


def test_vertex_descent_removes_a_needless_bend():
    """Idea 4: a vertex pushed 60 m off a straight line (two small bends) is removed by the descent."""
    from beeline_engine.spotting import _vertex_descent
    f, chk, sp = _empty_band_checker()
    c = f.length_m / 2
    pts = _polyline_towers(chk, sp, [(c - 2000, -300.0), (c - 150, -240.0), (c + 1700, -300.0)], 0)
    assert len(_vertices(pts)) >= 1
    out, saved, moves = _vertex_descent(chk, pts, sp)
    assert moves >= 1 and len(_vertices(out)) == 0, _vertices(out)


def test_refinement_keeps_a_feasible_route_no_dearer():
    """Targeted 25 m lattice: the refined search round the 50 m solution is feasible and no dearer at search
    prices; the route is still a legal crossing of the existing line."""
    from beeline_engine import solver as S_, pipeline as P_
    calls = []
    orig = S_.solve

    def spy(compiled, p, cancel_check=None, progress=None, corridor=None):
        out = orig(compiled, p, cancel_check=cancel_check, progress=progress, corridor=corridor)
        calls.append((p.site_spacing_m, corridor is not None, out.feasible, out.cost_rs))
        return out
    S_.solve = spy
    P_.solve = spy                                  # the pipeline imported the name
    try:
        f = frame()
        c = f.length_m / 2
        res = run_solve(solve_input({"power": _line_with_towers(f, c)}))
    finally:
        S_.solve = orig
        P_.solve = orig
    assert res["feasible"]
    fine = [x for x in calls if x[1]]
    coarse = [x for x in calls if not x[1]]
    assert fine and coarse and fine[0][0] == 25.0
    assert fine[0][2] and fine[0][3] <= coarse[0][3] + 1e-6
