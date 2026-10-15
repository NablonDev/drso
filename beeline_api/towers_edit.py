"""Manual tower moves on a solved route (Vivek, 10 Oct 2026).

An EditContext holds the solve's compiled rules (the same Compiled, SpotParams, DEM and feature
index the solve used) so that a dragged tower can be checked exactly, live, and a set of moves can
be applied: the moved tower and its two neighbours are re-typed, every span re-checked, violations
flagged (a drop that breaks a rule is allowed), cost and events recomputed, and the result saved as
a new solve version. Nothing is re-spotted or re-searched around the move.
"""
from __future__ import annotations

import io
import math
import threading
import time
from collections import OrderedDict

import mercantile
import numpy as np
import shapely
from PIL import Image
from shapely.geometry import LineString, Point

from beeline_engine import pipeline, spotting as SP
from beeline_engine.config import TOWER_TYPES, tower_type_for
from beeline_engine.solver import fit_type_for_spans

from . import db, layers as L, settings
from .solvejob import LayerInput, SolveInput, build_manifest


class _Stop(Exception):
    pass


class EditContext:
    def __init__(self, sid: int):
        self.sid = sid
        s = db.one("select * from solves where id=%s", (sid,))
        if not s:
            raise ValueError("solve not found")
        self.solve = s
        self.run = db.one("select * from runs where id=%s", (s["run_id"],))
        rls = L.run_layers(self.run["id"])
        used = (s["manifest"] or {}).get("layer_version_ids") or {}
        inputs = {}
        for r in rls:
            lv = used.get(r["layer_key"], r["layer_version_id"])
            g = L.read_version(lv) if r["status"] == "done" else None
            cov = shapely.geometry.shape(r["coverage_geojson"]) if r["coverage_geojson"] else None
            inputs[r["layer_key"]] = LayerInput(r["layer_key"], g, r["mode"], r["source"] or "", r["confidence"], cov, r["model"])
        self.inp = SolveInput(a=(self.run["a_lon"], self.run["a_lat"]), b=(self.run["b_lon"], self.run["b_lat"]),
                              band_km=self.run["band_km"], kv=self.run["kv"], circuit=self.run["circuit"], layers=inputs,
                              a_name=self.run.get("a_name") or "A", b_name=self.run.get("b_name") or "B",
                              routes_wanted=1, rule_overrides=self.run["rule_overrides"] or {},
                              cost_overrides=self.run["cost_overrides"] or {}, dem_dir=settings.DEM_DIR,
                              cache_dir=settings.DATA / "cache" / "rasters")
        cap: dict = {}
        _SP = pipeline.SpotParams
        def sp_cap(*a, **k):
            sp = _SP(*a, **k); cap["sp"] = sp; return sp
        def ge_cap(inp, frame, rules, comp, findex, layer_names, rule_layer, dem, dem_note):
            cap.update(frame=frame, rules=rules, comp=comp, findex=findex, layer_names=layer_names, dem=dem)
            raise _Stop()
        old = (pipeline.SpotParams, pipeline._global_events)
        pipeline.SpotParams, pipeline._global_events = sp_cap, ge_cap
        try:
            pipeline.run_solve(self.inp)
        except _Stop:
            pass
        finally:
            pipeline.SpotParams, pipeline._global_events = old
        self.frame, self.rules, self.comp = cap["frame"], cap["rules"], cap["comp"]
        self.findex, self.layer_names, self.dem, self.sp = cap["findex"], cap["layer_names"], cap["dem"], cap["sp"]
        self.chk = SP.Checker(self.comp, self.sp)
        self.chk.dem = self.dem
        self.ready_at = time.time()

    # ---- geometry helpers --------------------------------------------------------------
    def to_frame(self, lon: float, lat: float) -> tuple[float, float]:
        ux, uy = self.frame.lonlat_to_utm(lon, lat)
        x, y = self.frame.utm_to_frame(np.array([ux]), np.array([uy]))
        return float(x[0]), float(y[0])

    def route(self, route_id: int) -> dict:
        r = next((r for r in (self.solve["result"] or {}).get("routes", []) if r["id"] == route_id), None)
        if r is None:
            raise ValueError("route not in this solve")
        return r

    def pts(self, route: dict) -> list[tuple[float, float]]:
        xs = np.array([t["x_utm"] for t in route["towers"]]); ys = np.array([t["y_utm"] for t in route["towers"]])
        fx, fy = self.frame.utm_to_frame(xs, ys)
        return [(float(a), float(b)) for a, b in zip(fx, fy)]

    # ---- live check ----------------------------------------------------------------------
    def check(self, route_id: int, tower_no: int, lon: float, lat: float, baseline: bool = True) -> dict:
        """Exact checks with tower `tower_no` at lon/lat. Problems already present at the tower's
        original position are marked existing; `ok` means the move adds none."""
        route = self.route(route_id)
        pts = self.pts(route)
        n = len(pts)
        if tower_no <= 0 or tower_no >= n - 1:
            return {"ok": False, "error": "the gantries cannot be moved"}
        out = self._check_at(route, pts, tower_no, self.to_frame(lon, lat))
        if baseline:
            t0 = route["towers"][tower_no]
            base = self._check_at(route, self.pts(route), tower_no, self.to_frame(t0["lon"], t0["lat"]))
            seen = set(base["all_problems"])
            out["existing_problems"] = [x for x in out["all_problems"] if x in seen]
            out["new_problems"] = [x for x in out["all_problems"] if x not in seen]
            out["ok"] = not out["new_problems"]
        return out

    def _check_at(self, route: dict, pts: list, tower_no: int, p: tuple) -> dict:
        n = len(pts)
        pts[tower_no] = p
        chk, sp = self.chk, self.sp
        out: dict = {"tower": tower_no, "ok": True, "tower_rules": [], "spans": [], "turns": [], "types": []}
        # rules the tower itself breaks (with the nearest feature and its distance)
        P = Point(*p)
        if chk.in_go_zone(P):
            hits = [r for r in self.comp.go_overlap if r.hits(P)]
        else:
            hits = [r for r in self.comp.tower_nogo if r.hits(P)]
        if abs(p[1]) > self.frame.half_width_m:
            out["tower_rules"].append({"rule": "band", "name": "Outside the corridor band", "limit_m": self.frame.half_width_m, "distance_m": abs(p[1])})
        for r in hits:
            rid = r.rule_id
            rule = self.rules.get(rid)
            lyr = r.layer
            it, d = self.findex.nearest(P, layers={lyr} if lyr else None, max_dist=max(r.d, 50.0) + 100.0)
            out["tower_rules"].append({"rule": rid, "name": rule.name if rule else rid.replace("_", " "), "layer": lyr,
                                       "feature": it["label"] if it else None, "limit_m": round(r.d, 1),
                                       "distance_m": None if d is None else round(d, 1)})
        # the two spans
        for a, b in ((tower_no - 1, tower_no), (tower_no, tower_no + 1)):
            pa, pb = pts[a], pts[b]
            Lm = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            okx, items = chk.span_crossings(pa, pb)
            cond_ok = chk.conductor_ok(pa, pb)
            cond_hits = [] if cond_ok else sorted({r.rule_id for r in self.comp.conductor_nogo if r.hits(LineString([pa, pb]))})
            probs = []
            if Lm < sp.min_span_m:
                probs.append(f"span {Lm:.0f} m below the {sp.min_span_m:.0f} m minimum")
            long_ok = any(c.get("long_span") for c in items)
            cap = sp.span_max_m if not long_ok else sp.river_max_span_m
            if Lm > cap + 1e-3:
                probs.append(f"span {Lm:.0f} m over the {cap:.0f} m limit")
            for c in items:
                if c.get("violation") == "angle":
                    probs.append(f"{c['name']} crossed at {c['angle_deg']:.0f}°, outside its angle window")
                elif c.get("violation") == "span":
                    probs.append(f"{c['name']} crossing span {Lm:.0f} m over its limit")
            for rid in cond_hits:
                rule = self.rules.get(rid)
                probs.append(f"wire passes through the {rule.name if rule else rid} buffer")
            probs = list(dict.fromkeys(probs))          # the same crossing can appear once per voltage part
            out["spans"].append({"a": a, "b": b, "length_m": round(Lm, 1), "ok": okx and cond_ok and not probs,
                                 "crossings": [{"rule": c["rule"], "name": c["name"], "angle_deg": round(c["angle_deg"], 1)} for c in items],
                                 "problems": probs})
        # turns and types at the three towers
        for i in (tower_no - 1, tower_no, tower_no + 1):
            if i <= 0 or i >= n - 1:
                continue
            d, turn = SP._deflection(pts[i - 1], pts[i], pts[i + 1])
            t = tower_type_for(d, sp.type_bands)
            L1 = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); L2 = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
            probs = []
            if d > sp.max_dev_deg + 1e-6:
                probs.append(f"turn {d:.0f}° over the {sp.max_dev_deg:.0f}° maximum")
            if sp.tsd and t:
                ft = fit_type_for_spans(sp.tsd["span_sum"], t, d, L1, L2, float(sp.tsd.get("single_span_fraction", 0.6)))
                if ft is None:
                    probs.append("adjacent spans exceed the TSD table for this turn")
                else:
                    t = ft
                wind = float((sp.tsd.get("wind_span_m") or {}).get("base", 400))
                if (L1 + L2) / 2 > wind + 1e-3:
                    probs.append(f"wind span {(L1 + L2) / 2:.0f} m over {wind:.0f} m")
            out["turns"].append({"tower": i, "deflection_deg": round(d, 1), "turn": turn, "type": t or "D60", "problems": probs})
        probs = [f"{r['name']}: {r['limit_m']:.0f} m setback" + (f", {r['distance_m']:.0f} m from {r['feature']}" if r.get('distance_m') is not None and r.get('feature') else "") for r in out["tower_rules"]]
        probs += [f"span {s['a']}→{s['b']}: {x}" for s in out["spans"] for x in s["problems"]]
        probs += [f"tower {t['tower']}: {x}" for t in out["turns"] for x in t["problems"]]
        out["all_problems"] = probs
        out["ok"] = not probs
        return out

    # ---- why this angle tower (15 Oct 2026) -------------------------------------------------
    def explain(self, route_id: int, tower_no: int) -> dict:
        """Why the line bends at this tower, tested with the solve's own compiled rules:
        (1) the tower moved onto the straight line through its two neighbours, checked like a hand move;
        (2) the straight leg between the angle points either side: stretches where no tower may stand, wire
            buffers it would pass through, crossings it would make at an illegal angle;
        (3) what the straight leg would save on length and towers (an estimate: land, forest, foundation and
            existing-line pricing are left out)."""
        from collections import Counter
        route = self.route(route_id)
        T = route["towers"]
        n = len(T)
        if tower_no <= 0 or tower_no >= n - 1:
            raise ValueError("a gantry has no bend to explain")
        chk, sp, comp, fr = self.chk, self.sp, self.comp, self.frame
        t = T[tower_no]

        def ll(x, y):
            lon, lat = fr.frame_to_lonlat(x, y)
            return [round(float(lon), 7), round(float(lat), 7)]

        def rname(rid):
            r = self.rules.get(rid)
            return r.name if r else rid.replace("_", " ")

        out: dict = {"tower": tower_no, "type": t["type"], "deflection_deg": t["deflection_deg"],
                     "engine_commit": (self.solve.get("manifest") or {}).get("engine_commit")}

        # (1) this tower on the straight line through its neighbours
        pts0 = self.pts(route)
        a, b, p = pts0[tower_no - 1], pts0[tower_no + 1], pts0[tower_no]
        ax, ay = b[0] - a[0], b[1] - a[1]
        u = max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / ((ax * ax + ay * ay) or 1.0)))
        q = (a[0] + u * ax, a[1] + u * ay)
        base = self._check_at(route, self.pts(route), tower_no, p)
        moved = self._check_at(route, self.pts(route), tower_no, q)
        seen = set(base["all_problems"])
        out["on_line"] = {"lonlat": ll(*q), "shift_m": round(math.hypot(q[0] - p[0], q[1] - p[1]), 1),
                          "new_problems": [x for x in moved["all_problems"] if x not in seen],
                          "existing_problems": [x for x in moved["all_problems"] if x in seen]}

        # (2) the straight leg between the angle points either side
        prev = max(j for j in range(0, tower_no) if j == 0 or T[j].get("angle_point"))
        nxt = min(j for j in range(tower_no + 1, n) if j == n - 1 or T[j].get("angle_point"))
        P, N = pts0[prev], pts0[nxt]
        Ls = math.hypot(N[0] - P[0], N[1] - P[1])
        bend_len = sum(math.hypot(pts0[j + 1][0] - pts0[j][0], pts0[j + 1][1] - pts0[j][1]) for j in range(prev, nxt))
        step = 10.0
        ns = max(2, int(Ls // step) + 1)
        runs, cur = [], None
        for k in range(ns):
            s_ = Ls * k / (ns - 1)
            x, y = P[0] + (N[0] - P[0]) * s_ / Ls, P[1] + (N[1] - P[1]) * s_ / Ls
            if k in (0, ns - 1) or chk.tower_ok(x, y, False):
                if cur is not None:
                    runs.append(cur)
                    cur = None
                continue
            pt = Point(x, y)
            ids = ["band"] if abs(y) > fr.half_width_m else \
                [r.rule_id for r in (comp.go_overlap if chk.in_go_zone(pt) else comp.tower_nogo) if r.hits(pt)]
            if cur is None:
                cur = {"s0": s_, "s1": s_, "rules": Counter(), "layer": {}}
            cur["s1"] = s_
            cur["rules"].update(ids)
            for r in comp.tower_nogo:
                if r.rule_id in ids:
                    cur["layer"].setdefault(r.rule_id, r.layer)
        if cur is not None:
            runs.append(cur)
        blocked_runs = []
        for r_ in runs:
            length = r_["s1"] - r_["s0"] + step
            top = [rid for rid, _ in r_["rules"].most_common(3)]
            sm = (r_["s0"] + r_["s1"]) / 2
            mid = Point(P[0] + (N[0] - P[0]) * sm / Ls, P[1] + (N[1] - P[1]) * sm / Ls)
            feat = None
            if top and top[0] != "band":
                lyr = r_["layer"].get(top[0])
                it, d = self.findex.nearest(mid, layers={lyr} if lyr else None, max_dist=2000.0)
                feat = it["label"] if it else None
            s0, s1 = r_["s0"], r_["s1"] + step
            blocked_runs.append({"length_m": round(length), "from_m": round(s0), "to_m": round(min(s1, Ls)),
                                 "rules": [{"id": rid, "name": "Outside the corridor band" if rid == "band" else rname(rid)} for rid in top],
                                 "feature": feat, "spannable": length <= sp.span_max_m,
                                 "a": ll(P[0] + (N[0] - P[0]) * s0 / Ls, P[1] + (N[1] - P[1]) * s0 / Ls),
                                 "b": ll(P[0] + (N[0] - P[0]) * min(s1, Ls) / Ls, P[1] + (N[1] - P[1]) * min(s1, Ls) / Ls)})
        leg = LineString([P, N])
        wire = []
        if not chk.conductor_ok(P, N):
            for r in comp.conductor_nogo:
                if r.hits(leg):
                    it, d = self.findex.nearest(leg, layers={r.layer} if r.layer else None, max_dist=r.d + 100.0)
                    wire.append({"id": r.rule_id, "name": rname(r.rule_id), "buffer_m": round(r.d, 1), "feature": it["label"] if it else None})
            wire = list({w["id"]: w for w in wire}.values())
        _, items = chk.span_crossings(P, N)
        crossings = [{"rule": c["rule"], "name": c["name"], "angle_deg": c["angle_deg"], "angle_ok": c.get("violation") != "angle"}
                     for c in items]
        long_runs = [r_ for r_ in blocked_runs if not r_["spannable"]]
        bad_angle = [c for c in crossings if not c["angle_ok"]]
        allowed = not long_runs and not wire and not bad_angle

        # (3) estimate of what the straight leg saves
        cost, stats = route.get("cost") or {}, route.get("stats") or {}
        per_m = ((cost.get("conductor") or 0) + (cost.get("land_row") or 0)) / max(1.0, (stats.get("length_km") or 0) * 1000)
        ext = float(t.get("extension_m") or 0)
        type_rs = SP._tower_rs(sp, t["type"], ext) - SP._tower_rs(sp, "A", ext) if t["type"] != "A" else 0.0
        n_bend, n_straight = nxt - prev, max(1, math.ceil(Ls / sp.design_span_m))
        towers_rs = max(0, n_bend - n_straight) * SP._tower_rs(sp, "A", 0.0)
        length_rs = max(0.0, bend_len - Ls) * per_m
        out["straight"] = {
            "from_tower": prev, "to_tower": nxt, "from": ll(*P), "to": ll(*N),
            "length_m": round(Ls, 1), "bend_length_m": round(bend_len, 1), "allowed": allowed,
            "no_tower_stretches": blocked_runs, "wire_buffers": wire, "crossings": crossings,
            "saving_rs": {"length": round(length_rs), "tower_type": round(type_rs), "fewer_towers": round(towers_rs),
                          "total": round(length_rs + type_rs + towers_rs), "towers_bend": n_bend - 1, "towers_straight": n_straight - 1},
        }
        return out

    # ---- no-tower overlay tiles ------------------------------------------------------------
    def nogo_tile(self, z: int, x: int, y: int) -> bytes:
        f = self.frame
        xy = mercantile.xy_bounds(x, y, z)
        n = 256
        xs = np.linspace(xy.left, xy.right, n, endpoint=False) + (xy.right - xy.left) / (2 * n)
        ys = np.linspace(xy.top, xy.bottom, n, endpoint=False) - (xy.top - xy.bottom) / (2 * n)
        from pyproj import Transformer
        tr = Transformer.from_crs(3857, f.epsg, always_xy=True)
        gx, gy = np.meshgrid(xs, ys)
        ux, uy = tr.transform(gx.ravel(), gy.ravel())
        fx, fy = f.utm_to_frame(np.asarray(ux), np.asarray(uy))
        col = ((fx - f.x0) / f.res_m).astype(int); row = ((f.y_top - fy) / f.res_m).astype(int)
        inside = (col >= 0) & (col < f.width) & (row >= 0) & (row < f.height)
        forb = np.zeros(len(fx), bool)
        forb[inside] = self.comp.tower_forbidden[row[inside], col[inside]]
        rgba = np.zeros((n, n, 4), np.uint8)
        m = forb.reshape(n, n)
        rgba[m] = (224, 71, 58, 110)
        buf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, format="PNG"); return buf.getvalue()

    # ---- apply moves -> new solve version ------------------------------------------------
    def apply(self, route_id: int, moves: list[dict], note: str | None = None) -> dict:
        route = self.route(route_id)
        pts = self.pts(route)
        n = len(pts)
        moved = set()
        for mv in moves:
            no = int(mv["tower_no"])
            if no <= 0 or no >= n - 1:
                raise ValueError("the gantries cannot be moved")
            pts[no] = self.to_frame(float(mv["lon"]), float(mv["lat"]))
            moved.add(no)
        affected = sorted({j for i in moved for j in (i - 1, i, i + 1) if 0 < j < n - 1})
        chk, sp, comp = self.chk, self.sp, self.comp
        T0 = route["towers"]
        towers: list[SP.Tower] = []
        for i, (x, y) in enumerate(pts):
            t0 = T0[i]
            if i == 0 or i == n - 1:
                towers.append(SP.Tower(i, x, y, "G", 0.0, "-", False, ["gantry"], ground_z=t0.get("ground_z")))
                continue
            d, turn = SP._deflection(pts[i - 1], pts[i], pts[i + 1])
            if i in affected:
                t = tower_type_for(d, sp.type_bands) or "D60"
                reasons = [r for r in t0.get("reasons", []) if r in ("dead_end", "section", "section_hilly", "hilly")]
                if "dead_end" in reasons:
                    t = "D60"
                reasons.append("moved_by_hand" if i in moved else "retyped_after_move")
                tw = SP.Tower(i, x, y, t, round(d, 2), turn if d > 0 else "-", d > 0, reasons, ground_z=None, extension_m=0.0)
            else:
                tw = SP.Tower(i, x, y, t0["type"], round(d, 2), turn if d > 0 else "-", bool(t0.get("angle_point")) or d > 0,
                              list(t0.get("reasons", [])), ground_z=t0.get("ground_z"), extension_m=float(t0.get("extension_m") or 0),
                              violations=[], warnings=list(t0.get("warnings", [])))
            towers.append(tw)
        # ground heights for the affected towers
        if self.dem is not None and getattr(self.dem, "ok", False):
            idx = [i for i in affected]
            if idx:
                lon, lat = self.frame.frame_to_lonlat(np.array([pts[i][0] for i in idx]), np.array([pts[i][1] for i in idx]))
                for i, z in zip(idx, self.dem.sample(lon, lat)):
                    towers[i].ground_z = None if not np.isfinite(z) else round(float(z), 1)
        spans: list[SP.Span] = []
        for i in range(n - 1):
            a, b = pts[i], pts[i + 1]
            Lm = math.hypot(b[0] - a[0], b[1] - a[1])
            gantry = i == 0 or i == n - 2
            okx, items = (True, []) if gantry else chk.span_crossings(a, b)
            s = SP.Span(i, i + 1, Lm, items, hilly=(not gantry) and chk.is_hilly(a, b))
            if not gantry:
                if not okx:
                    for c in items:
                        if c.get("violation"):
                            towers[i].violations.append(f"crossing_{c['violation']}:{c['rule']}")
                    if Lm > (sp.river_max_span_m if any(c.get("long_span") for c in items) else sp.span_max_m) + 1e-3:
                        towers[i].violations.append(f"span_length:{Lm:.0f}m")
                if Lm < sp.min_span_m - 1e-6:
                    towers[i].violations.append(f"span_length:{Lm:.0f}m")
                if not chk.conductor_ok(a, b):
                    towers[i if i > 0 else 1].violations.append("conductor_buffer")
            spans.append(s)
        for i in range(1, n - 1):
            tw = towers[i]
            if not chk.tower_ok(tw.x, tw.y, angle_tower=tw.deflection_deg > 0):
                tw.violations.append("tower_buffer")
            if tw.deflection_deg > sp.max_dev_deg + 0.5:
                tw.violations.append(f"max_deviation:{tw.deflection_deg:.0f}deg")
            # crossing towers keep their minimum type
            for s in (spans[i - 1], spans[i]):
                for c in s.crossings:
                    mt = c.get("min_type")
                    if mt and mt in TOWER_TYPES and tw.type in TOWER_TYPES and TOWER_TYPES.index(mt) > TOWER_TYPES.index(tw.type):
                        tw.type = mt
                        if f"crossing:{c['rule']}" not in tw.reasons:
                            tw.reasons.append(f"crossing:{c['rule']}")
        if self.dem is not None and getattr(self.dem, "ok", False):
            for i in affected:
                for s in (spans[i - 1], spans[i]):
                    SP._clearance(comp, self.dem, towers, s, sp)
        if sp.tsd:
            SP._tsd_checks(towers, spans, sp)
        cost, stats = SP.route_cost(comp, towers, spans, sp)
        sr = SP.SpottedRoute(towers, spans, cost, stats, "manual")
        pipeline._approvals(sr, comp, self.rules)
        evs = pipeline._route_events(sr, self.frame, comp, self.findex, self.rules, sp, self.layer_names, self.inp, self.dem)
        from beeline_engine.events import event
        evs.insert(0, event("MANUAL_EDIT", {"route": route_id, "n": len(moved), "towers": ", ".join(str(i) for i in sorted(moved)),
                                            "parent": self.sid}))
        var = {"id": route["id"], "key": route["key"], "name": route["name"]}
        class _Out:  # the fields _route_json reads from the search outcome
            cost_rs = float(route.get("grid_cost_rs") or 0.0)
        new_route = pipeline._route_json(var, sr, self.frame, evs, 0.0, 0.0, _Out())
        new_route["edited"] = {"parent_solve": self.sid, "moves": [{"tower_no": int(m["tower_no"]), "lon": float(m["lon"]), "lat": float(m["lat"])} for m in moves]}
        result = dict(self.solve["result"])
        result["routes"] = [new_route if r["id"] == route_id else r for r in result["routes"]]
        rls = L.run_layers(self.run["id"])
        manifest = build_manifest(self.run, rls, result)
        manifest["layer_version_ids"] = (self.solve["manifest"] or {}).get("layer_version_ids") or {r["layer_key"]: r["layer_version_id"] for r in rls}
        manifest["exports_skipped"] = True
        manifest["edited_from"] = self.sid
        manifest["edit_note"] = note
        row = db.one("""insert into solves (run_id, job_id, feasible, result, manifest) values (%s,%s,%s,%s,%s)
                        returning id, created_at""", (self.run["id"], self.solve.get("job_id"), result["feasible"], db.J(result), db.J(manifest)))
        out_dir = settings.RUNS_DIR / self.run["id"] / f"solve_{row['id']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        db.exec_("update solves set out_dir=%s where id=%s", (str(out_dir), row["id"]))
        db.exec_("update runs set latest_solve_id=%s, updated_at=now() where id=%s", (row["id"], self.run["id"]))
        return {"solve_id": row["id"], "route": new_route}


_CTX: "OrderedDict[int, EditContext]" = OrderedDict()
_LOCK = threading.Lock()
_BUILDING: dict[int, threading.Event] = {}


def context(sid: int, wait: bool = True) -> EditContext | None:
    """The solve's edit context, built on first use (tens of seconds) and kept for the last two solves."""
    with _LOCK:
        if sid in _CTX:
            _CTX.move_to_end(sid)
            return _CTX[sid]
        ev = _BUILDING.get(sid)
        if ev is None:
            ev = threading.Event()
            _BUILDING[sid] = ev
            builder = True
        else:
            builder = False
    if builder:
        try:
            ctx = EditContext(sid)
            with _LOCK:
                _CTX[sid] = ctx
                while len(_CTX) > 2:
                    _CTX.popitem(last=False)
        finally:
            with _LOCK:
                _BUILDING.pop(sid, None)
            ev.set()
        return ctx
    if not wait:
        return None
    ev.wait()
    with _LOCK:
        return _CTX.get(sid)


def status(sid: int) -> dict:
    with _LOCK:
        if sid in _CTX:
            return {"ready": True}
        return {"ready": False, "building": sid in _BUILDING}
