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
