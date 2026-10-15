"""Independent rule checker for a solved 765 kV route (QA tool, 14 Oct 2026).

    source dev.env && python scripts/independent_check.py <run_id> [solve_id] [out_dir]

Reads the solve result + run layers straight from the database, re-derives every geometric rule in
config/rules.yaml from the vector geometry (EPSG:32642) and compares with the engine's own flags.
Nothing here imports beeline_engine; the only shared code is the DB / layer reader. Writes
violations.csv, crossings_compare.csv and run_meta.json to out_dir (default data/logs/independent_check/<run>_<solve>).
Rule parameters (setbacks, windows, side distances) are copied from config/rules.yaml as of 14 Oct 2026 — keep in step.
"""
import sys, json, math, csv, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import unary_union
from shapely import STRtree
from beeline_api import layers as L

import os
from beeline_api import db as _db
RUN = sys.argv[1] if len(sys.argv) > 1 else 'BHU-SHI-0222'
_SOLVE = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
_row = _db.one("select id, result from solves where run_id=%s and (%s::int is null or id=%s) order by id desc limit 1", (RUN, _SOLVE, _SOLVE))
if _row is None:
    sys.exit(f"no solve for {RUN}")
A = sys.argv[3] if len(sys.argv) > 3 else f"data/logs/independent_check/{RUN}_{_row['id']}"
os.makedirs(A, exist_ok=True)
_RESULT = _row['result']
EPSG = 32642
KV = 765
CHW = 33.5                 # corridor half-width
APPROACH_CHW = 27.0        # approach section half-width
APPROACH_LEN = 2000.0      # from each gantry
GO_ZONE_R = 500.0
YARD_R = 150.0             # terminal yard: substation within this of the gantry point
GANTRY_BUF = 10.0
GO_HALF_DIAG = math.sqrt(784.0) / math.sqrt(2.0)   # D60 base 784 m2 -> centre to corner 19.8 m
PARALLEL_ANGLE = 30.0      # engine.yaml: span within this angle of the feature = parallel run
DESIGN_SPAN = 400.0
MIN_SPAN = 100.0
LONG_SPAN = 600.0
RAIL_SPAN = 300.0
WIND_SPAN = 400.0
TYPE_BANDS = [('A', 2.0), ('B1', 15.0), ('C1', 30.0), ('D45', 45.0), ('D60', 60.0)]
RANK = {'A': 0, 'B1': 1, 'C1': 2, 'D45': 3, 'D60': 4, 'G': 9}
TOL = 0.05                 # metres, numeric slack
ANG_TOL = 0.05

# ----------------------------------------------------------------------------------------------- data
res = _RESULT
route = res['routes'][0]
T = route['towers']
S = route['spans']
N = len(T)
pts = [(t['x_utm'], t['y_utm']) for t in T]
gantry = [Point(pts[0]), Point(pts[-1])]
go_zone = unary_union([g.buffer(GO_ZONE_R) for g in gantry])

layers = {}
layer_notes = []
for r in L.run_layers(RUN):
    g = L.read_version(r['layer_version_id'])
    if g is None or len(g) == 0:
        layer_notes.append(f"{r['layer_key']} v{r['layer_version_id']}: empty layer")
        layers[r['layer_key']] = None
        continue
    g = g.to_crs(EPSG)
    bad = g.geometry.isna() | g.geometry.is_empty
    if bad.any():
        layer_notes.append(f"{r['layer_key']}: {int(bad.sum())} empty geometries dropped")
        g = g[~bad]
    layers[r['layer_key']] = g.reset_index(drop=True)
for k in ('defence', 'monuments', 'lulc'):
    if k not in layers:
        layer_notes.append(f"{k}: no layer in this run (rule cannot be checked)")

def attr(row, key, default=None):
    a = row.get('attrs')
    return a.get(key, default) if isinstance(a, dict) else default

def fname(g, i):
    row = g.iloc[i]
    n = row.get('name')
    if n is None or (isinstance(n, float) and math.isnan(n)) or n == '':
        n = row.get('src_id') or f"#{i}"
    return str(n)

class Idx:
    """STRtree over one layer / class subset."""
    def __init__(self, key, classes=None):
        g = layers.get(key)
        self.key, self.classes = key, classes
        if g is None:
            self.g, self.tree = None, None
            return
        if classes:
            g = g[g['class'].isin(classes)].reset_index(drop=True)
        self.g = g
        self.tree = STRtree(list(g.geometry)) if len(g) else None
    def within(self, geom, d):
        """[(dist, i)] of features within d of geom (exact)."""
        if self.tree is None:
            return []
        out = []
        for j in self.tree.query(geom.buffer(d + 1.0)):
            fg = self.g.geometry.iloc[j]
            dist = fg.distance(geom)
            if dist <= d + 1.0:
                out.append((dist, int(j)))
        out.sort()
        return out

IDX = {}
def idx(key, classes=None):
    k = (key, tuple(classes) if classes else None)
    if k not in IDX:
        IDX[k] = Idx(key, classes)
    return IDX[k]

# terminal yards: substations within YARD_R of either gantry point
subs = layers.get('substations')
yard_mask = subs.geometry.apply(lambda g: min(g.distance(gantry[0]), g.distance(gantry[1])) <= YARD_R)
yards = subs[yard_mask]
yard_union = unary_union(list(yards.geometry)) if len(yards) else None
layers['substations_nonterminal'] = subs[~yard_mask].reset_index(drop=True)
layer_notes.append(f"substations: {int(yard_mask.sum())} terminal yard(s) carved out (within {YARD_R:.0f} m of a gantry)")

def chw_at(p):
    return APPROACH_CHW if min(p.distance(gantry[0]), p.distance(gantry[1])) <= APPROACH_LEN else CHW

# ----------------------------------------------------------------------------------------------- output rows
rows = []
def add(kind, no, rule, check, measured, limit, unit='m', severity='violation', layer='', cls='', feature='',
        span=None, note='', in_go=False):
    t = T[no]
    rows.append(dict(kind=kind, tower_no=no, span=f"{span[0]}-{span[1]}" if span else '', chainage_km=t['chainage_km'],
                     tower_type=t['type'], deflection_deg=t['deflection_deg'], rule=rule, check=check,
                     measured=(round(measured, 2) if isinstance(measured, (int, float)) else measured),
                     limit=limit, unit=unit, severity=severity, layer=layer, cls=cls, feature=feature,
                     in_go_zone=in_go, note=note, engine_flag='', status=''))

# ----------------------------------------------------------------------------------------------- 1. tower setbacks
# (rule_id, layer, classes, setback) ; 'CHW' = corridor half-width (27 m in the approach section)
TOWER_SETBACKS = [
    ('airport_prohibition', 'aviation', ['airport'], 6000.0),
    ('buildings', 'buildings', None, 'CHW'),
    ('clusters', 'clusters', None, 200.0),
    ('plotted_land', 'plotted', None, 100.0),
    ('substations', 'substations_nonterminal', None, 100.0),
    ('dams', 'dams', ['dam'], 500.0),
    ('solar', 'solar', None, 150.0),
    ('creek', 'other_water', ['creek'], 500.0),
    ('reservoir', 'reservoir', None, 200.0),
    ('lake', 'lake', None, 'CHW'),
    ('pond', 'pond', None, 30.0),
    ('other_water', 'other_water', ['beach', 'coral_reef', 'lagoon', 'mud_flat', 'salt_marsh', 'mangrove', 'detected_water', 'marine_buffer'], 0.0),
    ('protected_areas', 'national_park', None, 0.0),
    ('protected_areas', 'wildlife_sanctuary', None, 0.0),
    ('protected_areas', 'forest_in_pa', None, 0.0),
    ('protected_areas', 'eco_sensitive_zone', None, 0.0),
    ('crz', 'crz', ['crz_ia', 'crz_ia_mangrove_buffer', 'crz_ib', 'crz_ii', 'crz_iii_ndz', 'crz_iiia', 'crz_iiib', 'crz_iva', 'crz_ivb', 'crz_water'], 'CHW'),
    ('cell_towers', 'telecom', ['cell_tower'], 210.0),
    ('wind_turbines', 'wind_turbines', None, 210.0),
    ('village_roads', 'village_roads', None, 30.0),
    ('railway_platforms', 'rail_platforms', ['platform'], 200.0),
    ('platform_crossing_zone', 'rail_platforms', ['platform_crossing_zone'], 0.0),
    ('railway', 'rail', None, 86.0),
    ('national_highway', 'roads', ['nh'], 92.5),
    ('state_highway', 'roads', ['sh'], 87.0),
    ('bridges', 'bridges', ['road_bridge', 'rail_bridge', 'flyover', 'underpass'], 0.0),
    ('river', 'river', None, 150.0),
    ('canal', 'canal', None, 45.0),          # 30 m from the edge + 15 m half-width
    ('nalla', 'nalla', None, 40.0),
    ('power_line', 'power', ['power_tower'], 72.5),
    ('power_line', 'power', ['power_line'], 80.0),
    ('power_line_lt', 'power_lt', ['power_tower'], 72.5),
    ('power_line_lt', 'power_lt', ['power_line'], 80.0),
    ('pipeline_footing', 'pipelines', None, 30.0),
    ('salt_pan_water', 'other_water', ['salt_pan'], 200.0),
    ('salt_pans', 'salt_pans', None, 200.0),  # landclass, Last resort -> the buffer holds towers too
]
# go zone: only these overlaps are checked (base = D60 square; existing tower = same size)
GO_OVERLAP = [('go_zone_substation', 'substations', None, GO_HALF_DIAG),
              ('go_zone_tower', 'power', ['power_tower'], 2 * GO_HALF_DIAG),
              ('go_zone_tower', 'power_lt', ['power_tower'], 2 * GO_HALF_DIAG),
              ('go_zone_tower', 'telecom', None, 2 * GO_HALF_DIAG),
              ('go_zone_tower', 'wind_turbines', None, 2 * GO_HALF_DIAG)]

# the span each tower belongs to, for the crossing exemptions on tower setbacks of crossing rules:
# a crossing tower of a power line is governed by the 87-123 m window, not by the 80 m setback (it is >= 87 anyway)
for i in range(1, N - 1):
    p = Point(pts[i])
    in_go = go_zone.contains(p)
    if in_go:
        for rid, key, cls, d in GO_OVERLAP:
            for dist, j in idx(key, cls).within(p, d):
                if dist < d - TOL:
                    add('tower', i, rid, 'go_zone_overlap', dist, d, layer=key, cls=cls[0] if cls else '', feature=fname(idx(key, cls).g, j), in_go=True)
        if yard_union is not None and yard_union.contains(p):
            add('tower', i, 'substations', 'tower_in_terminal_yard', 0.0, 0, severity='info', layer='substations', in_go=True,
                note='dead-end tower inside the terminal yard')
        # everything else is exempt in the go zone; record what would have hit as info
    hw = chw_at(p)
    for rid, key, cls, d in TOWER_SETBACKS:
        lim = hw if d == 'CHW' else d
        ix = idx(key, cls)
        if ix.tree is None:
            continue
        hits = ix.within(p, max(lim, 0.0))
        for dist, j in hits:
            if lim == 0.0:
                viol = ix.g.geometry.iloc[j].intersects(p)
                if not viol:
                    continue
                dist = 0.0
            elif dist >= lim - TOL:
                continue
            sev = 'info_go_zone' if in_go else 'violation'
            add('tower', i, rid, 'tower_setback', dist, lim, layer=key, cls=str(ix.g.iloc[j]['class']), feature=fname(ix.g, j),
                in_go=in_go, severity=sev, note=('go zone: exempt' if in_go else ''))
            break   # one row per rule/layer per tower (nearest feature)

# ----------------------------------------------------------------------------------------------- 2. conductor buffers
COND_BUFFERS = [
    ('airport_prohibition', 'aviation', ['airport'], 6000.0),
    ('buildings', 'buildings', None, 'CHW'),
    ('clusters', 'clusters', None, 200.0),
    ('plotted_land', 'plotted', None, 100.0),
    ('substations', 'substations_nonterminal', None, 'CHW'),
    ('substations', 'substations', None, 'CHW'),      # terminal yards too, all spans but the gantry span
    ('dams', 'dams', ['dam'], 500.0),
    ('solar', 'solar', None, 150.0),
    ('creek', 'other_water', ['creek'], 500.0),
    ('reservoir', 'reservoir', None, 200.0),
    ('lake', 'lake', None, 'CHW'),
    ('pond', 'pond', None, 30.0),
    ('other_water', 'other_water', ['beach', 'coral_reef', 'lagoon', 'mud_flat', 'salt_marsh', 'mangrove', 'detected_water', 'marine_buffer'], 0.0),
    ('protected_areas', 'national_park', None, 0.0),
    ('protected_areas', 'wildlife_sanctuary', None, 0.0),
    ('protected_areas', 'forest_in_pa', None, 0.0),
    ('protected_areas', 'eco_sensitive_zone', None, 0.0),
    ('crz', 'crz', ['crz_ia', 'crz_ia_mangrove_buffer', 'crz_ib', 'crz_ii', 'crz_iii_ndz', 'crz_iiia', 'crz_iiib', 'crz_iva', 'crz_ivb', 'crz_water'], 'CHW'),
    ('cell_towers', 'telecom', ['cell_tower'], 210.0),
    ('wind_turbines', 'wind_turbines', None, 210.0),
    ('railway_platforms', 'rail_platforms', ['platform'], 'CHW'),
    ('platform_crossing_zone', 'rail_platforms', ['platform_crossing_zone'], 0.0),
    ('power_line', 'power', ['power_tower'], 80.0),       # our conductor to their tower
    ('power_line_lt', 'power_lt', ['power_tower'], 'CHW'),  # our conductor to their pole
    ('salt_pan_water', 'other_water', ['salt_pan'], 200.0),
    ('salt_pans', 'salt_pans', None, 200.0),
]
# parallel-run buffers of the crossing rules: apply to spans that do NOT cross the feature
PARALLEL = [
    ('railway', 'rail', None, 'CHW'),
    ('national_highway', 'roads', ['nh'], 320.0),
    ('state_highway', 'roads', ['sh'], 40.5),
    ('river', 'river', None, 150.0),
    ('canal', 'canal', None, 45.0),
    ('nalla', 'nalla', None, 40.0),
    ('power_line', 'power', ['power_line'], 'CHW'),
    ('power_line_lt', 'power_lt', ['power_line'], 'CHW'),
    ('pipeline_crossing', 'pipelines', None, 30.0),
]

def heading(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

def acute(h1, h2):
    d = abs((h1 - h2) % 180.0)
    return min(d, 180.0 - d)

def local_dir(geom, pt):
    """Heading of the feature at the point of it nearest to pt (lines: the segment; polygons: the boundary)."""
    if geom.geom_type in ('Polygon', 'MultiPolygon'):
        geom = geom.boundary
    parts = list(getattr(geom, 'geoms', [geom]))
    best, bd = None, 1e18
    for part in parts:
        c = list(part.coords)
        for k in range(len(c) - 1):
            seg = LineString([c[k], c[k + 1]])
            d = seg.distance(pt)
            if d < bd:
                bd, best = d, heading(c[k], c[k + 1])
    return best

span_lines = [LineString([pts[s['a']], pts[s['b']]]) for s in S]

for si, s in enumerate(S):
    a, b = s['a'], s['b']
    line = span_lines[si]
    gantry_span = (si == 0 or si == len(S) - 1)
    hw = chw_at(line.centroid)
    head = heading(pts[a], pts[b])
    # the go zone exempts the part of the span inside it
    line_out = line.difference(go_zone) if line.intersects(go_zone) else line
    if line_out.is_empty:
        continue
    owner = a if a > 0 else 1   # engine attaches conductor flags to span.a (tower 1 for the first span)
    for rid, key, cls, d in COND_BUFFERS:
        if gantry_span and key == 'substations':
            continue              # the gantry span may enter its own terminal yard
        lim = hw if d == 'CHW' else d
        ix = idx(key, cls)
        if ix.tree is None:
            continue
        for dist, j in ix.within(line_out, max(lim, 0.0)):
            fg = ix.g.geometry.iloc[j]
            if lim == 0.0:
                if not fg.intersects(line_out):
                    continue
                dist = 0.0
            elif dist >= lim - TOL:
                continue
            add('span', owner, rid, 'conductor_buffer', dist, lim, layer=key, cls=str(ix.g.iloc[j]['class']), feature=fname(ix.g, j), span=(a, b),
                note=('crosses the feature' if fg.intersects(line_out) else ''))
            break
    for rid, key, cls, d in PARALLEL:
        lim = hw if d == 'CHW' else d
        ix = idx(key, cls)
        if ix.tree is None:
            continue
        for dist, j in ix.within(line_out, lim):
            fg = ix.g.geometry.iloc[j]
            if dist >= lim - TOL:
                continue
            if fg.intersects(line):
                continue      # a crossing span, governed by the crossing checks below
            np_ = shapely.ops.nearest_points(fg, line_out)[0]
            ang = acute(head, local_dir(fg, np_))
            sev = 'violation' if ang < PARALLEL_ANGLE else 'oblique_near_miss'
            add('span', owner, rid, 'parallel_run', dist, lim, layer=key, cls=str(ix.g.iloc[j]['class']), feature=fname(ix.g, j), span=(a, b),
                severity=sev, note=f"angle to feature {ang:.0f} deg (parallel = <{PARALLEL_ANGLE:.0f})")
            break

# ----------------------------------------------------------------------------------------------- 3. spans and crossings
def min_type_for(rule, kv=None):
    if rule in ('railway', 'bridges'):
        return 'D45'
    if rule == 'national_highway':
        return 'B1'
    if rule == 'power_line':
        return 'D45' if (kv or 0) >= 400 else 'B1'
    if rule == 'power_line_lt':
        return 'D45' if (kv or 0) >= 400 else 'A'
    return 'A'

LONG_SPAN_RULES = {'river', 'canal', 'nalla', 'national_highway', 'state_highway', 'bridges', 'pipeline_crossing'}
ANGLE_MIN = {'railway': 80.0, 'national_highway': 80.0, 'state_highway': 70.0, 'bridges': 80.0, 'river': 75.0,
             'pipeline_crossing': 75.0, 'power_line': 70.0, 'power_line_lt': 70.0, 'canal': None, 'nalla': None}
SIDE_MIN = {'railway': 86.0, 'national_highway': 92.5, 'state_highway': 87.0, 'river': 150.0, 'canal': 45.0,
            'nalla': 40.0, 'pipeline_crossing': 30.0}
MAX_SPAN = {'railway': RAIL_SPAN}

def wire_parts(geom):
    return [g for g in getattr(geom, 'geoms', [geom]) if g.geom_type == 'LineString']

# existing-tower positions along each wire (towers within 15 m of the wire), per power layer
WIRE_TOWERS = {}
def wire_tower_positions(key, wi, ln):
    k = (key, wi, id(ln))
    if k in WIRE_TOWERS:
        return WIRE_TOWERS[k]
    ix = idx(key, ['power_tower'])
    pos = sorted(ln.project(ix.g.geometry.iloc[j]) for d, j in ix.within(ln, 15.0) if d <= 15.0)
    WIRE_TOWERS[k] = pos
    return pos

crossings_found = []   # for the comparison with the engine's crossing records
for si, s in enumerate(S):
    a, b = s['a'], s['b']
    line = span_lines[si]
    Lm = line.length
    head = heading(pts[a], pts[b])
    gantry_span = (si == 0 or si == len(S) - 1)
    crossed = []    # (rule, dict)
    # --- line features: rail, roads, canal, nalla, pipelines, power lines
    for rule, key, cls in (('railway', 'rail', None), ('national_highway', 'roads', ['nh']), ('state_highway', 'roads', ['sh']),
                           ('canal', 'canal', None), ('nalla', 'nalla', None), ('pipeline_crossing', 'pipelines', None),
                           ('power_line', 'power', ['power_line']), ('power_line_lt', 'power_lt', ['power_line'])):
        ix = idx(key, cls)
        if ix.tree is None:
            continue
        for j in ix.tree.query(line):
            fg = ix.g.geometry.iloc[j]
            if not fg.intersects(line):
                continue
            for ln in wire_parts(fg):
                x = ln.intersection(line)
                if x.is_empty:
                    continue
                for g in getattr(x, 'geoms', [x]):
                    if g.geom_type != 'Point':
                        continue
                    ang = acute(head, local_dir(ln, g))
                    kv = attr(ix.g.iloc[j], 'voltage_kv')
                    rec = dict(rule=rule, span=(a, b), feature=fname(ix.g, j), kv=kv, angle=ang,
                               side_a=ln.distance(Point(pts[a])), side_b=ln.distance(Point(pts[b])),
                               along_a=line.project(g), in_go=go_zone.contains(g), xpt=g)
                    if rule in ('power_line', 'power_line_lt'):
                        pos = wire_tower_positions(key, j, ln)
                        sx = ln.project(g)
                        lo = max((v for v in pos if v <= sx), default=None)
                        hi = min((v for v in pos if v >= sx), default=None)
                        if lo is not None and hi is not None and hi - lo >= 1.0:
                            rec['null_pct'] = abs(sx - (lo + hi) / 2.0) / (hi - lo) * 100.0
                            rec['their_span'] = hi - lo
                        else:
                            rec['null_pct'] = None
                            near = min((abs(sx - v) for v in pos), default=None)
                            rec['clear_of_tower_m'] = near
                    crossed.append(rec)
    # --- polygon features: river, bridges
    for rule, key, cls in (('river', 'river', None), ('bridges', 'bridges', ['road_bridge', 'rail_bridge', 'flyover', 'underpass'])):
        ix = idx(key, cls)
        if ix.tree is None:
            continue
        for j in ix.tree.query(line):
            fg = ix.g.geometry.iloc[j]
            if not fg.intersects(line):
                continue
            inside = fg.intersection(line)
            width = inside.length
            if rule == 'river':
                # river direction = principal axis of the bank vertices within 150 m of the entry / exit points
                bpts = fg.boundary.intersection(line)
                bl = [g for g in getattr(bpts, 'geoms', [bpts]) if g.geom_type == 'Point']
                ang = float('nan')
                if bl:
                    near = unary_union(bl).buffer(150.0)
                    bnd = fg.boundary.intersection(near)
                    vs = []
                    for part in getattr(bnd, 'geoms', [bnd]):
                        if part.geom_type == 'LineString':
                            vs.extend(part.coords)
                    if len(vs) >= 3:
                        arr = np.array(vs) - np.mean(vs, axis=0)
                        u, sv, vt = np.linalg.svd(arr, full_matrices=False)
                        ang = acute(head, math.degrees(math.atan2(vt[0][1], vt[0][0])))
                    else:
                        ang = float(np.mean([acute(head, local_dir(fg, g)) for g in bl]))
                # second estimate: long axis of the river polygon clipped to a 400 m disc round the crossing
                ang2 = float('nan')
                clip = fg.intersection(fg.intersection(line).centroid.buffer(400.0)) if fg.intersects(line) else None
                if clip is not None and not clip.is_empty and clip.area > 0:
                    big = max(getattr(clip, 'geoms', [clip]), key=lambda g: g.area)
                    c = list(big.minimum_rotated_rectangle.exterior.coords)
                    e = [(LineString([c[k], c[k + 1]]).length, heading(c[k], c[k + 1])) for k in range(4)]
                    ang2 = acute(head, max(e)[1])
                side_a, side_b = fg.distance(Point(pts[a])), fg.distance(Point(pts[b]))
            else:
                mrr = fg.minimum_rotated_rectangle
                c = list(mrr.exterior.coords)
                e = [(LineString([c[k], c[k + 1]]).length, heading(c[k], c[k + 1])) for k in range(4)]
                ang = acute(head, max(e)[1])
                side_a, side_b = fg.distance(Point(pts[a])), fg.distance(Point(pts[b]))
            mid = inside.centroid if not inside.is_empty else line.centroid
            rec = dict(rule=rule, span=(a, b), feature=fname(ix.g, j), kv=None, angle=ang, side_a=side_a, side_b=side_b,
                       along_a=line.project(mid), in_go=go_zone.contains(mid), xpt=mid, width=width,
                       cls=str(ix.g.iloc[j]['class']))
            if rule == 'river':
                rec['angle2'] = ang2
            crossed.append(rec)
    cnt = collections.Counter(r['rule'] for r in crossed)
    for k, rec in enumerate(crossed):
        if cnt[rec['rule']] > 1:
            rec['multi'] = f"one of {cnt[rec['rule']]} {rec['rule']} features crossed by this span"
        crossings_found.append(rec)
    crossed_rules = {r['rule'] for r in crossed}
    # --- span length
    lim = DESIGN_SPAN
    note = ''
    if crossed_rules & LONG_SPAN_RULES:
        lim = LONG_SPAN
        note = 'long-span crossing: ' + ','.join(sorted(crossed_rules & LONG_SPAN_RULES))
    if 'railway' in crossed_rules:
        lim = min(lim, RAIL_SPAN)
        note = 'railway crossing'
    if Lm > lim + TOL:
        add('span', a, 'design_span', 'span_too_long', Lm, lim, span=(a, b), note=note)
    if Lm < MIN_SPAN - TOL:
        add('span', a, 'crossing_span_limits' if not gantry_span else 'substations', 'span_too_short' if not gantry_span else 'gantry_span_short', Lm, MIN_SPAN, span=(a, b))
    # --- crossing checks
    for rec in crossed:
        rule = rec['rule']
        fid = rec['feature']
        kvs = f" {rec['kv']:.0f} kV" if rec.get('kv') else ''
        tag = f"{fid}{kvs}"
        if gantry_span:
            add('span', a, rule, 'crossing_on_gantry_span', Lm, '-', unit='', severity='info', span=(a, b), feature=tag,
                note='engine skips crossing checks on gantry spans')
        amin = ANGLE_MIN.get(rule)
        if rule == 'power_line_lt' and (rec.get('kv') or 0) < 66:
            amin = None
        if rec['in_go'] and not (rule == 'power_line' and (rec.get('kv') or 0) >= 220):
            amin = None   # go zone: only crossings of lines from 220 kV up keep their angle window
        if amin is not None and rec['angle'] < amin - ANG_TOL:
            sev, extra = 'violation', ''
            if rule == 'river':
                a2 = rec.get('angle2')
                extra = f"bank-PCA {rec['angle']:.1f} deg, reach-axis {a2:.1f} deg; " if a2 == a2 else ''
                if a2 == a2 and a2 >= amin - ANG_TOL:
                    sev = 'disputed'
                    extra += 'second estimate passes -> method-sensitive; '
            add('span', a, rule, 'crossing_angle', rec['angle'], amin, unit='deg', span=(a, b), feature=tag, in_go=rec['in_go'], severity=sev,
                note=extra + ('go zone: angle window still applies; ' if rec['in_go'] else '') + rec.get('multi', ''))
        elif rule == 'river' and amin is not None and rec.get('angle2') == rec.get('angle2') and rec['angle2'] < amin - ANG_TOL:
            add('span', a, rule, 'crossing_angle', rec['angle2'], amin, unit='deg', span=(a, b), feature=tag, severity='disputed',
                note=f"reach-axis estimate {rec['angle2']:.1f} deg fails, bank-PCA {rec['angle']:.1f} deg passes -> method-sensitive")
        if rec['in_go']:
            continue    # go zone: nothing else applies
        # crossing-tower distances
        if rule in ('power_line', 'power_line_lt'):
            for side, ti in (('a', a), ('b', b)):
                v = rec['side_' + side]
                if v < 87.0 - TOL:
                    add('span', a, rule, 'crossing_side_min', v, 87, span=(a, b), feature=tag, note=f"tower {ti}")
                elif v > 123.0 + TOL:
                    add('span', a, rule, 'crossing_side_max', v, 123, span=(a, b), feature=tag, severity='soft',
                        note=f"tower {ti}; beyond the maximum is allowed at a search cost (rule side_relax_pct_per_25m)")
            if rec.get('null_pct') is not None:
                if rec['null_pct'] > 20.0 + 0.5:
                    add('span', a, rule, 'crossing_null_point', rec['null_pct'], 20, unit='%', span=(a, b), feature=tag,
                        note=f"their span {rec['their_span']:.0f} m; " + rec.get('multi', ''))
            else:
                cl = rec.get('clear_of_tower_m')
                if cl is None:
                    add('span', a, rule, 'crossing_null_point', 'n/a', 20, unit='%', span=(a, b), feature=tag, severity='info',
                        note='no existing towers mapped on this wire; null point not checkable')
                elif cl < 80.0 - TOL:
                    add('span', a, rule, 'crossing_tower_clear', cl, 80, span=(a, b), feature=tag,
                        note='their span not bracketed by mapped towers; fallback 80 m clear of a tower')
                else:
                    add('span', a, rule, 'crossing_null_point', 'n/a', 20, unit='%', span=(a, b), feature=tag, severity='info',
                        note=f"their span not bracketed by mapped towers; {cl:.0f} m clear of the nearest")
        elif rule in SIDE_MIN:
            for side, ti in (('a', a), ('b', b)):
                v = rec['side_' + side]
                if v < SIDE_MIN[rule] - TOL:
                    add('span', a, rule, 'crossing_side_min', v, SIDE_MIN[rule], span=(a, b), feature=tag, note=f"tower {ti}")
        # span limits by crossing rule
        if rule in MAX_SPAN and Lm > MAX_SPAN[rule] + TOL:
            add('span', a, rule, 'crossing_span', Lm, MAX_SPAN[rule], span=(a, b), feature=tag)
        if rule in ('power_line', 'power_line_lt') and Lm > DESIGN_SPAN + TOL:
            add('span', a, rule, 'crossing_span', Lm, DESIGN_SPAN, span=(a, b), feature=tag, note='power-line crossing span is not a long-span crossing')
        # tower type and deviation at the two crossing towers
        need = min_type_for(rule, rec.get('kv'))
        if rule == 'river' and rec.get('width', 0) >= 600.0:
            need = 'D45'
        for ti in (a, b):
            t = T[ti]
            if t['type'] == 'G':
                continue
            if RANK[t['type']] < RANK[need]:
                add('tower', ti, rule, 'crossing_tower_type', t['type'], need, unit='type', span=(a, b), feature=tag)
            if t['deflection_deg'] > 45.0 + 1e-6 and 0 < ti < N - 1:
                add('tower', ti, rule, 'crossing_deviation', t['deflection_deg'], 45, unit='deg', span=(a, b), feature=tag)
            if rule == 'bridges' and rec.get('cls') == 'flyover' and t['extension_m'] < 18.0:
                add('tower', ti, rule, 'flyover_extension', t['extension_m'], 18, span=(a, b), feature=tag)

# ----------------------------------------------------------------------------------------------- 4. deflection, types, wind span, hilly, sections
SPAN_SUM = {
    'A': [[0, 863], [1, 831], [2, 800]],
    'B1': [[3, 1200], [4, 1183], [5, 1148], [6, 1113], [7, 1078], [8, 1043], [9, 1008], [10, 973], [11, 938], [12, 904], [13, 869], [14, 834], [15, 800]],
    'C1': [[18, 1200], [19, 1175], [20, 1141], [21, 1106], [22, 1072], [23, 1038], [24, 1003], [25, 969], [26, 935], [27, 901], [28, 867], [29, 833], [30, 800]],
    'D45': [[32, 1200], [33, 1195], [34, 1161], [35, 1128], [36, 1095], [37, 1061], [38, 1028], [39, 995], [40, 962], [41, 930], [42, 897], [43, 864], [44, 832], [45, 800]],
    'D60': [[45, 800], [46, 758], [47, 716], [48, 674], [49, 633], [50, 592], [51, 551], [52, 510], [53, 469], [54, 428], [55, 388], [56, 348], [57, 308], [58, 268], [59, 228], [60, 189]],
}
def span_sum_limit(t, d):
    rows_ = SPAN_SUM.get(t)
    if not rows_:
        return None
    if d <= rows_[0][0]:
        return float(rows_[0][1])
    for (a0, v0), (a1, v1) in zip(rows_[:-1], rows_[1:]):
        if d <= a1:
            return float(v1)
    return float(rows_[-1][1])

def band_type(d):
    for t, lim in TYPE_BANDS:
        if d <= lim + 1e-6:
            return t
    return None

for i in range(1, N - 1):
    t = T[i]
    d = t['deflection_deg']
    if d > 60.0 + 1e-6:
        add('tower', i, 'max_deviation', 'max_deviation', d, 60, unit='deg')
    need = band_type(d) or 'D60'
    if RANK[t['type']] < RANK[need]:
        add('tower', i, 'tower_type_bands', 'tower_type_below_deflection_band', t['type'], need, unit='type', note=f"{d:.1f} deg needs {need}")
    if i in (1, N - 2) and d > 15.0 + 1e-6:
        add('tower', i, 'substations', 'dead_end_deviation', d, 15, unit='deg')
    L1, L2 = S[i - 1]['length_m'], S[i]['length_m']
    ws = (L1 + L2) / 2.0
    ext = float(t.get('extension_m') or 0)
    wlim = 250.0 if ext >= 45 else 270.0 if ext >= 33 else 300.0 if ext >= 18 else WIND_SPAN
    if ws > wlim + TOL:
        add('tower', i, 'design_span', 'wind_span', ws, wlim, note=f"spans {L1:.0f} + {L2:.0f}; extension {ext:.0f} m (TSD wind span base 400, +18 m 300, +33 m 270, +45 m 250)")
    # TSD table 3: permissible sum of adjacent spans by type and deviation; one span <= 0.6 x the sum
    lim = span_sum_limit(t['type'], d)
    if lim is not None and t['type'] != 'G':
        if L1 + L2 > lim + TOL:
            add('tower', i, 'design_span', 'span_sum', L1 + L2, lim, note=f"TSD span-sum table, {t['type']} at {d:.1f} deg; spans {L1:.0f} + {L2:.0f}")
        elif max(L1, L2) > 0.6 * lim + TOL:
            add('tower', i, 'design_span', 'span_sum_single', max(L1, L2), round(0.6 * lim, 1), note=f"one span <= 60 % of the TSD span sum {lim:.0f} ({t['type']} at {d:.1f} deg)")
# hilly spans: ground rises > 10 m between adjacent towers -> both towers B1 at least
for s in S[1:-1]:
    a, b = s['a'], s['b']
    za, zb = T[a].get('ground_z'), T[b].get('ground_z')
    if za is None or zb is None:
        continue
    if abs(za - zb) > 10.0:
        for ti in (a, b):
            if T[ti]['type'] == 'A':
                add('tower', ti, 'design_span', 'hilly_span_needs_tension_tower', abs(za - zb), 10, span=(a, b), severity='soft',
                    note='ground_z from the result; engine samples its own DEM profile')
# section points: <= 15 spans / 5 km between tension towers (plain)
run_m, nsp, last = 0.0, 0, 1
for i in range(2, N - 1):
    run_m += S[i - 1]['length_m']; nsp += 1
    if T[i]['type'] != 'A':
        run_m, nsp, last = 0.0, 0, i
        continue
    if run_m > 5000.0 + TOL or nsp > 15:
        add('tower', i, 'section_interval', 'section_interval', run_m if run_m > 5000 else nsp, '5000 m / 15 spans', unit='', severity='soft',
            note=f"{nsp} spans, {run_m:.0f} m since tension tower {last}")
        run_m, nsp, last = 0.0, 0, i

# ----------------------------------------------------------------------------------------------- 5. match with the engine's flags
engine_flags = []   # (tower_no, flag)
for t in T:
    for v in t.get('violations') or []:
        engine_flags.append((t['no'], v, 'violation'))
    for w in t.get('warnings') or []:
        engine_flags.append((t['no'], w, 'warning'))

def flag_matches(row, no, flag):
    rule = row['rule']; chk = row['check']
    if flag == 'tower_buffer':
        return row['kind'] == 'tower' and chk in ('tower_setback', 'go_zone_overlap') and row['tower_no'] == no
    if flag == 'conductor_buffer':
        if row['kind'] == 'span' and chk in ('crossing_null_point', 'crossing_tower_clear') and row['tower_no'] == no and row['severity'] == 'violation':
            row['note'] = (row.get('note') or '') + ' [engine also reports this null-point ban as conductor_buffer]'
            return True
        return row['kind'] == 'span' and chk in ('conductor_buffer', 'parallel_run') and row['tower_no'] == no
    if flag.startswith('crossing_null_point:'):
        return chk in ('crossing_null_point', 'crossing_tower_clear') and rule == flag.split(':')[1] and row['tower_no'] == no
    if flag.startswith('crossing_deviation:'):
        return chk == 'crossing_deviation' and rule == flag.split(':')[1] and row['tower_no'] == no
    if flag.startswith('crossing_angle:'):
        return chk == 'crossing_angle' and rule == flag.split(':')[1] and row['tower_no'] == no
    if flag.startswith('crossing_span:'):
        return chk in ('crossing_span', 'span_too_long') and row['tower_no'] == no
    if flag.startswith('crossing_side:'):
        return chk == 'crossing_side_min' and rule == flag.split(':')[1] and row['tower_no'] == no
    if flag.startswith('max_deviation:'):
        return chk == 'max_deviation' and row['tower_no'] == no
    if flag.startswith('gantry_span:'):
        return chk == 'gantry_span_short' and row['tower_no'] in (no, no - 1)
    if flag.startswith('dead_end_deviation:'):
        return chk == 'dead_end_deviation' and row['tower_no'] == no
    if flag.startswith('wind_span:'):
        return chk == 'wind_span' and row['tower_no'] == no
    if flag.startswith('span_sum:'):
        return chk in ('span_sum', 'span_sum_single') and row['tower_no'] == no
    return False

matched_flags = set()
for row in rows:
    hits = [f for f in engine_flags if flag_matches(row, f[0], f[1])]
    if hits:
        row['engine_flag'] = '; '.join(f"{f[1]}@{f[0]}" for f in hits)
        row['status'] = 'both'
        for f in hits:
            matched_flags.add(f)
    else:
        row['status'] = 'checker_only' if row['severity'] == 'violation' else row['severity']
unmatched = [f for f in engine_flags if f not in matched_flags]
for no, flag, kind in unmatched:
    t = T[no]
    rows.append(dict(kind='engine', tower_no=no, span='', chainage_km=t['chainage_km'], tower_type=t['type'], deflection_deg=t['deflection_deg'],
                     rule=flag.split(':')[1] if ':' in flag and not flag.startswith(('max_deviation', 'wind_span', 'dead_end', 'gantry_span', 'span_sum')) else flag.split(':')[0],
                     check=flag.split(':')[0], measured='', limit='', unit='', severity=f'engine_{kind}', layer='', cls='', feature='',
                     in_go_zone=False, note='engine flag the checker could not reproduce', engine_flag=f"{flag}@{no}", status='engine_only'))

# ----------------------------------------------------------------------------------------------- 6. write
cols = ['kind', 'tower_no', 'span', 'chainage_km', 'tower_type', 'deflection_deg', 'rule', 'check', 'measured', 'limit', 'unit', 'severity',
        'status', 'engine_flag', 'layer', 'cls', 'feature', 'in_go_zone', 'note']
rows.sort(key=lambda r: (r['tower_no'], r['kind'], r['rule'], r['check']))
with open(f'{A}/violations.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, '') for k in cols})

# crossings comparison with the engine's records
eng_x = {}
for s in S:
    for c in s.get('crossings') or []:
        eng_x.setdefault((s['a'], s['b'], c['rule']), []).append(c)
xrows = []
seen = set()
for rec in crossings_found:
    k = (rec['span'][0], rec['span'][1], rec['rule'])
    seen.add(k)
    e = (eng_x.get(k) or [None])[0]
    xrows.append(dict(span=f"{k[0]}-{k[1]}", rule=rec['rule'], feature=rec['feature'], kv=rec.get('kv') or '', in_go_zone=rec['in_go'],
                      chk_angle=round(rec['angle'], 1), chk_angle2=(round(rec['angle2'], 1) if rec.get('angle2') == rec.get('angle2') and rec.get('angle2') is not None else ''), eng_angle=e['angle_deg'] if e else '',
                      chk_side_a=round(rec['side_a'], 1), eng_side_a=e['side_a_m'] if e else '',
                      chk_side_b=round(rec['side_b'], 1), eng_side_b=e['side_b_m'] if e else '',
                      chk_null_pct=(round(rec['null_pct'], 1) if rec.get('null_pct') is not None else ''), eng_null_pct=(e.get('null_offset_pct') if e else ''),
                      chk_width=round(rec.get('width', 0), 1) if 'width' in rec else '', eng_width=(e.get('width_m') if e else ''),
                      eng_min_type=e['min_type'] if e else '', eng_violation=(e.get('violation') if e else ''),
                      status='both' if e else 'checker_only'))
for k, lst in eng_x.items():
    if k not in seen:
        c = lst[0]
        xrows.append(dict(span=f"{k[0]}-{k[1]}", rule=k[2], feature=c.get('name'), kv='', in_go_zone='', chk_angle='', chk_angle2='', eng_angle=c['angle_deg'],
                          chk_side_a='', eng_side_a=c['side_a_m'], chk_side_b='', eng_side_b=c['side_b_m'], chk_null_pct='',
                          eng_null_pct=c.get('null_offset_pct'), chk_width='', eng_width=c.get('width_m'), eng_min_type=c['min_type'],
                          eng_violation=c.get('violation'), status='engine_only'))
xrows.sort(key=lambda r: (int(r['span'].split('-')[0]), r['rule']))
with open(f'{A}/crossings_compare.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(xrows[0].keys()))
    w.writeheader(); w.writerows(xrows)

route_line = LineString(pts)
corr = route_line.buffer(CHW, cap_style=2)
forest_ha = {}
for key in ('reserve_forest', 'forest_in_pa'):
    g = layers.get(key)
    forest_ha[key] = round(sum(x.intersection(corr).area for x in g.geometry if x.intersects(corr)) / 1e4, 2) if g is not None else None
print("corridor (67 m) in forest, ha:", forest_ha, "engine stats forest_ha:", route['stats'].get('forest_ha'))
json.dump(dict(forest_ha=forest_ha, layer_notes=layer_notes, n_rows=len(rows), n_engine_flags=len(engine_flags), n_matched=len(matched_flags),
               n_unmatched=len(unmatched)), open(f'{A}/run_meta.json', 'w'), indent=1)
print("rows", len(rows), "engine flags", len(engine_flags), "matched", len(matched_flags), "unmatched", len(unmatched))
print("layer notes:"); [print("  ", n) for n in layer_notes]
