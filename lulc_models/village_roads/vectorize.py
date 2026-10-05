# Copied from roaddet/vectorize.py (2026-09-15). The research __main__ harness (site chips with
# relative paths) is removed; the functions are unchanged. MAX_GAP_M is overridden by blockvec.init().
"""Mask -> skeleton -> graph -> gap-bridged centrelines.

Port of the Microsoft geomgen PostProcessor order, with their ConnectionFinder
cost function (step distance + colour weight + angle weight) reimplemented in
Python over skimage's MCP_Geometric instead of their C# Dijkstra.
"""
import json, math, sys, collections
import numpy as np, rasterio, networkx as nx, sknw
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import (binary_closing, disk, remove_small_holes,
                                remove_small_objects, skeletonize)
from skimage.graph import MCP_Geometric
import scipy.ndimage as ndi

# --- tuned on our measurements, MS constants noted where they differ
LO, HI          = 0.15, 0.50   # hysteresis (measured best pair)
HOLE_PX         = 200
CLOSE_R         = 1            # MS uses 3x3 kernel == r1. NEVER open: r>=3 erases 4px tracks
MIN_OBJ_PX      = 250
SPUR_M          = 12.0         # prune dangles shorter than this
MAX_GAP_M       = 80.0         # MS FlyDistanceLimit 350px @0.3m ~= 105 m
ANGLE_GATE_DEG  = 60.0         # MS: 90 deg road-end, 60 deg junction
DETOUR_MAX      = 1.6          # path length / straight-line
MIN_MEAN_P      = 0.12         # mean probability along the bridge
JUNCT_MERGE_PX  = 4            # MS JunctionsMergeDistance


def build(prob, px_m):
    mask = apply_hysteresis_threshold(prob, LO, HI)
    mask = remove_small_holes(mask, connectivity=1, area_threshold=HOLE_PX)
    if CLOSE_R: mask = binary_closing(mask, footprint=disk(CLOSE_R))
    mask = remove_small_objects(mask, connectivity=1, min_size=MIN_OBJ_PX)
    skel = skeletonize(mask, method="zhang")
    G = sknw.build_sknw(skel, multi=False, iso=False, ring=True)
    return mask, skel, G


def edge_widths(G, mask, px_m):
    """Per-edge surface width in metres.

    The distance transform at a skeleton pixel is the radius of the largest
    inscribed disc, so 2*dt is the local mask breadth. Median along the edge
    resists junction blobs and spurs. Same quantity Microsoft fits by annealing.
    """
    dt = ndi.distance_transform_edt(mask)
    for u, v, d in G.edges(data=True):
        pts = d.get("pts")
        if pts is None or len(pts) == 0:
            d["width_m"] = None; continue
        r = np.clip(pts[:, 0].astype(int), 0, dt.shape[0] - 1)
        c = np.clip(pts[:, 1].astype(int), 0, dt.shape[1] - 1)
        vals = dt[r, c]
        vals = vals[vals > 0]
        if vals.size == 0:
            d["width_m"] = None; continue
        d["width_m"]  = float(2.0 * np.median(vals) * px_m)
        d["width_p90"] = float(2.0 * np.percentile(vals, 90) * px_m)
    return G


def prune(G, px_m, rounds=3):
    for _ in range(rounds):
        drop = [n for n in list(G.nodes())
                if G.degree(n) == 1 and
                any(G[n][m].get("weight", 0) * px_m < SPUR_M for m in G[n])]
        if not drop: break
        G.remove_nodes_from(drop)
    G.remove_nodes_from([n for n in list(G.nodes()) if G.degree(n) == 0])
    return G


def heading(G, n, back=10):
    """Outgoing direction at a degree-1 node, from the last `back` skeleton px."""
    for m in G[n]:
        pts = G[n][m].get("pts")
        if pts is None or len(pts) < 2: continue
        o = np.array(G.nodes[n]["o"])
        if np.linalg.norm(pts[0] - o) > np.linalg.norm(pts[-1] - o):
            pts = pts[::-1]
        k = min(back, len(pts) - 1)
        v = pts[0] - pts[k]
        nv = np.linalg.norm(v)
        if nv > 0: return v / nv
    return None


def bridge(G, prob, px_m):
    """MS ConnectionFinder: least-cost path between road-ends over 1/(p+eps),
    gated on heading, detour ratio and mean probability along the path."""
    ends = [n for n in G.nodes() if G.degree(n) == 1]
    if len(ends) < 2: return G, 0
    cost = 1.0 / (prob.astype(np.float64) + 0.05)
    mcp = MCP_Geometric(cost, fully_connected=True)
    coords = {n: np.array(G.nodes[n]["o"], dtype=float) for n in ends}
    head = {n: heading(G, n) for n in ends}
    maxpx = MAX_GAP_M / px_m
    added = 0
    used = set()
    order = sorted(ends)
    for i, a in enumerate(order):
        if a in used: continue
        pa, ha = coords[a], head[a]
        if ha is None: continue
        cands = []
        for b in order[i+1:]:
            if b in used or b == a: continue
            d = np.linalg.norm(coords[b] - pa)
            if d < 1e-6 or d > maxpx: continue
            dirv = (coords[b] - pa) / d
            if math.degrees(math.acos(np.clip(np.dot(ha, dirv), -1, 1))) > ANGLE_GATE_DEG: continue
            hb = head[b]
            if hb is not None and math.degrees(math.acos(np.clip(np.dot(hb, -dirv), -1, 1))) > ANGLE_GATE_DEG:
                continue
            cands.append((d, b))
        if not cands: continue
        cands.sort()
        for d, b in cands[:4]:
            try:
                mcp.find_costs([tuple(np.round(pa).astype(int))],
                               [tuple(np.round(coords[b]).astype(int))])
                path = np.array(mcp.traceback(tuple(np.round(coords[b]).astype(int))))
            except Exception:
                continue
            if len(path) < 2: continue
            plen = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
            if plen / max(d, 1e-6) > DETOUR_MAX: continue
            mp = prob[path[:, 0], path[:, 1]].mean()
            if mp < MIN_MEAN_P: continue
            G.add_edge(a, b, pts=path, weight=float(plen), bridged=True)
            used.add(a); used.add(b); added += 1
            break
    return G, added


def collapse(G):
    nodes = list(G.nodes())
    if not nodes: return G
    from scipy.spatial import cKDTree
    P = np.array([G.nodes[n]["o"] for n in nodes], dtype=float)
    pairs = cKDTree(P).query_pairs(JUNCT_MERGE_PX)
    for i, j in sorted(pairs, reverse=True):
        a, b = nodes[i], nodes[j]
        if a in G and b in G and a != b:
            try: nx.contracted_nodes(G, a, b, self_loops=False, copy=False)
            except Exception: pass
    return G


def to_lines(G, transform):
    out = []
    for u, v, d in G.edges(data=True):
        pts = d.get("pts")
        if pts is None or len(pts) < 2: continue
        xy = [rasterio.transform.xy(transform, int(r), int(c)) for r, c in pts]
        out.append((xy, bool(d.get("bridged", False)), d.get("width_m"), d.get("width_p90")))
    return out


def ncomp(G):
    return nx.number_connected_components(G) if G.number_of_nodes() else 0


