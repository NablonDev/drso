"""Experiment F: model-based turbine grouping using the scene's shadow geometry
plus the wind-farm layout prior.

WHY THE ANISOTROPIC DBSCAN SATURATED
------------------------------------
Exp E linked blobs generously along the sun azimuth and stingily across it, then
accepted any cluster with >= 700 px of total area. Its parameters saturate because
the *grouping* was never the bottleneck - inspection of the blob constellation
around matched turbines shows the grouping is already essentially correct. What
saturated was the *acceptance rule*: "total area >= 700 px" is a blunt proxy that
lets in any large bright cluster and shuts out weak but structurally-valid ones.

Two facts from the blob data drive this implementation:

1. A strong turbine is NOT an even scatter of fragments. It is one dominant
   streak blob (10k-20k px, major axis 130-220 m, major-axis angle within 1-3 deg
   of the scene azimuth) surrounded by a handful of 20-130 px specks. The "median
   3 fragments" case is the *weak* turbines. So the useful anchor is a blob whose
   own major axis is azimuth-aligned - columns 3,4,5 of B, not centroid position.

2. Shadow length is a scene constant (hub height / tan(sun elevation)), and it
   differs hugely between tiles: median streak major axis is 98 m on 2660409331
   and 205 m on 265889841, with dominant-blob areas of ~3.5k vs ~12k px. Any
   absolute pixel threshold is therefore wrong on at least one tile. Every size
   threshold here is expressed as a fraction of the scene's own measured shadow
   scale, calibrated the same way harness.azimuth() calibrates the azimuth.

THE METHOD
----------
  Stage 0  Scene calibration. From the tile's own confident streak blobs measure
           the azimuth (harness.azimuth), the characteristic shadow length L and
           the characteristic streak area A. Fewer than 3 such blobs means the
           scene has no coherent shadow structure and the method declines to run.

  Stage 1  Seeds. A blob is a turbine seed if its major axis is at least FMAJ*L
           AND its major-axis direction is within DA degrees of the scene azimuth;
           or, as a fallback for turbines whose mask is a fat clump rather than a
           streak, if its area is at least FAREA*A. Noise blobs are small AND
           randomly oriented, so they fail both.

  Stage 2  Grouping. Anisotropic linking (the Exp E kernel, eps=EPS, ANISO across)
           applied ONLY to seeds. Non-seed blobs cannot start a group; they are
           absorbed afterwards into an azimuth-aligned box around a group purely
           so the reported centroid and area cover the whole object.

  Stage 3  Layout prior. Turbines obey a hard minimum spacing (~111 m) and never
           occur alone - they occur in farms. So:
             (a) non-max suppression at SPACING metres: two candidates closer than
                 the physical spacing floor cannot both be turbines, keep the larger;
             (b) isolation rejection: drop any candidate whose nearest surviving
                 neighbour is further than ISOK x the scene's own median
                 nearest-neighbour distance. Scene-relative, so it adapts to a
                 dense farm and a sparse one alike.
           (b) is what carries the precision gain. (a) fires on no candidate at the
           tuned configuration - stage 1 already leaves no sub-spacing pairs - so it
           is kept as a cheap production guard, not because it earns anything here.
           The stage-2 absorption of leftover blobs is likewise inert on these three
           tiles: it shifts centroids by less than the 100 m match tolerance.

RESULT (threshold 160, combined over the 3 tiles)
    TP=73 FP=4  P=0.948 R=0.912 F1=0.930
versus the anisotropic-DBSCAN target F1=0.866 and production baseline F1=0.792.

HONEST CAVEAT - read before deploying
    Tile 2660409251 (zero turbines) returns zero candidates, but that is produced
    by the Stage 0 gate, not by the per-candidate geometry. With the gate replaced
    by a fallback shadow length, that tile emits 2-10 false positives depending on
    the fallback and threshold. The gate is therefore load-bearing and it has been
    validated against exactly one negative tile. See ablation "E" in __main__.

Copied from /mnt/drso-data/devesh/turbine_expg/turbine_detect.py (Exp-G, CFG unchanged). The
research report section (__main__, _row, _aniso_dbscan) was dropped: it scores against a Windows
share and is not used in production.
"""
import math

import numpy as np
from sklearn.cluster import DBSCAN

import turbine_harness as H  # azimuth, shadow_frame (copied from turbine_expg/harness.py)


# ----------------------------------------------------------------- configuration
CFG = dict(
    # Stage 0 - scene calibration
    NCONF_MIN=3,     # confident streak blobs needed to trust the scene scale
    # Stage 1 - seed test (all sizes relative to the scene's own shadow scale)
    FMAJ=0.08,       # seed major axis >= FMAJ * L
    DA=22.0,         # seed major-axis direction within DA deg of the scene azimuth
    FAREA=0.25,      # OR seed area >= FAREA * A (fat-clump turbines)
    # Stage 2 - grouping
    EPS=140.0,       # metres along the azimuth
    ANISO=4.0,       # across-azimuth tolerance is EPS / ANISO
    CW=25.0,         # absorption half-width across azimuth, metres
    FCL=0.75,        # absorption half-length along azimuth, as a fraction of L
    # Stage 3 - wind-farm layout prior
    SPACING=95.0,    # non-max suppression radius, metres (turbine spacing floor 111 m)
    ISOK=3.5,        # drop candidates isolated beyond ISOK x median nearest-neighbour
)

DEFAULT_THRESH = 160


# ------------------------------------------------------------- stage 0: calibrate
def calibrate(B, gsd, fallback_L=None):
    """Measure the scene's characteristic shadow length L (metres) and streak
    area A (pixels) from the tile's own confident streak blobs.

    Same blob population harness.azimuth() uses, for consistency. Returns
    L = 0 when the scene has too few coherent streaks to be calibrated, which is
    the signal that the scene is not a wind farm. fallback_L overrides that for
    ablation purposes only.
    """
    if len(B) == 0:
        return 0, 0.0, 0.0
    el = B[:, 3] / np.maximum(B[:, 4], 1e-6)
    maj = B[:, 3] * gsd
    m = (B[:, 2] >= 1000) & (el >= 3)
    if m.sum() < 5:
        m = (B[:, 2] >= 300) & (el >= 2)
    n = int(m.sum())
    if n < 3:
        return (n, float(fallback_L), 300.0) if fallback_L else (n, 0.0, 0.0)
    return n, float(np.median(maj[m])), float(np.median(B[m, 2]))


# --------------------------------------------------------- stages 1-2: build groups
def _groups(B, gsd, az, L, A, cfg):
    u, v = H.shadow_frame(B, gsd, az)
    maj = B[:, 3] * gsd
    da = np.abs(((B[:, 5] - az + 90.0) % 180.0) - 90.0)

    # stage 1 - seeds: azimuth-aligned streak, or a clump at the scene's streak scale
    seed = ((maj >= cfg["FMAJ"] * L) & (da <= cfg["DA"])) | (B[:, 2] >= cfg["FAREA"] * A)
    S = np.where(seed)[0]
    if not len(S):
        return []

    # stage 2 - anisotropic linking over seeds only
    lab = DBSCAN(eps=cfg["EPS"], min_samples=1).fit_predict(
        np.column_stack([u[S], v[S] * cfg["ANISO"]]))
    gd = {}
    for k, i in zip(lab, S):
        gd.setdefault(k, []).append(i)
    gl = list(gd.values())

    # absorb leftover blobs into an azimuth-aligned box so centroids/areas are complete
    taken = np.zeros(len(B), bool)
    taken[S] = True
    cen = [(np.mean([u[i] for i in g]), np.mean([v[i] for i in g])) for g in gl]
    for j in range(len(B)):
        if taken[j]:
            continue
        best, bd = -1, 1e18
        for k, (gu, gv) in enumerate(cen):
            if abs(v[j] - gv) <= cfg["CW"] and abs(u[j] - gu) <= cfg["FCL"] * L:
                d = abs(u[j] - gu) + cfg["ANISO"] * abs(v[j] - gv)
                if d < bd:
                    bd, best = d, k
        if best >= 0:
            gl[best].append(j)
            taken[j] = True

    out = []
    for g in gl:
        idx = np.array(g)
        a = B[idx, 2]
        tot = float(a.sum())
        out.append(dict(cy=float((B[idx, 0] * a).sum() / tot),
                        cx=float((B[idx, 1] * a).sum() / tot),
                        area=tot, n=len(idx)))
    return out


# ----------------------------------------------------- stage 3: wind-farm layout prior
def _layout_prior(C, gsd, cfg):
    if len(C) < 2:
        return C
    P = np.array([[c["cy"] * gsd, c["cx"] * gsd] for c in C])
    A = np.array([c["area"] for c in C])

    # (a) two candidates closer than the turbine spacing floor cannot both be real
    keep = []
    for i in np.argsort(-A):
        if all(math.hypot(P[i, 0] - P[j, 0], P[i, 1] - P[j, 1]) >= cfg["SPACING"]
               for j in keep):
            keep.append(i)
    keep = np.array(keep)
    if len(keep) < 3:
        return [C[i] for i in keep]

    # (b) turbines occur in farms - drop candidates isolated on the scene's own scale
    Q = P[keep]
    D = np.hypot(Q[:, None, 0] - Q[None, :, 0], Q[:, None, 1] - Q[None, :, 1])
    np.fill_diagonal(D, 1e12)
    nn1 = D.min(1)
    keep = keep[nn1 <= cfg["ISOK"] * np.median(nn1)]
    return [C[i] for i in keep]


# ------------------------------------------------------------------- the detector
def make(fallback_L=None, use_layout=True, use_seeds=True, **over):
    """Build a clustering fn(B, gsd, az) -> list of candidate dicts.

    fallback_L   : ablation - run even when the scene scale is uncalibrated
    use_layout   : ablation - disable stage 3
    use_seeds    : ablation - disable stage 1 (every blob becomes a seed)
    **over       : override any CFG entry
    """
    cfg = dict(CFG)
    cfg.update(over)

    def fn(B, gsd, az):
        if len(B) == 0:
            return []
        nconf, L, A = calibrate(B, gsd, fallback_L)
        if L <= 0 or nconf < cfg["NCONF_MIN"]:
            if fallback_L is None:
                return []                      # scene has no coherent shadow structure
        c = dict(cfg)
        if not use_seeds:
            c["FMAJ"], c["DA"], c["FAREA"] = 0.0, 90.0, 0.0
        C = _groups(B, gsd, az, L, A, c)
        return _layout_prior(C, gsd, c) if use_layout else C

    return fn


def detect(B, gsd, az):
    """Production entry point with the tuned configuration."""
    return make()(B, gsd, az)
