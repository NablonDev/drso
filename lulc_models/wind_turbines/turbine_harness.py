"""Scene-geometry helpers of the Exp-G turbine detector.

Copied from /mnt/drso-data/devesh/turbine_expg/harness.py: azimuth() and shadow_frame() are
verbatim. Dropped: the evaluation part (evaluate/score_all/_osm, which read ground truth from a
Windows share) and load()'s blob-cache lookup, including its stale /tmp/blobs_* fallback; the
plugin passes the blob table in directly (turbine_blobs.extract).

B is an (N,6) array of [cy, cx, area_px, major_px, minor_px, angle_deg] in PIXEL units.
"""
import math

import numpy as np


def azimuth(B):
    """Scene shadow azimuth, measured from the tile's own confident blobs.
    Never hardcode this - it is a sun-angle property and differs per scene."""
    if len(B) == 0:
        return 37.0, 0
    ratio = B[:, 3] / np.maximum(B[:, 4], 1e-6)
    m = (B[:, 2] >= 1000) & (ratio >= 3)
    if m.sum() < 5:
        m = (B[:, 2] >= 300) & (ratio >= 2)
    if m.sum() < 3:
        return 37.0, int(m.sum())
    # circular median on a 180-degree axis
    a = np.radians(B[m, 5] * 2.0)
    ang = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) / 2.0
    return float(ang % 180.0), int(m.sum())


def shadow_frame(B, gsd, az):
    """Rotate centroids into (along-azimuth, across-azimuth) metres."""
    th = math.radians(az); ct, st = math.cos(th), math.sin(th)
    y = B[:, 0] * gsd; x = B[:, 1] * gsd
    return y * ct + x * st, -y * st + x * ct
