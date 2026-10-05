# Copied verbatim from guj/code/vec_clip.py.
import numpy as np
def clip_polyline(P, lo, hi):
    """Clip polyline P (k,2) to the square [lo,hi]^2 (Liang-Barsky per segment).
    Returns the list of inside parts, each an (m,2) array."""
    parts, cur = [], []
    for a, b in zip(P[:-1], P[1:]):
        d = b - a; t0, t1 = 0.0, 1.0; ok = True
        for p, q in ((-d[0], a[0]-lo), (d[0], hi-a[0]), (-d[1], a[1]-lo), (d[1], hi-a[1])):
            if p == 0:
                if q < 0: ok = False; break
            else:
                t = q / p
                if p < 0: t0 = max(t0, t)
                else:     t1 = min(t1, t)
        if not ok or t0 > t1:
            if len(cur) > 1: parts.append(np.array(cur))
            cur = []; continue
        s, e = a + t0*d, a + t1*d
        if cur and np.allclose(cur[-1], s): cur.append(e)
        else:
            if len(cur) > 1: parts.append(np.array(cur))
            cur = [s, e]
        if t1 < 1.0:                       # left the box inside this segment
            parts.append(np.array(cur)); cur = []
    if len(cur) > 1: parts.append(np.array(cur))
    return parts
