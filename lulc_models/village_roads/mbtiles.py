"""Read z18 imagery tiles from local MBTiles (replaces guj_fetch.py: no network, ever).

A tile may be in any of the given files. Sources are tried newest date first; a tile that is
missing, undecodable or black (max < BLACK_MAX on every band) in one source falls through to
the next. Missing/black tiles leave zeros in the mosaic and are not counted as valid.
"""
from __future__ import annotations

import io
import re
import sqlite3

import numpy as np
from PIL import Image

from aoi_grid import H, N, NT, Z

BLACK_MAX = 8


def _connect(path):
    """Read-only connection; falls back to immutable=1 for WAL files on read-only mounts."""
    last = None
    for q in ("mode=ro", "mode=ro&immutable=1"):
        try:
            con = sqlite3.connect("file:%s?%s" % (path, q), uri=True, check_same_thread=False)
            con.execute("select 1 from tiles limit 1").fetchall()
            return con
        except sqlite3.Error as e:
            last = e
    raise sqlite3.OperationalError("cannot open %s: %s" % (path, last))


class Source:
    def __init__(self, item: dict, idx: int):
        self.item, self.idx = item, idx
        self.path = item["path"]
        self.con = _connect(self.path)
        try:
            self.meta = dict(self.con.execute("select name, value from metadata").fetchall())
        except sqlite3.Error:
            self.meta = {}
        self.xyz = str(self.meta.get("scheme", "tms")).lower() == "xyz"
        desc = " ".join(str(self.meta.get(k, "")) for k in ("description", "name"))
        m = re.search(r"release\s+(\d+)", desc)
        self.release = str(item.get("release") or (m.group(1) if m else "") or "")
        m = re.search(r"(\d{4}-\d{2}-\d{2})", desc)
        self.date = str(item.get("date") or (m.group(1) if m else "") or "")

    def row(self, y):          # xyz row <-> stored row
        return y if self.xyz else N - 1 - y

    def tiles(self, c_lo, c_hi, y_lo, y_hi, with_data=True):
        """Yield (col, xyz_row[, data]) for z18 tiles in the inclusive xyz range."""
        a, b = sorted((self.row(y_lo), self.row(y_hi)))
        cols = "tile_column, tile_row" + (", tile_data" if with_data else "")
        cur = self.con.execute(
            "select %s from tiles where zoom_level=? and tile_column between ? and ? "
            "and tile_row between ? and ?" % cols, (Z, c_lo, c_hi, a, b))
        for rec in cur:
            yield (rec[0], self.row(rec[1])) + tuple(rec[2:])

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass


def decode(data: bytes):
    """JPEG/PNG bytes -> (256,256,3) uint8, or None if undecodable or black."""
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None
    if im.size != (256, 256):
        im = im.resize((256, 256), Image.BOX)
    a = np.asarray(im)
    if int(a.max()) < BLACK_MAX:
        return None
    return a


class TileSources:
    def __init__(self, items: list[dict]):
        self.sources = []
        self.errors = []
        for i, it in enumerate(items):
            try:
                self.sources.append(Source(it, i))
            except Exception as e:     # one bad file must not sink the others
                self.errors.append("%s: %s" % (it.get("path"), e))
        # newest first (date from the catalog item, else the MBTiles metadata); stable on ties
        self.sources.sort(key=lambda s: s.date, reverse=True)

    def presence(self, plan) -> np.ndarray:
        """Bool grid (plan shape): tile present in at least one source (data not decoded)."""
        P = np.zeros((plan.h, plan.w), bool)
        for s in self.sources:
            for c, y in s.tiles(plan.c0, plan.c0 + plan.w - 1, plan.r0, plan.r0 + plan.h - 1,
                                with_data=False):
                P[y - plan.r0, c - plan.c0] = True
        return P

    def mosaic(self, bx, by, halo=H):
        """(T*256, T*256, 3) uint8 mosaic of block (bx,by) plus halo, the (NT,NT) bool grid of
        valid core tiles, and {source idx: valid core tiles taken from it}."""
        T = NT + 2 * halo
        mos = np.zeros((T * 256, T * 256, 3), np.uint8)
        filled = np.zeros((T, T), bool)
        used = {}
        for s in self.sources:
            if filled.all():
                break
            for c, y, data in s.tiles(bx - halo, bx + NT - 1 + halo, by - halo, by + NT - 1 + halo):
                dx, dy = c - (bx - halo), y - (by - halo)
                if filled[dy, dx]:
                    continue
                a = decode(data)
                if a is None:
                    continue
                mos[dy * 256:(dy + 1) * 256, dx * 256:(dx + 1) * 256] = a
                filled[dy, dx] = True
                if halo <= dx < halo + NT and halo <= dy < halo + NT:
                    used[s.idx] = used.get(s.idx, 0) + 1
        return mos, filled[halo:halo + NT, halo:halo + NT].copy(), used

    def close(self):
        for s in self.sources:
            s.close()
