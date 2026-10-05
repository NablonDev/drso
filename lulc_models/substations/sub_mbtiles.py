"""MBTiles data footprint (where tiles actually exist), computed from the `tiles` table.

The Wayback archives are clipped to a corridor polygon, so the raster extent GDAL reports is
the tile bounding box, not the imaged area. Coverage in result.json must be the imaged area,
so it is rebuilt here from the tile index: one cell per z18 tile, polygonised in EPSG:3857
(tile edges are straight lines there).
"""
from __future__ import annotations

import math
import sqlite3

import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, box
from shapely.ops import unary_union

R = 6378137.0 * math.pi


def _connect(path: str) -> sqlite3.Connection:
    # immutable=1: never create a journal/lock next to a read-only archive
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)


def metadata(path: str) -> dict:
    with _connect(path) as c:
        return {k: v for k, v in c.execute("select name, value from metadata")}


def zoom_of(path: str) -> int:
    md = metadata(path)
    if md.get("maxzoom") is not None:
        return int(md["maxzoom"])
    with _connect(path) as c:
        return int(c.execute("select max(zoom_level) from tiles").fetchone()[0])


def _merc_x_to_tile(x: float, z: int) -> int:
    return int(math.floor((x + R) / (2 * R) * 2 ** z))


def _merc_y_to_tile(y: float, z: int) -> int:          # XYZ (row 0 at the top)
    return int(math.floor((R - y) / (2 * R) * 2 ** z))


def footprint_3857(path: str, bounds_3857: tuple[float, float, float, float], z: int | None = None):
    """Union of the tiles present in `path` within `bounds_3857` (minx, miny, maxx, maxy).

    Returns a shapely geometry in EPSG:3857 (empty when no tile exists there) and the tile count.
    """
    z = zoom_of(path) if z is None else z
    n = 2 ** z
    minx, miny, maxx, maxy = bounds_3857
    x0, x1 = max(0, _merc_x_to_tile(minx, z)), min(n - 1, _merc_x_to_tile(maxx, z))
    y0, y1 = max(0, _merc_y_to_tile(maxy, z)), min(n - 1, _merc_y_to_tile(miny, z))
    if x1 < x0 or y1 < y0:
        return box(0, 0, 0, 0).buffer(0), 0
    # MBTiles stores TMS rows: tile_row = 2^z - 1 - y
    r0, r1 = n - 1 - y1, n - 1 - y0
    grid = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    with _connect(path) as c:
        cur = c.execute("select tile_column, tile_row from tiles where zoom_level=? and "
                        "tile_column between ? and ? and tile_row between ? and ?",
                        (z, x0, x1, r0, r1))
        for col, row in cur:
            grid[(n - 1 - row) - y0, col - x0] = 1
    cnt = int(grid.sum())
    if not cnt:
        return box(0, 0, 0, 0).buffer(0), 0
    size = 2 * R / n
    tf = Affine(size, 0, -R + x0 * size, 0, -size, R - y0 * size)
    polys = [shape(g) for g, v in shapes(grid, mask=grid.astype(bool), transform=tf) if v == 1]
    return unary_union(polys), cnt
