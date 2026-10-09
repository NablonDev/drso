"""Chainage frame: a UTM projection rotated so that x runs along the bee-line from A to B.

x = chainage (m) from A, y = lateral offset (m), positive to the left of A->B.
Rasters live on a grid aligned with this frame, so a raster column is a chainage slice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import transform as shp_transform


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


@dataclass
class Frame:
    a_lonlat: tuple[float, float]
    b_lonlat: tuple[float, float]
    half_width_m: float
    margin_m: float
    res_m: float

    def __post_init__(self):
        mid = ((self.a_lonlat[0] + self.b_lonlat[0]) / 2, (self.a_lonlat[1] + self.b_lonlat[1]) / 2)
        self.epsg = utm_epsg(*mid)
        self.crs = CRS.from_epsg(self.epsg)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)
        ax, ay = self._fwd.transform(*self.a_lonlat)
        bx, by = self._fwd.transform(*self.b_lonlat)
        self.A = np.array([ax, ay])
        self.B = np.array([bx, by])
        d = self.B - self.A
        self.length_m = float(np.hypot(*d))
        if self.length_m < 1000:
            raise ValueError("A and B are less than 1 km apart")
        self.u = d / self.length_m                 # along-line unit vector
        self.v = np.array([-self.u[1], self.u[0]])  # left normal
        self.bearing_rad = math.atan2(self.u[1], self.u[0])
        # raster extent in frame coordinates
        self.x0 = -self.margin_m
        self.x1 = self.length_m + self.margin_m
        self.y_top = self.half_width_m + self.margin_m
        self.y_bot = -self.y_top
        self.width = int(math.ceil((self.x1 - self.x0) / self.res_m))
        self.height = int(math.ceil((self.y_top - self.y_bot) / self.res_m))
        r = self.res_m
        ux, uy = self.u
        vx, vy = self.v
        # pixel (col,row) -> UTM, rotation included (rasterio/GDAL accept a rotated geotransform)
        self.transform = Affine(
            r * ux, -r * vx, self.A[0] + self.x0 * ux + self.y_top * vx,
            r * uy, -r * vy, self.A[1] + self.x0 * uy + self.y_top * vy,
        )

    # ---- coordinate conversions -------------------------------------------------------
    def lonlat_to_utm(self, lon, lat):
        return self._fwd.transform(lon, lat)

    def utm_to_lonlat(self, x, y):
        return self._inv.transform(x, y)

    def utm_to_frame(self, x, y):
        dx = np.asarray(x) - self.A[0]
        dy = np.asarray(y) - self.A[1]
        return dx * self.u[0] + dy * self.u[1], dx * self.v[0] + dy * self.v[1]

    def frame_to_utm(self, fx, fy):
        fx = np.asarray(fx, dtype=float)
        fy = np.asarray(fy, dtype=float)
        return (self.A[0] + fx * self.u[0] + fy * self.v[0],
                self.A[1] + fx * self.u[1] + fy * self.v[1])

    def frame_to_lonlat(self, fx, fy):
        return self.utm_to_lonlat(*self.frame_to_utm(fx, fy))

    def frame_to_pixel(self, fx, fy):
        """Continuous pixel coords (col, row) where integer+0.5 is a pixel centre."""
        return (np.asarray(fx) - self.x0) / self.res_m, (self.y_top - np.asarray(fy)) / self.res_m

    # ---- geometry helpers ------------------------------------------------------------
    def geom_to_utm(self, g):
        return shp_transform(self._fwd.transform, g)

    def geom_to_lonlat(self, g):
        return shp_transform(self._inv.transform, g)

    def geoms_utm_to_frame(self, geoms):
        """Vectorised geom_utm_to_frame over an array of shapely geometries (one coordinate pass)."""
        import shapely
        u, v, a = self.u, self.v, self.A

        def f(xy):
            dx = xy[:, 0] - a[0]
            dy = xy[:, 1] - a[1]
            return np.column_stack([dx * u[0] + dy * u[1], dx * v[0] + dy * v[1]])
        return shapely.transform(np.asarray(geoms, dtype=object), f)

    def geom_utm_to_frame(self, g):
        # translate then rotate by -bearing about origin
        g = affinity.translate(g, -self.A[0], -self.A[1])
        return affinity.rotate(g, -self.bearing_rad, origin=(0, 0), use_radians=True)

    def band_polygon_lonlat(self, margin: bool = False) -> Polygon:
        m = self.margin_m if margin else 0.0
        hw = self.half_width_m + m
        pts = [(-m, -hw), (self.length_m + m, -hw), (self.length_m + m, hw), (-m, hw)]
        xs, ys = self.frame_to_utm([p[0] for p in pts], [p[1] for p in pts])
        lon, lat = self.utm_to_lonlat(xs, ys)
        return Polygon(list(zip(lon, lat)))


def band_polygon(a_lonlat, b_lonlat, half_width_km: float, margin_m: float = 0.0) -> Polygon:
    """Band polygon in lon/lat without building rasters (used by the API for area / reuse)."""
    f = Frame(tuple(a_lonlat), tuple(b_lonlat), half_width_km * 1000.0, margin_m, 100.0)
    return f.band_polygon_lonlat(margin=margin_m > 0)
