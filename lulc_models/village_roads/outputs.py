"""Merge per-block centrelines, clip to the AOI, write out.gpkg and the coverage polygon.

Reprojection and clipping use the system ogr2ogr (needs GDAL built with GEOS). The final
GeoPackage is written directly with sqlite3 so the schema (column types) is fixed even when
the output is empty or a column is all-null.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import subprocess

import numpy as np

CLASS = "village_road"
FIELDS = [("class", "TEXT"), ("confidence", "REAL"), ("name", "TEXT"), ("src_id", "TEXT"),
          ("block", "TEXT"), ("bridged", "INTEGER"), ("surface_width_m", "REAL"),
          ("width_p90_m", "REAL"), ("lane_tier", "TEXT"), ("length_m", "REAL")]
WGS84_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
             'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
             'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],'
             'AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]')
CRS3857 = {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::3857"}}


def write_json(path, obj):
    tmp = str(path) + ".t"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def write_aoi(path, geom):
    write_json(path, {"type": "FeatureCollection",
                      "features": [{"type": "Feature", "properties": {}, "geometry": geom}]})


# ---- merge + clip ------------------------------------------------------------------------
def merge_blocks(vec_dir, tags):
    """Concatenate block GeoJSONs (EPSG:3857) in tag order; src_id = vr_<block>_<k>."""
    feats = []
    for bx, by in sorted(tags, key=lambda b: (b[1], b[0])):
        p = os.path.join(vec_dir, "%d_%d.geojson" % (bx, by))
        if not os.path.exists(p):
            continue
        with open(p) as f:
            fs = json.load(f).get("features", [])
        for k, ft in enumerate(fs):
            ft["properties"]["src_id"] = "vr_%d_%d_%d" % (bx, by, k)
            feats.append(ft)
    return feats


def ogr_clip(src, aoi_path, dst, s_srs="EPSG:3857"):
    """Reproject src (GeoJSON) to EPSG:4326 and clip to aoi_path (GeoJSON, 4326)."""
    if os.path.exists(dst):
        os.remove(dst)
    subprocess.run(["ogr2ogr", "-q", "-f", "GeoJSON", "-s_srs", s_srs, "-t_srs", "EPSG:4326",
                    "-clipdst", aoi_path, dst, src], check=True, capture_output=True, text=True)
    with open(dst) as f:
        return json.load(f).get("features", [])


def _parts(geom, kind):
    """Flatten a (Multi)LineString / (Multi)Polygon / GeometryCollection to parts of `kind`."""
    if geom is None:
        return []
    t = geom["type"]
    if t == kind:
        return [geom["coordinates"]]
    if t == "Multi" + kind:
        return list(geom["coordinates"])
    if t == "GeometryCollection":
        out = []
        for g in geom["geometries"]:
            out += _parts(g, kind)
        return out
    return []


def ground_length_m(coords):
    a = np.asarray(coords, float)
    if len(a) < 2:
        return 0.0
    lat = np.radians((a[1:, 1] + a[:-1, 1]) / 2)
    dx = np.radians(np.diff(a[:, 0])) * np.cos(lat); dy = np.radians(np.diff(a[:, 1]))
    return float(6371008.8 * np.sqrt(dx * dx + dy * dy).sum())


def clip_lines(feats_3857, aoi_path, work):
    """-> list of (coords 4326, props). Features cut into several pieces by the AOI get
    src_id suffixes _a, _b, ...; length_m is the ground length of the clipped piece."""
    if not feats_3857:
        return []
    src = os.path.join(work, "merged_3857.geojson"); dst = os.path.join(work, "clipped_4326.geojson")
    write_json(src, {"type": "FeatureCollection", "crs": CRS3857, "features": feats_3857})
    out = []
    for ft in ogr_clip(src, aoi_path, dst):
        parts = [p for p in _parts(ft.get("geometry"), "LineString") if len(p) >= 2]
        for i, p in enumerate(parts):
            props = dict(ft["properties"])
            if len(parts) > 1:
                props["src_id"] = "%s_%s" % (props["src_id"], _suffix(i))
            props["length_m"] = round(ground_length_m(p), 1)
            if props["length_m"] <= 0:
                continue
            out.append((p, props))
    return out


def _suffix(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return s


# ---- GeoPackage ---------------------------------------------------------------------------
def _gpkg_line(coords, srs_id=4326):
    a = np.asarray(coords, float)[:, :2]
    env = (a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max())
    hdr = b"GP" + bytes([0, 0b00000011]) + struct.pack("<i", srs_id) + struct.pack("<4d", *env)
    wkb = struct.pack("<BII", 1, 2, len(a)) + a.astype("<f8").tobytes()
    return hdr + wkb, env


def write_gpkg(path, rows, layer="features"):
    """rows: list of (coords [[lon,lat],...], props). Writes a GeoPackage 1.4 with one
    LINESTRING layer in EPSG:4326 and the FIELDS schema, then asks GDAL for a spatial index."""
    tmp = str(path) + ".part"
    for f in (tmp, path):
        if os.path.exists(f):
            os.remove(f)
    con = sqlite3.connect(tmp)
    con.execute("PRAGMA application_id = 1196444487")
    con.execute("PRAGMA user_version = 10400")
    con.executescript("""
    CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER NOT NULL PRIMARY KEY,
      organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL,
      description TEXT);
    CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
      identifier TEXT UNIQUE, description TEXT DEFAULT '',
      last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
      min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER,
      CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
    CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
      geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
      CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
      CONSTRAINT uk_gc_table_name UNIQUE (table_name),
      CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
      CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id));
    """)
    con.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", [
        ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", "undefined cartesian coordinate reference system"),
        ("Undefined geographic SRS", 0, "NONE", 0, "undefined", "undefined geographic coordinate reference system"),
        ("WGS 84 geodetic", 4326, "EPSG", 4326, WGS84_WKT, "longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid")])
    cols = ", ".join('"%s" %s' % (n, t) for n, t in FIELDS)
    con.execute('CREATE TABLE "%s" (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, geom LINESTRING, %s)'
                % (layer, cols))
    names = [n for n, _ in FIELDS]
    bb = [math.inf, math.inf, -math.inf, -math.inf]
    recs = []
    for coords, props in rows:
        blob, (x0, x1, y0, y1) = _gpkg_line(coords)
        bb = [min(bb[0], x0), min(bb[1], y0), max(bb[2], x1), max(bb[3], y1)]
        recs.append([blob] + [props.get(n) for n in names])
    con.executemany('INSERT INTO "%s" (geom, %s) VALUES (?, %s)'
                    % (layer, ", ".join('"%s"' % n for n in names), ", ".join("?" * len(names))), recs)
    if not recs:
        bb = [None] * 4
    con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id) "
                "VALUES (?, 'features', ?, ?, ?, ?, ?, 4326)", (layer, layer, *bb))
    con.execute("INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', 'LINESTRING', 4326, 0, 0)", (layer,))
    con.commit(); con.close()
    os.replace(tmp, path)
    warn = None
    try:
        r = subprocess.run(["ogrinfo", "-q", str(path), "-sql",
                            "SELECT CreateSpatialIndex('%s','geom')" % layer],
                           capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            warn = "spatial index not created: %s" % r.stderr.strip()[:200]
    except Exception as e:
        warn = "spatial index not created: %s" % e
    return warn


def feature_rows(clipped):
    rows = []
    for coords, p in clipped:
        rows.append((coords, {"class": CLASS, "confidence": None, "name": None, "src_id": p["src_id"],
                              "block": p.get("block"), "bridged": int(p.get("bridged") or 0),
                              "surface_width_m": p.get("surface_width_m"),
                              "width_p90_m": p.get("width_p90_m"), "lane_tier": p.get("lane_tier"),
                              "length_m": p.get("length_m")}))
    return rows


# ---- coverage -------------------------------------------------------------------------------
EMPTY_COVERAGE = {"type": "MultiPolygon", "coordinates": []}


def coverage(valid_tiles, transform, aoi_path, work):
    """Valid-tile grid (bool, plan grid) -> MultiPolygon (EPSG:4326) clipped to the AOI."""
    if valid_tiles is None or not valid_tiles.any():
        return EMPTY_COVERAGE
    from rasterio.features import shapes
    feats = [{"type": "Feature", "properties": {}, "geometry": g}
             for g, v in shapes(valid_tiles.astype(np.uint8), mask=valid_tiles, transform=transform,
                                connectivity=4) if v == 1]
    src = os.path.join(work, "coverage_3857.geojson"); dst = os.path.join(work, "coverage_4326.geojson")
    write_json(src, {"type": "FeatureCollection", "crs": CRS3857, "features": feats})
    polys = []
    for ft in ogr_clip(src, aoi_path, dst):
        polys += _parts(ft.get("geometry"), "Polygon")
    return {"type": "MultiPolygon", "coordinates": polys}


def multipolygon_area_km2(mp):
    """Approximate ground area of a 4326 MultiPolygon (local equirectangular per ring)."""
    tot = 0.0
    for poly in mp.get("coordinates", []):
        for k, ring in enumerate(poly):
            a = np.asarray(ring, float)
            if len(a) < 4:
                continue
            lat0 = math.radians(a[:, 1].mean())
            x = np.radians(a[:, 0]) * math.cos(lat0) * 6371.0088; y = np.radians(a[:, 1]) * 6371.0088
            ar = abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            tot += ar if k == 0 else -ar
    return tot
