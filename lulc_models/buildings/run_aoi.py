"""Buildings (footprint) model plugin — PLACEHOLDER (Vivek, 6 Oct 2026).

Follows lulc_models/CONTRACT.md so the wiring is real: it receives the AOI and the catalog
imagery, reports what it would read, and returns an EMPTY layer with empty coverage and a
warning, so a run outside the stored output shows "no data" instead of pretending the area has
no buildings. Replace `detect()` with the footprint model when it is delivered.

    python run_aoi.py [--check] --request request.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.environ.get("PYTHONPATH", os.path.join(os.path.dirname(__file__), "..")))
import common  # noqa: E402

MODEL = "buildings-placeholder"
VERSION = "0 (no detector installed)"


def imagery_summary(req):
    items = (req.get("imagery") or {}).get("esri_wayback_z18") or []
    return {"esri_wayback_z18": len(items), "releases": sorted({str(i.get("release")) for i in items}),
            "dates": sorted({i.get("date") for i in items if i.get("date")})}


def detect(req):
    """Placeholder: no footprint model yet. Returns no features."""
    return []


def main():
    req = common.load_request()
    img = imagery_summary(req)
    if "--check" in sys.argv:
        rep = {"plugin": "buildings", "ok": True, "placeholder": True, "imagery": img,
               "note": "placeholder model: runs, returns an empty layer and a warning"}
        print(json.dumps(rep, indent=1))
        return
    common.progress(0.1, "buildings placeholder: checking imagery")
    import geopandas as gpd
    feats = detect(req)
    g = gpd.GeoDataFrame({"class": [], "name": [], "confidence": [], "src_id": []}, geometry=[], crs=4326)
    g.to_file(req["out_gpkg"], layer="features", driver="GPKG")
    warn = ("buildings model is a placeholder (no detector installed) — no footprints were produced for this "
            "area; " + (f"{img['esri_wayback_z18']} Wayback item(s) were available"
                        if img["esri_wayback_z18"] else "and no Wayback imagery is on this machine for it"))
    common.write_result(req, coverage=None, image_source="Esri Wayback z18" if img["esri_wayback_z18"] else None,
                        image_dates=img["dates"], model=MODEL, model_version=VERSION,
                        counts={"features": len(feats)}, warnings=[warn])
    common.progress(1.0, "buildings placeholder: done (empty)")


if __name__ == "__main__":
    main()
