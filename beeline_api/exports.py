"""Exports: DXF (UTM), shapefile, KML, tower and crossing schedules, PDF report, manifest, zip."""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import ezdxf
import geopandas as gpd
import pyogrio
from shapely.geometry import LineString, Point

TYPE_COLOURS = {"A": 7, "B1": 3, "B2": 3, "C1": 2, "C2": 2, "D45": 1, "D60": 1}


def _route_gdfs(route: dict, epsg: int | None = None):
    towers = [t for t in route["towers"] if t["type"] != "G"]
    line = gpd.GeoDataFrame({"route": [route["id"]], "name": [route["name"]], "length_km": [route["stats"]["length_km"]],
                             "towers": [route["stats"]["towers"]], "cost_cr": [round(route["cost"]["total"] / 1e7, 2)]},
                            geometry=[LineString(route["geometry"])], crs=4326)
    tw = gpd.GeoDataFrame({
        "tower_no": [t["no"] for t in towers], "type": [t["type"] for t in towers],
        "chain_km": [t["chainage_km"] for t in towers], "defl_deg": [t["deflection_deg"] for t in towers],
        "turn": [t["turn"] for t in towers], "span_m": [t["span_ahead_m"] or 0 for t in towers],
        "ground_z": [t["ground_z"] if t["ground_z"] is not None else -9999 for t in towers],
        "ext_m": [t["extension_m"] for t in towers], "reasons": [",".join(t["reasons"]) for t in towers],
        "flags": [",".join(t["violations"] + t.get("warnings", [])) for t in towers],
    }, geometry=[Point(t["lon"], t["lat"]) for t in towers], crs=4326)
    if epsg:
        line, tw = line.to_crs(epsg), tw.to_crs(epsg)
    return line, tw


def dxf_bytes(route: dict, epsg: int) -> bytes:
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 6  # metres
    msp = doc.modelspace()
    for name, colour in (("ROUTE", 5), ("TOWERS", 7), ("TOWER_LABELS", 8), ("GANTRY", 6)):
        doc.layers.add(name, color=colour)
    for t, c in TYPE_COLOURS.items():
        doc.layers.add(f"TOWER_{t}", color=c)
    pts = [(t["x_utm"], t["y_utm"]) for t in route["towers"]]
    msp.add_lwpolyline(pts, dxfattribs={"layer": "ROUTE"})
    for t in route["towers"]:
        if t["type"] == "G":
            msp.add_circle((t["x_utm"], t["y_utm"]), 6, dxfattribs={"layer": "GANTRY"})
            continue
        layer = f"TOWER_{t['type']}"
        s = 6 if t["type"] == "A" else 9
        x, y = t["x_utm"], t["y_utm"]
        msp.add_lwpolyline([(x - s, y - s), (x + s, y - s), (x + s, y + s), (x - s, y + s)], close=True,
                           dxfattribs={"layer": layer})
        label = f"T{t['no']} {t['type']}" + (f" {t['deflection_deg']:.1f}°{t['turn']}" if t["angle_point"] else "")
        msp.add_text(label, height=6, dxfattribs={"layer": "TOWER_LABELS"}).set_placement((x + 12, y + 12))
    out = io.StringIO()
    doc.write(out)
    return out.getvalue().encode("utf-8")


def shp_zip_bytes(route: dict, epsg: int, tmpdir: Path) -> bytes:
    line, tw = _route_gdfs(route, epsg)
    d = tmpdir / f"shp_r{route['id']}"
    d.mkdir(parents=True, exist_ok=True)
    pyogrio.write_dataframe(line, d / f"route{route['id']}_alignment.shp")
    pyogrio.write_dataframe(tw, d / f"route{route['id']}_towers.shp")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(d.iterdir()):
            z.write(p, p.name)
    return buf.getvalue()


def kml_bytes(route: dict) -> bytes:
    towers = [t for t in route["towers"] if t["type"] != "G"]
    coords = " ".join(f"{lon},{lat},0" for lon, lat in route["geometry"])
    pms = []
    for t in towers:
        pms.append(f"""<Placemark><name>T{t['no']} {t['type']}</name><description>km {t['chainage_km']:.3f} · deflection {t['deflection_deg']:.1f}° {t['turn']}</description>
<styleUrl>#{'angle' if t['type'] != 'A' else 'susp'}</styleUrl><Point><coordinates>{t['lon']},{t['lat']},0</coordinates></Point></Placemark>""")
    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Route {route['id']} — {route['name']}</name>
<Style id="line"><LineStyle><color>ff2f6fd8</color><width>3</width></LineStyle></Style>
<Style id="angle"><IconStyle><color>ff0000ff</color><scale>0.8</scale></IconStyle></Style>
<Style id="susp"><IconStyle><color>ffffffff</color><scale>0.5</scale></IconStyle></Style>
<Placemark><name>Alignment</name><styleUrl>#line</styleUrl><LineString><tessellate>1</tessellate><coordinates>{coords}</coordinates></LineString></Placemark>
<Folder><name>Towers</name>{''.join(pms)}</Folder></Document></kml>"""
    return kml.encode("utf-8")


def tower_schedule_csv(route: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["tower", "type", "chainage_km", "offset_m", "lat", "lon", "easting", "northing", "deflection_deg", "turn",
                "span_ahead_m", "ground_z_m", "extension_m", "reasons", "violations", "warnings"])
    for t in route["towers"]:
        w.writerow(["G" if t["type"] == "G" else t["no"], "Gantry" if t["type"] == "G" else t["type"], t["chainage_km"],
                    t["offset_m"], t["lat"], t["lon"], t["x_utm"], t["y_utm"], t["deflection_deg"], t["turn"],
                    t["span_ahead_m"], t["ground_z"], t["extension_m"], ";".join(t["reasons"]),
                    ";".join(t["violations"]), ";".join(t.get("warnings", []))])
    return buf.getvalue().encode("utf-8")


def crossing_schedule_csv(route: dict) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["from_tower", "to_tower", "span_m", "rule", "feature", "angle_deg", "major", "min_tower_type", "cost_lakh"])
    names = {e["data"].get("tower_a"): e["data"].get("feature") for e in route["events"] if e["type"] == "CROSSING"}
    for s in route["spans"]:
        for c in s["crossings"]:
            w.writerow([s["a"], s["b"], s["length_m"], c["rule"], names.get(s["a"], c["name"]), c["angle_deg"],
                        c.get("major", False), c["min_type"], round(c["cost_rs"] / 1e5, 2)])
    return buf.getvalue().encode("utf-8")


def report_pdf(run: dict, solve: dict, manifest: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f"{run['id']} route report")
    ss = getSampleStyleSheet()
    body = ss["BodyText"]
    body.fontSize = 9
    body.leading = 12
    el = [Paragraph(f"{run['id']} · rev {run['rev']} — {run.get('a_name') or 'A'} → {run.get('b_name') or 'B'}", ss["Title"]),
          Paragraph(f"{run['kv']} kV {run['circuit'].upper()} · band ± {run['band_km']:g} km · solved {solve.get('created_at', '')}", body),
          Spacer(1, 6)]
    notes = manifest.get("notices", [])
    for n in notes:
        el.append(Paragraph(f"<b>Notice.</b> {n}", body))
    res = solve["result"]
    if not res.get("feasible"):
        el.append(Paragraph("<b>No viable route.</b> " + res["blocked"]["event"]["text"], body))
    else:
        hdr = ["Route", "Length km", "vs bee-line", "Towers", "Angle pts", "Susp.", "Forest ha", "Agri ha", "Crossings", "Cost ₹ Cr"]
        rows = [hdr]
        for r in res["routes"]:
            s = r["stats"]
            rows.append([f"{r['id']} {r['name']}", f"{s['length_km']:.2f}", f"{s['vs_beeline_pct']:+.1f}%", s["towers"],
                         s["angle_points"], s["suspension_towers"], f"{s['forest_ha']:.1f}", f"{s['agri_ha']:.0f}",
                         s["crossings_total"], f"{r['cost']['total'] / 1e7:.1f}"])
        t = Table(rows, repeatRows=1)
        t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee"))]))
        el += [Spacer(1, 6), t]
        for r in res["routes"]:
            el += [PageBreak(), Paragraph(f"Route {r['id']} — {r['name']}", ss["Heading2"])]
            c = r["cost"]
            el.append(Paragraph(" · ".join(f"{k.replace('_', ' ')} ₹{v / 1e7:.2f} Cr" for k, v in c.items() if k != "total")
                                + f" · <b>total ₹{c['total'] / 1e7:.2f} Cr</b>", body))
            mix = ", ".join(f"{n} × {k}" for k, n in r["stats"]["tower_mix"].items() if n)
            el.append(Paragraph(f"Tower mix: {mix}.", body))
            el.append(Paragraph("<b>Decision log</b>", ss["Heading4"]))
            for e in r["events"]:
                src = e.get("source") or {}
                tag = f" <font color='#777777'>[{src.get('layer_name')}, {src.get('mode')}, {src['confidence']:.2f}]</font>" \
                    if src.get("confidence") is not None else ""
                el.append(Paragraph(f"{e['text']}{tag}", body))
    el += [PageBreak(), Paragraph("Run-level notes", ss["Heading2"])]
    for e in res.get("events_global", []):
        el.append(Paragraph(e["text"], body))
    el.append(Paragraph("<b>Overrides</b>", ss["Heading4"]))
    if not manifest.get("overrides"):
        el.append(Paragraph("None — the rule set is at defaults.", body))
    for o in manifest.get("overrides", []):
        el.append(Paragraph(f"{o['rule']} · {o['param']}: {o['default']} → {o['value']} — {o.get('reason') or ''}", body))
    doc.build(el)
    return buf.getvalue()


def build_all(run: dict, solve: dict, manifest: dict, out_dir: Path) -> dict:
    """Write every export to out_dir; return {name: path}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    res = solve["result"]
    epsg = res["frame"]["epsg"]
    files = {}

    def put(name, data: bytes):
        p = out_dir / name
        p.write_bytes(data)
        files[name] = p

    put("manifest.json", json.dumps(manifest, indent=2, default=str).encode("utf-8"))
    put("report.pdf", report_pdf(run, solve, manifest))
    for r in res.get("routes", []):
        i = r["id"]
        put(f"route{i}.dxf", dxf_bytes(r, epsg))
        put(f"route{i}_shp.zip", shp_zip_bytes(r, epsg, out_dir / "_tmp"))
        put(f"route{i}.kml", kml_bytes(r))
        put(f"route{i}_towers.csv", tower_schedule_csv(r))
        put(f"route{i}_crossings.csv", crossing_schedule_csv(r))
    zp = out_dir / f"{run['id']}_all.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in files.items():
            z.write(p, name)
        lg = out_dir / "layers.gpkg"
        if lg.exists():
            z.write(lg, "layers.gpkg")
    files[zp.name] = zp
    return {k: str(v) for k, v in files.items()}
