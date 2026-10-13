#!/usr/bin/env python
"""Add the OSM power lines the import used to drop (Vivek, 13 Oct 2026): HVDC (500 / 800 kV) and lines
without a voltage tag (voltage inferred from a parallel tagged line, else 400 kV — see
osm_power.infer_untagged), with their towers. Additive: what is already stored (incl. the AOI refresh
of 9 Oct) is left as it is; a way or a (voltage, tower) already in the store is not added again.

    python scripts/add_dropped_power_lines.py [--pbf data/bucket/osm/india-power.osm.pbf]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osmium  # noqa: E402

from beeline_api import db, osm_power as op  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", default=str(op.OSM_DIR / "india-power.osm.pbf"))
    a = ap.parse_args()
    op.init_schema()
    towers, lines = {}, []

    class H(osmium.SimpleHandler):
        def node(self, n):
            if n.tags.get("power") in op.TOWER_TAGS:
                towers[n.id] = (n.location.lon, n.location.lat, dict(n.tags))

        def way(self, w):
            tags = dict(w.tags)
            if not op._is_line(tags):
                return
            coords = [(nd.lon, nd.lat) for nd in w.nodes if nd.location.valid()]
            lines.append((w.id, tags, coords, [nd.ref for nd in w.nodes]))

    H().apply_file(a.pbf, locations=True)
    rows, _ = op._rows_from(lines, towers)
    have_lines = {r["osm_id"] for r in db.all_("select osm_id from osm_power where kind='line'")}
    have_towers = {(r["voltage_kv"], r["osm_id"]) for r in db.all_("select voltage_kv, osm_id from osm_power where kind='tower'")}
    new_lines = [r for r in rows if r[1] == "line" and r[3] not in have_lines]
    new_ids = {r[3] for r in new_lines}
    nodes_of_new = {n for oid, tags, coords, nids in lines if oid in new_ids for n in nids}
    new_towers = [r for r in rows if r[1] == "tower" and r[3] in nodes_of_new and (r[0], r[3]) not in have_towers]
    counts = {}
    for r in new_lines + new_towers:
        k = f"{r[1]}_{r[0]}" + ("_assumed" if r[4].get("voltage_assumed") else "")
        counts[k] = counts.get(k, 0) + 1
    load = db.one("insert into osm_power_loads (kind, source, status, counts) values ('india', %s, 'done', %s) returning id",
                  (f"Added: HVDC and untagged lines from {Path(a.pbf).name} (13 Oct 2026)", db.J(counts)))
    with db.conn() as c:
        with c.transaction():
            with c.cursor() as cur:
                op._copy_rows(cur, new_lines + new_towers, load["id"])
    print("load", load["id"], counts, flush=True)


if __name__ == "__main__":
    main()
