"""Decision log: typed events + the templates that turn them into sentences.

No text is generated at read time beyond filling these templates. The log is the deliverable;
the "Why this route" page is only its renderer.
"""
from __future__ import annotations

TEMPLATES = {
    "TURN_TO_AVOID": "At km {chainage_km:.1f}, the route turns {deflection:.0f}° {dir} at tower {tower} ({type}) to clear {feature} by {clearance_m:.0f} m.",
    "TURN_TO_CROSS": "At km {chainage_km:.1f}, the route turns {deflection:.0f}° {dir} at tower {tower} ({type}) to cross {feature} under its crossing rule ({crossing_note}).",
    "RULE_WARNING": "Tower {tower} at km {chainage_km:.1f} {what}.",
    "DETOUR": "Between km {from_km:.1f} and km {to_km:.1f} the route runs up to {max_offset_m:.0f} m {side} of the bee-line to clear {feature}; closest approach {clearance_m:.0f} m.",
    "TURN_TO_KEEP_CLEAR": "At km {chainage_km:.1f}, the route turns {deflection:.0f}° {dir} at tower {tower} ({type}) to keep clear of {feature} ahead, passing it at {clearance_m:.0f} m.",
    "TURN_BACK": "At km {chainage_km:.1f}, the route turns {deflection:.0f}° {dir} at tower {tower} ({type}) back towards the bee-line, the obstacle behind it cleared.",
    "TURN": "At km {chainage_km:.1f}, the route turns {deflection:.0f}° {dir} at tower {tower} ({type}) — the cheapest line through the cost surface here, no single feature forces it.",
    "CROSSING": "Crosses {feature} at {angle_deg:.0f}° on a {span_m:.0f} m span between towers {tower_a} and {tower_b} (Type {type_a} / {type_b}).",
    "SECTION_POINT": "Tower {tower} at km {chainage_km:.1f} is a section point (Type {type}) — tension towers may not be more than {section_km:g} km apart.",
    "CLEARANCE_TOWER": "Tower {tower} at km {chainage_km:.1f} was added on the ridge: the span could not keep {clearance_req:g} m ground clearance even with +18 m extensions (DEM: {dem}).",
    "EXTENSION": "Tower {tower} at km {chainage_km:.1f} carries a +{extension_m:.0f} m extension to keep {clearance_req:g} m ground clearance (DEM: {dem}).",
    "CROSSING_TOWERS": "The {span_m:.0f} m crossing span between towers {tower_a} and {tower_b} needs crossing towers +{extension_m:.0f} m taller to keep {clearance_req:g} m ground clearance at mid-span sag (DEM: {dem}).",
    "CLEARANCE_SHORT": "Span from tower {tower_a} at km {chainage_km:.1f} has {clearance_m:.1f} m ground clearance against {clearance_req:g} m required on the {dem}. Check on site.",
    "VIOLATION": "Tower {tower} at km {chainage_km:.1f}: {what}. Spotting could not repair this on the 50 m solver grid — move the tower before issue.",
    "FOREST_ENTRY": "The corridor takes {forest_ha:.2f} ha of forest.{threshold_note}",
    "NOC_REQUIRED": "{n_towers} towers stand within {radius_km:g} km of {feature} — an AAI NOC is required; height governed by the {height_source}.",
    "BEELINE_BLOCKER": "The bee-line passes {length_km:.1f} km through {feature} ({rule}){stretches_note}.",
    "LAYER_EMPTY": "{rules} did not fire — the {layer} extraction found nothing in the band.",
    "LOW_CONF": "{decision} rests on {layer} at {confidence:.2f} confidence. Verify on site.",
    "NO_COVERAGE": "{layer} has no data over {uncovered_km2:.0f} km² of the band ({ranges}). Rules on this layer do not fire there.",
    "RULE_OFF": "Rule “{rule}” was switched off for this run.",
    "BAND_BLOCKED": "No route inside the ± {band_km:g} km band: the corridor is closed at km {from_km:.1f} – {to_km:.1f} by {feature}.",
    "DEM_NOTICE": "Ground clearance was checked on {dem}; the research spec asks for 10 m or finer.",
    "APART_FROM": "Route {route} is the second least-cost route: the least-cost line kept out of route {other}'s corridor.",
    "ROUTE_NOT_FOUND": "Route {route} ({name}) could not be found under its rules: {why}",
    "APPROVAL": "Approval: {item} — about {months:g} months.",
    "MANUAL_EDIT": "Route {route}: {n} tower(s) moved by hand ({towers}) from solve {parent}; the moved towers and their neighbours were re-typed and re-checked, nothing else changed.",
    "PRECEDENT": "Route {route} found no path under its rules; it was solved again with {n} restricted area(s) opened where existing power towers already stand: {items}.",
    "DIVERSITY": "Route {route} first came out within {overlap_pct:.0f}% of route {other}'s corridor; it was re-solved with that corridor priced up to give a distinct alternative.",
}


def render(ev: dict) -> str:
    if ev["type"] == "BEELINE_BLOCKER":
        n = ev["data"].get("stretches", 1)
        ev["data"]["stretches_note"] = f" in {n} stretches" if n > 1 else ""
    t = TEMPLATES.get(ev["type"])
    if t is None:
        return ev["type"]
    try:
        return t.format(**ev["data"])
    except (KeyError, ValueError, IndexError):
        return f"{ev['type']}: {ev['data']}"


def event(type_: str, data: dict, chainage_km: float | None = None, source: dict | None = None,
          severity: str = "info") -> dict:
    d = dict(data)
    if chainage_km is not None:
        d.setdefault("chainage_km", chainage_km)
    ev = {"type": type_, "chainage_km": chainage_km, "data": d, "source": source, "severity": severity}
    ev["text"] = render(ev)
    return ev
