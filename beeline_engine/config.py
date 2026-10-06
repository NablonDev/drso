"""Load the YAML rule / cost / engine / layer configs and resolve them for one line spec.

A run stores only its *overrides* (changed parameter values, switched-off rules, changed
cost items). `resolve_rules` / `resolve_costs` merge those onto the defaults and pick the
per-voltage value, so the engine only ever sees plain numbers and strings.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(os.environ.get("BEELINE_CONFIG_DIR", Path(__file__).resolve().parent.parent / "config"))

VOLTAGES = (765, 400, 220, 132)
CIRCUITS = ("sc", "dc", "qc")
TOWER_TYPES = ("A", "B1", "B2", "C1", "C2", "D45", "D60")


def _load(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_layers() -> dict:
    return _load("layers.yaml")


def load_rules() -> dict:
    return _load("rules.yaml")


def load_costs() -> dict:
    return _load("costs.yaml")


def load_engine() -> dict:
    return _load("engine.yaml")


def pick(value: Any, kv: int, circuit: str) -> Any:
    """Resolve a scalar-or-per-voltage value for (kv, circuit)."""
    if not isinstance(value, dict):
        return value
    for k in (f"{kv}_{circuit}", str(kv), kv, "default"):
        if k in value:
            return value[k]
    raise KeyError(f"no value for {kv}_{circuit} in {value}")


@dataclass
class LineSpec:
    kv: int
    circuit: str  # sc | dc | qc

    def __post_init__(self):
        if self.kv not in VOLTAGES:
            raise ValueError(f"voltage {self.kv} not supported; expected one of {VOLTAGES}")
        if self.circuit not in CIRCUITS:
            raise ValueError(f"circuit {self.circuit} not supported; expected one of {CIRCUITS}")


@dataclass
class ResolvedRule:
    id: str
    group: str
    name: str
    mechanism: str
    layer: str | None
    classes: list[str] | None
    params: dict[str, Any]
    enabled: bool = True
    statutory: bool = False
    crossing_cost: str | None = None
    cost_class: str | None = None
    also_lulc_classes: list[str] | None = None
    also_layers: list[str] | None = None        # landclass: further layers read whole
    overridden: list[str] = field(default_factory=list)


def rule_catalogue(overrides: dict | None = None) -> list[dict]:
    """Rules as shown on the Rules screen: defaults + the run's overrides (unresolved per-voltage)."""
    rules = load_rules()
    overrides = overrides or {}
    out = []
    for g in rules["groups"]:
        for r in g["rules"]:
            ov = overrides.get(r["id"], {})
            item = copy.deepcopy(r)
            item["group"] = g["key"]
            item["enabled"] = ov.get("enabled", True)
            item["reason"] = ov.get("reason")
            for p in item["params"]:
                p["default"] = p["value"]
                if p["key"] in ov.get("params", {}):
                    p["value"] = ov["params"][p["key"]]
            out.append(item)
    return out


def resolve_rules(spec: LineSpec, overrides: dict | None = None) -> dict[str, ResolvedRule]:
    """overrides: {rule_id: {"enabled": bool, "params": {key: value}, "reason": str}}
    A param override is a plain value for this run's voltage (the UI shows the resolved value)."""
    rules = load_rules()
    overrides = overrides or {}
    out: dict[str, ResolvedRule] = {}
    for g in rules["groups"]:
        for r in g["rules"]:
            ov = overrides.get(r["id"], {})
            params, changed = {}, []
            for p in r["params"]:
                v = pick(p["value"], spec.kv, spec.circuit)
                if p["key"] in ov.get("params", {}):
                    nv = ov["params"][p["key"]]
                    if nv != v:
                        changed.append(p["key"])
                    v = nv
                params[p["key"]] = v
            out[r["id"]] = ResolvedRule(
                id=r["id"], group=g["key"], name=r["name"], mechanism=r["mechanism"],
                layer=r.get("layer"), classes=r.get("classes"), params=params,
                enabled=ov.get("enabled", True), statutory=bool(g.get("statutory")),
                crossing_cost=r.get("crossing_cost"), cost_class=r.get("cost_class"),
                also_lulc_classes=r.get("also_lulc_classes"), also_layers=r.get("also_layers"),
                overridden=changed,
            )
    return out


def resolve_costs(spec: LineSpec, overrides: dict | None = None) -> dict:
    """Flat {category: {item: value}} plus the tower tonnage table for this line.
    overrides: {category: {item: value}} — plain values for this run's voltage."""
    costs = load_costs()
    overrides = overrides or {}
    flat: dict[str, dict] = {}
    for c in costs["categories"]:
        items = {}
        for it in c["items"]:
            items[it["key"]] = pick(it["value"], spec.kv, spec.circuit)
        items.update(overrides.get(c["key"], {}))
        flat[c["key"]] = items
    tower = next(c for c in costs["categories"] if c["key"] == "tower")
    tonnage = tower["tonnage"]
    factor = tonnage["circuit_factor"][spec.circuit]
    base = tonnage["by_voltage"][str(spec.kv)]
    t_over = overrides.get("tower_tonnage", {})
    flat["tonnage"] = {t: float(t_over.get(t, base[t] * factor)) for t in TOWER_TYPES}
    flat["insulator_sets_per_circuit"] = tower["insulator_sets_per_circuit"]
    flat["base_area_m2"] = pick(tower["base_area_m2"], spec.kv, spec.circuit)
    flat["n_circuits"] = {"sc": 1, "dc": 2, "qc": 4}[spec.circuit]
    return flat


def cost_catalogue(overrides: dict | None = None) -> list[dict]:
    costs = load_costs()
    overrides = overrides or {}
    out = []
    for c in costs["categories"]:
        item = copy.deepcopy(c)
        for it in item["items"]:
            it["default"] = it["value"]
            if it["key"] in overrides.get(c["key"], {}):
                it["value"] = overrides[c["key"]][it["key"]]
        out.append(item)
    return out


def tower_type_for(deflection_deg: float, bands: dict[str, float]) -> str | None:
    for t in TOWER_TYPES:
        if deflection_deg <= bands[t] + 1e-9:
            return t
    return None


def forced_min_type(code: str | None) -> str:
    """Rule tower-type option -> lowest acceptable tower type."""
    if not code:
        return "A"
    first = code.split("/")[0].strip().upper()
    return {"A": "A", "B": "B1", "C": "C1", "D": "D45"}.get(first, "A")


def max_type(a: str, b: str) -> str:
    return a if TOWER_TYPES.index(a) >= TOWER_TYPES.index(b) else b
