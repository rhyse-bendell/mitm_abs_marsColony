"""Authoritative structural support accounting.

Answers "what support infrastructure does this colony actually possess?" from the construction state
(validated structures and the connector graph), using atoms from the task package
(config/tasks/<task_id>/support_model.json). It is a pure, read-only analysis layer:

- it never mutates construction state and holds no Pilot beliefs;
- it separates PHYSICAL CONNECTION STATE (what the team built, including invalid designs such as a generator
  wired to many structures) from EFFECTIVE RESOURCE SERVICE (what the task rules allow those connections to
  provide), and NOMINAL capacity (validated structures) from effectively supported capacity;
- construction validity and mission support are separate: a validated house can lack food or water and a
  validated greenhouse can lack water; this module reports that, it never invalidates anything;
- it does NOT decide which civilians or VIPs receive scarce capacity, and it invents NO allocation winner:
  when a generator is connected to more structures than it can serve, which endpoints are served is
  UNDETERMINED by the task sources, so the report gives the slot count and serviceable ranges instead of
  choosing by connection order. Allocation and mission scoring remain unresolved task questions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "structural_support_accounting/2"
UNDETERMINED_WATER = "service_underdetermined_generator_overconnected"


def load_support_model(task_id: str = "mars_colony") -> Dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "config" / "tasks" / str(task_id) / "support_model.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _is_validated(project: Dict[str, Any]) -> bool:
    return (
        str(project.get("status") or "").lower() == "complete"
        and project.get("validated_complete") is True
        and project.get("correct", True) is not False
    )


def _range(low: int, high: int) -> Dict[str, int]:
    return {"min": low, "max": high}


def evaluate_structural_support(construction, model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the structural support report for ``construction`` (anything exposing ``.projects``)."""
    model = model or load_support_model()
    projects: Dict[str, Dict[str, Any]] = {
        str(pid): p for pid, p in (getattr(construction, "projects", {}) or {}).items() if isinstance(p, dict)
    }
    kind_of_type = {t: kind for kind, types in model["structure_types"].items() for t in types}
    connector_kind = {ctype: support for support, ctype in model["connector_project_types"].items()}
    water_cfg = model["water"]
    housing_cfg = model["housing"]
    greenhouse_cfg = model["greenhouse"]
    output_per_generator = float(water_cfg["output_per_generator"])
    max_served = int(water_cfg["max_supported_structures_per_generator"])

    def kind(pid: str) -> Optional[str]:
        project = projects.get(pid)
        return kind_of_type.get(str(project.get("type") or "").strip().lower()) if project else None

    structures: Dict[str, List[str]] = {"housing": [], "greenhouse": [], "water_generator": []}
    unvalidated: Dict[str, List[str]] = {"housing": [], "greenhouse": [], "water_generator": [], "connector": []}
    for pid in sorted(projects):
        project = projects[pid]
        structure_kind = kind_of_type.get(str(project.get("type") or "").strip().lower())
        if structure_kind:
            (structures if _is_validated(project) else unvalidated)[structure_kind].append(pid)
        elif str(project.get("type") or "").strip().lower() in connector_kind and not _is_validated(project):
            unvalidated["connector"].append(pid)
    valid = {pid for ids in structures.values() for pid in ids}

    violations: List[Dict[str, Any]] = []

    # --- the physical connector graph: validated connector projects, link entries with both endpoints ---
    links: List[Dict[str, Any]] = []
    seen = set()
    for pid in sorted(projects):
        connector = projects[pid]
        support = connector_kind.get(str(connector.get("type") or "").strip().lower())
        if not support:
            continue
        entries = [c for c in (connector.get("connections") or []) if isinstance(c, dict) and c.get("from_project_id") and c.get("to_project_id")]
        if entries and not _is_validated(connector):
            violations.append({"type": "connector_not_validated", "connector": pid, "links_ignored": len(entries)})
            continue
        endpoints = model["link_endpoints"][support]
        for index, entry in enumerate(entries):
            a, b = str(entry["from_project_id"]), str(entry["to_project_id"])
            ref = {"connector": pid, "link_index": index, "support": support, "from": a, "to": b}
            provider = a if kind(a) == endpoints["provider"] else (b if kind(b) == endpoints["provider"] else None)
            consumer = b if provider == a else (a if provider == b else None)
            if provider is None or kind(consumer) not in endpoints["consumers"]:
                violations.append({"type": "invalid_link_endpoints", **ref})
                continue
            if provider not in valid or consumer not in valid:
                violations.append({"type": "link_to_unvalidated_structure", **ref})
                continue
            key = (support, provider, consumer)
            if key in seen:
                violations.append({"type": "duplicate_link", **ref})
                continue
            seen.add(key)
            links.append({"support": support, "provider": provider, "consumer": consumer, "connector": pid})

    # --- water: a generator can serve at most N structures, output split evenly among those served ---
    # Physical connections are never capped. If a generator is connected to more than N structures, WHICH N
    # are served is not specified by the sources, so no endpoint is chosen: endpoints are only "underdetermined".
    certain_water: Dict[str, float] = {}
    possible_water: Dict[str, float] = {}
    water_linked = set()
    generators = []
    overconnected: List[str] = []
    overconnected_ranges = []
    for generator in structures["water_generator"]:
        connected = sorted({l["consumer"] for l in links if l["support"] == "water" and l["provider"] == generator})
        count = len(connected)
        slots = min(count, max_served)
        determined = count <= max_served
        share = output_per_generator / slots if slots else 0.0
        housing_ids = [c for c in connected if kind(c) == "housing"]
        greenhouse_ids = [c for c in connected if kind(c) == "greenhouse"]
        if determined:
            h_min, h_max, g_min, g_max = len(housing_ids), len(housing_ids), len(greenhouse_ids), len(greenhouse_ids)
        else:
            h_max, g_max = min(len(housing_ids), slots), min(len(greenhouse_ids), slots)
            h_min, g_min = max(0, slots - len(greenhouse_ids)), max(0, slots - len(housing_ids))
        for consumer in connected:
            water_linked.add(consumer)
            target = certain_water if determined else possible_water
            target[consumer] = target.get(consumer, 0.0) + share
        record = {
            "generator": generator,
            "nominal_output": output_per_generator,
            "connected_endpoints": connected,
            "connection_count": count,
            "max_serviceable_structures": max_served,
            "overconnected": not determined,
            "service_slots": slots,
            "total_water_units": output_per_generator if slots else 0.0,
            "water_units_per_served_structure": share,
            "excess_connections": count - slots,
            "allocation_determined": determined,
            "housing_connected": len(housing_ids),
            "greenhouses_connected": len(greenhouse_ids),
            "housing_serviceable": _range(h_min, h_max),
            "greenhouse_serviceable": _range(g_min, g_max),
        }
        generators.append(record)
        if not determined:
            overconnected.append(generator)
            overconnected_ranges.append((h_min, h_max, g_min, g_max))
            violations.append({"type": "generator_overconnected", "generator": generator, "connected": count, "limit": max_served, "connected_endpoints": connected, "allocation_determined": False})

    def water_state(structure_id: str) -> str:
        if structure_id in certain_water:
            return "served"
        if structure_id in possible_water:
            return UNDETERMINED_WATER
        return "no_water_link"

    # --- food: greenhouses feed housing; a greenhouse only provides food if it is itself watered ---
    greenhouse_rows = []
    gh_served = {gh for gh in structures["greenhouse"] if water_state(gh) == "served"}
    gh_possible = {gh for gh in structures["greenhouse"] if water_state(gh) == UNDETERMINED_WATER}
    for gh in structures["greenhouse"]:
        feeds = sorted({l["consumer"] for l in links if l["support"] == "food" and l["provider"] == gh})
        state = water_state(gh)
        greenhouse_rows.append({"greenhouse": gh, "water": state, "water_units": _range(certain_water.get(gh, 0.0), certain_water.get(gh, 0.0) + possible_water.get(gh, 0.0)), "housing_connected": feeds})
        if state == "no_water_link":
            violations.append({"type": "greenhouse_unwatered", "greenhouse": gh})
    food_providers: Dict[str, List[str]] = {}
    for l in links:
        if l["support"] == "food":
            food_providers.setdefault(l["consumer"], []).append(l["provider"])

    # --- housing ---
    civ_per_unit = int(housing_cfg["capacity_per_floorspace_unit"]["civilians"])
    vip_per_unit = int(housing_cfg["capacity_per_floorspace_unit"]["vips"])
    units = int(housing_cfg["floorspace_units_per_structure"])
    housing_rows = []
    for house in structures["housing"]:
        providers = food_providers.get(house, [])
        if not providers:
            food_state = "no_food_link"
        elif any(p in gh_served for p in providers):
            food_state = "effective"
        elif any(p in gh_possible for p in providers):
            food_state = "service_underdetermined"
        else:
            food_state = "food_source_unwatered"
        w_state = water_state(house)
        if w_state == "served" and food_state == "effective":
            support_state = "full"
        elif w_state == "no_water_link" or food_state in ("no_food_link", "food_source_unwatered"):
            support_state = "not_full"
        else:
            support_state = "underdetermined"
        missing = []
        if food_state != "effective":
            missing.append(food_state if food_state != "service_underdetermined" else "food_service_underdetermined")
        if w_state != "served":
            missing.append("water_service_underdetermined_generator_overconnected" if w_state == UNDETERMINED_WATER else w_state)
        housing_rows.append(
            {
                "housing": house,
                "food": food_state,
                "water": w_state,
                "water_units": _range(certain_water.get(house, 0.0), certain_water.get(house, 0.0) + possible_water.get(house, 0.0)),
                "support": support_state,
                "missing": missing,
            }
        )

    n_housing = len(housing_rows)
    h_water_certain = sum(1 for r in housing_rows if r["water"] == "served")
    h_water_possible = sum(1 for r in housing_rows if r["water"] == UNDETERMINED_WATER)
    h_water_lo = h_water_certain + min(max([r[0] for r in overconnected_ranges] or [0]), h_water_possible)
    h_water_hi = h_water_certain + min(sum(r[1] for r in overconnected_ranges), h_water_possible)
    h_food_certain = sum(1 for r in housing_rows if r["food"] == "effective")
    h_food_possible = sum(1 for r in housing_rows if r["food"] == "service_underdetermined")
    full = sum(1 for r in housing_rows if r["support"] == "full")
    undetermined = sum(1 for r in housing_rows if r["support"] == "underdetermined")
    full_lo, full_hi = full, min(full + undetermined, h_water_hi)

    reasons = Counter(m for r in housing_rows for m in r["missing"])
    if reasons:
        limiting, count = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        bottleneck = {"limiting_factor": limiting, "housing_affected": count, "of_validated_housing": n_housing, "all_reasons": dict(sorted(reasons.items()))}
    else:
        bottleneck = {"limiting_factor": None, "housing_affected": 0, "of_validated_housing": n_housing, "all_reasons": {}}
    bottleneck["water_excess_connections"] = sum(g["excess_connections"] for g in generators)

    soil_per = int(greenhouse_cfg["soil_units_per_structure"])
    soil_civ = int(greenhouse_cfg["capacity_per_soil_unit"]["civilians"])
    soil_vip = int(greenhouse_cfg["capacity_per_soil_unit"]["vips"])
    gh_possible_n = len(gh_possible)
    gh_lo = len(gh_served) + min(max([r[2] for r in overconnected_ranges] or [0]), gh_possible_n)
    gh_hi = len(gh_served) + min(sum(r[3] for r in overconnected_ranges), gh_possible_n)

    return {
        "schema": SCHEMA,
        "allocation_determined": not overconnected,
        "housing": {
            "validated_structures": n_housing,
            "civilian_capacity": n_housing * units * civ_per_unit,
            "vip_capacity_if_vip_compatible": n_housing * units * vip_per_unit,
            "vip_property_modeled": bool(housing_cfg.get("vip_property_modeled", False)),
            "with_food_link": sum(1 for r in housing_rows if r["food"] != "no_food_link"),
            "with_effective_food": _range(h_food_certain, h_food_certain + h_food_possible),
            "with_water_connection": sum(1 for r in housing_rows if r["water"] != "no_water_link"),
            "with_effective_water": _range(h_water_lo, h_water_hi),
            "fully_supported_structures": _range(full_lo, full_hi),
            "fully_supported_civilian_capacity": _range(full_lo * units * civ_per_unit, full_hi * units * civ_per_unit),
            "fully_supported_vip_capacity_if_vip_compatible": _range(full_lo * units * vip_per_unit, full_hi * units * vip_per_unit),
            "unsupported_structures": [{"housing": r["housing"], "support": r["support"], "missing": r["missing"]} for r in housing_rows if r["support"] != "full"],
            "structures": housing_rows,
        },
        "food": {
            "validated_greenhouses": len(greenhouse_rows),
            "soil_units": len(greenhouse_rows) * soil_per,
            "civilian_equivalent_capacity": len(greenhouse_rows) * soil_per * soil_civ,
            "vip_equivalent_capacity": len(greenhouse_rows) * soil_per * soil_vip,
            "watered_greenhouses": _range(gh_lo, gh_hi),
            "unwatered_greenhouses": sum(1 for r in greenhouse_rows if r["water"] == "no_water_link"),
            "effective_soil_units": _range(gh_lo * soil_per, gh_hi * soil_per),
            "effective_civilian_equivalent_capacity": _range(gh_lo * soil_per * soil_civ, gh_hi * soil_per * soil_civ),
            "effective_vip_equivalent_capacity": _range(gh_lo * soil_per * soil_vip, gh_hi * soil_per * soil_vip),
            "connected_housing": len(food_providers),
            "greenhouses": greenhouse_rows,
        },
        "water": {
            "validated_generators": len(generators),
            "total_nominal_output": output_per_generator * len(generators),
            "total_water_units_deliverable": sum(g["total_water_units"] for g in generators),
            "physical_connections": sum(g["connection_count"] for g in generators),
            "effective_connections": sum(g["service_slots"] for g in generators),
            "unsupported_connections": sum(g["excess_connections"] for g in generators),
            "overconnected_generators": overconnected,
            "connected_housing": sum(1 for h in structures["housing"] if h in water_linked),
            "connected_greenhouses": sum(1 for g in structures["greenhouse"] if g in water_linked),
            "generators": generators,
        },
        "unvalidated_structures": {k: list(v) for k, v in unvalidated.items()},
        "bottleneck": bottleneck,
        "violations": violations,
        "notes": [
            "Infrastructure only: no civilian/VIP allocation or scoring is implied (unresolved task question).",
            "Capacities are per colonist kind (civilians OR VIPs per unit); they are alternatives, not additive.",
            "Physical connections are never capped. For an overconnected generator the sources do not say which endpoints are served, so none is chosen: *_min/*_max ranges bound the effective values and structure states are 'underdetermined'.",
            "Construction validity and mission support are separate: nothing here invalidates a structure.",
            "vip_capacity_if_vip_compatible assumes the VIP housing property, which construction state does not model; one floorspace unit per house and one soil unit per greenhouse are simulator abstractions.",
        ],
    }
