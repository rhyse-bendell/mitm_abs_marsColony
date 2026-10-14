"""Authoritative structural support accounting.

Answers "what support infrastructure does this colony actually possess?" from the construction state
(validated structures and the connector graph), using atoms from the task package
(config/tasks/<task_id>/support_model.json). It is a pure, read-only analysis layer:

- it never mutates construction state and holds no Pilot beliefs;
- it separates NOMINAL capacity (validated structures) from EFFECTIVELY SUPPORTED capacity
  (structures actually served food and water through valid direct connections);
- it does NOT decide which civilians or VIPs receive scarce capacity. Allocation and mission scoring are
  unresolved task questions and are deliberately out of scope. Capacities are reported per colonist kind
  so that a later explicit allocation policy can consume them.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA = "structural_support_accounting/1"


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

    # --- the connector graph: only validated connector projects, only link entries with both endpoints ---
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

    # --- water: each validated generator serves at most N structures, output split evenly among those served ---
    water_delivered: Dict[str, float] = {}
    water_linked: Dict[str, List[str]] = {}
    generators = []
    effective_connections = 0
    unsupported_connections = 0
    overconnected = []
    for generator in structures["water_generator"]:
        connected = [l["consumer"] for l in links if l["support"] == "water" and l["provider"] == generator]
        served, unserved = connected[:max_served], connected[max_served:]
        share = output_per_generator / len(served) if served else 0.0
        for consumer in connected:
            water_linked.setdefault(consumer, []).append(generator)
        for consumer in served:
            water_delivered[consumer] = water_delivered.get(consumer, 0.0) + share
        effective_connections += len(served)
        unsupported_connections += len(unserved)
        record = {
            "generator": generator,
            "nominal_output": output_per_generator,
            "connected_structures": len(connected),
            "served": list(served),
            "unserved": list(unserved),
            "water_units_per_served_structure": share,
            "overconnected": len(connected) > max_served,
        }
        generators.append(record)
        if record["overconnected"]:
            overconnected.append(generator)
            violations.append({"type": "generator_overconnected", "generator": generator, "connected": len(connected), "limit": max_served, "unserved": list(unserved)})

    # --- food: greenhouses feed housing; a greenhouse only counts as food if it is itself watered ---
    greenhouse_rows = []
    watered = {gh for gh in structures["greenhouse"] if water_delivered.get(gh, 0.0) > 0.0}
    for gh in structures["greenhouse"]:
        feeds = [l["consumer"] for l in links if l["support"] == "food" and l["provider"] == gh]
        water_state = "served" if gh in watered else ("linked_but_unserved_generator_overconnected" if gh in water_linked else "no_water_link")
        greenhouse_rows.append({"greenhouse": gh, "watered": gh in watered, "water": water_state, "water_units_delivered": water_delivered.get(gh, 0.0), "housing_connected": feeds})
        if gh not in watered:
            violations.append({"type": "greenhouse_unwatered", "greenhouse": gh, "water": water_state})
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
        elif any(p in watered for p in providers):
            food_state = "effective"
        else:
            food_state = "food_source_unwatered"
        if house in water_delivered:
            water_state = "served"
        elif house in water_linked:
            water_state = "linked_but_unserved_generator_overconnected"
        else:
            water_state = "no_water_link"
        missing = []
        if food_state != "effective":
            missing.append(food_state)
        if water_state != "served":
            missing.append(water_state)
        housing_rows.append(
            {
                "housing": house,
                "food": food_state,
                "water": water_state,
                "water_units_delivered": water_delivered.get(house, 0.0),
                "fully_supported": not missing,
                "missing": missing,
            }
        )
    fully = [r for r in housing_rows if r["fully_supported"]]

    reasons = Counter(m for r in housing_rows for m in r["missing"])
    if reasons:
        limiting, count = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        bottleneck = {"limiting_factor": limiting, "housing_affected": count, "of_validated_housing": len(housing_rows), "all_reasons": dict(sorted(reasons.items()))}
    else:
        bottleneck = {"limiting_factor": None, "housing_affected": 0, "of_validated_housing": len(housing_rows), "all_reasons": {}}

    soil_units = len(structures["greenhouse"]) * int(greenhouse_cfg["soil_units_per_structure"])
    soil_civ = int(greenhouse_cfg["capacity_per_soil_unit"]["civilians"])
    soil_vip = int(greenhouse_cfg["capacity_per_soil_unit"]["vips"])
    effective_soil = len(watered) * int(greenhouse_cfg["soil_units_per_structure"])

    return {
        "schema": SCHEMA,
        "housing": {
            "validated_structures": len(housing_rows),
            "civilian_capacity": len(housing_rows) * units * civ_per_unit,
            "vip_capacity_if_vip_compatible": len(housing_rows) * units * vip_per_unit,
            "vip_property_modeled": bool(housing_cfg.get("vip_property_modeled", False)),
            "with_food_link": sum(1 for r in housing_rows if r["food"] != "no_food_link"),
            "with_effective_food": sum(1 for r in housing_rows if r["food"] == "effective"),
            "with_water_connection": sum(1 for r in housing_rows if r["water"] != "no_water_link"),
            "with_effective_water": sum(1 for r in housing_rows if r["water"] == "served"),
            "fully_supported_structures": len(fully),
            "fully_supported_civilian_capacity": len(fully) * units * civ_per_unit,
            "fully_supported_vip_capacity_if_vip_compatible": len(fully) * units * vip_per_unit,
            "unsupported_structures": [{"housing": r["housing"], "missing": r["missing"]} for r in housing_rows if not r["fully_supported"]],
            "structures": housing_rows,
        },
        "food": {
            "validated_greenhouses": len(greenhouse_rows),
            "soil_units": soil_units,
            "civilian_equivalent_capacity": soil_units * soil_civ,
            "vip_equivalent_capacity": soil_units * soil_vip,
            "watered_greenhouses": len(watered),
            "unwatered_greenhouses": len(greenhouse_rows) - len(watered),
            "effective_soil_units": effective_soil,
            "effective_civilian_equivalent_capacity": effective_soil * soil_civ,
            "effective_vip_equivalent_capacity": effective_soil * soil_vip,
            "connected_housing": len(food_providers),
            "greenhouses": greenhouse_rows,
        },
        "water": {
            "validated_generators": len(generators),
            "total_nominal_output": output_per_generator * len(generators),
            "total_delivered": sum(water_delivered.values()),
            "effective_connections": effective_connections,
            "unsupported_connections": unsupported_connections,
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
            "vip_capacity_if_vip_compatible assumes the VIP housing property, which construction state does not model.",
            "Validated structures only; connector projects must themselves be validated and only endpoint-bearing link entries are edges.",
        ],
    }
