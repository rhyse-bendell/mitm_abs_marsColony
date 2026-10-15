"""Knowledge-grounded support reasoning (generic engine; the task package supplies the knowledge gates).

Pure and read-only. Given what a Pilot HOLDS (DIK element ids), the observable construction state and the legal
action options, it states which support requirements the Pilot can recognise as unmet and which already-legal actions
would address each one. It does not choose an action, create a project, attach a link, or read knowledge the Pilot
has not acquired: a requirement (or the means to meet it) that the Pilot does not hold simply does not appear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_CACHE: Dict[str, Optional[Dict[str, Any]]] = {}


def load_support_reasoning(task_id: str = "mars_colony") -> Optional[Dict[str, Any]]:
    key = str(task_id)
    if key not in _CACHE:
        path = Path(__file__).resolve().parents[1] / "config" / "tasks" / key / "support_reasoning.json"
        try:
            _CACHE[key] = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            _CACHE[key] = None
    return _CACHE[key]


def plan_relevant_dik_ids(config: Optional[Dict[str, Any]]) -> set:
    """Every DIK element id the task package marks as relevant to support planning (what a plan artifact may carry)."""
    ids: set = set()
    if not config:
        return ids
    for by_kind in (config.get("requirement_knowledge") or {}).values():
        for values in by_kind.values():
            ids.update(values)
    for section in ("provision_knowledge", "provider_capacity_knowledge"):
        for values in (config.get(section) or {}).values():
            ids.update(values)
    return ids


def _held(ids: Iterable[str], held: set) -> List[str]:
    return sorted(i for i in ids if i in held)


def derive_support_subgoals(
    *,
    config: Optional[Dict[str, Any]],
    support_model: Optional[Dict[str, Any]],
    held_ids: set,
    projects: Dict[str, Any],
    links: List[Dict[str, Any]],
    connection_options: List[Dict[str, Any]],
    start_structure_types: Iterable[str],
) -> List[Dict[str, Any]]:
    if not config or not support_model:
        return []
    kind_of_type = {t: kind for kind, types in support_model["structure_types"].items() for t in types}
    start_types = {str(t) for t in start_structure_types}
    subgoals: List[Dict[str, Any]] = []
    for support, rule in support_model["link_endpoints"].items():
        means_basis = _held((config.get("provision_knowledge") or {}).get(support, []), held_ids)
        capacity_known = bool(_held((config.get("provider_capacity_knowledge") or {}).get(support, []), held_ids))
        max_links = (support_model.get(support) or {}).get("max_supported_structures_per_generator")
        provider_types = [t for t, k in kind_of_type.items() if k == rule["provider"]]
        connector_type = support_model["connector_project_types"][support]
        provider_ids = [pid for pid, p in projects.items() if isinstance(p, dict) and str(p.get("type") or "").lower() in provider_types]

        def provider_links(pid):
            return sum(1 for l in links if l.get("connection_type") == support and pid in (l.get("from_project_id"), l.get("to_project_id")))

        with_slack = [pid for pid in provider_ids if not (capacity_known and max_links and provider_links(pid) >= int(max_links))]
        for pid, project in sorted(projects.items()):
            if not isinstance(project, dict):
                continue
            kind = kind_of_type.get(str(project.get("type") or "").lower())
            if kind not in rule["consumers"]:
                continue
            requirement_basis = _held(((config.get("requirement_knowledge") or {}).get(kind) or {}).get(support, []), held_ids)
            if not requirement_basis:
                continue  # this Pilot does not hold the requirement: no deficit is recognised
            if any(l.get("connection_type") == support and pid in (l.get("from_project_id"), l.get("to_project_id")) for l in links):
                continue
            options: List[Dict[str, Any]] = []
            if means_basis:
                for opt in connection_options:
                    if opt["connection_type"] == support and pid in (opt["from_project_id"], opt["to_project_id"]):
                        other = opt["to_project_id"] if opt["from_project_id"] == pid else opt["from_project_id"]
                        if other in with_slack:
                            options.append({"kind": "connect", "connector_project_id": opt["connector_project_id"], "endpoint_a_id": opt["from_project_id"], "endpoint_b_id": opt["to_project_id"]})
                connectors = [c for c, p in projects.items() if isinstance(p, dict) and str(p.get("type") or "").lower() == connector_type]
                if not connectors:
                    if connector_type in start_types:
                        options.append({"kind": "start", "structure_type": connector_type})
                else:
                    options.extend({"kind": "taskwork", "project_id": c} for c in connectors if str(projects[c].get("status")) != "complete" and not projects[c].get("validated_complete"))
                if not with_slack:
                    options.extend({"kind": "start", "structure_type": t} for t in provider_types if t in start_types)
                else:
                    options.extend({"kind": "taskwork", "project_id": p} for p in with_slack if not projects[p].get("structurally_complete") and str(projects[p].get("status")) != "complete")
            subgoals.append(
                {
                    "subgoal_id": f"{pid}:{support}",
                    "consumer_id": pid,
                    "consumer_type": project.get("type"),
                    "support": support,
                    "status": "unmet",
                    "requirement_basis": requirement_basis,
                    "means_known": bool(means_basis),
                    "means_basis": means_basis,
                    "options": options,
                }
            )
    return subgoals


def annotate_affordances(affordances: List[Dict[str, Any]], subgoals: List[Dict[str, Any]]) -> None:
    """Mark legal affordances that address a recognised subgoal. Information only: no utility is changed here."""
    for aff in affordances:
        ids: List[str] = []
        action = aff.get("action_type")
        for sg in subgoals:
            for opt in sg["options"]:
                if opt["kind"] == "connect" and action == "connect_structures":
                    args = aff.get("action_args") or {}
                    if (args.get("connector_project_id"), args.get("endpoint_a_id"), args.get("endpoint_b_id")) == (opt["connector_project_id"], opt["endpoint_a_id"], opt["endpoint_b_id"]):
                        ids.append(sg["subgoal_id"])
                elif opt["kind"] == "start" and action == "start_construction" and aff.get("start_mode") == "new_project":
                    if (aff.get("action_args") or {}).get("structure_type") == opt["structure_type"]:
                        ids.append(sg["subgoal_id"])
                elif opt["kind"] == "taskwork" and action in {"transport_resources", "start_construction", "continue_construction", "validate_construction", "repair_or_correct_construction"}:
                    if aff.get("start_mode") != "new_project" and aff.get("target_id") == opt["project_id"]:
                        ids.append(sg["subgoal_id"])
        if ids:
            aff["addresses_support_subgoals"] = sorted(set(ids))