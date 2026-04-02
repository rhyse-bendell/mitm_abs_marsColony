# File: modules/construction.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from modules.task_model import normalize_rule_token


@dataclass
class ConstructionSite:
    site_id: str
    label: str
    position: Tuple[float, float]
    capacity: int
    buildable: bool
    started_structures: List[str] = field(default_factory=list)


@dataclass
class ResourcePile:
    pile_id: str
    site_id: str
    position: Tuple[float, float]
    quantity: int
    max_quantity: int


@dataclass
class BridgeState:
    bridge_id: str
    start_site_id: str
    end_site_id: str
    status: str
    delivered_resources: int
    required_resources: int


class ConstructionManager:
    DEFAULT_PARAMETERS = {
        "pile_a_quantity": 100,
        "pile_c_quantity": 100,
        "housing_cost": 10,
        "greenhouse_cost": 10,
        "water_generator_cost": 10,
        "bridge_bc_cost": 20,
        "site_a_capacity": 4,
        "site_b_capacity": 8,
        "site_c_capacity": 16,
        "move_time_per_unit": 4,
        "carry_capacity": 1,
    }
    STRUCTURE_STYLE_MAP = {
        "house": {"symbol": "square", "color": "#c6362f"},
        "housing": {"symbol": "square", "color": "#c6362f"},
        "greenhouse": {"symbol": "square", "color": "#2f8f46"},
        "water_generator": {"symbol": "square", "color": "#2f6fbf"},
    }
    PROJECT_TO_SITE = {
        "Build_Site_A": "site_a",
        "Build_Site_B": "site_b",
        "Build_Site_C": "site_c",
        "Build_Table_A": "site_a",
        "Build_Table_B": "site_b",
        "Build_Table_C": "site_c",
    }
    SITE_TO_BUILD_TARGET = {
        "site_a": "Build_Site_A",
        "site_b": "Build_Site_B",
        "site_c": "Build_Site_C",
    }
    DEFAULT_SITE_DEFINITIONS = {
        "site_a": {"label": "Site A", "position": (6.5, 3.4), "capacity_parameter": "site_a_capacity", "buildable": True},
        "site_b": {"label": "Site B", "position": (5.0, 4.4), "capacity_parameter": "site_b_capacity", "buildable": True},
        "site_c": {"label": "Site C", "position": (3.5, 3.4), "capacity_parameter": "site_c_capacity", "buildable": False, "buildable_when_bridge_complete": "bridge_bc"},
    }

    def __init__(self, task_model=None, parameters: Optional[Dict] = None):
        self.task_model = task_model
        self.parameters = dict(self.DEFAULT_PARAMETERS)
        if isinstance(parameters, dict):
            self.parameters.update(parameters)

        self.site_definitions = self._load_site_definitions()
        self.sites = self._build_sites()
        self.resource_nodes = {
            "pile_a": ResourcePile("pile_a", "site_a", (7.25, 3.7), int(self.parameters["pile_a_quantity"]), int(self.parameters["pile_a_quantity"])),
            "pile_c": ResourcePile("pile_c", "site_c", (2.75, 3.65), int(self.parameters["pile_c_quantity"]), int(self.parameters["pile_c_quantity"])),
        }
        self.site_resource_inventory = {site_id: 0 for site_id in self.sites}
        self.site_resource_inventory["site_a"] = self.resource_nodes["pile_a"].quantity
        self.site_resource_inventory["site_c"] = self.resource_nodes["pile_c"].quantity
        self.bridges = {
            "bridge_ab": BridgeState("bridge_ab", "site_a", "site_b", "complete", 0, 0),
            "bridge_bc": BridgeState("bridge_bc", "site_b", "site_c", "not_started", 0, max(1, int(self.parameters["bridge_bc_cost"]))),
        }
        self.connectors: List[Dict] = []
        self._active_transports: Dict[str, Dict] = {}
        self.project_templates = self._build_project_templates()
        self._project_counters: Dict[Tuple[str, str], int] = {}

        self.projects = {}
        self._seed_initial_projects()
        self.update()

    def _load_site_definitions(self):
        configured = self.parameters.get("site_definitions")
        if not isinstance(configured, dict):
            return dict(self.DEFAULT_SITE_DEFINITIONS)
        merged = dict(self.DEFAULT_SITE_DEFINITIONS)
        for site_id, row in configured.items():
            if not isinstance(row, dict):
                continue
            merged[site_id] = {**merged.get(site_id, {}), **row}
        return merged

    def _build_sites(self):
        sites = {}
        for site_id, conf in self.site_definitions.items():
            capacity_key = conf.get("capacity_parameter")
            fallback_capacity = conf.get("capacity", 1)
            raw_capacity = self.parameters.get(capacity_key, fallback_capacity)
            pos = tuple(conf.get("position", (0.0, 0.0)))
            sites[site_id] = ConstructionSite(
                site_id=site_id,
                label=str(conf.get("label", site_id.replace("_", " ").title())),
                position=(float(pos[0]), float(pos[1])),
                capacity=max(1, int(raw_capacity)),
                buildable=bool(conf.get("buildable", True)),
            )
        return sites

    def _build_project_templates(self):
        defaults = {
            "template_site_a_greenhouse": {
                "name": "Greenhouse at Site A",
                "type": "greenhouse",
                "artifact_type": "greenhouse_construction",
                "expected_rules": [normalize_rule_token("rule:greenhouse_requires_water")],
                "required": int(self.parameters["greenhouse_cost"]),
                "site_id": "site_a",
            },
            "template_site_b_house": {
                "name": "Housing at Site B",
                "type": "house",
                "artifact_type": "house_construction",
                "expected_rules": [normalize_rule_token("rule:house_enclosed")],
                "required": int(self.parameters["housing_cost"]),
                "site_id": "site_b",
            },
            "template_site_c_water_generator": {
                "name": "Water Generator at Site C",
                "type": "water_generator",
                "artifact_type": "water_generator_construction",
                "expected_rules": [normalize_rule_token("rule:water_generator_2x2")],
                "required": int(self.parameters["water_generator_cost"]),
                "site_id": "site_c",
            },
        }
        templates = {}
        if self.task_model and getattr(self.task_model, "construction_templates", None):
            for template in self.task_model.construction_templates.values():
                target_id = str(template.target_id or "").strip()
                site_id = self.PROJECT_TO_SITE.get(target_id)
                if site_id not in self.sites:
                    continue
                templates[template.project_id] = {
                    "name": template.name or f"{template.structure_type} @ {site_id}",
                    "type": template.structure_type,
                    "artifact_type": template.artifact_type,
                    "expected_rules": [normalize_rule_token(r) for r in template.expected_rules if normalize_rule_token(r)],
                    "required": int((template.required_resources or {}).get("bricks", 1) or 1),
                    "site_id": site_id,
                    "target_id": self.SITE_TO_BUILD_TARGET.get(site_id, target_id),
                }
        if templates:
            return templates
        for key, row in defaults.items():
            row = dict(row)
            row["target_id"] = self.SITE_TO_BUILD_TARGET.get(row["site_id"], "")
            templates[key] = row
        return templates

    def _site_templates(self, site_id):
        return [t for t in self.project_templates.values() if t.get("site_id") == site_id]

    def _default_template_for_site(self, site_id):
        templates = self._site_templates(site_id)
        return templates[0] if templates else None

    def _legacy_project_id_for_site(self, site_id):
        for legacy_id, mapped_site in self.PROJECT_TO_SITE.items():
            if mapped_site == site_id and legacy_id.startswith("Build_Table_"):
                return legacy_id
        return None

    def _seed_initial_projects(self):
        for site_id in self.sites:
            template = self._default_template_for_site(site_id)
            if template:
                legacy_id = self._legacy_project_id_for_site(site_id)
                self.create_project(
                    site_id=site_id,
                    structure_type=template["type"],
                    template=template,
                    author="system",
                    project_id_override=legacy_id,
                )

    def _site_project_ids(self, site_id):
        return [p.get("id") for p in self.projects.values() if p.get("site_id") == site_id]

    def _site_structure_position(self, site_id):
        site = self.sites[site_id]
        existing_count = len(self._site_project_ids(site_id))
        offset_patterns = [
            (0.0, 0.0),
            (-0.2, 0.17),
            (0.2, 0.17),
            (-0.2, -0.17),
            (0.2, -0.17),
            (0.0, 0.28),
            (-0.28, 0.0),
            (0.28, 0.0),
        ]
        dx, dy = offset_patterns[existing_count % len(offset_patterns)]
        return (site.position[0] + dx, site.position[1] + dy)

    def _next_project_id(self, site_id, structure_type):
        key = (site_id, str(structure_type))
        next_idx = int(self._project_counters.get(key, 0)) + 1
        self._project_counters[key] = next_idx
        return f"{site_id}_{structure_type}_{next_idx:03d}"

    def create_project(self, site_id, structure_type=None, *, template=None, target_id=None, author="system", project_id_override=None):
        site = self.sites.get(site_id)
        if not site:
            return None, "unknown_site"
        if len(self._site_project_ids(site_id)) >= site.capacity:
            return None, "site_capacity_reached"
        template = dict(template or self._default_template_for_site(site_id) or {})
        resolved_structure_type = str(structure_type or template.get("type") or "house").strip().lower()
        project_id = str(project_id_override or self._next_project_id(site_id, resolved_structure_type))
        if project_id in self.projects:
            return project_id, "exists"
        expected_rules = [normalize_rule_token(r) for r in template.get("expected_rules", []) if normalize_rule_token(r)]
        required = max(1, int(template.get("required", 1)))
        canonical_target_id = str(target_id or template.get("target_id") or self.SITE_TO_BUILD_TARGET.get(site_id, ""))
        project = {
            "id": project_id,
            "name": str(template.get("name") or f"{resolved_structure_type.replace('_', ' ').title()} at {self.sites[site_id].label}"),
            "type": resolved_structure_type,
            "location": self._site_structure_position(site_id),
            "site_id": site_id,
            "status": "not_started",
            "in_progress": False,
            "started": False,
            "correct": True,
            "required_resources": {"bricks": required},
            "delivered_resources": {"bricks": 0},
            "expected_rules": list(expected_rules),
            "resource_complete": False,
            "structurally_complete": False,
            "validated_complete": False,
            "builders": set(),
            "author": author,
            "artifact_type": str(template.get("artifact_type") or f"{resolved_structure_type}_construction"),
            "target_id": canonical_target_id,
            "last_actor": None,
            "last_event_time": None,
            "provenance": {
                "last_actor": None,
                "contributors": [],
                "last_update_time": None,
                "timeline": [],
                "expected_rules": list(expected_rules),
                "held_rule_ids_at_build": [],
                "held_information_ids_at_build": [],
                "held_data_ids_at_build": [],
                "held_expected_rules_locally": False,
                "missing_expected_rules": list(expected_rules),
                "team_rule_snapshot_ids": [],
            },
        }
        self.projects[project_id] = project
        return project_id, "created"

    def resolve_project_id(self, project_or_target_id, *, create_if_missing=False):
        requested = str(project_or_target_id or "").strip()
        if requested in self.projects:
            return requested
        site_id = self.PROJECT_TO_SITE.get(requested)
        if site_id is None:
            return None
        site_projects = [p for p in self.projects.values() if p.get("site_id") == site_id]
        unfinished = [p for p in site_projects if p.get("status") != "complete"]
        if unfinished:
            unfinished.sort(key=lambda p: (0 if p.get("started") else 1, p.get("id")))
            return unfinished[0].get("id")
        if not create_if_missing:
            return None
        template = self._default_template_for_site(site_id)
        legacy_id = requested if requested.startswith("Build_Table_") else None
        project_id, _ = self.create_project(site_id=site_id, template=template, author="system", project_id_override=legacy_id)
        return project_id

    def update_project_provenance(self, project_id, *, event, actor=None, sim_time=None, held_data_ids=None, held_information_ids=None, held_rule_ids=None, team_rule_snapshot_ids=None):
        project = self.projects.get(project_id)
        if not project:
            return
        prov = project.setdefault("provenance", {})
        expected = [
            normalize_rule_token(r)
            for r in (project.get("expected_rules") or prov.get("expected_rules") or [])
            if normalize_rule_token(r)
        ]
        held_rules = sorted({normalize_rule_token(r) for r in (held_rule_ids or prov.get("held_rule_ids_at_build") or []) if normalize_rule_token(r)})
        held_info = sorted({str(i) for i in (held_information_ids or prov.get("held_information_ids_at_build") or []) if str(i)})
        held_data = sorted({str(d) for d in (held_data_ids or prov.get("held_data_ids_at_build") or []) if str(d)})
        team_rules = sorted({normalize_rule_token(r) for r in (team_rule_snapshot_ids or prov.get("team_rule_snapshot_ids") or []) if normalize_rule_token(r)})
        expected_set = set(expected)
        held_expected_locally = bool(expected_set.issubset(set(held_rules))) if expected_set else True
        missing_expected = sorted(expected_set - set(held_rules))
        contributors = sorted(set(list(prov.get("contributors", [])) + ([actor] if actor else [])))
        prov.update(
            {
                "last_actor": actor or prov.get("last_actor"),
                "contributors": contributors,
                "last_update_time": sim_time,
                "expected_rules": expected,
                "held_rule_ids_at_build": held_rules,
                "held_information_ids_at_build": held_info,
                "held_data_ids_at_build": held_data,
                "held_expected_rules_locally": held_expected_locally,
                "missing_expected_rules": missing_expected,
                "team_rule_snapshot_ids": team_rules,
            }
        )
        timeline = list(prov.get("timeline", []))
        timeline.append(
            {
                "event": str(event),
                "actor": actor,
                "time": sim_time,
                "status": project.get("status"),
                "correct": bool(project.get("correct", True)),
                "delivered_resources": dict(project.get("delivered_resources", {})),
                "required_resources": dict(project.get("required_resources", {})),
                "held_expected_rules_locally": held_expected_locally,
                "missing_expected_rules": missing_expected,
            }
        )
        prov["timeline"] = timeline[-20:]
        project["last_actor"] = actor or project.get("last_actor")
        project["last_event_time"] = sim_time

    def _project_progress(self, project):
        req = float(project.get("required_resources", {}).get("bricks", 0) or 0)
        if req <= 0:
            return 0.0
        delivered = float(project.get("delivered_resources", {}).get("bricks", 0) or 0)
        return max(0.0, min(1.0, delivered / req))

    def _site_has_capacity(self, site_id):
        site = self.sites.get(site_id)
        if not site:
            return False
        return len(self._site_project_ids(site_id)) < site.capacity

    def _is_site_buildable(self, site_id):
        site_conf = self.site_definitions.get(site_id, {})
        required_bridge = site_conf.get("buildable_when_bridge_complete")
        if required_bridge:
            bridge = self.bridges.get(required_bridge)
            return bool(bridge and bridge.status == "complete")
        return bool(self.sites.get(site_id, ConstructionSite("", "", (0.0, 0.0), 1, False)).buildable)

    def _accessible_sites_from(self, site_id):
        adj = {
            "site_a": {"site_b"},
            "site_b": {"site_a"},
            "site_c": set(),
        }
        if self.bridges["bridge_bc"].status == "complete":
            adj["site_b"].add("site_c")
            adj["site_c"].add("site_b")
        seen = {site_id}
        frontier = [site_id]
        while frontier:
            cur = frontier.pop(0)
            for nxt in adj.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
        return seen

    def can_transport(self, from_site_id, to_site_id):
        if from_site_id == to_site_id:
            return True
        if {from_site_id, to_site_id} == {"site_a", "site_b"}:
            return True
        if {from_site_id, to_site_id} == {"site_b", "site_c"}:
            return self.bridges["bridge_bc"].status == "complete"
        return False

    def reserve_transport(self, agent_name, from_site_id, to_site_id, quantity):
        if agent_name in self._active_transports:
            return False
        quantity = max(1, int(quantity or 1))
        carry = max(1, int(self.parameters["carry_capacity"]))
        if quantity > carry:
            return False
        if not self.can_transport(from_site_id, to_site_id):
            return False
        if self.site_resource_inventory.get(from_site_id, 0) < quantity:
            return False
        self.site_resource_inventory[from_site_id] -= quantity
        self._active_transports[agent_name] = {
            "from_site_id": from_site_id,
            "to_site_id": to_site_id,
            "quantity": quantity,
            "remaining": int(self.parameters["move_time_per_unit"]) * quantity,
        }
        return True

    def is_agent_transporting(self, agent_name):
        return agent_name in self._active_transports

    def _advance_transports(self):
        finished = []
        for agent_name, tx in self._active_transports.items():
            tx["remaining"] -= 1
            if tx["remaining"] <= 0:
                self.site_resource_inventory[tx["to_site_id"]] = self.site_resource_inventory.get(tx["to_site_id"], 0) + tx["quantity"]
                finished.append(agent_name)
        for agent_name in finished:
            self._active_transports.pop(agent_name, None)

    def _consume_resource_for_site(self, site_id, quantity):
        quantity = max(1, int(quantity))
        accessible = self._accessible_sites_from(site_id)
        for candidate_id in sorted(accessible):
            available = self.site_resource_inventory.get(candidate_id, 0)
            if available >= quantity:
                self.site_resource_inventory[candidate_id] -= quantity
                pile = self.resource_nodes.get("pile_a") if candidate_id == "site_a" else self.resource_nodes.get("pile_c") if candidate_id == "site_c" else None
                if pile:
                    pile.quantity = max(0, min(pile.max_quantity, self.site_resource_inventory[candidate_id]))
                return True
        return False

    def update(self):
        self._advance_transports()
        self.sites["site_c"].buildable = self._is_site_buildable("site_c")
        for project in self.projects.values():
            required = int(project["required_resources"].get("bricks", 0) or 0)
            delivered = int(project["delivered_resources"].get("bricks", 0) or 0)
            project["resource_complete"] = required > 0 and delivered >= required
            project["structurally_complete"] = project["resource_complete"]
            if not project.get("started"):
                project["status"] = "not_started"
                project["in_progress"] = False
            elif project["resource_complete"] and project.get("validated_complete"):
                project["status"] = "complete"
                project["in_progress"] = False
            elif project["resource_complete"]:
                project["status"] = "ready_for_validation"
                project["in_progress"] = True
                project["validated_complete"] = False
            else:
                project["status"] = "in_progress"
                project["in_progress"] = True
                project["validated_complete"] = False

    def get_active_projects(self):
        return [p for p in self.projects.values() if p.get("started") and p["status"] != "complete"]

    def start_project(self, project_id):
        resolved_id = self.resolve_project_id(project_id, create_if_missing=True)
        project = self.projects.get(resolved_id)
        if not project:
            return False, "project_not_found"
        site_id = project["site_id"]
        if not self._is_site_buildable(site_id):
            return False, "site_not_buildable"
        if not project.get("started"):
            project["started"] = True
            if resolved_id not in self.sites[site_id].started_structures:
                self.sites[site_id].started_structures.append(resolved_id)
        self.update()
        self.update_project_provenance(resolved_id, event="project_started", sim_time=None)
        return True, "started"

    def deliver_resource(self, project_id, resource_type, quantity=1):
        resolved_id = self.resolve_project_id(project_id, create_if_missing=True)
        project = self.projects.get(resolved_id)
        if not project or resource_type != "bricks":
            return False
        started, _reason = self.start_project(resolved_id)
        if not started:
            return False
        if not self._consume_resource_for_site(project["site_id"], quantity):
            return False
        required = int(project["required_resources"].get(resource_type, 0) or 0)
        current = int(project["delivered_resources"].get(resource_type, 0) or 0)
        project["delivered_resources"][resource_type] = min(required, current + int(quantity))
        self.update()
        self.update_project_provenance(resolved_id, event="resource_delivered", sim_time=None)
        return True

    def mark_validated(self, project_id, is_valid=True):
        project = self.projects.get(project_id)
        if not project:
            return
        if not project.get("started"):
            return
        if not is_valid:
            project["correct"] = False
            project["validated_complete"] = False
            project["status"] = "needs_repair"
            project["in_progress"] = True
            self.update_project_provenance(project_id, event="validation_failed", sim_time=None)
            return
        if project.get("resource_complete"):
            project["validated_complete"] = True
            project["status"] = "complete"
            project["in_progress"] = False
        else:
            project["validated_complete"] = False
            project["status"] = "in_progress"
            project["in_progress"] = True
        self.update_project_provenance(project_id, event="validation_passed" if is_valid else "validation_failed", sim_time=None)

    def assign_builder(self, project_id, agent_name):
        resolved_id = self.resolve_project_id(project_id, create_if_missing=True)
        project = self.projects.get(resolved_id)
        if not project:
            return
        project["builders"].add(agent_name)
        self.start_project(resolved_id)
        self.update_project_provenance(resolved_id, event="builder_assigned", actor=agent_name, sim_time=None)

    def add_connector(self, from_project_id, to_project_id, connector_type="generic"):
        start_id = self.resolve_project_id(from_project_id, create_if_missing=False)
        end_id = self.resolve_project_id(to_project_id, create_if_missing=False)
        if not start_id or not end_id:
            return False, "unknown_structure_reference"
        start_project = self.projects.get(start_id)
        end_project = self.projects.get(end_id)
        if not start_project or not end_project:
            return False, "project_not_found"
        if start_project.get("site_id") != end_project.get("site_id"):
            return False, "cross_site_connectors_not_allowed"
        connector_id = f"connector_{len(self.connectors) + 1:03d}"
        self.connectors.append(
            {
                "connector_id": connector_id,
                "connector_type": str(connector_type or "generic"),
                "from_project_id": start_id,
                "to_project_id": end_id,
                "site_id": start_project.get("site_id"),
            }
        )
        return True, connector_id

    def build_bridge_bc(self, quantity=1):
        bridge = self.bridges["bridge_bc"]
        if bridge.status == "complete":
            return True
        if not self._consume_resource_for_site("site_b", quantity):
            return False
        bridge.delivered_resources = min(bridge.required_resources, bridge.delivered_resources + int(quantity))
        bridge.status = "in_progress" if bridge.delivered_resources < bridge.required_resources else "complete"
        self.update()
        return bridge.status == "complete"

    def get_visual_data(self):
        visuals = []
        for project in self.get_active_projects():
            visuals.append(
                {
                    "position": project["location"],
                    "radius": 0.34,
                    "progress": self._project_progress(project),
                    "border_color": "#444444",
                    "fill_color": "none",
                    "label": project["id"],
                }
            )
        return visuals

    def get_construction_scene_data(self):
        structures = []
        for project in self.projects.values():
            if not project.get("started"):
                continue
            structure_type = str(project.get("type") or "").strip().lower()
            style = self.STRUCTURE_STYLE_MAP.get(structure_type, {"symbol": "square", "color": "#666666"})
            structures.append(
                {
                    "project_id": project.get("id"),
                    "name": project.get("name"),
                    "structure_type": structure_type,
                    "site_id": project.get("site_id"),
                    "progress": self._project_progress(project),
                    "status": project.get("status", "unknown"),
                    "correct": bool(project.get("correct", True)),
                    "resource_complete": bool(project.get("resource_complete", False)),
                    "validated_complete": bool(project.get("validated_complete", False)),
                    "builders": sorted(project.get("builders", [])),
                    "symbol": style["symbol"],
                    "color": style["color"],
                }
            )

        resource_piles = []
        for pile in self.resource_nodes.values():
            remaining = max(0, int(pile.quantity))
            max_qty = max(1, int(pile.max_quantity))
            resource_piles.append(
                {
                    "pile_id": pile.pile_id,
                    "site_id": pile.site_id,
                    "position": pile.position,
                    "remaining": remaining,
                    "max_quantity": max_qty,
                    "fill_fraction": max(0.0, min(1.0, remaining / max_qty)),
                }
            )

        bridges = [
            {
                "bridge_id": "bridge_ab",
                "start_site_id": "site_a",
                "end_site_id": "site_b",
                "start": self.sites["site_a"].position,
                "end": self.sites["site_b"].position,
                "status": "complete",
                "progress": 1.0,
            }
        ]
        bridge_bc = self.bridges["bridge_bc"]
        if bridge_bc.status in {"in_progress", "complete"}:
            bridges.append(
                {
                    "bridge_id": "bridge_bc",
                    "start_site_id": "site_b",
                    "end_site_id": "site_c",
                    "start": self.sites["site_b"].position,
                    "end": self.sites["site_c"].position,
                    "status": bridge_bc.status,
                    "progress": max(0.0, min(1.0, bridge_bc.delivered_resources / max(1, bridge_bc.required_resources))),
                }
            )

        return {
            "sites": [
                {
                    "site_id": site.site_id,
                    "position": site.position,
                    "label": site.label,
                    "capacity": site.capacity,
                    "buildable": site.buildable,
                }
                for site in self.sites.values()
            ],
            "resource_piles": resource_piles,
            "bridges": bridges,
            "structures": structures,
            "connectors": list(self.connectors),
        }
