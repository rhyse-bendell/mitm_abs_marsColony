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
        "Build_Table_A": "site_a",
        "Build_Table_B": "site_b",
        "Build_Table_C": "site_c",
    }

    def __init__(self, task_model=None, parameters: Optional[Dict] = None):
        self.task_model = task_model
        self.parameters = dict(self.DEFAULT_PARAMETERS)
        if isinstance(parameters, dict):
            self.parameters.update(parameters)

        self.sites = {
            "site_a": ConstructionSite("site_a", "Site A", (6.5, 3.4), max(1, int(self.parameters["site_a_capacity"])), True),
            "site_b": ConstructionSite("site_b", "Site B", (5.0, 4.4), max(1, int(self.parameters["site_b_capacity"])), True),
            "site_c": ConstructionSite("site_c", "Site C", (3.5, 3.4), max(1, int(self.parameters["site_c_capacity"])), False),
        }
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

        self.projects = self._build_projects()
        self.update()

    def _build_projects(self):
        defaults = {
            "Build_Table_A": {
                "name": "Housing at Site A",
                "type": "house",
                "artifact_type": "house_construction",
                "expected_rules": [normalize_rule_token("rule:house_enclosed")],
                "required": int(self.parameters["housing_cost"]),
            },
            "Build_Table_B": {
                "name": "Greenhouse at Site B",
                "type": "greenhouse",
                "artifact_type": "greenhouse_construction",
                "expected_rules": [normalize_rule_token("rule:greenhouse_requires_water")],
                "required": int(self.parameters["greenhouse_cost"]),
            },
            "Build_Table_C": {
                "name": "Water Generator at Site C",
                "type": "water_generator",
                "artifact_type": "water_generator_construction",
                "expected_rules": [normalize_rule_token("rule:water_generator_2x2")],
                "required": int(self.parameters["water_generator_cost"]),
            },
        }
        projects = {}
        for project_id, conf in defaults.items():
            site_id = self.PROJECT_TO_SITE[project_id]
            projects[project_id] = {
                "id": project_id,
                "name": conf["name"],
                "type": conf["type"],
                "location": self.sites[site_id].position,
                "site_id": site_id,
                "status": "not_started",
                "in_progress": False,
                "started": False,
                "correct": True,
                "required_resources": {"bricks": max(1, int(conf["required"]))},
                "delivered_resources": {"bricks": 0},
                "expected_rules": list(conf["expected_rules"]),
                "resource_complete": False,
                "structurally_complete": False,
                "validated_complete": False,
                "builders": set(),
                "author": "system",
                "artifact_type": conf["artifact_type"],
                "target_id": project_id,
            }
        return projects

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
        return len(site.started_structures) < site.capacity

    def _is_site_buildable(self, site_id):
        if site_id != "site_c":
            return True
        return self.bridges["bridge_bc"].status == "complete"

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
        project = self.projects.get(project_id)
        if not project:
            return False, "project_not_found"
        site_id = project["site_id"]
        if not self._is_site_buildable(site_id):
            return False, "site_not_buildable"
        if not project.get("started") and not self._site_has_capacity(site_id):
            return False, "site_capacity_reached"
        if not project.get("started"):
            project["started"] = True
            if project_id not in self.sites[site_id].started_structures:
                self.sites[site_id].started_structures.append(project_id)
        self.update()
        return True, "started"

    def deliver_resource(self, project_id, resource_type, quantity=1):
        project = self.projects.get(project_id)
        if not project or resource_type != "bricks":
            return False
        started, _reason = self.start_project(project_id)
        if not started:
            return False
        if not self._consume_resource_for_site(project["site_id"], quantity):
            return False
        required = int(project["required_resources"].get(resource_type, 0) or 0)
        current = int(project["delivered_resources"].get(resource_type, 0) or 0)
        project["delivered_resources"][resource_type] = min(required, current + int(quantity))
        self.update()
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
            return
        if project.get("resource_complete"):
            project["validated_complete"] = True
            project["status"] = "complete"
            project["in_progress"] = False
        else:
            project["validated_complete"] = False
            project["status"] = "in_progress"
            project["in_progress"] = True

    def assign_builder(self, project_id, agent_name):
        project = self.projects.get(project_id)
        if not project:
            return
        project["builders"].add(agent_name)
        self.start_project(project_id)

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
