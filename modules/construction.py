# File: modules/construction.py

from modules.task_model import normalize_rule_token


class ConstructionManager:
    def __init__(self, task_model=None):
        self.task_model = task_model
        self.resource_nodes = {}
        self.projects = self._build_projects()

    def _build_projects(self):
        if self.task_model and getattr(self.task_model, "construction_templates", None):
            projects = {}
            for template in self.task_model.construction_templates.values():
                if not template.enabled:
                    continue
                projects[template.project_id] = {
                    "id": template.project_id,
                    "name": template.name,
                    "type": template.structure_type,
                    "location": (template.location_x, template.location_y),
                    "status": "in_progress",
                    "in_progress": True,
                    "correct": True,
                    "required_resources": dict(template.required_resources),
                    "delivered_resources": {k: 0 for k in template.required_resources},
                    "expected_rules": [normalize_rule_token(r) for r in template.expected_rules],
                    "resource_complete": False,
                    "structurally_complete": False,
                    "validated_complete": False,
                    "builders": set(),
                    "author": "system",
                    "artifact_type": template.artifact_type,
                    "target_id": template.target_id,
                }
            self.resource_nodes = {
                node.node_id: {
                    "id": node.node_id,
                    "zone_id": node.zone_id,
                    "resource_type": node.resource_type,
                    "quantity": node.quantity,
                    "position": (node.x, node.y),
                    "transport_time_scale": node.transport_time_scale,
                }
                for node in self.task_model.resource_nodes.values()
                if node.enabled
            }
            if projects:
                return projects

        return {
            "Build_Table_A": {
                "id": "Build_Table_A",
                "name": "Table A Construction",
                "type": "greenhouse",
                "location": (6.5, 3.4),
                "status": "in_progress",
                "in_progress": True,
                "correct": True,
                "required_resources": {"bricks": 12},
                "delivered_resources": {"bricks": 0},
                "expected_rules": [normalize_rule_token("rule:greenhouse_requires_water")],
                "resource_complete": False,
                "structurally_complete": False,
                "validated_complete": False,
                "builders": set(),
                "author": "system",
                "artifact_type": "greenhouse_construction",
            },
            "Build_Table_B": {
                "id": "Build_Table_B",
                "name": "Table B Construction",
                "type": "house",
                "location": (5.0, 4.4),
                "status": "in_progress",
                "in_progress": True,
                "correct": True,
                "required_resources": {"bricks": 14},
                "delivered_resources": {"bricks": 0},
                "expected_rules": [normalize_rule_token("rule:house_enclosed")],
                "resource_complete": False,
                "structurally_complete": False,
                "validated_complete": False,
                "builders": set(),
                "author": "system",
                "artifact_type": "house_construction",
            },
            "Build_Table_C": {
                "id": "Build_Table_C",
                "name": "Table C Construction",
                "type": "water_generator",
                "location": (3.5, 3.4),
                "status": "in_progress",
                "in_progress": True,
                "correct": True,
                "required_resources": {"bricks": 10},
                "delivered_resources": {"bricks": 0},
                "expected_rules": [normalize_rule_token("rule:water_generator_2x2")],
                "resource_complete": False,
                "structurally_complete": False,
                "validated_complete": False,
                "builders": set(),
                "author": "system",
                "artifact_type": "water_generator_construction",
            },
        }

    def update(self):
        for project in self.projects.values():
            required = project["required_resources"].get("bricks", 0)
            delivered = project["delivered_resources"].get("bricks", 0)
            resource_complete = delivered >= required if required > 0 else False
            project["resource_complete"] = resource_complete
            project["structurally_complete"] = resource_complete
            if resource_complete and project.get("validated_complete", False):
                project["status"] = "complete"
                project["in_progress"] = False
            elif resource_complete:
                project["status"] = "ready_for_validation"
                project["in_progress"] = True
                project["validated_complete"] = False
            else:
                project["status"] = "in_progress"
                project["in_progress"] = True
                project["validated_complete"] = False

    def get_active_projects(self):
        return [p for p in self.projects.values() if p["status"] != "complete"]

    def deliver_resource(self, project_id, resource_type, quantity=1):
        if project_id in self.projects:
            p = self.projects[project_id]
            if resource_type in p["required_resources"]:
                p["delivered_resources"][resource_type] += quantity
                self.update()

    def mark_validated(self, project_id, is_valid=True):
        project = self.projects.get(project_id)
        if not project:
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
        if project_id in self.projects:
            self.projects[project_id]["builders"].add(agent_name)

    def get_visual_data(self):
        visuals = []
        for p in self.get_active_projects():
            required = p["required_resources"].get("bricks", 0)
            delivered = p["delivered_resources"].get("bricks", 0)
            progress = min(1.0, delivered / required) if required > 0 else 0.0

            border = {
                "bridge": "blue",
                "greenhouse": "green",
                "water_generator": "teal",
                "house": "brown"
            }.get(p["type"], "gray")

            fill = "lightblue" if p["correct"] else "mistyrose"

            visuals.append({
                "position": p["location"],
                "radius": 0.4,
                "progress": progress,
                "border_color": border,
                "fill_color": fill,
                "label": p["id"]
            })
        return visuals
