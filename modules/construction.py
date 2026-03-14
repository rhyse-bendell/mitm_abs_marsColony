# File: modules/construction.py


class ConstructionManager:
    def __init__(self):
        self.projects = {
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
                "expected_rules": ["rule:greenhouse_requires_water"],
                "builders": set(),
                "author": "system",
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
                "expected_rules": ["rule:house_enclosed"],
                "builders": set(),
                "author": "system",
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
                "expected_rules": ["rule:water_generator_2x2"],
                "builders": set(),
                "author": "system",
            },
        }

    def update(self):
        for project in self.projects.values():
            if project["status"] == "in_progress":
                required = project["required_resources"]["bricks"]
                delivered = project["delivered_resources"]["bricks"]
                if delivered >= required:
                    project["status"] = "complete"
                    project["in_progress"] = False

    def get_active_projects(self):
        return [p for p in self.projects.values() if p["status"] != "complete"]

    def deliver_resource(self, project_id, resource_type, quantity=1):
        if project_id in self.projects:
            p = self.projects[project_id]
            if resource_type in p["required_resources"]:
                p["delivered_resources"][resource_type] += quantity
                self.update()

    def assign_builder(self, project_id, agent_name):
        if project_id in self.projects:
            self.projects[project_id]["builders"].add(agent_name)

    def get_visual_data(self):
        visuals = []
        for p in self.get_active_projects():
            required = p["required_resources"]["bricks"]
            delivered = p["delivered_resources"]["bricks"]
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
