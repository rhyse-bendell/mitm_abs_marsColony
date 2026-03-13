# File: modules/construction.py

class ConstructionManager:
    def __init__(self):
        self.projects = {

        }

    def update(self):
        for project in self.projects.values():
            if project["status"] == "in_progress":
                required = project["required_resources"]["bricks"]
                delivered = project["delivered_resources"]["bricks"]
                if delivered >= required:
                    project["status"] = "complete"

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
                "water_gen": "teal",
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
