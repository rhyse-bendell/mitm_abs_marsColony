# File: modules/environment.py

import math
from modules.construction import ConstructionManager


SHOW_LAYOUT = True

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    plt = None
    patches = None

try:
    import tkinter as tk
except ImportError:
    tk = None

from modules.knowledge import init_dik_packets


# ----------------------------
# Spatial Layout Configuration
# ----------------------------


# Default proximity radius required to access information packets,
# used when the object does not specify its own `access_radius`.
DEFAULT_INFO_ACCESS_RADIUS = 0.15

TABLE_INTERACTION_RADIUS = 0.2


RAW_OBJECTS = {
    "Team_Info": {
        "type": "rect",
        "position": (7.0, 6.4),
        "size": (2.0, 0.5),
        "label": "Team Info",
        "passable": True,
        "access_radius": DEFAULT_INFO_ACCESS_RADIUS
    },
    "Engineer_Info": {
        "type": "rect",
        "position": (3.25, 0.9),
        "size": (.4, 0.4),
        "label": "Engineer Info",
        "orientation": "left",
        "passable": True,
        "access_radius": DEFAULT_INFO_ACCESS_RADIUS
    },
    "Botanist_Info": {
        "type": "rect",
        "position": (4.75, 0.4),
        "size": (.4, 0.4),
        "label": "Botanist Info",
        "orientation": "down",
        "passable": True,
        "access_radius": DEFAULT_INFO_ACCESS_RADIUS
    },
    "Architect_Info": {
        "type": "rect",
        "position": (6.25, 0.9),
        "size": (.4, 0.4),
        "label": "Architect Info",
        "orientation": "right",
        "passable": True,
        "access_radius": DEFAULT_INFO_ACCESS_RADIUS
    },
    "Table_A": {
        "type": "circle",
        "position": (6.5, 3.4),
        "radius": 0.6,
        "label": "A"
    },
    "Table_B": {
        "type": "circle",
        "position": (5.0, 4.4),
        "radius": 0.6,
        "label": "B"
    },
    "Table_C": {
        "type": "circle",
        "position": (3.5, 3.4),
        "radius": 0.6,
        "label": "C"
    },
    "Bridge_A_B": {
        "type": "line",
        "start": (6.0, 3.65),
        "end": (5.4, 4.05),
        "label": "Bridge A-B"
    },
    "Blocked_Zone_AC": {
        "type": "blocked",
        "corners": [(4.1, 3.65), (5.9, 2.65)],
        "label": "Blocked"
    },
    "Blocked_Zone_leftWall": {
        "type": "blocked",
        "corners": [(0, 7.4), (.2, 0)],
        "label": "Blocked"
    },
    "Blocked_Zone_topWall": {
        "type": "blocked",
        "corners": [(0, 7.4), (10, 7.2)],
        "label": "Blocked"
    },
    "Blocked_Zone_rightWall": {
        "type": "blocked",
        "corners": [(9.8, 7.4), (10, 0)],
        "label": "Blocked"
    },
    "Blocked_Zone_bottomWall": {
        "type": "blocked",
        "corners": [(0, 0.2), (10, 0.0)],
        "label": "Blocked"
    }
}



ZONES = {
    "Zone_Team_Info": {"corners": [(7.0, 6.4), (9.0, 6.9)]},
    "Zone_Engineer_Info": {"corners": [(3.25, 0.9), (3.65, 1.3)]},
    "Zone_Botanist_Info": {"corners": [(4.75, 0.4), (5.15, 0.8)]},
    "Zone_Architect_Info": {"corners": [(6.25, 0.9), (6.65, 1.3)]},
    "Zone_Table_A": {"corners": [(5.9, 2.8), (7.1, 4.0)]},
    "Zone_Table_B": {"corners": [(4.4, 3.8), (5.6, 5.0)]},
    "Zone_Table_C": {"corners": [(2.9, 2.8), (4.1, 4.0)]},
    "Zone_Transition": {"default": True}
}


INTERACTION_TARGETS = {
    "Team_Info": {"kind": "information", "zone": "Zone_Team_Info"},
    "Engineer_Info": {"kind": "information", "zone": "Zone_Engineer_Info"},
    "Botanist_Info": {"kind": "information", "zone": "Zone_Botanist_Info"},
    "Architect_Info": {"kind": "information", "zone": "Zone_Architect_Info"},
    "Build_Table_A": {"kind": "build", "zone": "Zone_Table_A", "object": "Table_A"},
    "Build_Table_B": {"kind": "build", "zone": "Zone_Table_B", "object": "Table_B"},
    "Build_Table_C": {"kind": "build", "zone": "Zone_Table_C", "object": "Table_C"},
}

OBJECTS = RAW_OBJECTS

class Environment:
    def __init__(self, phases=None):
        self.objects = OBJECTS
        self.zones = ZONES
        self.phases = phases if phases else []
        self.current_phase_index = 0
        self.gas_environment = {"co2_level": 400}
        self.resources = []
        self.construction = ConstructionManager()
        self.knowledge_packets = init_dik_packets()
        self.interaction_targets = INTERACTION_TARGETS
        self._time = 0.0

    def _zone_corners(self, zone_name):
        zone = self.zones.get(zone_name, {})
        return zone.get("corners")

    def _zone_candidate_points(self, zone_name):
        corners = self._zone_corners(zone_name)
        if not corners:
            return []
        (x1, y1), (x2, y2) = corners
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return [
            (x_min, y_min),
            (x_min, y_max),
            (x_max, y_min),
            (x_max, y_max),
            (cx, y_min),
            (cx, y_max),
            (x_min, cy),
            (x_max, cy),
            (cx, cy),
        ]

    def is_point_navigable(self, point, threshold=0.15):
        for name, obj in self.objects.items():
            if obj.get("type") in {"rect", "circle", "blocked"} and not obj.get("passable", False):
                if self.is_near_object(point, name, threshold=threshold):
                    return False
        return True


    def _segment_is_navigable(self, start, end, samples=24):
        for i in range(1, samples + 1):
            t = i / samples
            px = start[0] + (end[0] - start[0]) * t
            py = start[1] + (end[1] - start[1]) * t
            if not self.is_point_navigable((px, py)):
                return False
        return True

    def get_interaction_target_position(self, target_name, from_position=None):
        target = self.interaction_targets.get(target_name)
        if not target:
            return None

        candidates = [p for p in self._zone_candidate_points(target["zone"]) if self.is_point_navigable(p)]
        if target.get("kind") == "build":
            candidates = [
                p for p in candidates
                if not any(
                    self.is_near_object(p, name, threshold=0.25)
                    for name, obj in self.objects.items()
                    if obj.get("type") == "blocked"
                )
            ] or candidates

        if not candidates:
            return None

        if from_position is None:
            return candidates[0]

        clear_candidates = [p for p in candidates if self._segment_is_navigable(from_position, p)]
        if clear_candidates:
            candidates = clear_candidates

        return min(candidates, key=lambda p: math.hypot(p[0] - from_position[0], p[1] - from_position[1]))

    def get_spawn_points(self):
        return [
            (6.0, 1.0),  # Near Architect Info
            (5.0, 1.0),  # Between tables but clear
            (4.0, 1.0)  # Near Engineer Info
        ]

    def update(self, time):
        self._time = time  # Store for later use
        if self.phases:
            self._update_phase(time)
        if self.construction:
            self.construction.update()

    def _update_phase(self, time):
        total_elapsed = 0
        for i, phase in enumerate(self.phases):
            phase_end = total_elapsed + (phase["duration_minutes"] * 60)
            if time < phase_end:
                self.current_phase_index = i
                break
            total_elapsed = phase_end

    def get_current_phase(self):
        return self.phases[self.current_phase_index] if self.phases else None

    def is_near_object(self, agent_pos, object_name, threshold=0.5):
        obj = self.objects[object_name]
        x, y = agent_pos

        if obj["type"] == "rect":
            ox, oy = obj["position"]
            w, h = obj["size"]
            nearest_x = max(ox, min(x, ox + w))
            nearest_y = max(oy, min(y, oy + h))
            dist = math.hypot(x - nearest_x, y - nearest_y)
            return dist <= threshold

        elif obj["type"] == "circle":
            ox, oy = obj["position"]
            radius = obj["radius"]
            dist = math.hypot(x - ox, y - oy)
            return dist <= (radius + threshold)

        elif obj["type"] == "blocked":
            (x1, y1), (x2, y2) = obj["corners"]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            return (x_min <= x <= x_max) and (y_min <= y <= y_max)

        return False

    def can_access_info(self, position, packet_name, role=None):
        """
        Determines whether a packet is accessible given agent's position and role.
        """
        # Match object name to packet name exactly or with a fallback
        object_key = None
        if packet_name in self.objects:
            object_key = packet_name
        elif packet_name + "_Info" in self.objects:
            object_key = packet_name + "_Info"
        else:
            return False

        obj = self.objects[object_key]
        access_radius = obj.get("access_radius", DEFAULT_INFO_ACCESS_RADIUS)

        if obj.get("type") == "rect":
            ox, oy = obj["position"]
            w, h = obj["size"]
            nearest_x = max(ox, min(position[0], ox + w))
            nearest_y = max(oy, min(position[1], oy + h))
            dist = math.hypot(position[0] - nearest_x, position[1] - nearest_y)
        elif obj.get("type") == "circle":
            ox, oy = obj["position"]
            radius = obj["radius"]
            center_dist = math.hypot(position[0] - ox, position[1] - oy)
            dist = max(0.0, center_dist - radius)
        else:
            obj_pos = obj.get("position", (0.0, 0.0))
            dist = math.hypot(position[0] - obj_pos[0], position[1] - obj_pos[1])

        if dist > access_radius:
            return False

        # If packet has a role restriction, ensure agent has the correct role
        if "role" in self.objects[packet_name]:
            if role is None or self.objects[packet_name]["role"] != role:
                return False

        return True

    def can_interact_with_table(self, agent_pos, table_name):
        return self.is_near_object(agent_pos, table_name, threshold=TABLE_INTERACTION_RADIUS)

    def get_visible_resources(self, agent_pos, radius=1.0):
        visible = []
        for r in self.resources:
            if r["carried_by"] is None:
                dist = math.hypot(agent_pos[0] - r["position"][0], agent_pos[1] - r["position"][1])
                if dist <= radius:
                    visible.append(r)
        return visible

    def _point_in_zone(self, pos, corners):
        x, y = pos
        (x1, y1), (x2, y2) = corners
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    def get_spawn_point(self, role):
        """
        Returns a safe, role-specific spawn point not inside any blocked zone.
        """
        candidate_spawns = {
            "Architect": (6.9, 1.2),  # Near Architect_Info, outside station geometry
            "Engineer": (3.9, 1.2),  # Near Engineer_Info, outside station geometry
            "Botanist": (5.0, 1.0)  # Between them
        }
        pos = candidate_spawns.get(role, (6.0, 3.0))
        if not self.is_in_blocked_zone(pos):
            return pos

        # Fallback safe spot
        return (5.0, 5.0)

    def is_in_blocked_zone(self, pos):
        x, y = pos
        for obj in self.objects.values():
            if obj.get("type") == "blocked":
                (x1, y1), (x2, y2) = obj["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
        return False


    def get_time(self):
        return getattr(self, "_time", 0.0)


# ----------------------------
# Layout Visualizer for Tweaking
# ----------------------------

def get_screen_size_inches(dpi=100):
    if tk is None:
        return 10, 8
    root = tk.Tk()
    root.withdraw()
    width_px = root.winfo_screenwidth()
    height_px = root.winfo_screenheight()
    return width_px / dpi, height_px / dpi

if __name__ == "__main__" and plt is not None:
    if SHOW_LAYOUT:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.set_title("Environment Layout Viewer")

        def draw_orientation_arrow(x, y, direction):
            angle_map = {
                'up': np.pi,
                'down': 0,
                'left': np.pi / 2,
                'right': -np.pi / 2
            }
            if direction in angle_map:
                arrow = patches.RegularPolygon((x, y), numVertices=3, radius=0.2, orientation=angle_map[direction])
                ax.add_patch(arrow)

        for name, e in OBJECTS.items():
            if e["type"] == "rect":
                x, y = e["position"]
                w, h = e["size"]
                ax.add_patch(patches.Rectangle((x, y), w, h, edgecolor='black', facecolor='lightgray'))
                ax.text(x + w / 2, y + h / 2, e["label"], ha='center', va='center')
                if "orientation" in e:
                    draw_orientation_arrow(x + w / 2, y + h / 2, e["orientation"])
            elif e["type"] == "circle":
                x, y = e["position"]
                ax.add_patch(patches.Circle((x, y), e["radius"], edgecolor='black', facecolor='lightblue'))
                ax.text(x, y, e["label"], ha='center', va='center', fontsize=12, weight='bold')
            elif e["type"] == "line":
                sx, sy = e["start"]
                ex, ey = e["end"]
                ax.plot([sx, ex], [sy, ey], color='gray', linewidth=4)
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                ax.text(mx, my + 0.2, e["label"], ha='center', va='center', fontsize=9, color='dimgray')
            elif e["type"] == "blocked":
                (x1, y1), (x2, y2) = e["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                               edgecolor='black', facecolor='darkgray'))
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, e["label"], ha='center', va='center')

        for zone_name, zone in ZONES.items():
            if "corners" in zone:
                (x1, y1), (x2, y2) = zone["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                               edgecolor='purple', linestyle='--', facecolor='none', linewidth=1.5))
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, zone_name.replace("Zone_", ""),
                        ha='center', va='center', fontsize=9, color='purple')

        plt.tight_layout()
        plt.show()
