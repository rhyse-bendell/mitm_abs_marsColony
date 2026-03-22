# File: modules/environment.py

import math
import heapq
from modules.construction import ConstructionManager
from modules.task_model import TaskModel


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
DEFAULT_INFO_ACCESS_RADIUS = 0.4
SOURCE_SLOT_DISTANCE = 0.28
SOURCE_QUEUE_SPACING = 0.22

TABLE_INTERACTION_RADIUS = 0.35
VIEWPORT_MARGIN = 0.2


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


def _objects_from_task_model(task_model):
    out = {}
    for obj in task_model.environment_objects.values():
        if not obj.enabled:
            continue
        row = {"type": obj.object_type, "label": obj.label}
        if obj.object_type == "rect":
            row["position"] = (obj.x, obj.y)
            row["size"] = (obj.width, obj.height)
        elif obj.object_type == "circle":
            row["position"] = (obj.x, obj.y)
            row["radius"] = obj.radius
        elif obj.object_type == "line":
            row["start"] = (obj.x, obj.y)
            row["end"] = (obj.end_x, obj.end_y)
        elif obj.object_type == "blocked":
            row["corners"] = [(obj.x, obj.y), (obj.end_x, obj.end_y)]
        if obj.passable:
            row["passable"] = True
        if obj.access_radius:
            row["access_radius"] = obj.access_radius
        if obj.orientation:
            row["orientation"] = obj.orientation
        if obj.role_restriction:
            row["role"] = obj.role_restriction
        out[obj.object_id] = row
    return out


def _zones_from_task_model(task_model):
    out = {}
    for zone in task_model.zones.values():
        if not zone.enabled:
            continue
        if zone.default_zone:
            out[zone.zone_id] = {"default": True}
        else:
            out[zone.zone_id] = {"corners": [(zone.x1, zone.y1), (zone.x2, zone.y2)]}
    return out


def _targets_from_task_model(task_model):
    out = {}
    for target in task_model.interaction_targets.values():
        if not target.enabled:
            continue
        row = {"kind": target.kind, "zone": target.zone_id}
        if target.object_id:
            row["object"] = target.object_id
        if target.role_scope:
            row["role_scope"] = target.role_scope
        out[target.target_id] = row
    return out

class Environment:
    SOURCE_PACKET_NAME_MAP = {
        "SRC_TEAM_SHARED": "Team_Info",
        "SRC_ARCHITECT_BRIEF": "Architect_Info",
        "SRC_ENGINEER_BRIEF": "Engineer_Info",
        "SRC_BOTANIST_BRIEF": "Botanist_Info",
    }

    def resolve_source_id(self, packet_name: str) -> str | None:
        """Resolve canonical source_id for a packet/interaction target name when available."""
        inverse = {v: k for k, v in self.source_packet_name_map.items()}
        if packet_name in inverse:
            return inverse[packet_name]
        if self.task_model and packet_name in self.task_model.sources:
            return packet_name
        return None

    def source_metadata_for_packet(self, packet_name: str) -> dict:
        """Return source/target metadata normalized for inspect/access semantics."""
        source_id = self.resolve_source_id(packet_name)
        source = self.task_model.sources.get(source_id) if self.task_model and source_id else None
        target = self.interaction_targets.get(packet_name, {})
        role_scope = target.get("role_scope", [])
        if isinstance(role_scope, str):
            role_scope = [role_scope]
        return {
            "packet_name": packet_name,
            "source_id": source_id,
            "source_type": getattr(source, "source_type", None),
            "access_scope": getattr(source, "access_scope", None),
            "role_scope": [str(r).lower() for r in role_scope],
        }

    @staticmethod
    def expected_role_for_packet(packet_name: str) -> str | None:
        packet = str(packet_name or "")
        if packet.endswith("_Info") and packet != "Team_Info":
            return packet.replace("_Info", "")
        return None

    def is_shared_information_source(self, packet_name: str) -> bool:
        meta = self.source_metadata_for_packet(packet_name)
        access_scope = str(meta.get("access_scope") or "").strip().lower()
        if access_scope in {"team", "all"}:
            return True
        role_scope = set(meta.get("role_scope") or [])
        if role_scope & {"team", "all"}:
            return True
        return str(packet_name).strip().lower().startswith("team_")

    def classify_source_access(self, packet_name: str, position=None, role=None, target_kind=None) -> dict:
        """Classify a source interaction for strict witness/metrics accounting."""
        packet_name = str(packet_name)
        target = self.interaction_targets.get(packet_name, {})
        kind = str(target_kind or target.get("kind") or "information").strip().lower()
        meta = self.source_metadata_for_packet(packet_name)
        is_shared = self.is_shared_information_source(packet_name)
        expected_role = None if is_shared else self.expected_role_for_packet(packet_name)

        role_mismatch = bool(expected_role and role and str(expected_role).lower() != str(role).lower())
        movement_only = False
        if position is not None and kind == "information":
            movement_only = not self.can_access_info(position, packet_name, role=role)

        if kind != "information":
            classification = "artifact_consultation" if kind == "artifact" else "movement_only_visit"
        elif is_shared:
            classification = "shared_team_source"
        elif role_mismatch:
            classification = "role_mismatch_visit"
        elif movement_only:
            classification = "movement_only_visit"
        else:
            classification = "role_private_source"

        return {
            "source_id": meta.get("source_id") or self.resolve_source_id(packet_name) or packet_name,
            "packet_name": packet_name,
            "classification": classification,
            "is_shared_source": bool(classification == "shared_team_source"),
            "is_private_source": bool(classification == "role_private_source"),
            "is_role_mismatch": bool(classification == "role_mismatch_visit"),
            "is_movement_only": bool(classification == "movement_only_visit"),
            "is_artifact_consultation": bool(classification == "artifact_consultation"),
            "access_scope": meta.get("access_scope"),
        }

    def __init__(self, phases=None, task_model: TaskModel | None = None):
        self.objects = _objects_from_task_model(task_model) if task_model and task_model.environment_objects else OBJECTS
        self.zones = _zones_from_task_model(task_model) if task_model and task_model.zones else ZONES
        self.phases = phases if phases else []
        self.current_phase_index = 0
        self.gas_environment = {"co2_level": 400}
        self.resources = []
        self.construction = ConstructionManager(task_model=task_model)
        self.task_model = task_model
        self.source_packet_name_map = dict(self.SOURCE_PACKET_NAME_MAP)
        if self.task_model is not None:
            self.knowledge_packets = init_dik_packets(
                task_model=self.task_model,
                source_name_map=self.source_packet_name_map,
            )
        else:
            self.knowledge_packets = init_dik_packets()
        self.interaction_targets = _targets_from_task_model(task_model) if task_model and task_model.interaction_targets else INTERACTION_TARGETS
        self._time = 0.0
        self._path_cache = {}
        self._source_slot_reservations = {}
        self._source_queue_reservations = {}

    @staticmethod
    def _quantize_point(point, step=0.2):
        return (round(float(point[0]) / step) * step, round(float(point[1]) / step) * step)

    @staticmethod
    def _heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from, node):
        out = [node]
        while node in came_from:
            node = came_from[node]
            out.append(node)
        out.reverse()
        return out

    def _neighbor_points(self, point, step=0.2):
        x, y = point
        deltas = [
            (-step, 0.0),
            (step, 0.0),
            (0.0, -step),
            (0.0, step),
            (-step, -step),
            (-step, step),
            (step, -step),
            (step, step),
        ]
        (x_min, x_max), (y_min, y_max) = self.get_viewport_bounds(margin=0.0)
        out = []
        for dx, dy in deltas:
            nx = round(x + dx, 3)
            ny = round(y + dy, 3)
            if nx < x_min or nx > x_max or ny < y_min or ny > y_max:
                continue
            out.append((nx, ny))
        return out

    def plan_path(self, start, target, mode="grid_astar", grid_step=0.35):
        start = (float(start[0]), float(start[1]))
        target = (float(target[0]), float(target[1]))
        if mode != "grid_astar":
            if self._segment_is_navigable(start, target):
                return {
                    "status": "ok",
                    "waypoints": [target],
                    "from_cache": False,
                    "path_mode": mode,
                    "blocker_category": None,
                }
            return {
                "status": "failed",
                "waypoints": [],
                "from_cache": False,
                "path_mode": mode,
                "blocker_category": "no_path_found",
            }

        q_start = self._quantize_point(start, step=grid_step)
        q_target = self._quantize_point(target, step=grid_step)

        if not self.is_point_navigable(q_target):
            return {
                "status": "failed",
                "waypoints": [],
                "from_cache": False,
                "path_mode": mode,
                "blocker_category": "target_unreachable",
            }

        cache_key = (q_start, q_target, float(grid_step), mode)
        cached = self._path_cache.get(cache_key)
        if cached:
            return {
                "status": "ok",
                "waypoints": list(cached),
                "from_cache": True,
                "path_mode": mode,
                "blocker_category": None,
            }

        if self._segment_is_navigable(q_start, q_target):
            path = [q_target]
            if self._heuristic(q_target, target) > 1e-6:
                path.append(target)
            self._path_cache[cache_key] = list(path)
            return {
                "status": "ok",
                "waypoints": path,
                "from_cache": False,
                "path_mode": mode,
                "blocker_category": None,
            }

        open_heap = []
        heapq.heappush(open_heap, (0.0, q_start))
        came_from = {}
        g_score = {q_start: 0.0}
        closed = set()
        max_nodes = 6000

        while open_heap and len(closed) < max_nodes:
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            if self._heuristic(current, q_target) <= (grid_step * 0.75):
                path = self._reconstruct_path(came_from, current)
                path[-1] = q_target
                if self._heuristic(path[-1], target) > 1e-6:
                    path.append(target)
                if path and self._heuristic(path[0], start) <= (grid_step * 0.75):
                    path = path[1:]
                if path:
                    self._path_cache[cache_key] = list(path)
                    return {
                        "status": "ok",
                        "waypoints": path,
                        "from_cache": False,
                        "path_mode": mode,
                        "blocker_category": None,
                    }
                break

            for neighbor in self._neighbor_points(current, step=grid_step):
                if neighbor in closed:
                    continue
                if not self.is_point_navigable(neighbor):
                    continue
                if not self._segment_is_navigable(current, neighbor, samples=3):
                    continue
                step_cost = self._heuristic(current, neighbor)
                tentative_g = g_score[current] + step_cost
                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + self._heuristic(neighbor, q_target)
                heapq.heappush(open_heap, (f_score, neighbor))

        return {
            "status": "failed",
            "waypoints": [],
            "from_cache": False,
            "path_mode": mode,
            "blocker_category": "no_path_found",
        }

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

    def _nearest_distance_to_zone(self, point, zone_name):
        corners = self._zone_corners(zone_name)
        if not corners:
            return float("inf")
        (x1, y1), (x2, y2) = corners
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        px, py = point
        nearest_x = max(x_min, min(px, x_max))
        nearest_y = max(y_min, min(py, y_max))
        return math.hypot(px - nearest_x, py - nearest_y)

    def get_viewport_bounds(self, margin=VIEWPORT_MARGIN):
        extents = []

        for obj in self.objects.values():
            obj_type = obj.get("type")
            if obj_type == "rect":
                x, y = obj["position"]
                w, h = obj["size"]
                extents.append((x, x + w, y, y + h))
            elif obj_type == "circle":
                x, y = obj["position"]
                r = obj["radius"]
                extents.append((x - r, x + r, y - r, y + r))
            elif obj_type == "line":
                sx, sy = obj["start"]
                ex, ey = obj["end"]
                extents.append((min(sx, ex), max(sx, ex), min(sy, ey), max(sy, ey)))
            elif obj_type == "blocked":
                (x1, y1), (x2, y2) = obj["corners"]
                extents.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))

        for zone in self.zones.values():
            corners = zone.get("corners")
            if not corners:
                continue
            (x1, y1), (x2, y2) = corners
            extents.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))

        if not extents:
            return (0.0, 10.0), (0.0, 10.0)

        x_min = min(e[0] for e in extents) - margin
        x_max = max(e[1] for e in extents) + margin
        y_min = min(e[2] for e in extents) - margin
        y_max = max(e[3] for e in extents) + margin
        return (x_min, x_max), (y_min, y_max)

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
        if not self.is_interaction_target_unlocked(target_name):
            return None

        if target.get("kind") == "information":
            slot = self.select_source_access_point(target_name, agent_id=None, from_position=from_position)
            if slot is not None:
                return tuple(slot.get("position"))

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

    def get_source_access_slots(self, packet_name):
        obj = self.objects.get(packet_name)
        if obj is None:
            return []

        slots = []
        if obj.get("type") == "rect":
            x, y = obj["position"]
            w, h = obj["size"]
            cx = x + (w / 2.0)
            cy = y + (h / 2.0)
            q1x = x + (w * 0.25)
            q3x = x + (w * 0.75)
            q1y = y + (h * 0.25)
            q3y = y + (h * 0.75)
            slots = [
                {"slot_id": "bottom_left", "position": (q1x, y - SOURCE_SLOT_DISTANCE), "queue_dir": (0.0, -1.0)},
                {"slot_id": "bottom_right", "position": (q3x, y - SOURCE_SLOT_DISTANCE), "queue_dir": (0.0, -1.0)},
                {"slot_id": "top_left", "position": (q1x, y + h + SOURCE_SLOT_DISTANCE), "queue_dir": (0.0, 1.0)},
                {"slot_id": "top_right", "position": (q3x, y + h + SOURCE_SLOT_DISTANCE), "queue_dir": (0.0, 1.0)},
                {"slot_id": "left_mid", "position": (x - SOURCE_SLOT_DISTANCE, cy), "queue_dir": (-1.0, 0.0)},
                {"slot_id": "right_mid", "position": (x + w + SOURCE_SLOT_DISTANCE, cy), "queue_dir": (1.0, 0.0)},
            ]
        elif obj.get("type") == "circle":
            x, y = obj["position"]
            r = float(obj.get("radius", 0.3)) + SOURCE_SLOT_DISTANCE
            slots = [
                {"slot_id": "north", "position": (x, y + r), "queue_dir": (0.0, 1.0)},
                {"slot_id": "south", "position": (x, y - r), "queue_dir": (0.0, -1.0)},
                {"slot_id": "west", "position": (x - r, y), "queue_dir": (-1.0, 0.0)},
                {"slot_id": "east", "position": (x + r, y), "queue_dir": (1.0, 0.0)},
            ]
        else:
            pos = tuple(obj.get("position", (0.0, 0.0)))
            slots = [{"slot_id": "default", "position": pos, "queue_dir": (0.0, -1.0)}]

        return [s for s in slots if self.is_point_navigable(s["position"]) or self.can_access_info(s["position"], packet_name)]

    def _slot_key(self, packet_name, slot_id):
        return f"{packet_name}:{slot_id}"

    def reserve_source_access_slot(self, packet_name, slot_id, agent_id):
        key = self._slot_key(packet_name, slot_id)
        owner = self._source_slot_reservations.get(key)
        if owner is None or owner == agent_id:
            self._source_slot_reservations[key] = agent_id
            return True
        return False

    def release_source_access_slot(self, packet_name, agent_id=None, slot_id=None):
        released = []
        for slot in self.get_source_access_slots(packet_name):
            sid = slot.get("slot_id")
            if slot_id is not None and sid != slot_id:
                continue
            key = self._slot_key(packet_name, sid)
            owner = self._source_slot_reservations.get(key)
            if owner is None:
                continue
            if agent_id is None or owner == agent_id:
                self._source_slot_reservations.pop(key, None)
                released.append(sid)
        self._source_queue_reservations = {
            k: v for k, v in self._source_queue_reservations.items()
            if not (k.startswith(f"{packet_name}:") and (agent_id is None or v == agent_id))
        }
        return released

    def select_source_access_point(self, packet_name, agent_id=None, from_position=None):
        slots = self.get_source_access_slots(packet_name)
        if not slots:
            return None

        def _distance(p):
            if from_position is None:
                return 0.0
            return math.hypot(p["position"][0] - from_position[0], p["position"][1] - from_position[1])

        ranked = sorted(slots, key=lambda s: (_distance(s), s["slot_id"]))
        free = []
        owned = []
        busy = []
        for slot in ranked:
            key = self._slot_key(packet_name, slot["slot_id"])
            owner = self._source_slot_reservations.get(key)
            if owner is None:
                free.append(slot)
            elif agent_id is not None and owner == agent_id:
                owned.append(slot)
            else:
                busy.append(slot)

        if owned:
            chosen = owned[0]
            return {"kind": "slot", "slot_id": chosen["slot_id"], "position": chosen["position"], "reason": "owned"}
        if free:
            chosen = free[0]
            if agent_id is not None:
                self.reserve_source_access_slot(packet_name, chosen["slot_id"], agent_id)
            return {"kind": "slot", "slot_id": chosen["slot_id"], "position": chosen["position"], "reason": "free"}
        if not busy:
            return None

        anchor = busy[0]
        queue_index = sum(1 for k in self._source_queue_reservations if k.startswith(f"{packet_name}:{anchor['slot_id']}:"))
        qx = anchor["position"][0] + (anchor["queue_dir"][0] * SOURCE_QUEUE_SPACING * (queue_index + 1))
        qy = anchor["position"][1] + (anchor["queue_dir"][1] * SOURCE_QUEUE_SPACING * (queue_index + 1))
        queue_pos = (qx, qy)
        queue_key = f"{packet_name}:{anchor['slot_id']}:q{queue_index + 1}:{agent_id}"
        if agent_id is not None:
            self._source_queue_reservations[queue_key] = agent_id
        return {
            "kind": "queue",
            "slot_id": anchor["slot_id"],
            "position": queue_pos,
            "queue_index": queue_index + 1,
            "reason": "all_slots_occupied",
        }

    def can_agent_use_source_slot(self, packet_name, agent_id, position, slot_id=None, role=None):
        if not self.can_access_info(position, packet_name, role=role):
            return False, "too_far_or_role_mismatch"
        slots = self.get_source_access_slots(packet_name)
        if not slots:
            return False, "no_slots"
        slot = None
        if slot_id is not None:
            slot = next((s for s in slots if s["slot_id"] == slot_id), None)
        if slot is None:
            slot = min(slots, key=lambda s: math.hypot(position[0] - s["position"][0], position[1] - s["position"][1]))
        if math.hypot(position[0] - slot["position"][0], position[1] - slot["position"][1]) > 0.24:
            return False, "not_at_interaction_slot"
        key = self._slot_key(packet_name, slot["slot_id"])
        owner = self._source_slot_reservations.get(key)
        if owner is not None and owner != agent_id:
            return False, "slot_reserved_by_other"
        return True, "slot_access_ok"

    def source_slot_snapshot(self, packet_name):
        slots = self.get_source_access_slots(packet_name)
        occupancy = {}
        for s in slots:
            occupancy[s["slot_id"]] = self._source_slot_reservations.get(self._slot_key(packet_name, s["slot_id"]))
        return {"slots": [s["slot_id"] for s in slots], "occupancy": occupancy}

    def is_source_slot_context(self, point, packet_name):
        for slot in self.get_source_access_slots(packet_name):
            if math.hypot(point[0] - slot["position"][0], point[1] - slot["position"][1]) <= 0.32:
                return True
        return False

    def get_spawn_points(self):
        if self.task_model and self.task_model.spawn_points:
            return [(sp.x, sp.y) for sp in self.task_model.spawn_points if sp.enabled]
        return [
            (7.35, 3.05),
            (7.75, 3.40),
            (7.35, 3.75)
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

    def has_phase_unlock(self, unlock_id):
        if not unlock_id or not self.phases:
            return False
        normalized = str(unlock_id).strip().lower()
        for idx, phase in enumerate(self.phases):
            if idx > self.current_phase_index:
                break
            unlocks = phase.get("unlocks") or []
            if any(str(u).strip().lower() == normalized for u in unlocks):
                return True
        return False

    def is_interaction_target_unlocked(self, target_name):
        if target_name == "Build_Table_C":
            return self.has_phase_unlock("bridge_to_zone_C")
        return True

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

        expected_role = self.expected_role_for_packet(object_key)
        if self.is_shared_information_source(object_key):
            expected_role = None
        if expected_role is not None:
            if str(role or "").strip().lower() != str(expected_role).strip().lower():
                return False

        # If packet has an explicit role restriction, ensure agent has the correct role
        if "role" in self.objects[object_key]:
            if role is None or self.objects[object_key]["role"] != role:
                return False

        target_meta = self.interaction_targets.get(object_key, {})
        zone_name = target_meta.get("zone")
        if zone_name and self._nearest_distance_to_zone(position, zone_name) <= access_radius:
            return True

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

        return True

    def can_interact_with_table(self, agent_pos, table_name):
        table_zone_name = f"Zone_{table_name}"
        if table_zone_name in self.zones and self._point_in_zone(agent_pos, self.zones[table_zone_name]["corners"]):
            return True
        return self.is_near_object(agent_pos, table_name, threshold=TABLE_INTERACTION_RADIUS)

    def get_interaction_access(self, position, target_name, role=None):
        target = self.interaction_targets.get(target_name)
        if not target:
            return {"accessible": False, "reason": "unknown_target"}

        if not self.is_interaction_target_unlocked(target_name):
            return {"accessible": False, "reason": "locked_until_bridge_access"}

        if target.get("kind") == "information":
            role_ok = self.can_access_info(position, target_name, role=role)
            if not role_ok:
                return {"accessible": False, "reason": "too_far_or_role_mismatch"}

            zone_name = target.get("zone")
            obj = self.objects.get(target_name)
            access_radius = (obj or {}).get("access_radius", DEFAULT_INFO_ACCESS_RADIUS)
            if zone_name and self._point_in_zone(position, self.zones[zone_name]["corners"]):
                return {"accessible": True, "reason": "in_zone"}
            if zone_name and self._nearest_distance_to_zone(position, zone_name) <= access_radius:
                return {"accessible": True, "reason": "near_zone"}
            return {"accessible": True, "reason": "distance_threshold"}

        if target.get("kind") == "build":
            table_name = target.get("object")
            if not table_name:
                return {"accessible": False, "reason": "missing_build_object"}
            if self.can_interact_with_table(position, table_name):
                if self._point_in_zone(position, self.zones[f"Zone_{table_name}"]["corners"]):
                    return {"accessible": True, "reason": "in_work_zone"}
                return {"accessible": True, "reason": "near_table"}
            return {"accessible": False, "reason": "not_in_work_zone_or_radius"}

        return {"accessible": False, "reason": "unsupported_target_kind"}

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
        """Returns a safe, role-specific spawn point not inside any blocked zone."""
        if self.task_model and self.task_model.spawn_points:
            candidates = [sp for sp in self.task_model.spawn_points if sp.enabled and (sp.role_id == role or sp.role_id == "all")]
            for sp in candidates:
                pos = (sp.x, sp.y)
                if not self.is_in_blocked_zone(pos):
                    return pos

        candidate_spawns = {
            "Architect": (7.35, 3.75),
            "Engineer": (7.35, 3.05),
            "Botanist": (7.75, 3.40)
        }
        pos = candidate_spawns.get(role, (6.0, 3.0))
        if not self.is_in_blocked_zone(pos):
            return pos
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

    def get_zone(self, pos):
        for zone_name, zone in self.zones.items():
            corners = zone.get("corners")
            if corners and self._point_in_zone(pos, corners):
                return zone_name
        return "Zone_Transition"


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
        (x_min, x_max), (y_min, y_max) = Environment().get_viewport_bounds()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
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
