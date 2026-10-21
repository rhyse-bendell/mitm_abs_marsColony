"""Participant-visible state: an explicit, role-filtered projection of the simulator for ONE human participant.

This module is READ-ONLY with respect to the simulation. It builds a JSON-able view from (a) the physical layout everybody can see, (b) what THIS agent
may legitimately observe (the same canonical context any controller receives), (c) exposure the participant has actually had (sources they opened,
messages they were shown) and (d) the contextual menus derived from the Middleware's own action affordances.

Never in a participant view (enforced by construction here, and by tests/test_human_participant_privacy.py):
    hidden construction correctness, other agents' held DIK or briefs, observer-only traces, Middleware utility scores/ranking, experimenter measures,
    controller prompts, TeamKnowledgeManager internals, derived "knowledge"/rules the simulator computed, the human's own simulated beliefs.

`participant_view/1` is the only thing a participant client ever receives. Menus carry opaque item ids; the intent behind an item stays server-side.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

from modules.action_schema import ExecutableActionType
from modules.controller_context import build_canonical_context
from modules.human_participant import ACTION_CATEGORY, HUMAN_SUBMITTABLE_ACTIONS

SCHEMA = "participant_view/1"
PILE_INTERACTION_RADIUS = 0.6
BRIDGE_INTERACTION_RADIUS = 0.9
ROLE_COLORS = {"Architect": "#e4572e", "Botanist": "#2e9e5b", "Engineer": "#2f7de1"}
DEFAULT_COLOR = "#8a8f98"
ACTION_LABELS = {
    ExecutableActionType.INSPECT_INFORMATION_SOURCE.value: "Read",
    ExecutableActionType.CONSULT_TEAM_ARTIFACT.value: "Read the whiteboard",
    ExecutableActionType.TRANSPORT_RESOURCES.value: "Carry bricks",
    ExecutableActionType.START_CONSTRUCTION.value: "Start building",
    ExecutableActionType.CONTINUE_CONSTRUCTION.value: "Continue building",
    ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value: "Repair",
    ExecutableActionType.VALIDATE_CONSTRUCTION.value: "Validate",
    ExecutableActionType.CONNECT_STRUCTURES.value: "Connect",
}
BUSY_LABELS = {
    ExecutableActionType.INSPECT_INFORMATION_SOURCE.value: "reading",
    ExecutableActionType.CONSULT_TEAM_ARTIFACT.value: "reading the whiteboard",
    ExecutableActionType.TRANSPORT_RESOURCES.value: "carrying bricks",
    ExecutableActionType.START_CONSTRUCTION.value: "building",
    ExecutableActionType.CONTINUE_CONSTRUCTION.value: "building",
    ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value: "repairing",
    ExecutableActionType.VALIDATE_CONSTRUCTION.value: "validating",
    ExecutableActionType.CONNECT_STRUCTURES.value: "connecting structures",
    ExecutableActionType.COMMUNICATE.value: "talking",
    ExecutableActionType.MOVE_TO_TARGET.value: "walking",
}


def role_color(role: str) -> str:
    return ROLE_COLORS.get(str(role), DEFAULT_COLOR)


# ------------------------------------------------------------------------------------------------------------------------------------
# Spatial reach (stateless; used by the view to say "move closer" and by the Mecha to enforce it)
# ------------------------------------------------------------------------------------------------------------------------------------
def _site_target_for_project(construction, project: Dict[str, Any]) -> Optional[str]:
    return construction.SITE_TO_BUILD_TARGET.get(project.get("site_id"))


def object_anchor(sim, object_id: str) -> Optional[Tuple[float, float]]:
    """World position of a selectable object (None if unknown)."""
    env = sim.environment
    construction = env.construction
    if object_id in construction.resource_nodes:
        return tuple(construction.resource_nodes[object_id].position)
    if object_id == "bridge_bc":
        b, c = construction.sites["site_b"].position, construction.sites["site_c"].position
        return ((b[0] + c[0]) / 2.0, (b[1] + c[1]) / 2.0)
    project = construction.projects.get(object_id)
    if isinstance(project, dict):
        loc = project.get("location")
        return tuple(loc) if loc else tuple(construction.sites[project["site_id"]].position)
    target = env.interaction_targets.get(object_id)
    if target:
        obj = env.objects.get(target.get("object") or object_id) or {}
        if obj.get("type") == "rect":
            (x, y), (w, h) = obj["position"], obj["size"]
            return (x + w / 2.0, y + h / 2.0)
        if obj.get("type") == "circle":
            return tuple(obj["position"])
    return None


def object_reach(sim, position, object_id: str) -> Dict[str, Any]:
    """{near, distance, radius} for any selectable object id (interaction target, project, pile, bridge). Purely spatial: no role/legality."""
    env = sim.environment
    construction = env.construction
    if object_id in construction.resource_nodes or object_id == "bridge_bc":
        anchor = object_anchor(sim, object_id)
        radius = PILE_INTERACTION_RADIUS if object_id != "bridge_bc" else BRIDGE_INTERACTION_RADIUS
        d = math.hypot(position[0] - anchor[0], position[1] - anchor[1])
        return {"near": d <= radius, "distance": round(d, 3), "radius": radius}
    project = construction.projects.get(object_id)
    if isinstance(project, dict):
        site_target = _site_target_for_project(construction, project)
        return env.interaction_proximity(position, site_target) if site_target else {"near": False, "distance": None, "radius": None}
    return env.interaction_proximity(position, object_id)


def resolve_move_for_object(sim, position, object_id: str) -> Dict[str, Any]:
    """Where "walk to <object>" means. Interaction targets and projects are named destinations the Environment already resolves; piles and the bridge have
    no interaction target, so the destination is a navigable point next to them (toward the participant). -> {"target_id": ..} | {"x","y"} | {}"""
    env = sim.environment
    if object_id in env.interaction_targets or object_id in env.construction.projects:
        return {"target_id": object_id}
    anchor = object_anchor(sim, object_id)
    if anchor is None:
        return {}
    dx, dy = position[0] - anchor[0], position[1] - anchor[1]
    d = math.hypot(dx, dy) or 1.0
    for off in (0.35, 0.0, 0.6, 0.9):
        for ang in (0.0, 0.6, -0.6, 1.2, -1.2, 2.4, -2.4, 3.14):
            c, s = math.cos(ang), math.sin(ang)
            ux, uy = (dx / d) * c - (dy / d) * s, (dx / d) * s + (dy / d) * c
            point = (anchor[0] + ux * off, anchor[1] + uy * off)
            if env.is_point_navigable(point):
                return {"x": round(point[0], 3), "y": round(point[1], 3)}
    return {}


# ------------------------------------------------------------------------------------------------------------------------------------
# Static layout (physical world everybody can see)
# ------------------------------------------------------------------------------------------------------------------------------------
def build_static_layout(sim) -> Dict[str, Any]:
    env = sim.environment
    xs, ys = [], []
    objects, walls = [], []
    for oid, obj in env.objects.items():
        kind = obj.get("type")
        if kind == "blocked":
            (x1, y1), (x2, y2) = obj["corners"]
            walls.append({"x": min(x1, x2), "y": min(y1, y2), "w": abs(x2 - x1), "h": abs(y2 - y1)})
            xs += [x1, x2]
            ys += [y1, y2]
        elif kind == "rect":
            (x, y), (w, h) = obj["position"], obj["size"]
            objects.append({"id": oid, "shape": "rect", "label": obj.get("label", oid), "x": x, "y": y, "w": w, "h": h})
        elif kind == "circle":
            (x, y) = obj["position"]
            objects.append({"id": oid, "shape": "circle", "label": obj.get("label", oid), "x": x, "y": y, "r": obj["radius"]})
        elif kind == "line":
            objects.append({"id": oid, "shape": "line", "label": obj.get("label", oid), "x1": obj["start"][0], "y1": obj["start"][1], "x2": obj["end"][0], "y2": obj["end"][1]})
    width = max(xs) if xs else 10.0
    height = max(ys) if ys else 7.4
    return {"bounds": [round(width, 3), round(height, 3)], "objects": objects, "walls": walls}


# ------------------------------------------------------------------------------------------------------------------------------------
# Menus derived from the Middleware's affordances
# ------------------------------------------------------------------------------------------------------------------------------------
def _intent_key(action: str, target_id: Optional[str], args: Dict[str, Any], ui_object: str = "") -> str:
    """Stable opaque id of a menu option. The object it was offered from is part of the identity (the same carry action is offered at the pile and at the project)."""
    raw = f"{ui_object}|{action}|{target_id or ''}|{sorted((args or {}).items())}"
    return "m" + hashlib.sha1(raw.encode()).hexdigest()[:10]


def _item(label: str, action: str, target_id: Optional[str], args: Dict[str, Any], *, enabled: bool = True, reason: Optional[str] = None, ui_object: str = "", detail: str = "") -> Dict[str, Any]:
    key = _intent_key(action, target_id, args, ui_object)
    return {"public": {"id": key, "label": label, "enabled": bool(enabled), "reason": reason, "detail": detail},
            "intent": {"action": action, "target_id": target_id, "action_args": dict(args or {}), "ui_object": ui_object, "label": label}}


def _project_label(pid: str, project_types: Dict[str, str]) -> str:
    kind = str(project_types.get(pid) or "structure").replace("_", " ")
    suffix = pid.rsplit("_", 1)[-1]
    return f"{kind} #{suffix}"


def build_menus(sim, agent, canonical: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """{object_id: {"title", "kind", "items": [item,...]}} where every actionable item mirrors a Middleware affordance (or, for another role's packet and
    for not-ready validate/repair, the advertised-but-blocked attempt). No second list of game actions lives here."""
    env = sim.environment
    construction = env.construction
    executable = [a for a in canonical.get("executable_actions", []) if a.get("action_type") in HUMAN_SUBMITTABLE_ACTIONS]
    blocked = [a for a in canonical.get("blocked_actions", []) if a.get("action_type") in HUMAN_SUBMITTABLE_ACTIONS]
    types = {pid: p.get("type") for pid, p in construction.projects.items() if isinstance(p, dict)}
    menus: Dict[str, Dict[str, Any]] = {}

    def menu(oid: str, title: str, kind: str) -> List[Dict[str, Any]]:
        return menus.setdefault(oid, {"title": title, "kind": kind, "items": []})["items"]

    held_sources = set(getattr(agent, "memory_seen_packets", set()) or set())
    # ---- information packets and the whiteboard
    for tid, target in env.interaction_targets.items():
        kind = target.get("kind")
        if kind == "information":
            obj = env.objects.get(target.get("object") or tid) or {}
            is_shared = env.is_shared_information_source(tid)
            own = tid == f"{agent.role}_Info"
            label = obj.get("label", tid)
            items = menu(tid, label, "info_shared" if is_shared else ("info_own" if own else "info_other"))
            if tid in held_sources:
                items.append({"public": {"id": f"reopen:{tid}", "label": "Reopen in Information panel", "enabled": True, "reason": None, "detail": "You have already read this."}, "intent": None})
            else:
                # Offered for every packet, including another role's: the attempt goes through the gate, which refuses it (the physical metaphor
                # allows trying; nothing is opened in its place).
                items.append(_item("Read the team brief" if is_shared else "Read briefing", ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, tid, {}, ui_object=tid))
        elif kind == "artifact":
            items = menu(tid, "Whiteboard", "artifact")
            if any(a.get("action_type") == ExecutableActionType.CONSULT_TEAM_ARTIFACT.value for a in executable):
                items.append(_item("Read the whiteboard", ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, "whiteboard", {}, ui_object=tid))
        elif kind == "build":
            site_id = target.get("site_id") or construction.PROJECT_TO_SITE.get(tid)
            site_label = f"Site {str(site_id or tid)[-1].upper()}"
            items = menu(tid, site_label, "site")
            for aff in executable:
                if aff.get("action_type") == ExecutableActionType.START_CONSTRUCTION.value and aff.get("target_id") == tid and aff.get("start_mode") == "new_project":
                    stype = str((aff.get("action_args") or {}).get("structure_type") or "structure")
                    need = (aff.get("required_resources") or {}).get("bricks")
                    items.append(_item(f"Start a {stype.replace('_', ' ')}", aff["action_type"], tid, dict(aff.get("action_args") or {}), ui_object=tid, detail=f"needs {need} bricks" if need is not None else ""))
    # ---- projects
    for pid, project in sorted(construction.projects.items()):
        if not isinstance(project, dict) or str(project.get("status") or "") == "complete" and project.get("validated_complete"):
            continue
        items = menu(pid, _project_label(pid, types), "project")
        for aff in executable + blocked:
            args = dict(aff.get("action_args") or {})
            at = aff.get("action_type")
            if at == ExecutableActionType.CONNECT_STRUCTURES.value:
                if args.get("connector_project_id") != pid:
                    continue
                a, b = args.get("endpoint_a_id"), args.get("endpoint_b_id")
                items.append(_item(f"Connect {_project_label(a, types)} and {_project_label(b, types)}", at, None, args, ui_object=pid))
                continue
            if aff.get("target_id") != pid or aff.get("start_mode") == "new_project":
                continue
            ready = aff in executable
            label = {ExecutableActionType.TRANSPORT_RESOURCES.value: "Bring bricks to this project", ExecutableActionType.START_CONSTRUCTION.value: "Begin building",
                     ExecutableActionType.CONTINUE_CONSTRUCTION.value: "Continue building", ExecutableActionType.VALIDATE_CONSTRUCTION.value: "Validate this structure",
                     ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value: "Repair this structure"}.get(at, ACTION_LABELS.get(at, at))
            items.append(_item(label, at, pid, args, enabled=ready, reason=None if ready else "Not ready yet.", ui_object=pid))
    # ---- bridge
    bridge_items = menu("bridge_bc", "Bridge (Site B to Site C)", "bridge")
    for aff in executable:
        if aff.get("target_id") == "bridge_bc":
            label = "Bring bricks to the bridge" if aff["action_type"] == ExecutableActionType.TRANSPORT_RESOURCES.value else "Build the bridge"
            bridge_items.append(_item(label, aff["action_type"], "bridge_bc", dict(aff.get("action_args") or {}), ui_object="bridge_bc"))
    # ---- resource piles: the carry action is offered where the bricks are
    carry = [a for a in executable if a.get("action_type") == ExecutableActionType.TRANSPORT_RESOURCES.value]
    for pile_id in construction.resource_nodes:
        items = menu(pile_id, "Brick pile", "pile")
        for aff in carry:
            tid = aff.get("target_id")
            dest = "the bridge" if tid == "bridge_bc" else _project_label(str(tid), types)
            items.append(_item(f"Carry bricks to {dest}", aff["action_type"], tid, dict(aff.get("action_args") or {}), ui_object=pile_id))
    return {k: v for k, v in menus.items() if v["items"] or v["kind"] in {"pile", "info_shared", "info_own", "info_other", "artifact", "site", "project", "bridge"}}


# ------------------------------------------------------------------------------------------------------------------------------------
# The view
# ------------------------------------------------------------------------------------------------------------------------------------
def _phase_info(sim) -> Dict[str, Any]:
    env = sim.environment
    phases = list(env.phases or [])
    total = sum(float(p.get("duration_minutes", 0.0) or 0.0) * 60.0 for p in phases)
    cur = env.get_current_phase() or {"name": "default"}
    start = 0.0
    for idx, p in enumerate(phases):
        end = start + float(p.get("duration_minutes", 0.0) or 0.0) * 60.0
        if idx == env.current_phase_index:
            return {"name": cur.get("name", "default"), "index": idx, "remaining_s": max(0.0, round(end - sim.time, 1)), "total_s": total}
        start = end
    return {"name": cur.get("name", "default"), "index": env.current_phase_index, "remaining_s": 0.0, "total_s": total}


def _teammate_activity(sim, other) -> str:
    if sim.environment.construction.is_agent_transporting(other.name):
        return "carrying bricks"
    for a in other.active_actions:
        label = BUSY_LABELS.get(a.get("decision_action"))
        if label and a.get("decision_action") != ExecutableActionType.MOVE_TO_TARGET.value:
            return label
    return "walking" if other.target else "idle"


def _exposed_items(agent, packet_name: str, packet: Dict[str, Any]) -> List[Dict[str, str]]:
    """Items of a packet this participant has actually been exposed to (direct/shared/artifact acquisition), as text. Derived items are excluded."""
    held = {getattr(o, "id", None) for o in list(agent.mental_model["data"]) + list(agent.mental_model["information"])}
    rows = []
    for bucket, kind in (("data", "data"), ("information", "information")):
        for item in packet.get(bucket, []) or []:
            if item.id in held and (item.acquired_by.get(agent.name, {}) or {}).get("mode") in {"direct", "shared", "artifact"}:
                rows.append({"id": item.id, "kind": kind, "text": str(item.content)})
    return rows


def build_participant_view(sim, agent, controller, telemetry=None, *, seq: int = 0, paused: bool = False, menu_registry: Optional[Dict[str, Dict[str, Any]]] = None, last_feedback_id: int = 0) -> Dict[str, Any]:
    """The complete participant-visible state. `menu_registry` (if given) is filled with {item_id: intent} for server-side lookup of selected items."""
    env = sim.environment
    construction = env.construction
    pos = (float(agent.position[0]), float(agent.position[1]))
    packet = sim.brain_context_builder.build(sim, agent)
    canonical = build_canonical_context(packet, agent_id=agent.agent_id, role=agent.role, controller_type="human_participant")
    menus = build_menus(sim, agent, canonical)
    built = {b["structure_id"]: b for b in (packet.world_snapshot.get("built_state") or [])}
    types = {pid: p.get("type") for pid, p in construction.projects.items() if isinstance(p, dict)}

    objects: List[Dict[str, Any]] = []
    for oid, m in menus.items():
        anchor = object_anchor(sim, oid)
        if anchor is None:
            continue
        reach = object_reach(sim, pos, oid)
        near = bool(reach.get("near"))
        row: Dict[str, Any] = {"id": oid, "kind": m["kind"], "title": m["title"], "x": round(anchor[0], 3), "y": round(anchor[1], 3), "near": near,
                               "radius": reach.get("radius"), "hint": "Click to interact" if near else "Move closer to interact"}
        if m["kind"] == "pile":
            node = construction.resource_nodes[oid]
            if near:
                row["status"] = "stocked" if int(node.quantity) > 0 else "empty"
        if m["kind"] == "project":
            b = built.get(oid, {})
            row.update({"structure_type": str(types.get(oid) or "structure"), "status": str(b.get("project_status") or "in_progress"), "progress": b.get("progress", 0.0),
                        "materials": f"{b.get('delivered_resources', 0)}/{b.get('required_resources', 0)}", "validation_failed": bool(b.get("validation_failed"))})
        if m["kind"] == "bridge":
            row["status"] = str(construction.bridges["bridge_bc"].status)
        if m["kind"] == "site":
            site_id = env.interaction_targets.get(oid, {}).get("site_id")
            site = construction.sites.get(site_id)
            row["buildable"] = bool(site.buildable) if site else True
        items = []
        for it in m["items"]:
            pub = dict(it["public"])
            items.append(pub)
            if menu_registry is not None and it["intent"] is not None:
                menu_registry[pub["id"]] = it["intent"]
        row["items"] = items
        objects.append(row)

    sources = []
    seen_sources = list(getattr(agent, "memory_seen_packets", []) or [])
    open_ids = set(telemetry.open_source_ids()) if telemetry is not None else set()
    for sid in seen_sources:
        pk = env.knowledge_packets.get(sid) if isinstance(sid, str) else None
        if not pk:
            continue
        items = _exposed_items(agent, sid, pk)
        if items:
            target = env.interaction_targets.get(sid, {})
            label = (env.objects.get(target.get("object") or sid) or {}).get("label", sid)
            sources.append({"id": sid, "label": label, "scope": "shared" if env.is_shared_information_source(sid) else "own role", "open": sid in open_ids, "items": items})

    chat = []
    for rec in sim.chat_log[-80:]:
        if rec["kind"] == "human_chat" and (rec["sender"] == agent.name or agent.name in rec["recipients"]) or rec["kind"] == "agent_message" and agent.name in rec["recipients"]:
            chat.append({"id": rec["id"], "t": rec["time"], "sender": rec["sender"], "text": rec["text"], "mine": rec["sender"] == agent.name, "kind": rec["kind"]})

    committed = agent._human_committed_action()
    busy = None
    if committed is not None:
        left = max(0.0, float(committed.get("duration", 0.0) or 0.0) - float(committed.get("progress", 0.0) or 0.0))
        busy = {"label": BUSY_LABELS.get(committed.get("decision_action"), "working"), "remaining_s": round(left, 1)}
    carrying = int(((getattr(agent, "transport_state", {}) or {}).get("carrying") or {}).get("quantity", 0) or 0)
    phase = _phase_info(sim)
    ended = bool(getattr(sim, "terminal_state", None))
    return {
        "schema": SCHEMA, "seq": seq,
        "you": {"agent_id": agent.agent_id, "name": agent.name, "role": agent.role, "color": role_color(agent.role), "x": round(pos[0], 3), "y": round(pos[1], 3),
                "busy": busy, "moving": [round(agent.target[0], 3), round(agent.target[1], 3)] if agent.target else None, "carrying": carrying},
        "mission": {"sim_time": round(float(sim.time), 1), "phase": phase["name"], "phase_remaining_s": phase["remaining_s"], "total_s": phase["total_s"],
                    "status": "ended" if ended else ("paused" if paused else "running"), "ended": ended},
        "agents": [{"name": o.name, "role": o.role, "color": role_color(o.role), "x": round(float(o.position[0]), 3), "y": round(float(o.position[1]), 3), "activity": _teammate_activity(sim, o)}
                   for o in sim.agents if o is not agent],
        "objects": objects,
        "bridge_bc_visible": str(construction.bridges["bridge_bc"].status) in {"in_progress", "complete"},
        "info": {"sources": sources},
        "chat": chat,
        "feedback": controller.feedback_since(last_feedback_id)[-20:],
    }
