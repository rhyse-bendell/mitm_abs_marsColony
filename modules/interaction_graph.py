from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

NODE_GROUP_COLORS = {
    "Agents": "#4C78A8",
    "Planner/LLM": "#F58518",
    "Sources / Knowledge": "#54A24B",
    "Derivation / Rules": "#E45756",
    "Environment / Movement": "#72B7B2",
    "Construction / Externalization": "#B279A2",
    "Runtime Witness Audit": "#FF9DA6",
    "Metrics / Logger / Phase": "#9D755D",
}

CANONICAL_NODES: Dict[str, Dict[str, Any]] = {
    "Agent:Architect": {"group": "Agents", "label": "Architect", "pos": (0.06, 0.84)},
    "Agent:Engineer": {"group": "Agents", "label": "Engineer", "pos": (0.06, 0.70)},
    "Agent:Botanist": {"group": "Agents", "label": "Botanist", "pos": (0.06, 0.56)},
    "Planner:OllamaQwen": {"group": "Planner/LLM", "label": "Ollama/Qwen", "pos": (0.30, 0.84)},
    "Planner:RuleBrain": {"group": "Planner/LLM", "label": "Rule Brain", "pos": (0.30, 0.68)},
    "Bootstrap:StartupSanity": {"group": "Planner/LLM", "label": "Startup Sanity", "pos": (0.30, 0.52)},
    "Source:TeamInfo": {"group": "Sources / Knowledge", "label": "Team Source", "pos": (0.50, 0.90)},
    "Source:ArchitectInfo": {"group": "Sources / Knowledge", "label": "Architect Source", "pos": (0.50, 0.78)},
    "Source:EngineerInfo": {"group": "Sources / Knowledge", "label": "Engineer Source", "pos": (0.50, 0.66)},
    "Source:BotanistInfo": {"group": "Sources / Knowledge", "label": "Botanist Source", "pos": (0.50, 0.54)},
    "Knowledge:AgentMentalModel": {"group": "Sources / Knowledge", "label": "Agent Mental Model", "pos": (0.50, 0.40)},
    "Knowledge:TeamKnowledgeStore": {"group": "Sources / Knowledge", "label": "Team Knowledge", "pos": (0.50, 0.28)},
    "Knowledge:DerivationEngine": {"group": "Derivation / Rules", "label": "Derivation Engine", "pos": (0.72, 0.74)},
    "Knowledge:RuleStore": {"group": "Derivation / Rules", "label": "Rule Store", "pos": (0.72, 0.60)},
    "World:Environment": {"group": "Environment / Movement", "label": "Environment", "pos": (0.72, 0.42)},
    "World:MovementPathing": {"group": "Environment / Movement", "label": "Pathing", "pos": (0.72, 0.30)},
    "World:ConstructionProjects": {"group": "Construction / Externalization", "label": "Construction", "pos": (0.90, 0.56)},
    "Audit:RuntimeWitness": {"group": "Runtime Witness Audit", "label": "Runtime Witness", "pos": (0.90, 0.78)},
    "System:Metrics": {"group": "Metrics / Logger / Phase", "label": "Metrics", "pos": (0.90, 0.34)},
    "System:Logger": {"group": "Metrics / Logger / Phase", "label": "Logger", "pos": (0.90, 0.22)},
    "System:PhaseManager": {"group": "Metrics / Logger / Phase", "label": "Phase Manager", "pos": (0.90, 0.10)},
}

CANONICAL_INTERACTION_TYPES = {
    "source_access",
    "dik_acquisition",
    "derivation_attempt",
    "derivation_success",
    "rule_adoption",
    "planner_request",
    "planner_response",
    "fallback_activation",
    "movement_plan",
    "movement_progress",
    "construction_update",
    "witness_block",
    "witness_recover",
    "metric_emit",
    "phase_transition",
}


@dataclass
class InteractionEvent:
    time: float
    interaction_id: str
    source_node: str
    target_node: str
    interaction_type: str
    status: str
    agent_id: Optional[str] = None
    payload_summary: str = ""
    category: Optional[str] = None
    severity: Optional[str] = None

    def to_row(self) -> Dict[str, Any]:
        row = {
            "time": round(float(self.time), 3),
            "interaction_id": self.interaction_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "interaction_type": self.interaction_type,
            "status": self.status,
            "agent_id": self.agent_id,
            "payload_summary": self.payload_summary,
            "category": self.category,
            "severity": self.severity,
        }
        return row


def canonical_node(node_id: str) -> str:
    if node_id in CANONICAL_NODES:
        return node_id
    if node_id.startswith("Agent:"):
        return node_id
    return "System:Logger"


def agent_node(agent_name: Optional[str]) -> str:
    if not agent_name:
        return "Agent:Unknown"
    return f"Agent:{str(agent_name)}"


def _summary(payload: Dict[str, Any], keys: Iterable[str]) -> str:
    out: List[str] = []
    for key in keys:
        value = payload.get(key)
        if value in (None, "", [], {}):
            continue
        out.append(f"{key}={value}")
    text = ", ".join(out)
    return text[:220]


def parse_payload_maybe(event_row: Dict[str, Any]) -> Dict[str, Any]:
    payload = event_row.get("payload_data")
    if isinstance(payload, dict):
        return payload
    payload = event_row.get("payload")
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str) and payload.strip():
        try:
            return json.loads(payload)
        except Exception:
            return {}
    return {}


def build_interaction_from_sim_event(time: float, event_type: str, payload: Dict[str, Any]) -> Optional[InteractionEvent]:
    payload = dict(payload or {})
    agent_id = str(payload.get("agent_id") or payload.get("agent") or "") or None
    src_agent = agent_node(payload.get("agent"))
    event_type = str(event_type or "")
    interaction_id = str(payload.get("trace_id") or payload.get("request_id") or payload.get("event_id") or f"{event_type}:{round(float(time), 3)}")

    if event_type in {"inspect_started", "target_resolved", "shared_source_inspect_started"}:
        source_id = payload.get("source_id") or payload.get("target_id") or "TeamInfo"
        target = f"Source:{str(source_id).replace('_', '')}"
        return InteractionEvent(time, interaction_id, src_agent, canonical_node(target), "source_access", "started", agent_id, _summary(payload, ["source_id", "goal"]))

    if event_type in {"dik_acquired", "shared_source_dik_acquired_agent", "shared_source_dik_acquired_team"}:
        return InteractionEvent(time, interaction_id, "Source:TeamInfo", "Knowledge:AgentMentalModel", "dik_acquisition", "completed", agent_id, _summary(payload, ["source_id", "new_information_ids", "new_rule_ids"]))

    if event_type in {"derivation_attempted"}:
        return InteractionEvent(time, interaction_id, "Knowledge:AgentMentalModel", "Knowledge:DerivationEngine", "derivation_attempt", "started", agent_id, _summary(payload, ["derivation_id", "rule_id"]))

    if event_type in {"derivation_succeeded"}:
        return InteractionEvent(time, interaction_id, "Knowledge:DerivationEngine", "Knowledge:RuleStore", "derivation_success", "completed", agent_id, _summary(payload, ["derivation_id", "output_id"]))

    if event_type in {"rule_adopted"}:
        return InteractionEvent(time, interaction_id, "Knowledge:RuleStore", src_agent, "rule_adoption", "completed", agent_id, _summary(payload, ["rule_id", "adoption_mode"]))

    if event_type in {"planner_request_started_async", "brain_provider_request_started", "planner_invocation_requested"}:
        planner_target = "Planner:OllamaQwen"
        if str(payload.get("configured_backend") or "").lower() == "rule_brain":
            planner_target = "Planner:RuleBrain"
        return InteractionEvent(time, interaction_id, src_agent, planner_target, "planner_request", "started", agent_id, _summary(payload, ["request_id", "backend", "trigger_reason"]))

    if event_type in {"planner_result_applied", "brain_decision_outcome"}:
        planner_source = "Planner:OllamaQwen"
        if str(payload.get("effective_brain_backend") or "").lower() == "rule_brain":
            planner_source = "Planner:RuleBrain"
        return InteractionEvent(time, interaction_id, planner_source, src_agent, "planner_response", "completed", agent_id, _summary(payload, ["selected_action", "decision_status"]))

    if event_type in {"brain_provider_fallback", "ui_safe_fallback_used"}:
        return InteractionEvent(time, interaction_id, "Planner:OllamaQwen", "Planner:RuleBrain", "fallback_activation", "failed", agent_id, _summary(payload, ["reason", "fallback_provider"]), severity="warning")

    if event_type in {"movement_started", "movement_plan_created"}:
        return InteractionEvent(time, interaction_id, src_agent, "World:MovementPathing", "movement_plan", "started", agent_id, _summary(payload, ["destination", "path_mode"]))

    if event_type in {"movement_progress", "movement_arrived", "movement_failed", "movement_blocked"}:
        status = "progressed"
        if event_type == "movement_arrived":
            status = "completed"
        if event_type in {"movement_failed", "movement_blocked"}:
            status = "failed"
        return InteractionEvent(time, interaction_id, "World:MovementPathing", "World:Environment", "movement_progress", status, agent_id, _summary(payload, ["destination", "failure_category", "blocker_category"]))

    if event_type.startswith("construction_") or event_type in {"externalization_created", "construction_externalization_update"}:
        return InteractionEvent(time, interaction_id, src_agent, "World:ConstructionProjects", "construction_update", "progressed", agent_id, _summary(payload, ["project_id", "status", "progress", "structure_type"]))

    if event_type in {"witness_expectation_failed", "source_access_blocked", "runtime_witness_audit_saved"}:
        return InteractionEvent(time, interaction_id, "Audit:RuntimeWitness", src_agent, "witness_block", "failed", agent_id, _summary(payload, ["target_id", "failure_category", "step_type"]))

    if event_type in {"witness_expectation_recovered", "witness_step_recovered", "source_access_recovered", "witness_step_recovered_after_late_success"}:
        return InteractionEvent(time, interaction_id, "Audit:RuntimeWitness", src_agent, "witness_recover", "completed", agent_id, _summary(payload, ["target_id", "step_type"]))

    if event_type in {"metrics_snapshot", "outputs_saved"}:
        return InteractionEvent(time, interaction_id, "System:Metrics", "System:Logger", "metric_emit", "completed", agent_id, _summary(payload, ["rows", "event_rows", "phase"]))

    if event_type in {"phase_transition"}:
        return InteractionEvent(time, interaction_id, "System:PhaseManager", "World:Environment", "phase_transition", "completed", agent_id, _summary(payload, ["from_phase", "to_phase"]))

    return None


def render_interaction_graph(ax, events: List[Dict[str, Any]], now_time: Optional[float] = None, window_s: float = 8.0):
    import matplotlib.patches as mpatches
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Interaction Graph")

    recent = list(events)
    if now_time is not None:
        recent = [e for e in recent if now_time - float(e.get("time", 0.0)) <= window_s and float(e.get("time", 0.0)) <= now_time]

    active_nodes = set()
    active_edges: Dict[Tuple[str, str], str] = {}
    for event in recent:
        s = canonical_node(str(event.get("source_node") or "System:Logger"))
        t = canonical_node(str(event.get("target_node") or "System:Logger"))
        active_nodes.update({s, t})
        active_edges[(s, t)] = str(event.get("status") or "progressed")

    for (source, target), status in active_edges.items():
        s_meta = CANONICAL_NODES.get(source)
        t_meta = CANONICAL_NODES.get(target)
        if not s_meta or not t_meta:
            continue
        sx, sy = s_meta["pos"]
        tx, ty = t_meta["pos"]
        color = "#2ca02c" if status in {"completed", "progressed"} else "#d62728"
        ax.annotate("", xy=(tx, ty), xytext=(sx, sy), arrowprops={"arrowstyle": "->", "color": color, "lw": 2.0, "alpha": 0.8})

    for node_id, meta in CANONICAL_NODES.items():
        x, y = meta["pos"]
        color = NODE_GROUP_COLORS.get(meta["group"], "#999999")
        edge = "#222222" if node_id in active_nodes else "#bbbbbb"
        size = 0.018 if node_id not in active_nodes else 0.026
        ax.add_patch(mpatches.Circle((x, y), size, facecolor=color, edgecolor=edge, linewidth=2.0))
        ax.text(x + 0.022, y, meta["label"], fontsize=7, va="center")

    if recent:
        newest = recent[-1]
        ax.text(0.01, 0.02, f"Recent: {newest.get('interaction_type')} {newest.get('source_node')} -> {newest.get('target_node')}", fontsize=7)
