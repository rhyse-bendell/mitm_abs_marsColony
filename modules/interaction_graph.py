from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class InteractionNode:
    node_id: str
    label: str
    group: str
    pos: Tuple[float, float]


CANONICAL_NODES: List[InteractionNode] = [
    InteractionNode("Agent:Architect", "Architect", "Agents", (0.10, 0.20)),
    InteractionNode("Agent:Engineer", "Engineer", "Agents", (0.10, 0.45)),
    InteractionNode("Agent:Botanist", "Botanist", "Agents", (0.10, 0.70)),
    InteractionNode("Planner:OllamaQwen", "OllamaQwen", "Planner/LLM", (0.35, 0.20)),
    InteractionNode("Planner:RuleBrain", "RuleBrain", "Planner/LLM", (0.35, 0.35)),
    InteractionNode("Bootstrap:StartupSanity", "StartupSanity", "Planner/LLM", (0.35, 0.50)),
    InteractionNode("Source:TeamInfo", "TeamInfo", "Sources / Knowledge", (0.35, 0.70)),
    InteractionNode("Source:ArchitectInfo", "ArchitectInfo", "Sources / Knowledge", (0.35, 0.82)),
    InteractionNode("Source:EngineerInfo", "EngineerInfo", "Sources / Knowledge", (0.52, 0.82)),
    InteractionNode("Source:BotanistInfo", "BotanistInfo", "Sources / Knowledge", (0.68, 0.82)),
    InteractionNode("Knowledge:AgentMentalModel", "AgentMentalModel", "Derivation / Rules", (0.58, 0.20)),
    InteractionNode("Knowledge:TeamKnowledgeStore", "TeamKnowledgeStore", "Derivation / Rules", (0.58, 0.35)),
    InteractionNode("Knowledge:DerivationEngine", "DerivationEngine", "Derivation / Rules", (0.58, 0.50)),
    InteractionNode("Knowledge:RuleStore", "RuleStore", "Derivation / Rules", (0.58, 0.65)),
    InteractionNode("World:MovementPathing", "MovementPathing", "Environment / Movement", (0.82, 0.20)),
    InteractionNode("World:Environment", "Environment", "Environment / Movement", (0.82, 0.35)),
    InteractionNode("World:ConstructionProjects", "ConstructionProjects", "Construction / Externalization", (0.82, 0.55)),
    InteractionNode("Audit:RuntimeWitness", "RuntimeWitness", "Runtime Witness Audit", (0.82, 0.75)),
    InteractionNode("System:Metrics", "Metrics", "Metrics / Logger / Phase", (0.58, 0.94)),
    InteractionNode("System:Logger", "Logger", "Metrics / Logger / Phase", (0.76, 0.94)),
    InteractionNode("System:PhaseManager", "PhaseManager", "Metrics / Logger / Phase", (0.90, 0.94)),
]

NODE_BY_ID = {n.node_id: n for n in CANONICAL_NODES}

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
}


def canonical_agent_node(agent_or_role: Optional[str]) -> str:
    role = str(agent_or_role or "Agent").split(":")[-1]
    role = role.replace(" ", "")
    candidate = f"Agent:{role}"
    if candidate in NODE_BY_ID:
        return candidate
    return "Agent:Architect"


def source_node_from_id(source_id: Optional[str]) -> str:
    sid = str(source_id or "Team_Info")
    mapping = {
        "Team_Info": "Source:TeamInfo",
        "Architect_Info": "Source:ArchitectInfo",
        "Engineer_Info": "Source:EngineerInfo",
        "Botanist_Info": "Source:BotanistInfo",
    }
    return mapping.get(sid, "Source:TeamInfo")


def make_interaction_event(
    *,
    time: float,
    interaction_id: str,
    source_node: str,
    target_node: str,
    interaction_type: str,
    status: str,
    agent_id: Optional[str] = None,
    payload_summary: str = "",
    category: Optional[str] = None,
    severity: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "time": round(float(time), 3),
        "interaction_id": str(interaction_id),
        "source_node": source_node,
        "target_node": target_node,
        "interaction_type": interaction_type,
        "status": status,
        "agent_id": agent_id,
        "payload_summary": payload_summary,
        "category": category,
        "severity": severity,
    }


class InteractionTraceWriter:
    def __init__(self, path: Path):
        self.path = path

    def append(self, row: Dict[str, Any]):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


class InteractionTelemetryBridge:
    def __init__(self, logger):
        self.logger = logger
        self.counter = 0

    def _next_id(self, event: Dict[str, Any]) -> str:
        self.counter += 1
        return f"itx-{int(float(event.get('time', 0))*1000)}-{self.counter}"

    def on_event(self, event: Dict[str, Any]):
        et = str(event.get("event_type", ""))
        payload = event.get("payload_data", {}) or {}
        agent = payload.get("agent") or payload.get("role") or payload.get("agent_id")
        src = canonical_agent_node(agent)
        t = float(event.get("time", 0.0) or 0.0)

        mapped: Optional[Dict[str, Any]] = None
        if et in {"inspect_started", "shared_source_inspect_started"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node=src, target_node=source_node_from_id(payload.get("source_id") or payload.get("target_id")), interaction_type="source_access", status="started", agent_id=payload.get("agent_id"), payload_summary=f"inspect {payload.get('source_id') or payload.get('target_id')}")
        elif et in {"inspect_progressed"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node=source_node_from_id(payload.get("source_id") or payload.get("target_id")), target_node="Knowledge:AgentMentalModel", interaction_type="dik_acquisition", status="progressed", agent_id=payload.get("agent_id"), payload_summary="DIK progressing")
        elif et in {"inspect_completed", "shared_source_inspect_completed"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node=source_node_from_id(payload.get("source_id") or payload.get("target_id")), target_node="Knowledge:AgentMentalModel", interaction_type="dik_acquisition", status="completed", agent_id=payload.get("agent_id"), payload_summary="DIK acquired")
        elif et in {"derivation_succeeded"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="Knowledge:DerivationEngine", target_node="Knowledge:RuleStore", interaction_type="derivation_success", status="completed", agent_id=payload.get("agent_id"), payload_summary=str(payload.get("rule_id") or "rule derived"))
        elif et in {"rule_adopted"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="Knowledge:RuleStore", target_node=src, interaction_type="rule_adoption", status="completed", agent_id=payload.get("agent_id"), payload_summary=str(payload.get("rule_id") or "rule adopted"))
        elif et in {"planner_invocation_requested", "planner_request_started_async", "brain_provider_request_started"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node=src, target_node="Planner:OllamaQwen", interaction_type="planner_request", status="started", agent_id=payload.get("agent_id"), payload_summary="planner request")
        elif et in {"planner_invocation_completed", "planner_request_completed_async", "brain_provider_response_received", "llm_response_received"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="Planner:OllamaQwen", target_node=src, interaction_type="planner_response", status="completed", agent_id=payload.get("agent_id"), payload_summary="planner response")
        elif et in {"brain_provider_fallback", "fallback_result_adopted", "ui_safe_fallback_used"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="Planner:OllamaQwen", target_node="Planner:RuleBrain", interaction_type="fallback_activation", status="failed", agent_id=payload.get("agent_id"), payload_summary=str(payload.get("reason") or "fallback"), severity="warning")
        elif et in {"movement_between_knowledge_locations", "moving_to_shared_source", "moving_to_externalization_site"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node=src, target_node="World:MovementPathing", interaction_type="movement_plan", status="started", agent_id=payload.get("agent_id"), payload_summary="movement plan")
        elif et in {"movement_progress", "movement_retried"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="World:MovementPathing", target_node="World:Environment", interaction_type="movement_progress", status="progressed", agent_id=payload.get("agent_id"), payload_summary="movement progress")
        elif et in {"construction_progress_updated", "construction_externalization_updated", "construction_resource_delivered", "construction_completed"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node=src, target_node="World:ConstructionProjects", interaction_type="construction_update", status="progressed", agent_id=payload.get("agent_id"), payload_summary=str(payload.get("project_id") or "construction"))
        elif et in {"source_access_blocked", "witness_step_blocked", "witness_expectation_failed"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="Audit:RuntimeWitness", target_node=src, interaction_type="witness_block", status="failed", agent_id=payload.get("agent_id"), payload_summary=str(payload.get("failure_category") or "witness block"), severity="warning")
        elif et in {"source_access_recovered", "witness_step_recovered", "witness_expectation_recovered"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="Audit:RuntimeWitness", target_node=src, interaction_type="witness_recover", status="completed", agent_id=payload.get("agent_id"), payload_summary="witness recovered")
        elif et in {"outputs_saved", "phase_transition", "brain_backend_runtime_status"}:
            mapped = make_interaction_event(time=t, interaction_id=self._next_id(event), source_node="System:Metrics", target_node="System:Logger", interaction_type="metric_emit", status="completed", payload_summary=et)

        if mapped:
            self.logger.log_interaction(mapped)
