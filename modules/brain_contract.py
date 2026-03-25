from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

from modules.action_schema import BrainDecision, ExecutableActionType


@dataclass
class AgentGoal:
    goal_id: str
    description: str
    priority: float
    status: str
    parent_goal_id: Optional[str] = None
    evidence: Optional[str] = None
    horizon: Optional[str] = None
    source: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentGoal":
        return cls(
            goal_id=str(payload.get("goal_id", "")),
            description=str(payload.get("description", "")),
            priority=float(payload.get("priority", 0.0)),
            status=str(payload.get("status", "active")),
            parent_goal_id=payload.get("parent_goal_id"),
            evidence=payload.get("evidence"),
            horizon=payload.get("horizon"),
            source=payload.get("source"),
        )


@dataclass
class PlannedActionStep:
    step_index: int
    action_type: ExecutableActionType
    target_id: Optional[str] = None
    target_zone: Optional[str] = None
    expected_purpose: str = ""
    preconditions: List[str] = field(default_factory=list)
    expected_outcome: Optional[str] = None
    fallback_if_blocked: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PlannedActionStep":
        action = payload.get("action_type", ExecutableActionType.WAIT.value)
        return cls(
            step_index=int(payload.get("step_index", 0)),
            action_type=ExecutableActionType(action),
            target_id=payload.get("target_id") or payload.get("target"),
            target_zone=payload.get("target_zone") or payload.get("location"),
            expected_purpose=str(payload.get("expected_purpose", "")),
            preconditions=list(payload.get("preconditions", [])),
            expected_outcome=payload.get("expected_outcome"),
            fallback_if_blocked=payload.get("fallback_if_blocked"),
        )

    def to_brain_decision(self, confidence: float, plan_method_id: Optional[str] = None, next_steps: Optional[List[str]] = None) -> BrainDecision:
        return BrainDecision(
            selected_action=self.action_type,
            target_id=self.target_id,
            target_zone=self.target_zone,
            reason_summary=self.expected_purpose,
            confidence=confidence,
            plan_method_id=plan_method_id,
            next_steps=list(next_steps or []),
        )


@dataclass
class AgentPlan:
    plan_id: str
    plan_horizon: int
    ordered_goals: List[AgentGoal]
    ordered_actions: List[PlannedActionStep]
    next_action: PlannedActionStep
    confidence: float
    replan_conditions: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    uncertainty_flags: List[str] = field(default_factory=list)
    plan_method_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentPlan":
        actions = [PlannedActionStep.from_dict(a) for a in payload.get("ordered_actions", [])]
        if not actions:
            actions = [
                PlannedActionStep(
                    step_index=0,
                    action_type=ExecutableActionType.WAIT,
                    expected_purpose="fallback wait due to empty plan",
                )
            ]
        next_payload = payload.get("next_action") or actions[0].__dict__
        next_action = PlannedActionStep.from_dict(next_payload)
        return cls(
            plan_id=str(payload.get("plan_id") or f"plan-{uuid.uuid4().hex[:8]}"),
            plan_horizon=max(1, int(payload.get("plan_horizon", len(actions)))),
            ordered_goals=[AgentGoal.from_dict(g) for g in payload.get("ordered_goals", [])],
            ordered_actions=actions,
            next_action=next_action,
            confidence=float(payload.get("confidence", 0.0)),
            replan_conditions=list(payload.get("replan_conditions", [])),
            notes=list(payload.get("notes", [])),
            uncertainty_flags=list(payload.get("uncertainty_flags", [])),
            plan_method_id=payload.get("plan_method_id"),
        )


@dataclass
class AgentBrainRequest:
    request_id: str
    tick: int
    sim_time: float
    agent_id: str
    display_name: str
    task_id: str
    phase: str
    local_context_summary: str
    local_observations: List[str]
    working_memory_summary: Dict[str, Any]
    inbox_summary: List[str]
    current_goal_stack: List[Dict[str, Any]]
    current_plan_summary: Dict[str, Any]
    allowed_actions: List[Dict[str, Any]]
    planning_horizon_config: Dict[str, Any]
    request_explanation: bool
    agent_label: Optional[str] = None
    explanation_style: Optional[str] = None
    task_context: Dict[str, Any] = field(default_factory=dict)
    rule_context: List[str] = field(default_factory=list)
    derivation_context: List[str] = field(default_factory=list)
    artifact_context: List[Dict[str, Any]] = field(default_factory=list)
    bootstrap_summary: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentBrainRequest":
        return cls(
            request_id=str(payload.get("request_id", "")),
            tick=int(payload.get("tick", 0)),
            sim_time=float(payload.get("sim_time", 0.0)),
            agent_id=str(payload.get("agent_id", "")),
            display_name=str(payload.get("display_name", "")),
            task_id=str(payload.get("task_id", "")),
            phase=str(payload.get("phase", "")),
            local_context_summary=str(payload.get("local_context_summary", "")),
            local_observations=[str(item) for item in payload.get("local_observations", [])],
            working_memory_summary=dict(payload.get("working_memory_summary", {})),
            inbox_summary=[str(item) for item in payload.get("inbox_summary", [])],
            current_goal_stack=list(payload.get("current_goal_stack", [])),
            current_plan_summary=dict(payload.get("current_plan_summary", {})),
            allowed_actions=list(payload.get("allowed_actions", [])),
            planning_horizon_config=dict(payload.get("planning_horizon_config", {})),
            request_explanation=bool(payload.get("request_explanation", False)),
            agent_label=payload.get("agent_label"),
            explanation_style=payload.get("explanation_style"),
            task_context=dict(payload.get("task_context", {})),
            rule_context=[str(item) for item in payload.get("rule_context", [])],
            derivation_context=[str(item) for item in payload.get("derivation_context", [])],
            artifact_context=list(payload.get("artifact_context", [])),
            bootstrap_summary=payload.get("bootstrap_summary"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tick": self.tick,
            "sim_time": self.sim_time,
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "agent_label": self.agent_label,
            "task_id": self.task_id,
            "phase": self.phase,
            "local_context_summary": self.local_context_summary,
            "local_observations": self.local_observations,
            "working_memory_summary": self.working_memory_summary,
            "inbox_summary": self.inbox_summary,
            "current_goal_stack": self.current_goal_stack,
            "current_plan_summary": self.current_plan_summary,
            "allowed_actions": self.allowed_actions,
            "planning_horizon_config": self.planning_horizon_config,
            "request_explanation": self.request_explanation,
            "explanation_style": self.explanation_style,
            "task_context": self.task_context,
            "rule_context": self.rule_context,
            "derivation_context": self.derivation_context,
            "artifact_context": self.artifact_context,
            "bootstrap_summary": self.bootstrap_summary,
        }


@dataclass
class AgentBrainResponse:
    response_id: str
    agent_id: str
    plan: AgentPlan
    confidence: float = 0.0
    belief_updates: Dict[str, Any] = field(default_factory=dict)
    uncertainty_estimates: Dict[str, Any] = field(default_factory=dict)
    communication_act: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    escalation_flags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentBrainResponse":
        plan_payload = payload.get("plan") or {}
        return cls(
            response_id=str(payload.get("response_id") or f"resp-{uuid.uuid4().hex[:8]}"),
            agent_id=str(payload.get("agent_id", "")),
            plan=AgentPlan.from_dict(plan_payload),
            confidence=float(payload.get("confidence", plan_payload.get("confidence", 0.0))),
            belief_updates=dict(payload.get("belief_updates", {})),
            uncertainty_estimates=dict(payload.get("uncertainty_estimates", {})),
            communication_act=payload.get("communication_act"),
            explanation=payload.get("explanation"),
            escalation_flags=list(payload.get("escalation_flags", [])),
        )


BRAIN_REQUEST_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": [
        "request_id", "tick", "sim_time", "agent_id", "display_name", "task_id", "phase",
        "local_context_summary", "local_observations", "working_memory_summary", "inbox_summary",
        "current_goal_stack", "current_plan_summary", "allowed_actions", "planning_horizon_config",
        "request_explanation",
    ],
    "properties": {
        "request_id": {"type": "string"},
        "tick": {"type": "integer"},
        "sim_time": {"type": "number"},
        "agent_id": {"type": "string"},
        "display_name": {"type": "string"},
        "agent_label": {"type": ["string", "null"]},
        "task_id": {"type": "string"},
        "phase": {"type": "string"},
        "local_context_summary": {"type": "string"},
        "local_observations": {"type": "array"},
        "working_memory_summary": {"type": "object"},
        "inbox_summary": {"type": "array"},
        "current_goal_stack": {"type": "array"},
        "current_plan_summary": {"type": "object"},
        "allowed_actions": {"type": "array"},
        "planning_horizon_config": {"type": "object"},
        "request_explanation": {"type": "boolean"},
        "explanation_style": {"type": ["string", "null"]},
        "bootstrap_summary": {"type": ["object", "null"]},
    },
}


BRAIN_RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["response_id", "agent_id", "plan"],
    "properties": {
        "response_id": {"type": "string"},
        "agent_id": {"type": "string"},
        "plan": {
            "type": "object",
            "required": ["plan_id", "plan_horizon", "ordered_goals", "ordered_actions", "next_action", "confidence"],
        },
        "communication_act": {"type": ["object", "null"]},
        "explanation": {"type": ["string", "null"]},
    },
}




@dataclass
class DIKCandidateUpdate:
    candidate_id: str
    evidence_ids: List[str] = field(default_factory=list)
    justification: Optional[str] = None
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DIKCandidateUpdate":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            evidence_ids=[str(item) for item in payload.get("evidence_ids", []) if str(item).strip()],
            justification=(str(payload.get("justification")).strip() if payload.get("justification") is not None else None),
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
        )


@dataclass
class AgentDIKIntegrationRequest:
    request_id: str
    tick: int
    sim_time: float
    agent_id: str
    display_name: str
    phase: str
    trigger_reason: str
    held_data_ids: List[str]
    held_information_ids: List[str]
    held_knowledge_ids: List[str]
    recent_new_item_ids: List[str] = field(default_factory=list)
    recent_communication_ids: List[str] = field(default_factory=list)
    recent_artifact_ids: List[str] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)
    contradiction_signals: List[str] = field(default_factory=list)
    candidate_information_ids: List[str] = field(default_factory=list)
    candidate_knowledge_ids: List[str] = field(default_factory=list)
    candidate_rule_ids: List[str] = field(default_factory=list)
    candidate_information_grounding: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    candidate_knowledge_grounding: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    candidate_rule_grounding: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    max_candidates_per_type: int = 8

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentDIKIntegrationRequest":
        return cls(
            request_id=str(payload.get("request_id", "")),
            tick=int(payload.get("tick", 0)),
            sim_time=float(payload.get("sim_time", 0.0)),
            agent_id=str(payload.get("agent_id", "")),
            display_name=str(payload.get("display_name", "")),
            phase=str(payload.get("phase", "")),
            trigger_reason=str(payload.get("trigger_reason", "")),
            held_data_ids=[str(item) for item in payload.get("held_data_ids", [])],
            held_information_ids=[str(item) for item in payload.get("held_information_ids", [])],
            held_knowledge_ids=[str(item) for item in payload.get("held_knowledge_ids", [])],
            recent_new_item_ids=[str(item) for item in payload.get("recent_new_item_ids", [])],
            recent_communication_ids=[str(item) for item in payload.get("recent_communication_ids", [])],
            recent_artifact_ids=[str(item) for item in payload.get("recent_artifact_ids", [])],
            unresolved_gaps=[str(item) for item in payload.get("unresolved_gaps", [])],
            contradiction_signals=[str(item) for item in payload.get("contradiction_signals", [])],
            candidate_information_ids=[str(item) for item in payload.get("candidate_information_ids", [])],
            candidate_knowledge_ids=[str(item) for item in payload.get("candidate_knowledge_ids", [])],
            candidate_rule_ids=[str(item) for item in payload.get("candidate_rule_ids", [])],
            candidate_information_grounding={
                str(k): [dict(entry) for entry in (v or []) if isinstance(entry, dict)]
                for k, v in dict(payload.get("candidate_information_grounding", {})).items()
            },
            candidate_knowledge_grounding={
                str(k): [dict(entry) for entry in (v or []) if isinstance(entry, dict)]
                for k, v in dict(payload.get("candidate_knowledge_grounding", {})).items()
            },
            candidate_rule_grounding={
                str(k): dict(v)
                for k, v in dict(payload.get("candidate_rule_grounding", {})).items()
                if isinstance(v, dict)
            },
            max_candidates_per_type=max(1, int(payload.get("max_candidates_per_type", 8) or 8)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "tick": self.tick,
            "sim_time": self.sim_time,
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "phase": self.phase,
            "trigger_reason": self.trigger_reason,
            "held_data_ids": list(self.held_data_ids),
            "held_information_ids": list(self.held_information_ids),
            "held_knowledge_ids": list(self.held_knowledge_ids),
            "recent_new_item_ids": list(self.recent_new_item_ids),
            "recent_communication_ids": list(self.recent_communication_ids),
            "recent_artifact_ids": list(self.recent_artifact_ids),
            "unresolved_gaps": list(self.unresolved_gaps),
            "contradiction_signals": list(self.contradiction_signals),
            "candidate_information_ids": list(self.candidate_information_ids),
            "candidate_knowledge_ids": list(self.candidate_knowledge_ids),
            "candidate_rule_ids": list(self.candidate_rule_ids),
            "candidate_information_grounding": dict(self.candidate_information_grounding),
            "candidate_knowledge_grounding": dict(self.candidate_knowledge_grounding),
            "candidate_rule_grounding": dict(self.candidate_rule_grounding),
            "max_candidates_per_type": self.max_candidates_per_type,
        }


@dataclass
class AgentDIKIntegrationResponse:
    response_id: str
    agent_id: str
    candidate_information_updates: List[DIKCandidateUpdate] = field(default_factory=list)
    candidate_knowledge_updates: List[DIKCandidateUpdate] = field(default_factory=list)
    candidate_rule_supports: List[DIKCandidateUpdate] = field(default_factory=list)
    unresolved_gaps: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AgentDIKIntegrationResponse":
        return cls(
            response_id=str(payload.get("response_id") or f"dik-{uuid.uuid4().hex[:8]}"),
            agent_id=str(payload.get("agent_id", "")),
            candidate_information_updates=[DIKCandidateUpdate.from_dict(item) for item in payload.get("candidate_information_updates", []) if isinstance(item, dict)],
            candidate_knowledge_updates=[DIKCandidateUpdate.from_dict(item) for item in payload.get("candidate_knowledge_updates", []) if isinstance(item, dict)],
            candidate_rule_supports=[DIKCandidateUpdate.from_dict(item) for item in payload.get("candidate_rule_supports", []) if isinstance(item, dict)],
            unresolved_gaps=[str(item) for item in payload.get("unresolved_gaps", [])],
            contradictions=[str(item) for item in payload.get("contradictions", [])],
            summary=str(payload.get("summary", "")),
            confidence=max(0.0, min(1.0, float(payload.get("confidence", 0.0) or 0.0))),
        )


def validate_agent_dik_integration_response(response: AgentDIKIntegrationResponse) -> List[str]:
    errors: List[str] = []
    for bucket_name, bucket in (
        ("candidate_information_updates", response.candidate_information_updates),
        ("candidate_knowledge_updates", response.candidate_knowledge_updates),
        ("candidate_rule_supports", response.candidate_rule_supports),
    ):
        for idx, candidate in enumerate(bucket):
            if not str(candidate.candidate_id).strip():
                errors.append(f"{bucket_name}[{idx}] missing candidate_id")
            if not isinstance(candidate.evidence_ids, list):
                errors.append(f"{bucket_name}[{idx}] evidence_ids must be list")
            if not (0.0 <= float(candidate.confidence) <= 1.0):
                errors.append(f"{bucket_name}[{idx}] confidence must be in [0,1]")
    if not (0.0 <= float(response.confidence) <= 1.0):
        errors.append("confidence must be in [0,1]")
    return errors

def validate_agent_brain_response(response: AgentBrainResponse, legal_action_ids: List[str]) -> List[str]:
    errors: List[str] = []
    if response.plan.next_action.action_type.value not in set(legal_action_ids):
        errors.append(f"illegal next_action '{response.plan.next_action.action_type.value}'")
    if not response.plan.ordered_actions:
        errors.append("plan ordered_actions cannot be empty")
    if not (0.0 <= float(response.plan.confidence) <= 1.0):
        errors.append("plan confidence must be in [0,1]")
    return errors
