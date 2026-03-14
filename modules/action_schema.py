from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ExecutableActionType(str, Enum):
    MOVE_TO_TARGET = "move_to_target"
    INSPECT_INFORMATION_SOURCE = "inspect_information_source"
    COMMUNICATE = "communicate"
    REQUEST_ASSISTANCE = "request_assistance"
    MEETING = "meeting"
    EXTERNALIZE_PLAN = "externalize_plan"
    CONSULT_TEAM_ARTIFACT = "consult_team_artifact"
    TRANSPORT_RESOURCES = "transport_resources"
    START_CONSTRUCTION = "start_construction"
    CONTINUE_CONSTRUCTION = "continue_construction"
    REPAIR_OR_CORRECT_CONSTRUCTION = "repair_or_correct_construction"
    VALIDATE_CONSTRUCTION = "validate_construction"
    OBSERVE_ENVIRONMENT = "observe_environment"
    REASSESS_PLAN = "reassess_plan"
    WAIT = "wait"


class InternalEventType(str, Enum):
    TRANSFORM_DATA_TO_INFORMATION = "transform_data_to_information"
    TRANSFORM_INFORMATION_TO_KNOWLEDGE = "transform_information_to_knowledge"
    UPDATE_TEAM_MODEL = "update_team_model"
    UPDATE_TEAMMATE_MODEL = "update_teammate_model"
    DETECT_CONFLICT_OR_GAP = "detect_conflict_or_gap"
    ADOPT_EXTERNALIZED_KNOWLEDGE = "adopt_externalized_knowledge"
    REVISE_GOAL_STACK = "revise_goal_stack"


class CommunicationIntent(str, Enum):
    TDP = "TDP"  # data provision
    TIP = "TIP"  # information provision
    TKP = "TKP"  # knowledge provision
    TGTO = "TGTO"  # goal/task objective
    TKRQ = "TKRQ"  # knowledge request
    TCR = "TCR"  # correction/repair
    TPP = "TPP"  # plan proposal
    TPA = "TPA"  # plan acknowledgement/agreement/disagreement


LEGACY_COMMUNICATION_TYPE_MAP: Dict[str, str] = {
    "TDP": "TDP",
    "TIP": "TIP",
    "TKP": "TKP",
    "TGTO": "TGTO",
    "TKR": "TKRQ",
    "TKRQ": "TKRQ",
    "TCR": "TCR",
}


@dataclass(frozen=True)
class SimulatorAction:
    action_type: ExecutableActionType
    target_id: Optional[str] = None
    target_zone: Optional[str] = None
    duration_s: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CognitiveEvent:
    event_type: InternalEventType
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BrainDecision:
    selected_action: ExecutableActionType
    target_id: Optional[str] = None
    target_zone: Optional[str] = None
    goal_update: Optional[str] = None
    plan_steps: List[str] = field(default_factory=list)
    plan_method_id: Optional[str] = None
    next_steps: List[str] = field(default_factory=list)
    communication_intent: Optional[CommunicationIntent] = None
    reason_summary: str = ""
    confidence: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    requests_for_context: List[str] = field(default_factory=list)


def validate_brain_decision(decision: BrainDecision, legal_actions: List[ExecutableActionType]) -> List[str]:
    errors: List[str] = []
    if decision.selected_action not in legal_actions:
        errors.append(f"selected_action '{decision.selected_action.value}' is not currently legal")
    if not (0.0 <= decision.confidence <= 1.0):
        errors.append("confidence must be between 0.0 and 1.0")
    if decision.selected_action == ExecutableActionType.TRANSPORT_RESOURCES:
        duration = decision.assumptions and next((a for a in decision.assumptions if a.startswith("duration_s=")), None)
        if duration is None:
            errors.append("transport_resources decisions should include duration_s assumption")
    return errors
