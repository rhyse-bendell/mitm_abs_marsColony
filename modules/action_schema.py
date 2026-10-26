"""Canonical action, communication, and planner decision schemas.

Strict enums and validation helpers protect simulator integrity by requiring
planner outputs to conform before translation and execution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ExecutableActionType(str, Enum):
    MOVE_TO_TARGET = "move_to_target"
    INSPECT_INFORMATION_SOURCE = "inspect_information_source"
    COMMUNICATE = "communicate"
    REQUEST_ASSISTANCE = "request_assistance"
    MEETING = "meeting"
    EXTERNALIZE_PLAN = "externalize_plan"
    EXTERNALIZE_KNOWLEDGE = "externalize_knowledge"
    CONSULT_TEAM_ARTIFACT = "consult_team_artifact"
    TRANSPORT_RESOURCES = "transport_resources"
    START_CONSTRUCTION = "start_construction"
    CONTINUE_CONSTRUCTION = "continue_construction"
    REPAIR_OR_CORRECT_CONSTRUCTION = "repair_or_correct_construction"
    VALIDATE_CONSTRUCTION = "validate_construction"
    CONNECT_STRUCTURES = "connect_structures"
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
    # Optional action-specific structured arguments (see ACTION_ARG_SPECS). Only actions that declare a spec
    # may carry arguments; every other action leaves this empty.
    action_args: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Single normalization point for the contract: a legacy packed connect target
        # ("connector|endpoint_a|endpoint_b") is consumed into structured args here, so everything
        # downstream sees one representation.
        self.action_args = dict(self.action_args) if isinstance(self.action_args, dict) else {"_malformed_action_args": repr(self.action_args)}
        if self.selected_action == ExecutableActionType.CONNECT_STRUCTURES:
            packed = parse_legacy_connection_target(self.target_id)
            if packed is not None:
                if not self.action_args:
                    self.action_args = dict(zip(CONNECT_STRUCTURES_ARG_KEYS, packed))
                self.target_id = None


CONNECT_STRUCTURES_ARG_KEYS: Tuple[str, str, str] = ("connector_project_id", "endpoint_a_id", "endpoint_b_id")
_LEGACY_CONNECTION_SEPARATOR = "|"
_ARG_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:\-]*$")
_STRUCTURE_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# Allowed argument keys per action. Actions absent here take no arguments.
#   start_construction: structure_type is optional at the schema level (a START without it continues an existing
#   unfinished project at the target site); creating a NEW project requires it (enforced by the action gate).
ACTION_ARG_SPECS: Dict[ExecutableActionType, Dict[str, Any]] = {
    ExecutableActionType.START_CONSTRUCTION: {"required": (), "allowed": ("structure_type", "design_slot", "design_option")},
    # design_slot/design_option: the specification choice the Pilot makes for the next build step (or, for a repair, the slot it reworks).
    # The option menu is the same for every agent and never marks an option as correct (see Documentation/research/MITM_construction_contract.md).
    ExecutableActionType.CONTINUE_CONSTRUCTION: {"required": (), "allowed": ("design_slot", "design_option")},
    ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION: {"required": (), "allowed": ("design_slot", "design_option")},
    ExecutableActionType.CONNECT_STRUCTURES: {"required": CONNECT_STRUCTURES_ARG_KEYS, "allowed": CONNECT_STRUCTURES_ARG_KEYS},
    # Knowledge externalization (not a team plan): `kind` is the epistemic intent, `element_id` the grounded DIK reference (a held
    # information/rule element, or for `uncertainty` a recognized gap), `project_id` an optional construction project it concerns.
    ExecutableActionType.EXTERNALIZE_KNOWLEDGE: {"required": ("kind", "element_id"), "allowed": ("kind", "element_id", "project_id")},
}
EXTERNALIZATION_KINDS: Tuple[str, ...] = ("claim", "evidence", "uncertainty", "design_note")


def decision_action_args(decision: Any) -> Dict[str, Any]:
    """The decision's action_args as a fresh dict; tolerant of duck-typed decisions that predate the field."""
    args = getattr(decision, "action_args", None)
    return dict(args) if isinstance(args, dict) else {}


def parse_legacy_connection_target(target_id: Any) -> Optional[Tuple[str, str, str]]:
    """Read-only compatibility parser for the retired packed connect target ``connector|a|b``."""
    parts = str(target_id or "").split(_LEGACY_CONNECTION_SEPARATOR)
    if len(parts) != 3 or not all(p.strip() for p in parts):
        return None
    return tuple(p.strip() for p in parts)  # type: ignore[return-value]


def validate_action_args(action: ExecutableActionType, args: Any, *, allowed_structure_types: Optional[List[str]] = None) -> List[str]:
    """Structural validation of ``action_args`` for one action; returns error tokens (empty when valid).

    Checks allowed keys, required keys, value types and id shape. It does not decide legality: the
    Environment/ConstructionManager stays authoritative for that.
    """
    if not isinstance(args, dict):
        return ["action_args_not_a_mapping"]
    spec = ACTION_ARG_SPECS.get(action)
    errors: List[str] = []
    if spec is None:
        return [f"unsupported_action_arg:{k}" for k in sorted(args)] if args else []
    for key in sorted(args):
        if key not in spec["allowed"]:
            errors.append(f"unsupported_action_arg:{key}")
    for key in spec["required"]:
        if key not in args or args.get(key) in (None, ""):
            errors.append(f"missing_action_arg:{key}")
    for key in spec["allowed"]:
        if key not in args or args.get(key) in (None, ""):
            continue
        value = args[key]
        if not isinstance(value, str):
            errors.append(f"action_arg_not_a_string:{key}")
        elif key == "kind":
            if value not in EXTERNALIZATION_KINDS:
                errors.append(f"unknown_externalization_kind:{value}")
        elif key == "structure_type":
            if not _STRUCTURE_TYPE_PATTERN.match(value):
                errors.append("malformed_structure_type")
            elif allowed_structure_types is not None and value not in allowed_structure_types:
                errors.append(f"unknown_structure_type:{value}")
        elif not _ARG_ID_PATTERN.match(value):
            errors.append(f"malformed_action_arg:{key}")
    return errors


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
