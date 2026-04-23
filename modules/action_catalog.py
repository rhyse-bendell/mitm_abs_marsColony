from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

from modules.action_schema import ExecutableActionType


@dataclass(frozen=True)
class ActionCatalogEntry:
    action_id: str
    enum_name: str
    label: str
    category: str
    planner_visible: bool
    executable: bool
    target_required: bool
    allowed_target_kinds: tuple[str, ...]
    expected_in_allowed_actions: bool
    aliases: tuple[str, ...]
    preferred_modes: tuple[str, ...]
    method_steps: tuple[str, ...]
    translation_destination: str
    execution_owner: str
    status: str


_ACTIONS: tuple[ActionCatalogEntry, ...] = (
    ActionCatalogEntry("move_to_target", "MOVE_TO_TARGET", "Move to target", "movement", True, True, True, ("information", "build", "artifact", "team", "self"), False, ("move", "move_to", "navigate", "go_to_target"), (), (), "agent.translate -> type=move_to", "modules/agent.py::_translate_brain_decision_to_legacy_action", "partial"),
    ActionCatalogEntry("inspect_information_source", "INSPECT_INFORMATION_SOURCE", "Inspect information source", "inspection", True, True, True, ("information",), True, ("inspect", "inspect_info", "inspect_information_source", "inspect_source"), ("BOOTSTRAP", "ACQUIRE_DIK", "RECOVERY"), ("move_to_shared_source", "inspect_shared_source", "identify_role_source", "move_to_role_source", "inspect_role_source"), "agent.translate -> type=move_to + source_target_id", "modules/agent.py::_translate_brain_decision_to_legacy_action + _inspect_source", "implemented"),
    ActionCatalogEntry("communicate", "COMMUNICATE", "Communicate", "communication", True, True, False, ("team",), True, ("communicate", "communicate_with_team", "share", "message_teammate"), ("COORDINATE",), ("select_teammate_or_artifact", "communicate_critical_dik", "formulate_team_plan", "adopt_team_plan_direction"), "agent.translate -> type=communicate", "modules/agent.py::_translate_brain_decision_to_legacy_action + _advance_active_actions", "implemented"),
    ActionCatalogEntry("request_assistance", "REQUEST_ASSISTANCE", "Request assistance", "communication", True, True, False, ("team",), True, ("request_assistance", "ask_for_help", "request_help", "ask_help"), ("COORDINATE", "ACQUIRE_DIK"), ("communicate_critical_dik",), "agent.translate -> type=communicate + assist_action", "modules/agent.py::_translate_brain_decision_to_legacy_action + _advance_active_actions", "implemented"),
    ActionCatalogEntry("meeting", "MEETING", "Meeting", "communication", True, True, False, ("team",), False, ("meeting", "team_meeting"), (), (), "agent.translate -> type=communicate", "modules/agent.py::_translate_brain_decision_to_legacy_action", "experimental"),
    ActionCatalogEntry("externalize_plan", "EXTERNALIZE_PLAN", "Externalize plan", "artifact use", True, True, True, ("artifact", "build"), True, ("externalize", "externalize_plan", "write_plan", "propose_plan"), ("COORDINATE", "INTEGRATE_DIK"), ("integrate_role_dik", "select_teammate_or_artifact", "formulate_team_plan", "externalize_team_plan", "consult_artifact"), "agent.translate -> type=idle + artifact_action", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("consult_team_artifact", "CONSULT_TEAM_ARTIFACT", "Consult team artifact", "artifact use", True, True, True, ("artifact",), True, ("consult_team_artifact", "consult_artifact", "inspect_artifact", "read_whiteboard"), ("INTEGRATE_DIK", "COORDINATE"), ("integrate_shared_dik", "integrate_role_dik", "diagnose_deadlock", "consult_team_plan", "adopt_team_plan_direction", "consult_artifact"), "agent.translate -> type=idle + artifact_action", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("transport_resources", "TRANSPORT_RESOURCES", "Transport resources", "logistics", True, True, True, ("logistics", "build"), True, ("transport", "transport_resources", "deliver_resources", "carry_resources"), ("LOGISTICS",), ("identify_viable_project", "bind_project_target", "ensure_project_binding", "choose_accessible_pile", "move_to_pile", "pickup", "move_to_project", "dropoff", "ensure_build_ready"), "agent.translate -> type=transport_resources", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("start_construction", "START_CONSTRUCTION", "Start construction", "construction", True, True, True, ("build",), True, ("start_construction", "build_start", "start_build"), ("CONSTRUCT",), ("bind_project_target", "start_or_continue_construction"), "agent.translate -> type=construct", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("continue_construction", "CONTINUE_CONSTRUCTION", "Continue construction", "construction", True, True, True, ("build",), True, ("continue_construction", "continue_build", "build_continue"), ("CONSTRUCT",), ("start_or_continue_construction",), "agent.translate -> type=construct", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("repair_or_correct_construction", "REPAIR_OR_CORRECT_CONSTRUCTION", "Repair or correct construction", "repair", True, True, True, ("build",), True, ("repair", "repair_or_correct_construction", "correct_construction"), ("REPAIR",), ("attempt_repair",), "agent.translate -> type=construct", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("validate_construction", "VALIDATE_CONSTRUCTION", "Validate construction", "validation", True, True, True, ("build",), True, ("validate", "validate_construction", "check_construction"), ("VALIDATE", "REPAIR"), ("perform_validation", "revalidate"), "agent.translate -> type=idle (validation branch)", "modules/agent.py::_translate_brain_decision_to_legacy_action + _apply_externalization_and_construction_effects", "implemented"),
    ActionCatalogEntry("observe_environment", "OBSERVE_ENVIRONMENT", "Observe environment", "inspection", True, True, False, ("self",), False, ("observe", "observe_environment", "scan_environment"), ("MONITOR", "RECOVERY", "VALIDATE"), ("identify_role_source", "diagnose_deadlock", "perform_validation"), "agent.translate -> type=idle", "modules/agent.py::_translate_brain_decision_to_legacy_action", "implemented"),
    ActionCatalogEntry("reassess_plan", "REASSESS_PLAN", "Reassess plan", "regulation", True, True, False, ("self",), False, ("reassess", "reassess_plan", "replan"), ("RECOVERY", "INTEGRATE_DIK", "MONITOR"), ("integrate_shared_dik", "reassess_plan_with_rules"), "agent.translate -> type=idle", "modules/agent.py::_translate_brain_decision_to_legacy_action", "implemented"),
    ActionCatalogEntry("wait", "WAIT", "Wait", "passive", True, True, False, ("self",), True, ("wait", "idle", "hold"), ("MONITOR",), (), "agent.translate -> type=idle", "modules/agent.py::_translate_brain_decision_to_legacy_action", "implemented"),
)

ACTION_CATALOG: Dict[str, ActionCatalogEntry] = {entry.action_id: entry for entry in _ACTIONS}
ACTION_ALIASES: Dict[str, str] = {alias: entry.action_id for entry in _ACTIONS for alias in entry.aliases}


def normalize_action_alias(value: str) -> str | None:
    lowered = str(value or "").strip().lower()
    if not lowered:
        return None
    if lowered in ExecutableActionType._value2member_map_:
        return lowered
    return ACTION_ALIASES.get(lowered)


def planner_expected_action_ids() -> set[str]:
    return {
        entry.action_id
        for entry in ACTION_CATALOG.values()
        if entry.planner_visible and entry.expected_in_allowed_actions
    }


def validate_action_catalog_coverage() -> list[str]:
    errors: list[str] = []
    enum_values = {action.value for action in ExecutableActionType}
    missing = sorted(enum_values - set(ACTION_CATALOG))
    extra = sorted(set(ACTION_CATALOG) - enum_values)
    if missing:
        errors.append(f"missing_catalog_entries:{'|'.join(missing)}")
    if extra:
        errors.append(f"catalog_entries_without_enum:{'|'.join(extra)}")
    for entry in ACTION_CATALOG.values():
        if entry.action_id != entry.action_id.lower():
            errors.append(f"non_canonical_action_id_case:{entry.action_id}")
    return errors


def validate_rulebrain_action_references(
    mode_action_preferences: Mapping[str, Mapping[str, float]],
    step_action_map: Mapping[str, Iterable[str]],
    fallback_order: Iterable[str],
) -> list[str]:
    errors: list[str] = []
    known = set(ACTION_CATALOG)
    for mode, prefs in mode_action_preferences.items():
        for action_id in prefs:
            if action_id not in known:
                errors.append(f"mode_preference_unknown_action:{mode}:{action_id}")
    for step, actions in step_action_map.items():
        for action_id in actions:
            if action_id not in known:
                errors.append(f"step_map_unknown_action:{step}:{action_id}")
    for action_id in fallback_order:
        if action_id not in known:
            errors.append(f"fallback_unknown_action:{action_id}")
    return errors
