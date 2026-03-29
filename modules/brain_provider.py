from __future__ import annotations

import json
import logging
import math
import random
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict
from urllib import error, request

from modules.action_schema import BrainDecision, CommunicationIntent, ExecutableActionType
from modules.brain_contract import (
    AgentBrainRequest,
    AgentBrainResponse,
    AgentDIKIntegrationRequest,
    AgentDIKIntegrationResponse,
    PlannedActionStep,
    validate_agent_dik_integration_response,
)


LOGGER = logging.getLogger(__name__)


RUNTIME_ACTION_ALIASES: dict[str, str] = {
    "inspect": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
    "inspect_info": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
    "observe": ExecutableActionType.OBSERVE_ENVIRONMENT.value,
    "communicate_with_team": ExecutableActionType.COMMUNICATE.value,
}


def _truncate_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _bounded_json_value(value: Any, *, max_chars: int) -> Any:
    try:
        serialized = json.dumps(value, default=str)
    except TypeError:
        serialized = json.dumps(str(value))
    if len(serialized) <= max_chars:
        return value
    return {"_truncated_json": _truncate_text(serialized, max_chars=max_chars)}


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _content_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "output_text"):
            if isinstance(value.get(key), str):
                return str(value.get(key)).strip()
    return ""


def select_productive_fallback_action(allowed_actions: list[dict[str, Any]]) -> PlannedActionStep:
    """Pick a deterministic safe fallback action with conservative progress bias."""
    if not allowed_actions:
        return PlannedActionStep(step_index=0, action_type=ExecutableActionType.WAIT, expected_purpose="fallback legal wait")

    indexed_allowed = [item for item in allowed_actions if isinstance(item, dict)]
    build_actions_present = any(
        item.get("action_type")
        in {
            ExecutableActionType.START_CONSTRUCTION.value,
            ExecutableActionType.CONTINUE_CONSTRUCTION.value,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
            ExecutableActionType.VALIDATE_CONSTRUCTION.value,
        }
        for item in indexed_allowed
    )
    preference_order = [
        ExecutableActionType.START_CONSTRUCTION.value,
        ExecutableActionType.CONTINUE_CONSTRUCTION.value,
        ExecutableActionType.VALIDATE_CONSTRUCTION.value,
        ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
        ExecutableActionType.TRANSPORT_RESOURCES.value,
        ExecutableActionType.EXTERNALIZE_PLAN.value,
        ExecutableActionType.COMMUNICATE.value,
        ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
        ExecutableActionType.REQUEST_ASSISTANCE.value,
        ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
        ExecutableActionType.OBSERVE_ENVIRONMENT.value,
        ExecutableActionType.WAIT.value,
    ]
    if not build_actions_present:
        preference_order.remove(ExecutableActionType.INSPECT_INFORMATION_SOURCE.value)
        preference_order.insert(5, ExecutableActionType.INSPECT_INFORMATION_SOURCE.value)

    for action_type in preference_order:
        choices = [item for item in indexed_allowed if item.get("action_type") == action_type]
        if not choices:
            continue
        reachable_first = sorted(choices, key=lambda item: 0 if item.get("reachable", True) else 1)
        choice = reachable_first[0]
        return PlannedActionStep(
            step_index=0,
            action_type=ExecutableActionType(action_type),
            target_id=choice.get("target_id"),
            target_zone=choice.get("target_zone"),
            expected_purpose="safe productive fallback action",
        )

    first = allowed_actions[0]
    return PlannedActionStep(
        step_index=0,
        action_type=ExecutableActionType(first.get("action_type", ExecutableActionType.WAIT.value)),
        target_id=first.get("target_id"),
        target_zone=first.get("target_zone"),
        expected_purpose="deterministic first legal action fallback",
    )


@dataclass(frozen=True)
class BrainBackendConfig:
    backend: str = "rule_brain"
    local_base_url: str = "http://127.0.0.1:11434"
    local_endpoint: str = "/v1/chat/completions"
    local_model: str = "qwen2.5:3b-instruct"
    timeout_s: float = 90.0
    warmup_timeout_s: float = 45.0
    completion_max_tokens: int = 1024
    startup_completion_max_tokens: int = 512
    permissive_timeout_ceiling_s: float = 1200.0
    permissive_completion_ceiling_tokens: int = 16384
    unrestricted_local_qwen_mode: bool = False
    max_retries: int = 0
    fallback_backend: str = "rule_brain"
    debug: bool = False
    planner_trace_enabled: bool = True
    planner_trace_mode: str = "full"
    planner_trace_max_chars: int = 12000


@dataclass(frozen=True)
class RuleBrainPolicyConfig:
    mode_selection_temperature: float = 0.45
    action_selection_temperature: float = 0.55
    dwell_bonus: float = 0.3
    switch_penalty: float = 0.25
    recovery_bonus: float = 0.75
    randomness_floor: float = 0.02
    min_mode_dwell_steps: int = 2
    max_history: int = 12
    max_logged_candidates: int = 6


@dataclass(frozen=True)
class RuleMethodDefinition:
    method_id: str
    applicable_modes: tuple[str, ...]
    ordered_steps: tuple[str, ...]
    preconditions: tuple[str, ...] = ()
    success_conditions: tuple[str, ...] = ()
    failure_conditions: tuple[str, ...] = ()
    switch_conditions: tuple[str, ...] = ()
    retry_budgets: dict[str, int] | None = None
    role_preference: tuple[str, ...] = ()
    source_cooldown_ticks: int = 0


def _rule_method_library() -> dict[str, RuleMethodDefinition]:
    return {
        "AcquireInitialGrounding": RuleMethodDefinition(
            method_id="AcquireInitialGrounding",
            applicable_modes=("BOOTSTRAP", "ACQUIRE_DIK"),
            ordered_steps=("move_to_shared_source", "inspect_shared_source", "integrate_shared_dik"),
            success_conditions=("shared_grounding_sufficient",),
            switch_conditions=("shared_source_exhausted_role_gap_remaining",),
            retry_budgets={"inspect_shared_source": 2, "move_to_shared_source": 3},
            source_cooldown_ticks=4,
        ),
        "AcquireRoleSpecificGrounding": RuleMethodDefinition(
            method_id="AcquireRoleSpecificGrounding",
            applicable_modes=("ACQUIRE_DIK",),
            ordered_steps=("identify_role_source", "move_to_role_source", "inspect_role_source", "integrate_role_dik"),
            success_conditions=("role_grounding_sufficient",),
            failure_conditions=("role_source_exhausted_without_gain",),
            retry_budgets={"inspect_role_source": 2, "move_to_role_source": 3},
        ),
        "ShareCriticalDIK": RuleMethodDefinition(
            method_id="ShareCriticalDIK",
            applicable_modes=("COORDINATE",),
            ordered_steps=("select_teammate_or_artifact", "communicate_critical_dik"),
            retry_budgets={"communicate_critical_dik": 2},
        ),
        "IntegrateColonyRules": RuleMethodDefinition(
            method_id="IntegrateColonyRules",
            applicable_modes=("INTEGRATE_DIK",),
            ordered_steps=("consult_artifact", "reassess_plan_with_rules"),
            success_conditions=("readiness_improved",),
            retry_budgets={"consult_artifact": 2},
        ),
        "SelectProjectAndSite": RuleMethodDefinition(
            method_id="SelectProjectAndSite",
            applicable_modes=("LOGISTICS", "CONSTRUCT"),
            ordered_steps=("identify_viable_project", "bind_project_target"),
            success_conditions=("project_bound",),
        ),
        "TransportResourcesToProject": RuleMethodDefinition(
            method_id="TransportResourcesToProject",
            applicable_modes=("LOGISTICS",),
            ordered_steps=("ensure_project_binding", "choose_accessible_pile", "move_to_pile", "pickup", "move_to_project", "dropoff"),
            success_conditions=("project_resource_complete",),
            retry_budgets={"move_to_pile": 3, "move_to_project": 3, "pickup": 2, "dropoff": 2},
        ),
        "ConstructProject": RuleMethodDefinition(
            method_id="ConstructProject",
            applicable_modes=("CONSTRUCT",),
            ordered_steps=("ensure_build_ready", "start_or_continue_construction"),
            success_conditions=("construction_progressed",),
            failure_conditions=("construction_blocked",),
            retry_budgets={"start_or_continue_construction": 3},
        ),
        "ValidateProject": RuleMethodDefinition(
            method_id="ValidateProject",
            applicable_modes=("VALIDATE",),
            ordered_steps=("perform_validation",),
            success_conditions=("validation_passed",),
            failure_conditions=("validation_failed",),
            retry_budgets={"perform_validation": 2},
        ),
        "RepairProject": RuleMethodDefinition(
            method_id="RepairProject",
            applicable_modes=("REPAIR",),
            ordered_steps=("attempt_repair", "revalidate"),
            success_conditions=("repair_succeeded",),
            failure_conditions=("repair_failed_repeatedly",),
            retry_budgets={"attempt_repair": 3},
        ),
    }

def create_brain_provider(config: BrainBackendConfig | None = None) -> BrainProvider:
    config = config or BrainBackendConfig()
    selected = config.backend.lower()
    if selected == "local_stub":
        return LocalLLMBrainStub()
    if selected in {"local_http", "openai_compatible_local", "ollama_local", "ollama"}:
        fallback = RuleBrain()
        return OllamaLocalBrainProvider(config=config, fallback=fallback)
    if selected == "cloud_stub":
        return CloudBrainStub()
    return RuleBrain()


class BrainProvider(ABC):
    @abstractmethod
    def generate_plan(self, request_packet: AgentBrainRequest) -> AgentBrainResponse:
        raise NotImplementedError

    def generate_dik_integration(self, request_packet: AgentDIKIntegrationRequest) -> AgentDIKIntegrationResponse:
        return AgentDIKIntegrationResponse.from_dict(
            {
                "response_id": f"dik-rule-{request_packet.request_id}",
                "agent_id": request_packet.agent_id,
                "summary": "No DIK integration candidates from deterministic fallback.",
                "confidence": 0.0,
            }
        )

    def decide(self, context_packet):
        decision = self.decision_from_context(context_packet)
        return decision

    def decision_from_context(self, context_packet) -> BrainDecision:
        request_packet = _request_from_context_packet(context_packet)
        response = self.generate_plan(request_packet)
        next_steps = [step.expected_purpose for step in response.plan.ordered_actions[:3] if step.expected_purpose]
        return response.plan.next_action.to_brain_decision(
            confidence=response.plan.confidence,
            plan_method_id=response.plan.plan_method_id,
            next_steps=next_steps,
        )


def _request_from_context_packet(context_packet) -> AgentBrainRequest:
    world = context_packet.world_snapshot
    cognitive = context_packet.individual_cognitive_state
    phase = world.get("phase_profile", {}).get("name", world.get("phase_state", {}).get("name", "default"))
    control_state = dict(cognitive.get("control_state", {}))
    return AgentBrainRequest(
        request_id=f"ctx-{int(time.time()*1000)}",
        tick=0,
        sim_time=float(world.get("sim_time", 0.0)),
        agent_id=str(context_packet.static_task_context.get("role", "agent")),
        display_name=str(context_packet.static_task_context.get("role", "agent")),
        agent_label=context_packet.static_task_context.get("role"),
        task_id="mars_colony",
        phase=str(phase),
        local_context_summary=f"build_status={cognitive.get('build_readiness', {}).get('status')}",
        local_observations=[str(x) for x in context_packet.history_bands.get("near_preceding_events", [])[:4]],
        working_memory_summary={
            "knowledge": list(cognitive.get("knowledge_summary", [])[:8]),
            "known_gaps": list(cognitive.get("known_gaps", [])[:6]),
        },
        inbox_summary=[],
        current_goal_stack=list(cognitive.get("goal_stack", [])),
        current_plan_summary=dict(cognitive.get("active_plan", {})),
        allowed_actions=list(context_packet.action_affordances),
        planning_horizon_config={"max_steps": 3},
        request_explanation=False,
        task_context={
            **dict(context_packet.static_task_context),
            "control_state": control_state,
            "build_readiness": dict(cognitive.get("build_readiness", {})),
            "built_state": list(world.get("built_state", [])),
            "loop_counters": dict(cognitive.get("loop_counters", {})),
            "progress_state": dict(cognitive.get("progress_state", {})),
            "seconds_since_dik_change": cognitive.get("seconds_since_dik_change"),
            "mismatch_signals": list(context_packet.history_bands.get("semantic_plan_evolution", {}).get("unresolved_contradictions", [])),
            "inspect_state": dict(cognitive.get("inspect_state", {})),
        },
        rule_context=list(cognitive.get("knowledge_summary", [])[:8]),
        derivation_context=[],
        artifact_context=list(context_packet.team_state.get("externalized_artifacts", []))[:4],
        control_mode=str(control_state.get("mode") or "BOOTSTRAP"),
        previous_control_mode=control_state.get("previous_mode"),
        mode_dwell_steps=int(control_state.get("mode_dwell_steps", 0) or 0),
        last_transition_reason=str(control_state.get("last_transition_reason") or "none"),
        control_state_snapshot={
            "mode": str(control_state.get("mode") or "BOOTSTRAP"),
            "previous_mode": control_state.get("previous_mode"),
            "mode_dwell_steps": int(control_state.get("mode_dwell_steps", 0) or 0),
            "last_transition_reason": str(control_state.get("last_transition_reason") or "none"),
            "recovery_active": bool(control_state.get("recovery_active")),
            "top_features": dict((control_state.get("last_policy_snapshot", {}) or {}).get("top_features", {}) or control_state.get("last_transition_features", {})),
            "policy_snapshot": dict(control_state.get("last_policy_snapshot", {})),
            "method_state": dict(control_state.get("method_state", {}) or {}),
            "inspect_state": dict(cognitive.get("inspect_state", {}) or {}),
        },
    )


class RuleBrain(BrainProvider):
    """Hierarchical stochastic fallback brain with simulator-side control-state."""

    MODES = (
        "BOOTSTRAP",
        "ACQUIRE_DIK",
        "INTEGRATE_DIK",
        "COORDINATE",
        "LOGISTICS",
        "CONSTRUCT",
        "VALIDATE",
        "REPAIR",
        "RECOVERY",
        "MONITOR",
    )
    MODE_ACTION_PREFERENCES = {
        "BOOTSTRAP": {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value: 1.3, ExecutableActionType.OBSERVE_ENVIRONMENT.value: 0.5},
        "ACQUIRE_DIK": {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value: 1.6, ExecutableActionType.CONSULT_TEAM_ARTIFACT.value: 0.4, ExecutableActionType.REQUEST_ASSISTANCE.value: 0.4},
        "INTEGRATE_DIK": {ExecutableActionType.CONSULT_TEAM_ARTIFACT.value: 1.1, ExecutableActionType.REASSESS_PLAN.value: 0.9, ExecutableActionType.EXTERNALIZE_PLAN.value: 0.5},
        "COORDINATE": {ExecutableActionType.COMMUNICATE.value: 1.2, ExecutableActionType.EXTERNALIZE_PLAN.value: 1.0, ExecutableActionType.REQUEST_ASSISTANCE.value: 0.8},
        "LOGISTICS": {ExecutableActionType.TRANSPORT_RESOURCES.value: 1.5, ExecutableActionType.COMMUNICATE.value: 0.3},
        "CONSTRUCT": {ExecutableActionType.START_CONSTRUCTION.value: 1.4, ExecutableActionType.CONTINUE_CONSTRUCTION.value: 1.5},
        "VALIDATE": {ExecutableActionType.VALIDATE_CONSTRUCTION.value: 1.6, ExecutableActionType.OBSERVE_ENVIRONMENT.value: 0.3},
        "REPAIR": {ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value: 1.7, ExecutableActionType.VALIDATE_CONSTRUCTION.value: 0.4},
        "RECOVERY": {ExecutableActionType.REASSESS_PLAN.value: 1.0, ExecutableActionType.INSPECT_INFORMATION_SOURCE.value: 0.8, ExecutableActionType.OBSERVE_ENVIRONMENT.value: 0.5},
        "MONITOR": {ExecutableActionType.OBSERVE_ENVIRONMENT.value: 0.9, ExecutableActionType.WAIT.value: 0.3, ExecutableActionType.REASSESS_PLAN.value: 0.4},
    }
    METHOD_LIBRARY = _rule_method_library()
    MODE_METHOD_PREFERENCES = {
        "BOOTSTRAP": ("AcquireInitialGrounding",),
        "ACQUIRE_DIK": ("AcquireRoleSpecificGrounding", "AcquireInitialGrounding"),
        "INTEGRATE_DIK": ("IntegrateColonyRules",),
        "COORDINATE": ("ShareCriticalDIK",),
        "LOGISTICS": ("SelectProjectAndSite", "TransportResourcesToProject"),
        "CONSTRUCT": ("ConstructProject", "SelectProjectAndSite"),
        "VALIDATE": ("ValidateProject",),
        "REPAIR": ("RepairProject",),
        "RECOVERY": ("AcquireRoleSpecificGrounding", "IntegrateColonyRules"),
        "MONITOR": ("ShareCriticalDIK",),
    }
    STEP_ACTION_MAP = {
        "move_to_shared_source": {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value},
        "inspect_shared_source": {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value},
        "integrate_shared_dik": {ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, ExecutableActionType.REASSESS_PLAN.value},
        "identify_role_source": {ExecutableActionType.OBSERVE_ENVIRONMENT.value, ExecutableActionType.REASSESS_PLAN.value},
        "move_to_role_source": {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value},
        "inspect_role_source": {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value},
        "integrate_role_dik": {ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, ExecutableActionType.REASSESS_PLAN.value},
        "select_teammate_or_artifact": {ExecutableActionType.COMMUNICATE.value, ExecutableActionType.EXTERNALIZE_PLAN.value},
        "communicate_critical_dik": {ExecutableActionType.COMMUNICATE.value, ExecutableActionType.EXTERNALIZE_PLAN.value, ExecutableActionType.REQUEST_ASSISTANCE.value},
        "consult_artifact": {ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, ExecutableActionType.REASSESS_PLAN.value},
        "reassess_plan_with_rules": {ExecutableActionType.REASSESS_PLAN.value, ExecutableActionType.EXTERNALIZE_PLAN.value},
        "identify_viable_project": {ExecutableActionType.OBSERVE_ENVIRONMENT.value, ExecutableActionType.REASSESS_PLAN.value},
        "bind_project_target": {ExecutableActionType.TRANSPORT_RESOURCES.value, ExecutableActionType.START_CONSTRUCTION.value},
        "ensure_project_binding": {ExecutableActionType.TRANSPORT_RESOURCES.value, ExecutableActionType.REASSESS_PLAN.value},
        "choose_accessible_pile": {ExecutableActionType.TRANSPORT_RESOURCES.value},
        "move_to_pile": {ExecutableActionType.TRANSPORT_RESOURCES.value},
        "pickup": {ExecutableActionType.TRANSPORT_RESOURCES.value},
        "move_to_project": {ExecutableActionType.TRANSPORT_RESOURCES.value},
        "dropoff": {ExecutableActionType.TRANSPORT_RESOURCES.value},
        "ensure_build_ready": {ExecutableActionType.REASSESS_PLAN.value, ExecutableActionType.CONSULT_TEAM_ARTIFACT.value},
        "start_or_continue_construction": {ExecutableActionType.START_CONSTRUCTION.value, ExecutableActionType.CONTINUE_CONSTRUCTION.value},
        "perform_validation": {ExecutableActionType.VALIDATE_CONSTRUCTION.value, ExecutableActionType.OBSERVE_ENVIRONMENT.value},
        "attempt_repair": {ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value, ExecutableActionType.REASSESS_PLAN.value},
        "revalidate": {ExecutableActionType.VALIDATE_CONSTRUCTION.value},
    }

    def __init__(self, policy_config: RuleBrainPolicyConfig | None = None):
        self.policy_config = policy_config or RuleBrainPolicyConfig()

    @staticmethod
    def _best_affordance(
        sorted_affordances: list[dict[str, Any]],
        allowed_action_types: set[str],
    ) -> dict[str, Any] | None:
        reachable = [
            item
            for item in sorted_affordances
            if item.get("action_type") in allowed_action_types and item.get("reachable", True)
        ]
        if reachable:
            return reachable[0]
        return next((item for item in sorted_affordances if item.get("action_type") in allowed_action_types), None)

    def _softmax_pick(self, weighted_scores: dict[str, float], temperature: float, rng: random.Random) -> tuple[str, dict[str, float]]:
        if not weighted_scores:
            return "MONITOR", {"MONITOR": 1.0}
        temp = max(0.05, float(temperature))
        max_score = max(weighted_scores.values())
        exp_scores: dict[str, float] = {}
        for key, score in weighted_scores.items():
            exp_scores[key] = math.exp((score - max_score) / temp)
        total = sum(exp_scores.values()) or 1.0
        probs = {k: v / total for k, v in exp_scores.items()}
        pick = rng.random()
        csum = 0.0
        selected = next(iter(probs))
        for key, prob in sorted(probs.items(), key=lambda item: item[1], reverse=True):
            csum += prob
            if pick <= csum:
                selected = key
                break
        return selected, probs

    def _extract_features(self, context_packet, *, is_request: bool = False) -> dict[str, float]:
        request_ctx = dict(getattr(context_packet, "task_context", {}) or {}) if is_request else {}
        cognitive = context_packet.individual_cognitive_state if not is_request else request_ctx
        world = context_packet.world_snapshot if not is_request else {"built_state": request_ctx.get("built_state", [])}
        known_gaps = cognitive.get("known_gaps", []) if not is_request else list(context_packet.working_memory_summary.get("known_gaps", []))
        readiness = cognitive.get("build_readiness", {}) if not is_request else dict(request_ctx.get("build_readiness", {}))
        built_state = world.get("built_state", []) if not is_request else list(request_ctx.get("built_state", []))
        active_projects = [p for p in built_state if p.get("state") in {"absent", "in_progress"} and float(p.get("progress", 0.0)) < 1.0]
        mismatch_signals = context_packet.history_bands.get("semantic_plan_evolution", {}).get("unresolved_contradictions", []) if not is_request else list(request_ctx.get("mismatch_signals", []))
        loop_counters = cognitive.get("loop_counters", {}) if not is_request else dict(request_ctx.get("loop_counters", {}))
        progress_state = cognitive.get("progress_state", {}) if not is_request else dict(request_ctx.get("progress_state", {}))
        repeated = max(int(loop_counters.get("action_repeats", 0) or 0), int(loop_counters.get("selected_action_repeats", 0) or 0))
        no_progress_streak = max(int(loop_counters.get("no_progress_streak", 0) or 0), int(progress_state.get("no_progress_streak", 0) or 0))
        observe_no_effect = int(loop_counters.get("observe_no_effect", 0) or 0)
        communication_no_effect = int(loop_counters.get("communication_no_effect", 0) or 0)
        goal_stack = cognitive.get("goal_stack", []) if not is_request else list(context_packet.current_goal_stack)
        externalized = context_packet.team_state.get("externalized_artifacts", []) if not is_request else list(context_packet.artifact_context)
        validated = sum(1 for a in externalized if a.get("validation_state") == "validated")
        seconds_since_dik_change = cognitive.get("seconds_since_dik_change") if not is_request else request_ctx.get("seconds_since_dik_change")
        inspect_state = cognitive.get("inspect_state", {}) if not is_request else dict(request_ctx.get("inspect_state", {}))
        if is_request:
            request_control = dict(context_packet.control_state_snapshot or {})
            if request_control:
                inspect_state = dict(request_control.get("inspect_state") or inspect_state)
            goal_stack = list(context_packet.current_goal_stack or goal_stack)
            plan_summary = dict(context_packet.current_plan_summary or {})
            if not goal_stack:
                derived_goal = plan_summary.get("goal_id") or plan_summary.get("summary")
                if derived_goal:
                    goal_stack = [{"goal_id": str(derived_goal)}]
            if known_gaps:
                known_gaps = list(known_gaps)[:8]
            elif context_packet.bootstrap_summary:
                known_gaps = ["bootstrap_missing_details"]
            if plan_summary.get("status") in {"blocked", "stalled"}:
                repeated = max(repeated, 3)
        source_exhaustion = inspect_state.get("source_exhaustion", {}) if isinstance(inspect_state, dict) else {}
        exhausted_shared = bool((source_exhaustion.get("Team_Info", {}) or {}).get("exhausted"))
        role_name = ""
        if not is_request:
            role_name = str((getattr(context_packet, "static_task_context", {}) or {}).get("role", ""))
        else:
            role_name = str((request_ctx or {}).get("role", ""))
        exhausted_role = bool((source_exhaustion.get(f"{role_name}_Info", {}) or {}).get("exhausted"))
        return {
            "epistemic_deficit": min(1.0, len(known_gaps) / 4.0),
            "build_opportunity": 1.0 if readiness.get("ready_for_build") else 0.0,
            "coordination_need": min(1.0, (len(known_gaps) + len(context_packet.team_state.get("teammate_help_signals", {}))) / 6.0) if not is_request else min(1.0, len(known_gaps) / 4.0),
            "repair_pressure": min(1.0, (len(mismatch_signals) / 3.0) + (0.5 if any(p.get("needs_repair") for p in built_state) else 0.0)),
            "loop_pressure": min(1.0, repeated / 4.0),
            "no_progress_pressure": min(1.0, no_progress_streak / 4.0),
            "observe_ineffective_pressure": min(1.0, observe_no_effect / 4.0),
            "communication_ineffective_pressure": min(1.0, communication_no_effect / 4.0),
            "readiness_blocked": 0.0 if readiness.get("ready_for_build") else 1.0,
            "contradiction_pressure": min(1.0, len(mismatch_signals) / 3.0),
            "active_incomplete_projects": min(1.0, len(active_projects) / 3.0),
            "goal_pressure": min(1.0, len(goal_stack) / 4.0),
            "dik_change_recency": 1.0 if (seconds_since_dik_change is not None and float(seconds_since_dik_change) <= 4.0) else 0.0,
            "artifact_validation_available": min(1.0, validated / 2.0),
            "teammate_relevance": min(1.0, len(context_packet.team_state.get("tom_summary", {})) / 3.0) if not is_request else 0.0,
            "shared_source_exhausted": 1.0 if exhausted_shared else 0.0,
            "role_source_exhausted": 1.0 if exhausted_role else 0.0,
        }

    @staticmethod
    def _control_state_from_request(request_packet: AgentBrainRequest) -> dict[str, Any]:
        snapshot = dict(request_packet.control_state_snapshot or {})
        task_control = dict((request_packet.task_context or {}).get("control_state") or {})
        plan_control = dict((request_packet.current_plan_summary or {}).get("control_state") or {})
        merged = {**plan_control, **task_control, **snapshot}
        mode = request_packet.control_mode or merged.get("mode") or "BOOTSTRAP"
        previous = request_packet.previous_control_mode if request_packet.previous_control_mode is not None else merged.get("previous_mode")
        dwell = request_packet.mode_dwell_steps if request_packet.mode_dwell_steps is not None else merged.get("mode_dwell_steps", 0)
        reason = request_packet.last_transition_reason if request_packet.last_transition_reason is not None else merged.get("last_transition_reason", "none")
        normalized = {
            "mode": str(mode),
            "previous_mode": previous,
            "mode_dwell_steps": int(dwell or 0),
            "last_transition_reason": str(reason or "none"),
            "recovery_active": bool(merged.get("recovery_active")),
        }
        if merged.get("last_transition_features"):
            normalized["last_transition_features"] = dict(merged.get("last_transition_features") or {})
        if merged.get("top_features"):
            normalized["top_features"] = dict(merged.get("top_features") or {})
        if merged.get("policy_snapshot"):
            normalized["last_policy_snapshot"] = dict(merged.get("policy_snapshot") or {})
        if merged.get("method_state"):
            normalized["method_state"] = dict(merged.get("method_state") or {})
        return normalized

    def _compute_mode_scores(self, features: dict[str, float], legal_types: set[str], control_state: dict[str, Any], traits: dict[str, float]) -> tuple[dict[str, float], dict[str, bool]]:
        mode_scores = {m: 0.0 for m in self.MODES}
        mode_scores["BOOTSTRAP"] = 0.3 + 1.4 * features["epistemic_deficit"] + 0.4 * features["readiness_blocked"] - 1.25 * features["shared_source_exhausted"]
        mode_scores["ACQUIRE_DIK"] = 0.4 + 1.5 * features["epistemic_deficit"] + 0.3 * features["coordination_need"]
        mode_scores["INTEGRATE_DIK"] = 0.3 + 1.2 * features["dik_change_recency"] + 0.8 * features["artifact_validation_available"]
        mode_scores["COORDINATE"] = 0.3 + 1.0 * features["coordination_need"] + 0.4 * float(traits.get("communication_propensity", 0.5))
        mode_scores["LOGISTICS"] = 0.35 + 1.5 * features["build_opportunity"] + 0.9 * features["active_incomplete_projects"]
        mode_scores["CONSTRUCT"] = 0.35 + 1.7 * features["build_opportunity"] + 0.9 * features["active_incomplete_projects"]
        mode_scores["VALIDATE"] = 0.2 + 1.4 * features["artifact_validation_available"] + 0.5 * features["goal_pressure"]
        mode_scores["REPAIR"] = 0.2 + 1.8 * features["repair_pressure"]
        mode_scores["RECOVERY"] = 0.2 + 1.8 * features["loop_pressure"] + self.policy_config.recovery_bonus * features["contradiction_pressure"]
        mode_scores["LOGISTICS"] += 1.3 * features.get("no_progress_pressure", 0.0)
        mode_scores["MONITOR"] = 0.35 + 0.2 * (1.0 - features["goal_pressure"])
        current_mode = str(control_state.get("mode") or "BOOTSTRAP")
        if current_mode in mode_scores:
            mode_scores[current_mode] += self.policy_config.dwell_bonus
        for mode in mode_scores:
            if mode != current_mode:
                mode_scores[mode] -= self.policy_config.switch_penalty
        mode_guards = {
            "CONSTRUCT": features["build_opportunity"] > 0 and any(t in legal_types for t in {ExecutableActionType.START_CONSTRUCTION.value, ExecutableActionType.CONTINUE_CONSTRUCTION.value}),
            "LOGISTICS": ExecutableActionType.TRANSPORT_RESOURCES.value in legal_types,
            "VALIDATE": ExecutableActionType.VALIDATE_CONSTRUCTION.value in legal_types and features["active_incomplete_projects"] > 0,
            "REPAIR": ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value in legal_types and features["repair_pressure"] > 0.0,
            "ACQUIRE_DIK": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value in legal_types,
            "COORDINATE": any(t in legal_types for t in {ExecutableActionType.COMMUNICATE.value, ExecutableActionType.EXTERNALIZE_PLAN.value, ExecutableActionType.REQUEST_ASSISTANCE.value}),
        }
        return mode_scores, mode_guards

    @staticmethod
    def _top_features(features: dict[str, float], *, limit: int = 4) -> dict[str, float]:
        return {k: round(v, 3) for k, v in sorted(features.items(), key=lambda item: item[1], reverse=True)[:limit]}

    def _apply_transition_guards(self, mode_scores: dict[str, float], mode_guards: dict[str, bool], features: dict[str, float], legal_types: set[str]) -> tuple[dict[str, float], list[str]]:
        notes: list[str] = []
        guarded_scores = {mode: score for mode, score in mode_scores.items() if mode_guards.get(mode, True)}
        if features.get("build_opportunity", 0.0) > 0.0 and features.get("active_incomplete_projects", 0.0) > 0.0:
            if ExecutableActionType.START_CONSTRUCTION.value in legal_types or ExecutableActionType.CONTINUE_CONSTRUCTION.value in legal_types:
                guarded_scores["CONSTRUCT"] = guarded_scores.get("CONSTRUCT", mode_scores.get("CONSTRUCT", 0.0)) + 1.25
                notes.append("build_ready_incomplete_projects_bias_construct")
            if ExecutableActionType.TRANSPORT_RESOURCES.value in legal_types:
                guarded_scores["LOGISTICS"] = guarded_scores.get("LOGISTICS", mode_scores.get("LOGISTICS", 0.0)) + 1.05
                notes.append("build_ready_incomplete_projects_bias_logistics")
            for mode in ("BOOTSTRAP", "ACQUIRE_DIK"):
                if mode in guarded_scores:
                    guarded_scores[mode] -= 1.1
                    notes.append("build_ready_deprioritize_epistemic_churn")

        if features.get("loop_pressure", 0.0) >= 0.75:
            guarded_scores["RECOVERY"] = guarded_scores.get("RECOVERY", mode_scores.get("RECOVERY", 0.0)) + 1.35
            notes.append("loop_pressure_bias_recovery")
            for mode in ("ACQUIRE_DIK", "COORDINATE"):
                if mode in guarded_scores:
                    guarded_scores[mode] -= 0.8
                    notes.append("loop_pressure_deprioritize_assistance_modes")
        if features.get("no_progress_pressure", 0.0) >= 0.5:
            guarded_scores["LOGISTICS"] = guarded_scores.get("LOGISTICS", mode_scores.get("LOGISTICS", 0.0)) + 1.2
            guarded_scores["RECOVERY"] = guarded_scores.get("RECOVERY", mode_scores.get("RECOVERY", 0.0)) + 0.8
            for mode in ("MONITOR", "COORDINATE"):
                if mode in guarded_scores:
                    guarded_scores[mode] -= 0.9
            notes.append("no_progress_pressure_bias_logistics_recovery")

        if features.get("contradiction_pressure", 0.0) > 0.0 or features.get("repair_pressure", 0.0) > 0.0:
            for mode in ("REPAIR", "VALIDATE"):
                if mode in guarded_scores:
                    guarded_scores[mode] += 0.8
            notes.append("contradiction_repair_bias")

        if features.get("shared_source_exhausted", 0.0) > 0.0 and features.get("role_source_exhausted", 0.0) > 0.0 and features.get("dik_change_recency", 0.0) <= 0.0:
            for mode in ("BOOTSTRAP", "ACQUIRE_DIK"):
                if mode in guarded_scores:
                    guarded_scores[mode] -= 1.2
            guarded_scores["MONITOR"] = guarded_scores.get("MONITOR", mode_scores.get("MONITOR", 0.0)) + 0.5
            notes.append("source_exhaustion_deprioritize_repeated_epistemic_requests")
        return guarded_scores, notes

    def _select_mode(
        self,
        *,
        mode_scores: dict[str, float],
        control_state: dict[str, Any],
        context_packet,
        features: dict[str, float],
        is_request: bool,
        rng: random.Random,
    ) -> tuple[str, dict[str, float], str]:
        current_mode = str(control_state.get("mode") or "BOOTSTRAP")
        transition_reason = "stochastic_mode_selection"
        if int(control_state.get("mode_dwell_steps", 0) or 0) < self.policy_config.min_mode_dwell_steps and current_mode in mode_scores:
            selected_mode = current_mode
            mode_probs = {k: (1.0 if k == current_mode else 0.0) for k in mode_scores}
            transition_reason = "mode_dwell_guard_hold"
        else:
            selected_mode, mode_probs = self._softmax_pick(mode_scores, self.policy_config.mode_selection_temperature, rng)

        force_bootstrap_exit, bootstrap_exit_reason = self._bootstrap_exit_triggered(
            context_packet,
            features,
            control_state,
            is_request=is_request,
        )
        if current_mode == "BOOTSTRAP" and selected_mode == "BOOTSTRAP" and force_bootstrap_exit:
            bootstrap_candidates = [
                ("ACQUIRE_DIK", mode_scores.get("ACQUIRE_DIK", -999.0) + 0.6),
                ("INTEGRATE_DIK", mode_scores.get("INTEGRATE_DIK", -999.0) + 0.2),
                ("COORDINATE", mode_scores.get("COORDINATE", -999.0)),
                ("MONITOR", mode_scores.get("MONITOR", -999.0)),
            ]
            selected_mode = sorted(bootstrap_candidates, key=lambda item: item[1], reverse=True)[0][0]
            mode_probs = {k: v for k, v in mode_probs.items() if k != "BOOTSTRAP"}
            transition_reason = bootstrap_exit_reason
        return selected_mode, mode_probs, transition_reason

    @staticmethod
    def _method_state_from_control(control_state: dict[str, Any], sim_step: int) -> dict[str, Any]:
        state = dict(control_state.get("method_state") or {})
        state.setdefault("active_method_id", None)
        state.setdefault("active_method_step", None)
        state.setdefault("active_method_instance", None)
        state.setdefault("method_started_tick", sim_step)
        state.setdefault("step_started_tick", sim_step)
        state.setdefault("step_retry_count", 0)
        state.setdefault("recent_step_outcomes", [])
        state.setdefault("method_history", [])
        state.setdefault("method_transition_history", [])
        state.setdefault("abandoned_methods", [])
        state.setdefault("method_cooldowns", {})
        state.setdefault("source_cooldowns", {})
        state.setdefault("source_exhaustion", {})
        state.setdefault("last_method_switch_reason", "initialized")
        return state

    def _candidate_methods(self, selected_mode: str, method_state: dict[str, Any]) -> list[str]:
        now = int(method_state.get("sim_step", 0) or 0)
        candidates = []
        for method_id in self.MODE_METHOD_PREFERENCES.get(selected_mode, ()):
            cooldown_until = int((method_state.get("method_cooldowns", {}) or {}).get(method_id, -1) or -1)
            if cooldown_until > now:
                continue
            method_def = self.METHOD_LIBRARY.get(method_id)
            if method_def and selected_mode in method_def.applicable_modes:
                candidates.append(method_id)
        return candidates

    def _switch_method(self, method_state: dict[str, Any], method_id: str, *, reason: str) -> None:
        now = int(method_state.get("sim_step", 0) or 0)
        prev_method = method_state.get("active_method_id")
        if prev_method and prev_method != method_id:
            transitions = list(method_state.get("method_transition_history", []))
            transitions.append({"tick": now, "from_method": prev_method, "to_method": method_id, "reason": reason})
            method_state["method_transition_history"] = transitions[-self.policy_config.max_history :]
        method_state["active_method_id"] = method_id
        method_state["active_method_instance"] = f"{method_id}-{now}"
        method_state["method_started_tick"] = now
        method_state["active_method_step"] = self.METHOD_LIBRARY[method_id].ordered_steps[0]
        method_state["step_started_tick"] = now
        method_state["step_retry_count"] = 0
        method_state["last_method_switch_reason"] = reason
        history = list(method_state.get("method_history", []))
        history.append({"tick": now, "method_id": method_id, "event": "method_started", "reason": reason})
        method_state["method_history"] = history[-self.policy_config.max_history :]

    def _select_or_continue_method(self, *, selected_mode: str, method_state: dict[str, Any], features: dict[str, float], affordances: list[dict[str, Any]], is_request: bool) -> tuple[str | None, list[str]]:
        notes: list[str] = []
        now = int(method_state.get("sim_step", 0) or 0)
        candidates = self._candidate_methods(selected_mode, method_state)
        active = method_state.get("active_method_id")
        if active in candidates:
            return active, notes
        if selected_mode == "ACQUIRE_DIK":
            source_exhaustion = dict(method_state.get("source_exhaustion", {}))
            team = dict(source_exhaustion.get("Team_Info", {}))
            role_gap = features.get("epistemic_deficit", 0.0) > 0.0
            if team.get("exhausted") and role_gap and "AcquireRoleSpecificGrounding" in candidates:
                method_state.setdefault("source_cooldowns", {})["Team_Info"] = now + 4
                self._switch_method(method_state, "AcquireRoleSpecificGrounding", reason="shared_source_exhausted_role_gap_remaining")
                notes.append("method_switched_due_to_team_info_exhaustion")
                return method_state.get("active_method_id"), notes
        if candidates:
            self._switch_method(method_state, candidates[0], reason=f"mode_method_selection:{selected_mode}")
            notes.append(f"method_started:{candidates[0]}")
            return candidates[0], notes
        if not is_request:
            method_state["last_method_switch_reason"] = f"no_method_for_mode:{selected_mode}"
        return None, notes

    def _evaluate_method_step(self, *, method_id: str | None, method_state: dict[str, Any], selected_mode: str, chosen_affordances: list[dict[str, Any]], features: dict[str, float]) -> tuple[str | None, list[str]]:
        notes: list[str] = []
        if method_id is None:
            return None, notes
        method_def = self.METHOD_LIBRARY[method_id]
        step = method_state.get("active_method_step") or method_def.ordered_steps[0]
        retry_budget = int((method_def.retry_budgets or {}).get(step, 2))
        source_exhaustion = dict(method_state.get("source_exhaustion", {}))
        team_exhausted = bool((source_exhaustion.get("Team_Info", {}) or {}).get("exhausted"))
        role_gap = features.get("epistemic_deficit", 0.0) > 0.0
        if method_id == "AcquireInitialGrounding" and team_exhausted and role_gap:
            method_state["source_cooldowns"]["Team_Info"] = int(method_state.get("sim_step", 0) or 0) + int(method_def.source_cooldown_ticks or 4)
            method_state["last_method_switch_reason"] = "team_info_exhausted_no_new_dik"
            self._switch_method(method_state, "AcquireRoleSpecificGrounding", reason="team_info_exhausted_no_new_dik")
            notes.append("source_marked_exhausted:Team_Info")
            notes.append("source_cooldown_started:Team_Info")
            return method_state.get("active_method_step"), notes
        if int(method_state.get("step_retry_count", 0) or 0) >= retry_budget:
            idx = list(method_def.ordered_steps).index(step)
            if idx + 1 < len(method_def.ordered_steps):
                new_step = method_def.ordered_steps[idx + 1]
                method_state["active_method_step"] = new_step
                method_state["step_started_tick"] = int(method_state.get("sim_step", 0) or 0)
                method_state["step_retry_count"] = 0
                notes.append(f"method_step_switched:{step}->{new_step}")
                return new_step, notes
            if selected_mode != "ACQUIRE_DIK":
                method_state["abandoned_methods"] = (list(method_state.get("abandoned_methods", [])) + [{"tick": int(method_state.get("sim_step", 0) or 0), "method_id": method_id, "reason": "retry_budget_exhausted"}])[-self.policy_config.max_history :]
            notes.append("method_abandoned_retry_budget_exhausted")
        return step, notes

    def _score_actions_for_method_step(self, *, sorted_affordances: list[dict[str, Any]], step_id: str | None, method_state: dict[str, Any]) -> dict[str, float]:
        if not step_id:
            return {}
        allowed = self.STEP_ACTION_MAP.get(step_id, set())
        scores: dict[str, float] = {}
        source_cooldowns = dict(method_state.get("source_cooldowns", {}))
        source_exhaustion = dict(method_state.get("source_exhaustion", {}))
        sim_step = int(method_state.get("sim_step", 0) or 0)
        role_source = str(method_state.get("role_source_id") or "")
        preferred_source: str | None = None
        suppressed_sources: set[str] = set()
        if step_id in {"move_to_shared_source", "inspect_shared_source"}:
            preferred_source = "Team_Info"
            if role_source:
                suppressed_sources.add(role_source)
        elif step_id in {"move_to_role_source", "inspect_role_source", "identify_role_source"} and role_source:
            preferred_source = role_source
            suppressed_sources.add("Team_Info")
        for idx, affordance in enumerate(sorted_affordances):
            action_type = str(affordance.get("action_type") or ExecutableActionType.WAIT.value)
            if allowed and action_type not in allowed:
                continue
            score = 1.0 + float(affordance.get("utility", 0.0))
            if action_type == ExecutableActionType.INSPECT_INFORMATION_SOURCE.value:
                target_id = str(affordance.get("target_id") or "")
                if int(source_cooldowns.get(target_id, -1) or -1) > sim_step:
                    score -= 6.0
                exhausted = bool((source_exhaustion.get(target_id, {}) or {}).get("exhausted"))
                if exhausted:
                    score -= 3.0
                if preferred_source:
                    score += 5.0 if target_id == preferred_source else -5.0
                if target_id in suppressed_sources:
                    score -= 4.0
            scores[f"{idx}:{action_type}"] = score
        return scores

    def _score_actions_for_mode(
        self,
        *,
        sorted_affordances: list[dict[str, Any]],
        selected_mode: str,
        features: dict[str, float],
        traits: dict[str, float],
    ) -> dict[str, float]:
        action_scores: dict[str, float] = {}
        for idx, affordance in enumerate(sorted_affordances):
            action_type = str(affordance.get("action_type") or ExecutableActionType.WAIT.value)
            utility = float(affordance.get("utility", 0.0))
            score = utility + self.MODE_ACTION_PREFERENCES.get(selected_mode, {}).get(action_type, 0.0)
            score += 0.6 * features["goal_pressure"] + 0.4 * features["build_opportunity"]
            if action_type == ExecutableActionType.REQUEST_ASSISTANCE.value:
                score += 0.9 * features["epistemic_deficit"] + 0.5 * float(traits.get("help_tendency", 0.5))
                score -= 0.8 * features["loop_pressure"]
            if action_type in {ExecutableActionType.COMMUNICATE.value, ExecutableActionType.REQUEST_ASSISTANCE.value}:
                if not bool(affordance.get("reachable", True)):
                    score -= 2.5
                if not bool(affordance.get("productive", True)):
                    score -= 1.8
                no_effect_streak = int(affordance.get("no_effect_streak", 0) or 0)
                if no_effect_streak > 0:
                    score -= min(2.2, 0.6 * no_effect_streak)
                score -= 1.4 * features.get("communication_ineffective_pressure", 0.0)
            if (
                features["build_opportunity"] > 0.0
                and features["active_incomplete_projects"] > 0.0
                and action_type == ExecutableActionType.START_CONSTRUCTION.value
            ):
                score += 1.15
            if (
                features["build_opportunity"] > 0.0
                and features["active_incomplete_projects"] > 0.0
                and features["dik_change_recency"] > 0.0
                and action_type == ExecutableActionType.TRANSPORT_RESOURCES.value
            ):
                score += 1.25
            if (
                features["shared_source_exhausted"] > 0.0
                and features["dik_change_recency"] <= 0.0
                and action_type in {ExecutableActionType.INSPECT_INFORMATION_SOURCE.value, ExecutableActionType.REQUEST_ASSISTANCE.value}
            ):
                score -= 1.25
            if action_type == ExecutableActionType.WAIT.value:
                score -= 0.35
            if action_type == ExecutableActionType.OBSERVE_ENVIRONMENT.value:
                score -= 1.2 * features.get("observe_ineffective_pressure", 0.0)
            if action_type in {ExecutableActionType.START_CONSTRUCTION.value, ExecutableActionType.CONTINUE_CONSTRUCTION.value} and features.get("readiness_blocked", 0.0) > 0.0:
                score -= 1.6
            if action_type == ExecutableActionType.TRANSPORT_RESOURCES.value and features.get("readiness_blocked", 0.0) > 0.0:
                score += 1.25
            score += self.policy_config.randomness_floor * (1.0 - (idx / max(1, len(sorted_affordances))))
            action_scores[f"{idx}:{action_type}"] = score
        return action_scores

    def _select_action(
        self,
        *,
        sorted_affordances: list[dict[str, Any]],
        action_scores: dict[str, float],
        features: dict[str, float],
        step_allowed_actions: set[str] | None,
        rng: random.Random,
    ) -> tuple[ExecutableActionType, dict[str, Any], dict[str, float]]:
        selected_action_key, action_probs = self._softmax_pick(
            action_scores or {"0:wait": 0.0},
            self.policy_config.action_selection_temperature,
            rng,
        )
        selected_idx = int(selected_action_key.split(":")[0])
        chosen = sorted_affordances[selected_idx] if sorted_affordances else {"action_type": ExecutableActionType.WAIT.value}
        if (
            features["build_opportunity"] > 0.0
            and features["active_incomplete_projects"] > 0.0
            and features["dik_change_recency"] > 0.0
            and (
                not step_allowed_actions
                or ExecutableActionType.TRANSPORT_RESOURCES.value in step_allowed_actions
            )
        ):
            transport = next((a for a in sorted_affordances if a.get("action_type") == ExecutableActionType.TRANSPORT_RESOURCES.value), None)
            if transport is not None:
                chosen = transport
        selected_action = ExecutableActionType(chosen.get("action_type", ExecutableActionType.WAIT.value))
        return selected_action, chosen, action_probs

    def _build_policy_snapshot(
        self,
        *,
        selected_mode: str,
        previous_mode: str,
        selected_action: ExecutableActionType,
        features: dict[str, float],
        mode_probs: dict[str, float],
        action_probs: dict[str, float],
        transition_reason: str,
        recovery_active: bool,
    ) -> dict[str, Any]:
        top_mode_probs = sorted(mode_probs.items(), key=lambda item: item[1], reverse=True)[: self.policy_config.max_logged_candidates]
        top_action_probs = sorted(action_probs.items(), key=lambda item: item[1], reverse=True)[: self.policy_config.max_logged_candidates]
        return {
            "current_mode": selected_mode,
            "previous_mode": previous_mode,
            "top_features": self._top_features(features, limit=4),
            "top_mode_probabilities": [(m, round(p, 4)) for m, p in top_mode_probs],
            "top_action_probabilities": [(m, round(p, 4)) for m, p in top_action_probs],
            "selected_action": selected_action.value,
            "transition_reason": transition_reason,
            "recovery_active": bool(recovery_active),
        }

    def _bootstrap_exit_triggered(self, context_packet, features: dict[str, float], control_state: dict[str, Any], *, is_request: bool) -> tuple[bool, str]:
        if is_request:
            return False, ""
        inspect_state = context_packet.individual_cognitive_state.get("inspect_state", {})
        source_exhaustion = inspect_state.get("source_exhaustion", {}) if isinstance(inspect_state, dict) else {}
        shared = source_exhaustion.get("Team_Info", {}) or {}
        role_source = f"{context_packet.static_task_context.get('role', '')}_Info"
        role_state = source_exhaustion.get(role_source, {}) or {}
        shared_exhausted = bool(shared.get("exhausted"))
        shared_no_new_streak = int(shared.get("no_new_dik_streak", 0) or 0)
        role_missing = bool(
            features["epistemic_deficit"] > 0.0
            and (not role_state.get("inspected"))
            and (not role_state.get("exhausted"))
        )
        role_still_blocked = bool(features["readiness_blocked"] > 0.0 and features["epistemic_deficit"] > 0.0)
        if shared_exhausted and (role_missing or role_still_blocked):
            return True, "bootstrap_exit_shared_source_exhausted_role_gap_remaining"
        if shared_no_new_streak >= 2 and role_missing:
            return True, "bootstrap_exit_shared_source_no_new_dik_streak"
        if features["shared_source_exhausted"] > 0 and features["dik_change_recency"] <= 0 and role_still_blocked:
            return True, "bootstrap_exit_no_marginal_shared_value"
        return False, ""

    def _apply_control_state_update(self, context_packet, selected_mode: str, features: dict[str, float], reason: str, policy_snapshot: dict[str, Any]) -> dict[str, Any]:
        control_state = context_packet.individual_cognitive_state.get("control_state", {}) if hasattr(context_packet, "individual_cognitive_state") else {}
        sim_step = int(context_packet.world_snapshot.get("sim_time", 0.0) if hasattr(context_packet, "world_snapshot") else 0)
        if not isinstance(control_state, dict):
            return {"mode": selected_mode, "mode_dwell_steps": 1}
        current_mode = str(control_state.get("mode") or "BOOTSTRAP")
        transition_applied = current_mode != selected_mode
        if not transition_applied:
            control_state["mode_dwell_steps"] = int(control_state.get("mode_dwell_steps", 0) or 0) + 1
        else:
            control_state["previous_mode"] = current_mode
            control_state["mode"] = selected_mode
            control_state["mode_entered_step"] = sim_step
            control_state["mode_dwell_steps"] = 1
            control_state["last_transition_reason"] = reason
            control_state["last_transition_features"] = self._top_features(features, limit=4)
            history = list(control_state.get("mode_history", []))
            history.append({"step": sim_step, "mode": selected_mode, "reason": reason})
            control_state["mode_history"] = history[-self.policy_config.max_history :]
            transition_history = list(control_state.get("transition_history", []))
            transition_history.append({"step": sim_step, "from_mode": current_mode, "to_mode": selected_mode, "reason": reason})
            control_state["transition_history"] = transition_history[-self.policy_config.max_history :]
        if transition_applied:
            control_state["last_transition_reason"] = reason
            control_state["last_transition_features"] = self._top_features(features, limit=4)
        control_state["recovery_active"] = bool(selected_mode == "RECOVERY" or policy_snapshot.get("recovery_active"))
        control_state["last_policy_snapshot"] = dict(policy_snapshot)
        control_state.setdefault("method_state", dict(control_state.get("method_state") or {}))
        return control_state

    def _policy_core(self, context_packet, *, control_state: dict[str, Any] | None = None, is_request: bool = False) -> dict[str, Any]:
        # Authoritative fallback controller pipeline:
        # 1) derive features, 2) select/continue macro mode, 3) select/continue method,
        # 4) select/advance method step, 5) choose legal action for that step,
        # 6) emit policy snapshot, 7) hand action to executor bridge (Agent).
        affordances = list(context_packet.action_affordances if not is_request else context_packet.allowed_actions)
        sorted_affordances = sorted(affordances, key=lambda a: float(a.get("utility", 0.0)), reverse=True)
        legal_types = {a.get("action_type") for a in affordances if a.get("action_type")}
        traits = (context_packet.individual_cognitive_state.get("traits", {}) if not is_request else {}) or {}
        features = self._extract_features(context_packet, is_request=is_request)
        control_state = dict(control_state or (context_packet.individual_cognitive_state.get("control_state", {}) if not is_request else {}))
        if not control_state:
            control_state = {"mode": "BOOTSTRAP", "mode_dwell_steps": 0}
        mode_scores, mode_guards = self._compute_mode_scores(features, legal_types, control_state, traits)
        mode_scores, guard_notes = self._apply_transition_guards(mode_scores, mode_guards, features, legal_types)
        rng_seed = f"{getattr(context_packet, 'request_id', 'ctx')}-{len(affordances)}-{control_state.get('mode')}-{int(features['goal_pressure']*100)}"
        rng = random.Random(rng_seed)
        selected_mode, mode_probs, transition_reason = self._select_mode(
            mode_scores=mode_scores,
            control_state=control_state,
            context_packet=context_packet,
            features=features,
            is_request=is_request,
            rng=rng,
        )
        sim_step = int((context_packet.world_snapshot.get("sim_time", 0.0) if not is_request else context_packet.sim_time) or 0)
        method_state = self._method_state_from_control(control_state, sim_step)
        method_state["sim_step"] = sim_step
        role_name = str(
            (
                (context_packet.static_task_context if not is_request else (context_packet.task_context or {}))
                or {}
            ).get("role", "")
        )
        method_state["role_source_id"] = f"{role_name}_Info" if role_name else ""
        inspect_state = (
            dict((context_packet.individual_cognitive_state.get("inspect_state") or {}))
            if not is_request
            else dict((context_packet.task_context or {}).get("inspect_state") or {})
        )
        method_state["source_exhaustion"] = dict(inspect_state.get("source_exhaustion") or method_state.get("source_exhaustion") or {})
        method_id, method_notes = self._select_or_continue_method(
            selected_mode=selected_mode,
            method_state=method_state,
            features=features,
            affordances=sorted_affordances,
            is_request=is_request,
        )
        step_id, step_notes = self._evaluate_method_step(
            method_id=method_id,
            method_state=method_state,
            selected_mode=selected_mode,
            chosen_affordances=sorted_affordances,
            features=features,
        )
        action_scores = self._score_actions_for_mode(
            sorted_affordances=sorted_affordances,
            selected_mode=selected_mode,
            features=features,
            traits=traits,
        )
        step_action_scores = self._score_actions_for_method_step(
            sorted_affordances=sorted_affordances,
            step_id=step_id,
            method_state=method_state,
        )
        if step_action_scores:
            action_scores.update({k: action_scores.get(k, 0.0) + (v * 1.6) for k, v in step_action_scores.items()})
        selected_action, chosen, action_probs = self._select_action(
            sorted_affordances=sorted_affordances,
            action_scores=action_scores,
            features=features,
            step_allowed_actions=set(self.STEP_ACTION_MAP.get(step_id, set())) if step_id else None,
            rng=rng,
        )
        reason = f"mode={selected_mode};transition={transition_reason}"
        assumptions = [f"policy_mode={selected_mode}", f"mode_dwell={control_state.get('mode_dwell_steps', 0)}"]
        if method_id:
            assumptions.append(f"method={method_id}")
        if step_id:
            assumptions.append(f"step={step_id}")
        if selected_action == ExecutableActionType.TRANSPORT_RESOURCES:
            assumptions.append(f"duration_s={chosen.get('duration_s', 30)}")
        assumptions.extend([f"guard={note}" for note in guard_notes[:2]])
        assumptions.extend([f"method_note={note}" for note in (method_notes + step_notes)[:3]])
        recovery_active = selected_mode == "RECOVERY" or features.get("loop_pressure", 0.0) >= 0.75
        policy_snapshot = self._build_policy_snapshot(
            selected_mode=selected_mode,
            previous_mode=str(control_state.get("mode") or "BOOTSTRAP"),
            selected_action=selected_action,
            features=features,
            mode_probs=mode_probs,
            action_probs=action_probs,
            transition_reason=transition_reason,
            recovery_active=recovery_active,
        )
        if not is_request:
            updated_control = self._apply_control_state_update(context_packet, selected_mode, features, reason, policy_snapshot)
            context_packet.individual_cognitive_state["control_state"] = updated_control
        else:
            updated_control = control_state
        method_state.pop("sim_step", None)
        updated_control["method_state"] = dict(method_state)
        top_mode_probs = sorted(mode_probs.items(), key=lambda item: item[1], reverse=True)[: self.policy_config.max_logged_candidates]
        top_action_probs = sorted(action_probs.items(), key=lambda item: item[1], reverse=True)[: self.policy_config.max_logged_candidates]
        return {
            "selected_action": selected_action,
            "chosen_affordance": chosen,
            "selected_mode": selected_mode,
            "features": features,
            "mode_probs": top_mode_probs,
            "action_probs": top_action_probs,
            "reason": reason,
            "assumptions": assumptions,
            "control_state": updated_control,
            "policy_snapshot": policy_snapshot,
            "selected_method": method_id,
            "selected_step": step_id,
            "method_notes": method_notes + step_notes,
            "goal_update": "execute_build" if selected_mode in {"LOGISTICS", "CONSTRUCT"} else ("repair_detected_mismatch" if selected_mode == "REPAIR" else "maintain_forward_progress"),
        }

    def decide(self, context_packet):
        result = self._policy_core(context_packet, is_request=False)
        communication_intent = CommunicationIntent.TIP if result["selected_mode"] in {"BOOTSTRAP", "ACQUIRE_DIK"} else None
        if result["selected_mode"] == "COORDINATE":
            communication_intent = CommunicationIntent.TPP
        return BrainDecision(
            selected_action=result["selected_action"],
            target_id=result["chosen_affordance"].get("target_id"),
            target_zone=result["chosen_affordance"].get("target_zone"),
            goal_update=result["goal_update"],
            plan_steps=[
                f"macro_state:{result['selected_mode']}",
                f"method:{result.get('selected_method') or 'none'}",
                f"step:{result.get('selected_step') or 'none'}",
                "execute legal action aligned with weighted affordance policy",
            ],
            reason_summary=result["reason"],
            communication_intent=communication_intent,
            confidence=max(prob for _, prob in result["action_probs"]) if result["action_probs"] else 0.6,
            assumptions=result["assumptions"]
            + [
                f"mode_probs={[(m, round(p,3)) for m,p in result['mode_probs']]}",
                f"action_probs={[(a, round(p,3)) for a,p in result['action_probs']]}",
                f"features={{{', '.join(f'{k}:{round(v,2)}' for k,v in sorted(result['features'].items(), key=lambda i: i[1], reverse=True)[:4])}}}",
            ],
        )

    def generate_plan(self, request_packet: AgentBrainRequest) -> AgentBrainResponse:
        seed_control_state = self._control_state_from_request(request_packet)
        result = self._policy_core(request_packet, control_state=seed_control_state, is_request=True)
        step = PlannedActionStep(
            step_index=0,
            action_type=result["selected_action"],
            target_id=result["chosen_affordance"].get("target_id"),
            target_zone=result["chosen_affordance"].get("target_zone"),
            expected_purpose=result["reason"],
        )
        payload = {
            "response_id": f"rule-{request_packet.request_id}",
            "agent_id": request_packet.agent_id,
            "plan": {
                "plan_id": f"plan-{request_packet.request_id}",
                "plan_horizon": int(request_packet.planning_horizon_config.get("max_steps", 3)),
                "ordered_goals": [
                    {
                        "goal_id": "stability",
                        "description": "maintain forward progress with legal actions",
                        "priority": 0.7,
                        "status": "active",
                        "source": "planner",
                    }
                ],
                "ordered_actions": [step.__dict__],
                "next_action": step.__dict__,
                "confidence": max(prob for _, prob in result["action_probs"]) if result["action_probs"] else 0.65,
                "plan_method_id": "rule_brain_hierarchical_policy_v1",
                "notes": [
                    f"mode={result['selected_mode']}",
                    f"method={result.get('selected_method')}",
                    f"step={result.get('selected_step')}",
                    f"mode_probs={[(m, round(p, 3)) for m, p in result['mode_probs']]}",
                    f"action_probs={[(a, round(p, 3)) for a, p in result['action_probs']]}",
                    f"method_notes={result.get('method_notes', [])}",
                    f"policy_snapshot={result.get('policy_snapshot', {})}",
                ],
            },
            "explanation": (
                f"rule fallback selected {result['selected_action'].value} in {result['selected_mode']} mode"
                if request_packet.request_explanation
                else None
            ),
            "confidence": max(prob for _, prob in result["action_probs"]) if result["action_probs"] else 0.65,
        }
        return AgentBrainResponse.from_dict(payload)

    def generate_dik_integration(self, request_packet: AgentDIKIntegrationRequest) -> AgentDIKIntegrationResponse:
        held_ids = set(request_packet.held_data_ids) | set(request_packet.held_information_ids) | set(request_packet.held_knowledge_ids)
        held_information = set(request_packet.held_information_ids)
        held_knowledge = set(request_packet.held_knowledge_ids)
        max_items = max(1, int(request_packet.max_candidates_per_type or 8))

        def _confidence_for(required_ids: list[str]) -> float:
            count = max(1, len([rid for rid in required_ids if str(rid).strip()]))
            return max(0.55, min(0.95, 0.55 + (0.08 * min(count, 5))))

        candidate_information_updates: list[dict[str, Any]] = []
        for candidate_id in request_packet.candidate_information_ids:
            if candidate_id in held_information:
                continue
            grounding_paths = list((request_packet.candidate_information_grounding or {}).get(candidate_id, []))
            for path in grounding_paths:
                required_inputs = [str(x) for x in path.get("required_inputs", []) if str(x).strip()]
                if required_inputs and all(req in held_ids for req in required_inputs):
                    candidate_information_updates.append(
                        {
                            "candidate_id": candidate_id,
                            "evidence_ids": required_inputs,
                            "justification": f"deterministic_derivation:{path.get('derivation_id', 'unknown')}",
                            "confidence": _confidence_for(required_inputs),
                        }
                    )
                    break
            if len(candidate_information_updates) >= max_items:
                break

        candidate_knowledge_updates: list[dict[str, Any]] = []
        for candidate_id in request_packet.candidate_knowledge_ids:
            if candidate_id in held_knowledge:
                continue
            grounding_paths = list((request_packet.candidate_knowledge_grounding or {}).get(candidate_id, []))
            for path in grounding_paths:
                required_inputs = [str(x) for x in path.get("required_inputs", []) if str(x).strip()]
                if required_inputs and all(req in held_ids for req in required_inputs):
                    candidate_knowledge_updates.append(
                        {
                            "candidate_id": candidate_id,
                            "evidence_ids": required_inputs,
                            "justification": f"deterministic_derivation:{path.get('derivation_id', 'unknown')}",
                            "confidence": _confidence_for(required_inputs),
                        }
                    )
                    break
            if len(candidate_knowledge_updates) >= max_items:
                break

        candidate_rule_supports: list[dict[str, Any]] = []
        for candidate_id in request_packet.candidate_rule_ids:
            if candidate_id in held_knowledge:
                continue
            grounding = dict((request_packet.candidate_rule_grounding or {}).get(candidate_id, {}))
            required = [str(x) for x in grounding.get("required_evidence_ids", []) if str(x).strip()]
            if required and all(req in held_ids for req in required):
                candidate_rule_supports.append(
                    {
                        "candidate_id": candidate_id,
                        "evidence_ids": required,
                        "justification": "deterministic_rule_prerequisites_satisfied",
                        "confidence": _confidence_for(required),
                    }
                )
            if len(candidate_rule_supports) >= max_items:
                break

        total = len(candidate_information_updates) + len(candidate_knowledge_updates) + len(candidate_rule_supports)
        confidence = 0.0 if total == 0 else max(
            0.55,
            min(
                0.95,
                sum(
                    [x["confidence"] for x in candidate_information_updates]
                    + [x["confidence"] for x in candidate_knowledge_updates]
                    + [x["confidence"] for x in candidate_rule_supports]
                )
                / float(total),
            ),
        )
        return AgentDIKIntegrationResponse.from_dict(
            {
                "response_id": f"dik-rule-{request_packet.request_id}",
                "agent_id": request_packet.agent_id,
                "candidate_information_updates": candidate_information_updates,
                "candidate_knowledge_updates": candidate_knowledge_updates,
                "candidate_rule_supports": candidate_rule_supports,
                "unresolved_gaps": list(request_packet.unresolved_gaps[:6]),
                "contradictions": list(request_packet.contradiction_signals[:6]),
                "summary": (
                    "Rule deterministic DIK integration completed."
                    if total
                    else "Rule deterministic DIK integration found no fully grounded candidates."
                ),
                "confidence": confidence,
            }
        )


class OllamaLocalBrainProvider(BrainProvider):
    """Optional local model backend with safe fallback to RuleBrain."""

    def __init__(self, config: BrainBackendConfig, fallback: BrainProvider):
        self.config = config
        self.fallback = fallback
        self.last_outcome = {"fallback": False, "reason": None, "latency_ms": None, "outcome_category": "llm_success", "result_source": "ollama"}
        self.last_trace: Dict[str, Any] = {}

    def warmup_probe(self) -> Dict[str, Any]:
        endpoint = f"{self.config.local_base_url.rstrip('/')}{self.config.local_endpoint}"
        started = time.perf_counter()
        warmup_timeout_s = float(getattr(self.config, "warmup_timeout_s", self.config.timeout_s) or self.config.timeout_s)
        payload = {
            "model": self.config.local_model,
            "messages": [{"role": "user", "content": "{}"}],
            "temperature": 0,
            "max_tokens": 8,
            "response_format": {"type": "json_object"},
        }
        try:
            req = request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=warmup_timeout_s) as response:
                raw = response.read().decode("utf-8")
            return {
                "ok": bool(raw),
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "endpoint": endpoint,
                "model": self.config.local_model,
                "warmup_timeout_s": warmup_timeout_s,
                "probe_type": "startup_warmup",
                "warmup_timeout": False,
            }
        except TimeoutError as exc:  # noqa: PERF203
            return {
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "endpoint": endpoint,
                "model": self.config.local_model,
                "warmup_timeout_s": warmup_timeout_s,
                "probe_type": "startup_warmup",
                "warmup_timeout": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        except Exception as exc:  # noqa: BLE001 - warmup probe should not fail simulation startup
            return {
                "ok": False,
                "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
                "endpoint": endpoint,
                "model": self.config.local_model,
                "warmup_timeout_s": warmup_timeout_s,
                "probe_type": "startup_warmup",
                "warmup_timeout": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def backend_settings(self) -> Dict[str, Any]:
        return {
            "configured_backend": self.config.backend,
            "provider_class": self.__class__.__name__,
            "local_model_name": self.config.local_model,
            "local_base_url": self.config.local_base_url,
            "local_endpoint": self.config.local_endpoint,
            "timeout_s": self.config.timeout_s,
            "completion_max_tokens": self.config.completion_max_tokens,
            "startup_completion_max_tokens": self.config.startup_completion_max_tokens,
            "unrestricted_local_qwen_mode": bool(getattr(self.config, "unrestricted_local_qwen_mode", False)),
            "fallback_backend": self.config.fallback_backend,
        }

    def _log_debug(self, message: str, payload: Dict[str, Any]) -> None:
        if self.config.debug:
            LOGGER.info("%s %s", message, payload)

    def _build_request_payload(self, request_packet: AgentBrainRequest) -> Dict[str, Any]:
        max_actions = 14
        max_observations = 6
        max_goal_stack = 5
        max_rules = 8
        max_derivations = 6
        max_artifacts = 4
        max_steps = max(1, int(request_packet.planning_horizon_config.get("max_steps", 3) or 3))

        compact_contract = {
            "request_id": request_packet.request_id,
            "agent": {
                "agent_id": request_packet.agent_id,
                "display_name": request_packet.display_name,
                "agent_label": request_packet.agent_label,
                "phase": request_packet.phase,
            },
            "sim": {
                "tick": int(request_packet.tick),
                "sim_time": float(request_packet.sim_time),
                "task_id": request_packet.task_id,
            },
            "planning_horizon_config": {"max_steps": max_steps},
            "local_context_summary": _truncate_text(request_packet.local_context_summary, max_chars=220),
            "local_observations": [_truncate_text(item, max_chars=140) for item in request_packet.local_observations[:max_observations]],
            "working_memory_summary": _bounded_json_value(request_packet.working_memory_summary, max_chars=1200),
            "current_goal_stack": [_bounded_json_value(item, max_chars=320) for item in request_packet.current_goal_stack[:max_goal_stack]],
            "current_plan_summary": _bounded_json_value(request_packet.current_plan_summary, max_chars=700),
            "allowed_actions": [_bounded_json_value(item, max_chars=320) for item in request_packet.allowed_actions[:max_actions]],
            "rule_context": [_truncate_text(item, max_chars=120) for item in request_packet.rule_context[:max_rules]],
            "derivation_context": [_truncate_text(item, max_chars=120) for item in request_packet.derivation_context[:max_derivations]],
            "artifact_context": [_bounded_json_value(item, max_chars=300) for item in request_packet.artifact_context[:max_artifacts]],
            "bootstrap_summary": _bounded_json_value(request_packet.bootstrap_summary, max_chars=500) if request_packet.bootstrap_summary else None,
        }
        return {
            "model": self.config.local_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object that matches AgentBrainResponse. "
                        "Reason internally if needed, but output only the final JSON object. "
                        "Output JSON only. No markdown fences. No prose before or after JSON. "
                        "Do not output reasoning, analysis, or thinking text. "
                        "Do not place the answer in a reasoning/thinking field. "
                        "Do not leave message.content empty; put the final JSON object in message.content. "
                        "Do not wrap the object under keys like response/result/data. "
                        "The top-level JSON must be the planner response object. "
                        "plan.next_action must be an object, not a string. "
                        "plan.ordered_actions must be a list of action objects. "
                        "Include plan.next_action and non-empty plan.ordered_actions. "
                        "Keep plan concise, action-oriented, and bounded to immediate horizon."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(compact_contract, separators=(",", ":"), default=str),
                },
            ],
            "temperature": 0,
            "max_tokens": int(self.config.completion_max_tokens),
            "response_format": {"type": "json_object"},
        }

    def _parse_response(self, response_json: Dict[str, Any]) -> Dict[str, Any]:
        choices = response_json.get("choices") or []
        if not choices:
            raise ValueError("local backend response missing choices")
        choice0 = choices[0] if isinstance(choices[0], dict) else {}
        message = choice0.get("message", {}) if isinstance(choice0.get("message"), dict) else {}
        content = message.get("content")
        if isinstance(content, dict):
            return {"payload": content, "parse_source": "choices[0].message.content(dict)"}

        candidate_texts: list[tuple[str, str]] = []
        for source, value in [
            ("choices[0].message.content", content),
            ("choices[0].message.output_text", message.get("output_text")),
            ("choices[0].text", choice0.get("text")),
            ("response", response_json.get("response")),
            ("choices[0].message.reasoning", message.get("reasoning")),
            ("choices[0].message.thinking", message.get("thinking")),
            ("choices[0].reasoning", choice0.get("reasoning")),
        ]:
            extracted = _content_text(value)
            if extracted:
                candidate_texts.append((source, extracted))
        if not candidate_texts:
            raise ValueError("local backend response content is empty")

        for source, candidate in candidate_texts:
            cleaned = candidate.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            try:
                return {"payload": json.loads(cleaned), "parse_source": source}
            except json.JSONDecodeError:
                continue
        raise ValueError("local backend response did not contain parseable JSON")

    def _normalize_action_type(self, value: Any, note_sink: list[dict[str, Any]], *, field: str) -> str | None:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if not raw:
            return None
        if raw in ExecutableActionType._value2member_map_:
            return raw
        lowered = raw.lower()
        if lowered in RUNTIME_ACTION_ALIASES:
            mapped = RUNTIME_ACTION_ALIASES[lowered]
            note_sink.append(
                {
                    "step": "normalized_action_alias",
                    "field": field,
                    "original_action": raw,
                    "normalized_action": mapped,
                    "automatic": True,
                }
            )
            return mapped
        if lowered == "build":
            note_sink.append(
                {
                    "step": "ambiguous_action_alias_unresolved",
                    "field": field,
                    "original_action": raw,
                    "note": "build alias is ambiguous; no automatic mapping performed",
                }
            )
        return None

    def _extract_action_type_from_entry(self, entry: dict[str, Any]) -> tuple[str | None, str | None]:
        for key in ("action_type", "action", "type", "action_name", "name"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return key, value
        return None, None

    def _find_nested_planner_candidate(self, payload: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any] | None:
        queue: list[tuple[str, Any]] = [("top", payload)]
        seen: set[int] = set()
        while queue:
            path, node = queue.pop(0)
            if not isinstance(node, dict):
                continue
            node_id = id(node)
            if node_id in seen:
                continue
            seen.add(node_id)
            if path != "top" and (
                isinstance(node.get("plan"), dict)
                or isinstance(node.get("next_action"), (dict, str))
                or isinstance(node.get("ordered_actions"), list)
            ):
                steps.append({"step": "found_nested_planner_candidate", "candidate_path": path})
                return node
            for key, value in node.items():
                if isinstance(value, dict):
                    queue.append((f"{path}.{key}" if path != "top" else key, value))
        return None

    def _extract_minimal_action_payload(
        self,
        payload: Any,
        *,
        steps: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not isinstance(payload, dict):
            return None, None
        plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else payload
        next_action = plan.get("next_action")
        if isinstance(next_action, str):
            canonical = self._normalize_action_type(next_action, steps, field="salvage.next_action")
            if canonical is not None:
                steps.append({"step": "minimal_salvage_from_next_action_string", "normalized_action": canonical})
                return {"step_index": 0, "action_type": canonical}, "next_action"
        if isinstance(next_action, dict):
            key, action_value = self._extract_action_type_from_entry(next_action)
            if action_value is not None:
                canonical = self._normalize_action_type(action_value, steps, field="salvage.next_action")
                if canonical is not None:
                    action = dict(next_action)
                    action["action_type"] = canonical
                    action.setdefault("step_index", 0)
                    if key and key != "action_type":
                        steps.append(
                            {
                                "step": "minimal_salvage_promoted_next_action_alias",
                                "source_field": key,
                                "normalized_action": canonical,
                            }
                        )
                    steps.append({"step": "minimal_salvage_from_next_action_object", "normalized_action": canonical})
                    return action, "next_action"
        ordered = plan.get("ordered_actions")
        if isinstance(ordered, list):
            for idx, entry in enumerate(ordered):
                if isinstance(entry, str):
                    canonical = self._normalize_action_type(entry, steps, field=f"salvage.ordered_actions[{idx}]")
                    if canonical is None:
                        continue
                    steps.append(
                        {
                            "step": "minimal_salvage_from_ordered_action_string",
                            "index": idx,
                            "normalized_action": canonical,
                        }
                    )
                    return {"step_index": idx, "action_type": canonical}, "ordered_actions_first_usable"
                if isinstance(entry, dict):
                    key, action_value = self._extract_action_type_from_entry(entry)
                    if action_value is None:
                        continue
                    canonical = self._normalize_action_type(action_value, steps, field=f"salvage.ordered_actions[{idx}]")
                    if canonical is None:
                        continue
                    action = dict(entry)
                    action["action_type"] = canonical
                    action.setdefault("step_index", idx)
                    if key and key != "action_type":
                        steps.append(
                            {
                                "step": "minimal_salvage_promoted_ordered_action_alias",
                                "index": idx,
                                "source_field": key,
                                "normalized_action": canonical,
                            }
                        )
                    steps.append({"step": "minimal_salvage_from_ordered_action_object", "index": idx, "normalized_action": canonical})
                    return action, "ordered_actions_first_usable"
        return None, None

    def _build_minimal_salvage_response(
        self,
        request_packet: AgentBrainRequest,
        *,
        source_payload: Any,
        failure_reason: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str | None]:
        steps: list[dict[str, Any]] = []
        action, salvage_source = self._extract_minimal_action_payload(source_payload, steps=steps)
        if action is None:
            return None, steps, None
        plan_horizon = max(1, int(request_packet.planning_horizon_config.get("max_steps", 3) or 3))
        payload = {
            "response_id": f"salvaged-{request_packet.request_id}",
            "agent_id": request_packet.agent_id,
            "plan": {
                "plan_id": f"salvaged-plan-{request_packet.request_id}",
                "plan_horizon": plan_horizon,
                "ordered_goals": [],
                "ordered_actions": [dict(action)],
                "next_action": dict(action),
                "confidence": 0.35,
                "notes": [f"minimal_action_salvage:{failure_reason}"],
            },
            "confidence": 0.35,
        }
        steps.append(
            {
                "step": "constructed_minimal_salvage_response",
                "salvage_source": salvage_source,
                "failure_reason": failure_reason,
            }
        )
        return payload, steps, "accepted_via_minimal_action_salvage"

    def _normalize_payload(self, payload: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
        steps: list[dict[str, Any]] = []
        if not isinstance(payload, dict):
            return None, steps, "rejected_schema_invalid"
        working = deepcopy(payload)
        wrapper_source = "top_level_plan"
        wrapper_payload: dict[str, Any] | None = None
        if "plan" not in working:
            for wrapper in ("response", "result", "data"):
                wrapped = working.get(wrapper)
                if isinstance(wrapped, dict) and isinstance(wrapped.get("plan"), dict):
                    working = {**working, **wrapped}
                    wrapper_payload = wrapped
                    wrapper_source = f"{wrapper}.plan"
                    steps.append({"step": "unwrapped_payload", "wrapper_source": wrapper_source})
                    break
        if "plan" not in working:
            nested_candidate = self._find_nested_planner_candidate(working, steps)
            if isinstance(nested_candidate, dict):
                merged = dict(working)
                if isinstance(nested_candidate.get("plan"), dict):
                    merged["plan"] = deepcopy(nested_candidate.get("plan"))
                for field in ("next_action", "ordered_actions"):
                    if field in nested_candidate and field not in merged:
                        merged[field] = deepcopy(nested_candidate.get(field))
                working = merged
                wrapper_payload = nested_candidate
                wrapper_source = "nested_candidate"
                steps.append({"step": "unwrapped_nested_planner_candidate"})
        if "plan" not in working or not isinstance(working.get("plan"), dict):
            return None, steps, "rejected_missing_plan"

        plan = working.get("plan") or {}
        sibling_sources = [working]
        if isinstance(wrapper_payload, dict):
            sibling_sources.insert(0, wrapper_payload)
        for source in sibling_sources:
            if not isinstance(source, dict):
                continue
            if "next_action" not in plan and isinstance(source.get("next_action"), (dict, str)):
                plan["next_action"] = deepcopy(source.get("next_action"))
                steps.append(
                    {
                        "step": "promoted_sibling_next_action_to_plan",
                        "from": "wrapper" if source is wrapper_payload else "top_level",
                        "source_key": "next_action",
                    }
                )
            if "ordered_actions" not in plan and isinstance(source.get("ordered_actions"), list):
                plan["ordered_actions"] = deepcopy(source.get("ordered_actions"))
                steps.append(
                    {
                        "step": "promoted_sibling_ordered_actions_to_plan",
                        "from": "wrapper" if source is wrapper_payload else "top_level",
                        "source_key": "ordered_actions",
                    }
                )
        ordered = plan.get("ordered_actions")
        normalized_ordered: list[dict[str, Any]] = []
        if isinstance(ordered, list):
            for idx, entry in enumerate(ordered):
                if isinstance(entry, str):
                    normalized = {"step_index": idx, "action_type": entry}
                    steps.append({"step": "ordered_action_string_to_object", "index": idx, "original": entry})
                elif isinstance(entry, dict):
                    normalized = dict(entry)
                else:
                    steps.append({"step": "ordered_action_unusable_entry", "index": idx, "entry_type": type(entry).__name__})
                    continue

                key, action_value = self._extract_action_type_from_entry(normalized)
                if action_value is None:
                    steps.append({"step": "ordered_action_missing_action_type", "index": idx})
                    continue
                canonical = self._normalize_action_type(action_value, steps, field=f"plan.ordered_actions[{idx}]")
                if canonical is None:
                    steps.append({"step": "ordered_action_illegal_action", "index": idx, "original_action": action_value})
                    continue
                normalized["action_type"] = canonical
                if key and key != "action_type":
                    steps.append(
                        {
                            "step": "ordered_action_alias_field_promoted",
                            "index": idx,
                            "original_field": key,
                            "normalized_action": canonical,
                        }
                    )
                normalized.setdefault("step_index", idx)
                if "step_index" not in entry if isinstance(entry, dict) else True:
                    steps.append({"step": "ordered_action_step_index_filled", "index": idx, "value": idx})
                normalized_ordered.append(normalized)
        elif ordered is not None:
            steps.append({"step": "ordered_actions_not_list", "entry_type": type(ordered).__name__})

        if normalized_ordered:
            plan["ordered_actions"] = normalized_ordered

        next_action = plan.get("next_action")
        if isinstance(next_action, str):
            canonical_next = self._normalize_action_type(next_action, steps, field="plan.next_action")
            if canonical_next is None:
                return None, steps, "rejected_unusable_next_action"
            matched = next((item for item in normalized_ordered if item.get("action_type") == canonical_next), None)
            if matched:
                plan["next_action"] = dict(matched)
                steps.append(
                    {
                        "step": "resolved_string_next_action_from_ordered_actions",
                        "original_action": next_action,
                        "normalized_action": canonical_next,
                    }
                )
            else:
                plan["next_action"] = {"step_index": 0, "action_type": canonical_next}
                steps.append(
                    {
                        "step": "constructed_minimal_next_action",
                        "original_action": next_action,
                        "normalized_action": canonical_next,
                    }
                )
        elif isinstance(next_action, dict):
            key, action_value = self._extract_action_type_from_entry(next_action)
            if action_value is None:
                return None, steps, "rejected_unusable_next_action"
            canonical = self._normalize_action_type(action_value, steps, field="plan.next_action")
            if canonical is None:
                return None, steps, "rejected_illegal_action"
            next_action["action_type"] = canonical
            next_action.setdefault("step_index", 0)
            if key and key != "action_type":
                steps.append({"step": "next_action_alias_field_promoted", "original_field": key, "normalized_action": canonical})
            plan["next_action"] = next_action
        elif isinstance(plan.get("ordered_actions"), list):
            recovered_next = next((item for item in plan.get("ordered_actions", []) if isinstance(item, dict)), None)
            if recovered_next is not None:
                plan["next_action"] = dict(recovered_next)
                steps.append({"step": "promoted_first_ordered_action_to_next_action"})
            else:
                return None, steps, "rejected_unusable_next_action"
        else:
            return None, steps, "rejected_unusable_next_action"

        if not isinstance(plan.get("ordered_actions"), list) or not plan.get("ordered_actions"):
            if isinstance(plan.get("next_action"), dict):
                plan["ordered_actions"] = [dict(plan["next_action"])]
                steps.append({"step": "filled_ordered_actions_from_next_action"})
            else:
                return None, steps, "rejected_schema_invalid"

        working["plan"] = plan
        disposition = "accepted_as_is"
        if wrapper_source != "top_level_plan":
            disposition = "accepted_after_unwrap"
        if steps and disposition == "accepted_as_is":
            disposition = "accepted_after_repair"
        elif steps and disposition == "accepted_after_unwrap":
            disposition = "accepted_after_repair"
        return working, steps, disposition

    def _build_schema_repair_payload(
        self,
        *,
        invalid_payload: Any,
        failure_reason: str,
    ) -> Dict[str, Any]:
        repair_contract = {
            "failure_reason": failure_reason,
            "invalid_payload": invalid_payload,
            "required_shape": "Return exactly one top-level AgentBrainResponse JSON object.",
            "rules": [
                "Output JSON only.",
                "No wrapper keys like response/result/data.",
                "Include plan as an object.",
                "Include plan.next_action as an object with a legal action_type.",
                "Include non-empty plan.ordered_actions.",
            ],
        }
        return {
            "model": self.config.local_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Your previous response was structurally invalid. "
                        "Repair it and return exactly one valid top-level AgentBrainResponse JSON object. JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(repair_contract, separators=(",", ":"), default=str),
                },
            ],
            "temperature": 0,
            "max_tokens": min(2048, int(self.config.completion_max_tokens)),
            "response_format": {"type": "json_object"},
        }

    def _build_dik_integration_payload(self, request_packet: AgentDIKIntegrationRequest) -> Dict[str, Any]:
        contract = {
            "request_id": request_packet.request_id,
            "agent_id": request_packet.agent_id,
            "display_name": request_packet.display_name,
            "phase": request_packet.phase,
            "tick": int(request_packet.tick),
            "sim_time": float(request_packet.sim_time),
            "trigger_reason": request_packet.trigger_reason,
            "held_data_ids": list(request_packet.held_data_ids),
            "held_information_ids": list(request_packet.held_information_ids),
            "held_knowledge_ids": list(request_packet.held_knowledge_ids),
            "recent_new_item_ids": list(request_packet.recent_new_item_ids[:24]),
            "recent_communication_ids": list(request_packet.recent_communication_ids[:16]),
            "recent_artifact_ids": list(request_packet.recent_artifact_ids[:16]),
            "unresolved_gaps": list(request_packet.unresolved_gaps[:12]),
            "contradiction_signals": list(request_packet.contradiction_signals[:12]),
            "candidate_information_ids": list(request_packet.candidate_information_ids),
            "candidate_knowledge_ids": list(request_packet.candidate_knowledge_ids),
            "candidate_rule_ids": list(request_packet.candidate_rule_ids),
            "max_candidates_per_type": int(request_packet.max_candidates_per_type),
        }
        return {
            "model": self.config.local_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one JSON object matching AgentDIKIntegrationResponse. "
                        "Output JSON only with no markdown or prose. "
                        "Use only held IDs provided. Do not invent unsupported candidates. "
                        "Each candidate must include candidate_id and evidence_ids from held IDs. "
                        "Keep candidate updates separate from unresolved gaps and contradictions."
                    ),
                },
                {"role": "user", "content": json.dumps(contract, separators=(",", ":"), default=str)},
            ],
            "temperature": 0,
            "max_tokens": min(2048, int(self.config.completion_max_tokens)),
            "response_format": {"type": "json_object"},
        }

    def _normalize_dik_integration_payload(self, payload: Any) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
        steps: list[dict[str, Any]] = []
        if not isinstance(payload, dict):
            return None, steps, "rejected_schema_invalid"
        working = deepcopy(payload)
        if "response" in working and isinstance(working.get("response"), dict):
            working = {**working, **working["response"]}
            steps.append({"step": "unwrapped_response_wrapper"})
        for field in ("candidate_information_updates", "candidate_knowledge_updates", "candidate_rule_supports"):
            value = working.get(field)
            if value is None:
                working[field] = []
                continue
            if not isinstance(value, list):
                steps.append({"step": "coerced_candidate_bucket_to_empty", "field": field, "source_type": type(value).__name__})
                working[field] = []
                continue
            cleaned = []
            for idx, entry in enumerate(value):
                if isinstance(entry, str):
                    cleaned.append({"candidate_id": entry, "evidence_ids": [], "confidence": 0.2})
                    steps.append({"step": "candidate_string_to_object", "field": field, "index": idx})
                elif isinstance(entry, dict):
                    item = dict(entry)
                    if "candidate_id" not in item:
                        for alias in ("id", "rule_id", "element_id"):
                            if isinstance(item.get(alias), str):
                                item["candidate_id"] = item.get(alias)
                                steps.append({"step": "candidate_alias_promoted", "field": field, "index": idx, "alias": alias})
                                break
                    if "evidence_ids" not in item:
                        aliases = item.get("evidence") or item.get("evidence_id")
                        if isinstance(aliases, list):
                            item["evidence_ids"] = [str(x) for x in aliases]
                        elif isinstance(aliases, str) and aliases.strip():
                            item["evidence_ids"] = [aliases.strip()]
                        else:
                            item["evidence_ids"] = []
                    cleaned.append(item)
            working[field] = cleaned
        working.setdefault("unresolved_gaps", [])
        working.setdefault("contradictions", [])
        working.setdefault("summary", "")
        working.setdefault("confidence", 0.0)
        disposition = "accepted_as_is" if not steps else "accepted_after_repair"
        return working, steps, disposition

    def generate_dik_integration(self, request_packet: AgentDIKIntegrationRequest) -> AgentDIKIntegrationResponse:
        endpoint = f"{self.config.local_base_url.rstrip('/')}{self.config.local_endpoint}"
        payload = self._build_dik_integration_payload(request_packet)
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        parsed_result = self._parse_response(parsed)
        normalized, _steps, _disposition = self._normalize_dik_integration_payload(parsed_result.get("payload"))
        if normalized is None:
            return self.fallback.generate_dik_integration(request_packet)
        normalized.setdefault("response_id", f"dik-{request_packet.request_id}")
        normalized.setdefault("agent_id", request_packet.agent_id)
        response_obj = AgentDIKIntegrationResponse.from_dict(normalized)
        if validate_agent_dik_integration_response(response_obj):
            return self.fallback.generate_dik_integration(request_packet)
        return response_obj

    def generate_plan(self, request_packet: AgentBrainRequest) -> AgentBrainResponse:
        started_at = time.perf_counter()
        started_at_epoch_ms = int(time.time() * 1000)
        endpoint = f"{self.config.local_base_url.rstrip('/')}{self.config.local_endpoint}"
        fallback_reason = None
        fallback_hint = None
        trace: Dict[str, Any] = {
            "request_id": request_packet.request_id,
            "provider_class": self.__class__.__name__,
            "backend": self.config.backend,
            "model": self.config.local_model,
            "endpoint": endpoint,
            "timeout_s": self.config.timeout_s,
            "completion_max_tokens": int(self.config.completion_max_tokens),
            "unrestricted_local_qwen_mode": bool(getattr(self.config, "unrestricted_local_qwen_mode", False)),
            "request_started_epoch_ms": started_at_epoch_ms,
            "agent_brain_request_size_chars": len(json.dumps(request_packet.to_dict(), default=str)),
            "provider_request_payload": None,
            "provider_request_payload_size_chars": 0,
            "provider_response_parse_source": None,
            "reasoning_suppression_requested": False,
            "attempts": [],
            "llm_response_received": False,
            "llm_response_parsed": False,
            "llm_response_validated": False,
            "timeout_occurred": False,
            "fallback_used": False,
            "fallback_source": None,
            "result_source": "ollama",
            "trace_outcome_category": "no_result_generated",
            "runtime_disposition": "no_result_generated",
            "repair_retry_requested": False,
            "repair_retry_reason": None,
            "repair_retry_attempted": False,
            "repair_retry_count": 0,
            "minimal_action_salvage_attempted": False,
            "minimal_action_salvage_used": False,
            "minimal_action_salvage_reason": None,
        }
        attempts = max(1, int(self.config.max_retries) + 1)
        for attempt in range(1, attempts + 1):
            try:
                attempted_kinds = ["primary"]
                for attempt_kind in attempted_kinds:
                    payload = (
                        self._build_request_payload(request_packet)
                        if attempt_kind == "primary"
                        else self._build_schema_repair_payload(
                            invalid_payload=response_payload,
                            failure_reason=fallback_reason or "provider payload schema invalid",
                        )
                    )
                    if attempt_kind == "primary":
                        trace["provider_request_payload"] = payload
                        trace["provider_request_payload_size_chars"] = len(json.dumps(payload, default=str))
                    trace["reasoning_suppression_requested"] = False
                    attempt_trace: Dict[str, Any] = {"attempt": attempt, "attempt_kind": attempt_kind}
                    req = request.Request(
                        endpoint,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with request.urlopen(req, timeout=self.config.timeout_s) as response:
                        raw = response.read().decode("utf-8")
                    attempt_trace["http_body_non_empty"] = bool(raw.strip())
                    attempt_trace["raw_http_response_text"] = raw
                    trace["llm_response_received"] = True
                    parsed = json.loads(raw)
                    attempt_trace["parsed_response_json"] = parsed
                    trace["llm_response_parsed"] = True
                    parse_result = self._parse_response(parsed)
                    response_payload = parse_result["payload"]
                    attempt_trace["response_parse_source"] = parse_result.get("parse_source")
                    trace["provider_response_parse_source"] = parse_result.get("parse_source")
                    attempt_trace["extracted_response_payload"] = response_payload
                    normalized_payload, normalization_steps, runtime_disposition = self._normalize_payload(response_payload)
                    attempt_trace["normalization_steps"] = normalization_steps
                    attempt_trace["normalized_response_payload"] = normalized_payload
                    trace["runtime_disposition"] = runtime_disposition
                    trace["attempts"].append(attempt_trace)

                    if normalized_payload is None:
                        failure_reason = "provider payload schema invalid"
                        if runtime_disposition == "rejected_missing_plan":
                            failure_reason = "provider payload missing plan"
                        elif runtime_disposition == "rejected_unusable_next_action":
                            failure_reason = "provider payload has unusable next_action"
                        elif runtime_disposition == "rejected_illegal_action":
                            failure_reason = "provider payload has illegal action"
                        fallback_reason = f"attempt={attempt}/{attempts} error={failure_reason}"
                        trace["trace_outcome_category"] = "llm_parsed_invalid_payload"
                        trace["repair_retry_reason"] = failure_reason
                        can_repair = (
                            attempt_kind == "primary"
                            and attempt < attempts
                            and trace.get("llm_response_received")
                            and trace.get("llm_response_parsed")
                            and response_payload is not None
                        )
                        if can_repair:
                            trace["repair_retry_requested"] = True
                            trace["repair_retry_attempted"] = True
                            trace["repair_retry_count"] = int(trace.get("repair_retry_count", 0)) + 1
                            attempted_kinds.append("schema_repair")
                            continue
                        trace["minimal_action_salvage_attempted"] = True
                        trace["minimal_action_salvage_reason"] = failure_reason
                        salvage_payload, salvage_steps, salvage_disposition = self._build_minimal_salvage_response(
                            request_packet,
                            source_payload=response_payload,
                            failure_reason=failure_reason,
                        )
                        if salvage_payload is not None and salvage_disposition:
                            attempt_trace["minimal_salvage_steps"] = salvage_steps
                            attempt_trace["normalized_response_payload"] = salvage_payload
                            trace["minimal_action_salvage_used"] = True
                            trace["runtime_disposition"] = salvage_disposition
                            response_obj = AgentBrainResponse.from_dict(salvage_payload)
                            latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                            trace.update(
                                {
                                    "fallback": False,
                                    "fallback_used": False,
                                    "fallback_reason": None,
                                    "fallback_hint": None,
                                    "latency_ms": latency_ms,
                                    "latency_until_outcome_ms": latency_ms,
                                    "response_payload_valid": True,
                                    "llm_response_validated": True,
                                    "result_source": "ollama",
                                    "trace_outcome_category": "llm_success_via_minimal_action_salvage",
                                    "exception": None,
                                    "normalization_steps": list(normalization_steps) + list(salvage_steps),
                                    "normalized_response_payload": salvage_payload,
                                    "extracted_response_payload": response_payload,
                                    "runtime_disposition": salvage_disposition,
                                }
                            )
                            self.last_outcome = {
                                "fallback": False,
                                "reason": None,
                                "latency_ms": latency_ms,
                                "outcome_category": "llm_success_via_minimal_action_salvage",
                                "result_source": "ollama",
                            }
                            self.last_trace = trace
                            return response_obj
                        raise ValueError(failure_reason)

                    response_obj = AgentBrainResponse.from_dict(normalized_payload)
                    latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                    trace_outcome = "llm_success"
                    resolved_disposition = runtime_disposition
                    if attempt_kind == "schema_repair":
                        trace_outcome = "llm_success_after_schema_retry"
                        resolved_disposition = "accepted_after_schema_retry"
                    elif runtime_disposition == "accepted_after_repair":
                        trace_outcome = "llm_success_after_normalization_repair"
                    self.last_outcome = {"fallback": False, "reason": None, "latency_ms": latency_ms}
                    trace.update(
                        {
                            "fallback": False,
                            "fallback_used": False,
                            "fallback_reason": None,
                            "fallback_hint": None,
                            "latency_ms": latency_ms,
                            "latency_until_outcome_ms": latency_ms,
                            "response_payload_valid": True,
                            "llm_response_validated": True,
                            "result_source": "ollama",
                            "trace_outcome_category": trace_outcome,
                            "exception": None,
                            "normalization_steps": list(normalization_steps),
                            "normalized_response_payload": normalized_payload,
                            "extracted_response_payload": response_payload,
                            "runtime_disposition": resolved_disposition,
                        }
                    )
                    self.last_outcome.update({"outcome_category": trace_outcome, "result_source": "ollama"})
                    self.last_trace = trace
                    return response_obj
            except (TimeoutError, error.URLError, error.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                attempt_trace = {"attempt": attempt, "exception": {"type": type(exc).__name__, "message": str(exc), "repr": repr(exc)}}
                if "raw" in locals() and isinstance(raw, str):
                    attempt_trace["raw_http_response_text"] = raw
                if "parsed" in locals() and isinstance(parsed, dict):
                    attempt_trace["parsed_response_json"] = parsed
                if "response_payload" in locals() and isinstance(response_payload, dict):
                    attempt_trace["extracted_response_payload"] = response_payload
                trace["attempts"].append(attempt_trace)
                if isinstance(exc, TimeoutError) or "timed out" in str(exc).lower():
                    trace["timeout_occurred"] = True
                    trace["runtime_disposition"] = "timed_out"
                    trace["trace_outcome_category"] = "llm_timeout"
                elif isinstance(exc, (error.URLError, error.HTTPError)):
                    trace["runtime_disposition"] = trace.get("runtime_disposition") or "transport_error"
                    trace["trace_outcome_category"] = "llm_transport_error"
                elif trace.get("trace_outcome_category") != "llm_parsed_invalid_payload":
                    trace["runtime_disposition"] = trace.get("runtime_disposition") or "parsed_invalid_payload"
                    trace["trace_outcome_category"] = "llm_parsed_invalid_payload"
                if isinstance(exc, error.HTTPError) and getattr(exc, "code", None) == 404:
                    fallback_hint = "HTTP 404 from local backend may indicate missing/incorrect model name"
                fallback_reason = f"attempt={attempt}/{attempts} error={exc}"
                if attempt >= attempts:
                    break

        latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        self.last_outcome = {
            "fallback": True,
            "reason": fallback_reason,
            "latency_ms": latency_ms,
            "outcome_category": "llm_timeout_with_fallback" if trace.get("timeout_occurred") else "llm_error_with_fallback",
            "result_source": "fallback_safe_policy",
            "hint": fallback_hint,
            "configured_backend": self.config.backend,
            "configured_model": self.config.local_model,
            "configured_base_url": self.config.local_base_url,
            "configured_endpoint": self.config.local_endpoint,
            "timeout_s": self.config.timeout_s,
            "completion_max_tokens": int(self.config.completion_max_tokens),
            "unrestricted_local_qwen_mode": bool(getattr(self.config, "unrestricted_local_qwen_mode", False)),
        }
        trace.update(
            {
                "fallback": True,
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "fallback_hint": fallback_hint,
                "fallback_source": "fallback_safe_policy",
                "latency_ms": latency_ms,
                "latency_until_outcome_ms": latency_ms,
                "response_payload_valid": False,
                "result_source": "fallback_safe_policy",
                "trace_outcome_category": (
                    "llm_timeout_with_fallback"
                    if trace.get("timeout_occurred")
                    else ("llm_invalid_after_schema_retry_with_fallback" if trace.get("repair_retry_attempted") else "llm_error_with_fallback")
                ),
                "exception": trace.get("attempts", [{}])[-1].get("exception") if trace.get("attempts") else None,
                "runtime_disposition": trace.get("runtime_disposition") or ("timed_out" if trace.get("timeout_occurred") else "fallback_generated"),
            }
        )
        self.last_trace = trace
        LOGGER.warning("OllamaLocalBrainProvider fallback to RuleBrain: %s", self.last_outcome)
        step = select_productive_fallback_action(request_packet.allowed_actions)
        payload = {
            "response_id": f"fallback-{request_packet.request_id}",
            "agent_id": request_packet.agent_id,
            "plan": {
                "plan_id": f"fallback-plan-{request_packet.request_id}",
                "plan_horizon": int(request_packet.planning_horizon_config.get("max_steps", 3)),
                "ordered_goals": [
                    {
                        "goal_id": "fallback_progress",
                        "description": "maintain safe deterministic forward progress",
                        "priority": 0.6,
                        "status": "active",
                        "source": "fallback",
                    }
                ],
                "ordered_actions": [step.__dict__],
                "next_action": step.__dict__,
                "confidence": 0.6,
                "notes": ["fallback_safe_policy"],
            },
            "explanation": "fallback safe policy selected legal action" if request_packet.request_explanation else None,
            "confidence": 0.6,
        }
        return AgentBrainResponse.from_dict(payload)


class LocalLLMBrainStub(BrainProvider):
    def generate_plan(self, request_packet: AgentBrainRequest) -> AgentBrainResponse:
        payload = {
            "response_id": f"stub-{request_packet.request_id}",
            "agent_id": request_packet.agent_id,
            "plan": {
                "plan_id": f"stub-plan-{request_packet.request_id}",
                "plan_horizon": 2,
                "ordered_goals": [{"goal_id": "g_stub", "description": "request context", "priority": 0.25, "status": "active"}],
                "ordered_actions": [{"step_index": 0, "action_type": "reassess_plan", "expected_purpose": "stub backend"}],
                "next_action": {"step_index": 0, "action_type": "reassess_plan", "expected_purpose": "stub backend"},
                "confidence": 0.25,
                "notes": ["backend is stubbed"],
            },
            "explanation": "stub response" if request_packet.request_explanation else None,
        }
        return AgentBrainResponse.from_dict(payload)


class CloudBrainStub(BrainProvider):
    def generate_plan(self, request_packet: AgentBrainRequest) -> AgentBrainResponse:
        payload = {
            "response_id": f"cloud-stub-{request_packet.request_id}",
            "agent_id": request_packet.agent_id,
            "plan": {
                "plan_id": f"cloud-plan-{request_packet.request_id}",
                "plan_horizon": 1,
                "ordered_goals": [{"goal_id": "observe", "description": "observe world", "priority": 0.2, "status": "active"}],
                "ordered_actions": [{"step_index": 0, "action_type": "observe_environment", "expected_purpose": "no API configured"}],
                "next_action": {"step_index": 0, "action_type": "observe_environment", "expected_purpose": "no API configured"},
                "confidence": 0.2,
            },
        }
        return AgentBrainResponse.from_dict(payload)

# Backward compatible alias
LocalHTTPBrain = OllamaLocalBrainProvider
