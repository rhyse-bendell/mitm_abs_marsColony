from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from urllib import error, request

from modules.action_schema import BrainDecision, CommunicationIntent, ExecutableActionType
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse, PlannedActionStep


LOGGER = logging.getLogger(__name__)


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
    local_model: str = "qwen3.5:9b"
    timeout_s: float = 75.0
    warmup_timeout_s: float = 45.0
    completion_max_tokens: int = 2048
    startup_completion_max_tokens: int = 1024
    permissive_timeout_ceiling_s: float = 1200.0
    permissive_completion_ceiling_tokens: int = 16384
    unrestricted_local_qwen_mode: bool = False
    max_retries: int = 0
    fallback_backend: str = "rule_brain"
    debug: bool = False
    planner_trace_enabled: bool = True
    planner_trace_mode: str = "full"
    planner_trace_max_chars: int = 12000


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
        task_context=dict(context_packet.static_task_context),
        rule_context=list(cognitive.get("knowledge_summary", [])[:8]),
        derivation_context=[],
        artifact_context=list(context_packet.team_state.get("externalized_artifacts", []))[:4],
    )


class RuleBrain(BrainProvider):
    """Deterministic baseline brain used as default backend."""

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

    def _decision_logic(self, context_packet) -> BrainDecision:
        affordances = context_packet.action_affordances
        affordance_types = [item["action_type"] for item in affordances]

        traits = context_packet.individual_cognitive_state.get("traits", {})
        communication_propensity = float(traits.get("communication_propensity", 0.5))
        goal_alignment = float(traits.get("goal_alignment", 0.5))
        help_tendency = float(traits.get("help_tendency", 0.5))

        phase_profile = context_packet.world_snapshot.get("phase_profile", {})
        stage = phase_profile.get("stage", "execution")
        readiness = context_packet.individual_cognitive_state.get("build_readiness", {})
        ready_for_build = bool(readiness.get("ready_for_build"))

        has_validated_artifact = any(
            a.get("validation_state") == "validated" for a in context_packet.team_state.get("externalized_artifacts", [])
        )
        has_known_gaps = bool(context_packet.individual_cognitive_state.get("known_gaps"))
        mismatch_signals = context_packet.history_bands.get("semantic_plan_evolution", {}).get("unresolved_contradictions", [])

        sorted_affordances = sorted(affordances, key=lambda a: float(a.get("utility", 0.0)), reverse=True)
        top_affordance = sorted_affordances[0] if sorted_affordances else None
        built_state = context_packet.world_snapshot.get("built_state", [])
        active_incomplete_projects = [
            item
            for item in built_state
            if item.get("state") in {"absent", "in_progress"} and float(item.get("progress", 0.0)) < 1.0
        ]
        needs_resource_delivery = any(float(item.get("progress", 0.0)) < 1.0 for item in active_incomplete_projects)
        in_progress_resource_incomplete = any(
            item.get("state") == "in_progress" and float(item.get("progress", 0.0)) < 1.0
            for item in active_incomplete_projects
        )
        productive_build_types = {
            ExecutableActionType.TRANSPORT_RESOURCES.value,
            ExecutableActionType.START_CONSTRUCTION.value,
            ExecutableActionType.CONTINUE_CONSTRUCTION.value,
        }
        loop_counters = context_packet.individual_cognitive_state.get("loop_counters", {})
        repeated_action_count = int(loop_counters.get("action_repeats", 0) or 0)
        repeated_selected_action_count = int(loop_counters.get("selected_action_repeats", 0) or 0)
        seconds_since_dik_change = context_packet.individual_cognitive_state.get("seconds_since_dik_change")
        recent_meaningful_epistemic_change = (
            seconds_since_dik_change is not None
            and float(seconds_since_dik_change) <= 2.0
            and (bool(mismatch_signals) or (has_known_gaps and not ready_for_build))
        )
        productive_build_affordance = None
        if ready_for_build and active_incomplete_projects:
            if needs_resource_delivery and (in_progress_resource_incomplete or not recent_meaningful_epistemic_change):
                productive_build_affordance = self._best_affordance(
                    sorted_affordances, {ExecutableActionType.TRANSPORT_RESOURCES.value}
                )
            if productive_build_affordance is None:
                productive_build_affordance = self._best_affordance(sorted_affordances, productive_build_types)
        post_readiness_pivot_active = (
            ready_for_build
            and bool(active_incomplete_projects)
            and productive_build_affordance is not None
            and not recent_meaningful_epistemic_change
        )

        assistance_stalled = (
            max(repeated_action_count, repeated_selected_action_count) >= 3
            and (seconds_since_dik_change is None or float(seconds_since_dik_change) > 8.0)
        )

        if post_readiness_pivot_active:
            selected = ExecutableActionType(productive_build_affordance["action_type"])
            goal_update = "execute_build"
            reason = "Readiness unlocked with incomplete projects; pivot from epistemic actions to productive build progression."
            if selected == ExecutableActionType.TRANSPORT_RESOURCES:
                goal_update = "satisfy_build_logistics"
                reason = "Readiness unlocked with incomplete projects; prioritize logistics delivery before construction."
            return BrainDecision(
                selected_action=selected,
                target_id=productive_build_affordance.get("target_id"),
                target_zone=productive_build_affordance.get("target_zone"),
                goal_update=goal_update,
                plan_steps=["advance active construction project"],
                reason_summary=reason,
                confidence=0.84,
                assumptions=["duration_s=30"] if selected == ExecutableActionType.TRANSPORT_RESOURCES else [],
            )

        if (
            stage in {"early", "execution"}
            and ExecutableActionType.EXTERNALIZE_PLAN.value in affordance_types
            and communication_propensity >= 0.7
            and context_packet.individual_cognitive_state.get("knowledge_summary")
        ):
            return BrainDecision(
                selected_action=ExecutableActionType.EXTERNALIZE_PLAN,
                goal_update="share_plan",
                communication_intent=CommunicationIntent.TPP,
                plan_steps=["externalize rule summary", "invite uptake"],
                reason_summary="Communication propensity favors whiteboard/team externalization.",
                confidence=0.78,
            )

        if (
            ExecutableActionType.CONSULT_TEAM_ARTIFACT.value in affordance_types
            and goal_alignment >= 0.65
            and has_validated_artifact
        ):
            return BrainDecision(
                selected_action=ExecutableActionType.CONSULT_TEAM_ARTIFACT,
                goal_update="align_with_team_plan",
                plan_steps=["consult validated artifact", "align local plan"],
                reason_summary="Goal alignment favors consulting validated team artifacts.",
                confidence=0.82,
                assumptions=[f"phase_stage={stage}"],
            )

        if (
            ExecutableActionType.REQUEST_ASSISTANCE.value in affordance_types
            and help_tendency >= 0.7
            and has_known_gaps
            and (
                not ready_for_build
                or not active_incomplete_projects
                or productive_build_affordance is None
            )
            and not assistance_stalled
        ):
            return BrainDecision(
                selected_action=ExecutableActionType.REQUEST_ASSISTANCE,
                goal_update="request_or_offer_help",
                communication_intent=CommunicationIntent.TKRQ,
                plan_steps=["ask for clarification", "integrate support"],
                reason_summary="Help tendency and known gaps prompt assistance-seeking.",
                confidence=0.76,
            )

        if assistance_stalled:
            anti_loop_affordance = self._best_affordance(
                sorted_affordances,
                {
                    ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
                    ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
                    ExecutableActionType.TRANSPORT_RESOURCES.value,
                    ExecutableActionType.START_CONSTRUCTION.value,
                    ExecutableActionType.CONTINUE_CONSTRUCTION.value,
                },
            )
            if anti_loop_affordance is not None:
                selected = ExecutableActionType(anti_loop_affordance["action_type"])
                return BrainDecision(
                    selected_action=selected,
                    target_id=anti_loop_affordance.get("target_id"),
                    target_zone=anti_loop_affordance.get("target_zone"),
                    goal_update="break_assistance_loop",
                    plan_steps=["de-prioritize repeated assistance", "execute next productive affordance"],
                    reason_summary="Repeated assistance without recent DIK/team-state change; force productive anti-loop action.",
                    confidence=0.78,
                )

        if productive_build_affordance is not None:
            selected = ExecutableActionType(productive_build_affordance["action_type"])
            goal_update = "execute_build"
            reason = "Build readiness unlocked; prioritize highest-utility productive construction progression affordance."
            if selected == ExecutableActionType.TRANSPORT_RESOURCES:
                goal_update = "satisfy_build_logistics"
                reason = "Build readiness unlocked with incomplete projects; prioritize resource transport progression."
            return BrainDecision(
                selected_action=selected,
                target_id=productive_build_affordance.get("target_id"),
                target_zone=productive_build_affordance.get("target_zone"),
                goal_update=goal_update,
                plan_steps=["advance active construction project"],
                reason_summary=reason,
                confidence=0.82,
                assumptions=["duration_s=30"] if selected == ExecutableActionType.TRANSPORT_RESOURCES else [],
            )

        if stage == "late" and mismatch_signals and ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value in affordance_types:
            return BrainDecision(
                selected_action=ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
                goal_update="repair_detected_mismatch",
                plan_steps=["locate contradiction", "apply corrective construction"],
                reason_summary="Late-phase contradictions increase correction priority.",
                confidence=0.83,
            )

        if stage == "early" and ExecutableActionType.INSPECT_INFORMATION_SOURCE.value in affordance_types:
            target = next(
                (a for a in sorted_affordances if a["action_type"] == ExecutableActionType.INSPECT_INFORMATION_SOURCE.value),
                None,
            )
            return BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id=target.get("target_id") if target else None,
                target_zone=target.get("target_zone") if target else None,
                goal_update="seek_info",
                plan_steps=["inspect team/role information", "share useful findings"],
                communication_intent=CommunicationIntent.TIP,
                reason_summary="Early phase prioritizes information gathering.",
                confidence=0.82,
            )

        if ready_for_build and stage in {"execution", "late"}:
            transport = next((a for a in sorted_affordances if a["action_type"] == ExecutableActionType.TRANSPORT_RESOURCES.value), None)
            if transport and float(transport.get("utility", 0.0)) >= 0.6:
                return BrainDecision(
                    selected_action=ExecutableActionType.TRANSPORT_RESOURCES,
                    goal_update="satisfy_build_logistics",
                    plan_steps=["move resources to active work zone"],
                    reason_summary="Execution phase elevates logistics before construction.",
                    confidence=0.79,
                    assumptions=["duration_s=30"],
                )

            start = next((a for a in sorted_affordances if a["action_type"] == ExecutableActionType.START_CONSTRUCTION.value), None)
            if start:
                return BrainDecision(
                    selected_action=ExecutableActionType.START_CONSTRUCTION,
                    target_id=start.get("target_id"),
                    target_zone=start.get("target_zone"),
                    goal_update="execute_build",
                    plan_steps=["start construction at viable work zone"],
                    reason_summary="Build readiness and phase progression support execution.",
                    confidence=0.8,
                )

        if ExecutableActionType.INSPECT_INFORMATION_SOURCE.value in affordance_types and not ready_for_build:
            target = next(
                (a for a in sorted_affordances if a["action_type"] == ExecutableActionType.INSPECT_INFORMATION_SOURCE.value),
                None,
            )
            return BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id=target.get("target_id") if target else None,
                target_zone=target.get("target_zone") if target else None,
                goal_update="seek_info",
                plan_steps=["inspect needed source", "update build readiness"],
                communication_intent=CommunicationIntent.TIP,
                reason_summary="Continue information gathering until readiness is plausible.",
                confidence=0.77,
            )

        if top_affordance and top_affordance.get("action_type") in affordance_types:
            selected = ExecutableActionType(top_affordance["action_type"])
            assumptions = []
            if selected == ExecutableActionType.TRANSPORT_RESOURCES:
                assumptions.append("duration_s=30")
            return BrainDecision(
                selected_action=selected,
                target_id=top_affordance.get("target_id"),
                target_zone=top_affordance.get("target_zone"),
                reason_summary="Selected highest-utility legal affordance.",
                confidence=0.7,
                assumptions=assumptions,
            )

        return BrainDecision(
            selected_action=ExecutableActionType.WAIT,
            reason_summary="No specific high-value affordance available; wait and reassess.",
            confidence=0.6,
            assumptions=["holding position is legal"],
        )

    def decide(self, context_packet):
        return self._decision_logic(context_packet)

    def generate_plan(self, request_packet: AgentBrainRequest) -> AgentBrainResponse:
        legal = request_packet.allowed_actions
        step = PlannedActionStep(step_index=0, action_type=ExecutableActionType.WAIT, expected_purpose="hold position")
        if legal:
            first = legal[0]
            step = PlannedActionStep(
                step_index=0,
                action_type=ExecutableActionType(first.get("action_type", ExecutableActionType.WAIT.value)),
                target_id=first.get("target_id"),
                target_zone=first.get("target_zone"),
                expected_purpose="deterministic first legal action",
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
                "confidence": 0.7,
            },
            "explanation": "rule fallback selected legal first action" if request_packet.request_explanation else None,
            "confidence": 0.7,
        }
        return AgentBrainResponse.from_dict(payload)


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
                        "Output JSON only. No markdown. No analysis or chain-of-thought in visible output. "
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
        }
        attempts = max(1, int(self.config.max_retries) + 1)
        for attempt in range(1, attempts + 1):
            try:
                payload = self._build_request_payload(request_packet)
                trace["provider_request_payload"] = payload
                trace["provider_request_payload_size_chars"] = len(json.dumps(payload, default=str))
                trace["reasoning_suppression_requested"] = False
                attempt_trace: Dict[str, Any] = {"attempt": attempt}
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
                if not isinstance(response_payload, dict) or "plan" not in response_payload:
                    raise ValueError("provider payload missing plan")
                plan_payload = response_payload.get("plan") or {}
                if "next_action" not in plan_payload or "ordered_actions" not in plan_payload:
                    raise ValueError("provider payload missing required plan fields")
                response_obj = AgentBrainResponse.from_dict(response_payload)
                trace["attempts"].append(attempt_trace)
                latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
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
                        "trace_outcome_category": "llm_success",
                        "exception": None,
                    }
                )
                self.last_outcome.update({"outcome_category": "llm_success", "result_source": "ollama"})
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
                "trace_outcome_category": "llm_timeout_with_fallback" if trace.get("timeout_occurred") else "llm_error_with_fallback",
                "exception": trace.get("attempts", [{}])[-1].get("exception") if trace.get("attempts") else None,
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
