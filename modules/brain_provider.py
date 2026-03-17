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
        ):
            return BrainDecision(
                selected_action=ExecutableActionType.REQUEST_ASSISTANCE,
                goal_update="request_or_offer_help",
                communication_intent=CommunicationIntent.TKRQ,
                plan_steps=["ask for clarification", "integrate support"],
                reason_summary="Help tendency and known gaps prompt assistance-seeking.",
                confidence=0.76,
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
        contract = request_packet.to_dict()
        return {
            "model": self.config.local_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON object matching AgentBrainResponse with plan/next_action."
                },
                {
                    "role": "user",
                    "content": json.dumps(contract, default=str),
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
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            raise ValueError("local backend response content is not JSON string")
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        return json.loads(cleaned)

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
                response_payload = self._parse_response(parsed)
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
