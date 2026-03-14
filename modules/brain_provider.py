from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict
from urllib import error, request

from modules.action_schema import BrainDecision, CommunicationIntent, ExecutableActionType


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrainBackendConfig:
    backend: str = "rule_brain"
    local_base_url: str = "http://127.0.0.1:11434"
    local_endpoint: str = "/v1/chat/completions"
    local_model: str = "local-model"
    timeout_s: float = 1.5
    max_retries: int = 0
    fallback_backend: str = "rule_brain"
    debug: bool = False


def create_brain_provider(config: BrainBackendConfig | None = None) -> BrainProvider:
    config = config or BrainBackendConfig()
    selected = config.backend.lower()
    if selected == "local_stub":
        return LocalLLMBrainStub()
    if selected in {"local_http", "openai_compatible_local"}:
        fallback = RuleBrain() if config.fallback_backend.lower() == "rule_brain" else RuleBrain()
        return LocalHTTPBrain(config=config, fallback=fallback)
    if selected == "cloud_stub":
        return CloudBrainStub()
    return RuleBrain()


class BrainProvider(ABC):
    @abstractmethod
    def decide(self, context_packet):
        raise NotImplementedError


def _decision_from_payload(payload: Dict[str, Any]) -> BrainDecision:
    selected_action = payload.get("selected_action")
    if not selected_action:
        raise ValueError("missing selected_action in local model response")

    communication_intent = payload.get("communication_intent")
    return BrainDecision(
        selected_action=ExecutableActionType(selected_action),
        target_id=payload.get("target_id"),
        target_zone=payload.get("target_zone"),
        goal_update=payload.get("goal_update"),
        plan_steps=list(payload.get("plan_steps", [])),
        communication_intent=CommunicationIntent(communication_intent) if communication_intent else None,
        reason_summary=str(payload.get("reason_summary", "")),
        confidence=float(payload.get("confidence", 0.0)),
        assumptions=list(payload.get("assumptions", [])),
        requests_for_context=list(payload.get("requests_for_context", [])),
        plan_method_id=payload.get("plan_method_id"),
        next_steps=list(payload.get("next_steps", payload.get("plan_steps", []))),
    )


class RuleBrain(BrainProvider):
    """Deterministic baseline brain used as default backend."""

    def decide(self, context_packet):
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


class LocalHTTPBrain(BrainProvider):
    """Optional local model backend with safe fallback to RuleBrain."""

    def __init__(self, config: BrainBackendConfig, fallback: BrainProvider):
        self.config = config
        self.fallback = fallback
        self.last_outcome = {"fallback": False, "reason": None, "latency_ms": None}

    def _log_debug(self, message: str, payload: Dict[str, Any]) -> None:
        if self.config.debug:
            LOGGER.info("%s %s", message, payload)

    def _build_request_payload(self, context_packet) -> Dict[str, Any]:
        context_json = json.dumps(context_packet.__dict__, default=str)
        return {
            "model": self.config.local_model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only JSON object matching BrainDecision fields."
                },
                {
                    "role": "user",
                    "content": context_json,
                },
            ],
            "temperature": 0,
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

    def decide(self, context_packet):
        started_at = time.perf_counter()
        context_meta = {
            "sim_time": context_packet.world_snapshot.get("sim_time"),
            "affordance_count": len(context_packet.action_affordances),
            "build_plausibility": context_packet.individual_cognitive_state.get("build_readiness", {}).get("status"),
        }
        self._log_debug("brain_local_query", {"provider": "local_http", "context_meta": context_meta})

        endpoint = f"{self.config.local_base_url.rstrip('/')}{self.config.local_endpoint}"
        fallback_reason = None
        attempts = max(1, int(self.config.max_retries) + 1)
        for attempt in range(1, attempts + 1):
            try:
                payload = self._build_request_payload(context_packet)
                req = request.Request(
                    endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=self.config.timeout_s) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                decision_payload = self._parse_response(parsed)
                decision = _decision_from_payload(decision_payload)
                latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
                self._log_debug(
                    "brain_local_outcome",
                    {
                        "provider": "local_http",
                        "fallback": False,
                        "selected_action": decision.selected_action.value,
                        "confidence": decision.confidence,
                        "latency_ms": latency_ms,
                        "attempt": attempt,
                    },
                )
                self.last_outcome = {"fallback": False, "reason": None, "latency_ms": latency_ms}
                return decision
            except (TimeoutError, error.URLError, error.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
                fallback_reason = f"attempt={attempt}/{attempts} error={exc}"
                if attempt >= attempts:
                    break

        latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        LOGGER.warning(
            "LocalHTTPBrain fallback to RuleBrain: %s",
            fallback_reason,
        )
        self._log_debug(
            "brain_local_outcome",
            {
                "provider": "local_http",
                "fallback": True,
                "fallback_reason": fallback_reason,
                "latency_ms": latency_ms,
            },
        )
        self.last_outcome = {"fallback": True, "reason": fallback_reason, "latency_ms": latency_ms}
        return self.fallback.decide(context_packet)


class LocalLLMBrainStub(BrainProvider):
    def decide(self, context_packet):
        return BrainDecision(
            selected_action=ExecutableActionType.REASSESS_PLAN,
            reason_summary="LocalLLMBrain stub: no model call configured yet.",
            confidence=0.25,
            assumptions=["backend is stubbed"],
            requests_for_context=["role-specific constraints"],
        )


class CloudBrainStub(BrainProvider):
    def decide(self, context_packet):
        return BrainDecision(
            selected_action=ExecutableActionType.OBSERVE_ENVIRONMENT,
            reason_summary="CloudBrain stub: no external API call configured yet.",
            confidence=0.2,
            assumptions=["backend is stubbed", "simulator truth remains authoritative"],
            requests_for_context=["recent validation failures"],
        )
