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
    debug: bool = False


def create_brain_provider(config: BrainBackendConfig | None = None) -> BrainProvider:
    config = config or BrainBackendConfig()
    selected = config.backend.lower()
    if selected == "local_stub":
        return LocalLLMBrainStub()
    if selected in {"local_http", "openai_compatible_local"}:
        return LocalHTTPBrain(config=config, fallback=RuleBrain())
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
    )


class RuleBrain(BrainProvider):
    """Deterministic baseline brain used as default backend."""

    def decide(self, context_packet):
        affordances = context_packet.action_affordances
        affordance_types = [item["action_type"] for item in affordances]

        if ExecutableActionType.INSPECT_INFORMATION_SOURCE.value in affordance_types:
            target = next(
                (a for a in affordances if a["action_type"] == ExecutableActionType.INSPECT_INFORMATION_SOURCE.value),
                None,
            )
            return BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id=target.get("target_id") if target else None,
                target_zone=target.get("target_zone") if target else None,
                goal_update="seek_info",
                plan_steps=["inspect team/role information", "share useful findings"],
                communication_intent=CommunicationIntent.TIP,
                reason_summary="Prioritize gathering information before construction.",
                confidence=0.8,
                assumptions=["simulator validates packet access"],
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
        return json.loads(content)

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
                },
            )
            self.last_outcome = {"fallback": False, "reason": None, "latency_ms": latency_ms}
            return decision
        except (TimeoutError, error.URLError, error.HTTPError, ValueError, KeyError, json.JSONDecodeError) as exc:
            fallback_reason = str(exc)

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
