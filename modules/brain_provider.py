from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modules.action_schema import BrainDecision, CommunicationIntent, ExecutableActionType


@dataclass(frozen=True)
class BrainBackendConfig:
    backend: str = "rule_brain"


def create_brain_provider(config: BrainBackendConfig | None = None) -> BrainProvider:
    selected = (config.backend if config else "rule_brain").lower()
    if selected == "local_stub":
        return LocalLLMBrainStub()
    if selected == "cloud_stub":
        return CloudBrainStub()
    return RuleBrain()


class BrainProvider(ABC):
    @abstractmethod
    def decide(self, context_packet):
        raise NotImplementedError


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
