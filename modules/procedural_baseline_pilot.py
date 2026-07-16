"""Adapter for the Procedural Baseline Pilot (legacy RuleBrain backend)."""

from __future__ import annotations

from typing import Optional, Any

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse
from modules.brain_provider import ProceduralBaselinePilot, select_productive_fallback_action


class ProceduralBaselinePilotAdapter:
    pilot_id = "procedural_baseline"
    display_name = "Procedural Baseline Pilot"

    def __init__(self, provider: Optional[ProceduralBaselinePilot] = None):
        self.provider = provider or ProceduralBaselinePilot()

    def choose_action(self, request: AgentBrainRequest) -> AgentBrainResponse:
        return self.provider.generate_plan(request)

    def choose_fallback_action(self, *, agent: Any, context_packet: Any, reason: str, sim_state: Any = None) -> Optional[BrainDecision]:
        # Procedural-baseline-specific fallback delegation; not generic mecha behavior.
        allowed_actions = list(getattr(context_packet, "action_affordances", []) or [])
        step = select_productive_fallback_action(allowed_actions)
        return step.to_brain_decision(confidence=0.25, plan_method_id="procedural_baseline_fallback", next_steps=[reason])

    def handle_blocked_action(
        self,
        *,
        agent: Any,
        original_decision: BrainDecision,
        gate_result: Any,
        sim_state: Any = None,
    ) -> Optional[BrainDecision]:
        blockers = set(getattr(gate_result, "blockers", []) or [])
        if blockers & {"physical_build_not_started", "physical_incomplete"}:
            priority = [
                ExecutableActionType.START_CONSTRUCTION,
                ExecutableActionType.CONTINUE_CONSTRUCTION,
                ExecutableActionType.TRANSPORT_RESOURCES,
            ]
            reroutes = list(getattr(gate_result, "available_reroutes", []) or [])
            for action_type in priority:
                for candidate in reroutes:
                    if getattr(candidate, "selected_action", None) == action_type:
                        return candidate
        return None
