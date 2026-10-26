"""Adapter for the Procedural Baseline Pilot (legacy RuleBrain backend)."""

from __future__ import annotations

from typing import Optional, Any

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse
from modules.controller_context import partition_affordances
from modules.brain_provider import ProceduralBaselinePilot, resolve_design_variant, select_productive_fallback_action


class ProceduralBaselinePilotAdapter:
    pilot_id = "procedural_baseline"
    display_name = "Procedural Baseline Pilot"

    def __init__(self, provider: Optional[ProceduralBaselinePilot] = None):
        self.provider = provider or ProceduralBaselinePilot()

    def choose_action(self, request: AgentBrainRequest) -> AgentBrainResponse:
        return self.provider.generate_plan(request)

    def choose_fallback_action(self, *, agent: Any, context_packet: Any, reason: str, sim_state: Any = None) -> Optional[BrainDecision]:
        # Procedural-baseline-specific fallback delegation; not generic mecha behavior.
        # the fallback, like the main policy, may only choose currently-executable affordances (blocked ones are information, never selectable)
        allowed_actions, _blocked = partition_affordances(list(getattr(context_packet, "action_affordances", []) or []))
        step = select_productive_fallback_action(allowed_actions)
        if step.action_args.get("design_slot"):
            # the fallback applies the same documented specification-choice rule as the main policy (it never defaults to the first listed option)
            first = next((a for a in allowed_actions if a.get("action_type") == step.action_type.value and a.get("target_id") == step.target_id and a.get("design_slot") == step.action_args.get("design_slot")), None)
            held, task_id, seed_key = ProceduralBaselinePilot._design_policy_inputs(context_packet, False)
            seed_key = f"fallback:{getattr(agent, 'run_seed', None)}:{getattr(agent, 'agent_id', '')}"
            if first is not None:
                picked = resolve_design_variant(first, allowed_actions, held, task_id, seed_key)
                step = type(step)(step_index=step.step_index, action_type=step.action_type, target_id=step.target_id, target_zone=step.target_zone, expected_purpose=step.expected_purpose, action_args=dict(picked.get("action_args") or {}))
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
