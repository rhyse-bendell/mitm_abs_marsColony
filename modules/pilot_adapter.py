"""Pilot adapter seam for action-selection policies.

Pilots choose actions from AgentBrainRequest context. Agent Mecha and environment
layers remain responsible for action execution constraints and world-truth outcomes.
"""

from __future__ import annotations

from typing import Optional, Protocol, Any

from modules.action_schema import BrainDecision
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse


class PilotAdapter(Protocol):
    pilot_id: str
    display_name: str

    def choose_action(self, request: AgentBrainRequest) -> AgentBrainResponse:
        ...

    def choose_fallback_action(self, *, agent: Any, context_packet: Any, reason: str, sim_state: Any = None) -> Optional[BrainDecision]:
        ...

    def handle_blocked_action(
        self,
        *,
        agent: Any,
        original_decision: BrainDecision,
        gate_result: Any,
        sim_state: Any = None,
    ) -> Optional[BrainDecision]:
        ...
