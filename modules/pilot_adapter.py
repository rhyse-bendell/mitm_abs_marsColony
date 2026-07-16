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



class GenericBrainProviderPilotAdapter:
    """Adapter for non-procedural pilots backed by the current BrainProvider."""

    pilot_id = "generic"
    display_name = "Generic Brain Provider Pilot"

    def __init__(self, provider: Any, *, pilot_id: str = "generic", display_name: str = "Generic Brain Provider Pilot"):
        self.provider = provider
        self.pilot_id = pilot_id
        self.display_name = display_name

    def choose_action(self, request: AgentBrainRequest) -> AgentBrainResponse:
        return self.provider.generate_plan(request)

    def choose_fallback_action(self, *, agent: Any, context_packet: Any, reason: str, sim_state: Any = None) -> Optional[BrainDecision]:
        # Non-baseline pilots should observe failures and replan rather than
        # silently receiving Procedural Baseline strategy.
        return None

    def handle_blocked_action(
        self,
        *,
        agent: Any,
        original_decision: BrainDecision,
        gate_result: Any,
        sim_state: Any = None,
    ) -> Optional[BrainDecision]:
        return None
