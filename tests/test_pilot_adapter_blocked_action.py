from types import SimpleNamespace

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.action_gate import ActionGateResult
from modules.pilot_adapter import GenericBrainProviderPilotAdapter
from modules.procedural_baseline_pilot import ProceduralBaselinePilotAdapter


def test_procedural_baseline_accepts_construction_logistics_reroute_for_physical_blocker():
    reroutes = [
        BrainDecision(ExecutableActionType.TRANSPORT_RESOURCES, target_id="p1"),
        BrainDecision(ExecutableActionType.CONTINUE_CONSTRUCTION, target_id="p1"),
        BrainDecision(ExecutableActionType.START_CONSTRUCTION, target_id="p1"),
    ]
    result = ActionGateResult(False, BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="p1"), blockers=["physical_incomplete"], available_reroutes=reroutes)

    selected = ProceduralBaselinePilotAdapter().handle_blocked_action(agent=None, original_decision=result.normalized_decision, gate_result=result)

    assert selected.selected_action == ExecutableActionType.START_CONSTRUCTION


def test_generic_adapter_returns_none_for_blocked_validation():
    result = ActionGateResult(False, BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="p1"), blockers=["physical_incomplete"], available_reroutes=[BrainDecision(ExecutableActionType.START_CONSTRUCTION, target_id="p1")])
    adapter = GenericBrainProviderPilotAdapter(provider=SimpleNamespace(generate_plan=lambda request: None), pilot_id="llm")

    assert adapter.handle_blocked_action(agent=None, original_decision=result.normalized_decision, gate_result=result) is None
