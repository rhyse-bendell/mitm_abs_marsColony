"""Procedural Baseline Pilot runs through the canonical Pilot contract.

Normal ``rule_brain`` runtime: agent -> per-agent PilotAdapter -> AgentBrainRequest /
AgentBrainResponse -> shared response validation/grounding -> ActionGate/translation.
The baseline stays synchronous; it must not call ``provider.decide(context)`` directly.
"""

import copy
import tempfile

import pytest

import modules.agent as agent_module
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse
from modules.procedural_baseline_pilot import ProceduralBaselinePilotAdapter
from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="baseline_contract")
        try:
            yield state
        finally:
            state.stop()


class _RecordingAdapter:
    """Delegates to the real adapter while recording the requests/responses it sees."""

    def __init__(self, inner):
        self.inner = inner
        self.provider = inner.provider
        self.pilot_id = inner.pilot_id
        self.display_name = inner.display_name
        self.requests = []
        self.responses = []

    def choose_action(self, request):
        self.requests.append(request)
        response = self.inner.choose_action(request)
        self.responses.append(response)
        return response

    def choose_fallback_action(self, **kwargs):
        return self.inner.choose_fallback_action(**kwargs)

    def handle_blocked_action(self, **kwargs):
        return self.inner.handle_blocked_action(**kwargs)


def _wrap_adapters(sim):
    wrapped = {}
    for agent in sim.agents:
        runtime = sim.get_agent_brain_runtime(agent)
        runtime["adapter"] = _RecordingAdapter(runtime["adapter"])
        wrapped[agent.name] = runtime["adapter"]
    return wrapped


def _forbid_decide(sim):
    def _boom(*_args, **_kwargs):
        raise AssertionError("provider.decide() must not be used for authoritative Pilot selection")

    for agent in sim.agents:
        sim.get_agent_brain_runtime(agent)["provider"].decide = _boom
    sim.brain_provider.decide = _boom


def test_rule_brain_runtime_invokes_each_agents_procedural_baseline_adapter(sim):
    adapters = _wrap_adapters(sim)
    for _ in range(40):
        sim.update(1.0)

    assert set(adapters) == {"Architect", "Engineer", "Botanist"}
    for name, adapter in adapters.items():
        assert isinstance(adapter.inner, ProceduralBaselinePilotAdapter), name
        assert adapter.requests, f"{name}'s adapter was never asked to choose an action"
        assert all(isinstance(r, AgentBrainRequest) for r in adapter.requests)
        assert all(isinstance(r, AgentBrainResponse) for r in adapter.responses)
        # requests identify their own agent (no cross-agent routing)
        agent = next(a for a in sim.agents if a.name == name)
        assert {r.agent_id for r in adapter.requests} == {agent.agent_id}


def test_normal_rule_brain_path_never_calls_provider_decide(sim):
    _forbid_decide(sim)
    adapters = _wrap_adapters(sim)

    for _ in range(40):
        sim.update(1.0)  # would raise AssertionError if decide() were used

    assert all(a.requests for a in adapters.values())


def test_three_rule_brain_agents_keep_distinct_adapters_and_providers(sim):
    runtimes = [sim.get_agent_brain_runtime(a) for a in sim.agents]
    assert len({id(r["adapter"]) for r in runtimes}) == 3
    assert len({id(r["provider"]) for r in runtimes}) == 3
    assert all(r["adapter"].provider is r["provider"] for r in runtimes)


def test_selected_response_passes_canonical_validation_before_translation(sim, monkeypatch):
    validated = []
    original_validate = agent_module.validate_agent_brain_response

    def _spy(response, legal_action_ids):
        validated.append(response)
        return original_validate(response, legal_action_ids)

    monkeypatch.setattr(agent_module, "validate_agent_brain_response", _spy)
    adapters = _wrap_adapters(sim)
    for _ in range(25):
        sim.update(1.0)

    asked = sum(len(a.responses) for a in adapters.values())
    assert asked > 0
    assert len(validated) >= asked
    responses = [r for a in adapters.values() for r in a.responses]
    assert all(any(v is r for v in validated) for r in responses)


def test_invalid_baseline_response_is_repaired_by_the_shared_contract_not_executed(sim):
    agent = sim.agents[0]
    runtime = sim.get_agent_brain_runtime(agent)
    context = sim.brain_context_builder.build(sim, agent)
    offered = {a["action_type"] for a in context.action_affordances}
    assert ExecutableActionType.MEETING.value not in offered

    class _IllegalAdapter(_RecordingAdapter):
        def choose_action(self, request):
            return AgentBrainResponse.from_dict(
                {
                    "response_id": "illegal",
                    "agent_id": request.agent_id,
                    "plan": {
                        "plan_id": "p",
                        "plan_horizon": 1,
                        "ordered_goals": [],
                        "ordered_actions": [{"step_index": 0, "action_type": "meeting", "expected_purpose": "illegal"}],
                        "next_action": {"step_index": 0, "action_type": "meeting", "expected_purpose": "illegal"},
                        "confidence": 1.0,
                    },
                }
            )

    runtime["adapter"] = _IllegalAdapter(runtime["adapter"])
    decision = agent._select_decision_through_pilot_adapter(sim, context, "unit_test", runtime=runtime)

    assert decision.selected_action.value in offered
    assert decision.selected_action != ExecutableActionType.MEETING


def test_baseline_target_and_transport_duration_survive_grounding_and_validation(sim):
    agent = sim.agents[0]
    runtime = sim.get_agent_brain_runtime(agent)
    context = sim.brain_context_builder.build(sim, agent)
    request = agent._build_sync_pilot_request(sim, context, "unit_test")
    response = runtime["adapter"].choose_action(request)
    # the baseline's internal method id is not a task plan-method and must not trigger
    # the unknown-method fallback (which drops targets)
    assert response.plan.plan_method_id is None
    grounded = agent._ground_validate_and_decide(response, context, request, sim, runtime)
    assert "unknown_method_fallback" not in grounded["response"].plan.notes
    assert grounded["decision"].target_id == response.plan.next_action.target_id

    project_id, _ = sim.environment.construction.create_project("site_b", structure_type="house")  # transport names an exact project
    context = sim.brain_context_builder.build(sim, agent)  # affordances must reflect the new project
    request = agent._build_sync_pilot_request(sim, context, "unit_test")
    transport = AgentBrainResponse.from_dict(
        {
            "response_id": "t",
            "agent_id": agent.agent_id,
            "plan": {
                "plan_id": "p",
                "plan_horizon": 1,
                "ordered_goals": [],
                "ordered_actions": [{"step_index": 0, "action_type": "transport_resources", "target_id": project_id, "expected_purpose": "t"}],
                "next_action": {"step_index": 0, "action_type": "transport_resources", "target_id": project_id, "expected_purpose": "t"},
                "confidence": 0.9,
            },
        }
    )
    decision = agent._ground_validate_and_decide(transport, context, request, sim, runtime)["decision"]
    assert any(str(a).startswith("duration_s=") for a in decision.assumptions)


def _comparable_control(control):
    method = dict(control.get("method_state") or {})
    return {
        "mode": control.get("mode"),
        "previous_mode": control.get("previous_mode"),
        "mode_dwell_steps": control.get("mode_dwell_steps"),
        "mode_history": [(h.get("mode"), h.get("reason")) for h in control.get("mode_history", [])],
        "active_method_id": method.get("active_method_id"),
        "active_method_step": method.get("active_method_step"),
    }


def test_adapter_path_matches_legacy_decide_decision_and_controller_state(sim):
    """Contract parity: same context -> same action/target and same controller-state update."""
    checked = 0
    for tick in range(1, 181):
        sim.update(1.0)
        if tick % 15 != 0:
            continue
        for agent in sim.agents:
            context = sim.brain_context_builder.build(sim, agent)
            context.individual_cognitive_state.setdefault("control_state", dict(agent.control_state))
            context.individual_cognitive_state["control_state"]["post_acquisition_until"] = int(agent.post_acquisition_stabilization_until)
            legacy_ctx = copy.copy(context)
            legacy_ctx.individual_cognitive_state = dict(context.individual_cognitive_state)
            legacy_ctx.individual_cognitive_state["control_state"] = copy.deepcopy(context.individual_cognitive_state["control_state"])
            adapter_ctx = copy.copy(context)
            adapter_ctx.individual_cognitive_state = dict(context.individual_cognitive_state)
            adapter_ctx.individual_cognitive_state["control_state"] = copy.deepcopy(context.individual_cognitive_state["control_state"])

            provider = sim.get_agent_brain_runtime(agent)["provider"]
            legacy = provider.decide(legacy_ctx)
            via_adapter = agent._select_decision_through_pilot_adapter(sim, adapter_ctx, "parity_check")

            assert (via_adapter.selected_action, via_adapter.target_id, via_adapter.target_zone) == (
                legacy.selected_action,
                legacy.target_id,
                legacy.target_zone,
            ), (tick, agent.name)
            assert _comparable_control(adapter_ctx.individual_cognitive_state["control_state"]) == _comparable_control(
                legacy_ctx.individual_cognitive_state["control_state"]
            ), (tick, agent.name)
            checked += 1
    assert checked >= 30


def test_controller_state_persists_and_advances_across_successive_decisions(sim):
    agent = sim.agents[0]
    history_lengths = []
    dwell_seen = set()
    for _ in range(60):
        sim.update(1.0)
        history_lengths.append(len(agent.control_state.get("mode_history", [])))
        dwell_seen.add(int(agent.control_state.get("mode_dwell_steps", 0) or 0))

    assert history_lengths == sorted(history_lengths)  # history is never reset
    assert history_lengths[-1] >= 2  # BOOTSTRAP entry plus at least one transition
    assert len(dwell_seen) > 1  # dwell advances (state is carried between decisions)
    assert agent.control_state.get("last_policy_snapshot")
    assert agent.control_state.get("method_state")


def test_reroute_through_rulebrain_uses_adapter_not_provider_decide(sim):
    for _ in range(30):
        sim.update(1.0)
    agent = sim.agents[0]
    adapter = _RecordingAdapter(sim.get_agent_brain_runtime(agent)["adapter"])
    sim.get_agent_brain_runtime(agent)["adapter"] = adapter
    _forbid_decide(sim)
    context = sim.brain_context_builder.build(sim, agent)
    fallback = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="pivot fallback", confidence=0.5)

    decision = agent._reroute_decision_through_rulebrain_controller(
        context, fallback, sim_state=sim, pivot_origin="unit_test", reason="readiness_unlocked"
    )

    assert decision is not None
    assert adapter.requests and adapter.requests[-1].task_context.get("policy_seed_key") == "ctx"
    method_state = (agent.control_state.get("method_state") or {})
    assert any(o.get("origin") == "unit_test" for o in method_state.get("recent_step_outcomes", []))
