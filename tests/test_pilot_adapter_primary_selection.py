from types import SimpleNamespace

from modules.agent import Agent
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse
from modules.action_schema import ExecutableActionType


def _response(agent_id="agent-1"):
    return AgentBrainResponse.from_dict(
        {
            "response_id": "response-1",
            "agent_id": agent_id,
            "plan": {
                "plan_id": "plan-1",
                "plan_horizon": 1,
                "ordered_goals": [],
                "ordered_actions": [
                    {"step_index": 0, "action_type": "wait", "expected_purpose": "adapter-selected wait"}
                ],
                "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "adapter-selected wait"},
                "confidence": 1.0,
            },
        }
    )


class _ContextBuilder:
    def build(self, sim_state, agent):
        return SimpleNamespace(
            individual_cognitive_state={},
            action_affordances=[{"action_type": "wait"}],
            team_state={},
            static_task_context={},
            world_snapshot={},
            history_bands={},
        )


class _Provider:
    def __init__(self):
        self.generate_plan_called = False
        self.last_trace = {"runtime_disposition": "provider_trace", "result_source": "provider"}

    def generate_plan(self, request_packet):
        self.generate_plan_called = True
        return _response(request_packet.agent_id)


class _Adapter:
    def __init__(self, provider):
        self.provider = provider
        self.choose_action_called = False

    def choose_action(self, request_packet):
        self.choose_action_called = True
        return _response(request_packet.agent_id)


def _request():
    return AgentBrainRequest(
        request_id="request-1",
        tick=0,
        sim_time=0.0,
        agent_id="agent-1",
        display_name="Agent One",
        task_id="task-1",
        phase="test",
        local_context_summary="test",
        local_observations=[],
        working_memory_summary={},
        inbox_summary=[],
        current_goal_stack=[],
        current_plan_summary={},
        allowed_actions=[{"action_type": "wait"}],
        planning_horizon_config={},
        request_explanation=False,
    )


def _sim_state(provider, adapter_marker=True):
    sim_state = SimpleNamespace(
        brain_context_builder=_ContextBuilder(),
        brain_provider=provider,
        brain_backend_config=SimpleNamespace(backend="test", local_model="none"),
        configured_brain_backend="test",
        effective_brain_backend="test",
        time=0.0,
        task_model=None,
    )
    if adapter_marker is not False:
        sim_state.pilot_adapter = adapter_marker
    return sim_state


def _run_planner(sim_state):
    agent = Agent("agent-1", "Engineer", agent_id="agent-1", display_name="Agent One")
    agent._apply_trait_bias_to_decision = lambda decision, context, sim, trigger: decision
    return agent._execute_planner_request_sync(
        sim_state,
        "test_trigger",
        _request(),
        False,
        0,
        0.0,
        0.0,
        "trace-1",
    )


def test_primary_planner_path_uses_pilot_adapter_choose_action():
    provider = _Provider()
    adapter = _Adapter(provider)

    result = _run_planner(_sim_state(provider, adapter))

    assert adapter.choose_action_called is True
    assert provider.generate_plan_called is False
    assert result["decision"].selected_action == ExecutableActionType.WAIT
    assert result["status"] == "accepted"


def test_primary_planner_path_falls_back_to_provider_without_pilot_adapter():
    provider = _Provider()

    result = _run_planner(_sim_state(provider, adapter_marker=False))

    assert provider.generate_plan_called is True
    assert result["decision"].selected_action == ExecutableActionType.WAIT
    assert result["status"] == "accepted"


def test_primary_planner_trace_metadata_resolves_from_wrapped_provider():
    provider = _Provider()
    provider.last_trace = {"runtime_disposition": "accepted_as_is", "result_source": "wrapped_provider"}
    adapter = _Adapter(provider)

    result = _run_planner(_sim_state(provider, adapter))

    assert result["runtime_disposition"] == "accepted_as_is"
    assert result["result_source"] == "wrapped_provider"
    assert result["trace"]["provider_trace"] == provider.last_trace


def test_agent_static_boundary_routes_primary_selection_through_pilot_adapter():
    text = open("modules/agent.py", encoding="utf-8").read()

    assert ".choose_action(request_packet)" in text
    assert "pilot_adapter.choose_action(request_packet)" in text
    assert "Compatibility fallback" in text
