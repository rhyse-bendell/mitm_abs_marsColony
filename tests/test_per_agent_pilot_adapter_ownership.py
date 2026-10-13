"""Per-agent PilotAdapter ownership.

Each agent's brain runtime owns a coherent {provider, adapter} pair, and normal
pilot selection resolves through that agent's adapter (not the simulation-wide one).
"""

import tempfile
import time

import pytest

from modules.brain_contract import AgentBrainResponse
from modules.pilot_adapter import GenericBrainProviderPilotAdapter
from modules.procedural_baseline_pilot import ProceduralBaselinePilotAdapter
from modules.simulation import SimulationState


def _agent_configs(backends):
    return [
        {
            "name": role,
            "role": role,
            "template_id": f"mars_{role.lower()}",
            "traits": {},
            "brain_config": {"backend": backend, "fallback_backend": "rule_brain"},
        }
        for role, backend in backends.items()
    ]


@pytest.fixture
def make_sim():
    sims = []

    def _make(backends):
        tmp = tempfile.TemporaryDirectory()
        sim = SimulationState(phases=[], project_root=tmp.name, agent_configs=_agent_configs(backends))
        sims.append((sim, tmp))
        return sim, {a.role: a for a in sim.agents}

    yield _make
    for sim, tmp in sims:
        sim.stop()
        tmp.cleanup()


def _wait_response(agent_id, marker):
    return AgentBrainResponse.from_dict(
        {
            "response_id": f"resp-{marker}",
            "agent_id": agent_id,
            "plan": {
                "plan_id": f"plan-{marker}",
                "plan_horizon": 1,
                "ordered_goals": [],
                "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": marker}],
                "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": marker},
                "confidence": 1.0,
            },
        }
    )


class _StubProvider:
    def __init__(self, marker):
        self.marker = marker
        self.calls = 0
        self.last_trace = {"runtime_disposition": f"disp-{marker}", "result_source": f"src-{marker}"}

    def generate_plan(self, request_packet):
        self.calls += 1
        return _wait_response(request_packet.agent_id, self.marker)


def _install_stub_pilot(runtime, marker):
    provider = _StubProvider(marker)
    runtime["provider"] = provider
    runtime["adapter"] = GenericBrainProviderPilotAdapter(provider, pilot_id=f"stub-{marker}")
    return provider, runtime["adapter"]


def _select_action(sim, agent):
    context = sim.brain_context_builder.build(sim, agent)
    request = agent._build_brain_request(sim, context, False, "test_trigger")
    return agent._execute_planner_request_sync(sim, "test_trigger", request, False, 0, 0.0, time.perf_counter(), "trace-1")


def test_agents_with_different_backends_own_distinct_provider_adapter_pairs(make_sim):
    sim, agents = make_sim({"Architect": "rule_brain", "Engineer": "local_stub", "Botanist": "rule_brain"})
    rt = {role: sim.get_agent_brain_runtime(a) for role, a in agents.items()}

    assert isinstance(rt["Architect"]["adapter"], ProceduralBaselinePilotAdapter)
    assert isinstance(rt["Engineer"]["adapter"], GenericBrainProviderPilotAdapter)
    assert rt["Architect"]["pilot_id"] == "procedural_baseline"
    assert rt["Engineer"]["pilot_id"] == "local_stub"
    assert rt["Engineer"]["adapter"].pilot_id == "local_stub"
    assert len({id(r["adapter"]) for r in rt.values()}) == 3
    assert len({id(r["provider"]) for r in rt.values()}) == 3
    for r in rt.values():
        assert r["adapter"].provider is r["provider"]
        assert r["adapter"] is not sim.pilot_adapter
        assert r["provider"] is not sim.brain_provider


def test_procedural_baseline_agents_do_not_share_pilot_instances(make_sim):
    sim, agents = make_sim({"Architect": "rule_brain", "Engineer": "rule_brain", "Botanist": "rule_brain"})
    rt = [sim.get_agent_brain_runtime(a) for a in agents.values()]

    assert all(isinstance(r["adapter"], ProceduralBaselinePilotAdapter) for r in rt)
    assert len({id(r["provider"]) for r in rt}) == 3
    assert len({id(r["adapter"]) for r in rt}) == 3
    assert all(r["adapter"].provider is r["provider"] for r in rt)


def test_selection_uses_only_the_selecting_agents_adapter(make_sim):
    sim, agents = make_sim({"Architect": "rule_brain", "Engineer": "rule_brain", "Botanist": "rule_brain"})
    a_provider, a_adapter = _install_stub_pilot(sim.get_agent_brain_runtime(agents["Architect"]), "A")
    b_provider, b_adapter = _install_stub_pilot(sim.get_agent_brain_runtime(agents["Engineer"]), "B")
    sim_provider = _StubProvider("SIM")
    sim.brain_provider = sim_provider
    sim.pilot_adapter = GenericBrainProviderPilotAdapter(sim_provider, pilot_id="stub-SIM")

    result = _select_action(sim, agents["Architect"])

    assert a_provider.calls == 1
    assert b_provider.calls == 0
    assert sim_provider.calls == 0
    assert result["decision"].selected_action.value == "wait"
    assert result["decision"].reason_summary == "A"


def test_trace_metadata_comes_from_the_provider_wrapped_by_the_agents_adapter(make_sim):
    sim, agents = make_sim({"Architect": "rule_brain", "Engineer": "rule_brain", "Botanist": "rule_brain"})
    a_provider, _ = _install_stub_pilot(sim.get_agent_brain_runtime(agents["Architect"]), "A")
    _install_stub_pilot(sim.get_agent_brain_runtime(agents["Engineer"]), "B")
    sim.brain_provider = _StubProvider("SIM")
    sim.pilot_adapter = GenericBrainProviderPilotAdapter(sim.brain_provider, pilot_id="stub-SIM")

    result = _select_action(sim, agents["Architect"])

    assert result["runtime_disposition"] == "disp-A"
    assert result["result_source"] == "src-A"
    assert result["trace"]["provider_trace"] == a_provider.last_trace


def test_hard_demotion_rebuilds_only_that_agents_adapter(make_sim):
    sim, agents = make_sim({"Architect": "rule_brain", "Engineer": "local_stub", "Botanist": "rule_brain"})
    eng_rt = sim.get_agent_brain_runtime(agents["Engineer"])
    others = {role: sim.get_agent_brain_runtime(agents[role]) for role in ("Architect", "Botanist")}
    before = {role: (r["provider"], r["adapter"], r["pilot_id"], r["hard_demoted"]) for role, r in others.items()}
    old_provider, old_adapter = eng_rt["provider"], eng_rt["adapter"]

    assert sim.hard_demote_agent_backend(agents["Engineer"], reason="test", activate_bootstrap=False)

    assert eng_rt["provider"] is not old_provider
    assert eng_rt["adapter"] is not old_adapter
    assert eng_rt["adapter"].provider is eng_rt["provider"]
    assert isinstance(eng_rt["adapter"], ProceduralBaselinePilotAdapter)
    assert eng_rt["pilot_id"] == "procedural_baseline"
    assert eng_rt["configured_pilot_id"] == "local_stub"
    assert eng_rt["hard_demoted"] is True
    for role, r in others.items():
        assert (r["provider"], r["adapter"], r["pilot_id"], r["hard_demoted"]) == before[role]
        assert sim.get_agent_brain_runtime(agents[role]) is r
