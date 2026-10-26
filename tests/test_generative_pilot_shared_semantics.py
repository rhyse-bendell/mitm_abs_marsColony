"""A generative (Ollama-style) Pilot sees and uses the same legal actions, arguments and artifact semantics as the Procedural Baseline."""

import json

import pytest

from modules.action_gate import AgentActionGate
from modules.action_schema import BrainDecision
from modules.controller_context import partition_affordances
from modules.knowledge import Information
from tests.test_ollama_controller import FakeOllama, OLLAMA_A, _adapter, _never_a_real_service, _sim, _stop  # noqa: F401  (the autouse fake service: no test needs a real Ollama)


@pytest.fixture
def world():
    st = _sim({"Engineer": OLLAMA_A})
    agent = next(a for a in st.agents if a.role == "Engineer")
    yield st, agent
    _stop(st)


def _pick(monkeypatch, st, agent, predicate):
    """Offer the generative adapter the real executable affordances and let a scripted model choose the first one satisfying ``predicate``."""
    ctx = st.brain_context_builder.build(st, agent)
    executable = partition_affordances(ctx.action_affordances)[0]
    idx, chosen = next((i, a) for i, a in enumerate(executable) if predicate(a))
    request = agent._build_brain_request(st, ctx, False, "t", request_id="r1")
    adapter = _adapter(None)
    monkeypatch.setattr(adapter, "client", FakeOllama(reply=lambda m, o, mod: json.dumps({"affordance_id": f"a{idx}", "arguments": dict(chosen.get("action_args") or {}), "decision_note": "scripted"})))
    response = adapter.choose_action(request)
    return chosen, response.plan.next_action, adapter


def test_knowledge_externalization_is_offered_and_executed_with_the_same_arguments(monkeypatch, world):
    st, agent = world
    agent.mental_model["information"].add(Information("I_ARCH_HOUSING_DEPENDENCIES", "held, task-relevant"))
    chosen, step, adapter = _pick(monkeypatch, st, agent, lambda a: a["action_type"] == "externalize_knowledge")
    assert step.action_type.value == "externalize_knowledge" and dict(step.action_args) == {"kind": "evidence", "element_id": "I_ARCH_HOUSING_DEPENDENCIES"}
    assert adapter.last_trace["validation_outcome"] == "valid"
    decision = BrainDecision(selected_action=step.action_type, target_id=step.target_id, action_args=dict(step.action_args), confidence=0.5)
    assert AgentActionGate().evaluate(agent=agent, decision=decision, sim_state=st).legal  # the same gate every Pilot faces


def test_a_response_to_a_received_request_is_a_Pilot_choice_through_the_same_affordance(monkeypatch, world):
    st, agent = world
    requester = next(a for a in st.agents if a is not agent)
    agent.mental_model["information"].add(Information("I_ENG_CONNECTOR_RULES", "held"))
    agent.request_ledger.receive(requester.name, {"information:I_ENG_CONNECTOR_RULES"}, float(st.time))
    agent.position = (requester.position[0] + 1.0, requester.position[1])
    chosen, step, _adapter_obj = _pick(monkeypatch, st, agent, lambda a: bool(a.get("response_to_request")))
    assert chosen["target_id"] == requester.name and chosen["can_provide"] == ["information:I_ENG_CONNECTOR_RULES"]
    assert step.action_type.value == "communicate" and step.target_id == requester.name
    assert not [m for m in agent.team_plan_state.get("pending_outbound_messages", []) if m["type"] in {"TDP", "TIP", "TKP"}]  # nothing was sent for the Pilot


def test_construction_options_are_the_same_menu_with_no_hint(monkeypatch, world):
    st, agent = world
    cm = st.environment.construction
    pid, _ = cm.create_project("site_b", structure_type="house", project_id_override="site_b_house_001")
    p = cm.projects[pid]
    required = p["required_resources"]["bricks"]
    p.update({"started": True, "staged_resources": {"bricks": required}, "delivered_resources": {"bricks": required}})
    cm.update()
    chosen, step, adapter = _pick(monkeypatch, st, agent, lambda a: a.get("design_option") == "mixed_floor")
    assert dict(step.action_args) == {"design_slot": "floor_wall_colour", "design_option": "mixed_floor"}  # a non-compliant option is just another offered choice
    ctx = st.brain_context_builder.build(st, agent)
    menu = {a["design_option"] for a in partition_affordances(ctx.action_affordances)[0] if a.get("design_slot") == "floor_wall_colour"}
    assert menu == {"single_solid_floor_walls_differ", "single_solid_floor_walls_same", "mixed_floor"}
    prompt_blob = json.dumps(adapter.last_trace, default=str).lower() + json.dumps(partition_affordances(ctx.action_affordances)[0], default=str).lower()
    assert "compliant" not in prompt_blob and "correct" not in prompt_blob
