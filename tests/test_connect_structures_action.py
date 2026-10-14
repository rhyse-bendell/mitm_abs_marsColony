"""connect_structures: the canonical contract can express and execute a direct connector link between any legal endpoints."""

import copy
import tempfile

import pytest

from modules.action_catalog import ACTION_CATALOG
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construction import ConstructionManager
from modules.simulation import SimulationState
from modules.support_accounting import evaluate_structural_support

CONNECT = ExecutableActionType.CONNECT_STRUCTURES


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="connect_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _built(cm, structure_type):
    pid, _ = cm.create_project("site_b", structure_type=structure_type, author="system")
    assert pid, structure_type
    p = cm.projects[pid]
    p.update({"structurally_complete": True, "validated_complete": True, "status": "complete", "correct": True})
    return pid


def _target(cm, connector, a, b):
    return cm.encode_connection_target(connector, a, b)


def _decision(cm, connector, a, b):
    return BrainDecision(selected_action=CONNECT, target_id=_target(cm, connector, a, b), confidence=0.8)


def _colony(sim):
    cm = sim.environment.construction
    cm.sites["site_b"].capacity = 30
    ids = {t: _built(cm, t) for t in ("water_generator", "greenhouse", "house")}
    ids["water_connector"] = _built(cm, "water_connector")
    ids["food_connector"] = _built(cm, "food_connector")
    return cm, ids


def _gate(sim, agent, decision):
    return sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)


def test_the_action_contract_can_represent_a_generator_to_greenhouse_water_connection(sim):
    cm, ids = _colony(sim)
    decision = _decision(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"])
    assert decision.selected_action is CONNECT and CONNECT.value in ACTION_CATALOG
    assert cm.parse_connection_target(decision.target_id) == (ids["water_connector"], ids["water_generator"], ids["greenhouse"])
    assert sim.task_model is not None and CONNECT.value in sim.task_model.enabled_actions_for_role("Engineer")


def test_the_gate_accepts_a_legal_connection_for_any_role(sim):
    cm, ids = _colony(sim)
    decision = _decision(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"])
    for agent in sim.agents:  # Architect, Engineer, Botanist alike
        result = _gate(sim, agent, decision)
        assert result.legal, (agent.role, result.blockers)
    assert sim.task_model.enabled_actions_for_role("Architect") and CONNECT.value in sim.task_model.enabled_actions_for_role("Architect")


def test_the_gate_rejects_illegal_or_unbuilt_connections(sim):
    cm, ids = _colony(sim)
    agent = sim.agents[0]
    house2 = _built(cm, "house")
    cases = [
        (ids["water_connector"], ids["house"], house2, "invalid_connector_endpoint_types"),
        (ids["water_connector"], ids["greenhouse"], ids["house"], "invalid_connector_endpoint_types"),  # greenhouse<->house is food, not water
        (ids["food_connector"], ids["water_generator"], ids["greenhouse"], "invalid_connector_endpoint_types"),
        (ids["food_connector"], ids["water_generator"], ids["house"], "invalid_connector_endpoint_types"),
        (ids["water_connector"], ids["water_generator"], ids["water_generator"], "self_connection"),
    ]
    for connector, a, b, blocker in cases:
        result = _gate(sim, agent, _decision(cm, connector, a, b))
        assert not result.legal and blocker in result.blockers, (a, b, result.blockers)
    cm.projects[ids["greenhouse"]]["structurally_complete"] = False
    assert "endpoint_not_structurally_complete" in _gate(sim, agent, _decision(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"])).blockers
    assert "malformed_connection_target" in _gate(sim, agent, BrainDecision(CONNECT, target_id="nonsense", confidence=0.5)).blockers
    assert "connector_project_not_found" in _gate(sim, agent, BrainDecision(CONNECT, target_id="a|b|c", confidence=0.5)).blockers
    assert "unknown_structure_reference" in _gate(sim, agent, _decision(cm, ids["water_connector"], ids["water_generator"], "nope")).blockers


def test_food_semantics_are_unchanged(sim):
    cm, ids = _colony(sim)
    assert _gate(sim, sim.agents[0], _decision(cm, ids["food_connector"], ids["greenhouse"], ids["house"])).legal
    options = {(o["connection_type"], frozenset((o["from_project_id"], o["to_project_id"]))) for o in cm.list_connection_options()}
    assert options == {
        ("water", frozenset((ids["water_generator"], ids["house"]))),
        ("water", frozenset((ids["water_generator"], ids["greenhouse"]))),
        ("food", frozenset((ids["greenhouse"], ids["house"]))),
    }


def test_execution_adds_the_connector_graph_edge_and_support_accounting_sees_it(sim):
    cm, ids = _colony(sim)
    agent = next(a for a in sim.agents if a.role != "Engineer")  # no Engineer/Botanist requirement
    decision = _decision(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"])
    assert evaluate_structural_support(cm)["food"]["watered_greenhouses"] == {"min": 0, "max": 0}  # dry before
    action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
    assert action["decision_action"] == CONNECT.value
    assert not any(c.get("connector_project_id") == ids["water_connector"] for c in cm.connectors)  # translation is not mutation
    assert agent._execute_connect_structures(action, sim) is True
    edge = [c for c in cm.connectors if c["connector_project_id"] == ids["water_connector"]]
    assert len(edge) == 1 and {edge[0]["from_project_id"], edge[0]["to_project_id"]} == {ids["water_generator"], ids["greenhouse"]}
    food = evaluate_structural_support(cm)["food"]
    assert food["greenhouses"][0]["water"] == "served" and food["watered_greenhouses"] == {"min": 1, "max": 1}
    assert not _gate(sim, agent, decision).legal and "already_connected" in _gate(sim, agent, decision).blockers


def test_a_watered_greenhouse_connected_to_a_house_contributes_effective_food(sim):
    cm, ids = _colony(sim)
    agent = sim.agents[0]
    for decision in (
        _decision(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"]),
        _decision(cm, ids["food_connector"], ids["greenhouse"], ids["house"]),
        _decision(cm, ids["water_connector"], ids["house"], ids["water_generator"]),
    ):
        action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
        assert agent._execute_connect_structures(action, sim)
    report = evaluate_structural_support(cm)
    assert report["food"]["effective_civilian_equivalent_capacity"] == {"min": 5, "max": 5}
    assert report["housing"]["fully_supported_structures"]["min"] == 1


def test_affordance_generation_is_read_only_and_lists_the_legal_connections(sim):
    cm, ids = _colony(sim)
    before = (copy.deepcopy(cm.connectors), copy.deepcopy(cm.projects), len(cm.projects))
    for agent in sim.agents:
        affordances = sim.brain_context_builder.build(sim, agent).action_affordances
        connects = [a for a in affordances if a["action_type"] == CONNECT.value]
        targets = {a["target_id"] for a in connects}
        assert _target(cm, ids["water_connector"], ids["greenhouse"], ids["water_generator"]) in targets or _target(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"]) in targets
        assert len(connects) == 3
    assert (cm.connectors, cm.projects, len(cm.projects)) == before
    cm.list_connection_options(); cm.check_connection(ids["water_connector"], ids["water_generator"], ids["greenhouse"])
    assert (cm.connectors, cm.projects, len(cm.projects)) == before


def test_a_pilot_without_the_task_knowledge_is_not_made_to_choose_the_connection(sim):
    """The affordance is offered with a low, knowledge-independent utility; nothing here asserts the baseline picks it."""
    cm, ids = _colony(sim)
    affordances = sim.brain_context_builder.build(sim, sim.agents[0]).action_affordances
    connects = [a for a in affordances if a["action_type"] == CONNECT.value]
    assert connects and all(a["utility"] <= 0.1 for a in connects)


def test_a_connect_decision_survives_the_runtime_action_queue_and_executes_on_completion(sim):
    """Regression: the active-action copy dropped the connection target, so a selected connect failed as malformed."""
    cm, ids = _colony(sim)
    agent = sim.agents[0]
    decision = _decision(cm, ids["water_connector"], ids["water_generator"], ids["greenhouse"])
    agent.current_action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    agent.active_actions = []
    for _ in range(6):
        agent._advance_active_actions(1.0, sim)
    edges = [c for c in cm.connectors if c["connector_project_id"] == ids["water_connector"]]
    assert len(edges) == 1
