"""Connector link authority: connect_structures is the only runtime path that creates a connector graph link."""

import ast
import copy
import tempfile
from pathlib import Path

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState

CONNECT = ExecutableActionType.CONNECT_STRUCTURES
MODULES = Path(__file__).resolve().parents[1] / "modules"


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="authority_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _built(cm, structure_type):
    pid, _ = cm.create_project("site_b", structure_type=structure_type, author="system")
    assert pid, structure_type
    cm.projects[pid].update({"structurally_complete": True, "validated_complete": True, "status": "complete", "correct": True})
    return pid


def _world(sim, *, with_providers=True):
    cm = sim.environment.construction
    cm.sites["site_b"].capacity = 30
    ids = {"house": _built(cm, "house")}
    if with_providers:
        for t in ("water_generator", "greenhouse", "water_connector", "food_connector"):
            ids[t] = _built(cm, t)
    return cm, ids


def _snapshot(cm):
    return copy.deepcopy((cm.projects, cm.connectors))


def _run(sim, agent, cm, connector, a, b):
    decision = BrainDecision(selected_action=CONNECT, target_id=cm.encode_connection_target(connector, a, b), confidence=0.9)
    pivoted = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="test")
    assert pivoted.selected_action is CONNECT  # a legal explicit connect is not rewritten
    action = agent._translate_brain_decision_to_legacy_action(pivoted, sim.environment, sim_state=sim)[0]
    return agent._execute_connect_structures(action, sim)


@pytest.mark.parametrize("action", [ExecutableActionType.WAIT, ExecutableActionType.VALIDATE_CONSTRUCTION, ExecutableActionType.REASSESS_PLAN, ExecutableActionType.START_CONSTRUCTION])
def test_an_unsupported_house_causes_no_middleware_connection_or_project_creation(sim, action):
    cm, ids = _world(sim, with_providers=False)
    agent = sim.agents[0]
    assert cm.get_support_deficit_report()  # the house lacks food and water
    before = _snapshot(cm)
    decision = BrainDecision(selected_action=action, target_id=ids["house"], confidence=0.8)
    for _ in range(3):
        result = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="test")
        assert result.selected_action is not CONNECT
        target = cm.projects.get(str(result.target_id or ""), {})
        assert target.get("type") not in {"water_generator", "greenhouse", "water_connector", "food_connector"}
    assert _snapshot(cm) == before  # no provider/connector project created, no link attached


def test_with_built_providers_and_connectors_middleware_still_attaches_nothing(sim):
    cm, ids = _world(sim)
    agent = sim.agents[0]
    before = _snapshot(cm)
    agent._apply_policy_pivots(BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=None, confidence=0.5), sim.environment, sim_state=sim, pivot_origin="test")
    assert cm.connectors == before[1] == []
    assert cm.get_support_deficit_report()[0]["missing_support"] == {"food": 1, "water": 1}


def test_explicit_water_and_food_connections_use_the_same_canonical_action(sim):
    cm, ids = _world(sim)
    agent = sim.agents[1]
    assert _run(sim, agent, cm, ids["water_connector"], ids["house"], ids["water_generator"])  # house -> generator
    assert _run(sim, agent, cm, ids["water_connector"], ids["greenhouse"], ids["water_generator"])  # greenhouse -> generator
    assert _run(sim, agent, cm, ids["food_connector"], ids["greenhouse"], ids["house"])  # food
    pairs = {(c["connection_type"], frozenset((c["from_project_id"], c["to_project_id"]))) for c in cm.connectors}
    assert pairs == {
        ("water", frozenset((ids["house"], ids["water_generator"]))),
        ("water", frozenset((ids["greenhouse"], ids["water_generator"]))),
        ("food", frozenset((ids["greenhouse"], ids["house"]))),
    }
    assert cm.get_support_deficit_report() == []  # house now has its food and water links


def test_context_and_deficit_generation_are_read_only_and_offer_without_choosing(sim):
    cm, ids = _world(sim)
    before = _snapshot(cm)
    for agent in sim.agents:
        packet = sim.brain_context_builder.build(sim, agent)
        deficits = packet.world_snapshot["support_deficits"] if hasattr(packet, "world_snapshot") else None
        if deficits is not None:
            assert deficits and deficits[0]["structure_id"] == ids["house"]
            options = deficits[0]["legal_connect_options"]
            assert {o["connection_type"] for o in options} == {"water", "food"}
        offered = [a for a in packet.action_affordances if a["action_type"] == CONNECT.value]
        assert offered and all(a["utility"] <= 0.1 for a in offered)  # offered, never preferred
    assert _snapshot(cm) == before
    assert cm.connectors == []


def test_no_runtime_module_attaches_connectors_or_creates_support_projects_outside_the_canonical_path():
    """Static guard: attach_connector / connector-project creation may only be called from the connect execution."""
    allowed_callers = {"attach_connector": {("agent.py", "_execute_connect_structures")}}
    forbidden = {"find_or_create_connector_project", "add_connector"}
    offenders = []
    for path in sorted(MODULES.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                    if name in forbidden:
                        offenders.append((path.name, fn.name, name))
                    if name == "attach_connector" and (path.name, fn.name) not in allowed_callers["attach_connector"]:
                        offenders.append((path.name, fn.name, name))
    assert offenders == []


def test_agent_has_no_support_deficit_rewrite_or_provider_site_chooser():
    from modules.agent import Agent

    assert not hasattr(Agent, "_resolve_support_deficit_decision")
    assert not hasattr(Agent, "_choose_support_site")
