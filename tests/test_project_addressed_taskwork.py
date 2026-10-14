"""Project-addressed taskwork: the Pilot names the exact project; nothing downstream swaps it for another one at the site."""

import copy
import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.agent import Agent
from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="addressing_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _site_b_house_and_connector(sim):
    """The original defect: a materially complete house and an unfinished connector at the same site."""
    cm = sim.environment.construction
    house, _ = cm.create_project("site_b", "house")
    connector, _ = cm.create_project("site_b", "water_connector")
    for _ in range(int(cm.projects[house]["required_resources"]["bricks"])):
        assert cm.deliver_resource(house, "bricks", 1)
    assert cm.projects[house]["resource_complete"] and not cm.projects[connector]["resource_complete"]
    return cm, house, connector


def _affordances(sim, agent, action):
    return [a for a in sim.brain_context_builder.build(sim, agent).action_affordances if a["action_type"] == action.value]


def _state(cm):
    return copy.deepcopy((cm.projects, cm.connectors, cm._project_counters, {k: v.quantity for k, v in cm.resource_nodes.items()}))


def test_affordances_name_each_exact_project_and_are_read_only(sim):
    cm, house, connector = _site_b_house_and_connector(sim)
    before = _state(cm)
    for agent in sim.agents:
        context = sim.brain_context_builder.build(sim, agent)
        targets = {(a["action_type"], a["target_id"]) for a in context.action_affordances}
        assert (A.TRANSPORT_RESOURCES.value, connector) in targets  # only the project that needs bricks
        assert (A.TRANSPORT_RESOURCES.value, house) not in targets
        assert {t for _, t in targets} .isdisjoint({"resource_zone_to_work_zone", "active_construction"})  # no Mecha-resolved placeholders
        assert (A.START_CONSTRUCTION.value, house) in targets and (A.START_CONSTRUCTION.value, connector) in targets  # existing projects, by id
    assert _state(cm) == before


def test_each_family_offers_every_eligible_project_separately(sim):
    cm = sim.environment.construction
    a, _ = cm.create_project("site_b", "house")
    b, _ = cm.create_project("site_b", "greenhouse")
    agent = sim.agents[0]
    for pid in (a, b):
        cm.projects[pid]["build_steps"][0]["completed"] = True  # physically in progress
    assert {x["target_id"] for x in _affordances(sim, agent, A.CONTINUE_CONSTRUCTION)} >= {a, b}
    assert {x["target_id"] for x in _affordances(sim, agent, A.REPAIR_OR_CORRECT_CONSTRUCTION)} == {a, b}
    for pid in (a, b):
        for step in cm.projects[pid]["build_steps"]:
            step["completed"] = True
        cm.projects[pid]["structurally_complete"] = True
    assert {x["target_id"] for x in _affordances(sim, agent, A.VALIDATE_CONSTRUCTION)} == {a, b}
    assert not _affordances(sim, agent, A.CONTINUE_CONSTRUCTION)


def test_gate_and_translation_never_redirect_an_exact_project_target(sim):
    cm, house, connector = _site_b_house_and_connector(sim)
    agent = sim.agents[0]
    before = _state(cm)
    to_house = BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id=house, assumptions=["duration_s=30"], confidence=0.9)
    gate = sim.action_gate.evaluate(agent=agent, decision=to_house, sim_state=sim)
    assert not gate.legal and "project_does_not_need_material" in gate.blockers
    action = agent._translate_brain_decision_to_legacy_action(to_house, sim.environment, sim_state=sim)[0]
    assert action["type"] == "idle" and action.get("project_id") != connector  # blocked, not quietly sent to the connector
    to_connector = BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id=connector, assumptions=["duration_s=30"], confidence=0.9)
    assert sim.action_gate.evaluate(agent=agent, decision=to_connector, sim_state=sim).legal
    action = agent._translate_brain_decision_to_legacy_action(to_connector, sim.environment, sim_state=sim)[0]
    assert action["type"] == "transport_resources" and action["project_id"] == connector
    assert _state(cm) == before  # translation mutates nothing


@pytest.mark.parametrize("family", [A.CONTINUE_CONSTRUCTION, A.VALIDATE_CONSTRUCTION, A.REPAIR_OR_CORRECT_CONSTRUCTION, A.START_CONSTRUCTION])
def test_construction_families_bind_the_named_project_even_with_another_at_the_site(sim, family):
    cm = sim.environment.construction
    first, _ = cm.create_project("site_b", "house")
    second, _ = cm.create_project("site_b", "greenhouse")
    for pid in (first, second):
        for step in cm.projects[pid]["build_steps"]:
            step["completed"] = True
        cm.projects[pid]["structurally_complete"] = True
        cm.projects[pid]["started"] = True
    agent = sim.agents[0]
    decision = BrainDecision(selected_action=family, target_id=second, confidence=0.9)  # `first` would win any site-level tiebreak
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    bound = {a.get("project_id") for a in actions if a.get("project_id")}
    assert bound <= {second}, (family, bound)


def test_a_pilot_chosen_transport_reaches_only_the_named_connector_and_identity_survives_the_build(sim, monkeypatch):
    """Original failure: a materially complete house must not intercept bricks meant for the unfinished connector."""
    cm, house, connector = _site_b_house_and_connector(sim)
    target = sim.agents[0]
    mode = {"value": "transport"}

    def pilot(self, sim_state, context, trigger_reason, runtime=None):
        if self is not target:
            return BrainDecision(selected_action=A.WAIT, confidence=0.9)
        if mode["value"] == "transport":
            return BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id=connector, assumptions=["duration_s=30"], confidence=0.9)
        return BrainDecision(selected_action=A.CONTINUE_CONSTRUCTION, target_id=connector, confidence=0.9)

    monkeypatch.setattr(Agent, "_select_decision_through_pilot_adapter", pilot)
    events = []
    sim.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
    house_before = copy.deepcopy(cm.projects[house]["delivered_resources"])
    for _ in range(600):
        sim.update(1.0)
        if cm.projects[connector]["resource_complete"]:
            mode["value"] = "build"
        if cm.projects[connector]["structurally_complete"]:
            break
    assert cm.projects[connector]["delivered_resources"]["bricks"] == cm.projects[connector]["required_resources"]["bricks"] == 4
    assert cm.projects[connector]["resource_complete"] and cm.projects[connector]["structurally_complete"]
    assert cm.projects[house]["delivered_resources"] == house_before and not cm.projects[house]["structurally_complete"]  # untouched
    worked = {p.get("project_id") for name, p in events if name == "construction_build_episode"}
    assert worked == {connector}  # every build episode was on the project the Pilot named
    delivered_to = {p.get("project_id") for name, p in events if name == "construction_transport_pickup" and p.get("project_id")}
    assert delivered_to <= {connector}