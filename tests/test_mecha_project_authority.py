"""Mecha project authority: the Pilot chooses which project gets bricks / work; middleware only grounds, blocks and reports."""

import copy
import tempfile

import pytest

from tests.construction_helpers import design_args_for

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.agent import Agent
from modules.simulation import SimulationState

TASKWORK = {A.TRANSPORT_RESOURCES, A.START_CONSTRUCTION, A.CONTINUE_CONSTRUCTION, A.VALIDATE_CONSTRUCTION, A.REPAIR_OR_CORRECT_CONSTRUCTION}


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="authority_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _state(cm):
    return copy.deepcopy((cm.projects, cm.connectors, cm._project_counters, {k: v.quantity for k, v in cm.resource_nodes.items()}))


def _physical(cm, pid):
    """World-changing state of a project (epistemic bookkeeping such as provenance/discussion is excluded)."""
    p = cm.projects[pid]
    keys = ("status", "started", "resource_complete", "structurally_complete", "validated_complete", "delivered_resources", "correct")
    return copy.deepcopy(({k: p.get(k) for k in keys}, [dict(s) for s in p.get("build_steps", [])]))


def _bricks(cm, pid):
    return int(cm.projects[pid]["delivered_resources"].get("bricks", 0) or 0)


def _stub_pilot(monkeypatch, sim, chooser):
    """Replace the Pilot with a scripted chooser(tick) -> BrainDecision for the first agent; the others wait."""
    target = sim.agents[0]
    picks = []

    def pilot(self, sim_state, context, trigger_reason, runtime=None):
        if self is not target:
            return BrainDecision(selected_action=A.WAIT, confidence=0.9)
        decision = chooser(len(picks))
        picks.append((decision.selected_action, decision.target_id))
        return decision

    monkeypatch.setattr(Agent, "_select_decision_through_pilot_adapter", pilot)
    return picks


def _transport(pid):
    return BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id=pid, assumptions=["duration_s=30"], confidence=0.9)


def _two_connectors(sim):
    cm = sim.environment.construction
    a, _ = cm.create_project("site_b", "water_connector")
    b, _ = cm.create_project("site_b", "food_connector")
    return cm, a, b


@pytest.mark.parametrize("chosen", ["first", "second"])
def test_pilot_selected_transport_delivers_only_to_the_chosen_project(sim, monkeypatch, chosen):
    cm, a, b = _two_connectors(sim)
    pid, other = (a, b) if chosen == "first" else (b, a)
    picks = _stub_pilot(monkeypatch, sim, lambda i: _transport(pid))
    for _ in range(400):
        sim.update(1.0)
        if cm.projects[pid]["resource_complete"]:
            break
    assert _bricks(cm, pid) > 0
    assert _bricks(cm, other) == 0  # a second unfinished project at the same site never receives them
    assert {t for _, t in picks} == {pid}


def _blocked_world(sim, family):
    """X is the Pilot's (blocked) choice; Y at the same site is fully actionable for the same family."""
    cm = sim.environment.construction
    x, _ = cm.create_project("site_b", "water_connector")
    y, _ = cm.create_project("site_b", "food_connector")
    if family == A.TRANSPORT_RESOURCES:
        for _ in range(int(cm.projects[x]["required_resources"]["bricks"])):
            cm.deliver_resource(x, "bricks", 1)  # X needs no more material: transport to it is illegal; Y still needs bricks
    elif family in (A.CONTINUE_CONSTRUCTION, A.START_CONSTRUCTION):
        for _ in range(int(cm.projects[y]["required_resources"]["bricks"])):
            cm.deliver_resource(y, "bricks", 1)  # Y could be built; X has no material
    else:  # VALIDATE / REPAIR: Y is structurally complete and ready; X has nothing built
        for pid in (y,):
            for _ in range(int(cm.projects[pid]["required_resources"]["bricks"])):
                cm.deliver_resource(pid, "bricks", 1)
            for step in cm.projects[pid]["build_steps"]:
                step["completed"] = True
            cm.projects[pid].update({"structurally_complete": True, "started": True})
    return cm, x, y


@pytest.mark.parametrize("family", [A.TRANSPORT_RESOURCES, A.CONTINUE_CONSTRUCTION, A.VALIDATE_CONSTRUCTION, A.REPAIR_OR_CORRECT_CONSTRUCTION])
def test_a_blocked_project_specific_choice_is_never_redirected_to_another_project(sim, monkeypatch, family):
    cm, x, y = _blocked_world(sim, family)
    y_before = _physical(cm, y)
    x_before = _physical(cm, x)
    decision = BrainDecision(selected_action=family, target_id=x, assumptions=["duration_s=30"], confidence=0.9)
    _stub_pilot(monkeypatch, sim, lambda i: decision)
    events = []
    sim.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
    for _ in range(120):
        sim.update(1.0)
    for name, payload in events:
        if name in {"construction_build_episode", "construction_transport_pickup", "construction_validation_episode"}:
            assert payload.get("project_id") in (x, None), (name, payload)
    assert _physical(cm, y) == y_before  # the actionable alternative was not worked on, delivered to, validated or repaired\n    if family != A.TRANSPORT_RESOURCES:\n        assert _physical(cm, x) == x_before  # nor did the blocked choice get anything


def test_pivots_hand_back_the_pilots_wait_even_when_a_legal_project_action_exists(sim):
    cm, x, y = _blocked_world(sim, A.CONTINUE_CONSTRUCTION)
    agent = sim.agents[0]
    agent.metacognitive_state["execution_window_origin"] = "productive_action"
    agent.metacognitive_state["execution_window_until"] = 1e9
    agent.metacognitive_state["execution_window_project_id"] = y
    agent._start_execution_lock(sim.environment, y, sim_state=sim, reason="test")
    agent._force_recovery_pivot(sim, to_action=A.TRANSPORT_RESOURCES.value, reason="test", ttl_s=60.0)
    assert not agent.progress_tracker.get("forced_pivot")  # recovery may not force project taskwork
    for _ in range(4):
        for original in (BrainDecision(selected_action=A.WAIT, confidence=0.8), BrainDecision(selected_action=A.REASSESS_PLAN, confidence=0.8)):
            out = agent._apply_policy_pivots(original, sim.environment, sim_state=sim, pivot_origin="test")
            assert out.selected_action not in TASKWORK, out.selected_action


def test_metacognition_exposes_legal_alternatives_read_only_and_does_not_apply_them(sim):
    cm, x, y = _blocked_world(sim, A.CONTINUE_CONSTRUCTION)
    agent = sim.agents[0]
    context = sim.brain_context_builder.build(sim, agent)
    before = _state(cm)
    alternatives = agent._metacognitive_project_alternatives(sim.environment, sim_state=sim, context=context)
    assert alternatives and all(a.target_id in cm.projects for a in alternatives)  # exact project ids, legal now
    assert _state(cm) == before  # candidate generation mutates no construction state
    events = []
    sim.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
    wait = BrainDecision(selected_action=A.WAIT, confidence=0.8)
    out = agent._apply_metacognitive_regulation(wait, sim.environment, sim_state=sim, context=context)
    assert out is wait  # reported, not applied
    assert {i["project_id"] for i in agent.metacognitive_state["legal_project_alternatives"]} == {a.target_id for a in alternatives}
    reported = [p for n, p in events if n == "metacognitive_project_alternatives_reported"]
    assert reported and reported[-1]["applied"] is False
    assert _state(cm) == before


def test_active_construction_and_site_aliases_never_resolve_ambiguously(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    for pid in list(cm.projects):
        cm.projects[pid]["status"] = "complete"
    only, _ = cm.create_project("site_b", "house")
    legacy = BrainDecision(selected_action=A.CONTINUE_CONSTRUCTION, target_id="active_construction", confidence=0.8)
    assert sim.action_gate.evaluate(agent=agent, decision=legacy, sim_state=sim).blockers == []  # sole unfinished project: unambiguous
    second, _ = cm.create_project("site_b", "greenhouse")
    for target in ("active_construction", "", next(iter(cm.PROJECT_TO_SITE))):
        for family in (A.CONTINUE_CONSTRUCTION, A.TRANSPORT_RESOURCES, A.VALIDATE_CONSTRUCTION, A.REPAIR_OR_CORRECT_CONSTRUCTION, A.START_CONSTRUCTION):
            decision = BrainDecision(selected_action=family, target_id=target or None, confidence=0.8)
            gate = sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)
            site = cm.PROJECT_TO_SITE.get(target)
            ambiguous = site is None or site == "site_b"
            if ambiguous:
                assert not gate.legal and gate.blockers[0] in {"ambiguous_project_target", "project_target_required"}, (target, family, gate.blockers)
                assert set(gate.metadata["legal_candidate_project_ids"]) >= {only, second}  # information, not a choice
            action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
            assert action.get("project_id") not in {only, second} or not ambiguous
    exact = BrainDecision(selected_action=A.CONTINUE_CONSTRUCTION, target_id=second, confidence=0.8)
    assert "ambiguous_project_target" not in sim.action_gate.evaluate(agent=agent, decision=exact, sim_state=sim).blockers


def test_every_world_changing_project_event_traces_to_a_pilot_selected_project(sim, monkeypatch):
    cm, a, b = _two_connectors(sim)
    third, _ = cm.create_project("site_b", "house")

    def chooser(i):
        if not cm.projects[a]["resource_complete"]:
            return _transport(a)
        if not cm.projects[b]["resource_complete"]:
            return _transport(b)
        if not cm.projects[a]["structurally_complete"]:
            return BrainDecision(selected_action=A.CONTINUE_CONSTRUCTION, target_id=a, action_args=design_args_for(cm, a), confidence=0.9)  # includes the Pilot's specification choice
        return BrainDecision(selected_action=A.WAIT, confidence=0.9)

    picks = _stub_pilot(monkeypatch, sim, chooser)
    events = []
    sim.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
    for _ in range(700):
        sim.update(1.0)
        if cm.projects[a]["structurally_complete"] and cm.projects[b]["resource_complete"]:
            break
    chosen_transport = {t for act, t in picks if act == A.TRANSPORT_RESOURCES}
    chosen_build = {t for act, t in picks if act in {A.START_CONSTRUCTION, A.CONTINUE_CONSTRUCTION, A.VALIDATE_CONSTRUCTION, A.REPAIR_OR_CORRECT_CONSTRUCTION}}
    delivered = {pid for pid in cm.projects if _bricks(cm, pid) > 0}
    assert delivered and delivered <= chosen_transport  # every material delivery has a matching Pilot target
    assert _bricks(cm, third) == 0
    episodes = {p.get("project_id") for n, p in events if n == "construction_build_episode"}
    assert episodes and episodes <= chosen_build  # every build episode was on a project the Pilot chose
    assert not [n for n, _ in events if n == "recovery_pivot_applied" and _.get("forced_action") in {x.value for x in TASKWORK}]