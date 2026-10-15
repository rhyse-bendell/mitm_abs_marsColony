"""Task-defined plan methods ground into typed legal actions; one chosen intent counts as one decision episode."""

import random
import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.agent import Agent
from modules.simulation import SimulationState

CONNECT = A.CONNECT_STRUCTURES.value


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="methods_intents", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _blank(agent):
    for kind in ("data", "information"):
        for item in list(agent.mental_model[kind]):
            agent.mental_model[kind].discard(item)
    agent.mental_model["knowledge"].rules.clear()
    return agent


def _give(sim, agent, *element_ids):
    for eid in element_ids:
        element = sim.task_model.dik_elements[eid]
        obj = agent._create_dik_object_from_element(element, source_id="test")
        agent.mental_model["data" if element.element_type == "data" else "information"].add(obj)
    return agent


def _equip(sim, agent, method_id, *, complete=True):
    """Give the agent exactly the DIK the method requires (information + data, plus rules/knowledge when complete)."""
    method = sim.task_model.plan_methods[method_id]
    _give(sim, agent, *method.required_information, *method.required_data)
    if complete:
        for rule in [*method.required_rules, *method.required_knowledge]:
            agent.mental_model["knowledge"].add_rule(rule, [])
    return agent


def _ctx(sim, agent):
    return sim.brain_context_builder.build(sim, agent)


def _starts(ctx):
    return {a["action_args"]["structure_type"]: a for a in ctx.action_affordances if a["action_type"] == A.START_CONSTRUCTION.value and a.get("start_mode") == "new_project"}


def _served(aff):
    return [m["method_id"] for m in aff.get("serves_task_methods", [])]


def _start_scores(sim, agent):
    ctx = _ctx(sim, agent)
    provider = sim.brain_provider
    feats = provider._extract_features(ctx)
    mech = {"goal_alignment": 0.5, "help_tendency": 0.5, "artifact_consultation_tendency": 0.5}
    affs = sorted(ctx.action_affordances, key=lambda a: float(a.get("utility", 0.0)), reverse=True)
    scores = provider._score_actions_for_mode(sorted_affordances=affs, selected_mode="CONSTRUCT", features=feats, mechanism_profile=mech)
    return {a["action_args"]["structure_type"]: scores[f"{i}:{a['action_type']}"] for i, a in enumerate(affs) if a["action_type"] == A.START_CONSTRUCTION.value and a.get("start_mode") == "new_project"}


METHODS = [("PM_BUILD_VALID_HOUSE", 0, "house"), ("PM_BUILD_VALID_GREENHOUSE", 2, "greenhouse"), ("PM_BUILD_VALID_WATERGEN", 1, "water_generator")]


@pytest.mark.parametrize("method_id,agent_index,structure_type", METHODS)
def test_each_role_method_prefers_its_typed_start(sim, method_id, agent_index, structure_type):
    agent = _equip(sim, _blank(sim.agents[agent_index]), method_id)
    starts = _starts(_ctx(sim, agent))
    assert structure_type in starts
    assert _served(starts[structure_type]) == [method_id]
    assert not [t for t, a in starts.items() if t != structure_type and _served(a)]  # only the method's own instance
    scores = _start_scores(sim, agent)
    assert scores[structure_type] > max(v for t, v in scores.items() if t != structure_type) + 0.5


def test_typed_argument_survives_pilot_to_middleware_to_execution(sim):
    agent = _equip(sim, _blank(sim.agents[0]), "PM_BUILD_VALID_HOUSE")
    aff = _starts(_ctx(sim, agent))["house"]
    decision = BrainDecision(selected_action=A.START_CONSTRUCTION, target_id=aff["target_id"], action_args=dict(aff["action_args"]), confidence=0.9)
    kept = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="test")
    assert (kept.selected_action, dict(kept.action_args)) == (A.START_CONSTRUCTION, {"structure_type": "house"})
    assert sim.action_gate.evaluate(agent=agent, decision=kept, sim_state=sim).legal
    assert not sim.environment.construction.projects or all(p["type"] != "house" for p in sim.environment.construction.projects.values())
    action = agent._translate_brain_decision_to_legacy_action(kept, sim.environment, sim_state=sim)[0]
    assert sim.environment.construction.projects[action["project_id"]]["type"] == "house"  # created only by this committed START


def test_a_method_cannot_select_an_unavailable_typed_affordance(sim):
    agent = _equip(sim, _blank(sim.agents[0]), "PM_BUILD_VALID_HOUSE")
    for site in sim.environment.construction.sites.values():
        site.capacity = 0  # no site can host anything: no typed START is legal
    ctx = _ctx(sim, agent)
    assert not _starts(ctx)
    assert not [a for a in ctx.action_affordances if a.get("serves_task_methods")]
    assert any(m["method_id"] == "PM_BUILD_VALID_HOUSE" and m["weight"] > 0 for m in ctx.individual_cognitive_state["task_methods"])  # competent, but nothing legal to express it with


def test_missing_required_dik_prevents_method_competence_and_acquiring_it_enables_it(sim):
    agent = _blank(sim.agents[0])
    status = {m["method_id"]: m for m in _ctx(sim, agent).individual_cognitive_state["task_methods"]}
    assert status["PM_BUILD_VALID_HOUSE"]["weight"] == 0.0 and status["PM_BUILD_VALID_HOUSE"]["missing"]["information"]
    assert not _served(_starts(_ctx(sim, agent))["house"])
    _equip(sim, agent, "PM_BUILD_VALID_HOUSE", complete=False)
    partial = {m["method_id"]: m for m in _ctx(sim, agent).individual_cognitive_state["task_methods"]}["PM_BUILD_VALID_HOUSE"]
    assert partial["weight"] == pytest.approx(0.35)  # information held, rules/knowledge not (the existing low-trust state)
    _equip(sim, agent, "PM_BUILD_VALID_HOUSE")
    full = {m["method_id"]: m for m in _ctx(sim, agent).individual_cognitive_state["task_methods"]}["PM_BUILD_VALID_HOUSE"]
    assert full["weight"] == 1.0 and _served(_starts(_ctx(sim, agent))["house"]) == ["PM_BUILD_VALID_HOUSE"]


def test_a_method_belongs_to_its_task_defined_roles_not_to_whoever_holds_the_dik(sim):
    engineer = _equip(sim, _blank(sim.agents[1]), "PM_BUILD_VALID_HOUSE")  # holds the Architect's method DIK
    assert "PM_BUILD_VALID_HOUSE" not in {m["method_id"] for m in _ctx(sim, engineer).individual_cognitive_state["task_methods"]}
    assert not _served(_starts(_ctx(sim, engineer))["house"])


def _built(cm, structure_type):
    pid, _ = cm.create_project("site_b", structure_type=structure_type, author="system")
    cm.projects[pid].update({"structurally_complete": True, "status": "complete", "validated_complete": True, "correct": True})
    return pid


def test_connection_method_prefers_the_explicit_connect_support_reasoning_exposes(sim):
    cm = sim.environment.construction
    greenhouse, generator, connector = _built(cm, "greenhouse"), _built(cm, "water_generator"), _built(cm, "water_connector")
    other_house = _built(cm, "house")
    agent = _equip(sim, _blank(sim.agents[2]), "PM_CONNECT_GREENHOUSE_RESOURCES")
    ctx = _ctx(sim, agent)
    connects = [a for a in ctx.action_affordances if a["action_type"] == CONNECT]
    served = [a for a in connects if _served(a)]
    assert len(served) == 1 and _served(served[0]) == ["PM_CONNECT_GREENHOUSE_RESOURCES"]
    args = served[0]["action_args"]
    # exact connector/endpoints come from the support subgoal, never from the method definition
    method = sim.task_model.plan_methods["PM_CONNECT_GREENHOUSE_RESOURCES"]
    assert "connect_structures" in method.candidate_steps[0] and not method.action_bindings
    subgoal = ctx.individual_cognitive_state["support_reasoning"]["subgoals"][0]
    option = [o for o in subgoal["options"] if o["kind"] == "connect"][0]
    assert (args["connector_project_id"], args["endpoint_a_id"], args["endpoint_b_id"]) == (option["connector_project_id"], option["endpoint_a_id"], option["endpoint_b_id"])
    assert {args["endpoint_a_id"], args["endpoint_b_id"]} == {greenhouse, generator}
    assert [a for a in connects if a is not served[0]] == [] or all(not _served(a) for a in connects if a is not served[0])
    # and without the method's DIK the connect is not method-preferred
    bare = _blank(sim.agents[0])
    assert not [a for a in _ctx(sim, bare).action_affordances if _served(a)]


# ------------------------------------------------------------ intent episodes
def _pilot(monkeypatch, sim, chooser, who=0):
    target = sim.agents[who]
    target.control_state['mode'] = 'COORDINATE'  # past the startup grounding guard, which would otherwise override the stub Pilot
    picks = []

    def pilot(self, sim_state, context, trigger_reason, runtime=None):
        if self is not target:
            return BrainDecision(selected_action=A.WAIT, confidence=0.9)
        decision = chooser(len(picks), sim_state)
        picks.append((sim_state.time, decision.selected_action, decision.target_id))
        self._pilot_decision_anchor = decision
        self._track_pilot_selection(sim_state, decision)
        return decision

    monkeypatch.setattr(Agent, "_select_decision_through_pilot_adapter", pilot)
    return target, picks


def _events(sim):
    events = []
    sim.logger.register_event_listener(lambda e: events.append((e["time"], e["event_type"], e["payload_data"])))
    return events


def _plan_artifact(sim):
    sim.team_knowledge_manager.propose_team_plan(
        plan_id="P1", proposed_by="Botanist", sim_time=0.0, goal_ids=["G"], goal_summary="s", trigger_reason="no_active_plan", project_targets=[], assignments_by_role={"Botanist": "x"},
        evidence_refs=[], blocked_reasons=[], success_criteria=[], review_at=50.0, expires_at=500.0, supporters=["Botanist"],
    )


def test_a_consult_requiring_travel_is_one_decision_episode_that_completes(sim, monkeypatch):
    _plan_artifact(sim)
    events = _events(sim)
    agent, picks = _pilot(monkeypatch, sim, lambda i, s: BrainDecision(selected_action=A.CONSULT_TEAM_ARTIFACT, target_id="whiteboard", confidence=0.9))
    for _ in range(80):
        sim.update(1.0)
        if any(n == "artifact_consulted" for _, n, _p in events):
            break
    for _ in range(3):
        sim.update(1.0)  # the action completes the tick after its effect ran
    consulted_at = next(t for t, n, p in events if n == "artifact_consulted" and p.get("agent") == agent.name)
    before = [p for p in picks if p[0] <= consulted_at]
    assert len(before) == 1  # one selection covers the whole walk and the read; movement ticks do not re-select
    started = [p for t, n, p in events if n == "pilot_intent_started" and p["agent"] == agent.name]
    assert started and started[0]["action"] == "consult_team_artifact"
    done = [p for t, n, p in events if n == "pilot_intent_completed" and p["agent"] == agent.name]
    assert done and done[0]["episode_id"] == started[0]["episode_id"] and done[0]["ticks"] >= 3 and done[0]["selections"] == 1


def test_a_blocked_selection_closes_its_episode_and_returns_control_to_the_pilot(sim, monkeypatch):
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", "house")
    events = _events(sim)
    agent, picks = _pilot(monkeypatch, sim, lambda i, s: BrainDecision(selected_action=A.VALIDATE_CONSTRUCTION, target_id=pid, confidence=0.9))
    for _ in range(12):
        sim.update(1.0)
    blocked = [p for t, n, p in events if n == "pilot_intent_blocked" and p["agent"] == agent.name]
    assert blocked and len(picks) >= 2  # blocked -> the Pilot is asked again
    assert agent.pilot_episode_stats["blocked"] >= 1


def test_transport_keeps_its_exact_target_for_the_whole_episode_and_a_state_change_ends_it(sim, monkeypatch):
    cm = sim.environment.construction
    a, _ = cm.create_project("site_b", "water_connector")
    b, _ = cm.create_project("site_b", "food_connector")
    events = _events(sim)
    agent, picks = _pilot(monkeypatch, sim, lambda i, s: BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id=b, assumptions=["duration_s=30"], confidence=0.9))
    seen = set()
    for _ in range(25):
        sim.update(1.0)
        seen |= {x.get("project_id") for x in agent.active_actions if x.get("type") == "transport_resources"}
    assert seen == {b}
    assert len([p for p in picks if p[2] == b]) <= 3  # a handful of decisions, not one per tick
    for _ in range(int(cm.projects[b]["required_resources"]["bricks"]) - int(cm.projects[b]["delivered_resources"]["bricks"])):
        cm.deliver_resource(b, "bricks", 1)  # the world changes underneath the in-flight intent
    for _ in range(40):
        sim.update(1.0)
    assert cm.projects[a]["delivered_resources"]["bricks"] == 0  # never silently redirected to the other project
    assert any(n in {"pilot_intent_blocked", "pilot_intent_completed", "pilot_intent_invalidated"} for _, n, _p in events)


def test_scaffold_pivots_cannot_turn_a_pilot_consult_into_externalization_or_communication(sim):
    agent = sim.agents[0]
    consult = BrainDecision(selected_action=A.CONSULT_TEAM_ARTIFACT, target_id="whiteboard", confidence=0.8)
    agent._pilot_decision_anchor = consult
    for rewritten in (BrainDecision(selected_action=A.EXTERNALIZE_PLAN, confidence=0.8), BrainDecision(selected_action=A.COMMUNICATE, confidence=0.8), BrainDecision(selected_action=A.CONNECT_STRUCTURES, action_args={"connector_project_id": "c", "endpoint_a_id": "a", "endpoint_b_id": "b"}, confidence=0.8)):
        assert agent._enforce_pilot_project_authority(rewritten, sim_state=sim) is consult
    assert agent._enforce_pilot_project_authority(BrainDecision(selected_action=A.REASSESS_PLAN, confidence=0.8), sim_state=sim).selected_action == A.REASSESS_PLAN  # non-mutating scaffolds are unchanged


def _run_trace(seed):
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="det", seed=seed)
        trace = []
        state.logger.register_event_listener(lambda e: trace.append((round(e["time"], 1), e["event_type"], e["payload_data"].get("agent"), e["payload_data"].get("action"))) if e["event_type"].startswith("pilot_intent_") or e["event_type"] == "pilot_selection_via_adapter" else None)
        try:
            for _ in range(70):
                state.update(1.0)
            stats = [dict(a.pilot_episode_stats) for a in state.agents]
        finally:
            state.stop()
    return trace, stats


def test_seeded_runs_with_episode_tracking_are_deterministic():
    random.seed(3)
    first = _run_trace(3)
    random.seed(3)
    second = _run_trace(3)
    assert first == second
    stats = first[1]
    assert all(s["episodes"] <= s["selections"] and s["selections"] > 0 for s in stats)