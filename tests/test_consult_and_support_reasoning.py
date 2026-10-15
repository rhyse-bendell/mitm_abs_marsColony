"""Consultation usefulness (state-sensitive) and knowledge-grounded support reasoning for the Procedural Baseline."""

import copy
import random
import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.support_reasoning import annotate_affordances, derive_support_subgoals, load_support_reasoning
from modules.support_accounting import load_support_model
from modules.simulation import SimulationState

CONNECT = A.CONNECT_STRUCTURES.value


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="consult_support", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _blank(agent):
    """An agent that holds no DIK at all: knowledge must then be acquired explicitly."""
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


def _built(cm, structure_type):
    pid, _ = cm.create_project("site_b", structure_type=structure_type, author="system")
    cm.projects[pid].update({"structurally_complete": True, "status": "complete", "validated_complete": True, "correct": True})
    return pid


def _ctx(sim, agent):
    return sim.brain_context_builder.build(sim, agent)


def _subgoals(sim, agent):
    return {s["subgoal_id"]: s for s in _ctx(sim, agent).individual_cognitive_state["support_reasoning"]["subgoals"]}


def _world_state(cm):
    return copy.deepcopy((cm.projects, cm.connectors, cm._project_counters))


GREENHOUSE_WATER = ("I_BOT_GREENHOUSE_DEPENDENCIES",)
MEANS = ("I_ENG_CONNECTOR_RULES",)


# ---------------------------------------------------------------- consultation
def _artifact(sim, aid="whiteboard:plan:1", summary="plan v1", author="Botanist"):
    return sim.team_knowledge_manager.externalize_artifact(aid, "whiteboard_plan", summary, {"rules": ["r1"]}, author, 1.0, knowledge_summary=["r1"])


def test_unseen_relevant_artifact_is_useful_to_consult(sim):
    agent = sim.agents[0]
    sim.team_knowledge_manager.artifacts.clear()
    _artifact(sim)
    a = agent._consult_assessment(sim.environment, sim_state=sim)
    assert a["useful"] and "unseen" in a["reasons"] and a["artifact_id"] == "whiteboard:plan:1"
    consult = [x for x in _ctx(sim, agent).action_affordances if x["action_type"] == A.CONSULT_TEAM_ARTIFACT.value][0]
    assert consult["consult_useful"] and not consult["consult_redundant"]


def test_revised_artifact_can_be_reconsulted_but_an_unchanged_one_becomes_redundant(sim):
    agent = sim.agents[0]
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim)
    agent._record_artifact_consult(art.artifact_id, sim.environment, sim_state=sim, dik_gained=0, useful=True)
    again = agent._consult_assessment(sim.environment, sim_state=sim)
    assert not again["useful"] and again["suppression_reason"] == "same_version_no_new_context"
    consult = [x for x in _ctx(sim, agent).action_affordances if x["action_type"] == A.CONSULT_TEAM_ARTIFACT.value][0]
    assert consult["consult_redundant"] and consult["utility"] < 0.1
    art.summary = "plan v2"  # revised since the last read
    revised = agent._consult_assessment(sim.environment, sim_state=sim)
    assert revised["useful"] and revised["artifact_changed_since_consult"]


def test_new_goal_context_makes_reconsultation_useful_again(sim):
    agent = sim.agents[0]
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim)
    agent.goal = "G_ONE"
    agent._record_artifact_consult(art.artifact_id, sim.environment, sim_state=sim)
    assert not agent._consult_assessment(sim.environment, sim_state=sim)["useful"]
    agent.goal = "G_TWO"
    changed = agent._consult_assessment(sim.environment, sim_state=sim)
    assert changed["useful"] and changed["goal_relevant"]


def test_artifact_carrying_unheld_dik_is_useful_until_acquired(sim):
    author, reader = sim.agents[2], _blank(sim.agents[0])
    _blank(author)
    _give(sim, author, *GREENHOUSE_WATER)
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim, author=author.name)
    author._attach_carried_information(art, sim)
    assert [i.id for i in art.carried_information] == ["I_BOT_GREENHOUSE_DEPENDENCIES"]
    assessment = reader._consult_assessment(sim.environment, sim_state=sim)
    assert "carries_unheld_dik" in assessment["reasons"]
    assert reader._adopt_artifact_information(art, sim) == 1
    reader._record_artifact_consult(art.artifact_id, sim.environment, sim_state=sim, dik_gained=1, useful=True)
    assert not reader._consult_assessment(sim.environment, sim_state=sim)["useful"]  # now held, nothing left to gain


def test_assessment_is_read_only(sim):
    agent = sim.agents[0]
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim)
    before = (copy.deepcopy(agent.artifact_consult_memory), copy.deepcopy(agent.artifact_coordination_state), art.uptake_count, list(art.consulted_by), _world_state(sim.environment.construction))
    for _ in range(3):
        agent._consult_assessment(sim.environment, sim_state=sim)
        _ctx(sim, agent)
    after = (agent.artifact_consult_memory, agent.artifact_coordination_state, art.uptake_count, list(art.consulted_by), _world_state(sim.environment.construction))
    assert before == after


def _scores(sim, agent, **profile):
    provider = sim.brain_provider
    context = _ctx(sim, agent)
    features = provider._extract_features(context)
    mech = {"goal_alignment": 0.5, "artifact_consultation_tendency": 0.5, "help_tendency": 0.5, **profile}
    affordances = sorted(context.action_affordances, key=lambda a: float(a.get("utility", 0.0)), reverse=True)
    scores = provider._score_actions_for_mode(sorted_affordances=affordances, selected_mode="INTEGRATE_DIK", features=features, mechanism_profile=mech)
    return {a["action_type"] + ":" + str(a.get("target_id")): scores[f"{i}:{a['action_type']}"] for i, a in enumerate(affordances)}


def test_consultation_tendency_raises_useful_consults_but_cannot_rescue_a_redundant_one(sim):
    agent = sim.agents[0]
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim)
    key = f"{A.CONSULT_TEAM_ARTIFACT.value}:whiteboard"
    low, high = _scores(sim, agent, artifact_consultation_tendency=0.1)[key], _scores(sim, agent, artifact_consultation_tendency=0.95)[key]
    assert high > low  # useful consultation is more likely when the tendency is high
    agent._record_artifact_consult(art.artifact_id, sim.environment, sim_state=sim)
    low_r, high_r = _scores(sim, agent, artifact_consultation_tendency=0.1)[key], _scores(sim, agent, artifact_consultation_tendency=0.95)[key]
    assert high_r == pytest.approx(low_r)  # a redundant consult gets no help from the tendency
    assert high_r < high - 2.5  # and is strongly disfavoured, so it cannot dominate indefinitely


def test_unchanged_artifact_with_no_dik_gain_does_not_dominate_selection(sim):
    agent = sim.agents[0]
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim)
    agent._record_artifact_consult(art.artifact_id, sim.environment, sim_state=sim)
    picks = []
    for i in range(30):
        agent._rng = random.Random(i)
        picks.append(agent._select_decision_through_pilot_adapter(sim, _ctx(sim, agent), "test").selected_action)
    assert picks.count(A.CONSULT_TEAM_ARTIFACT) <= len(picks) // 3, picks.count(A.CONSULT_TEAM_ARTIFACT)


# ----------------------------------------------------- knowledge authority
def test_agent_with_greenhouse_water_knowledge_recognises_the_deficit(sim):
    cm = sim.environment.construction
    greenhouse = _built(cm, "greenhouse")
    agent = _give(sim, _blank(sim.agents[1]), *GREENHOUSE_WATER, *MEANS)
    sg = _subgoals(sim, agent)[f"{greenhouse}:water"]
    assert sg["status"] == "unmet" and sg["requirement_basis"] == ["I_BOT_GREENHOUSE_DEPENDENCIES"] and sg["means_known"]
    kinds = {(o["kind"], o.get("structure_type")) for o in sg["options"]}
    assert ("start", "water_generator") in kinds and ("start", "water_connector") in kinds  # nothing exists yet: both are options, in no order


def test_agent_without_the_knowledge_does_not_infer_the_private_rule_from_world_truth(sim):
    cm = sim.environment.construction
    _built(cm, "greenhouse")
    agent = _give(sim, _blank(sim.agents[0]), "I_ARCH_HOUSING_GEOMETRY" if "I_ARCH_HOUSING_GEOMETRY" in sim.task_model.dik_elements else "I_ENG_CONNECTOR_RULES")
    ctx = _ctx(sim, agent)
    assert not [s for s in ctx.individual_cognitive_state["support_reasoning"]["subgoals"] if s["consumer_type"] == "greenhouse"]
    assert not [a for a in ctx.action_affordances if a.get("addresses_support_subgoals")]


def test_knowing_the_requirement_without_the_means_recognises_the_deficit_but_offers_no_remedy(sim):
    cm = sim.environment.construction
    greenhouse = _built(cm, "greenhouse")
    agent = _give(sim, _blank(sim.agents[2]), *GREENHOUSE_WATER)
    sg = _subgoals(sim, agent)[f"{greenhouse}:water"]
    assert not sg["means_known"] and sg["options"] == []
    assert not [a for a in _ctx(sim, agent).action_affordances if a.get("addresses_support_subgoals")]


def test_communicated_rule_enables_subsequent_support_reasoning(sim):
    cm = sim.environment.construction
    greenhouse = _built(cm, "greenhouse")
    sender = _give(sim, _blank(sim.agents[2]), *GREENHOUSE_WATER, *MEANS)
    receiver = _blank(sim.agents[0])
    assert f"{greenhouse}:water" not in _subgoals(sim, receiver)
    infos = [i for i in sender.mental_model["information"]]
    receiver.receive_message({"type": "TIP", "sender": sender.name, "content": infos}, from_agent=sender.name, sim_state=sim)
    assert f"{greenhouse}:water" in _subgoals(sim, receiver)


def test_artifact_mediated_acquisition_enables_the_same_reasoning(sim):
    cm = sim.environment.construction
    greenhouse = _built(cm, "greenhouse")
    author = _give(sim, _blank(sim.agents[2]), *GREENHOUSE_WATER, *MEANS)
    reader = _blank(sim.agents[0])
    sim.team_knowledge_manager.artifacts.clear()
    art = _artifact(sim, author=author.name)
    author._attach_carried_information(art, sim)
    assert f"{greenhouse}:water" not in _subgoals(sim, reader)
    reader._adopt_artifact_information(art, sim)
    assert f"{greenhouse}:water" in _subgoals(sim, reader)
    assert reader.mental_model["information"] and any(i.acquired_by[reader.name]["mode"] == "artifact" for i in reader.mental_model["information"])


# ---------------------------------------------------------- support policy
def _annotated(sim, agent):
    return [a for a in _ctx(sim, agent).action_affordances if a.get("addresses_support_subgoals")]


def test_known_unmet_requirement_raises_missing_infrastructure_starts_above_their_unannotated_selves(sim):
    cm = sim.environment.construction
    greenhouse = _built(cm, "greenhouse")
    agent = _give(sim, _blank(sim.agents[1]), *GREENHOUSE_WATER, *MEANS)
    starts = [a for a in _annotated(sim, agent) if a["action_type"] == A.START_CONSTRUCTION.value]
    assert {a["action_args"]["structure_type"] for a in starts} == {"water_generator", "water_connector"}
    assert all(a["addresses_support_subgoals"] == [f"{greenhouse}:water"] for a in starts)
    blank = _blank(sim.agents[0])
    plain = {a["action_args"]["structure_type"] for a in _ctx(sim, blank).action_affordances if a["action_type"] == A.START_CONSTRUCTION.value and a.get("start_mode") == "new_project"}
    assert {"water_generator", "water_connector"} <= plain  # both are legal for everyone; knowledge only grounds the reason
    scored = _scores(sim, agent)
    start_key = [k for k in scored if k.startswith(A.START_CONSTRUCTION.value)]
    assert start_key  # scoring runs over the same affordances; annotated ones carry the subgoal term
    ctx = _ctx(sim, agent)
    provider = sim.brain_provider
    feats = provider._extract_features(ctx)
    mech = {"goal_alignment": 0.5, "help_tendency": 0.5, "artifact_consultation_tendency": 0.5}
    aff = [a for a in ctx.action_affordances if a.get("addresses_support_subgoals")]
    stripped = [{k: v for k, v in a.items() if k != "addresses_support_subgoals"} for a in aff]
    with_term = provider._score_actions_for_mode(sorted_affordances=aff, selected_mode="CONSTRUCT", features=feats, mechanism_profile=mech)
    without = provider._score_actions_for_mode(sorted_affordances=stripped, selected_mode="CONSTRUCT", features=feats, mechanism_profile=mech)
    assert all(with_term[k] > without[k] + 0.9 for k in with_term)
    lo = provider._score_actions_for_mode(sorted_affordances=aff, selected_mode="CONSTRUCT", features=feats, mechanism_profile={**mech, "goal_alignment": 0.1})
    assert all(with_term[k] > lo[k] for k in with_term)  # goal alignment, an existing mechanism, scales the term


def test_built_generator_connector_and_greenhouse_lead_to_an_explicit_connect_that_survives_unchanged(sim):
    cm = sim.environment.construction
    greenhouse, generator, connector = _built(cm, "greenhouse"), _built(cm, "water_generator"), _built(cm, "water_connector")
    agent = _give(sim, _blank(sim.agents[1]), *GREENHOUSE_WATER, *MEANS)
    connects = [a for a in _annotated(sim, agent) if a["action_type"] == CONNECT]
    assert len(connects) == 1
    args = connects[0]["action_args"]
    assert {args["endpoint_a_id"], args["endpoint_b_id"]} == {greenhouse, generator} and args["connector_project_id"] == connector
    decision = BrainDecision(selected_action=A.CONNECT_STRUCTURES, action_args=dict(args), confidence=0.9)
    before = _world_state(cm)
    pivoted = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="test")
    assert (pivoted.selected_action, dict(pivoted.action_args)) == (A.CONNECT_STRUCTURES, dict(args))  # middleware changed nothing
    assert _world_state(cm) == before  # no hidden construction or connection before the committed action
    action = agent._translate_brain_decision_to_legacy_action(pivoted, sim.environment, sim_state=sim)[0]
    assert action["action_args"] == args
    assert agent._execute_connect_structures(action, sim)
    link = cm.connectors[-1]
    assert (link["connector_project_id"], {link["from_project_id"], link["to_project_id"]}) == (connector, {greenhouse, generator})
    assert greenhouse not in _subgoals(sim, agent)  # the deficit is resolved; reasoning follows state


def test_connect_is_selected_by_the_baseline_for_a_known_deficit_but_not_without_it(sim):
    cm = sim.environment.construction
    _built(cm, "greenhouse"), _built(cm, "water_generator"), _built(cm, "water_connector")
    knowing = _give(sim, _blank(sim.agents[1]), *GREENHOUSE_WATER, *MEANS)
    ignorant = _blank(sim.agents[0])
    for agent in (knowing, ignorant):
        agent.control_state['mode'] = 'COORDINATE'  # a mode in which no method step already owns the choice; same for both agents
    def rate(agent):
        n = 0
        for i in range(40):
            agent._rng = random.Random(i)
            n += agent._select_decision_through_pilot_adapter(sim, _ctx(sim, agent), "test").selected_action == A.CONNECT_STRUCTURES
        return n
    assert rate(knowing) > rate(ignorant)
    assert rate(knowing) >= 5


def test_house_food_deficit_leads_to_greenhouse_and_food_connector_reasoning(sim):
    cm = sim.environment.construction
    house = _built(cm, "house")
    agent = _give(sim, _blank(sim.agents[0]), "I_ARCH_HOUSING_DEPENDENCIES", *MEANS)
    goals = _subgoals(sim, agent)
    assert {f"{house}:food", f"{house}:water"} <= set(goals)
    food_starts = {o["structure_type"] for o in goals[f"{house}:food"]["options"] if o["kind"] == "start"}
    assert food_starts == {"food_connector", "greenhouse"}
    water_starts = {o["structure_type"] for o in goals[f"{house}:water"]["options"] if o["kind"] == "start"}
    assert water_starts == {"water_connector", "water_generator"}


def test_options_follow_world_state_not_a_fixed_build_order(sim):
    cm = sim.environment.construction
    greenhouse = _built(cm, "greenhouse")
    agent = _give(sim, _blank(sim.agents[1]), *GREENHOUSE_WATER, *MEANS)
    empty = {(o["kind"], o.get("structure_type") or o.get("project_id")) for o in _subgoals(sim, agent)[f"{greenhouse}:water"]["options"]}
    generator = _built(cm, "water_generator")
    with_provider = {(o["kind"], o.get("structure_type") or o.get("project_id")) for o in _subgoals(sim, agent)[f"{greenhouse}:water"]["options"]}
    assert ("start", "water_generator") in empty and ("start", "water_generator") not in with_provider  # provider exists: not re-offered
    assert ("start", "water_connector") in with_provider  # but the connector is still the way forward; no step is forced first
    connector = _built(cm, "water_connector")
    last = {o["kind"] for o in _subgoals(sim, agent)[f"{greenhouse}:water"]["options"]}
    assert last == {"connect"}


def test_generator_capacity_stays_simulator_owned(sim):
    cm = sim.environment.construction
    cm.sites["site_b"].capacity = 30
    generator, connector = _built(cm, "water_generator"), _built(cm, "water_connector")
    served = [_built(cm, "greenhouse"), _built(cm, "greenhouse")]
    for s in served:
        assert cm.attach_connector(connector, from_project_id=s, to_project_id=generator, actor="t", sim_time=0.0)[0]
    third = _built(cm, "greenhouse")
    knows_limit = _give(sim, _blank(sim.agents[1]), *GREENHOUSE_WATER, *MEANS, "I_ENG_DISTRIBUTION_TOPOLOGY")
    unaware = _give(sim, _blank(sim.agents[2]), *GREENHOUSE_WATER, *MEANS)
    assert not [o for o in _subgoals(sim, knows_limit)[f"{third}:water"]["options"] if o["kind"] == "connect"]
    assert [o for o in _subgoals(sim, unaware)[f"{third}:water"]["options"] if o["kind"] == "connect"]
    # The simulator, not the Pilot, owns legality: the over-connection is still legal when explicitly requested.
    assert cm.check_connection(connector, third, generator)[0]