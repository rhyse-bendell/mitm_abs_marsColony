"""Knowledge externalization is its own action (claim / evidence / uncertainty / design note); it is not a team plan."""

import tempfile

import pytest

from modules.action_gate import AgentActionGate
from modules.action_schema import BrainDecision, ExecutableActionType as A, validate_action_args
from modules.knowledge import Information
from modules.simulation import SimulationState

EVID = "I_ARCH_HOUSING_DEPENDENCIES"  # a task-declared plan-relevant information element
RULE = "K_HOUSE_VALIDITY_REQUIREMENTS"


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="extern", seed=1)
        state.events = []
        state.logger.register_event_listener(lambda e: state.events.append((e["event_type"], e["payload_data"])))
        try:
            yield state
        finally:
            state.stop()


def _ev(sim, name):
    return [p for n, p in sim.events if n == name]


def _at_whiteboard(sim, agent):
    agent.position = tuple(sim.environment.get_interaction_target_position("whiteboard", from_position=agent.position))


def _run(sim, agent, decision):
    gate = AgentActionGate().evaluate(agent=agent, decision=decision, sim_state=sim)
    if not gate.legal:
        return gate
    action = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
    _at_whiteboard(sim, agent)
    agent.active_actions = [dict(action, progress=0.0)]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 0.1)
    return gate


def _knowledge(kind, element_id, project_id=None):
    args = {"kind": kind, "element_id": element_id, **({"project_id": project_id} if project_id else {})}
    return BrainDecision(selected_action=A.EXTERNALIZE_KNOWLEDGE, target_id="whiteboard", action_args=args, confidence=0.6)


def _consult():
    return BrainDecision(selected_action=A.CONSULT_TEAM_ARTIFACT, target_id="whiteboard", confidence=0.6)


def test_action_contract_validates_kind_and_required_reference():
    assert validate_action_args(A.EXTERNALIZE_KNOWLEDGE, {"kind": "evidence", "element_id": EVID}) == []
    assert "unknown_externalization_kind:plan" in validate_action_args(A.EXTERNALIZE_KNOWLEDGE, {"kind": "plan", "element_id": EVID})
    assert "missing_action_arg:element_id" in validate_action_args(A.EXTERNALIZE_KNOWLEDGE, {"kind": "claim"})


def test_externalization_occurs_without_planning(sim):
    author = sim.agents[0]
    author.mental_model["information"].add(Information(EVID, "housing dependencies"))
    _run(sim, author, _knowledge("evidence", EVID))
    ext = _ev(sim, "knowledge_externalized")
    assert ext and ext[0]["kind"] == "evidence" and ext[0]["element_id"] == EVID
    assert not _ev(sim, "team_plan_proposed") and author.experimental_metrics["team_plan_proposals"] == 0
    assert author.experimental_metrics["externalization_episodes"] == 1 and author.experimental_metrics["knowledge_externalizations"] == 1
    art = sim.team_knowledge_manager.artifacts["knowledge_workspace"]
    assert art.artifact_type == "knowledge_note" and [i.id for i in art.carried_information] == [EVID]
    assert not [a for a in sim.team_knowledge_manager.artifacts.values() if a.artifact_type == "team_plan"]  # no plan was created or revised


def test_planning_occurs_without_transferring_new_knowledge(sim):
    author = sim.agents[0]
    assert not [i for i in author.mental_model["information"]]
    _run(sim, author, BrainDecision(selected_action=A.EXTERNALIZE_PLAN, target_id="whiteboard", confidence=0.6))
    assert author.experimental_metrics["team_plan_proposals"] == 1 and _ev(sim, "team_plan_proposed")
    assert author.experimental_metrics["externalization_episodes"] == 0 and not _ev(sim, "knowledge_externalized")  # a plan with nothing carried is not a knowledge externalization


def test_plan_that_carries_held_information_is_both_and_counted_as_distinct_events(sim):
    author = sim.agents[0]
    author.mental_model["information"].add(Information(EVID, "housing dependencies"))
    _run(sim, author, BrainDecision(selected_action=A.EXTERNALIZE_PLAN, target_id="whiteboard", confidence=0.6))
    assert author.experimental_metrics["team_plan_proposals"] == 1
    carried = [p for p in _ev(sim, "knowledge_externalized") if p["kind"] == "plan_carried_information"]
    assert carried and carried[0]["element_ids"] == [EVID] and author.experimental_metrics["externalized_dik_elements"] == 1


def test_consultation_is_not_uptake(sim):
    author, informed, naive = sim.agents[0], sim.agents[1], sim.agents[2]
    info = Information(EVID, "housing dependencies")
    author.mental_model["information"].add(info)
    informed.mental_model["information"].add(Information(EVID, "housing dependencies"))
    _run(sim, author, _knowledge("evidence", EVID))
    sim.events.clear()
    _run(sim, informed, _consult())
    assert _ev(sim, "artifact_consulted") and not _ev(sim, "dik_acquired_from_artifact")  # read it, learned nothing new
    _run(sim, naive, _consult())
    got = _ev(sim, "dik_acquired_from_artifact")
    assert got and got[0]["information_id"] == EVID
    assert EVID in {i.id for i in naive.mental_model["information"]}


def test_rule_claim_is_transferred_by_consultation(sim):
    author, reader = sim.agents[0], sim.agents[1]
    author.mental_model["knowledge"].add_rule(RULE, [])
    reader.mental_model["knowledge"].rules.discard(RULE) if hasattr(reader.mental_model["knowledge"].rules, "discard") else None
    _run(sim, author, _knowledge("claim", RULE))
    assert sim.team_knowledge_manager.artifacts["knowledge_workspace"].carried_rules == [RULE]
    _run(sim, reader, _consult())
    assert RULE in {r for r in reader.mental_model["knowledge"].rules}


def test_uncertainty_requires_a_recognized_gap_and_transfers_nothing(sim):
    author = sim.agents[0]
    gate = _run(sim, author, _knowledge("uncertainty", "I_ENG_CONNECTOR_RULES"))
    assert not gate.legal and "uncertainty_reference_not_a_recognized_gap" in gate.blockers
    author.known_gaps.add("information:I_ENG_CONNECTOR_RULES")
    _run(sim, author, _knowledge("uncertainty", "I_ENG_CONNECTOR_RULES"))
    assert author.experimental_metrics["uncertainties_externalized"] == 1 and author.experimental_metrics["externalization_episodes"] == 0
    art = sim.team_knowledge_manager.artifacts["knowledge_workspace"]
    assert not art.carried_information and art.content["entries"][0]["kind"] == "uncertainty"


def test_cannot_externalize_what_is_not_held(sim):
    author = sim.agents[0]
    gate = _run(sim, author, _knowledge("evidence", EVID))
    assert not gate.legal and "externalization_reference_not_held" in gate.blockers
    assert "knowledge_workspace" not in sim.team_knowledge_manager.artifacts


def test_reference_forgotten_between_selection_and_execution_is_blocked(sim):
    author = sim.agents[0]
    info = Information(EVID, "x")
    author.mental_model["information"].add(info)
    decision = _knowledge("evidence", EVID)
    action = author._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]  # legal when selected
    assert action["decision_action"] == "externalize_knowledge"
    author.mental_model["information"].discard(info)  # forgotten while travelling to the whiteboard
    _at_whiteboard(sim, author)
    author.active_actions = [dict(action, progress=0.0)]
    author._apply_externalization_and_construction_effects(sim.environment, sim, 0.1)
    assert _ev(sim, "knowledge_externalization_blocked")[0]["reason"] == "reference_not_held"
    assert "knowledge_workspace" not in sim.team_knowledge_manager.artifacts


def test_second_author_contributes_to_the_existing_shared_artifact(sim):
    a, b = sim.agents[0], sim.agents[1]
    a.mental_model["information"].add(Information(EVID, "x"))
    b.mental_model["knowledge"].add_rule(RULE, [])
    _run(sim, a, _knowledge("evidence", EVID))
    assert not _ev(sim, "shared_artifact_contribution")  # the first entry created it
    _run(sim, b, _knowledge("claim", RULE))
    contribution = _ev(sim, "shared_artifact_contribution")
    assert contribution and set(contribution[0]["contributors"]) == {a.name, b.name}


def test_duplicate_entries_are_suppressed_not_counted(sim):
    a = sim.agents[0]
    a.mental_model["information"].add(Information(EVID, "x"))
    _run(sim, a, _knowledge("evidence", EVID))
    _run(sim, a, _knowledge("evidence", EVID))
    assert a.experimental_metrics["knowledge_externalizations"] == 1
    assert _ev(sim, "knowledge_externalization_blocked")[-1]["reason"] == "already_on_workspace"


def test_design_note_needs_an_existing_project_and_a_held_reference(sim):
    a = sim.agents[0]
    a.mental_model["knowledge"].add_rule(RULE, [])
    gate = _run(sim, a, _knowledge("design_note", RULE, "no_such_project"))
    assert not gate.legal and "design_note_requires_existing_project" in gate.blockers
    pid, _ = sim.environment.construction.create_project("site_b", "house")
    _run(sim, a, _knowledge("design_note", RULE, pid))
    entry = sim.environment.construction.projects[pid]["epistemic_workspace"]["entries"][-1]
    assert entry["entry_type"] == "design_note" and RULE in entry["references"]


def test_candidates_are_grounded_and_exposed_identically_as_affordances(sim):
    a = sim.agents[0]
    assert a.externalization_candidates(sim, sim.environment) == []  # nothing held, nothing recognized: no option
    a.mental_model["information"].add(Information(EVID, "x"))
    a.mental_model["knowledge"].add_rule(RULE, [])
    a.known_gaps.add("information:I_ENG_CONNECTOR_RULES")
    cands = a.externalization_candidates(sim, sim.environment)
    assert {(c["kind"], c["element_id"]) for c in cands} == {("evidence", EVID), ("claim", RULE), ("uncertainty", "I_ENG_CONNECTOR_RULES")}
    offers = [x for x in sim.brain_context_builder.build(sim, a).action_affordances if x["action_type"] == "externalize_knowledge"]
    assert len(offers) == len(cands) and all(o["action_args"]["element_id"] for o in offers)
    assert not any(x.get("externalize_gap") for x in sim.brain_context_builder.build(sim, a).action_affordances if x["action_type"] == "externalize_plan")


def test_a_forced_recovery_pivot_cannot_replace_a_pilots_knowledge_externalization(sim):
    """Audit finding: externalize_knowledge was missing from the Pilot-authority guard, so a forced inspect pivot could replace it."""
    agent = sim.agents[0]
    agent.mental_model["information"].add(Information(EVID, "x"))
    decision = _knowledge("evidence", EVID)
    agent.progress_tracker["forced_pivot"] = A.INSPECT_INFORMATION_SOURCE.value
    agent.progress_tracker["forced_pivot_until"] = sim.time + 100.0
    result = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="audit")
    assert result.selected_action == A.EXTERNALIZE_KNOWLEDGE and dict(result.action_args) == dict(decision.action_args)
    assert A.EXTERNALIZE_KNOWLEDGE in agent._SUBSTANTIVE_PILOT_ACTIONS and A.EXTERNALIZE_KNOWLEDGE in agent._PILOT_STATE_MUTATING_ACTIONS
