"""Observed validation failure -> project-specific repair grounding -> explicit revalidation (no hidden-truth leak)."""

import tempfile

import pytest

from tests.construction_helpers import inject_defect, realize_compliant

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_provider import RuleBrain
from modules.construction import ConstructionManager
from modules.simulation import SimulationState

REPAIR_T = ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION
VALIDATE_T = ExecutableActionType.VALIDATE_CONSTRUCTION
REPAIR = REPAIR_T.value
VALIDATE = VALIDATE_T.value
HOUSE_RULE = "rule:house_enclosed"


def _ready_house(cm, project_id="site_b_house_001"):
    cm.create_project("site_b", structure_type="house", project_id_override=project_id)
    p = cm.projects[project_id]
    p["started"] = True
    p["delivered_resources"]["bricks"] = p["required_resources"]["bricks"]
    p["resource_complete"] = True
    p["structurally_complete"] = True
    p["functional_support_complete"] = True
    p["support_requirements"] = {}
    p["epistemic_workspace"]["entries"] = [{"entry_type": "claim"}, {"entry_type": "evidence"}, {"entry_type": "design_note"}]
    p["staged_resources"] = {"bricks": p["required_resources"]["bricks"]}
    for step in p["build_steps"]:
        step["completed"] = True
    for _step, spec in cm.project_slots(p):  # every slot realized as intended and compliant: a structurally valid house
        good = sorted(spec.compliant)[0]
        cm.realize_specification(project_id, slot_id=spec.slot_id, intended=good, realized=good, slipped=False, fidelity=0.5, actor=None, sim_time=0.0)
    cm.update()
    return project_id


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="observed_repair", seed=3)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _summary(sim, pid):
    return next(s for s in sim.brain_context_builder._summarize_structures(sim.environment) if s["structure_id"] == pid)


def _affordances(sim, agent, kind, pid):
    return [a for a in sim.brain_context_builder.build(sim, agent).action_affordances if a["action_type"] == kind and a.get("target_id") == pid]


DEFECT_SLOT = "airlock_opening"


def _fail_validation(cm, pid):
    """A realized defect (an execution slip on one slot), then a failed validation by a validator who can assess that slot."""
    spec = next(sp for _st, sp in cm.project_slots(cm.projects[pid]) if sp.slot_id == DEFECT_SLOT)
    wrong = next(o for o in spec.options if o not in spec.compliant)
    cm.realize_specification(pid, slot_id=DEFECT_SLOT, intended=sorted(spec.compliant)[0], realized=wrong, slipped=True, fidelity=0.5, actor="X", sim_time=0.5)
    cm.update()
    assessment = cm.assess_validation(pid, {"K_HOUSE_VALIDITY_REQUIREMENTS"})
    cm.mark_validated(pid, is_valid=False, actor="X", sim_time=1.0)
    cm.record_validation_observation(pid, actor="X", sim_time=1.0, passed=False, assessment=assessment)
    cm.update()


def _repair_args(cm, pid, slot=DEFECT_SLOT):
    spec = next(sp for _st, sp in cm.project_slots(cm.projects[pid]) if sp.slot_id == slot)
    return {"design_slot": slot, "design_option": sorted(spec.compliant)[0]}


def _hold_rule(agent, held=True):
    rules = agent.mental_model["knowledge"].rules
    if held and HOUSE_RULE not in rules:
        rules.append(HOUSE_RULE)
    if not held:
        while HOUSE_RULE in rules:
            rules.remove(HOUSE_RULE)


def _decide(kind, pid, args=None):
    return BrainDecision(selected_action=kind, target_id=pid, confidence=0.9, action_args=dict(args or {}))


def _run(sim, agent, decision):
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    agent.position = sim.environment.get_interaction_target_position(decision.target_id, from_position=agent.position)
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)
    return actions


def test_1_hidden_incorrect_not_visible_before_validation(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    inject_defect(cm, pid)
    cm.update()
    assert cm.projects[pid]["status"] == "needs_repair"  # world truth
    s = _summary(sim, pid)
    assert s["project_status"] != "needs_repair" and s["validated_correctness"] is True and s["observed_repair_need"] is False
    assert "correct" not in s and s.get("needs_repair") is False  # the key means OBSERVED failure only


def test_2_complete_but_incorrect_project_is_still_a_validation_candidate(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    inject_defect(cm, pid)
    cm.update()
    aff = _affordances(sim, sim.agents[0], VALIDATE, pid)
    assert aff and aff[0]["validation_candidate"] is True and aff[0]["validation_failed"] is False
    assert RuleBrain._project_validation_pending(_summary(sim, pid))


def test_3_first_failed_validation_yields_observable_evidence(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    s = _summary(sim, pid)
    assert s["validation_failed"] is True and s["observed_repair_need"] is True and s["validated_correctness"] is False
    art = sim.team_knowledge_manager.upsert_construction_artifact(cm.projects[pid], 2.0)
    assert art.content["validation_failed"] is True and art.content["correct"] is False


def test_4_failed_project_stops_dominating_validation_pending(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    assert not RuleBrain._project_validation_pending(_summary(sim, pid))
    aff = _affordances(sim, sim.agents[0], VALIDATE, pid)
    assert not aff or aff[0]["validation_failed"] is True


def test_5_failed_project_gives_repair_pressure(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent)
    aff = _affordances(sim, agent, REPAIR, pid)
    assert aff and aff[0]["observed_repair_need"] is True and aff[0]["repair_ready"] is True
    assert aff[0]["repair_blockers"] == []


def test_6_repair_not_advertised_on_healthy_or_unobserved_project(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    agent = sim.agents[0]
    _hold_rule(agent)
    assert not _affordances(sim, agent, REPAIR, pid)
    inject_defect(cm, pid)  # hidden error, no observed failure
    cm.update()
    assert not _affordances(sim, agent, REPAIR, pid)


def test_7_grounded_pilot_can_select_repair_even_with_low_source_memory(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent)
    agent.source_memory_state = {k: dict(v, memory_confidence=0.05) for k, v in agent.source_memory_state.items()}
    assert agent._epistemic_sufficiency(sim.environment, sim_state=sim)["sufficient_for_validation"] is False
    aff = _affordances(sim, agent, REPAIR, pid)
    assert aff and aff[0]["repair_ready"] is True  # source-memory confidence is not a hard gate
    agent._rng.random = lambda: 0.0
    _run(sim, agent, _decide(REPAIR_T, pid, _repair_args(cm, pid)))
    assert cm.projects[pid]["correct"] is True and cm.projects[pid]["specification"][DEFECT_SLOT]["realized"] == "one_brick_opening"  # the artifact's feature changed


def test_8_ungrounded_pilot_does_not_repair_gap_visible_acquisition_possible(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent, held=False)
    aff = _affordances(sim, agent, REPAIR, pid)
    assert aff and aff[0]["repair_ready"] is False
    expected = cm.projects[pid]["expected_rules"]
    assert aff[0]["repair_blockers"] == sorted("missing_expected_rule:" + r for r in expected)  # names rules, not the hidden error
    kinds = {a["action_type"] for a in sim.brain_context_builder.build(sim, agent).action_affordances}
    assert "inspect_information_source" in kinds or "consult_team_artifact" in kinds
    agent._rng.random = lambda: 0.0
    _run(sim, agent, _decide(REPAIR_T, pid))
    assert cm.projects[pid]["correct"] is False and cm.projects[pid].get("repair_count", 0) == 0
    assert agent.experimental_metrics["repair_blocked"] >= 1
    _hold_rule(agent)  # once the rule is actually held the same repair proceeds
    assert _affordances(sim, agent, REPAIR, pid)[0]["repair_ready"] is True
    _run(sim, agent, _decide(REPAIR_T, pid, _repair_args(cm, pid)))
    assert cm.projects[pid]["correct"] is True


def test_9_repair_is_exact_project_addressed(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent)
    aff = _affordances(sim, agent, REPAIR, pid)
    assert aff and all(a["target_id"] == pid for a in aff)
    agent._rng.random = lambda: 0.0
    actions = _run(sim, agent, _decide(REPAIR_T, pid, _repair_args(cm, pid)))
    assert actions and all(a.get("project_id") in (None, pid) for a in actions)
    assert cm.projects[pid]["repair_count"] == 1


def test_10_middleware_does_not_change_project_or_action(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent, held=False)  # blocked: reported, never rewritten into another action/project
    decision = _decide(REPAIR_T, pid)
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    assert decision.selected_action == REPAIR_T and decision.target_id == pid
    for a in actions:
        assert a.get("project_id") in (None, pid)
        assert "validate" not in str(a.get("type", "")) and "start_construction" not in str(a.get("type", ""))


def test_11_successful_repair_leads_to_explicit_revalidation(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent)
    agent._rng.random = lambda: 0.0
    _run(sim, agent, _decide(REPAIR_T, pid, _repair_args(cm, pid)))
    p = cm.projects[pid]
    assert p["correct"] is True and p["status"] != "complete" and p["awaiting_revalidation"] is True
    assert not ConstructionManager.observed_validation_failure(p)
    assert _affordances(sim, agent, VALIDATE, pid)  # revalidation is offered, never automatic
    _run(sim, agent, _decide(VALIDATE_T, pid))
    assert p["status"] == "complete" and p["last_validation_was_revalidation"] is True


def test_12_repair_is_not_guaranteed_to_succeed(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    _fail_validation(cm, pid)
    agent = sim.agents[0]
    _hold_rule(agent)
    agent._rng.random = lambda: 0.9999  # the application-fidelity roll fails
    agent._trait_value = lambda name, *a, **k: 0.0 if name == "rule_accuracy" else 1.0
    _run(sim, agent, _decide(REPAIR_T, pid, _repair_args(cm, pid)))
    cm.update()
    assert cm.projects[pid]["correct"] is False  # attempted, executed, still wrong: the slip realized a different, non-compliant option
    assert cm.projects[pid]["specification"][DEFECT_SLOT]["slipped"] is True and cm.projects[pid]["specification"][DEFECT_SLOT]["intended"] == "one_brick_opening"
    assert any(n == "construction_execution_slip" and d.get("repair") is True for n, d in sim.captured_events)
    assert agent.experimental_metrics["repair_applications"] == 1 and agent.experimental_metrics["repair_successes"] == 0
    assert _summary(sim, pid)["validation_failed"] is False  # and not visibly so until revalidated


def test_13_state_changing_events_are_pilot_authored():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="authorship", seed=1)
        events = []
        state.logger.register_event_listener(lambda e: events.append(e["event_type"]))
        try:
            for _ in range(400):
                state.update(1.0)
        finally:
            state.stop()
    m = lambda k: sum(a.experimental_metrics[k] for a in state.agents)
    assert events.count("construction_repair_episode") == m("repair_applications")
    assert m("repair_applications") == 0 or m("repair_decision_episodes") > 0  # an application traces to a Pilot repair intent
    assert m("repair_successes") <= m("repair_applications")
    assert events.count("construction_validation_attempted") == m("validation_attempts")


def test_14_no_raw_correct_in_team_artifacts_or_context_before_observation(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    inject_defect(cm, pid)
    cm.update()
    art = sim.team_knowledge_manager.upsert_construction_artifact(cm.projects[pid], 2.0)
    assert art.content["correct"] is True and art.content["status"] != "needs_repair" and art.validation_state != "mismatch"
    packet = sim.brain_context_builder.build(sim, sim.agents[0])
    assert "'needs_repair': True" not in repr(packet.world_snapshot) and "needs_repair" not in repr(packet.action_affordances)
    assert all(s.get("needs_repair") is False for s in sim.brain_context_builder._summarize_structures(sim.environment))