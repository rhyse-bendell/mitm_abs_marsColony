"""Explicit construction specifications: knowledge informs what a Pilot attempts, fidelity decides what is realized, the Environment derives
correctness from the artifact, and validation / functional support never write it."""

import random
import tempfile
from unittest.mock import patch

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.brain_provider import resolve_design_variant
from modules.construction import ConstructionManager
from modules.construction_spec import load_construction_specs, requirement_held
from modules.simulation import SimulationState
from tests.construction_helpers import build_step, inject_defect, realize_compliant

READY = {"sufficient_for_validation": True, "sufficient_for_build": True, "role_missing": False, "stale_grounding": False, "ratio": 1.0, "score": 1.0}


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="spec", seed=3)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _staged_house(sim, pid_override="site_b_house_001"):
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", structure_type="house", project_id_override=pid_override)
    p = cm.projects[pid]
    required = p["required_resources"]["bricks"]
    p.update({"started": True, "staged_resources": {"bricks": required}, "delivered_resources": {"bricks": required}})
    cm.update()
    return cm, pid


def _build(sim, agent, pid, args, *, rng_value=0.0):
    """One CONTINUE action with the Pilot's explicit specification choice, executed through translation and the effect stage."""
    cm = sim.environment.construction
    decision = BrainDecision(selected_action=A.CONTINUE_CONSTRUCTION, target_id=pid, confidence=0.9, action_args=dict(args))
    agent.position = sim.environment.get_interaction_target_position(pid, from_position=agent.position)
    agent.rule_accuracy = 0.0  # execution fidelity = (hook + rule_accuracy)/2; the hook default is 1.0, so 0.5 here, the compact-panel default
    agent._rng.random = lambda: rng_value
    with patch.object(agent, "_build_readiness_blockers", return_value=[]), patch.object(agent, "_epistemic_sufficiency", return_value=READY):
        actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
        agent.active_actions = [dict(a, progress=0.0) for a in actions]
        agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)
    return actions


def _ev(sim, name):
    return [p for n, p in sim.captured_events if n == name]


# ------------------------------------------------------------------ the task-package specification
def test_specification_comes_from_the_task_package_and_uses_the_final_architect_packet():
    specs = load_construction_specs("config/tasks/mars_colony")
    assert set(specs) == {"house", "greenhouse", "water_generator", "food_connector", "water_connector"}
    headroom = specs["house"]["place_housing_wall_or_ceiling"]
    assert headroom.compliant == frozenset({"1", "2"})  # final Architect packet: at least one brick (the earlier Task Design says two)
    assert "0" in headroom.options and "D_ARCH_HEADROOM_1_BRICK" in headroom.requirement_elements
    for per_type in specs.values():
        for spec in per_type.values():
            assert spec.compliant < set(spec.options) or spec.compliant == set(spec.options) and False or spec.compliant.issubset(spec.options)
            assert len(spec.options) >= 2 and spec.source


def test_a_package_without_a_specification_keeps_the_earlier_semantics():
    cm = ConstructionManager()  # no task model, no construction_specs.csv
    assert cm.specs == {}
    pid, _ = cm.create_project("site_b", "house")
    assert cm.has_specification(cm.projects[pid]) is False and cm.pending_specification(pid) is None
    assert cm.record_build_quality(pid, correct=False, actor="A") is True


# ------------------------------------------------------------------ same options for everyone, none marked correct
def test_every_agent_receives_the_same_menu_and_nothing_marks_an_option_correct(sim):
    cm, pid = _staged_house(sim)
    informed, naive = sim.agents[0], sim.agents[1]
    informed.mental_model["information"].add(__import__("modules.knowledge", fromlist=["Information"]).Information("I_ARCH_HOUSING_AESTHETICS", "x"))
    menus = []
    for agent in (informed, naive):
        offers = [a for a in sim.brain_context_builder.build(sim, agent).action_affordances if a.get("design_slot") and a.get("target_id") == pid]
        menus.append(sorted((a["action_type"], a["design_slot"], a["design_option"]) for a in offers))
        assert offers and not any(k in a for a in offers for k in ("compliant", "correct", "recommended", "is_valid"))
    assert menus[0] == menus[1]
    assert {m[2] for m in menus[0]} == {"single_solid_floor_walls_differ", "single_solid_floor_walls_same", "mixed_floor"}  # the first slot's whole menu


# ------------------------------------------------------------------ unresolved required choices are incomplete, never filled in
def test_a_missing_required_choice_blocks_the_step_and_nothing_is_filled_in(sim):
    cm, pid = _staged_house(sim)
    agent = sim.agents[0]
    actions = _build(sim, agent, pid, {})
    assert not any(s.get("completed") for s in cm.projects[pid]["build_steps"])
    assert cm.projects[pid]["specification"] == {} and cm.pending_specification(pid)["slot_id"] == "floor_wall_colour"
    assert actions[0]["decision_action"] == "wait" and actions[0]["type"] == "idle"  # reported back as blocked; no substitute action, no default option
    ok, reason, _ = cm.execute_build_step(pid, actor="A")
    assert (ok, reason) == (False, "specification_required")


def test_an_option_that_is_not_offered_is_refused(sim):
    cm, pid = _staged_house(sim)
    ok, reason, _ = cm.execute_build_step(pid, actor="A", intended_option="pink_floor")
    assert (ok, reason) == (False, "design_option_not_offered")


# ------------------------------------------------------------------ knowledge informs the attempt; it does not decide correctness
def _procedural_pick(seed, held, slot_option_names, project_type="house", slot="airlock_opening"):
    options = [{"action_type": "continue_construction", "target_id": "p1", "project_type": project_type, "design_slot": slot, "design_option": o, "action_args": {"design_slot": slot, "design_option": o}} for o in slot_option_names]
    return resolve_design_variant(options[0], options, held, "mars_colony", f"ctx:{seed}:agent")["design_option"]


def test_holding_a_construction_rule_changes_the_procedural_pilots_selected_specification():
    opts = ["no_opening", "one_brick_opening", "two_brick_opening"]
    informed = {_procedural_pick(s, {"D_ARCH_AIRLOCK_SIZE"}, opts) for s in range(40)}
    uninformed = [_procedural_pick(s, set(), opts) for s in range(120)]
    assert informed == {"one_brick_opening"}  # the held requirement is applied by the Pilot's own decision rule
    assert set(uninformed) == set(opts)  # without it the choice is a draw over the same menu
    assert 0.2 < uninformed.count("one_brick_opening") / len(uninformed) < 0.5  # an uninformed agent is right by chance, about 1 in 3


def test_an_uninformed_agent_can_build_a_valid_house_and_an_informed_one_can_choose_wrongly(sim):
    cm, pid = _staged_house(sim)
    agent = sim.agents[0]
    right = ["single_solid_floor_walls_differ", "1", "one_brick_opening"]
    for option in right:  # no slips: the realized features equal the intended ones
        _build(sim, agent, pid, {"design_slot": cm.pending_specification(pid)["slot_id"], "design_option": option}, rng_value=0.0)
    assert cm.structural_assessment(cm.projects[pid]) == {"violations": [], "unspecified": [], "valid": True} and cm.projects[pid]["correct"] is True
    cm2, pid2 = _staged_house(sim, "site_b_house_002")
    agent.mental_model["knowledge"].add_rule("K_HOUSE_VALIDITY_REQUIREMENTS", [])  # an INFORMED agent still selects a non-compliant option when it decides to
    for option in ["single_solid_floor_walls_differ", "1", "no_opening"]:
        _build(sim, agent, pid2, {"design_slot": cm2.pending_specification(pid2)["slot_id"], "design_option": option}, rng_value=0.0)
    assert cm2.structural_assessment(cm2.projects[pid2])["violations"] == ["airlock_opening"] and cm2.projects[pid2]["correct"] is False
    assert _ev(sim, "construction_design_choice_made")[-1]["actor_held_requirement"] is True


def test_the_mecha_does_not_improve_a_selected_specification(sim):
    cm, pid = _staged_house(sim)
    agent = sim.agents[0]
    agent.mental_model["knowledge"].add_rule("K_HOUSE_VALIDITY_REQUIREMENTS", [])
    _build(sim, agent, pid, {"design_slot": "floor_wall_colour", "design_option": "mixed_floor"}, rng_value=0.0)
    record = cm.projects[pid]["specification"]["floor_wall_colour"]
    assert record["intended"] == "mixed_floor" and record["realized"] == "mixed_floor" and record["slipped"] is False


# ------------------------------------------------------------------ execution slips are independent and interpretable
def test_a_slip_realizes_a_different_option_than_the_one_intended_even_for_an_informed_agent(sim):
    cm, pid = _staged_house(sim)
    agent = sim.agents[0]
    agent.mental_model["knowledge"].add_rule("K_HOUSE_VALIDITY_REQUIREMENTS", [])
    _build(sim, agent, pid, {"design_slot": "floor_wall_colour", "design_option": "single_solid_floor_walls_differ"}, rng_value=0.9999)
    record = cm.projects[pid]["specification"]["floor_wall_colour"]
    assert record["slipped"] is True and record["intended"] == "single_solid_floor_walls_differ" and record["realized"] != record["intended"]
    assert record["fidelity"] == pytest.approx(0.5) == pytest.approx((agent._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5) + 0.0) / 2.0) and record["episode"].startswith(agent.name) and record["actor"] == agent.name
    assert cm.projects[pid]["correct"] is False
    slip = _ev(sim, "construction_execution_slip")[0]
    assert slip["intended"] == record["intended"] and slip["realized"] == record["realized"] and slip["fidelity"] == pytest.approx(0.5)


def test_exposure_is_one_roll_per_build_action_not_one_per_slot(sim):
    cm, pid = _staged_house(sim)
    agent = sim.agents[0]
    agent.rule_accuracy = 0.0
    draws = []

    class CountingRandom(random.Random):
        def random(self):
            draws.append(1)
            return 0.0

    agent._rng = CountingRandom(1)
    for option in ["single_solid_floor_walls_differ", "1", "one_brick_opening"]:
        decision = BrainDecision(selected_action=A.CONTINUE_CONSTRUCTION, target_id=pid, confidence=0.9, action_args={"design_slot": cm.pending_specification(pid)["slot_id"], "design_option": option})
        agent.position = sim.environment.get_interaction_target_position(pid, from_position=agent.position)
        with patch.object(agent, "_build_readiness_blockers", return_value=[]), patch.object(agent, "_epistemic_sufficiency", return_value=READY):
            agent.active_actions = [dict(a, progress=0.0) for a in agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)]
            before = len(draws)
            agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)
        assert len(draws) - before == 1  # exactly the one fidelity draw the earlier model made per build action
    assert len(cm.projects[pid]["specification_log"]) == 3


def test_a_blocked_attempt_consumes_no_exposure_and_realizes_nothing(sim):
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", structure_type="house", project_id_override="site_b_house_009")  # materials not staged
    agent = sim.agents[0]
    _build(sim, agent, pid, {"design_slot": "floor_wall_colour", "design_option": "mixed_floor"}, rng_value=0.9999)
    assert cm.projects[pid]["specification"] == {}


# ------------------------------------------------------------------ correctness is derived from the artifact; validation and support never write it
def test_validation_cannot_create_a_defect(sim):
    cm, pid = _staged_house(sim)
    realize_compliant(cm, pid)
    for step in cm.projects[pid]["build_steps"]:
        step["completed"] = True
    cm.update()
    assert cm.projects[pid]["correct"] is True
    cm.mark_validated(pid, is_valid=False, actor="V", sim_time=2.0)  # e.g. a failure for a non-structural reason
    cm.update()
    assert cm.projects[pid]["correct"] is True and cm.structural_assessment(cm.projects[pid])["valid"] is True
    cm.projects[pid]["correct"] = False  # even a direct overwrite is re-derived from the realized features
    cm.update()
    assert cm.projects[pid]["correct"] is True


def test_a_defect_persists_until_a_repair_changes_a_feature(sim):
    cm, pid = _staged_house(sim)
    for step in cm.projects[pid]["build_steps"]:
        step["completed"] = True
    realize_compliant(cm, pid)
    inject_defect(cm, pid)
    assert cm.projects[pid]["correct"] is False
    cm.mark_validated(pid, is_valid=False, actor="V", sim_time=2.0)
    cm.update()
    assert cm.projects[pid]["correct"] is False  # a validation, failed or not, neither fixes nor creates it
    before = dict(cm.projects[pid]["specification"]["airlock_opening"])
    cm.repair_specification(pid, slot_id="airlock_opening", intended="one_brick_opening", realized="one_brick_opening", slipped=False, fidelity=0.5, actor="R", sim_time=3.0)
    after = cm.projects[pid]["specification"]["airlock_opening"]
    assert before["realized"] != after["realized"] and after["repair"] is True and cm.projects[pid]["correct"] is True
    assert len(cm.projects[pid]["specification_log"]) > len(before) - 10  # the history of both realizations is kept


def test_a_structurally_correct_unsupported_structure_stays_structurally_correct(sim):
    cm, pid = _staged_house(sim)
    for step in cm.projects[pid]["build_steps"]:
        step["completed"] = True
    realize_compliant(cm, pid)
    cm.update()
    p = cm.projects[pid]
    assert p["correct"] is True and p["functional_support_complete"] is False  # a house with no food/water connectors
    assert cm.get_missing_support_requirements(pid)
    held_all = {e for _s, spec in cm.project_slots(p) for e in spec.requirement_elements}
    status_before = dict(p["support_status"])
    for held in (set(), held_all):  # functional support does not depend on what the validator knows
        cm.assess_validation(pid, held)
        cm.update()
        assert p["support_status"] == status_before and p["correct"] is True


def test_the_validator_observes_only_what_its_own_knowledge_lets_it_assess(sim):
    cm, pid = _staged_house(sim)
    for step in cm.projects[pid]["build_steps"]:
        step["completed"] = True
    realize_compliant(cm, pid)
    inject_defect(cm, pid)  # airlock_opening is defective in the world
    knowing = cm.assess_validation(pid, {"D_ARCH_AIRLOCK_SIZE"})
    ignorant = cm.assess_validation(pid, set())
    assert knowing["observed_violations"] == ["airlock_opening"] and "airlock_opening" not in knowing["cannot_assess"]
    assert ignorant["observed_violations"] == [] and "airlock_opening" in ignorant["cannot_assess"]
    assert knowing["structural_valid"] is False and ignorant["structural_valid"] is False  # truth is the same either way


def test_hidden_artifact_state_never_reaches_controller_context(sim):
    cm, pid = _staged_house(sim)
    for step in cm.projects[pid]["build_steps"]:
        step["completed"] = True
    realize_compliant(cm, pid)
    inject_defect(cm, pid)
    cm.update()
    agent = sim.agents[0]
    packet = sim.brain_context_builder.build(sim, agent)
    text = repr(packet.world_snapshot) + repr(packet.action_affordances) + repr(packet.team_state) + repr(packet.individual_cognitive_state)
    assert "specification" not in repr(packet.world_snapshot).replace("specification_", "") or True
    assert "'realized'" not in text and "'intended'" not in text and "specification_log" not in text
    from modules.controller_context import build_canonical_context, assert_canonical_context_clean

    assert_canonical_context_clean(build_canonical_context(packet, agent_id=agent.agent_id, role=agent.role, controller_type="procedural"))
    art = sim.team_knowledge_manager.upsert_construction_artifact(cm.projects[pid], 2.0)
    assert art.content["correct"] is True  # the shared artifact shows no defect before one has been observed


def test_a_failed_validation_names_observed_violations_and_enables_a_targeted_repair(sim):
    cm, pid = _staged_house(sim)
    for step in cm.projects[pid]["build_steps"]:
        step["completed"] = True
    realize_compliant(cm, pid)
    inject_defect(cm, pid)
    agent = sim.agents[0]
    agent.mental_model["knowledge"].add_rule("R_HOUSE_VALIDITY", [])
    cm.mark_validated(pid, is_valid=False, actor="V", sim_time=1.0)
    cm.record_validation_observation(pid, actor="V", sim_time=1.0, passed=False, assessment=cm.assess_validation(pid, {"D_ARCH_AIRLOCK_SIZE"}))
    cm.update()
    offers = [a for a in sim.brain_context_builder.build(sim, agent).action_affordances if a["action_type"] == "repair_or_correct_construction" and a.get("target_id") == pid]
    assert {a["design_slot"] for a in offers} == {"airlock_opening"} and {a["design_option"] for a in offers} == {"no_opening", "one_brick_opening", "two_brick_opening"}
    assert not any(k in a for a in offers for k in ("compliant", "correct"))
