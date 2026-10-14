"""Procedural Baseline treats validation-ready projects as pending work, and validation survives.

Covers: pending-work recognition, VALIDATE selection, legal-VALIDATE preservation through generic
pivots, blockability of not-ready validation, ActionGate/execution readiness agreement, and the
Construction validation transition.
"""

import random
import tempfile
from unittest.mock import patch

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain
from modules.simulation import SimulationState


# --------------------------------------------------------------------------- policy-level


def _project(structure_id, *, status, validation_ready, progress=1.0):
    return {
        "structure_id": structure_id,
        "state": "built" if status == "complete" else "in_progress",
        "progress": progress,
        "physical_build_progress": progress,
        "project_status": status,
        "ready_for_validation": status == "ready_for_validation",
        "validation_readiness": {"validation_ready": validation_ready},
        "needs_repair": False,
        "resource_complete": True,
        "required_resources": 10,
        "delivered_resources": 10,
    }


def _context(built_state, *, mode="LOGISTICS", dwell=79, no_progress=4, affordances=None):
    affordances = affordances or [
        {"action_type": "validate_construction", "utility": 0.35, "target_id": "active_construction", "reachable": True},
        {"action_type": "consult_team_artifact", "utility": 0.6, "target_id": "whiteboard"},
        {"action_type": "externalize_plan", "utility": 0.8, "target_id": "whiteboard"},
        {"action_type": "transport_resources", "utility": 0.3, "target_id": "resource_zone_to_work_zone", "reachable": True},
        {"action_type": "communicate", "utility": 0.1, "target_id": "nearby_agent", "reachable": True},
        {"action_type": "wait", "utility": 0.1, "target_id": None},
    ]
    return BrainContextPacket(
        static_task_context={"role": "Botanist"},
        world_snapshot={"sim_time": 300.0, "phase_profile": {"stage": "execution", "name": "Phase 1"}, "built_state": built_state},
        individual_cognitive_state={
            "traits": {},
            "known_gaps": [],
            "build_readiness": {"ready_for_build": True},
            "goal_stack": [{"goal_id": f"g{i}"} for i in range(4)],  # live agents carry several goals
            "loop_counters": {"action_repeats": 0, "selected_action_repeats": 0, "no_progress_streak": no_progress},
            "progress_state": {"no_progress_streak": no_progress},
            "seconds_since_dik_change": 50.0,
            "control_state": {"mode": mode, "mode_dwell_steps": dwell, "mode_history": [], "transition_history": []},
            "inspect_state": {"source_exhaustion": {}},
            "epistemic_sufficiency": {"score": 0.15, "refresh_pressure": 0.0},
        },
        team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
        history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
        action_affordances=affordances,
    )


READY = [_project("p1", status="ready_for_validation", validation_ready=True)]
NOT_READY = [_project("p1", status="in_progress", validation_ready=False)]
DONE = [_project("p1", status="complete", validation_ready=True)]


def test_ready_for_validation_project_is_pending_work_even_when_physically_complete():
    brain = RuleBrain()
    ready = brain._extract_features(_context(READY))
    assert ready["validation_work_pending"] == 1.0
    assert ready["active_incomplete_projects"] == 0.0  # physical construction is complete...
    assert brain._closure_only_work(ready)  # ...but the project is still unfinished work

    assert brain._extract_features(_context(NOT_READY))["validation_work_pending"] == 0.0
    assert brain._extract_features(_context(DONE))["validation_work_pending"] == 0.0


def test_pending_validation_is_not_a_coordination_deadlock_but_a_missing_action_still_is():
    brain = RuleBrain()
    assert brain._extract_features(_context(READY))["coordination_deadlock_pressure"] == 0.0
    # no validate affordance offered -> the existing deadlock reading is unchanged
    no_validate = [a for a in _context(READY).action_affordances if a["action_type"] != "validate_construction"]
    assert brain._extract_features(_context(READY, affordances=no_validate))["coordination_deadlock_pressure"] > 0.0
    # nothing pending -> unchanged
    assert brain._extract_features(_context(NOT_READY))["coordination_deadlock_pressure"] > 0.0


@pytest.mark.parametrize("mode", ["LOGISTICS", "COORDINATE", "CONSTRUCT", "MONITOR"])
def test_baseline_selects_validate_for_a_ready_project_from_any_stuck_mode(mode):
    brain = RuleBrain()
    context = _context(READY, mode=mode)
    chosen = []
    for _ in range(6):  # control state carries between decisions, as in the live controller
        chosen.append(brain.decide(context).selected_action)
        if chosen[-1] == ExecutableActionType.VALIDATE_CONSTRUCTION:
            break
    assert chosen[-1] == ExecutableActionType.VALIDATE_CONSTRUCTION, chosen
    # and it stays on validation rather than flapping back
    assert brain.decide(context).selected_action == ExecutableActionType.VALIDATE_CONSTRUCTION


def test_baseline_does_not_select_validate_when_nothing_is_validation_ready():
    for built in (NOT_READY, DONE):
        brain = RuleBrain()
        context = _context(built)
        assert all(brain.decide(context).selected_action != ExecutableActionType.VALIDATE_CONSTRUCTION for _ in range(6))


# --------------------------------------------------------------------------- simulator-level


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="validation_pending")
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _make_project(sim, *, ready, project_id="site_b_house_001"):
    cm = sim.environment.construction
    cm.create_project("site_b", structure_type="house", project_id_override=project_id)
    project = cm.projects[project_id]
    project["started"] = True
    if ready:
        project["delivered_resources"]["bricks"] = project["required_resources"]["bricks"]
        project["resource_complete"] = True
        project["structurally_complete"] = True
        project["functional_support_complete"] = True
        project["support_requirements"] = {}
        project["epistemic_workspace"]["entries"] = [{"entry_type": "claim"}, {"entry_type": "evidence"}, {"entry_type": "design_note"}]
    # status is a per-tick cache and is deliberately left stale (Construction.update() has not run)
    project["status"] = "in_progress"
    return project_id


def _validate_decision(project_id):
    return BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id, confidence=0.9)


def _force_pivot(agent, sim, action="inspect_information_source"):
    agent.control_state["mode"] = "LOGISTICS"  # past the startup bootstrap guard, which bypasses all pivots
    agent.progress_tracker["forced_pivot"] = action
    agent.progress_tracker["forced_pivot_until"] = sim.time + 1000.0


def test_legal_pilot_validate_is_not_overwritten_by_recovery_pivot(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=True)
    _force_pivot(agent, sim)
    decision = _validate_decision(project_id)

    result = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="unit_test")

    assert result.selected_action == ExecutableActionType.VALIDATE_CONSTRUCTION
    assert result.target_id == project_id
    assert any(name == "policy_pivot_skipped_legal_validation" for name, _ in sim.captured_events)


def test_not_ready_validate_is_still_pivoted_and_blockable(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=False)
    _force_pivot(agent, sim)
    decision = _validate_decision(project_id)

    result = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="unit_test")
    assert result.selected_action != ExecutableActionType.VALIDATE_CONSTRUCTION  # existing pivot still applies

    gate = sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)
    assert gate.legal is False
    assert {"physical_incomplete", "physical_build_not_started"} & set(gate.blockers)
    # and the translation path rejects it rather than executing it
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    assert actions[0].get("decision_action") != "validate_construction"
    assert any(name == "mecha_action_blocked" and p.get("selected_action") == "validate_construction" for name, p in sim.captured_events)


def _run_validation(sim, agent, project_id):
    decision = _validate_decision(project_id)
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    agent.position = sim.environment.get_interaction_target_position(project_id, from_position=agent.position)
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)
    return actions


def test_gate_and_execution_agree_on_readiness_despite_stale_cached_status(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=True)
    construction = sim.environment.construction
    assert construction.projects[project_id]["status"] == "in_progress"  # cached status is stale
    decision = _validate_decision(project_id)
    assert sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim).legal is True
    assert construction.evaluate_project_validation_readiness(project_id)["validation_ready"] is True

    _run_validation(sim, agent, project_id)

    blocked = [p for name, p in sim.captured_events if name == "construction_validation_blocked"]
    assert not blocked, blocked
    assert construction.projects[project_id]["status"] == "complete"


def test_successful_validation_goes_through_construction_mark_validated(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=True)
    construction = sim.environment.construction
    with patch.object(construction, "mark_validated", wraps=construction.mark_validated) as spy:
        _run_validation(sim, agent, project_id)

    assert spy.called
    assert spy.call_args.args[0] == project_id
    assert spy.call_args.kwargs.get("is_valid") is True
    project = construction.projects[project_id]
    assert project["validated_complete"] is True
    assert project["status"] == "complete"
    assert any(name == "construction_validated_correct" for name, _ in sim.captured_events)


def test_execution_still_blocks_a_validation_that_is_not_ready(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=False)
    construction = sim.environment.construction
    # bypass the gate on purpose: execution must independently refuse an unready project
    decision = _validate_decision(project_id)
    action = {
        "type": "idle",
        "duration": 1.0,
        "priority": 1,
        "decision_action": "validate_construction",
        "project_id": project_id,
        "target_id": project_id,
        "target_kind": "construction_project",
        "target": sim.environment.get_interaction_target_position(project_id, from_position=agent.position),
        "progress": 0.0,
    }
    agent.position = action["target"]
    agent.active_actions = [action]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)

    assert [p for name, p in sim.captured_events if name == "construction_validation_blocked" and p.get("failure_category") == "illegal_project_status"]
    assert construction.projects[project_id].get("validated_complete") is not True
    assert decision.selected_action == ExecutableActionType.VALIDATE_CONSTRUCTION


# --------------------------------------------------------------------------- seeded end-to-end


def test_seeded_baseline_run_validates_built_projects_instead_of_waiting():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="validation_e2e")
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["time"], e["event_type"], e["payload_data"])))
        try:
            for _ in range(240):
                state.update(1.0)
            projects = state.environment.construction.projects
        finally:
            state.stop()

    validated = [p for _, name, p in events if name == "construction_validated_correct"]
    assert len(validated) >= 3
    assert sum(1 for p in projects.values() if p.get("status") == "complete") >= 3
    assert not [p for _, name, p in events if name == "construction_validation_blocked" and p.get("failure_category") == "illegal_project_status"]
