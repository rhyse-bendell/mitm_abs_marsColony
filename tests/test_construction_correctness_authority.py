"""Gate H: project correctness is written only by the construction subsystem, via modeled mechanisms."""

import tempfile
from unittest.mock import patch

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construction import ConstructionManager
from modules.simulation import SimulationState


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
    for step in p["build_steps"]:  # physically built: Construction.update() recomputes completeness from the steps
        step["completed"] = True
    cm.update()
    return project_id


def test_build_error_is_recorded_by_construction_and_only_ever_introduces_an_error():
    cm = ConstructionManager()
    pid = _ready_house(cm)
    assert cm.record_build_quality(pid, correct=True, actor="A") is False  # a build cannot "make correct"
    assert cm.projects[pid]["correct"] is True
    assert cm.record_build_quality(pid, correct=False, actor="A", sim_time=5.0) is True
    assert cm.projects[pid]["correct"] is False and cm.projects[pid]["last_build_error"]["actor"] == "A"
    cm.update()
    assert cm.projects[pid]["status"] == "needs_repair"
    assert cm.record_build_quality(pid, correct=False, actor="A") is False  # already incorrect


def test_repair_clears_the_error_but_does_not_validate_the_project():
    cm = ConstructionManager()
    pid = _ready_house(cm)
    cm.record_build_quality(pid, correct=False, actor="A")
    cm.update()
    assert cm.repair_project(pid, actor="B", sim_time=9.0) is True
    p = cm.projects[pid]
    assert p["correct"] is True and p["validated_complete"] is False
    assert p["status"] != "complete"  # a separate VALIDATE action must still validate it
    cm.mark_validated(pid, is_valid=True, actor="B")
    assert cm.projects[pid]["status"] == "complete"
    assert ConstructionManager().repair_project("missing") is False


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="correctness_authority", seed=3)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def test_agent_rule_disagreement_never_writes_project_correctness(sim):
    agent = sim.agents[0]
    cm = sim.environment.construction
    pid = _ready_house(cm)
    cm.projects[pid]["status"] = "in_progress"
    cm.projects[pid]["in_progress"] = True
    agent.mental_model["knowledge"].rules.append("K_UNRELATED_RULE")  # agent knows rules, but not the project's expected rule
    for step in range(40):
        sim.time = 100.0 + 6.0 * step  # beyond the detection/repair cooldowns
        agent.compare_and_repair_construction(cm, sim_state=sim)
    assert cm.projects[pid]["correct"] is True
    assert not [1 for name, _ in sim.captured_events if name == "construction_repair_episode"]
    # a mismatch can still be perceived and raised with the team
    assert any(name == "construction_mismatch_detected" for name, _ in sim.captured_events)

    cm.record_build_quality(pid, correct=False, actor="X")
    for step in range(40):
        sim.time = 500.0 + 6.0 * step
        agent.compare_and_repair_construction(cm, sim_state=sim)
    assert cm.projects[pid]["correct"] is False  # no ambient repair either


def test_repair_action_executes_through_construction_then_requires_validation(sim):
    agent = sim.agents[0]
    cm = sim.environment.construction
    pid = _ready_house(cm)
    cm.record_build_quality(pid, correct=False, actor="X")
    cm.update()
    assert cm.projects[pid]["status"] == "needs_repair"
    cm.mark_validated(pid, is_valid=False, actor="X", sim_time=1.0)  # a repair needs an OBSERVED error: the failed validation
    agent._rng.random = lambda: 0.0  # the fidelity roll applied to the repair itself succeeds in this test

    decision = BrainDecision(selected_action=ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, target_id=pid, confidence=0.9)
    # the agent's own information-readiness precondition is held satisfied; this test targets execution + Construction
    grounded = {"score": 1.0, "role_missing": False, "stale_grounding": False, "refresh_pressure": 0.0, "construction_state_needs_recheck": False, "sufficient_for_validation": True}
    with patch.object(agent, "_build_readiness_blockers", return_value=[]), patch.object(agent, "_epistemic_sufficiency", return_value=grounded):
        actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    agent.position = sim.environment.get_interaction_target_position(pid, from_position=agent.position)
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)

    assert cm.projects[pid]["correct"] is True
    assert cm.projects[pid]["status"] != "complete"
    assert any(name == "construction_repair_episode" for name, _ in sim.captured_events)

    cm.update()
    validate = BrainDecision(selected_action=ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=pid, confidence=0.9)
    actions = agent._translate_brain_decision_to_legacy_action(validate, sim.environment, sim_state=sim)
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)
    assert cm.projects[pid]["status"] == "complete"


def test_invalid_repair_invocation_does_not_corrupt_state(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    before = dict(cm.projects[pid])
    assert cm.repair_project("no_such_project") is False
    unstarted, _ = cm.create_project("site_b", structure_type="greenhouse")
    assert cm.repair_project(unstarted) is False
    assert cm.projects[pid]["correct"] == before["correct"]
    assert cm.projects[unstarted].get("validated_complete") in (False, None)
