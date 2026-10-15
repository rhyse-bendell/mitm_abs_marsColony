"""Existing-project START is offered only once materials are staged; typed project creation and TRANSPORT are untouched."""

import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState

START = ExecutableActionType.START_CONSTRUCTION
TRANSPORT = ExecutableActionType.TRANSPORT_RESOURCES.value


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="start_gate", seed=3)
        try:
            yield state
        finally:
            state.stop()


def _offers(sim, agent):
    return sim.brain_context_builder.build(sim, agent).action_affordances


def _existing_starts(offers, pid):
    return [a for a in offers if a["action_type"] == START.value and a.get("target_id") == pid and a.get("start_mode") == "existing_project"]


def _new_project(sim):
    pid, _ = sim.environment.construction.create_project("site_b", structure_type="house", author="system")
    return pid


def test_1_typed_new_project_start_remains_available(sim):
    offers = _offers(sim, sim.agents[0])
    assert [a for a in offers if a["action_type"] == START.value and a.get("start_mode") == "new_project"]


def test_2_unstaged_existing_project_offers_no_physical_start_and_states_the_deficit(sim):
    pid = _new_project(sim)
    agent = sim.agents[0]
    offers = _offers(sim, agent)
    assert not _existing_starts(offers, pid)
    stated = [a for a in offers if a.get("target_id") == pid]
    assert stated and all(a["materials_staged"] is False and a["materials_deficit"] > 0 for a in stated)


def test_3_transport_to_the_exact_project_remains_available(sim):
    pid = _new_project(sim)
    offers = _offers(sim, sim.agents[0])
    assert [a for a in offers if a["action_type"] == TRANSPORT and a.get("target_id") == pid]


def test_4_start_becomes_available_once_materials_are_staged(sim):
    cm = sim.environment.construction
    pid = _new_project(sim)
    required = cm.projects[pid]["required_resources"]["bricks"]
    assert cm.deliver_resource(pid, "bricks", required, from_carried=True)
    assert cm.projects[pid]["build_ready"] is True
    starts = _existing_starts(_offers(sim, sim.agents[0]), pid)
    assert starts and starts[0]["materials_staged"] is True and starts[0]["materials_deficit"] == 0


def test_5_no_middleware_project_substitution(sim):
    cm = sim.environment.construction
    pid = _new_project(sim)
    other = cm.create_project("site_b", structure_type="greenhouse", author="system")[0]
    agent = sim.agents[0]
    decision = BrainDecision(selected_action=START, target_id=pid, confidence=0.9)
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    assert decision.target_id == pid and decision.selected_action == START
    assert all(a.get("project_id") in (None, pid) for a in actions)
    assert not any(a.get("project_id") == other for a in actions)
    assert not any(a.get("type") in ("build", "construct") for a in actions)  # blocked, not rewritten into a build on another project
    assert not cm._project_steps_started(pid) if hasattr(cm, "_project_steps_started") else not any(s.get("completed") for s in cm.projects[pid]["build_steps"])


def test_6_no_repeated_unstaged_existing_project_start_in_seeded_run():
    from modules.agent import Agent

    counted = []
    original = Agent._select_decision_through_pilot_adapter

    def watch(self, sim_state, context, trigger_reason, runtime=None):
        decision = original(self, sim_state, context, trigger_reason, runtime)
        project = sim_state.environment.construction.projects.get(str(decision.target_id or ""))
        if decision.selected_action == START and project is not None and not project.get("build_ready") \
                and not any(s.get("completed") for s in (project.get("build_steps") or [])):
            counted.append((round(sim_state.time, 1), self.name, decision.target_id))
        return decision

    Agent._select_decision_through_pilot_adapter = watch
    try:
        with tempfile.TemporaryDirectory() as tmp:
            state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="start_loop", seed=2,
                                   experimental_settings={"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}})
            try:
                for _ in range(900):  # the pre-fix loop appears after ~700 s at this seed/condition (46 unstaged START selections by 900 s)
                    state.update(1.0)
            finally:
                state.stop()
    finally:
        Agent._select_decision_through_pilot_adapter = original
    assert not counted, counted[:10]