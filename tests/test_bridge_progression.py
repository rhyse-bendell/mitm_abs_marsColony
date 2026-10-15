"""Bridge (bridge_bc) progression helpers must not crash and must reflect real bridge state."""

import tempfile

import pytest

from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="bridge_progression")
        try:
            yield state
        finally:
            state.stop()


def test_execution_lock_helpers_handle_the_bridge_without_error(sim):
    agent = sim.agents[0]
    env = sim.environment
    assert env.construction.bridges["bridge_bc"].status != "complete"
    assert agent._execution_lock_viable_project(env, "bridge_bc", sim_state=sim) is True
    assert agent._start_execution_lock(env, "bridge_bc", sim_state=sim) is False  # no project lock on a bridge

    env.construction.bridges["bridge_bc"].status = "complete"
    assert agent._execution_lock_viable_project(env, "bridge_bc", sim_state=sim) is False


from modules.action_schema import BrainDecision, ExecutableActionType
import random


def _bridge_transport():
    return BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="bridge_bc", confidence=0.9)


def test_bridge_needs_transport_tracks_real_bridge_state(sim):
    agent = sim.agents[0]
    env = sim.environment
    bridge = env.construction.bridges["bridge_bc"]
    assert agent._bridge_needs_transport(env, "bridge_bc") is True
    assert agent._bridge_needs_transport(env, "site_b_house_001") is False  # not a bridge
    bridge.delivered_resources = bridge.required_resources
    assert agent._bridge_needs_transport(env, "bridge_bc") is False
    bridge.delivered_resources = 0
    bridge.status = "complete"
    assert agent._bridge_needs_transport(env, "bridge_bc") is False


def test_unfinished_bridge_transport_is_not_rewritten_as_a_satisfied_project(sim):
    agent = sim.agents[0]
    agent.control_state["mode"] = "LOGISTICS"
    decision = _bridge_transport()
    kept = agent._no_active_plan_successor_for_materially_satisfied_project(decision, sim.environment, sim_state=sim)
    assert (kept.selected_action, kept.target_id) == (ExecutableActionType.TRANSPORT_RESOURCES, "bridge_bc")
    pivoted = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, pivot_origin="unit_test")
    assert (pivoted.selected_action, pivoted.target_id) == (ExecutableActionType.TRANSPORT_RESOURCES, "bridge_bc")


def test_stagnation_regrounding_does_not_replace_a_productive_pilot_action(sim):
    agent = sim.agents[0]
    agent.control_state["mode"] = "LOGISTICS"
    stagnation = agent.metacognitive_state.setdefault("stagnation_counters", {})
    stagnation.update({"epistemic": 12, "support": 12, "closure": 0, "wait": 0})
    events = []
    sim.logger.register_event_listener(lambda e: events.append(e["event_type"]))

    out = agent._apply_metacognitive_regulation(_bridge_transport(), sim.environment, sim_state=sim, context=None, pivot_origin="unit_test")
    assert (out.selected_action, out.target_id) == (ExecutableActionType.TRANSPORT_RESOURCES, "bridge_bc")
    assert "metacognitive_support_loop_broken" not in events

    wait = BrainDecision(selected_action=ExecutableActionType.WAIT, confidence=0.9)
    agent._apply_metacognitive_regulation(wait, sim.environment, sim_state=sim, context=None, pivot_origin="unit_test")
    assert "metacognitive_support_loop_broken" in events  # the loop breaker still works for non-productive behavior


@pytest.mark.xfail(
    reason=(
        "Pilot-authority cleanup: this seeded colony outcome used to be reached partly because the Mecha swapped the Pilot's WAIT/consult choices "
        "and blocked project targets for taskwork on projects of its own choosing. The Procedural Baseline has no policy of its own that finishes "
        "Site B and the bridge; that is a Pilot-policy question, not something middleware may rescue."
    ),
    strict=False,
)
def test_seeded_run_completes_site_b_builds_the_bridge_and_works_at_site_c():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="bridge_e2e")
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["time"], e["event_type"], e["payload_data"])))
        try:
            for _ in range(2000):  # pace-independent: run until the bridge is built, then a little longer
                state.update(1.0)
                if state.environment.construction.bridges["bridge_bc"].status == "complete":
                    for _ in range(120):
                        state.update(1.0)
                    break
            cm = state.environment.construction
            statuses = {pid: p.get("status") for pid, p in cm.projects.items()}
            bridge_status = cm.bridges["bridge_bc"].status
        finally:
            state.stop()
    site_b = [pid for pid in statuses if pid.startswith("site_b_")]
    site_c = [pid for pid in statuses if pid.startswith("site_c_")]
    assert site_b and all(statuses[p] == "complete" for p in site_b)  # nothing stranded at Site B
    assert bridge_status == "complete"
    assert any(name == "site_unlocked_by_bridge" for _, name, _ in events)
    assert any(statuses[p] == "complete" for p in site_c)  # Site C is reachable and productive
    assert not [1 for _, name, p in events if name == "construction_transport_blocked" and p.get("failure_category") == "construction_delivery_rejected"]
