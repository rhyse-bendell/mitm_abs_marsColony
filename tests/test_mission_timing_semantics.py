"""Mars mission timing: phases and the mission end are clock-driven; brick exhaustion is a recorded world condition."""

import tempfile

import pytest

from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="timing_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _empty_piles(state):
    for pile in state.environment.construction.resource_nodes.values():
        pile.quantity = 0
    for site_id in state.environment.construction.site_resource_inventory:
        state.environment.construction.site_resource_inventory[site_id] = 0


def _mission_seconds(state):
    return sum(float(p["duration_minutes"]) * 60.0 for p in state.environment.phases)


def test_mars_does_not_terminate_when_the_piles_are_empty_and_records_exhaustion_once(sim):
    events = []
    sim.logger.register_event_listener(lambda e: events.append(e["event_type"]))
    _empty_piles(sim)
    for _ in range(3):
        assert sim._check_terminal_state() is None
    assert sim.terminal_state is None
    assert sim.resources_exhausted_time == sim.time
    assert events.count("resources_exhausted") == 1


def test_engine_still_supports_early_termination_when_a_task_policy_asks_for_it(sim):
    sim.task_model.manifest["terminal_policy"] = {"end_on_resources_exhausted": True}
    _empty_piles(sim)
    assert sim._check_terminal_state() == "resources_exhausted"


def test_mars_terminates_at_the_summed_phase_durations_not_a_hard_coded_time(sim):
    total = _mission_seconds(sim)
    assert total == 3000.0  # 30 + 20 minutes in the task package
    sim.time = total - 1.0
    assert sim._check_terminal_state() is None
    sim.time = total
    assert sim._check_terminal_state() == "time_expired"
    assert sim.terminal_time == total


def test_phase_two_starts_on_the_clock_with_the_second_bridge_unbuilt(sim):
    env = sim.environment
    assert env.construction.bridges["bridge_bc"].status != "complete"
    env.update(1799.0)
    assert env.get_current_phase()["name"] == "Phase 1"
    env.update(1800.0)
    assert env.get_current_phase()["name"] == "Phase 2"
    assert env.construction.bridges["bridge_bc"].status != "complete"


def test_site_c_and_its_pile_stay_locked_until_the_bridge_is_complete(sim):
    env, cm = sim.environment, sim.environment.construction
    assert not env.is_interaction_target_unlocked("Build_Site_C")
    assert "site_c" not in cm._accessible_sites_from("site_b")
    assert not cm.can_transport("site_c", "site_b")
    assert cm.create_project("site_c", structure_type="house")[1] == "site_locked_by_bridge"
    for _ in range(cm.bridges["bridge_bc"].required_resources):
        cm.build_bridge_bc(1)
    assert cm.bridges["bridge_bc"].status == "complete"
    assert env.is_interaction_target_unlocked("Build_Site_C")
    assert "site_c" in cm._accessible_sites_from("site_b")
    assert cm.can_transport("site_c", "site_b")


def test_a_brick_already_in_transit_can_still_be_delivered_after_the_piles_empty(sim):
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", structure_type="house")
    cm.site_resource_inventory["site_a"] = 1
    assert cm.reserve_transport("tester", "site_a", "site_b", 1)
    _empty_piles(sim)  # nothing left at the sources; the carried brick is in flight
    sim._check_terminal_state()
    assert sim.terminal_state is None
    for _ in range(int(cm.parameters["move_time_per_unit"]) + 1):
        cm.update()
    cm.site_resource_inventory["site_b"] = max(1, cm.site_resource_inventory.get("site_b", 0))
    assert cm.deliver_resource(pid, "bricks", 1)
    assert cm.projects[pid]["delivered_resources"]["bricks"] == 1
