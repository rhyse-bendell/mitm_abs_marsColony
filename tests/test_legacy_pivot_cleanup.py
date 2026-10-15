"""Legacy scaffold pivots no longer ATTEMPT to substitute state-creating actions for a Pilot's choice."""

import random
import tempfile

import pytest

from modules.simulation import SimulationState

ATTEMPT_EVENTS = {"mecha_project_substitution_rejected", "recovery_pivot_declined_project_taskwork"}


@pytest.mark.parametrize("seed", [0, 2])
def test_a_normal_seeded_run_produces_zero_attempted_state_creating_substitutions(seed):
    random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="pivot_cleanup", seed=seed)
        attempts = []
        state.logger.register_event_listener(lambda e: attempts.append((e["time"], e["event_type"], e["payload_data"])) if e["event_type"] in ATTEMPT_EVENTS else None)
        try:
            for _ in range(260):
                state.update(1.0)
        finally:
            state.stop()
    assert attempts == []  # not merely rejected by the guard: never attempted


def test_completed_connections_record_whether_they_are_cross_site():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="cross_site", seed=1)
        try:
            cm = state.environment.construction
            cm.sites["site_b"].capacity = 30
            def built(site, kind):
                pid, _ = cm.create_project(site, structure_type=kind, author="system")
                assert pid, (site, kind)
                cm.projects[pid].update({"structurally_complete": True, "status": "complete", "validated_complete": True, "correct": True})
                return pid
            generator, connector, greenhouse_b = built("site_b", "water_generator"), built("site_b", "water_connector"), built("site_b", "greenhouse")
            cm.bridges["bridge_bc"].status = "complete"  # unlock site_c for the structure setup only`n            cm.sites["site_c"].capacity = 30
            greenhouse_c = built("site_c", "greenhouse")
            assert cm.attach_connector(connector, from_project_id=greenhouse_b, to_project_id=generator, actor="t", sim_time=0.0)[0]
            assert cm.attach_connector(connector, from_project_id=greenhouse_c, to_project_id=generator, actor="t", sim_time=0.0)[0]  # permissive: not forbidden
            flags = {(l["from_project_id"], l["to_project_id"]): l["cross_site_connection"] for l in cm.connectors}
            assert flags[(greenhouse_b, generator)] is False and flags[(greenhouse_c, generator)] is True
        finally:
            state.stop()