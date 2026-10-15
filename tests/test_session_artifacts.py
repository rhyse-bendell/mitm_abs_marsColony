"""Gate K: a finished run leaves a useful, bounded research session."""

import json
import pytest
import tempfile
from pathlib import Path

from modules.simulation import SimulationState


def _run(ticks, seed=5):
    with tempfile.TemporaryDirectory() as tmp:
        sim = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="artifact_test", seed=seed)
        events = []
        sim.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        for _ in range(ticks):
            sim.update(1.0)
        sim.stop()
        session = Path(sim.logger.output_session.session_folder)
        measures = {p.name: p.read_text(encoding="utf-8") for p in (session / "measures").glob("*")}
        manifest = json.loads((session / "session_manifest.json").read_text(encoding="utf-8"))
    return events, measures, manifest


def test_final_measures_replace_the_placeholder_and_report_authoritative_state():
    _, measures, manifest = _run(260)
    assert "final_measures_placeholder.json" not in measures
    final = json.loads(measures["final_measures.json"])
    assert final["run"]["seed"] == 5 == manifest["seed"]
    assert final["run"]["sim_time_end"] == 260.0
    assert set(final["run"]["pilots"].values()) == {"procedural_baseline"}
    construction = final["construction"]
    # Whether any structure is validated in 260 s is Pilot-policy performance (no Mecha rescue stands in for it); the report
    # only has to state the authoritative count.
    assert isinstance(construction["structures_validated"], int) and construction["structures_validated"] >= 0
    assert construction["bricks_delivered_to_projects"] > 0
    assert {"pile_a", "pile_c"} <= set(construction["resource_piles"])
    assert final["mission"]["mission_success"] is None  # not evaluated: the task package leaves capacity-per-structure undefined
    assert final["mission"]["phase_targets"]


def test_affordance_events_record_the_offered_set_once_per_change_and_selections_point_at_it():
    events, _, _ = _run(120)
    listings = [p for name, p in events if name == "action_affordances_generated"]
    selections = [p for name, p in events if name == "pilot_selection_via_adapter"]
    assert listings and selections
    assert len(listings) < len(selections)  # not one listing per decision
    listed = {p["affordance_signature"]: p for p in listings}
    assert all(p["affordance_signature"] in listed for p in selections)  # every selection resolves to a listed offer
    sample = listings[0]["offered"][0]
    assert len(sample) == 4  # [action_type, target_id, reachable, utility]


def test_inferred_rule_ids_are_bounded_in_event_payloads():
    events, _, _ = _run(60)
    for name, payload in events:
        if name == "project_state_recomputed_after_dik_change":
            assert all(len(r) <= 60 for r in payload["changed_rule_ids"])


def test_terminal_state_is_derived_from_authoritative_state_only():
    with tempfile.TemporaryDirectory() as tmp:
        sim = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="terminal_test", seed=1)
        sim.task_model.manifest["terminal_policy"] = {"end_on_resources_exhausted": True}  # Mars defaults to non-terminal
        try:
            cm = sim.environment.construction
            assert sim._check_terminal_state() is None  # bricks remain: not terminal
            sim.terminal_state = None
            pid, _ = cm.create_project("site_b", structure_type="house")
            cm.projects[pid]["started"] = True
            for pile in cm.resource_nodes.values():
                pile.quantity = 0
            # unfinished, materially incomplete project and no bricks anywhere: nothing can advance
            assert sim._check_terminal_state() == "resources_exhausted"

            sim.terminal_state = None
            cm.projects[pid]["resource_complete"] = True  # materials are in place: validation/building can still advance
            assert sim._check_terminal_state() is None

            sim.terminal_state = None
            sim.time = 3000.0
            assert sim._check_terminal_state() == "time_expired"
            assert sim.terminal_state == "time_expired" and sim.terminal_time == 3000.0
        finally:
            sim.stop()


def test_headless_runner_stops_at_the_terminal_state_and_records_it(tmp_path, monkeypatch):
    import modules.headless_runner as hr
    from modules.headless_runner import run_batch_experiment

    real = hr.SimulationState
    holder = {}

    def _factory(*args, **kwargs):
        kwargs["project_root"] = str(tmp_path)
        sim = real(*args, **kwargs)
        sim.task_model.manifest["terminal_policy"] = {"end_on_resources_exhausted": True}
        for pile in sim.environment.construction.resource_nodes.values():
            pile.quantity = 0  # a run that is terminal from the start
        holder["sim"] = sim
        return sim

    monkeypatch.setattr(hr, "SimulationState", _factory)
    run_batch_experiment({"experiment_name": "terminal_cli", "timesteps_per_run": 500, "base_dt": 1.0, "seed": 2, "num_runs": 1, "brain_backend": "rule_brain"})
    sim = holder["sim"]
    assert sim.terminal_state == "resources_exhausted"
    assert sim.time < 20  # stopped right after the first terminal check, not after 500 steps
    final = json.loads(next(tmp_path.glob("Outputs/terminal_cli_*/measures/final_measures.json")).read_text(encoding="utf-8"))
    assert final["run"]["terminal_state"] == "resources_exhausted"
