"""Gate J: an explicit run seed makes the baseline reproducible and is recorded in the session."""

import json
import tempfile
from pathlib import Path

from modules.headless_runner import run_batch_experiment
from modules.knowledge import Information, OrderedSet
from modules.simulation import SimulationState


def _run(seed, ticks=150):
    events = []
    with tempfile.TemporaryDirectory() as tmp:
        sim = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="seed_test", seed=seed)
        sim.logger.register_event_listener(
            lambda e: events.append((e["time"], e["event_type"], json.dumps({k: v for k, v in e["payload_data"].items() if k not in ("session_folder", "path")}, sort_keys=True, default=str)))
        )
        try:
            for _ in range(ticks):
                sim.update(1.0)
            manifest = json.loads((Path(sim.logger.output_session.session_folder) / "session_manifest.json").read_text(encoding="utf-8"))
        finally:
            sim.stop()
    return events, manifest


def test_same_seed_reproduces_the_event_sequence_and_different_seed_does_not():
    a, manifest = _run(7)
    b, _ = _run(7)
    c, _ = _run(8)
    assert a == b
    assert a != c
    assert manifest["seed"] == 7


def test_unseeded_simulation_records_no_seed():
    _, manifest = _run(None, ticks=3)
    assert manifest["seed"] is None


def test_headless_settings_accept_a_seed_and_write_it_to_the_manifest(tmp_path, monkeypatch):
    captured = {}
    import modules.headless_runner as hr

    real = hr.SimulationState

    def _factory(*args, **kwargs):
        kwargs["project_root"] = str(tmp_path)
        sim = real(*args, **kwargs)
        captured["seed"] = sim.seed
        return sim

    monkeypatch.setattr(hr, "SimulationState", _factory)
    run_batch_experiment({"experiment_name": "seed_cli", "timesteps_per_run": 2, "base_dt": 1.0, "seed": 41, "num_runs": 1, "brain_backend": "rule_brain"})
    assert captured["seed"] == 41
    manifest = json.loads(next(tmp_path.glob("Outputs/seed_cli_*/session_manifest.json")).read_text(encoding="utf-8"))
    assert manifest["seed"] == 41


def test_ordered_set_iterates_in_insertion_order_and_behaves_like_a_set():
    infos = [Information(f"I{i}", f"c{i}") for i in range(20)]
    holdings = OrderedSet()
    for info in reversed(infos):
        holdings.add(info)
    holdings.add(infos[0])  # duplicate add keeps position
    assert [i.id for i in holdings] == [f"I{i}" for i in reversed(range(20))]
    assert infos[3] in holdings and len(holdings) == 20
    holdings.discard(infos[3])
    assert infos[3] not in holdings and len(holdings) == 19
    holdings.update([infos[3]])
    assert list(holdings)[-1] is infos[3]
