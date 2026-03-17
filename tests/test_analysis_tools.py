import csv
import json
from pathlib import Path

from modules.analysis_loader import load_analysis_session
from modules.analysis_stats import aggregate_statistics, phase_statistics
from modules.replay_engine import ReplayEngine


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_session(tmp_path: Path, include_optional: bool = True) -> Path:
    session = tmp_path / "Outputs" / "experiment_20260101"
    _write_json(session / "session_manifest.json", {"experiment_name": "x"})
    _write_json(session / "measures" / "run_summary.json", {"run_metadata": {"effective_brain_backend": "rule_brain"}, "process": {"planner_responsiveness": {"requests_started": 3}}})
    _write_json(session / "measures" / "team_summary.json", {"backend": {"fallback_count": 1}})
    _write_json(session / "measures" / "phase_summary.json", [{"phase_name": "Phase 1", "duration_seconds": 12, "events": {"movement_started": 2}}])
    _write_csv(session / "measures" / "agent_summary.csv", [{"agent": "Architect", "time_moving": 1.2}])
    _write_csv(
        session / "logs" / "events.csv",
        [
            {"time": "0.0", "event_type": "phase_transition", "payload": "{}"},
            {"time": "1.0", "event_type": "movement_started", "payload": "{}"},
            {"time": "2.0", "event_type": "brain_provider_fallback", "payload": "{}"},
        ],
    )
    _write_csv(
        session / "logs" / "experiment_20260101.csv",
        [
            {"time": "0.0", "agent": "Architect", "goal": "build", "x": "10", "y": "15"},
            {"time": "1.0", "agent": "Architect", "goal": "build", "x": "12", "y": "15"},
        ],
    )
    (session / "logs").mkdir(parents=True, exist_ok=True)
    (session / "logs" / "planner_trace.jsonl").write_text('{"trace": 1}\n', encoding="utf-8")
    (session / "logs" / "interaction_trace.jsonl").write_text('{"time":0.5,"interaction_type":"planner_request","source_node":"Agent:Architect","target_node":"Planner:OllamaQwen","status":"started"}\n', encoding="utf-8")

    if include_optional:
        _write_json(session / "measures" / "runtime_witness_coverage.json", {"coverage": 0.9})
        _write_json(session / "measures" / "startup_llm_sanity.json", {"startup_llm_sanity_enabled": True})
    return session


def test_analysis_loader_full_artifacts(tmp_path):
    session_dir = _build_session(tmp_path, include_optional=True)
    loaded = load_analysis_session(session_dir)
    assert loaded.artifacts.run_summary is not None
    assert len(loaded.artifacts.events) == 3
    assert len(loaded.artifacts.state_rows) == 2
    assert loaded.artifacts.runtime_witness_coverage == {"coverage": 0.9}
    assert len(loaded.artifacts.interaction_trace) == 1


def test_analysis_loader_missing_optional_files(tmp_path):
    session_dir = _build_session(tmp_path, include_optional=False)
    loaded = load_analysis_session(session_dir)
    assert loaded.artifacts.runtime_witness_coverage is None
    assert any("runtime_witness_coverage.json" in msg for msg in loaded.warnings)


def test_replay_engine_frame_construction(tmp_path):
    session_dir = _build_session(tmp_path, include_optional=True)
    loaded = load_analysis_session(session_dir)
    engine = ReplayEngine(loaded)
    assert len(engine.frames) >= 3
    assert engine.frames[-1].agent_states["Architect"]["x"] == "12"


def test_aggregate_and_phase_stats(tmp_path):
    session_dir = _build_session(tmp_path, include_optional=True)
    loaded = load_analysis_session(session_dir)
    aggregate = aggregate_statistics(loaded)
    phases = phase_statistics(loaded)

    assert aggregate["planner"]["requests_started"] == 3
    assert aggregate["backend"]["fallback_count"] == 1
    assert phases[0]["phase_name"] == "Phase 1"
