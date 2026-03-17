import json
from pathlib import Path

from modules.analysis_loader import load_analysis_session
from modules.interaction_graph import (
    CANONICAL_NODES,
    build_interaction_from_sim_event,
    canonical_node,
)
from modules.logging_tools import SimulationLogger
from modules.replay_engine import ReplayEngine


def test_canonical_node_mapping_consistency():
    assert "Agent:Architect" in CANONICAL_NODES
    assert canonical_node("Agent:Architect") == "Agent:Architect"
    assert canonical_node("Unknown:Node") == "System:Logger"


def test_interaction_event_emission_helper_maps_planner_request():
    event = build_interaction_from_sim_event(
        1.5,
        "planner_request_started_async",
        {"agent": "Architect", "agent_id": "Architect", "request_id": "r1", "backend": "ollama"},
    )
    assert event is not None
    row = event.to_row()
    assert row["interaction_type"] == "planner_request"
    assert row["source_node"] == "Agent:Architect"


def test_interaction_trace_written_and_replay_loaded(tmp_path):
    logger = SimulationLogger(experiment_name="interaction_test", project_root=tmp_path)
    logger.log_event(0.0, "planner_request_started_async", {"agent": "Architect", "agent_id": "Architect", "request_id": "r1"})
    logger.log_event(0.5, "brain_decision_outcome", {"agent": "Architect", "agent_id": "Architect", "decision_status": "ok"})
    logger.save_csv()

    session_dir = logger.output_session.session_folder
    trace_path = session_dir / "logs" / "interaction_trace.jsonl"
    assert trace_path.exists()
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 2

    loaded = load_analysis_session(session_dir)
    assert len(loaded.artifacts.interaction_trace) >= 2
    replay = ReplayEngine(loaded)
    assert replay.frames
    assert any(frame.interaction_events_at_time for frame in replay.frames)


def test_analysis_loader_missing_interaction_trace_graceful(tmp_path):
    session = tmp_path / "Outputs" / "missing_interaction_20260101"
    (session / "logs").mkdir(parents=True, exist_ok=True)
    (session / "session_manifest.json").write_text("{}", encoding="utf-8")
    (session / "logs" / "events.csv").write_text("time,event_type,payload\n0.0,phase_transition,{}\n", encoding="utf-8")
    (session / "logs" / "state.csv").write_text("time,agent,goal,x,y\n0.0,Architect,build,1,1\n", encoding="utf-8")

    loaded = load_analysis_session(session)
    assert loaded.artifacts.interaction_trace == []
    assert any("interaction_trace.jsonl" in msg for msg in loaded.warnings)
