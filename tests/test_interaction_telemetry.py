import json
from pathlib import Path

from modules.analysis_loader import load_analysis_session
from modules.interaction_graph import (
    CANONICAL_INTERACTION_TYPES,
    CANONICAL_NODES,
    InteractionTelemetryBridge,
    InteractionTraceWriter,
    make_interaction_event,
)
from modules.replay_engine import ReplayEngine


class DummyLogger:
    def __init__(self):
        self.rows = []

    def log_interaction(self, row):
        self.rows.append(row)


def test_interaction_event_helper_schema():
    row = make_interaction_event(
        time=1.23,
        interaction_id="itx-1",
        source_node="Agent:Architect",
        target_node="Source:TeamInfo",
        interaction_type="source_access",
        status="started",
        agent_id="Architect",
        payload_summary="inspect Team_Info",
    )
    assert row["time"] == 1.23
    assert row["interaction_type"] in CANONICAL_INTERACTION_TYPES
    assert row["source_node"] == "Agent:Architect"


def test_interaction_trace_writer(tmp_path: Path):
    path = tmp_path / "logs" / "interaction_trace.jsonl"
    writer = InteractionTraceWriter(path)
    writer.append({"time": 0.0, "interaction_type": "metric_emit"})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["interaction_type"] == "metric_emit"


def test_canonical_node_mapping_consistency():
    ids = [n.node_id for n in CANONICAL_NODES]
    assert len(ids) == len(set(ids))
    assert "Agent:Architect" in ids
    assert "Audit:RuntimeWitness" in ids


def test_interaction_loader_and_replay_support(tmp_path: Path):
    session = tmp_path / "session"
    logs = session / "logs"
    logs.mkdir(parents=True)
    (logs / "events.csv").write_text("time,event_type,payload\n", encoding="utf-8")
    (logs / "run.csv").write_text("time,agent,x,y,goal\n0.0,Architect,0,0,seek_info\n", encoding="utf-8")
    (logs / "interaction_trace.jsonl").write_text(
        json.dumps({
            "time": 0.0,
            "interaction_id": "itx-0",
            "source_node": "Agent:Architect",
            "target_node": "Planner:OllamaQwen",
            "interaction_type": "planner_request",
            "status": "started",
            "agent_id": "Architect",
            "payload_summary": "planner",
        }) + "\n",
        encoding="utf-8",
    )
    loaded = load_analysis_session(session)
    assert len(loaded.artifacts.interaction_trace) == 1
    engine = ReplayEngine(loaded)
    assert engine.frames
    assert engine.frames[0].interaction_events_at_time


def test_missing_interaction_trace_graceful(tmp_path: Path):
    session = tmp_path / "session2"
    logs = session / "logs"
    logs.mkdir(parents=True)
    (logs / "events.csv").write_text("time,event_type,payload\n", encoding="utf-8")
    (logs / "run.csv").write_text("time,agent,x,y,goal\n", encoding="utf-8")
    loaded = load_analysis_session(session)
    assert loaded.artifacts.interaction_trace == []
    assert any("interaction_trace.jsonl" in w for w in loaded.warnings)


def test_bridge_maps_core_events():
    logger = DummyLogger()
    bridge = InteractionTelemetryBridge(logger)
    bridge.on_event({"time": 0.1, "event_type": "inspect_started", "payload_data": {"agent": "Architect", "source_id": "Team_Info", "agent_id": "Architect"}})
    bridge.on_event({"time": 0.2, "event_type": "brain_provider_fallback", "payload_data": {"agent": "Architect", "reason": "timeout"}})
    assert len(logger.rows) == 2
    assert logger.rows[0]["interaction_type"] == "source_access"
    assert logger.rows[1]["interaction_type"] == "fallback_activation"
