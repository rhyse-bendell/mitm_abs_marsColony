import csv
import json
from pathlib import Path

from scripts.check_run_consistency import check_run_consistency


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_events(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "event_type", "payload"])
        writer.writeheader()
        writer.writerows(rows)


def _base_session(tmp_path: Path) -> Path:
    session = tmp_path / "Outputs" / "consistency_20260101"
    _write_json(session / "measures" / "run_summary.json", {"run_metadata": {"fallback_count": 0, "fallback_occurred": False}, "process": {"planner_responsiveness": {"requests_completed_with_fallback": 0}, "construction_event_counts": {"construction_progress_updated": 0, "construction_resource_delivered": 0, "construction_completed": 0}, "readiness_world_state_alignment": {"execution_readiness_failed_count": 0}}})
    _write_json(session / "measures" / "team_summary.json", {"backend": {"fallback_count": 0}, "events": {"construction_progress_updated": 0, "construction_resource_delivered": 0}})
    _write_json(
        session / "measures" / "runtime_witness_coverage.json",
        {
            "critical_targets": [],
            "summary": {"witness_step_failures_by_category": {}},
        },
    )
    _write_jsonl(session / "logs" / "planner_trace.jsonl", [])
    _write_jsonl(session / "logs" / "interaction_trace.jsonl", [])
    return session


def test_source_access_contradiction_detected(tmp_path):
    session = _base_session(tmp_path)
    _write_events(
        session / "logs" / "events.csv",
        [
            {"time": "1.0", "event_type": "source_access_succeeded", "payload": json.dumps({"agent": "Architect", "source_id": "Team_Info"})},
        ],
    )
    _write_json(
        session / "measures" / "runtime_witness_coverage.json",
        {
            "critical_targets": [
                {
                    "target_id": "goal:G",
                    "failure_category": "source_not_accessed",
                    "ordered_witness_steps": [
                        {"raw_step": "source_access:Team_Info", "status": "blocked", "blocked_by": "Architect"}
                    ],
                }
            ],
            "summary": {"witness_step_failures_by_category": {"source_not_accessed": 1}},
        },
    )

    report = check_run_consistency(session)
    assert report["counts_by_type"].get("source_access_contradiction", 0) == 1


def test_role_packet_dik_acquisition_contradiction_detected(tmp_path):
    session = _base_session(tmp_path)
    _write_events(
        session / "logs" / "events.csv",
        [
            {"time": "1.5", "event_type": "dik_acquired_from_inspect", "payload": json.dumps({"agent": "Engineer", "source_id": "Engineer_Info"})},
        ],
    )
    _write_json(
        session / "measures" / "runtime_witness_coverage.json",
        {
            "critical_targets": [
                {
                    "target_id": "rule:R",
                    "failure_category": "inspect_not_completed",
                    "ordered_witness_steps": [
                        {"raw_step": "source_access:Engineer_Info", "status": "pending", "blocked_by": "Engineer"}
                    ],
                }
            ],
            "summary": {"witness_step_failures_by_category": {}},
        },
    )

    report = check_run_consistency(session)
    assert report["counts_by_type"].get("role_packet_acquisition_contradiction", 0) == 1


def test_productive_execution_contradiction_detected(tmp_path):
    session = _base_session(tmp_path)
    _write_events(
        session / "logs" / "events.csv",
        [
            {"time": "2.0", "event_type": "construction_progress_updated", "payload": json.dumps({"agent": "Botanist", "project_id": "P1"})},
        ],
    )

    report = check_run_consistency(session)
    assert report["counts_by_type"].get("productive_execution_contradiction", 0) == 1


def test_fallback_backend_contradiction_detected(tmp_path):
    session = _base_session(tmp_path)
    _write_events(session / "logs" / "events.csv", [{"time": "0.2", "event_type": "brain_provider_fallback", "payload": json.dumps({"agent": "Architect"})}])
    _write_jsonl(
        session / "logs" / "planner_trace.jsonl",
        [{"agent": "Architect", "sim_time": 0.2, "fallback_used": True, "result_source": "fallback_safe_policy"}],
    )

    report = check_run_consistency(session)
    assert report["counts_by_type"].get("fallback_backend_contradiction", 0) == 1
