from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from modules.analysis_models import AnalysisSession


def aggregate_statistics(session: AnalysisSession) -> Dict[str, Any]:
    artifacts = session.artifacts
    run_summary = artifacts.run_summary or {}
    team_summary = artifacts.team_summary or {}

    events = artifacts.events
    event_counts = Counter(str(e.get("event_type", "unknown")) for e in events)

    return {
        "run_metadata": run_summary.get("run_metadata", {}),
        "backend": (team_summary.get("backend") or run_summary.get("run_metadata", {})),
        "dik": run_summary.get("process", {}).get("dik_acquisition_counts_by_agent", {}),
        "planner": run_summary.get("process", {}).get("planner_responsiveness", {}),
        "witness": artifacts.runtime_witness_coverage or run_summary.get("runtime_witness_coverage", {}),
        "externalization": run_summary.get("externalization_metrics", {}),
        "movement": run_summary.get("movement_path_metrics", run_summary.get("breakdown_metrics", {}).get("movement_path_metrics", {})),
        "startup": run_summary.get("startup_llm_sanity", {}),
        "agent_summary_rows": artifacts.agent_summary_rows,
        "top_event_counts": dict(event_counts.most_common(25)),
    }


def phase_statistics(session: AnalysisSession) -> List[Dict[str, Any]]:
    phase_rows = list(session.artifacts.phase_summary or [])
    if phase_rows:
        return phase_rows

    phases: Dict[str, Dict[str, Any]] = {}
    for event in session.artifacts.events:
        phase = str(event.get("phase") or "unscoped")
        if phase not in phases:
            phases[phase] = {"phase_name": phase, "events": Counter(), "event_count": 0}
        phases[phase]["events"][str(event.get("event_type", "unknown"))] += 1
        phases[phase]["event_count"] += 1

    rows: List[Dict[str, Any]] = []
    for phase, data in phases.items():
        rows.append({
            "phase_name": phase,
            "event_count": data["event_count"],
            "events": dict(data["events"]),
            "duration_seconds": None,
        })
    return rows
