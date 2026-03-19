#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

CANONICAL_FILES = ["logs/events.csv", "logs/planner_trace.jsonl", "logs/interaction_trace.jsonl"]
DERIVED_FILES = ["measures/runtime_witness_coverage.json", "measures/run_summary.json", "measures/team_summary.json"]
ROLE_SOURCE_IDS = {"Architect_Info", "Engineer_Info", "Botanist_Info"}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _read_events(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = {}
            raw = row.get("payload")
            if raw:
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
            rows.append(
                {
                    "time": float(row.get("time") or 0.0),
                    "event_type": row.get("event_type") or "",
                    "payload": payload,
                }
            )
    return rows


def _paths_for_root(root: Path) -> Dict[str, Path]:
    return {
        "events": root / "logs" / "events.csv",
        "planner_trace": root / "logs" / "planner_trace.jsonl",
        "interaction_trace": root / "logs" / "interaction_trace.jsonl",
        "witness": root / "measures" / "runtime_witness_coverage.json",
        "run_summary": root / "measures" / "run_summary.json",
        "team_summary": root / "measures" / "team_summary.json",
    }


def _normalize_run_dir(path: Path) -> Path:
    if (path / "logs").exists() and (path / "measures").exists():
        return path
    if (path / "Outputs").exists():
        sessions = sorted([p for p in (path / "Outputs").iterdir() if p.is_dir()])
        if sessions:
            return sessions[-1]
    return path


def _add_contradiction(contradictions: List[Dict[str, Any]], ctype: str, severity: str, canonical: Dict[str, Any], derived: Dict[str, Any], agent: str | None = None, source_id: str | None = None):
    row: Dict[str, Any] = {
        "type": ctype,
        "severity": severity,
        "canonical_evidence": canonical,
        "derived_evidence": derived,
    }
    if agent:
        row["agent"] = agent
    if source_id:
        row["source_id"] = source_id
    contradictions.append(row)


def _witness_source_steps(witness: Dict[str, Any]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for target in witness.get("critical_targets", []):
        failure_category = target.get("failure_category")
        target_id = target.get("target_id")
        for step in target.get("ordered_witness_steps", []):
            raw = str(step.get("raw_step") or "")
            if not raw.startswith("source_access:"):
                continue
            source_id = raw.split(":", 1)[1]
            agent = step.get("blocked_by") or step.get("completed_by")
            index[(str(agent or ""), source_id)].append(
                {
                    "target_id": target_id,
                    "step_status": step.get("status"),
                    "failure_category": failure_category,
                    "step": step,
                }
            )
            index[("", source_id)].append(
                {
                    "target_id": target_id,
                    "step_status": step.get("status"),
                    "failure_category": failure_category,
                    "step": step,
                }
            )
    return index


def check_run_consistency(run_dir: Path) -> Dict[str, Any]:
    run_dir = _normalize_run_dir(run_dir)
    paths = _paths_for_root(run_dir)

    events = _read_events(paths["events"])
    planner_trace = _read_jsonl(paths["planner_trace"])
    interaction_trace = _read_jsonl(paths["interaction_trace"])
    witness = _read_json(paths["witness"])
    run_summary = _read_json(paths["run_summary"])
    team_summary = _read_json(paths["team_summary"])

    contradictions: List[Dict[str, Any]] = []
    witness_steps = _witness_source_steps(witness)

    # Canonical facts
    source_successes: List[Dict[str, Any]] = []
    role_source_dik: List[Dict[str, Any]] = []
    derivation_or_rule: List[Dict[str, Any]] = []
    construction_progress_events: List[Dict[str, Any]] = []
    fallback_canonical_events: List[Dict[str, Any]] = []

    for e in events:
        etype = e["event_type"]
        payload = e["payload"]
        if etype in {"source_access_succeeded", "shared_source_access_success", "inspect_completed"}:
            sid = str(payload.get("source_id") or "")
            if sid:
                source_successes.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent"), "source_id": sid})

        if etype == "dik_acquired_from_inspect":
            sid = str(payload.get("source_id") or "")
            if sid in ROLE_SOURCE_IDS:
                role_source_dik.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent"), "source_id": sid})
        if etype == "source_access_succeeded":
            sid = str(payload.get("source_id") or "")
            gained = bool(payload.get("new_information_ids") or payload.get("new_data_ids") or payload.get("team_dik_added_ids"))
            if sid in ROLE_SOURCE_IDS and gained:
                role_source_dik.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent"), "source_id": sid, "gained": True})

        if etype in {"derivation_succeeded", "rule_adopted", "inspect_success_rule_adopted", "execution_readiness_passed"}:
            derivation_or_rule.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent")})

        if etype in {"construction_resource_delivered", "construction_progress_updated", "construction_completed"}:
            construction_progress_events.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent"), "project_id": payload.get("project_id")})

        if etype in {"brain_provider_fallback", "ui_safe_fallback_used"}:
            fallback_canonical_events.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent")})
        if etype == "brain_decision_outcome" and payload.get("fallback_used"):
            fallback_canonical_events.append({"event_type": etype, "time": e["time"], "agent": payload.get("agent"), "fallback_used": True})

    for row in planner_trace:
        if row.get("fallback_used"):
            fallback_canonical_events.append(
                {
                    "event_type": "planner_trace_fallback_used",
                    "time": row.get("sim_time"),
                    "agent": row.get("agent") or row.get("agent_id"),
                    "result_source": row.get("result_source"),
                }
            )

    for row in interaction_trace:
        if bool(row.get("fallback_used")):
            fallback_canonical_events.append(
                {
                    "event_type": "interaction_trace_fallback_used",
                    "time": row.get("time") or row.get("sim_time"),
                    "agent": row.get("agent") or row.get("agent_id"),
                }
            )

    # A + B: source and role packet contradictions against witness steps
    for fact in source_successes:
        sid = str(fact.get("source_id") or "")
        agent = str(fact.get("agent") or "")
        related = witness_steps.get((agent, sid), []) or witness_steps.get(("", sid), [])
        blocked = [
            s
            for s in related
            if s.get("step_status") in {"pending", "blocked"}
            or str(s.get("failure_category") or "") in {"source_not_accessed", "inspect_not_started", "inspect_not_completed"}
        ]
        if blocked:
            _add_contradiction(
                contradictions,
                "source_access_contradiction",
                "high",
                canonical={"source_access_success": fact},
                derived={"witness_source_steps": blocked[:4]},
                agent=agent or None,
                source_id=sid,
            )

    for fact in role_source_dik:
        sid = str(fact.get("source_id") or "")
        agent = str(fact.get("agent") or "")
        related = witness_steps.get((agent, sid), []) or witness_steps.get(("", sid), [])
        blocked = [s for s in related if s.get("step_status") in {"pending", "blocked"}]
        if blocked:
            _add_contradiction(
                contradictions,
                "role_packet_acquisition_contradiction",
                "high",
                canonical={"role_packet_dik_acquired": fact},
                derived={"witness_role_source_steps": blocked[:4]},
                agent=agent or None,
                source_id=sid,
            )

    # C: derivation/readiness contradiction
    readiness_failed_count = int(((run_summary.get("process") or {}).get("readiness_world_state_alignment") or {}).get("execution_readiness_failed_count", 0) or 0)
    witness_failure_counts = ((witness.get("summary") or {}).get("witness_step_failures_by_category") or {})
    readiness_blocked_in_witness = sum(
        int(v)
        for k, v in witness_failure_counts.items()
        if "readiness" in str(k)
    )
    if derivation_or_rule and (readiness_failed_count > 0 or readiness_blocked_in_witness > 0):
        _add_contradiction(
            contradictions,
            "derivation_readiness_contradiction",
            "medium",
            canonical={"derivation_or_rule_events": derivation_or_rule[:8], "count": len(derivation_or_rule)},
            derived={
                "run_summary_readiness_failed_count": readiness_failed_count,
                "witness_readiness_failure_count": readiness_blocked_in_witness,
                "witness_failure_categories": {k: v for k, v in witness_failure_counts.items() if "readiness" in str(k)},
            },
        )

    # D: productive execution contradiction
    if construction_progress_events:
        canonical_progress_count = len(construction_progress_events)
        summary_progress = int((((run_summary.get("process") or {}).get("construction_event_counts") or {}).get("construction_progress_updated", 0) or 0)
                               + (((run_summary.get("process") or {}).get("construction_event_counts") or {}).get("construction_resource_delivered", 0) or 0)
                               + (((run_summary.get("process") or {}).get("construction_event_counts") or {}).get("construction_completed", 0) or 0))
        team_events = team_summary.get("events") or {}
        team_progress = int(team_events.get("construction_progress_updated", 0) or 0) + int(team_events.get("construction_resource_delivered", 0) or 0)
        if summary_progress == 0 or team_progress == 0:
            _add_contradiction(
                contradictions,
                "productive_execution_contradiction",
                "high",
                canonical={"construction_progress_events_count": canonical_progress_count, "examples": construction_progress_events[:8]},
                derived={"run_summary_progress_count": summary_progress, "team_summary_progress_count": team_progress},
            )

    # E: fallback/backend contradiction
    if fallback_canonical_events:
        rs_meta = run_summary.get("run_metadata") or {}
        planner = (run_summary.get("process") or {}).get("planner_responsiveness") or {}
        team_backend = (team_summary.get("backend") or {})
        derived_fallback_count = int(rs_meta.get("fallback_count", 0) or 0) + int(planner.get("requests_completed_with_fallback", 0) or 0) + int(team_backend.get("fallback_count", 0) or 0)
        if derived_fallback_count == 0:
            _add_contradiction(
                contradictions,
                "fallback_backend_contradiction",
                "medium",
                canonical={"fallback_events_count": len(fallback_canonical_events), "examples": fallback_canonical_events[:8]},
                derived={
                    "run_summary_fallback_count": rs_meta.get("fallback_count", 0),
                    "run_summary_requests_completed_with_fallback": planner.get("requests_completed_with_fallback", 0),
                    "team_summary_fallback_count": team_backend.get("fallback_count", 0),
                    "run_summary_fallback_occurred": rs_meta.get("fallback_occurred"),
                },
            )

    counts = Counter(c["type"] for c in contradictions)
    result = {
        "run_dir": str(run_dir),
        "canonical_sources_used": CANONICAL_FILES,
        "derived_views_checked": DERIVED_FILES,
        "contradictions": contradictions,
        "counts_by_type": dict(counts),
        "canonical_fact_counts": {
            "source_access_successes": len(source_successes),
            "role_source_dik_acquisitions": len(role_source_dik),
            "derivation_or_rule_events": len(derivation_or_rule),
            "construction_progress_events": len(construction_progress_events),
            "fallback_events": len(fallback_canonical_events),
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check consistency between canonical run history and derived summaries.")
    parser.add_argument("run_dir", type=Path, help="Run/session directory (containing logs/ and measures/) or repo root containing Outputs/")
    parser.add_argument("--output", type=Path, default=None, help="Optional output report path. Defaults to <run_dir>/artifacts/run_consistency_report.json")
    args = parser.parse_args()

    report = check_run_consistency(args.run_dir)
    run_dir = Path(report["run_dir"])
    output_path = args.output or (run_dir / "artifacts" / "run_consistency_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Run: {report['run_dir']}")
    print(f"Canonical sources: {', '.join(report['canonical_sources_used'])}")
    print(f"Derived views: {', '.join(report['derived_views_checked'])}")
    print(f"Contradictions found: {len(report['contradictions'])}")
    for ctype, count in sorted(report["counts_by_type"].items()):
        print(f"  - {ctype}: {count}")
    print(f"Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
