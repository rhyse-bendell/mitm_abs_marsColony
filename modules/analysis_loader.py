from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.analysis_models import AnalysisSession, SessionArtifacts


OPTIONAL_PATHS = {
    "session_manifest": Path("session_manifest.json"),
    "run_summary": Path("measures/run_summary.json"),
    "team_summary": Path("measures/team_summary.json"),
    "phase_summary": Path("measures/phase_summary.json"),
    "runtime_witness_coverage": Path("measures/runtime_witness_coverage.json"),
    "startup_llm_sanity": Path("measures/startup_llm_sanity.json"),
    "agent_summary": Path("measures/agent_summary.csv"),
    "events": Path("logs/events.csv"),
    "planner_trace": Path("logs/planner_trace.jsonl"),
    "movement": Path("logs/movement.csv"),
    "clock": Path("logs/clock.csv"),
}


def _safe_json(path: Path, warnings: List[str]) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # tolerant read
        warnings.append(f"Failed to parse JSON: {path.name} ({exc})")
        return None


def _safe_csv(path: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        warnings.append(f"Failed to parse CSV: {path.name} ({exc})")
        return []


def _safe_jsonl(path: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                rows.append(json.loads(text))
    except Exception as exc:
        warnings.append(f"Failed to parse JSONL: {path.name} ({exc})")
    return rows


def _load_state_rows(session_dir: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    logs_dir = session_dir / "logs"
    if not logs_dir.exists():
        return []
    state_files = [p for p in logs_dir.glob("*.csv") if p.name != "events.csv"]
    if not state_files:
        return []
    state_file = sorted(state_files)[0]
    return _safe_csv(state_file, warnings)


def load_analysis_session(session_dir: Path | str) -> AnalysisSession:
    session_path = Path(session_dir)
    warnings: List[str] = []
    if not session_path.exists() or not session_path.is_dir():
        raise FileNotFoundError(f"Session folder not found: {session_path}")

    files_found: Dict[str, Optional[Path]] = {}
    for key, rel in OPTIONAL_PATHS.items():
        path = session_path / rel
        files_found[key] = path if path.exists() else None

    artifacts = SessionArtifacts(
        session_manifest=_safe_json(session_path / OPTIONAL_PATHS["session_manifest"], warnings),
        run_summary=_safe_json(session_path / OPTIONAL_PATHS["run_summary"], warnings),
        team_summary=_safe_json(session_path / OPTIONAL_PATHS["team_summary"], warnings),
        phase_summary=_safe_json(session_path / OPTIONAL_PATHS["phase_summary"], warnings) or [],
        runtime_witness_coverage=_safe_json(session_path / OPTIONAL_PATHS["runtime_witness_coverage"], warnings),
        startup_llm_sanity=_safe_json(session_path / OPTIONAL_PATHS["startup_llm_sanity"], warnings),
        agent_summary_rows=_safe_csv(session_path / OPTIONAL_PATHS["agent_summary"], warnings),
        events=_safe_csv(session_path / OPTIONAL_PATHS["events"], warnings),
        state_rows=_load_state_rows(session_path, warnings),
        planner_trace=_safe_jsonl(session_path / OPTIONAL_PATHS["planner_trace"], warnings),
        movement_rows=_safe_csv(session_path / OPTIONAL_PATHS["movement"], warnings),
        clock_rows=_safe_csv(session_path / OPTIONAL_PATHS["clock"], warnings),
    )

    for key, rel in OPTIONAL_PATHS.items():
        if not (session_path / rel).exists():
            warnings.append(f"Optional artifact missing: {rel}")

    return AnalysisSession(session_dir=session_path, artifacts=artifacts, warnings=warnings, files_found=files_found)
