from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionArtifacts:
    session_manifest: Optional[Dict[str, Any]] = None
    run_summary: Optional[Dict[str, Any]] = None
    team_summary: Optional[Dict[str, Any]] = None
    phase_summary: Optional[List[Dict[str, Any]]] = None
    runtime_witness_coverage: Optional[Dict[str, Any]] = None
    startup_llm_sanity: Optional[Dict[str, Any]] = None
    agent_summary_rows: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    state_rows: List[Dict[str, Any]] = field(default_factory=list)
    planner_trace: List[Dict[str, Any]] = field(default_factory=list)
    interaction_trace: List[Dict[str, Any]] = field(default_factory=list)
    movement_rows: List[Dict[str, Any]] = field(default_factory=list)
    clock_rows: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisSession:
    session_dir: Path
    artifacts: SessionArtifacts
    warnings: List[str] = field(default_factory=list)
    files_found: Dict[str, Optional[Path]] = field(default_factory=dict)

    @property
    def agent_names(self) -> List[str]:
        names = {str(r.get("agent", "")) for r in self.artifacts.state_rows if r.get("agent")}
        if names:
            return sorted(names)
        return sorted({str(r.get("agent", "")) for r in self.artifacts.events if r.get("agent")})
