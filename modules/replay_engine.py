from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from modules.analysis_models import AnalysisSession


@dataclass
class ReplayFrame:
    index: int
    time: float
    agent_states: Dict[str, Dict[str, Any]]
    events_at_time: List[Dict[str, Any]]
    construction_state: Dict[str, Any]


class ReplayEngine:
    def __init__(self, session: AnalysisSession):
        self.session = session
        self.frames: List[ReplayFrame] = self._build_frames()

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _parse_payload(row: Dict[str, Any]) -> Dict[str, Any]:
        payload = row.get("payload")
        if isinstance(payload, dict):
            return payload
        if not payload:
            return {}
        try:
            return json.loads(payload)
        except Exception:
            return {}

    def _build_frames(self) -> List[ReplayFrame]:
        state_rows = list(self.session.artifacts.state_rows)
        events = list(self.session.artifacts.events)
        state_rows.sort(key=lambda r: self._to_float(r.get("time")))
        events.sort(key=lambda r: self._to_float(r.get("time")))

        times = sorted(
            {
                *[round(self._to_float(r.get("time")), 3) for r in state_rows],
                *[round(self._to_float(r.get("time")), 3) for r in events],
            }
        )
        if not times:
            return []

        frames: List[ReplayFrame] = []
        latest_agent_state: Dict[str, Dict[str, Any]] = {}
        construction_state: Dict[str, Any] = {}
        state_idx = 0
        event_idx = 0

        for idx, t in enumerate(times):
            while state_idx < len(state_rows) and round(self._to_float(state_rows[state_idx].get("time")), 3) <= t:
                row = state_rows[state_idx]
                agent = str(row.get("agent") or row.get("display_name") or "unknown")
                latest_agent_state[agent] = row
                state_idx += 1

            events_at_time: List[Dict[str, Any]] = []
            while event_idx < len(events) and round(self._to_float(events[event_idx].get("time")), 3) <= t:
                event = events[event_idx]
                event_time = round(self._to_float(event.get("time")), 3)
                if event_time == t:
                    events_at_time.append(event)
                if event.get("event_type") == "construction_externalization_update":
                    payload = self._parse_payload(event)
                    key = str(payload.get("project_id") or payload.get("project") or f"unknown_{event_idx}")
                    construction_state[key] = payload
                event_idx += 1

            frames.append(
                ReplayFrame(
                    index=idx,
                    time=t,
                    agent_states=dict(latest_agent_state),
                    events_at_time=events_at_time,
                    construction_state=dict(construction_state),
                )
            )
        return frames

    def important_events(self) -> List[Dict[str, Any]]:
        candidates = {
            "phase_changed",
            "phase_transition",
            "brain_provider_fallback",
            "witness_expectation_failed",
            "witness_expectation_recovered",
            "planner_timeout",
            "movement_blocked",
        }
        return [e for e in self.session.artifacts.events if str(e.get("event_type")) in candidates]
