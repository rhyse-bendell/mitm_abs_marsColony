# File: modules/logging_tools.py

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from modules.interaction_graph import build_interaction_from_sim_event


class OutputSessionManager:
    def __init__(self, experiment_name="experiment", timestamp=None, project_root=None):
        self.experiment_name = experiment_name or "experiment"
        self.sanitized_prefix = self._sanitize_prefix(self.experiment_name)
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.outputs_root = self.project_root / "Outputs"
        self.session_folder = self.outputs_root / f"{self.sanitized_prefix}_{self.timestamp}"

        self.logs_dir = self.session_folder / "logs"
        self.measures_dir = self.session_folder / "measures"
        self.snapshots_dir = self.session_folder / "snapshots"

    @staticmethod
    def _sanitize_prefix(experiment_name):
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", experiment_name.strip())
        safe_name = safe_name.strip("_")
        return safe_name or "experiment"

    def setup_session_dirs(self):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.measures_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    def build_log_path(self, filename):
        return self.logs_dir / filename

    def build_measure_path(self, filename):
        return self.measures_dir / filename

    def write_manifest(self, speed=None, flash_mode=None, active_agents=None, extra_metadata=None):
        manifest = {
            "experiment_name": self.experiment_name,
            "sanitized_prefix": self.sanitized_prefix,
            "timestamp": self.timestamp,
            "session_folder": str(self.session_folder),
            "speed": speed,
            "flash_mode": flash_mode,
            "active_agents": active_agents or []
        }
        if extra_metadata:
            manifest.update(dict(extra_metadata))
        manifest_path = self.session_folder / "session_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def ensure_measures_placeholder(self):
        placeholder = self.build_measure_path("final_measures_placeholder.json")
        if placeholder.exists():
            return
        with placeholder.open("w", encoding="utf-8") as f:
            json.dump({"status": "placeholder"}, f)

class PlannerTraceWriter:
    def __init__(self, output_session, enabled=True, mode="full", max_chars=12000):
        self.output_session = output_session
        self.enabled = bool(enabled)
        mode = str(mode or "summary").lower()
        self.mode = mode if mode in {"summary", "full"} else "summary"
        self.max_chars = max(200, int(max_chars or 12000))
        self.trace_path = self.output_session.build_log_path("planner_trace.jsonl")

    def _truncate_text(self, value):
        if value is None:
            return None
        text = str(value)
        if len(text) <= self.max_chars:
            return text
        return text[: self.max_chars] + f"...<truncated {len(text)-self.max_chars} chars>"

    def _trim_value(self, value):
        if value is None:
            return None
        if isinstance(value, str):
            return self._truncate_text(value)
        encoded = json.dumps(value, default=str)
        if len(encoded) <= self.max_chars:
            return value
        return self._truncate_text(encoded)

    def _sanitize_attempt(self, attempt):
        item = dict(attempt or {})
        if self.mode == "summary":
            item.pop("raw_http_response_text", None)
            item.pop("parsed_response_json", None)
            item.pop("extracted_response_payload", None)
        else:
            if "raw_http_response_text" in item:
                item["raw_http_response_text"] = self._truncate_text(item.get("raw_http_response_text"))
            if "parsed_response_json" in item:
                item["parsed_response_json"] = self._trim_value(item.get("parsed_response_json"))
            if "extracted_response_payload" in item:
                item["extracted_response_payload"] = self._trim_value(item.get("extracted_response_payload"))
        if "exception" in item:
            item["exception"] = self._trim_value(item.get("exception"))
        return item

    def _sanitize(self, payload):
        cleaned = dict(payload or {})
        if self.mode == "summary":
            for key in [
                "raw_http_response_text",
                "provider_request_payload",
                "parsed_response_json",
                "extracted_response_payload",
                "normalized_agent_brain_response",
                "agent_brain_request_payload",
                "provider_trace",
            ]:
                cleaned.pop(key, None)
        else:
            for key in [
                "raw_http_response_text",
                "provider_request_payload",
                "parsed_response_json",
                "extracted_response_payload",
                "normalized_agent_brain_response",
                "agent_brain_request_payload",
                "provider_trace",
            ]:
                if key in cleaned:
                    cleaned[key] = self._trim_value(cleaned.get(key))
        if "exception" in cleaned:
            cleaned["exception"] = self._trim_value(cleaned.get("exception"))
        attempts = cleaned.get("provider_attempts")
        if isinstance(attempts, list):
            cleaned["provider_attempts"] = [self._sanitize_attempt(a) for a in attempts]
        return cleaned

    def append(self, payload):
        if not self.enabled:
            return
        row = self._sanitize(payload)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


class SimulationLogger:
    def __init__(self, filename=None, interval=5.0, experiment_name="experiment", project_root=None):
        self.output_session = OutputSessionManager(experiment_name=experiment_name, project_root=project_root)
        self.output_session.setup_session_dirs()
        timestamp = self.output_session.timestamp
        safe_name = self.output_session.sanitized_prefix
        if filename is None:
            filename = f"{safe_name}_{timestamp}.csv"

        self.filename = filename
        self.interval = interval
        self.buffer = []
        self.event_buffer = []
        self.recent_events = []
        self.max_recent_events = 300
        self.event_listeners = []
        self.last_dump_time = 0.0
        self.planner_trace_writer = PlannerTraceWriter(self.output_session, enabled=False)
        self.interaction_trace_path = self.output_session.build_log_path("interaction_trace.jsonl")
        self.recent_interactions = []
        self.max_recent_interactions = 300

    def _append_recent_interaction(self, interaction):
        self.recent_interactions.append(interaction)
        if len(self.recent_interactions) > self.max_recent_interactions:
            self.recent_interactions = self.recent_interactions[-self.max_recent_interactions:]

    def get_recent_interactions(self, count=80):
        return self.recent_interactions[-count:]

    def _append_interaction_trace(self, interaction):
        self.interaction_trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.interaction_trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(interaction, default=str) + "\n")

    def _append_recent_event(self, event):
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]

    def log_agent_state(self, time, agent):
        self.buffer.append({
            "time": round(time, 2),
            "agent": agent.name,
            "display_name": getattr(agent, "display_name", agent.name),
            "agent_label": getattr(agent, "agent_label", None),
            "agent_id": getattr(agent, "agent_id", agent.name),
            "role": agent.role,
            "brain_backend": getattr(agent, "brain_config", {}).get("backend"),
            "brain_local_model": getattr(agent, "brain_config", {}).get("local_model"),
            "brain_fallback_backend": getattr(agent, "brain_config", {}).get("fallback_backend"),
            "planner_interval_steps": getattr(getattr(agent, "planner_cadence", None), "planner_interval_steps", None),
            "planner_timeout_seconds": getattr(getattr(agent, "planner_cadence", None), "planner_timeout_seconds", None),
            "planner_degraded_mode": getattr(agent, "planner_state", {}).get("degraded_mode"),
            "x": round(agent.position[0], 2),
            "y": round(agent.position[1], 2),
            "goal": agent.goal or "None",
            "heart_rate": agent.heart_rate,
            "gsr": round(agent.gsr, 4),
            "temperature": round(agent.temperature, 2),
            "co2": round(agent.co2_output, 3),
            "num_data": len(agent.mental_model["data"]),
            "num_info": len(agent.mental_model["information"]),
            "num_rules": len(agent.mental_model["knowledge"].rules),
            "last_action": getattr(agent, "status_last_action", "") or (agent.activity_log[-1] if agent.activity_log else "")
        })

    def maybe_dump(self, current_time):
        if current_time - self.last_dump_time >= self.interval:
            self.save_csv()
            self.last_dump_time = current_time

    def log_event(self, time, event_type, payload):
        payload = dict(payload or {})
        event = {
            "time": round(time, 2),
            "event_type": event_type,
            "payload": json.dumps(payload, default=str),
            "payload_data": payload,
        }
        self.event_buffer.append(event)
        self._append_recent_event(event)
        interaction = build_interaction_from_sim_event(time, event_type, payload)
        if interaction is not None:
            interaction_row = interaction.to_row()
            self._append_interaction_trace(interaction_row)
            self._append_recent_interaction(interaction_row)
        for listener in self.event_listeners:
            listener(event)

    def register_event_listener(self, listener):
        self.event_listeners.append(listener)

    def get_recent_events(self, count=80):
        return self.recent_events[-count:]

    def save_csv(self):
        if not self.buffer:
            return

        save_path = self.output_session.build_log_path(self.filename)
        write_header = not save_path.exists()

        with save_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.buffer[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(self.buffer)

        event_rows = len(self.event_buffer)
        if self.event_buffer:
            event_path = self.output_session.build_log_path("events.csv")
            event_header = not event_path.exists()
            with event_path.open("a", newline="", encoding="utf-8") as f:
                write_rows = [{k: v for k, v in row.items() if k != "payload_data"} for row in self.event_buffer]
                writer = csv.DictWriter(f, fieldnames=write_rows[0].keys())
                if event_header:
                    writer.writeheader()
                writer.writerows(write_rows)
            self.event_buffer = []

        self._append_recent_event(
            {
                "time": self.buffer[-1]["time"],
                "event_type": "outputs_saved",
                "payload": json.dumps({"path": str(save_path), "rows": len(self.buffer), "event_rows": event_rows}),
            }
        )

        self.buffer = []
        print(f"✅ Agent logs saved to {save_path}")

    def configure_planner_trace(self, enabled=True, mode="full", max_chars=12000):
        self.planner_trace_writer = PlannerTraceWriter(
            self.output_session,
            enabled=enabled,
            mode=mode,
            max_chars=max_chars,
        )

    def append_planner_trace(self, payload):
        self.planner_trace_writer.append(payload)

    def initialize_session_outputs(self, speed=None, flash_mode=None, active_agents=None, extra_metadata=None):
        self.output_session.write_manifest(
            speed=speed,
            flash_mode=flash_mode,
            active_agents=active_agents,
            extra_metadata=extra_metadata,
        )
        self.output_session.ensure_measures_placeholder()

    def update_session_manifest(self, extra_metadata=None):
        if not extra_metadata:
            return
        manifest_path = self.output_session.session_folder / "session_manifest.json"
        payload = {}
        if manifest_path.exists():
            with manifest_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        payload.update(dict(extra_metadata))
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
