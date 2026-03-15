# File: modules/logging_tools.py

import csv
import json
import re
from datetime import datetime
from pathlib import Path


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

    def _append_recent_event(self, event):
        self.recent_events.append(event)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events = self.recent_events[-self.max_recent_events:]

    def log_agent_state(self, time, agent):
        self.buffer.append({
            "time": round(time, 2),
            "agent": agent.name,
            "role": agent.role,
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
        event = {
            "time": round(time, 2),
            "event_type": event_type,
            "payload": json.dumps(payload, default=str),
            "payload_data": payload,
        }
        self.event_buffer.append(event)
        self._append_recent_event(event)
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
