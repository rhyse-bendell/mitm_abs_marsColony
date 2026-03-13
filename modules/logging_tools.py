# File: modules/logging_tools.py

import csv
import os
from datetime import datetime

class SimulationLogger:
    def __init__(self, filename=None, interval=5.0, experiment_name="experiment"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = experiment_name.strip().replace(" ", "_") or "experiment"
        if filename is None:
            filename = f"{safe_name}_{timestamp}.csv"

        self.filename = filename
        self.interval = interval
        self.buffer = []
        self.last_dump_time = 0.0

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
            "last_action": agent.activity_log[-1] if agent.activity_log else ""
        })

    def maybe_dump(self, current_time):
        if current_time - self.last_dump_time >= self.interval:
            self.save_csv()
            self.last_dump_time = current_time

    def save_csv(self):
        if not self.buffer:
            return

        save_path = os.path.join(os.getcwd(), self.filename)
        write_header = not os.path.exists(save_path)

        with open(save_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.buffer[0].keys())
            if write_header:
                writer.writeheader()
            writer.writerows(self.buffer)

        self.buffer = []
        print(f"✅ Agent logs saved to {self.filename}")
