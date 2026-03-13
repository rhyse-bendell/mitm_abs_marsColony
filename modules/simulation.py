# File: modules/simulation.py

import math
from modules.agent import Agent
from modules.environment import Environment
from modules.logging_tools import SimulationLogger


class SimulationState:

    SPEED_MULTIPLIERS = {
        "Slow": 0.5,
        "Normal": 1.0,
        "Fast": 2.0
    }

    def __init__(self, agent_configs=None, num_runs=1, speed="Normal", experiment_name=None, phases=None, flash_mode=False):
        self.environment = Environment(phases=phases)
        self.agents = []
        self.num_runs = num_runs
        self.flash_mode = flash_mode
        self.time = 0.0
        self.logger = SimulationLogger(experiment_name=experiment_name or "experiment")
        self.save_interval = 10.0
        self._last_save_time = 0.0

        # Determine speed multiplier
        if isinstance(speed, (float, int)):
            self.speed_multiplier = float(speed)
        else:
            self.speed_multiplier = self.SPEED_MULTIPLIERS.get(speed, 1.0)

        if agent_configs is None:
            agent_configs = [
                {"name": "Architect", "role": "Architect", "traits": {}, "packet_access": "Team_Packet"},
                {"name": "Engineer", "role": "Engineer", "traits": {}, "packet_access": "Team_Packet"},
                {"name": "Botanist", "role": "Botanist", "traits": {}, "packet_access": "Team_Packet"},
            ]


        for config in agent_configs:
            position = self.environment.get_spawn_point(config["role"])
            agent = Agent(
                name=config["name"],
                role=config["role"],
                position=position
            )
            for trait, value in config["traits"].items():
                setattr(agent, trait, value)
            agent.allowed_packet = config["packet_access"]
            self.agents.append(agent)

        self.environment.agents = self.agents

    def update(self, base_dt):
        dt = base_dt * self.speed_multiplier
        self.environment.update(self.time)

        for i, agent in enumerate(self.agents):
            for j in range(i + 1, len(self.agents)):
                other = self.agents[j]
                if self._distance(agent.position, other.position) < 1.5:
                    agent.communicate_with(other)

        for agent in self.agents:
            agent.current_time = self.time
            agent.update(dt, self.environment)
            agent.compare_and_repair_construction(self.environment.construction)
            self.logger.log_agent_state(self.time, agent)

        self.time += dt

        if self.flash_mode or (self.time - self._last_save_time >= self.save_interval):
            self.logger.save_csv()
            self._last_save_time = self.time

    def stop(self):
        self.logger.save_csv()


    def _distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)
