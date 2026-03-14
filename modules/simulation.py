# File: modules/simulation.py

import math
from modules.agent import Agent
from modules.brain_context import BrainContextBuilder
from modules.brain_provider import BrainBackendConfig, create_brain_provider
from modules.environment import Environment
from modules.logging_tools import SimulationLogger
from modules.metrics import MetricsCollector
from modules.team_knowledge import TeamKnowledgeManager
from modules.construct_mapping import ConstructMapper


class SimulationState:

    SPEED_MULTIPLIERS = {
        "Slow": 0.5,
        "Normal": 1.0,
        "Fast": 2.0
    }

    def __init__(
        self,
        agent_configs=None,
        num_runs=1,
        speed="Normal",
        experiment_name=None,
        phases=None,
        flash_mode=False,
        project_root=None,
        brain_backend="rule_brain",
        brain_backend_options=None,
    ):
        self.environment = Environment(phases=phases)
        self.agents = []
        self.num_runs = num_runs
        self.flash_mode = flash_mode
        self.time = 0.0
        self.logger = SimulationLogger(experiment_name=experiment_name or "experiment", project_root=project_root)
        self.team_knowledge_manager = TeamKnowledgeManager()
        self.brain_context_builder = BrainContextBuilder()
        backend_options = brain_backend_options or {}
        self.brain_backend_config = BrainBackendConfig(backend=brain_backend, **backend_options)
        self.brain_provider = create_brain_provider(self.brain_backend_config)
        self.logger.log_event(
            self.time,
            "brain_backend_selected",
            {"backend": self.brain_backend_config.backend, "provider_class": self.brain_provider.__class__.__name__},
        )
        self.save_interval = 10.0
        self._last_save_time = 0.0
        self.construct_mapper = ConstructMapper()
        if self.construct_mapper.validation_issues:
            self.logger.log_event(self.time, "construct_mapping_validation_issues", {"issues": self.construct_mapper.validation_issues})
        self.logger.log_event(
            self.time,
            "construct_mapping_loaded",
            {
                "construct_count": len(self.construct_mapper.constructs),
                "construct_to_mechanism_rows": len(self.construct_mapper.construct_to_mechanism),
                "mechanism_to_hook_rows": len(self.construct_mapper.mechanism_to_hook),
            },
        )

        # Determine speed multiplier
        if isinstance(speed, (float, int)):
            self.speed_multiplier = float(speed)
        else:
            self.speed_multiplier = self.SPEED_MULTIPLIERS.get(speed, 1.0)

        if agent_configs is None:
            agent_configs = [
                {"name": "Architect", "role": "Architect", "traits": {}, "packet_access": ["Team_Packet", "Architect_Packet"]},
                {"name": "Engineer", "role": "Engineer", "traits": {}, "packet_access": ["Team_Packet", "Engineer_Packet"]},
                {"name": "Botanist", "role": "Botanist", "traits": {}, "packet_access": ["Team_Packet", "Botanist_Packet"]},
            ]


        for config in agent_configs:
            position = self.environment.get_spawn_point(config["role"])
            agent = Agent(
                name=config["name"],
                role=config["role"],
                position=position
            )
            incoming_traits = dict(config.get("traits", {}))
            construct_values = dict(config.get("constructs", {}))
            mechanism_overrides = dict(config.get("mechanism_overrides", incoming_traits))
            resolved_constructs, resolved_mechanisms, resolved_hooks = self.construct_mapper.resolve_agent_profile(
                construct_values=construct_values,
                mechanism_overrides=mechanism_overrides,
            )
            agent.construct_values = resolved_constructs
            agent.mechanism_profile = resolved_mechanisms
            agent.hook_effects = resolved_hooks
            for mechanism, value in resolved_mechanisms.items():
                setattr(agent, mechanism, value)
            self.logger.log_event(
                self.time,
                "agent_construct_profile",
                {"agent": agent.name, "constructs": resolved_constructs},
            )
            self.logger.log_event(
                self.time,
                "agent_mechanism_profile",
                {"agent": agent.name, "mechanisms": resolved_mechanisms},
            )
            agent.allowed_packet = config["packet_access"]
            self.agents.append(agent)

        self.environment.agents = self.agents
        self.metrics = MetricsCollector(self)
        self.logger.register_event_listener(self.metrics.on_event)
        self.logger.initialize_session_outputs(
            speed=speed,
            flash_mode=self.flash_mode,
            active_agents=[{"name": agent.name, "role": agent.role} for agent in self.agents],
        )
        self.logger.log_event(
            self.time,
            "session_initialized",
            {
                "session_folder": str(self.logger.output_session.session_folder),
                "speed": speed,
                "flash_mode": self.flash_mode,
                "agents": [agent.name for agent in self.agents],
            },
        )

    def update(self, base_dt):
        dt = base_dt * self.speed_multiplier
        self.environment.update(self.time)
        for project in self.environment.construction.projects.values():
            if isinstance(project, dict):
                self.team_knowledge_manager.upsert_construction_artifact(project, self.time)

        for i, agent in enumerate(self.agents):
            for j in range(i + 1, len(self.agents)):
                other = self.agents[j]
                if self._distance(agent.position, other.position) < 1.5:
                    agent.communicate_with(other, sim_state=self)

        for agent in self.agents:
            agent.current_time = self.time
            agent.update(dt, self.environment, sim_state=self)
            agent.compare_and_repair_construction(self.environment.construction, sim_state=self)
            self.logger.log_agent_state(self.time, agent)

        self.metrics.on_step(dt)

        self.time += dt

        if self.flash_mode or (self.time - self._last_save_time >= self.save_interval):
            self.logger.save_csv()
            self._last_save_time = self.time

    def stop(self):
        self.metrics.finalize()
        self.logger.save_csv()


    def _distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)
