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
from modules.task_model import load_task_model


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
        planner_config=None,
        task_id="mars_colony",
    ):
        self.task_model = load_task_model(task_id=task_id)
        if phases is None and self.task_model.phases:
            phases = [
                {
                    "id": p.phase_id,
                    "name": p.name,
                    "duration_minutes": p.duration_minutes,
                    "colonist_manifest": dict(p.colonist_manifest),
                    "unlocks": list(p.unlocks),
                    "required_structures": dict(p.required_structures),
                    "description": p.description,
                }
                for p in self.task_model.phases
            ]
        self.environment = Environment(phases=phases, task_model=self.task_model)
        self.agents = []
        self.num_runs = num_runs
        self.flash_mode = flash_mode
        self.time = 0.0
        self.logger = SimulationLogger(experiment_name=experiment_name or "experiment", project_root=project_root)
        self.team_knowledge_manager = TeamKnowledgeManager()
        self.brain_context_builder = BrainContextBuilder()
        self.planner_defaults = dict(self.task_model.manifest.get("planner_defaults", {}))
        if planner_config:
            self.planner_defaults.update(dict(planner_config))
        backend_options = dict(brain_backend_options or {})
        if "timeout_s" not in backend_options and "planner_timeout_seconds" in self.planner_defaults:
            backend_options["timeout_s"] = self.planner_defaults.get("planner_timeout_seconds")
        if "max_retries" not in backend_options and "planner_max_retries" in self.planner_defaults:
            backend_options["max_retries"] = self.planner_defaults.get("planner_max_retries")
        if "fallback_backend" not in backend_options and "planner_fallback_backend" in self.planner_defaults:
            backend_options["fallback_backend"] = self.planner_defaults.get("planner_fallback_backend")
        self.brain_backend_config = BrainBackendConfig(backend=brain_backend, **backend_options)
        self.configured_brain_backend = self.brain_backend_config.backend
        self.brain_provider = create_brain_provider(self.brain_backend_config)
        self.effective_brain_backend = self.configured_brain_backend
        self.fallback_occurred = False
        self.backend_fallback_count = 0
        self._last_backend_outcome_signature = None
        self.logger.log_event(
            self.time,
            "task_model_loaded",
            {
                "task_id": self.task_model.task_id,
                "source_count": len(self.task_model.sources),
                "dik_element_count": len(self.task_model.dik_elements),
                "derivation_count": len(self.task_model.derivations),
                "rule_count": len(self.task_model.rules),
                "goal_count": len(self.task_model.goals),
                "plan_method_count": len(self.task_model.plan_methods),
                "artifact_count": len(self.task_model.artifacts),
            },
        )
        self.logger.log_event(
            self.time,
            "brain_backend_selected",
            {
                "configured_brain_backend": self.configured_brain_backend,
                "effective_brain_backend": self.effective_brain_backend,
                "provider_class": self.brain_provider.__class__.__name__,
                "fallback_backend": self.brain_backend_config.fallback_backend,
                "local_model_name": self.brain_backend_config.local_model if self.configured_brain_backend != "rule_brain" else None,
                "local_base_url": self.brain_backend_config.local_base_url if self.configured_brain_backend != "rule_brain" else None,
            },
        )
        self.logger.log_event(
            self.time,
            "planner_cadence_configured",
            {"planner_defaults": self.planner_defaults},
        )
        self.logger.log_event(
            self.time,
            "brain_backend_runtime_status",
            {
                "configured_brain_backend": self.configured_brain_backend,
                "effective_brain_backend": self.effective_brain_backend,
                "fallback_backend": self.brain_backend_config.fallback_backend,
                "local_model_name": self.brain_backend_config.local_model if self.configured_brain_backend != "rule_brain" else None,
                "local_base_url": self.brain_backend_config.local_base_url if self.configured_brain_backend != "rule_brain" else None,
            },
        )
        self.save_interval = 10.0
        self._last_save_time = 0.0
        self._last_phase_index = self.environment.current_phase_index
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
            if self.task_model.agent_defaults:
                agent_configs = []
                for d in self.task_model.agent_defaults:
                    agent_configs.append(
                        {
                            "name": d.agent_name,
                            "display_name": d.display_name or d.agent_name,
                            "agent_id": d.agent_id or d.agent_name,
                            "label": d.agent_label or d.role_id,
                            "role": d.role_id,
                            "template_id": d.template_id or None,
                            "constructs": {
                                "teamwork_potential": d.teamwork_potential,
                                "taskwork_potential": d.taskwork_potential,
                            },
                            "mechanism_overrides": dict(d.mechanism_overrides),
                            "traits": dict(d.mechanism_overrides),
                            "packet_access": list(d.source_access_override),
                            "accessible_packet_ids": list(d.source_access_override),
                            "initial_goal_seeds": list(d.initial_goal_seeds or []),
                            "communication_params": dict(d.communication_params or {}),
                            "brain_config": dict(d.brain_config or {}),
                            "task_overrides": dict(d.task_overrides or {}),
                            "planner_config": dict(d.planner_config),
                        }
                    )
            else:
                agent_configs = [
                    {"name": "Architect", "role": "Architect", "traits": {}},
                    {"name": "Engineer", "role": "Engineer", "traits": {}},
                    {"name": "Botanist", "role": "Botanist", "traits": {}},
                ]


        for config in agent_configs:
            config = self._resolve_agent_config_with_template(config)
            role_id = config.get("role", config.get("label", "Agent"))
            position = self.environment.get_spawn_point(role_id)
            merged_planner_config = dict(self.planner_defaults)
            merged_planner_config.update(dict(config.get("planner_config", {})))
            agent = Agent(
                name=config.get("name", config.get("display_name", role_id)),
                role=role_id,
                position=position,
                planner_config=merged_planner_config,
                agent_id=config.get("agent_id"),
                display_name=config.get("display_name", config.get("name", role_id)),
                agent_label=config.get("label") or config.get("alias"),
                template_id=config.get("template_id"),
                brain_config=config.get("brain_config"),
                communication_params=config.get("communication_params"),
                initial_goal_seeds=config.get("initial_goal_seeds"),
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
            role_sources = self.task_model.source_ids_for_role(role_id)
            mapped_packets = [
                self.environment.source_packet_name_map.get(source_id, source_id)
                for source_id in role_sources
                if self.environment.source_packet_name_map.get(source_id, source_id) in self.environment.knowledge_packets
            ]
            fallback = config.get("accessible_packet_ids") or config.get("packet_access")
            agent.allowed_packet = mapped_packets or fallback
            agent.task_model = self.task_model
            agent.initial_goal_seeds = list(config.get("initial_goal_seeds", []) or [])
            agent._seed_task_defined_goals(sim_state=self)
            for seed in agent.initial_goal_seeds[:3]:
                if isinstance(seed, str) and seed.strip() and seed not in agent.goal_registry:
                    agent.push_goal(seed.strip(), target=None)
            agent.update_current_goal()
            self.agents.append(agent)

        self.environment.agents = self.agents
        self.metrics = MetricsCollector(self)
        self.logger.register_event_listener(self.metrics.on_event)
        self.logger.initialize_session_outputs(
            speed=speed,
            flash_mode=self.flash_mode,
            active_agents=[{"name": agent.name, "role": agent.role} for agent in self.agents],
            extra_metadata=self._backend_settings_for_manifest(),
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


    def _backend_settings_for_manifest(self):
        cfg = self.brain_backend_config
        return {
            "configured_brain_backend": self.configured_brain_backend,
            "effective_brain_backend": self.effective_brain_backend,
            "fallback_backend": cfg.fallback_backend,
            "local_model_name": cfg.local_model if self.configured_brain_backend != "rule_brain" else None,
            "local_base_url": cfg.local_base_url if self.configured_brain_backend != "rule_brain" else None,
            "local_endpoint": cfg.local_endpoint if self.configured_brain_backend != "rule_brain" else None,
            "fallback_occurred": self.backend_fallback_count > 0,
            "fallback_count": self.backend_fallback_count,
        }

    def _refresh_backend_effective_state(self, reason="runtime_update"):
        configured = self.configured_brain_backend
        provider = self.brain_provider
        effective = configured
        fallback_happened = False
        fallback_reason = None
        if hasattr(provider, "last_outcome") and isinstance(provider.last_outcome, dict):
            outcome = provider.last_outcome
            if outcome.get("fallback"):
                signature = (outcome.get("reason"), outcome.get("latency_ms"))
                if signature != self._last_backend_outcome_signature:
                    fallback_happened = True
                    self._last_backend_outcome_signature = signature
                fallback_reason = outcome.get("reason")
                effective = self.brain_backend_config.fallback_backend or "rule_brain"
            elif outcome.get("fallback") is False:
                self._last_backend_outcome_signature = None
        if fallback_happened:
            self.backend_fallback_count += 1
            self.fallback_occurred = True
            self.logger.log_event(
                self.time,
                "brain_provider_fallback",
                {
                    "configured_brain_backend": configured,
                    "effective_brain_backend": effective,
                    "provider": provider.__class__.__name__,
                    "fallback_provider": "RuleBrain",
                    "reason": fallback_reason,
                    "fallback_count": self.backend_fallback_count,
                },
            )
        previous_effective = self.effective_brain_backend
        self.effective_brain_backend = effective
        if previous_effective != self.effective_brain_backend:
            self.logger.log_event(
                self.time,
                "effective_brain_backend_updated",
                {
                    "configured_brain_backend": configured,
                    "effective_brain_backend": self.effective_brain_backend,
                    "reason": reason,
                },
            )



    def _agent_templates(self):
        return dict(self.task_model.manifest.get("agent_templates", {}))

    def _resolve_agent_config_with_template(self, config):
        cfg = dict(config)
        template_id = cfg.get("template_id")
        templates = self._agent_templates()
        if template_id and template_id in templates:
            merged = dict(templates.get(template_id, {}))
            for k, v in cfg.items():
                if isinstance(v, dict) and isinstance(merged.get(k), dict):
                    base = dict(merged.get(k, {}))
                    base.update(v)
                    merged[k] = base
                else:
                    merged[k] = v
            return merged
        return cfg

    def update(self, base_dt):
        dt = base_dt * self.speed_multiplier
        previous_phase_index = self.environment.current_phase_index
        previous_phase = self.environment.get_current_phase() or {"name": "default"}
        self.environment.update(self.time)
        if self.environment.current_phase_index != previous_phase_index:
            current_phase = self.environment.get_current_phase() or {"name": "default"}
            self.logger.log_event(
                self.time,
                "phase_transition",
                {
                    "from_phase": previous_phase.get("name", "default"),
                    "to_phase": current_phase.get("name", "default"),
                    "from_index": previous_phase_index,
                    "to_index": self.environment.current_phase_index,
                },
            )
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
            self._refresh_backend_effective_state(reason="planner_call")
            self.logger.log_agent_state(self.time, agent)

        self.metrics.on_step(dt)

        self.time += dt

        if self.flash_mode or (self.time - self._last_save_time >= self.save_interval):
            self.logger.save_csv()
            self._last_save_time = self.time

    def stop(self):
        self.metrics.finalize()
        self.logger.update_session_manifest(extra_metadata=self._backend_settings_for_manifest())
        self.logger.save_csv()


    def _distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)
