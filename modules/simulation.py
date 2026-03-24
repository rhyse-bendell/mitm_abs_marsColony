# File: modules/simulation.py

import math
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from modules.agent import Agent
from modules.brain_context import BrainContextBuilder
from modules.brain_provider import BrainBackendConfig, create_brain_provider
from modules.environment import Environment
from modules.logging_tools import SimulationLogger
from modules.llm_sanity import StartupLLMSanityConfig, run_startup_llm_sanity_check
from modules.interaction_graph import InteractionTelemetryBridge
from modules.metrics import MetricsCollector
from modules.runtime_witness_audit import RuntimeWitnessAudit
from modules.team_knowledge import TeamKnowledgeManager
from modules.construct_mapping import ConstructMapper
from modules.task_model import load_task_model

LOCAL_BACKEND_ALIASES = {"local_http", "openai_compatible_local", "ollama_local", "ollama"}
UNRESTRICTED_QWEN_DEFAULTS = {
    "planner_interval_steps": 24,
    "planner_interval_time": 20.0,
    "planner_timeout_seconds": 900.0,
    "startup_llm_sanity_timeout_seconds": 900.0,
    "startup_llm_sanity_completion_max_tokens": 24576,
    "planner_completion_max_tokens": 24576,
    "warmup_timeout_seconds": 600.0,
    "degraded_consecutive_failures_threshold": 24,
    "degraded_cooldown_seconds": 300.0,
    "degraded_step_interval_multiplier": 8.0,
    "high_latency_stale_result_grace_s": 1800.0,
    "permissive_timeout_ceiling_s": 1800.0,
    "permissive_completion_ceiling_tokens": 32768,
    "sticky_backend_demotion_enabled": False,
}


def _planner_defaults_with_high_latency_mode(planner_defaults, configured_backend):
    defaults = dict(planner_defaults or {})
    high_latency_enabled = bool(defaults.get("high_latency_local_llm_mode", configured_backend in LOCAL_BACKEND_ALIASES))
    unrestricted_local_qwen_mode = bool(defaults.get("unrestricted_local_qwen_mode", high_latency_enabled and configured_backend in LOCAL_BACKEND_ALIASES))
    defaults["high_latency_local_llm_mode"] = high_latency_enabled
    defaults["unrestricted_local_qwen_mode"] = unrestricted_local_qwen_mode
    if high_latency_enabled:
        defaults.setdefault("planner_interval_steps", 8)
        defaults.setdefault("planner_interval_time", 6.0)
        defaults.setdefault("planner_timeout_seconds", 90.0)
        defaults.setdefault("degraded_consecutive_failures_threshold", 6)
        defaults.setdefault("degraded_cooldown_seconds", 45.0)
        defaults.setdefault("degraded_step_interval_multiplier", 3.0)
        defaults.setdefault("startup_llm_sanity_timeout_seconds", 45.0)
        defaults.setdefault("high_latency_stale_result_grace_s", 60.0)
    if unrestricted_local_qwen_mode:
        defaults["high_latency_local_llm_mode"] = True
        for key, value in UNRESTRICTED_QWEN_DEFAULTS.items():
            existing = defaults.get(key)
            if isinstance(value, bool):
                defaults[key] = bool(existing) if isinstance(existing, bool) else value
            elif isinstance(value, int):
                if isinstance(existing, (int, float)):
                    defaults[key] = max(int(existing), int(value))
                else:
                    defaults[key] = int(value)
            elif isinstance(value, float):
                if isinstance(existing, (int, float)):
                    defaults[key] = max(float(existing), float(value))
                else:
                    defaults[key] = float(value)
            else:
                defaults.setdefault(key, value)
    return defaults


def _load_task_construction_defaults(task_id):
    path = Path(__file__).resolve().parents[1] / "config" / "tasks" / task_id / "construction_parameters.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


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
        construction_parameters=None,
        task_id="mars_colony",
        startup_progress_callback=None,
        execution_metadata=None,
    ):
        self.task_model = load_task_model(task_id=task_id)
        self.construction_parameters = _load_task_construction_defaults(task_id)
        if isinstance(construction_parameters, dict):
            self.construction_parameters.update(construction_parameters)
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
        self.environment = Environment(
            phases=phases,
            task_model=self.task_model,
            construction_parameters=self.construction_parameters,
        )
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
        self.planner_defaults = _planner_defaults_with_high_latency_mode(self.planner_defaults, brain_backend)
        self.startup_llm_sanity_config = StartupLLMSanityConfig(
            enabled=bool(self.planner_defaults.get("enable_startup_llm_sanity", True)),
            timeout_s=float(self.planner_defaults.get("startup_llm_sanity_timeout_seconds", 45.0) or 45.0),
            max_sources=max(1, int(self.planner_defaults.get("startup_llm_sanity_max_sources", 2) or 2)),
            max_items_per_type=max(1, int(self.planner_defaults.get("startup_llm_sanity_max_items_per_type", 3) or 3)),
            completion_max_tokens=max(256, int(self.planner_defaults.get("startup_llm_sanity_completion_max_tokens", 1024) or 1024)),
            json_only_mode=bool(self.planner_defaults.get("startup_llm_sanity_json_only_mode", True)),
            reasoning_suppression=bool(self.planner_defaults.get("startup_llm_sanity_reasoning_suppression", True)),
            raw_response_max_chars=max(500, int(self.planner_defaults.get("startup_llm_sanity_raw_response_max_chars", 4000) or 4000)),
            artifact_name=str(self.planner_defaults.get("startup_llm_sanity_artifact_name", "startup_llm_sanity.json") or "startup_llm_sanity.json"),
        )
        self.bootstrap_reuse_enabled = bool(self.planner_defaults.get("enable_bootstrap_summary_reuse", True))
        self.bootstrap_summary_max_chars = max(80, int(self.planner_defaults.get("bootstrap_summary_max_chars", 280) or 280))
        self.startup_llm_sanity_summary = {
            "startup_llm_sanity_enabled": bool(self.startup_llm_sanity_config.enabled),
            "startup_llm_sanity_agent_count": 0,
            "startup_llm_sanity_success_count": 0,
            "startup_llm_sanity_failure_count": 0,
            "startup_llm_sanity_timeout_count": 0,
            "startup_llm_sanity_parse_failure_count": 0,
            "startup_llm_sanity_artifact": None,
            "bootstrap_reuse_enabled": bool(self.bootstrap_reuse_enabled),
            "bootstrap_reuse_agent_count": 0,
        }
        self.bootstrap_reuse_included_count = 0
        self.startup_progress_callback = startup_progress_callback
        self.execution_metadata = dict(execution_metadata or {})
        planner_trace_enabled = bool(self.planner_defaults.get("enable_planner_trace", True))
        planner_trace_mode = str(self.planner_defaults.get("planner_trace_mode", "full") or "full").lower()
        planner_trace_max_chars = int(self.planner_defaults.get("planner_trace_max_chars", 12000) or 12000)
        self.logger.configure_planner_trace(
            enabled=planner_trace_enabled,
            mode=planner_trace_mode,
            max_chars=planner_trace_max_chars,
        )
        backend_options = dict(brain_backend_options or {})
        if "timeout_s" not in backend_options and "planner_timeout_seconds" in self.planner_defaults:
            backend_options["timeout_s"] = self.planner_defaults.get("planner_timeout_seconds")
        if "max_retries" not in backend_options and "planner_max_retries" in self.planner_defaults:
            backend_options["max_retries"] = self.planner_defaults.get("planner_max_retries")
        if "warmup_timeout_s" not in backend_options:
            backend_options["warmup_timeout_s"] = self.planner_defaults.get("warmup_timeout_seconds", self.planner_defaults.get("startup_llm_sanity_timeout_seconds"))
        if "fallback_backend" not in backend_options and "planner_fallback_backend" in self.planner_defaults:
            backend_options["fallback_backend"] = self.planner_defaults.get("planner_fallback_backend")
        if "completion_max_tokens" not in backend_options:
            backend_options["completion_max_tokens"] = self.planner_defaults.get("planner_completion_max_tokens", 2048)
        if "startup_completion_max_tokens" not in backend_options:
            backend_options["startup_completion_max_tokens"] = self.planner_defaults.get("startup_llm_sanity_completion_max_tokens", 1024)
        if "permissive_timeout_ceiling_s" not in backend_options:
            backend_options["permissive_timeout_ceiling_s"] = self.planner_defaults.get("permissive_timeout_ceiling_s", 1200.0)
        if "permissive_completion_ceiling_tokens" not in backend_options:
            backend_options["permissive_completion_ceiling_tokens"] = self.planner_defaults.get("permissive_completion_ceiling_tokens", 16384)
        backend_options.setdefault("unrestricted_local_qwen_mode", self.planner_defaults.get("unrestricted_local_qwen_mode", False))
        self.brain_backend_config = BrainBackendConfig(backend=brain_backend, **backend_options)
        if self.brain_backend_config.unrestricted_local_qwen_mode:
            effective_timeout_ceiling = max(60.0, float(self.brain_backend_config.permissive_timeout_ceiling_s))
            effective_completion_ceiling = max(512, int(self.brain_backend_config.permissive_completion_ceiling_tokens))
            startup_timeout_floor = float(self.planner_defaults.get("startup_llm_sanity_timeout_seconds", self.startup_llm_sanity_config.timeout_s) or self.startup_llm_sanity_config.timeout_s)
            startup_tokens_floor = int(self.planner_defaults.get("startup_llm_sanity_completion_max_tokens", self.startup_llm_sanity_config.completion_max_tokens) or self.startup_llm_sanity_config.completion_max_tokens)
            planner_tokens_floor = int(self.planner_defaults.get("planner_completion_max_tokens", self.brain_backend_config.completion_max_tokens) or self.brain_backend_config.completion_max_tokens)
            requested_planner_timeout = max(
                float(self.brain_backend_config.timeout_s),
                float(self.planner_defaults.get("planner_timeout_seconds", self.brain_backend_config.timeout_s) or self.brain_backend_config.timeout_s),
            )
            requested_warmup_timeout = max(
                float(self.brain_backend_config.warmup_timeout_s),
                float(self.planner_defaults.get("warmup_timeout_seconds", self.brain_backend_config.warmup_timeout_s) or self.brain_backend_config.warmup_timeout_s),
            )
            effective_planner_timeout = min(requested_planner_timeout, effective_timeout_ceiling)
            effective_warmup_timeout = min(requested_warmup_timeout, effective_timeout_ceiling)
            effective_startup_timeout = min(max(float(self.startup_llm_sanity_config.timeout_s), startup_timeout_floor), effective_timeout_ceiling)
            effective_startup_tokens = min(max(int(self.startup_llm_sanity_config.completion_max_tokens), startup_tokens_floor), effective_completion_ceiling)
            effective_planner_tokens = min(max(int(self.brain_backend_config.completion_max_tokens), planner_tokens_floor), effective_completion_ceiling)
            self.startup_llm_sanity_config = StartupLLMSanityConfig(
                enabled=self.startup_llm_sanity_config.enabled,
                timeout_s=effective_startup_timeout,
                max_sources=self.startup_llm_sanity_config.max_sources,
                max_items_per_type=self.startup_llm_sanity_config.max_items_per_type,
                completion_max_tokens=effective_startup_tokens,
                json_only_mode=self.startup_llm_sanity_config.json_only_mode,
                reasoning_suppression=self.startup_llm_sanity_config.reasoning_suppression,
                raw_response_max_chars=self.startup_llm_sanity_config.raw_response_max_chars,
                artifact_name=self.startup_llm_sanity_config.artifact_name,
            )
            self.brain_backend_config = BrainBackendConfig(
                backend=self.brain_backend_config.backend,
                local_base_url=self.brain_backend_config.local_base_url,
                local_endpoint=self.brain_backend_config.local_endpoint,
                local_model=self.brain_backend_config.local_model,
                timeout_s=effective_planner_timeout,
                warmup_timeout_s=effective_warmup_timeout,
                completion_max_tokens=effective_planner_tokens,
                startup_completion_max_tokens=effective_startup_tokens,
                permissive_timeout_ceiling_s=effective_timeout_ceiling,
                permissive_completion_ceiling_tokens=effective_completion_ceiling,
                unrestricted_local_qwen_mode=True,
                max_retries=self.brain_backend_config.max_retries,
                fallback_backend=self.brain_backend_config.fallback_backend,
                debug=self.brain_backend_config.debug,
                planner_trace_enabled=self.brain_backend_config.planner_trace_enabled,
                planner_trace_mode=self.brain_backend_config.planner_trace_mode,
                planner_trace_max_chars=self.brain_backend_config.planner_trace_max_chars,
            )
            self.planner_defaults["planner_timeout_seconds"] = effective_planner_timeout
            self.planner_defaults["startup_llm_sanity_timeout_seconds"] = effective_startup_timeout
            self.planner_defaults["warmup_timeout_seconds"] = effective_warmup_timeout
            self.planner_defaults["planner_completion_max_tokens"] = effective_planner_tokens
            self.planner_defaults["startup_llm_sanity_completion_max_tokens"] = effective_startup_tokens
        self.configured_brain_backend = self.brain_backend_config.backend
        self.brain_provider = create_brain_provider(self.brain_backend_config)
        self.provider_warmup_status = None
        if hasattr(self.brain_provider, "warmup_probe") and callable(getattr(self.brain_provider, "warmup_probe")):
            self.provider_warmup_status = self.brain_provider.warmup_probe()
        self.agent_brain_runtime = {}
        worker_count = int(self.planner_defaults.get("planner_async_workers", max(2, len(self.task_model.agent_defaults) if self.task_model.agent_defaults else 3)))
        self.planner_executor = ThreadPoolExecutor(max_workers=max(1, worker_count), thread_name_prefix="planner")
        self.planner_barrier_policy = str(self.planner_defaults.get("planner_barrier_policy", "global") or "global").lower()
        self.planner_barrier_state = {
            "active": False,
            "reason": None,
            "blocking_agent_ids": [],
            "blocking_request_ids": [],
            "pause_started_wallclock_at": None,
            "pause_started_sim_time": None,
            "pause_started_tick": None,
            "pause_count": 0,
            "total_wallclock_wait_s": 0.0,
            "last_wallclock_wait_s": 0.0,
            "last_active_emit_at": 0.0,
        }
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
                "local_backend_alias": "ollama_openai_compatible" if self.configured_brain_backend in {"local_http", "openai_compatible_local", "ollama_local", "ollama"} else None,
                "local_model_name": self.brain_backend_config.local_model if self.configured_brain_backend != "rule_brain" else None,
                "local_base_url": self.brain_backend_config.local_base_url if self.configured_brain_backend != "rule_brain" else None,
                "local_endpoint": self.brain_backend_config.local_endpoint if self.configured_brain_backend != "rule_brain" else None,
                "timeout_s": self.brain_backend_config.timeout_s if self.configured_brain_backend != "rule_brain" else None,
                "backend_warmup": self.provider_warmup_status,
            },
        )
        self.logger.log_event(
            self.time,
            "planner_cadence_configured",
            {"planner_defaults": self.planner_defaults},
        )
        self.logger.log_event(
            self.time,
            "planner_barrier_configured",
            {
                "planner_barrier_policy": self.planner_barrier_policy,
                "planner_blocks_sim_time_default": self.planner_defaults.get("planner_blocks_sim_time"),
                "high_latency_local_llm_mode": bool(self.planner_defaults.get("high_latency_local_llm_mode", False)),
                "unrestricted_local_qwen_mode": bool(self.planner_defaults.get("unrestricted_local_qwen_mode", False)),
            },
        )
        self.logger.log_event(
            self.time,
            "brain_backend_runtime_status",
            {
                "configured_brain_backend": self.configured_brain_backend,
                "effective_brain_backend": self.effective_brain_backend,
                "fallback_backend": self.brain_backend_config.fallback_backend,
                "local_backend_alias": "ollama_openai_compatible" if self.configured_brain_backend in {"local_http", "openai_compatible_local", "ollama_local", "ollama"} else None,
                "local_model_name": self.brain_backend_config.local_model if self.configured_brain_backend != "rule_brain" else None,
                "local_base_url": self.brain_backend_config.local_base_url if self.configured_brain_backend != "rule_brain" else None,
                "local_endpoint": self.brain_backend_config.local_endpoint if self.configured_brain_backend != "rule_brain" else None,
                "timeout_s": self.brain_backend_config.timeout_s if self.configured_brain_backend != "rule_brain" else None,
                "backend_warmup": self.provider_warmup_status,
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
            merged_brain_config = dict(config.get("brain_config", {}))
            merged_brain_config.setdefault("backend", self.configured_brain_backend)
            merged_brain_config.setdefault("local_model", self.brain_backend_config.local_model)
            merged_brain_config.setdefault("fallback_backend", self.brain_backend_config.fallback_backend)
            merged_brain_config.setdefault("timeout_s", self.brain_backend_config.timeout_s)
            merged_brain_config.setdefault("local_base_url", self.brain_backend_config.local_base_url)
            merged_brain_config.setdefault("local_endpoint", self.brain_backend_config.local_endpoint)
            agent = Agent(
                name=config.get("name", config.get("display_name", role_id)),
                role=role_id,
                position=position,
                planner_config=merged_planner_config,
                agent_id=config.get("agent_id"),
                display_name=config.get("display_name", config.get("name", role_id)),
                agent_label=config.get("label") or config.get("alias"),
                template_id=config.get("template_id"),
                brain_config=merged_brain_config,
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
            self._register_agent_brain_runtime(agent)
            self.agents.append(agent)

        self.environment.agents = self.agents
        self.runtime_witness_audit = RuntimeWitnessAudit(self)
        self.metrics = MetricsCollector(self)
        self.logger.register_event_listener(self.runtime_witness_audit.on_event)
        self.logger.register_event_listener(self.metrics.on_event)
        self.interaction_telemetry = InteractionTelemetryBridge(self.logger)
        self.logger.register_event_listener(self.interaction_telemetry.on_event)
        self.logger.initialize_session_outputs(
            speed=speed,
            flash_mode=self.flash_mode,
            active_agents=[self._agent_manifest_row(agent) for agent in self.agents],
            extra_metadata={
                **self._backend_settings_for_manifest(),
                **self.execution_metadata,
            },
        )
        self.run_startup_llm_sanity_check()
        self.logger.log_event(
            self.time,
            "session_initialized",
            {
                "session_folder": str(self.logger.output_session.session_folder),
                "speed": speed,
                "flash_mode": self.flash_mode,
                "agents": [agent.name for agent in self.agents],
                "execution_metadata": self.execution_metadata,
            },
        )


    def run_startup_llm_sanity_check(self):
        if not self.startup_llm_sanity_config.enabled:
            self.startup_llm_sanity_summary.update(
                {
                    "startup_llm_sanity_enabled": False,
                    "bootstrap_reuse_enabled": bool(self.bootstrap_reuse_enabled),
                    "bootstrap_reuse_agent_count": len(self.agents) if self.bootstrap_reuse_enabled else 0,
                }
            )
            self.logger.log_event(self.time, "startup_llm_sanity_disabled", {"enabled": False})
            return dict(self.startup_llm_sanity_summary)
        try:
            summary = run_startup_llm_sanity_check(
                self,
                config=self.startup_llm_sanity_config,
                progress_callback=self.startup_progress_callback,
            )
            self.startup_llm_sanity_summary.update(summary)
            results_by_agent = {
                str(item.get("agent_id")): item
                for item in (summary.get("startup_llm_sanity_results") or [])
                if isinstance(item, dict) and item.get("agent_id")
            }
            for agent in self.agents:
                row = results_by_agent.get(str(agent.agent_id), {})
                validation_success = bool(row.get("validation_success"))
                agent.fallback_bootstrap["startup_sanity_status"] = "success" if validation_success else "failed"
                if not validation_success:
                    if self.sticky_backend_demotion_enabled(agent):
                        self.hard_demote_agent_backend(agent, reason="startup_sanity_failed", activate_bootstrap=False)
                    agent.activate_fallback_bootstrap(sim_state=self, reason="startup_sanity_failed")
            self.logger.update_session_manifest(extra_metadata=self._backend_settings_for_manifest())
            return dict(self.startup_llm_sanity_summary)
        except Exception as exc:  # noqa: BLE001
            self.startup_llm_sanity_summary.update(
                {
                    "startup_llm_sanity_enabled": True,
                    "startup_llm_sanity_failure_count": len(self.agents),
                    "startup_llm_sanity_agent_count": len(self.agents),
                    "startup_llm_sanity_error": f"{type(exc).__name__}: {exc}",
                }
            )
            for agent in self.agents:
                agent.fallback_bootstrap["startup_sanity_status"] = "failed"
                if self.sticky_backend_demotion_enabled(agent):
                    self.hard_demote_agent_backend(agent, reason="startup_sanity_exception", activate_bootstrap=False)
                agent.activate_fallback_bootstrap(sim_state=self, reason="startup_sanity_exception")
            self.logger.log_event(
                self.time,
                "startup_llm_sanity_failed",
                {"error": f"{type(exc).__name__}: {exc}", "agent_count": len(self.agents)},
            )
            self.logger.update_session_manifest(extra_metadata=self._backend_settings_for_manifest())
            return dict(self.startup_llm_sanity_summary)

    def _agent_manifest_row(self, agent):
        runtime = self.get_agent_brain_runtime(agent)
        return {
            "name": agent.name,
            "display_name": agent.display_name,
            "label": agent.agent_label,
            "role": agent.role,
            "configured_backend": runtime["configured_backend"],
            "effective_backend": runtime.get("effective_backend"),
            "hard_demoted": bool(runtime.get("hard_demoted")),
            "hard_demoted_reason": runtime.get("hard_demoted_reason"),
            "fallback_count": runtime.get("fallback_count", 0),
            "fallback_backend": runtime["config"].fallback_backend,
            "local_model": runtime["config"].local_model if runtime["configured_backend"] != "rule_brain" else None,
            "planner_interval_steps": agent.planner_cadence.planner_interval_steps,
            "planner_timeout_seconds": agent.planner_cadence.planner_timeout_seconds,
            "planner_max_retries": agent.planner_cadence.planner_max_retries,
            "degraded_consecutive_failures_threshold": agent.planner_cadence.degraded_consecutive_failures_threshold,
            "degraded_cooldown_seconds": agent.planner_cadence.degraded_cooldown_seconds,
            "degraded_step_interval_multiplier": agent.planner_cadence.degraded_step_interval_multiplier,
            "high_latency_local_llm_mode": bool(agent.planner_cadence.high_latency_local_llm_mode),
            "unrestricted_local_qwen_mode": bool(agent.planner_cadence.unrestricted_local_qwen_mode),
            "high_latency_stale_result_grace_s": float(agent.planner_cadence.high_latency_stale_result_grace_s),
            "sticky_backend_demotion_enabled": bool(agent.planner_cadence.sticky_backend_demotion_enabled),
            "planner_blocks_sim_time": agent.planner_cadence.planner_blocks_sim_time,
        }

    def _register_agent_brain_runtime(self, agent):
        cfg = BrainBackendConfig(
            backend=str(agent.brain_config.get("backend", self.configured_brain_backend) or self.configured_brain_backend),
            local_base_url=str(agent.brain_config.get("local_base_url", self.brain_backend_config.local_base_url) or self.brain_backend_config.local_base_url),
            local_endpoint=str(agent.brain_config.get("local_endpoint", self.brain_backend_config.local_endpoint) or self.brain_backend_config.local_endpoint),
            local_model=str(agent.brain_config.get("local_model", self.brain_backend_config.local_model) or self.brain_backend_config.local_model),
            timeout_s=float(agent.brain_config.get("timeout_s", self.brain_backend_config.timeout_s)),
            warmup_timeout_s=float(agent.brain_config.get("warmup_timeout_s", self.brain_backend_config.warmup_timeout_s)),
            completion_max_tokens=int(agent.brain_config.get("completion_max_tokens", self.brain_backend_config.completion_max_tokens)),
            startup_completion_max_tokens=int(agent.brain_config.get("startup_completion_max_tokens", self.brain_backend_config.startup_completion_max_tokens)),
            permissive_timeout_ceiling_s=float(agent.brain_config.get("permissive_timeout_ceiling_s", self.brain_backend_config.permissive_timeout_ceiling_s)),
            permissive_completion_ceiling_tokens=int(agent.brain_config.get("permissive_completion_ceiling_tokens", self.brain_backend_config.permissive_completion_ceiling_tokens)),
            unrestricted_local_qwen_mode=bool(agent.brain_config.get("unrestricted_local_qwen_mode", self.brain_backend_config.unrestricted_local_qwen_mode)),
            max_retries=int(agent.brain_config.get("max_retries", self.brain_backend_config.max_retries)),
            fallback_backend=str(agent.brain_config.get("fallback_backend", self.brain_backend_config.fallback_backend) or self.brain_backend_config.fallback_backend),
            debug=bool(agent.brain_config.get("debug", self.brain_backend_config.debug)),
        )
        provider = create_brain_provider(cfg)
        self.agent_brain_runtime[agent.agent_id] = {
            "config": cfg,
            "provider": provider,
            "configured_backend": cfg.backend,
            "effective_backend": cfg.backend,
            "hard_demoted": False,
            "hard_demoted_reason": None,
            "fallback_count": 0,
            "last_outcome_signature": None,
            "bootstrap": {
                "status": "not_run",
                "latency_ms": None,
                "validated_response": None,
                "summary_text": None,
                "summary_structured": None,
                "included_count": 0,
            },
        }

    def get_agent_brain_runtime(self, agent):
        return self.agent_brain_runtime.get(agent.agent_id) or {
            "config": self.brain_backend_config,
            "provider": self.brain_provider,
            "configured_backend": self.configured_brain_backend,
            "effective_backend": self.effective_brain_backend,
            "fallback_count": self.backend_fallback_count,
            "last_outcome_signature": None,
            "bootstrap": {
                "status": "not_run",
                "latency_ms": None,
                "validated_response": None,
                "summary_text": None,
                "summary_structured": None,
                "included_count": 0,
            },
        }

    def refresh_agent_backend_effective_state(self, agent, reason="runtime_update"):
        runtime = self.get_agent_brain_runtime(agent)
        if runtime.get("hard_demoted"):
            runtime["effective_backend"] = runtime["config"].fallback_backend or "rule_brain"
            agent.planner_state["fallback_only_ticks"] = int(agent.planner_state.get("fallback_only_ticks", 0)) + 1
            any_fallback = any(
                (rt.get("effective_backend") != rt.get("configured_backend")) or bool(rt.get("hard_demoted"))
                for rt in self.agent_brain_runtime.values()
            )
            self.effective_brain_backend = self.brain_backend_config.fallback_backend if any_fallback else self.configured_brain_backend
            self.backend_fallback_count = sum(int(rt.get("fallback_count", 0)) for rt in self.agent_brain_runtime.values())
            self.fallback_occurred = self.backend_fallback_count > 0
            return
        provider = runtime["provider"]
        configured = runtime["configured_backend"]
        effective = configured
        if hasattr(provider, "last_outcome") and isinstance(provider.last_outcome, dict):
            outcome = provider.last_outcome
            if outcome.get("fallback"):
                signature = (outcome.get("reason"), outcome.get("latency_ms"))
                if signature != runtime.get("last_outcome_signature"):
                    runtime["fallback_count"] += 1
                    runtime["last_outcome_signature"] = signature
                effective = runtime["config"].fallback_backend or "rule_brain"
            elif outcome.get("fallback") is False:
                runtime["last_outcome_signature"] = None
        runtime["effective_backend"] = effective
        if effective != configured:
            agent.planner_state["fallback_only_ticks"] = int(agent.planner_state.get("fallback_only_ticks", 0)) + 1

        any_fallback = any(
            (rt.get("effective_backend") != rt.get("configured_backend")) or bool(rt.get("hard_demoted"))
            for rt in self.agent_brain_runtime.values()
        )
        self.effective_brain_backend = self.brain_backend_config.fallback_backend if any_fallback else self.configured_brain_backend
        self.backend_fallback_count = sum(int(rt.get("fallback_count", 0)) for rt in self.agent_brain_runtime.values())
        self.fallback_occurred = self.backend_fallback_count > 0

    def is_agent_backend_hard_demoted(self, agent):
        runtime = self.get_agent_brain_runtime(agent)
        return bool(runtime.get("hard_demoted"))

    def sticky_backend_demotion_enabled(self, agent=None):
        if agent is not None and hasattr(agent, "planner_cadence"):
            return bool(getattr(agent.planner_cadence, "sticky_backend_demotion_enabled", False))
        return bool(
            self.planner_defaults.get(
                "sticky_backend_demotion_enabled",
                self.planner_defaults.get("allow_persistent_backend_demotion", False),
            )
        )

    def hard_demote_agent_backend(self, agent, reason, activate_bootstrap=True):
        runtime = self.get_agent_brain_runtime(agent)
        if runtime.get("hard_demoted"):
            return False
        fallback_backend = runtime["config"].fallback_backend or "rule_brain"
        runtime["provider"] = create_brain_provider(BrainBackendConfig(backend=fallback_backend))
        runtime["effective_backend"] = fallback_backend
        runtime["hard_demoted"] = True
        runtime["hard_demoted_reason"] = str(reason)
        runtime["fallback_count"] = int(runtime.get("fallback_count", 0)) + 1
        self.backend_fallback_count = sum(int(rt.get("fallback_count", 0)) for rt in self.agent_brain_runtime.values())
        self.fallback_occurred = True
        if hasattr(agent, "planner_state") and isinstance(agent.planner_state, dict):
            agent.planner_state["fallback_only_ticks"] = int(agent.planner_state.get("fallback_only_ticks", 0)) + 1
        if hasattr(agent, "clear_planner_inflight_state"):
            agent.clear_planner_inflight_state(sim_state=self, reason=f"hard_demote:{reason}")
        if activate_bootstrap and hasattr(agent, "activate_fallback_bootstrap"):
            agent.activate_fallback_bootstrap(sim_state=self, reason=f"hard_demote:{reason}")
        self.logger.log_event(
            self.time,
            "agent_backend_hard_demoted",
            {
                "agent": agent.name,
                "agent_id": agent.agent_id,
                "configured_backend": runtime.get("configured_backend"),
                "effective_backend": runtime.get("effective_backend"),
                "reason": str(reason),
                "fallback_backend": fallback_backend,
            },
        )
        self.logger.update_session_manifest(extra_metadata=self._backend_settings_for_manifest())
        return True


    def _backend_settings_for_manifest(self):
        cfg = self.brain_backend_config
        return {
            "configured_brain_backend": self.configured_brain_backend,
            "effective_brain_backend": self.effective_brain_backend,
            "fallback_backend": cfg.fallback_backend,
            "local_model_name": cfg.local_model if self.configured_brain_backend != "rule_brain" else None,
            "local_base_url": cfg.local_base_url if self.configured_brain_backend != "rule_brain" else None,
            "local_endpoint": cfg.local_endpoint if self.configured_brain_backend != "rule_brain" else None,
            "timeout_s": cfg.timeout_s if self.configured_brain_backend != "rule_brain" else None,
            "warmup_timeout_s": cfg.warmup_timeout_s if self.configured_brain_backend != "rule_brain" else None,
            "high_latency_local_llm_mode": bool(self.planner_defaults.get("high_latency_local_llm_mode", False)),
            "unrestricted_local_qwen_mode": bool(self.planner_defaults.get("unrestricted_local_qwen_mode", False)),
            "planner_blocks_sim_time_default": self.planner_defaults.get("planner_blocks_sim_time"),
            "effective_planner_timeout_seconds": float(self.planner_defaults.get("planner_timeout_seconds", 0.0) or 0.0),
            "effective_startup_llm_sanity_timeout_seconds": float(self.planner_defaults.get("startup_llm_sanity_timeout_seconds", 0.0) or 0.0),
            "effective_warmup_timeout_seconds": float(self.brain_backend_config.warmup_timeout_s or 0.0),
            "effective_startup_llm_sanity_completion_max_tokens": int(self.startup_llm_sanity_config.completion_max_tokens),
            "effective_planner_completion_max_tokens": int(self.brain_backend_config.completion_max_tokens),
            "stale_result_relaxation_enabled": bool(self.planner_defaults.get("high_latency_local_llm_mode", False)) and float(self.planner_defaults.get("high_latency_stale_result_grace_s", 0.0) or 0.0) > 0.0,
            "high_latency_stale_result_grace_s": float(self.planner_defaults.get("high_latency_stale_result_grace_s", 0.0) or 0.0),
            "sticky_backend_demotion_enabled": bool(self.sticky_backend_demotion_enabled()),
            "permissive_timeout_ceiling_s": float(self.brain_backend_config.permissive_timeout_ceiling_s or 0.0),
            "permissive_completion_ceiling_tokens": int(self.brain_backend_config.permissive_completion_ceiling_tokens or 0),
            "provider_class": self.brain_provider.__class__.__name__,
            "local_backend_alias": "ollama_openai_compatible" if self.configured_brain_backend in {"local_http", "openai_compatible_local", "ollama_local", "ollama"} else None,
            "fallback_occurred": self.backend_fallback_count > 0,
            "fallback_count": self.backend_fallback_count,
            "planner_trace_enabled": bool(self.planner_defaults.get("enable_planner_trace", True)),
            "planner_trace_mode": str(self.planner_defaults.get("planner_trace_mode", "full") or "full"),
            "planner_trace_max_chars": int(self.planner_defaults.get("planner_trace_max_chars", 12000) or 12000),
            "planner_trace_artifact": "logs/planner_trace.jsonl" if bool(self.planner_defaults.get("enable_planner_trace", True)) else None,
            "backend_warmup": self.provider_warmup_status,
            "bootstrap_summary_max_chars": self.bootstrap_summary_max_chars,
            "bootstrap_reuse_included_count": int(getattr(self, "bootstrap_reuse_included_count", 0)),
            "planner_barrier_policy": self.planner_barrier_policy,
            "planner_barrier_active": bool(self.planner_barrier_state.get("active")),
            "planner_barrier_pause_count": int(self.planner_barrier_state.get("pause_count", 0)),
            "planner_barrier_total_wallclock_wait_s": float(self.planner_barrier_state.get("total_wallclock_wait_s", 0.0)),
            "planner_blocking_backends_detected": sorted(
                {
                    str(self.get_agent_brain_runtime(agent).get("effective_backend", self.configured_brain_backend))
                    for agent in self.agents
                    if hasattr(agent, "planner_request_blocks_sim_time")
                    and bool(agent.planner_request_blocks_sim_time(sim_state=self, runtime=self.get_agent_brain_runtime(agent)))
                }
            ),
            **dict(self.startup_llm_sanity_summary),
        }

    def _collect_blocking_planner_requests(self):
        active = []
        for agent in self.agents:
            planner_state = getattr(agent, "planner_state", {}) or {}
            if planner_state.get("status") != "in_flight":
                continue
            runtime = self.get_agent_brain_runtime(agent)
            if not hasattr(agent, "planner_request_blocks_sim_time"):
                continue
            if not bool(agent.planner_request_blocks_sim_time(sim_state=self, runtime=runtime)):
                continue
            active.append(
                {
                    "agent_id": agent.agent_id,
                    "agent_name": agent.name,
                    "request_id": planner_state.get("request_id"),
                    "reason": planner_state.get("barrier_reason") or "planner_request_in_flight",
                    "effective_backend": runtime.get("effective_backend", self.configured_brain_backend),
                    "configured_backend": runtime.get("configured_backend", self.configured_brain_backend),
                }
            )
        return active

    def _refresh_planner_barrier_state(self):
        now_wallclock = time.perf_counter()
        blocking_requests = self._collect_blocking_planner_requests()
        barrier_active = bool(blocking_requests) and self.planner_barrier_policy == "global"
        state = self.planner_barrier_state
        state["blocking_agent_ids"] = [row["agent_id"] for row in blocking_requests]
        state["blocking_request_ids"] = [row["request_id"] for row in blocking_requests if row.get("request_id")]
        state["reason"] = blocking_requests[0]["reason"] if blocking_requests else None
        if barrier_active and not state.get("active"):
            state["active"] = True
            state["pause_started_wallclock_at"] = now_wallclock
            state["pause_started_sim_time"] = float(self.time)
            state["pause_started_tick"] = int(round(self.time * 1000.0))
            state["pause_count"] = int(state.get("pause_count", 0)) + 1
            self.logger.log_event(
                self.time,
                "planner_barrier_started",
                {
                    "cognition_pause_reason": state.get("reason"),
                    "blocking_request_ids": list(state.get("blocking_request_ids", [])),
                    "blocking_agent_ids": list(state.get("blocking_agent_ids", [])),
                    "planner_barrier_policy": self.planner_barrier_policy,
                },
            )
        elif barrier_active and state.get("active"):
            if now_wallclock - float(state.get("last_active_emit_at", 0.0) or 0.0) >= 0.2:
                wait_s = max(0.0, now_wallclock - float(state.get("pause_started_wallclock_at") or now_wallclock))
                state["last_active_emit_at"] = now_wallclock
                self.logger.log_event(
                    self.time,
                    "planner_barrier_active",
                    {
                        "cognition_pause_reason": state.get("reason"),
                        "blocking_request_ids": list(state.get("blocking_request_ids", [])),
                        "blocking_agent_ids": list(state.get("blocking_agent_ids", [])),
                        "wallclock_wait_s": wait_s,
                    },
                )
        elif (not barrier_active) and state.get("active"):
            wait_s = max(0.0, now_wallclock - float(state.get("pause_started_wallclock_at") or now_wallclock))
            state["total_wallclock_wait_s"] = float(state.get("total_wallclock_wait_s", 0.0)) + wait_s
            state["last_wallclock_wait_s"] = wait_s
            self.logger.log_event(
                self.time,
                "planner_barrier_resolved",
                {
                    "cognition_pause_reason": state.get("reason"),
                    "blocking_request_ids": list(state.get("blocking_request_ids", [])),
                    "blocking_agent_ids": list(state.get("blocking_agent_ids", [])),
                    "wallclock_wait_s": wait_s,
                },
            )
            self.logger.log_event(
                self.time,
                "planner_barrier_released",
                {
                    "wallclock_wait_s": wait_s,
                    "planner_barrier_pause_count": int(state.get("pause_count", 0)),
                    "planner_barrier_total_wallclock_wait_s": float(state.get("total_wallclock_wait_s", 0.0)),
                },
            )
            state["active"] = False
            state["reason"] = None
            state["blocking_agent_ids"] = []
            state["blocking_request_ids"] = []
            state["pause_started_wallclock_at"] = None
            state["pause_started_sim_time"] = None
            state["pause_started_tick"] = None
            state["last_active_emit_at"] = 0.0

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
                    "fallback_hint": getattr(provider, "last_outcome", {}).get("hint"),
                    "local_model_name": self.brain_backend_config.local_model if configured != "rule_brain" else None,
                    "local_base_url": self.brain_backend_config.local_base_url if configured != "rule_brain" else None,
                    "local_endpoint": self.brain_backend_config.local_endpoint if configured != "rule_brain" else None,
                    "timeout_s": self.brain_backend_config.timeout_s if configured != "rule_brain" else None,
                    "fallback_count": self.backend_fallback_count,
                },
            )
            if self.backend_fallback_count >= 3:
                self.logger.log_event(
                    self.time,
                    "repeated_backend_fallback_detected",
                    {
                        "configured_brain_backend": configured,
                        "effective_brain_backend": effective,
                        "reason": fallback_reason,
                        "repetition_count": self.backend_fallback_count,
                        "window_size": 3,
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
        for agent in self.agents:
            agent.current_time = self.time
            agent._check_inflight_timeout(self)
            agent._poll_planner_request(self, self.environment)
        self._refresh_planner_barrier_state()
        if self.planner_barrier_state.get("active"):
            for agent in self.agents:
                self.logger.log_agent_state(self.time, agent)
            return

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
            if self.environment.construction.is_agent_transporting(agent.name):
                self.logger.log_event(
                    self.time,
                    "agent_occupied_transport",
                    {"agent": agent.name},
                )
                self.logger.log_agent_state(self.time, agent)
                continue
            agent.update(dt, self.environment, sim_state=self, planner_lifecycle_already_polled=True)
            agent.compare_and_repair_construction(self.environment.construction, sim_state=self)
            self.refresh_agent_backend_effective_state(agent, reason="planner_call")
            self.logger.log_agent_state(self.time, agent)

        self.metrics.on_step(dt)

        self.time += dt

        if self.flash_mode or (self.time - self._last_save_time >= self.save_interval):
            self.logger.save_csv()
            self._last_save_time = self.time

    def stop(self):
        self.runtime_witness_audit_result = self.runtime_witness_audit.finalize()
        self.metrics.finalize()
        self.logger.update_session_manifest(
            extra_metadata={
                **self._backend_settings_for_manifest(),
                **self.execution_metadata,
            }
        )
        self.logger.save_csv()
        self.planner_executor.shutdown(wait=False, cancel_futures=True)


    def _distance(self, p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return math.hypot(dx, dy)
