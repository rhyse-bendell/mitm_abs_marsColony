# File: modules/agent.py

import math
import random
import threading
import time
import uuid
from dataclasses import dataclass

try:
    import matplotlib.patches as patches
except ImportError:
    patches = None

from modules.knowledge import Data, Information, Knowledge
from modules.action_schema import (
    BrainDecision,
    CommunicationIntent,
    ExecutableActionType,
    InternalEventType,
    LEGACY_COMMUNICATION_TYPE_MAP,
    validate_brain_decision,
)
from modules.brain_contract import (
    AgentBrainRequest,
    AgentBrainResponse,
    AgentDIKIntegrationRequest,
    validate_agent_brain_response,
)
from modules.brain_provider import select_productive_fallback_action
from modules.goal_manager import GoalManager
from modules.goal_state import GOAL_SOURCES, GOAL_STATUSES
from modules.plan_state import PlanRecord
from modules.task_model import normalize_rule_token

DIK_LOG = []

ROLE_COLORS = {
    "Architect": "red",
    "Engineer": "blue",
    "Botanist": "green"
}

COMMUNICATION_RADIUS = 20  # how close agents must be to talk

# Communication types (based on your codebook)
MESSAGE_TYPES = {
    "TDP": "Team Data Provision",
    "TIP": "Team Information Provision",
    "TKP": "Team Knowledge Provision",
    "TGTO": "Team Goal/Task Objective",
    "TKRQ": "Team Knowledge Request",
    "TCR": "Team Correction or Repair"
}

@dataclass
class PlannerCadenceConfig:
    planner_enabled: bool = True
    planner_interval_steps: int = 4
    planner_interval_time: float = 3.0
    planner_trigger_mask: set[str] | None = None
    planner_timeout_seconds: float = 1.5
    planner_fallback_backend: str = "rule_brain"
    planner_max_retries: int = 0
    explanation_mode: str = "never"
    explanation_every_n_calls: int = 5
    explanation_probability: float = 0.1
    degraded_consecutive_failures_threshold: int = 3
    degraded_cooldown_seconds: float = 12.0
    degraded_step_interval_multiplier: float = 2.0
    high_latency_local_llm_mode: bool = False
    unrestricted_local_qwen_mode: bool = False
    high_latency_stale_result_grace_s: float = 0.0
    sticky_backend_demotion_enabled: bool = False
    planner_blocks_sim_time: bool | None = None
    planner_request_policy: str = "cadence_with_dik_integration"
    split_mode_planning_interval_steps: int = 60
    split_mode_dik_integration_cooldown_steps: int = 12
    split_mode_dik_batch_threshold: int = 2

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload or {})
        mask = payload.get("planner_trigger_mask")
        if isinstance(mask, str):
            mask = {m.strip() for m in mask.split("|") if m.strip()}
        elif isinstance(mask, (list, tuple, set)):
            mask = {str(m) for m in mask if str(m).strip()}
        else:
            mask = None
        return cls(
            planner_enabled=bool(payload.get("planner_enabled", True)),
            planner_interval_steps=max(1, int(payload.get("planner_interval_steps", 4))),
            planner_interval_time=max(0.0, float(payload.get("planner_interval_time", 3.0))),
            planner_trigger_mask=mask,
            planner_timeout_seconds=max(0.1, float(payload.get("planner_timeout_seconds", 1.5))),
            planner_fallback_backend=str(payload.get("planner_fallback_backend", "rule_brain")),
            planner_max_retries=max(0, int(payload.get("planner_max_retries", 0))),
            explanation_mode=str(payload.get("explanation_mode", "never")).lower(),
            explanation_every_n_calls=max(1, int(payload.get("explanation_every_n_calls", 5))),
            explanation_probability=max(0.0, min(1.0, float(payload.get("explanation_probability", 0.1)))),
            degraded_consecutive_failures_threshold=max(1, int(payload.get("degraded_consecutive_failures_threshold", 3))),
            degraded_cooldown_seconds=max(0.0, float(payload.get("degraded_cooldown_seconds", 12.0))),
            degraded_step_interval_multiplier=max(1.0, float(payload.get("degraded_step_interval_multiplier", 2.0))),
            high_latency_local_llm_mode=bool(payload.get("high_latency_local_llm_mode", False)),
            unrestricted_local_qwen_mode=bool(payload.get("unrestricted_local_qwen_mode", False)),
            high_latency_stale_result_grace_s=max(0.0, float(payload.get("high_latency_stale_result_grace_s", 0.0))),
            sticky_backend_demotion_enabled=bool(
                payload.get(
                    "sticky_backend_demotion_enabled",
                    payload.get("allow_persistent_backend_demotion", False),
                )
            ),
            planner_blocks_sim_time=(
                bool(payload.get("planner_blocks_sim_time"))
                if payload.get("planner_blocks_sim_time") is not None
                else None
            ),
            planner_request_policy=str(payload.get("planner_request_policy", "cadence_with_dik_integration") or "cadence_with_dik_integration").lower(),
            split_mode_planning_interval_steps=max(1, int(payload.get("planning_interval_steps", payload.get("split_mode_planning_interval_steps", 60)) or 60)),
            split_mode_dik_integration_cooldown_steps=max(1, int(payload.get("dik_integration_cooldown_steps", payload.get("split_mode_dik_integration_cooldown_steps", 12)) or 12)),
            split_mode_dik_batch_threshold=max(1, int(payload.get("dik_integration_batch_threshold", payload.get("split_mode_dik_batch_threshold", 2)) or 2)),
        )


class Agent:
    def __init__(self, name, role, position=(0.0, 0.0), orientation=0.0, speed=1.0, planner_config=None, agent_id=None, display_name=None, agent_label=None, template_id=None, brain_config=None, communication_params=None, initial_goal_seeds=None):
        self.name = name
        self.role = role
        self.agent_id = agent_id or name
        self.display_name = display_name or name
        self.agent_label = agent_label
        self.template_id = template_id
        self.brain_config = dict(brain_config or {})
        self.communication_params = dict(communication_params or {})
        self.initial_goal_seeds = list(initial_goal_seeds or [])
        self.position = position
        self.orientation = orientation
        self.speed = speed
        self.active_actions = []  # List of {"type": ..., "duration": ..., "progress": ...}

        # Physiological state
        self.heart_rate = 70
        self.gsr = 0.01
        self.temperature = 98.6
        self.co2_output = 0.04

        # Taskwork-related state
        self.inventory = []
        self.activity_log = []
        self.goal_stack = []  # Legacy compatibility mirror of active/queued goals
        self.goal_registry = {}
        self.goal_order = []
        self.goal = None  # Synchronized with highest-priority active goal
        self.goal_status_history = []
        self.goal_transition_counter = 0
        self.goal_manager = GoalManager(self)
        self.target = None
        self.spawn_position = tuple(position)
        self.has_shared = False
        self.detour_target = None

        # Cognitive state
        self.data_memory = set()
        self.information_memory = set()
        self.knowledge_memory = set()
        self.known_gaps = set()
        self.communication_log = []
        self.memory_seen_packets = set()
        self.source_inspection_state = {}
        self.source_exhaustion_state = {}
        self.inspect_stall_counts = {}
        self.current_inspect_target_id = None
        self.inspect_session = {
            "source_id": None,
            "target": None,
            "state": "idle",
            "started_at": None,
            "last_updated_at": None,
            "restarts": 0,
        }
        self.inspect_pursuit = {
            "action_type": None,
            "source_id": None,
            "slot_id": None,
            "target_position": None,
            "started_at": None,
            "expires_at": None,
            "blocked_attempts": 0,
            "no_progress_ticks": 0,
            "last_distance_to_target": None,
        }
        self.inspect_pursuit_lease_seconds = 10.0
        self.inspect_pursuit_no_progress_limit = 6
        self.inspect_pursuit_blocked_attempt_limit = 8
        self.source_access_state = {
            "source_id": None,
            "slot_id": None,
            "slot_position": None,
            "target_kind": None,
            "blocked_attempts": 0,
        }
        self.post_inspect_handoff = {
            "pending": False,
            "source_id": None,
            "dik_changed": False,
            "readiness_changed": False,
            "blockers": [],
            "blocker_category": None,
            "outcome": None,
            "expires_at": 0.0,
        }
        self.status_last_action = ""
        self.construction_validation_state = {"mismatch_last_ts": {}, "repair_last_ts": {}}

        # Team dynamics
        self.shared_knowledge = set()
        self.emotion_state = "neutral"
        self.current_action = None

        # Mental Model
        self.mental_model = {
            "data": set(),
            "information": set(),
            "knowledge": Knowledge()
        }

        # Theory of Mind (ToM) about teammates
        self.theory_of_mind = {}  # {agent_name: {"goals": [], "knowledge": set(), "last_seen": time}}

        # Brain-backed decision routing state
        self.current_plan = None
        self.plan_counter = 0
        self.plan_expiry_s = 12.0
        self.plan_review_interval_s = 3.0
        self.last_phase_name = None
        self.last_info_count = 0
        self.last_knowledge_count = 0
        self.last_build_readiness = 0
        self.last_team_update_count = 0
        self.last_replan_time = None
        self.last_phase_stage = "early"
        self.last_build_blockers = []
        self.inventory_resources = {"bricks": 0}
        self.transport_state = {
            "stage": "idle",
            "carrying": {"resource_type": None, "quantity": 0},
            "pickup_source_id": None,
            "bound_project_id": None,
        }
        self.history_compaction = {
            "recent_history_summary": "",
            "plan_evolution": [],
            "compaction_count": 0,
        }
        self.planner_cadence = PlannerCadenceConfig.from_dict(planner_config)
        self.sim_step_count = 0
        self.last_planner_step = -1
        self.last_planner_time = -1.0
        self.planner_call_count = 0
        self.loop_counters = {"action_signature": None, "action_repeats": 0, "plan_signature": None, "plan_repeats": 0, "target_failures": {}}
        self.selection_loop_guard = {"last_action": None, "consecutive_count": 0}
        self.control_state = {
            "mode": "BOOTSTRAP",
            "previous_mode": None,
            "mode_entered_step": 0,
            "mode_dwell_steps": 0,
            "last_transition_reason": "agent_initialized",
            "last_transition_features": {},
            "mode_history": [{"step": 0, "mode": "BOOTSTRAP", "reason": "agent_initialized"}],
            "transition_history": [],
            "recovery_active": False,
            "last_policy_snapshot": {},
        }
        # Durable fallback method/step state (simulator authoritative; bounded history).
        self.active_method_id = None
        self.active_method_instance = None
        self.active_method_step = None
        self.method_started_tick = None
        self.step_started_tick = None
        self.step_retry_count = 0
        self.recent_step_outcomes = []
        self.method_history = []
        self.method_transition_history = []
        self.abandoned_methods = []
        self.method_cooldowns = {}
        self.source_cooldowns = {}
        self.source_exhaustion = {}
        self.last_method_switch_reason = None
        self._planner_request_seq = 0
        self.planner_state = {
            "status": "idle",
            "request_id": None,
            "request_tick": None,
            "requested_at": None,
            "requested_wallclock_at": None,
            "completed_at": None,
            "error": None,
            "last_latency_s": None,
            "last_result": None,
            "last_result_request_id": None,
            "degraded_mode": False,
            "cooldown_until": 0.0,
            "consecutive_failures": 0,
            "total_started": 0,
            "total_completed": 0,
            "total_timed_out": 0,
            "total_failed": 0,
            "total_skipped_inflight": 0,
            "consecutive_inflight_skips": 0,
            "total_skipped_cooldown": 0,
            "total_stale_discarded": 0,
            "stale_plan_reuse_count": 0,
            "ui_safe_fallback_count": 0,
            "degraded_mode_episodes": 0,
            "consecutive_failure_sum": 0,
            "consecutive_failure_samples": 0,
            "requests_completed_with_llm": 0,
            "requests_completed_with_fallback": 0,
            "llm_success_count": 0,
            "llm_timeout_count": 0,
            "llm_invalid_count": 0,
            "llm_transport_error_count": 0,
            "fallback_generated_count": 0,
            "fallback_adopted_count": 0,
            "fallback_rejected_count": 0,
            "productive_fallback_action_count": 0,
            "idle_fallback_action_count": 0,
            "fallback_only_ticks": 0,
            "startup_target_resolution_failures": 0,
            "startup_movement_blockers": 0,
            "startup_plan_invalidations": 0,
            "blocking_sim_barrier": False,
            "barrier_reason": None,
        }
        self._planner_future = None
        self._planner_future_lock = threading.Lock()
        self._timed_out_request_ids = set()
        self.dik_integration_state = {
            "status": "idle",
            "request_id": None,
            "trigger_reason": None,
            "request_payload": None,
            "last_completed_step": -1,
            "total_started": 0,
            "total_completed": 0,
            "total_rejected": 0,
            "last_result": None,
            "accepted_updates": {"information": [], "knowledge": [], "rules": []},
            "recent_candidates": {"information": [], "knowledge": [], "rules": []},
            "last_sent_held_count": 0,
            "last_sent_comm_count": 0,
            "last_sent_artifact_count": 0,
        }
        self._dik_future = None
        self._dik_future_lock = threading.Lock()
        self.startup_state = {
            "initial_goal_selected": False,
            "initial_plan_selected": False,
            "first_productive_action_started": False,
            "first_movement_started": False,
            "left_spawn": False,
            "initial_goal_time": None,
            "initial_plan_time": None,
            "first_productive_action_time": None,
            "first_movement_time": None,
            "left_spawn_time": None,
        }
        self.fallback_bootstrap = {
            "active": False,
            "required_sources": [],
            "stage": "shared",
            "activation_reason": None,
            "activated_at": None,
            "completed_at": None,
            "runtime_fallback_triggers": 0,
            "last_forced_action": None,
        }
        self.navigation = {
            "path_mode": str((planner_config or {}).get("path_mode", "grid_astar") or "grid_astar"),
            "ignore_agent_collision": bool((planner_config or {}).get("ignore_agent_collision", True)),
            "active_path": [],
            "path_target": None,
            "path_index": 0,
            "retry_count": 0,
            "last_blocker_category": None,
            "last_position": tuple(position),
            "last_target": None,
            "last_arrival_position": tuple(position),
        }


        # Experiment-facing trait defaults
        self.communication_propensity = 0.5
        self.goal_alignment = 0.5
        self.help_tendency = 0.5
        self.build_speed = 0.5
        self.rule_accuracy = 0.5

        self.construct_values = {}
        self.mechanism_profile = {}
        self.hook_effects = {}

        self.last_dik_change_time = -1.0
        self.executed_derivations = set()
        self.derivation_events = []
        self.derivation_attempt_counts = {}
        self._construction_attempted_projects = set()
        self.support_goal_activation_state = {}
        self.support_goal_nonexec_counts = {}
        self.task_model = None


    def _emit_event(self, sim_state, event_type, payload):
        if sim_state is None:
            return
        enriched = {"agent_id": self.agent_id, "agent": self.name, **dict(payload or {})}
        sim_state.logger.log_event(sim_state.time, event_type, enriched)

    def _normalize_packet_name(self, packet_name):
        """Map UI packet labels and aliases to canonical environment packet keys."""
        mapping = {
            "Team_Packet": "Team_Info",
            "Architect_Packet": "Architect_Info",
            "Engineer_Packet": "Engineer_Info",
            "Botanist_Packet": "Botanist_Info",
        }
        return mapping.get(packet_name, packet_name)

    def _has_packet_access(self, packet_name):
        allowed = getattr(self, "allowed_packet", None)
        if allowed is None:
            return True

        if isinstance(allowed, str):
            allowed = [allowed]

        normalized_allowed = {self._normalize_packet_name(p) for p in allowed}
        return packet_name in normalized_allowed

    def _choose_info_target(self, environment):
        """Select an information interaction target using soft role-aware scoring."""
        candidates = []
        for packet_name in environment.knowledge_packets.keys():
            if not self._has_packet_access(packet_name):
                continue

            target = environment.get_interaction_target_position(packet_name, from_position=self.position)
            if target is None:
                continue

            distance = math.hypot(target[0] - self.position[0], target[1] - self.position[1])
            score = -distance
            if packet_name == f"{self.role}_Info":
                score += 2.0
            if packet_name == "Team_Info":
                score += 1.5

            candidates.append((score, packet_name, target))

        if not candidates:
            return self.position

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0]
        self.activity_log.append(f"Selected info target {chosen[1]} (score={chosen[0]:.2f})")
        return chosen[2]

    def _set_status(self, message, log_activity=True):
        self.status_last_action = message
        if log_activity:
            self.activity_log.append(message)

    def get_control_state_snapshot(self):
        control = dict(self.control_state or {})
        snapshot = dict(control.get("last_policy_snapshot") or {})
        top_features = dict(snapshot.get("top_features") or control.get("last_transition_features") or {})
        method_state = dict(control.get("method_state") or {})
        return {
            "mode": str(control.get("mode") or "BOOTSTRAP"),
            "previous_mode": control.get("previous_mode"),
            "mode_dwell_steps": int(control.get("mode_dwell_steps", 0) or 0),
            "last_transition_reason": str(control.get("last_transition_reason") or "none"),
            "last_transition_features": dict(control.get("last_transition_features") or {}),
            "recovery_active": bool(control.get("recovery_active")),
            "top_features": top_features,
            "policy_snapshot": snapshot,
            "mode_history": list(control.get("mode_history", [])),
            "transition_history": list(control.get("transition_history", [])),
            "method_state": method_state,
        }

    def _sync_method_state_from_control(self):
        method_state = dict((self.control_state or {}).get("method_state") or {})
        self.active_method_id = method_state.get("active_method_id")
        self.active_method_instance = method_state.get("active_method_instance")
        self.active_method_step = method_state.get("active_method_step")
        self.method_started_tick = method_state.get("method_started_tick")
        self.step_started_tick = method_state.get("step_started_tick")
        self.step_retry_count = int(method_state.get("step_retry_count", 0) or 0)
        self.recent_step_outcomes = list(method_state.get("recent_step_outcomes", []))[-8:]
        self.method_history = list(method_state.get("method_history", []))[-12:]
        self.method_transition_history = list(method_state.get("method_transition_history", []))[-12:]
        self.abandoned_methods = list(method_state.get("abandoned_methods", []))[-8:]
        self.method_cooldowns = dict(method_state.get("method_cooldowns", {}))
        self.source_cooldowns = dict(method_state.get("source_cooldowns", {}))
        self.source_exhaustion = dict(method_state.get("source_exhaustion", {}))
        self.last_method_switch_reason = method_state.get("last_method_switch_reason")

    def get_runtime_state_snapshot(self):
        current_goals = [g for g in list(self.goal_stack or []) if isinstance(g, dict)]
        top_goals = current_goals[:3]
        control_snapshot = self.get_control_state_snapshot()
        planner_last_result = dict(self.planner_state.get("last_result") or {})
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name or self.name,
            "role": self.role,
            "control_state": control_snapshot,
            "method_state": dict(control_snapshot.get("method_state") or {}),
            "planner_state": dict(self.planner_state or {}),
            "dik_integration_state": dict(self.dik_integration_state or {}),
            "fallback_bootstrap": dict(self.fallback_bootstrap or {}),
            "inspect_session": dict(self.inspect_session or {}),
            "inspect_pursuit": dict(self.inspect_pursuit or {}),
            "transport_state": dict(self.transport_state or {}),
            "current_goal": self.goal,
            "top_goals": top_goals,
            "current_plan_id": getattr(self.current_plan, "plan_id", None),
            "current_plan_method": getattr(self.current_plan, "plan_method_id", None),
            "next_action": getattr(getattr(self.current_plan, "decision", None), "selected_action", None),
            "current_target": self.target,
            "last_status": self.status_last_action or (self.activity_log[-1] if self.activity_log else ""),
            "last_action": planner_last_result.get("next_action", {}),
        }

    def get_agent_state_summary(self):
        snapshot = self.get_runtime_state_snapshot()
        control = snapshot["control_state"]
        planner = snapshot["planner_state"]
        return {
            "identity": f"{snapshot['display_name']} ({snapshot['role']})",
            "macro_mode": control.get("mode"),
            "active_method": snapshot.get("method_state", {}).get("active_method_id"),
            "active_step": snapshot.get("method_state", {}).get("active_method_step"),
            "previous_mode": control.get("previous_mode"),
            "mode_dwell_steps": control.get("mode_dwell_steps"),
            "planner_status": planner.get("status"),
            "dik_status": snapshot["dik_integration_state"].get("status"),
            "transport_stage": snapshot["transport_state"].get("stage"),
            "inspect_state": snapshot["inspect_session"].get("state"),
            "target": snapshot.get("current_target"),
            "last_status": snapshot.get("last_status"),
        }

    def _emit_startup_once(self, sim_state, flag_name, event_type, payload=None):
        if sim_state is None:
            return
        if not self.startup_state.get(flag_name):
            self.startup_state[flag_name] = True
            self.startup_state[f"{flag_name}_time"] = float(sim_state.time)
            self._emit_event(sim_state, event_type, payload or {})

    def _fallback_bootstrap_required_sources(self):
        role_source = f"{self.role}_Info"
        allowed = getattr(self, "allowed_packet", None)
        if isinstance(allowed, str):
            allowed = [allowed]
        normalized_allowed = {self._normalize_packet_name(p) for p in (allowed or [])}
        required = ["Team_Info"]
        if (not normalized_allowed) or role_source in normalized_allowed:
            required.append(role_source)
        return required

    def activate_fallback_bootstrap(self, sim_state=None, reason="runtime_fallback"):
        required_sources = self._fallback_bootstrap_required_sources()
        if not self.fallback_bootstrap.get("active"):
            self.fallback_bootstrap.update(
                {
                    "active": True,
                    "required_sources": list(required_sources),
                    "stage": "shared",
                    "activation_reason": reason,
                    "activated_at": float(getattr(sim_state, "time", getattr(self, "current_time", 0.0))),
                    "completed_at": None,
                }
            )
            self._emit_event(
                sim_state,
                "fallback_bootstrap_mode_activated",
                {"reason": reason, "required_sources": list(required_sources), "left_spawn": bool(self.startup_state.get("left_spawn"))},
            )
        elif required_sources:
            merged = []
            seen = set()
            for source_id in list(self.fallback_bootstrap.get("required_sources", [])) + list(required_sources):
                if source_id in seen:
                    continue
                seen.add(source_id)
                merged.append(source_id)
            self.fallback_bootstrap["required_sources"] = merged

    def _is_source_bootstrap_satisfied(self, source_id):
        if not source_id:
            return True
        if self.source_inspection_state.get(source_id) == "inspected":
            return True
        if bool(self.source_exhaustion_state.get(source_id, {}).get("exhausted")):
            return True
        return False

    def _fallback_bootstrap_complete(self, sim_state=None):
        required_sources = list(self.fallback_bootstrap.get("required_sources", []))
        completed_all = all(self._is_source_bootstrap_satisfied(source_id) for source_id in required_sources)
        return bool(self.startup_state.get("left_spawn")) and completed_all

    def clear_planner_inflight_state(self, sim_state=None, reason="manual_reset"):
        with self._planner_future_lock:
            fut = self._planner_future
            self._planner_future = None
        if fut is not None and not fut.done():
            fut.cancel()
        if self.planner_state.get("status") == "in_flight":
            self._emit_event(
                sim_state,
                "planner_inflight_cleared",
                {"agent": self.name, "request_id": self.planner_state.get("request_id"), "reason": reason},
            )
        self.planner_state["status"] = "idle"
        self.planner_state["request_id"] = None
        self.planner_state["requested_wallclock_at"] = None
        self.planner_state["requested_at"] = None
        self.planner_state["request_tick"] = None
        self.planner_state["consecutive_inflight_skips"] = 0
        self.planner_state["blocking_sim_barrier"] = False
        self.planner_state["barrier_reason"] = None

    def planner_request_blocks_sim_time(self, sim_state=None, runtime=None):
        explicit = self.planner_cadence.planner_blocks_sim_time
        if explicit is not None:
            return bool(explicit)
        runtime = runtime or (sim_state.get_agent_brain_runtime(self) if sim_state is not None and hasattr(sim_state, "get_agent_brain_runtime") else {})
        configured_backend = str(runtime.get("configured_backend", self.brain_config.get("backend", "rule_brain")) or "rule_brain").lower()
        effective_backend = str(runtime.get("effective_backend", configured_backend) or configured_backend).lower()
        if self.planner_cadence.unrestricted_local_qwen_mode or self.planner_cadence.high_latency_local_llm_mode:
            return True
        if configured_backend in {"ollama", "ollama_local"} or effective_backend in {"ollama", "ollama_local"}:
            return True
        return False

    def _maybe_hard_demote_backend(self, sim_state, reason, activate_bootstrap=True):
        if sim_state is None or not hasattr(sim_state, "hard_demote_agent_backend"):
            return False
        allow_sticky = True
        if hasattr(sim_state, "sticky_backend_demotion_enabled"):
            allow_sticky = bool(sim_state.sticky_backend_demotion_enabled(self))
        if not allow_sticky:
            self._emit_event(
                sim_state,
                "backend_hard_demotion_skipped",
                {"agent_id": self.agent_id, "reason": str(reason), "sticky_backend_demotion_enabled": False},
            )
            return False
        return bool(sim_state.hard_demote_agent_backend(self, reason=reason, activate_bootstrap=activate_bootstrap))

    def _next_bootstrap_source(self, environment):
        stage = str(self.fallback_bootstrap.get("stage") or "shared")
        required_sources = list(self.fallback_bootstrap.get("required_sources", []))
        team_source = "Team_Info" if "Team_Info" in required_sources else None
        role_source = next((s for s in required_sources if s != "Team_Info"), None)

        if stage == "shared":
            if self._is_source_bootstrap_satisfied(team_source):
                self.fallback_bootstrap["stage"] = "role"
                stage = "role"
            elif team_source and self._has_packet_access(team_source) and team_source in environment.knowledge_packets:
                return team_source
            else:
                return None

        if stage == "role":
            if not role_source:
                self.fallback_bootstrap["stage"] = "complete"
                return None
            if self._is_source_bootstrap_satisfied(role_source):
                self.fallback_bootstrap["stage"] = "complete"
                return None
            if self._has_packet_access(role_source) and role_source in environment.knowledge_packets:
                return role_source
            return None
        if stage == "complete":
            return None

        required_sources = list(self.fallback_bootstrap.get("required_sources", []))
        for source_id in required_sources:
            if not self._has_packet_access(source_id):
                continue
            status = self.source_inspection_state.get(source_id)
            if status == "inspected":
                continue
            if status == "in_progress" and int(self.inspect_stall_counts.get(source_id, 0)) >= 2:
                continue
            if (
                source_id == self.source_access_state.get("source_id")
                and int(self.source_access_state.get("blocked_attempts", 0)) >= 3
            ):
                continue
            if source_id in environment.knowledge_packets:
                return source_id
        candidates = self._candidate_information_sources(environment)
        if candidates:
            return candidates[0][1]
        return None

    def _bootstrap_override_decision(self, environment, sim_state=None):
        if not self.fallback_bootstrap.get("active"):
            return None
        if self._fallback_bootstrap_complete(sim_state=sim_state):
            self.fallback_bootstrap["active"] = False
            self.fallback_bootstrap["completed_at"] = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
            self.clear_planner_inflight_state(sim_state=sim_state, reason="bootstrap_completed")
            self.current_plan = None
            self.last_planner_step = -1
            self.last_planner_time = -1.0
            self._emit_event(
                sim_state,
                "fallback_bootstrap_mode_completed",
                {"required_sources": list(self.fallback_bootstrap.get("required_sources", []))},
            )
            return None
        source_id = self._next_bootstrap_source(environment)
        if source_id:
            if isinstance(getattr(self, "post_inspect_handoff", None), dict):
                self.post_inspect_handoff["pending"] = False
            self.fallback_bootstrap["last_forced_action"] = "inspect_information_source"
            return BrainDecision(
                selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                target_id=source_id,
                reason_summary="forced mission bootstrap for fallback continuity",
                confidence=0.95,
            )
        self.fallback_bootstrap["last_forced_action"] = "observe_environment"
        return BrainDecision(
            selected_action=ExecutableActionType.OBSERVE_ENVIRONMENT,
            reason_summary="bootstrap mode awaiting source accessibility",
            confidence=0.7,
        )

    def _trait_value(self, name, default=0.5):
        value = getattr(self, name, default)
        return max(0.0, min(1.0, float(value)))

    def _hook_value(self, hook_type, hook_target, parameter, default=None):
        if default is None:
            default = 0.0 if parameter in {"utility_weight", "priority_weight", "externalization_weight", "persistence_weight", "adoption_weight", "sensitivity"} else 1.0
        return float(getattr(self, "hook_effects", {}).get((hook_type, hook_target, parameter), default))

    def _epistemic_success_probability(self, hook_target, *, base_trait="rule_accuracy", context_modifier=0.0, retry_bonus=0.0):
        hook_base = self._hook_value("dik_update", hook_target, "success_probability", default=0.5)
        trait_base = self._trait_value(base_trait, default=0.5)
        mechanism_base = max(hook_base, trait_base)
        deterministic_component = 0.2 + (0.75 * mechanism_base)
        residual_noise = random.uniform(-0.04, 0.04)
        probability = deterministic_component + context_modifier + retry_bonus + residual_noise
        return max(0.05, min(0.98, probability))

    def _attempt_epistemic_transition(
        self,
        *,
        hook_target,
        sim_state=None,
        event_payload=None,
        attempt_event,
        failed_event,
        context_modifier=0.0,
        retry_bonus=0.0,
    ):
        probability = self._epistemic_success_probability(
            hook_target,
            context_modifier=context_modifier,
            retry_bonus=retry_bonus,
        )
        roll = random.random()
        payload = dict(event_payload or {})
        payload.update(
            {
                "hook_target": hook_target,
                "success_probability": round(probability, 4),
                "stochastic_roll": round(roll, 4),
                "context_modifier": round(context_modifier, 4),
                "retry_bonus": round(retry_bonus, 4),
            }
        )
        if sim_state is not None:
            self._emit_event(sim_state, attempt_event, payload)
        if roll <= probability:
            return True, payload
        if sim_state is not None:
            self._emit_event(sim_state, failed_event, payload)
        return False, payload

    def _derivation_context_modifier(self, derivation):
        if not getattr(self, "task_model", None):
            return 0.0
        required = set(derivation.required_inputs)
        role_scopes = set()
        for required_id in required:
            element = self.task_model.dik_elements.get(required_id)
            if element is None:
                continue
            scope = (element.role_scope or "").strip().lower()
            if scope:
                role_scopes.add(scope)
        role_key = (self.role or "").strip().lower()
        local_scopes = {"", "all", "team", "shared", role_key}
        external_role_dependencies = [scope for scope in role_scopes if scope not in local_scopes]
        if external_role_dependencies:
            return -0.08
        if "shared" in role_scopes or "team" in role_scopes:
            return -0.03
        return 0.0

    def _duration_scale(self, action_name):
        scale = self._hook_value("action_duration", action_name, "duration_scale", default=1.0)
        return max(0.2, float(scale))

    def _readiness_threshold(self):
        shift = self._hook_value("decision_threshold", "start_construction", "readiness_threshold", default=0.0)
        return max(1.0, 3.0 + shift)

    def _scaled_duration(self, base_duration):
        build_speed = self._trait_value("build_speed", 0.5)
        factor = 1.25 - (0.75 * build_speed)
        return max(0.2, round(base_duration * factor, 3))

    def _help_context_available(self, sim_state):
        for teammate_name, model in self.theory_of_mind.items():
            goals = model.get("goals", [])
            if any(g in {"stalled", "idle"} for g in goals):
                return True
        return bool(sim_state and len(self.known_gaps) > 0)

    def _ensure_source_state(self, environment):
        for packet_name in environment.knowledge_packets.keys():
            self.source_inspection_state.setdefault(packet_name, "unseen")
            self.source_exhaustion_state.setdefault(
                packet_name,
                {"inspect_count": 0, "last_dik_changed": None, "no_new_dik_streak": 0, "inspected": False, "exhausted": False},
            )

    def _release_source_slot(self, environment, source_id=None, emit=False, sim_state=None, reason="released"):
        source = source_id or self.source_access_state.get("source_id")
        slot_id = self.source_access_state.get("slot_id")
        if not source or not hasattr(environment, "release_source_access_slot"):
            return
        released = environment.release_source_access_slot(source, agent_id=self.agent_id, slot_id=slot_id)
        if released and emit:
            self._emit_event(sim_state, "source_slot_released", {"source_id": source, "slot_id": slot_id, "reason": reason})
        self.source_access_state = {
            "source_id": None,
            "slot_id": None,
            "slot_position": None,
            "target_kind": None,
            "blocked_attempts": 0,
        }

    def _clear_inspect_pursuit(self, reason=None, sim_state=None, release_slot=False, environment=None):
        source_id = self.inspect_pursuit.get("source_id")
        if release_slot and source_id and environment is not None:
            self._release_source_slot(environment, source_id=source_id, emit=True, sim_state=sim_state, reason=reason or "pursuit_cleared")
        self.inspect_pursuit = {
            "action_type": None,
            "source_id": None,
            "slot_id": None,
            "target_position": None,
            "started_at": None,
            "expires_at": None,
            "blocked_attempts": 0,
            "no_progress_ticks": 0,
            "last_distance_to_target": None,
        }
        if source_id and sim_state is not None:
            self._emit_event(sim_state, "pursuit_abandoned", {"source_id": source_id, "reason": reason or "cleared"})

    def _inspect_pursuit_active_for(self, source_id, now_ts):
        if self.inspect_pursuit.get("action_type") != ExecutableActionType.INSPECT_INFORMATION_SOURCE.value:
            return False
        if self.inspect_pursuit.get("source_id") != source_id:
            return False
        expires_at = self.inspect_pursuit.get("expires_at")
        return expires_at is not None and float(expires_at) >= float(now_ts)

    def _commit_inspect_pursuit(self, source_id, target_position, now_ts, slot_id=None, sim_state=None):
        self.inspect_pursuit.update(
            {
                "action_type": ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
                "source_id": source_id,
                "slot_id": slot_id,
                "target_position": target_position,
                "started_at": float(now_ts),
                "expires_at": float(now_ts) + float(self.inspect_pursuit_lease_seconds),
                "blocked_attempts": 0,
                "no_progress_ticks": 0,
                "last_distance_to_target": None,
            }
        )
        if sim_state is not None:
            self._emit_event(
                sim_state,
                "pursuit_committed",
                {
                    "source_id": source_id,
                    "target": target_position,
                    "slot_id": slot_id,
                    "lease_seconds": self.inspect_pursuit_lease_seconds,
                    "expires_at": self.inspect_pursuit.get("expires_at"),
                },
            )

    def _mark_inspect_pursuit_progress(self, source_id, distance, sim_state=None):
        if self.inspect_pursuit.get("source_id") != source_id:
            return
        previous = self.inspect_pursuit.get("last_distance_to_target")
        if previous is not None and distance <= (float(previous) - 0.05):
            self.inspect_pursuit["no_progress_ticks"] = 0
            if sim_state is not None:
                self._emit_event(
                    sim_state,
                    "pursuit_progressed",
                    {"source_id": source_id, "last_distance": round(float(previous), 4), "distance": round(float(distance), 4)},
                )
        elif previous is not None:
            self.inspect_pursuit["no_progress_ticks"] = int(self.inspect_pursuit.get("no_progress_ticks", 0)) + 1
        self.inspect_pursuit["last_distance_to_target"] = float(distance)

    def _expire_or_stall_inspect_pursuit(self, source_id, now_ts, sim_state=None, environment=None, reason_hint=None):
        if self.inspect_pursuit.get("source_id") != source_id:
            return False
        expires_at = float(self.inspect_pursuit.get("expires_at") or 0.0)
        if expires_at and float(now_ts) > expires_at:
            self._emit_event(sim_state, "pursuit_expired", {"source_id": source_id, "expires_at": expires_at, "now": float(now_ts)})
            self.inspect_stall_counts[source_id] = self.inspect_stall_counts.get(source_id, 0) + 1
            self._clear_inspect_pursuit(reason="lease_expired", sim_state=sim_state, release_slot=True, environment=environment)
            return True
        blocked_attempts = int(self.inspect_pursuit.get("blocked_attempts", 0))
        no_progress = int(self.inspect_pursuit.get("no_progress_ticks", 0))
        if blocked_attempts >= int(self.inspect_pursuit_blocked_attempt_limit) or no_progress >= int(self.inspect_pursuit_no_progress_limit):
            reason = reason_hint or ("blocked_threshold" if blocked_attempts >= int(self.inspect_pursuit_blocked_attempt_limit) else "no_progress_threshold")
            self._emit_event(
                sim_state,
                "pursuit_stalled",
                {
                    "source_id": source_id,
                    "blocked_attempts": blocked_attempts,
                    "no_progress_ticks": no_progress,
                    "reason": reason,
                },
            )
            self.inspect_stall_counts[source_id] = self.inspect_stall_counts.get(source_id, 0) + 1
            self._clear_inspect_pursuit(reason=reason, sim_state=sim_state, release_slot=True, environment=environment)
            return True
        return False

    def _select_source_access_target(self, environment, source_id, sim_state=None):
        if not hasattr(environment, "select_source_access_point"):
            point = environment.get_interaction_target_position(source_id, from_position=self.position)
            return {"kind": "slot", "slot_id": None, "position": point, "reason": "legacy_selection"} if point is not None else None

        selection = environment.select_source_access_point(source_id, agent_id=self.agent_id, from_position=self.position)
        if selection is None:
            return None
        slot_id = selection.get("slot_id")
        position = selection.get("position")
        kind = selection.get("kind")
        self.source_access_state.update(
            {
                "source_id": source_id,
                "slot_id": slot_id,
                "slot_position": position,
                "target_kind": kind,
            }
        )
        self._emit_event(sim_state, "source_slot_selected", {"source_id": source_id, "slot_id": slot_id, "target_kind": kind, "selection_reason": selection.get("reason")})
        if kind == "slot":
            self._emit_event(sim_state, "source_slot_reserved", {"source_id": source_id, "slot_id": slot_id})
        else:
            self._emit_event(sim_state, "source_access_queue_wait", {"source_id": source_id, "slot_id": slot_id, "queue_index": selection.get("queue_index", 1)})
        return selection

    def _critical_unmet_source_targets(self, sim_state, environment):
        audit = getattr(sim_state, "runtime_witness_audit", None)
        if audit is None:
            return {}
        priorities = {}
        for target in getattr(audit, "targets", {}).values():
            steps = target.get("ordered_witness_steps", [])
            first_pending = next((s for s in steps if getattr(s, "status", None) != "completed"), None)
            if first_pending is None or not str(first_pending.raw_step).startswith("source_access:"):
                continue
            source_ref = str(first_pending.raw_step).split(":", 1)[1]
            packet_name = environment.source_packet_name_map.get(source_ref, source_ref)
            if packet_name not in environment.knowledge_packets:
                continue
            priorities[packet_name] = max(priorities.get(packet_name, 0), 1)
        return priorities

    def _candidate_information_sources(self, environment, sim_state=None):
        self._ensure_source_state(environment)
        critical_needs = self._critical_unmet_source_targets(sim_state, environment)
        top_goal = next((g for g in self.goal_stack if g.get("status") in {"active", "queued", "candidate"}), self.goal_stack[0] if self.goal_stack else {})
        goal_text = str(top_goal.get("goal") or top_goal.get("goal_id") or "").lower()
        role_source = f"{self.role}_Info"
        missing_baseline_sources = {
            source_id
            for source_id in ("Team_Info", role_source)
            if source_id in environment.knowledge_packets
            and self._has_packet_access(source_id)
            and self.source_inspection_state.get(source_id) != "inspected"
            and not bool(self.source_exhaustion_state.get(source_id, {}).get("exhausted"))
        }
        info_pressure = float(min(3.0, len(critical_needs) + len(missing_baseline_sources)))
        candidates = []
        for packet_name in environment.knowledge_packets.keys():
            if not self._has_packet_access(packet_name):
                continue
            point = environment.get_interaction_target_position(packet_name, from_position=self.position)
            if point is None:
                continue

            status = self.source_inspection_state.get(packet_name, "unseen")
            stalled = self.inspect_stall_counts.get(packet_name, 0)
            score = 0.0
            if status == "unseen":
                score += 5.0
            elif status == "revisitable_due_to_gap":
                score += 4.0
            elif status == "in_progress":
                score += 2.0
            elif status == "inspected":
                score -= 4.0
            exhaustion = self.source_exhaustion_state.get(packet_name, {})
            if exhaustion.get("exhausted"):
                score -= 6.0
            no_new_streak = int(exhaustion.get("no_new_dik_streak", 0) or 0)
            score -= min(4.5, 1.5 * no_new_streak)
            if packet_name in critical_needs:
                score += 5.0 + (0.85 * info_pressure)
            if packet_name in missing_baseline_sources:
                score += 2.6 + (0.65 * info_pressure)
            if packet_name == f"{self.role}_Info":
                score += 1.5
            if packet_name == "Team_Info":
                score += 1.0
            if packet_name == "Team_Info" and (bool(exhaustion.get("exhausted")) or no_new_streak >= 2):
                score -= 3.5
            if self.role == "Architect" and packet_name == "Architect_Info" and any(k in goal_text for k in {"shelter", "build", "construction"}):
                score += 2.2
            if self.role == "Engineer" and packet_name == "Engineer_Info" and any(k in goal_text for k in {"water", "power", "connectivity", "logistics"}):
                score += 2.2
            if self.role == "Botanist" and packet_name == "Botanist_Info" and any(k in goal_text for k in {"food", "greenhouse", "sustain"}):
                score += 2.2
            score -= stalled * 1.5
            score -= math.hypot(point[0] - self.position[0], point[1] - self.position[1]) * 0.2
            candidates.append((score, packet_name, point, status, stalled, bool(packet_name in critical_needs), bool(exhaustion.get("exhausted")), no_new_streak))

        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates

    def _resolve_inspect_target(self, decision, environment, sim_state=None):
        self._ensure_source_state(environment)
        explicit_target = decision.target_id
        self._emit_event(sim_state, "target_resolution_started", {"target_type": "information_source", "requested_target_id": explicit_target})
        if explicit_target:
            selection = self._select_source_access_target(environment, explicit_target, sim_state=sim_state)
            if selection is not None and selection.get("position") is not None:
                exhausted = bool(self.source_exhaustion_state.get(explicit_target, {}).get("exhausted"))
                if exhausted:
                    self._emit_event(sim_state, "source_revisit_deferred", {"current_source_target": explicit_target, "reason": "source_exhausted_for_agent"})
                else:
                    self._set_status(f"Inspect target selected: {explicit_target}")
                    self._emit_event(sim_state, "target_resolved", {"target_type": "information_source", "target_id": explicit_target, "candidate_count": 1})
                    return explicit_target, selection.get("position")

        candidates = self._candidate_information_sources(environment, sim_state=sim_state)
        if not candidates:
            self._set_status("Inspect target resolution failed: no accessible information sources")
            self._emit_event(sim_state, "target_resolution_failed", {"target_type": "information_source", "failure_category": "no_information_source_available"})
            return None, None

        # Conservative retargeting away from repeatedly stalled targets when alternatives exist.
        non_stalled = [c for c in candidates if c[4] < 3 and not c[6]]
        if not non_stalled:
            non_stalled = [c for c in candidates if c[4] < 3]
        chosen = non_stalled[0] if non_stalled else candidates[0]
        if chosen[6]:
            self._emit_event(sim_state, "source_revisit_required", {"source_id": chosen[1], "reason": "all_sources_exhausted_or_stalled"})
        if explicit_target is None:
            self._set_status(
                f"Inspect decision missing explicit target; resolved to {chosen[1]} (status={chosen[3]}, stalled={chosen[4]})"
            )
        elif chosen[1] != explicit_target:
            self._set_status(
                f"Inspect target {explicit_target} unreachable; retargeted to {chosen[1]} (status={chosen[3]}, stalled={chosen[4]})"
            )
            self._emit_event(sim_state, "source_revisit_suppressed", {"source_id": explicit_target, "next_source_target": chosen[1], "reason": "source_exhausted_or_unreachable"})
        self._emit_event(sim_state, "next_source_target_selected", {"current_location": self.position, "current_source_target": explicit_target, "next_source_target": chosen[1], "critical_witness_source_need": chosen[5], "reason": "candidate_scoring"})
        self._emit_event(
            sim_state,
            "source_target_ranking",
            {
                "top_candidates": [
                    {
                        "source_id": c[1],
                        "score": round(c[0], 3),
                        "status": c[3],
                        "stalled": c[4],
                        "exhausted": c[6],
                        "no_new_dik_streak": c[7],
                    }
                    for c in candidates[:3]
                ],
                "selected_source": chosen[1],
            },
        )
        if chosen[5] and chosen[1] == "Team_Info":
            self._emit_event(sim_state, "shared_source_target_selected", {"source_id": chosen[1], "reason": "critical_witness_source_need"})
        self._emit_event(sim_state, "target_resolved", {"target_type": "information_source", "target_id": chosen[1], "candidate_count": len(candidates)})
        if self.source_access_state.get("source_id") and self.source_access_state.get("source_id") != chosen[1]:
            self._release_source_slot(environment, emit=True, sim_state=sim_state, reason="retargeted_source")
        selection = self._select_source_access_target(environment, chosen[1], sim_state=sim_state)
        return chosen[1], (selection or {}).get("position")

    def mark_source_revisitable(self, source_id, reason="identified_gap"):
        self.source_inspection_state[source_id] = "revisitable_due_to_gap"
        self._set_status(f"Source marked revisitable due to gap: {source_id} ({reason})")

    @staticmethod
    def _team_dik_key(element_id, element_type):
        return f"{element_type}:{element_id}"

    def _write_shared_source_to_team_knowledge(self, sim_state, source_id, packet, new_info_ids, new_data_ids, new_rule_ids):
        if sim_state is None or not hasattr(sim_state, "team_knowledge_manager"):
            return {"added": [], "adopted": []}
        manager = sim_state.team_knowledge_manager
        added = []
        adopted = []

        for item in packet.get("information", []):
            key = self._team_dik_key(item.id, "information")
            summary = f"{source_id}:{item.id}:{getattr(item, 'content', '')}"
            if key not in manager.validated_knowledge:
                manager.add_validated_knowledge(key, summary, self.name, float(getattr(sim_state, "time", 0.0)))
                added.append(item.id)
            elif item.id in set(new_info_ids):
                adopted.append(item.id)

        for item in packet.get("data", []):
            key = self._team_dik_key(item.id, "data")
            summary = f"{source_id}:{item.id}:{getattr(item, 'content', '')}"
            if key not in manager.validated_knowledge:
                manager.add_validated_knowledge(key, summary, self.name, float(getattr(sim_state, "time", 0.0)))
                added.append(item.id)
            elif item.id in set(new_data_ids):
                adopted.append(item.id)

        for rule_id in new_rule_ids:
            key = self._team_dik_key(rule_id, "rule")
            if key not in manager.validated_knowledge:
                manager.add_validated_knowledge(key, f"{source_id}:{rule_id}", self.name, float(getattr(sim_state, "time", 0.0)))
                added.append(rule_id)
            else:
                adopted.append(rule_id)

        return {"added": sorted(set(added)), "adopted": sorted(set(adopted))}

    def _inspect_source(self, environment, source_id, sim_state=None):
        packet = environment.knowledge_packets.get(source_id)
        source_meta = environment.source_metadata_for_packet(source_id) if hasattr(environment, "source_metadata_for_packet") else {}
        source_access_classification = (
            environment.classify_source_access(source_id, position=self.position, role=self.role, target_kind="information")
            if hasattr(environment, "classify_source_access")
            else {
                "classification": "shared_team_source" if bool(getattr(environment, "is_shared_information_source", lambda _: False)(source_id)) else "role_private_source",
                "is_shared_source": bool(getattr(environment, "is_shared_information_source", lambda _: False)(source_id)),
                "is_private_source": not bool(getattr(environment, "is_shared_information_source", lambda _: False)(source_id)),
                "is_role_mismatch": False,
                "is_movement_only": False,
            }
        )
        shared_source = bool(source_access_classification.get("is_shared_source"))
        if packet is None:
            self._set_status(f"Inspect failed: unknown source {source_id}")
            self._emit_event(sim_state, "inspect_completion_failed", {"source_id": source_id, "failure_category": "unknown_source"})
            return False

        now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
        if self._expire_or_stall_inspect_pursuit(source_id, now_ts, sim_state=sim_state, environment=environment):
            return False
        if self.inspect_session.get("source_id") != source_id:
            self.inspect_session = {
                "source_id": source_id,
                "target": None,
                "state": "target_selected",
                "started_at": now_ts,
                "last_updated_at": now_ts,
                "restarts": 0,
            }
        self.source_inspection_state[source_id] = "in_progress"
        if self._inspect_pursuit_active_for(source_id, now_ts) and self.inspect_pursuit.get("target_position") is not None:
            selection = {
                "position": self.inspect_pursuit.get("target_position"),
                "slot_id": self.inspect_pursuit.get("slot_id"),
                "kind": self.source_access_state.get("target_kind"),
            }
        else:
            selection = self._select_source_access_target(environment, source_id, sim_state=sim_state)
            if selection is not None and selection.get("position") is not None:
                self._commit_inspect_pursuit(
                    source_id,
                    selection.get("position"),
                    now_ts,
                    slot_id=selection.get("slot_id"),
                    sim_state=sim_state,
                )
        target_pos = (selection or {}).get("position")
        self.inspect_session["target"] = target_pos
        if target_pos is None:
            self._set_status(f"Inspect failed: no navigable target for {source_id}")
            self._emit_event(sim_state, "inspect_completion_blocked", {"source_id": source_id, "failure_category": "target_resolution_failed"})
            return False
        self.target = target_pos

        # Distinguish source vicinity from legal slot-based interaction access.
        slot_id = self.source_access_state.get("slot_id")
        usable, access_reason = (True, "legacy")
        if sim_state is None and hasattr(environment, "can_access_info"):
            usable = bool(environment.can_access_info(self.position, source_id, role=self.role))
            access_reason = "legacy_direct_inspection" if usable else "not_at_interaction_slot"
        elif hasattr(environment, "can_agent_use_source_slot"):
            usable, access_reason = environment.can_agent_use_source_slot(
                source_id,
                agent_id=self.agent_id,
                position=self.position,
                slot_id=slot_id,
                role=self.role,
            )

        arrival_distance = math.hypot(self.position[0] - target_pos[0], self.position[1] - target_pos[1]) if target_pos is not None else None
        self._emit_event(
            sim_state,
            "source_access_legality_checked",
            {
                "source_id": source_id,
                "slot_id": slot_id,
                "target_kind": self.source_access_state.get("target_kind"),
                "target_position": target_pos,
                "agent_position": self.position,
                "arrival_distance": round(float(arrival_distance), 4) if arrival_distance is not None else None,
                "access_legal": bool(usable),
                "reason": access_reason,
            },
        )
        if arrival_distance is not None:
            self._mark_inspect_pursuit_progress(source_id, arrival_distance, sim_state=sim_state)

        if not usable:
            self.source_access_state["blocked_attempts"] = int(self.source_access_state.get("blocked_attempts", 0)) + 1
            blocked_attempts = int(self.source_access_state.get("blocked_attempts", 0))
            if self.inspect_pursuit.get("source_id") == source_id:
                self.inspect_pursuit["blocked_attempts"] = blocked_attempts
            self._set_status(f"Inspect pending: usable source access not obtained for {source_id} ({access_reason})")
            self.inspect_session["last_updated_at"] = now_ts
            self.inspect_session["state"] = "target_reached" if self.inspect_session.get("state") == "target_reached" else "target_selected"
            self._emit_event(sim_state, "inspect_progressed", {"source_id": source_id, "stage": self.inspect_session.get("state"), "target": target_pos, "goal": self.goal})
            self._emit_event(sim_state, "source_access_blocked_by_occupancy", {"source_id": source_id, "slot_id": slot_id, "reason": access_reason, "blocked_attempts": blocked_attempts})
            if access_reason in {"slot_reserved_by_other", "not_at_interaction_slot"}:
                alt = self._select_source_access_target(environment, source_id, sim_state=sim_state)
                if alt is not None and alt.get("slot_id") != slot_id:
                    self.target = alt.get("position")
                    if self.inspect_pursuit.get("source_id") == source_id:
                        self.inspect_pursuit["target_position"] = alt.get("position")
                        self.inspect_pursuit["slot_id"] = alt.get("slot_id")
                    self._emit_event(sim_state, "source_access_retargeted_alternate_slot", {"source_id": source_id, "previous_slot_id": slot_id, "next_slot_id": alt.get("slot_id")})
                    self._emit_event(sim_state, "same_source_slot_reselected", {"source_id": source_id, "previous_slot_id": slot_id, "next_slot_id": alt.get("slot_id")})
            stalled = self._expire_or_stall_inspect_pursuit(
                source_id,
                now_ts,
                sim_state=sim_state,
                environment=environment,
                reason_hint=f"blocked:{access_reason}",
            )
            if stalled:
                self._emit_event(
                    sim_state,
                    "source_access_unstuck_backoff",
                    {"source_id": source_id, "slot_id": slot_id, "blocked_attempts": blocked_attempts, "backoff_target": "same_source"},
                )
            if shared_source:
                transient_block = access_reason in {"not_at_interaction_slot", "slot_reserved_by_other", "too_far_or_role_mismatch"}
                self._emit_event(
                    sim_state,
                    "shared_source_access_blocked",
                    {
                        "source_id": source_id,
                        "reason": access_reason,
                        "transient": bool(transient_block),
                        "terminal": bool(not transient_block),
                        "source_meta": source_meta,
                        "source_access_classification": source_access_classification.get("classification"),
                    },
                )
            return False

        self.source_access_state["blocked_attempts"] = 0
        if self.inspect_pursuit.get("source_id") == source_id:
            self.inspect_pursuit["blocked_attempts"] = 0
            self.inspect_pursuit["no_progress_ticks"] = 0

        self.inspect_session["state"] = "target_reached"
        self.inspect_session["last_updated_at"] = now_ts
        self._emit_event(sim_state, "inspect_progressed", {"source_id": source_id, "stage": "target_reached", "target": target_pos, "goal": self.goal})
        self._emit_event(sim_state, "source_access_arrival_confirmed", {"source_id": source_id, "slot_id": slot_id, "target": target_pos, "agent_position": self.position})
        self.inspect_session["state"] = "inspection_started"
        self.inspect_session["last_updated_at"] = now_ts
        self._emit_event(sim_state, "inspect_started", {"source_id": source_id, "target": target_pos, "goal": self.goal, "slot_id": slot_id})
        self._emit_event(sim_state, "inspect_progressed", {"source_id": source_id, "stage": "inspection_started", "target": target_pos, "goal": self.goal})
        if shared_source:
            self._emit_event(sim_state, "shared_source_inspect_started", {"source_id": source_id, "target": target_pos, "source_meta": source_meta})
        self._set_status(f"Arrived at source zone: {source_id}")
        readiness_before = set(self._build_readiness_blockers(environment, sim_state=sim_state))
        before_data_ids = {d.id for d in self.mental_model["data"]}
        before_ids = {info.id for info in self.mental_model["information"]}
        before_rules = set(self.mental_model["knowledge"].rules)
        derivations_before = set(self.executed_derivations)
        self.absorb_packet(packet, accuracy=0.95, sim_state=sim_state, source_id=source_id)
        after_ids = {info.id for info in self.mental_model["information"]}
        packet_info_ids = {info.id for info in packet.get("information", [])}
        new_ids = after_ids - before_ids
        after_data_ids = {d.id for d in self.mental_model["data"]}
        new_data_from_source = after_data_ids - before_data_ids
        self.memory_seen_packets.add(source_id)

        if new_ids:
            self._set_status(f"Source access succeeded: {source_id} (+{len(new_ids)} new items)")
        elif packet_info_ids.issubset(after_ids):
            self._set_status(f"Source already inspected: {source_id}")
        else:
            self._set_status(f"Source access had no uptake: {source_id}")

        if packet_info_ids.issubset(after_ids):
            self.source_inspection_state[source_id] = "inspected"
            self.inspect_session["state"] = "inspection_completed"
            self.inspect_session["last_updated_at"] = now_ts
            self._emit_event(sim_state, "inspect_completed", {"source_id": source_id, "target": target_pos, "goal": self.goal})
        else:
            if shared_source:
                self.source_inspection_state[source_id] = "inspected"
                self._emit_event(sim_state, "inspect_completed", {"source_id": source_id, "target": target_pos, "goal": self.goal, "completion_mode": "partial_packet_uptake"})
            else:
                self.source_inspection_state[source_id] = "in_progress"
                self._emit_event(sim_state, "inspect_completion_failed", {"source_id": source_id, "failure_category": "partial_packet_uptake"})

        packet_data_ids = {d.id for d in packet.get("data", [])}
        held_data_ids = {d.id for d in self.mental_model["data"]}
        new_data_ids = sorted(packet_data_ids & held_data_ids)
        self._apply_task_derivations(sim_state=sim_state, trigger_source=source_id)
        derivation_ids_triggered = sorted(set(self.executed_derivations) - derivations_before)
        after_rules = set(self.mental_model["knowledge"].rules)
        new_rule_ids = sorted(after_rules - before_rules)
        dik_changed = bool(new_ids or new_data_from_source or new_rule_ids)
        shared_team_delta = {"added": [], "adopted": []}
        if shared_source:
            shared_team_delta = self._write_shared_source_to_team_knowledge(
                sim_state,
                source_id,
                packet,
                sorted(new_ids),
                sorted(new_data_from_source),
                new_rule_ids,
            )
        team_changed = bool(shared_team_delta["added"] or shared_team_delta["adopted"])
        net_dik_changed = bool(dik_changed or team_changed)
        source_state = self.source_exhaustion_state.setdefault(source_id, {"inspect_count": 0, "last_dik_changed": None, "exhausted": False})
        source_state["inspect_count"] = int(source_state.get("inspect_count", 0)) + 1
        source_state["last_dik_changed"] = bool(net_dik_changed)
        if net_dik_changed:
            source_state["no_new_dik_streak"] = 0
        else:
            source_state["no_new_dik_streak"] = int(source_state.get("no_new_dik_streak", 0) or 0) + 1
        source_state["inspected"] = bool(self.source_inspection_state.get(source_id) == "inspected")
        source_state["exhausted"] = bool((not net_dik_changed) and self.source_inspection_state.get(source_id) == "inspected")
        if sim_state is not None and source_state["exhausted"]:
            self._emit_event(sim_state, "source_already_inspected_no_new_dik", {"source_id": source_id, "inspect_count": source_state["inspect_count"]})
            self._emit_event(sim_state, "source_exhausted_for_agent", {"source_id": source_id, "inspect_count": source_state["inspect_count"]})
            if shared_source:
                self._emit_event(sim_state, "shared_source_exhausted_for_team", {"source_id": source_id, "inspect_count": source_state["inspect_count"]})
        if sim_state is not None:
            self._emit_event(
                sim_state,
                "inspect_success_no_new_dik" if not dik_changed else "inspect_success_dik_changed",
                {
                    "source_id": source_id,
                    "dik_changed": dik_changed,
                    "new_information_count": len(new_ids),
                    "new_data_count": len(new_data_from_source),
                    "new_rule_count": len(new_rule_ids),
                },
            )
            self._emit_event(
                sim_state,
                "inspect_success_derivation_triggered" if derivation_ids_triggered else "inspect_success_dik_no_derivation",
                {
                    "source_id": source_id,
                    "dik_changed": dik_changed,
                    "derivation_triggered": bool(derivation_ids_triggered),
                    "derivation_ids": derivation_ids_triggered,
                },
            )
            if new_rule_ids:
                self._emit_event(
                    sim_state,
                    "inspect_success_rule_adopted",
                    {
                        "source_id": source_id,
                        "dik_changed": dik_changed,
                        "rule_ids": new_rule_ids,
                    },
                )
            elif dik_changed:
                self._emit_event(
                    sim_state,
                    "inspect_success_rule_not_adopted",
                    {
                        "source_id": source_id,
                        "dik_changed": dik_changed,
                    },
                )
            if shared_source:
                self._emit_event(sim_state, "shared_source_inspect_completed", {"source_id": source_id, "local_dik_changed": bool(dik_changed), "team_dik_changed": bool(team_changed), "source_meta": source_meta})
                if new_ids or new_data_from_source or new_rule_ids:
                    self._emit_event(sim_state, "shared_source_dik_acquired_agent", {"source_id": source_id, "new_information_ids": sorted(new_ids), "new_data_ids": sorted(new_data_from_source), "new_rule_ids": new_rule_ids})
                if shared_team_delta["added"]:
                    self._emit_event(sim_state, "shared_source_dik_acquired_team", {"source_id": source_id, "added_ids": shared_team_delta["added"]})
                if shared_team_delta["adopted"]:
                    self._emit_event(sim_state, "shared_source_dik_adopted", {"source_id": source_id, "adopted_ids": shared_team_delta["adopted"]})
                if not net_dik_changed:
                    self._emit_event(sim_state, "shared_source_exhausted_for_agent", {"source_id": source_id, "inspect_count": source_state["inspect_count"]})
        if dik_changed:
            self.inspect_session["state"] = "dik_acquired"
            self.inspect_session["last_updated_at"] = now_ts
            self._emit_event(
                sim_state,
                "dik_acquired_from_inspect",
                {
                    "source_id": source_id,
                    "new_information_ids": sorted(new_ids),
                    "new_data_ids": sorted(new_data_from_source),
                    "new_rule_ids": new_rule_ids,
                    "dik_changed": True,
                },
            )
        readiness_after = set(self._build_readiness_blockers(environment, sim_state=sim_state))
        readiness_changed = readiness_before != readiness_after
        blocker_category = self._categorize_readiness_blockers(readiness_after)
        post_inspect_outcome = self._categorize_post_inspect_outcome(
            dik_changed=net_dik_changed,
            derivation_triggered=bool(derivation_ids_triggered),
            rule_adopted=bool(new_rule_ids),
            readiness_after=readiness_after,
        )
        now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
        self.post_inspect_handoff = {
            "pending": True,
            "source_id": source_id,
            "dik_changed": net_dik_changed,
            "readiness_changed": readiness_changed,
            "blockers": sorted(readiness_after),
            "blocker_category": blocker_category,
            "outcome": post_inspect_outcome,
            "expires_at": now_ts + 5.0,
        }
        if sim_state is not None:
            self._emit_event(
                sim_state,
                "readiness_recomputed_after_inspect",
                {
                    "source_id": source_id,
                    "readiness_changed": readiness_changed,
                    "readiness_unlocked": not readiness_after,
                    "before_blockers": sorted(readiness_before),
                    "after_blockers": sorted(readiness_after),
                },
            )
            if net_dik_changed and readiness_after:
                self._emit_event(
                    sim_state,
                    "inspect_completion_blocked",
                    {
                        "source_id": source_id,
                        "failure_category": "readiness_not_unlocked_after_inspect_success",
                        "remaining_blockers": sorted(readiness_after),
                    },
                )
            self._emit_event(
                sim_state,
                "inspect_success_readiness_changed" if readiness_changed else "inspect_success_no_readiness_change",
                {
                    "source_id": source_id,
                    "dik_changed": net_dik_changed,
                    "team_dik_changed": bool(team_changed),
                    "readiness_changed": readiness_changed,
                    "readiness_unlocked": not readiness_after,
                    "remaining_blockers": sorted(readiness_after),
                    "post_inspect_blocker_category": blocker_category,
                    "post_inspect_outcome": post_inspect_outcome,
                },
            )
            self._emit_event(
                sim_state,
                "inspect_post_handoff_classified",
                {
                    "source_id": source_id,
                    "dik_changed": net_dik_changed,
                    "team_dik_changed": bool(team_changed),
                    "readiness_changed": readiness_changed,
                    "post_inspect_outcome": post_inspect_outcome,
                    "post_inspect_blocker_category": blocker_category,
                    "remaining_blockers": sorted(readiness_after),
                },
            )
        self._emit_event(
            sim_state,
            "source_access_succeeded",
            {
                "source_id": source_id,
                "reason": source_id,
                "new_information_ids": sorted(new_ids),
                "new_data_ids": new_data_ids,
                "team_dik_added_ids": shared_team_delta["added"],
                "team_dik_adopted_ids": shared_team_delta["adopted"],
                "source_access_classification": source_access_classification.get("classification"),
                "is_shared_source": bool(source_access_classification.get("is_shared_source")),
                "is_private_source": bool(source_access_classification.get("is_private_source")),
                "is_role_mismatch": bool(source_access_classification.get("is_role_mismatch")),
                "is_movement_only": bool(source_access_classification.get("is_movement_only")),
            },
        )
        if shared_source:
            self._emit_event(
                sim_state,
                "shared_source_access_success" if net_dik_changed else "shared_source_access_blocked",
                {
                    "source_id": source_id,
                    "reason": source_id if net_dik_changed else "no_new_dik",
                    "local_dik_changed": bool(dik_changed),
                    "team_dik_changed": bool(team_changed),
                    "new_information_ids": sorted(new_ids),
                    "new_data_ids": sorted(new_data_from_source),
                    "new_rule_ids": new_rule_ids,
                    "team_dik_added_ids": shared_team_delta["added"],
                    "team_dik_adopted_ids": shared_team_delta["adopted"],
                    "witness_step_satisfied": bool(net_dik_changed),
                    "source_access_classification": source_access_classification.get("classification"),
                },
            )
        self.inspect_session["state"] = "post_inspect_derivation_attempted"
        self.inspect_session["last_updated_at"] = now_ts
        self._release_source_slot(environment, source_id=source_id, emit=True, sim_state=sim_state, reason="inspection_completed")
        self._clear_inspect_pursuit(reason="inspection_succeeded", sim_state=sim_state, release_slot=False, environment=environment)
        return bool(net_dik_changed)

    @staticmethod
    def _categorize_readiness_blockers(readiness_after):
        blockers = set(readiness_after or set())
        if not blockers:
            return "none"
        if any(b in blockers for b in {"missing_task_prerequisite_rules", "insufficient_rule_knowledge"}):
            return "missing_rule"
        if "no_navigable_build_target" in blockers:
            return "missing_target"
        if any("artifact" in b or "externalization" in b for b in blockers):
            return "missing_artifact"
        if any("phase" in b for b in blockers):
            return "phase"
        return "other"

    def _categorize_post_inspect_outcome(self, dik_changed, derivation_triggered, rule_adopted, readiness_after):
        if not dik_changed:
            return "inspect_success_no_new_dik"
        if not derivation_triggered:
            return "inspect_success_dik_no_derivation"
        if not rule_adopted:
            return "inspect_success_rule_not_adopted"
        if not readiness_after:
            return "inspect_success_ready_for_action"
        blocker_category = self._categorize_readiness_blockers(readiness_after)
        mapping = {
            "missing_rule": "inspect_success_readiness_blocked_missing_rule",
            "missing_target": "inspect_success_readiness_blocked_missing_target",
            "missing_artifact": "inspect_success_readiness_blocked_missing_artifact",
            "phase": "inspect_success_readiness_blocked_phase",
            "other": "inspect_success_readiness_blocked_missing_rule",
        }
        return mapping.get(blocker_category, "inspect_success_readiness_blocked_missing_rule")

    def _choose_post_inspect_followup_decision(self, environment, sim_state=None):
        critical_sources = self._critical_unmet_source_targets(sim_state, environment)
        if critical_sources:
            preferred = "Team_Info" if "Team_Info" in critical_sources else sorted(critical_sources.keys())[0]
            if preferred in environment.knowledge_packets and not self.source_exhaustion_state.get(preferred, {}).get("exhausted"):
                return BrainDecision(
                    selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE,
                    target_id=preferred,
                    reason_summary="Post-inspect pivot to unmet critical witness source access.",
                    confidence=0.84,
                )
        if self._is_build_eligible(environment):
            build_selection = self._select_build_target(environment, require_readiness=True, include_project=True)
            if isinstance(build_selection, dict):
                return BrainDecision(
                    selected_action=ExecutableActionType.START_CONSTRUCTION,
                    target_id=build_selection.get("project_id"),
                    reason_summary="Post-inspect readiness unlocked; pivoting to construction.",
                    confidence=0.81,
                )
        build_selection = self._select_build_target(environment, require_readiness=False, include_project=True)
        if isinstance(build_selection, dict):
            return BrainDecision(
                selected_action=ExecutableActionType.TRANSPORT_RESOURCES,
                target_id=build_selection.get("project_id"),
                reason_summary="Post-inspect pivot to logistics handoff.",
                confidence=0.74,
            )
        return BrainDecision(
            selected_action=ExecutableActionType.COMMUNICATE,
            reason_summary="Post-inspect fallback to communication handoff.",
            confidence=0.66,
        )



    def _held_dik_ids(self):
        data_ids = {d.id for d in self.mental_model["data"]}
        info_ids = {i.id for i in self.mental_model["information"]}
        knowledge_ids = {normalize_rule_token(r) for r in self.mental_model["knowledge"].rules}
        return data_ids, info_ids, knowledge_ids

    def _team_validated_ids(self, sim_state=None):
        if sim_state is None or not hasattr(sim_state, "team_knowledge_manager"):
            return set(), set(), set()
        keys = set(sim_state.team_knowledge_manager.validated_knowledge.keys())
        data_ids = {k.split(":", 1)[1] for k in keys if isinstance(k, str) and k.startswith("data:") and ":" in k}
        info_ids = {k.split(":", 1)[1] for k in keys if isinstance(k, str) and k.startswith("information:") and ":" in k}
        rule_ids = {k.split(":", 1)[1] for k in keys if isinstance(k, str) and k.startswith("rule:") and ":" in k}
        return data_ids, info_ids, rule_ids

    def _create_dik_object_from_element(self, element, source_id):
        tags = [element.role_scope, element.phase_scope, element.element_type]
        if element.element_type == "data":
            return Data(element.element_id, element.description, source=source_id, tags=tags)
        if element.element_type == "information":
            return Information(element.element_id, element.description, source=source_id, tags=tags)
        return None

    def _apply_task_derivations(self, sim_state=None, trigger_source=None):
        if not getattr(self, "task_model", None):
            return

        now = getattr(self, "current_time", 0.0)
        data_ids, info_ids, knowledge_ids = self._held_dik_ids()
        team_data_ids, team_info_ids, team_rule_ids = self._team_validated_ids(sim_state=sim_state)
        held_ids = data_ids | info_ids | knowledge_ids | team_data_ids | team_info_ids | team_rule_ids

        ready_count = 0
        attempted_count = 0
        for derivation in self.task_model.derivations.values():
            if not derivation.enabled or derivation.derivation_id in self.executed_derivations:
                continue

            required = set(derivation.required_inputs)
            if required and not required.issubset(held_ids):
                if sim_state is not None and trigger_source is not None:
                    self._emit_event(
                        sim_state,
                        "derivation_blocked_missing_prereq",
                        {
                            "derivation_id": derivation.derivation_id,
                            "missing_required_inputs": sorted(required - held_ids),
                            "trigger_source": trigger_source,
                        },
                    )
                continue
            if derivation.min_required_count and len(required & held_ids) < derivation.min_required_count:
                continue

            ready_count += 1
            if sim_state is not None:
                self._emit_event(
                    sim_state,
                    "derivation_prerequisites_satisfied",
                    {
                        "derivation_id": derivation.derivation_id,
                        "required_inputs": sorted(required),
                        "trigger_source": trigger_source,
                    },
                )

            output_id = derivation.output_element_id
            element = self.task_model.dik_elements.get(output_id)
            if element is None or not element.enabled:
                if sim_state is not None:
                    self._emit_event(
                        sim_state,
                        "derivation_ready_but_not_attempted",
                        {
                            "derivation_id": derivation.derivation_id,
                            "reason": "output_disabled_or_missing",
                            "output_element_id": output_id,
                            "trigger_source": trigger_source,
                        },
                    )
                continue

            attempted_count += 1
            hook_target = "transform_information_to_knowledge" if (derivation.output_type == "knowledge" or element.element_type == "knowledge") else "transform_data_to_information"
            prior_attempts = int(self.derivation_attempt_counts.get(derivation.derivation_id, 0))
            retry_bonus = min(0.12, 0.03 * prior_attempts)
            context_modifier = self._derivation_context_modifier(derivation)
            derivation_allowed, attempt_payload = self._attempt_epistemic_transition(
                hook_target=hook_target,
                sim_state=sim_state,
                event_payload={
                    "derivation_id": derivation.derivation_id,
                    "output_element_id": output_id,
                    "required_inputs": sorted(required),
                    "trigger_source": trigger_source,
                    "attempt_index": prior_attempts + 1,
                },
                attempt_event="derivation_attempted",
                failed_event="derivation_failed",
                context_modifier=context_modifier,
                retry_bonus=retry_bonus,
            )
            self.derivation_attempt_counts[derivation.derivation_id] = prior_attempts + 1
            if not derivation_allowed:
                self.activity_log.append(
                    f"Derivation attempt failed {derivation.derivation_id} "
                    f"(p={attempt_payload['success_probability']:.2f}, roll={attempt_payload['stochastic_roll']:.2f})"
                )
                continue

            produced = False
            rule_ready_not_adopted = False
            if derivation.output_type == "knowledge" or element.element_type == "knowledge":
                if output_id not in self.mental_model["knowledge"].rules:
                    if sim_state is not None:
                        self._emit_event(
                            sim_state,
                            "rule_candidate_generated",
                            {
                                "derivation_id": derivation.derivation_id,
                                "rule_id": output_id,
                                "required_inputs": sorted(required),
                                "trigger_source": trigger_source,
                            },
                        )
                    self.mental_model["knowledge"].add_rule(output_id, sorted(required), inferred_by_agents=[self.name])
                    produced = True
                else:
                    rule_ready_not_adopted = True
            elif derivation.output_type == "information" or element.element_type == "information":
                if output_id not in info_ids:
                    info_obj = self._create_dik_object_from_element(element, source_id=f"DRV:{derivation.derivation_id}")
                    if info_obj is not None:
                        self.mental_model["information"].add(info_obj)
                        produced = True
            elif derivation.output_type == "data" or element.element_type == "data":
                if output_id not in data_ids:
                    data_obj = self._create_dik_object_from_element(element, source_id=f"DRV:{derivation.derivation_id}")
                    if data_obj is not None:
                        self.mental_model["data"].add(data_obj)
                        produced = True

            if produced:
                event = {
                    "time": now,
                    "agent": self.name,
                    "derivation_id": derivation.derivation_id,
                    "output_element_id": output_id,
                    "derivation_kind": derivation.derivation_kind,
                    "required_inputs": sorted(required),
                    "trigger_source": trigger_source,
                }
                self.executed_derivations.add(derivation.derivation_id)
                self.derivation_events.append(event)
                self.derivation_attempt_counts.pop(derivation.derivation_id, None)
                self.activity_log.append(f"Executed derivation {derivation.derivation_id} -> {output_id}")
                DIK_LOG.append({
                    "time": now,
                    "agent": self.name,
                    "type": "Derivation",
                    "id": derivation.derivation_id,
                    "mode": "derived",
                    "from": sorted(required),
                })
                self.last_dik_change_time = now
                if sim_state is not None:
                    self._emit_event(sim_state, "derivation_succeeded", event)
                    sim_state.logger.log_event(now, "dik_derivation_executed", event)
                    if output_id in self.mental_model["knowledge"].rules:
                        self._emit_event(sim_state, "rule_adopted", {"rule_id": output_id, "adoption_mode": "derived", "derivation_id": derivation.derivation_id})
            else:
                if sim_state is not None:
                    self._emit_event(
                        sim_state,
                        "derivation_attempted_no_output",
                        {
                            "derivation_id": derivation.derivation_id,
                            "output_element_id": output_id,
                            "reason": "output_already_held_or_not_materialized",
                            "trigger_source": trigger_source,
                        },
                    )
                    if rule_ready_not_adopted:
                        self._emit_event(
                            sim_state,
                            "rule_ready_but_not_adopted",
                            {
                                "derivation_id": derivation.derivation_id,
                                "rule_id": output_id,
                                "reason": "rule_already_present",
                                "trigger_source": trigger_source,
                            },
                        )

        if sim_state is not None and ready_count > 0 and attempted_count == 0:
            self._emit_event(
                sim_state,
                "derivation_ready_but_not_attempted",
                {
                    "ready_count": ready_count,
                    "reason": "ready_derivations_filtered_before_execution",
                    "trigger_source": trigger_source,
                },
            )

    def _build_readiness_score(self):
        info_count = len(self.mental_model["information"])
        knowledge_count = len(self.mental_model["knowledge"].rules)
        inspected_sources = sum(1 for state in self.source_inspection_state.values() if state == "inspected")
        artifact_count = len([p for p in self.memory_seen_packets if isinstance(p, str) and p.startswith("whiteboard:")])
        return info_count + (2 * knowledge_count) + inspected_sources + min(artifact_count, 1)

    def _effective_knowledge_for_readiness(self, sim_state=None):
        info_ids = {i.id for i in self.mental_model["information"]}
        knowledge_ids = {str(r).strip() for r in self.mental_model["knowledge"].rules if str(r).strip()}
        rule_ids = {normalize_rule_token(r) for r in knowledge_ids}
        _, team_info_ids, team_rule_ids = self._team_validated_ids(sim_state=sim_state)
        info_ids.update(team_info_ids)
        rule_ids.update(normalize_rule_token(r) for r in team_rule_ids)
        if getattr(self, "task_model", None):
            for rule in self.task_model.rules.values():
                if not rule.enabled:
                    continue
                needed_k = {k for k in rule.required_knowledge if k}
                needed_i = {i for i in rule.required_information if i}
                if needed_k.issubset(knowledge_ids) and needed_i.issubset(info_ids):
                    rule_ids.add(rule.rule_id)
        return info_ids, rule_ids

    def _construction_project_for_action(self, decision, action, environment):
        requested_project_id = decision.target_id or action.get("project_id")
        if requested_project_id in environment.construction.projects:
            return requested_project_id
        if decision.selected_action in {ExecutableActionType.VALIDATE_CONSTRUCTION, ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION}:
            return None
        build_selection = self._select_build_target(environment, require_readiness=False, include_project=True)
        if isinstance(build_selection, dict):
            return build_selection.get("project_id")
        return None

    def _construction_rule_match(self, project_id, environment=None, sim_state=None, include_team=True):
        project = None
        if environment is not None and getattr(environment, "construction", None) is not None:
            project = environment.construction.projects.get(project_id)
        elif sim_state is not None and getattr(sim_state, "environment", None) is not None:
            project = sim_state.environment.construction.projects.get(project_id)
        expected = {
            normalize_rule_token(r)
            for r in ((project or {}).get("expected_rules") or [])
            if normalize_rule_token(r)
        }
        if not expected:
            return True, []
        local_rules = {normalize_rule_token(r) for r in self.mental_model["knowledge"].rules}
        team_rules = set()
        if include_team:
            _, _, team_rule_ids = self._team_validated_ids(sim_state=sim_state)
            team_rules = {normalize_rule_token(r) for r in team_rule_ids}
        held_rules = local_rules | team_rules
        missing = sorted(expected - held_rules)
        return len(missing) == 0, missing

    def _construction_action_blockers(self, decision, action, environment, sim_state=None):
        blockers = []
        project_id = self._construction_project_for_action(decision, action, environment)
        if project_id is None:
            blockers.append("no_navigable_build_target")
            return blockers, None
        project = environment.construction.projects.get(project_id)
        if not isinstance(project, dict):
            blockers.append("unknown_project")
            return blockers, project_id

        action_type = decision.selected_action
        if action_type in {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}:
            blockers.extend(self._build_readiness_blockers(environment, sim_state=sim_state))
            if self.current_plan is not None and self.current_plan.plan_method_status == "low_trust":
                notes = set(self.current_plan.validation_notes or [])
                if any(n.startswith("missing_") for n in notes):
                    blockers.append("plan_method_not_grounded")
            if project.get("status") == "complete":
                blockers.append("project_already_complete")
        elif action_type == ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION:
            blockers.extend(self._build_readiness_blockers(environment, sim_state=sim_state))
            mismatch_detected = (project.get("correct", True) is False) or any(
                "mismatch with construction" in str(e).lower() for e in self.activity_log[-8:]
            )
            if not mismatch_detected:
                blockers.append("no_detected_mismatch")
        elif action_type == ExecutableActionType.VALIDATE_CONSTRUCTION:
            has_match, missing_rules = self._construction_rule_match(project_id, environment=environment, sim_state=sim_state, include_team=True)
            if not has_match:
                blockers.append("missing_validation_rule_knowledge")
                blockers.extend([f"missing_expected_rule:{rid}" for rid in missing_rules[:3]])

        return sorted(set(blockers)), project_id

    def _build_readiness_blockers(self, environment, sim_state=None):
        blockers = []
        info_ids, rule_ids = self._effective_knowledge_for_readiness(sim_state=sim_state)
        if len(info_ids) < 2:
            blockers.append("insufficient_information_inspection")
        if len(rule_ids) < 1:
            blockers.append("insufficient_rule_knowledge")

        if getattr(self, "task_model", None):
            role_rules = [
                r.rule_id for r in self.task_model.rules.values()
                if r.enabled and r.role_scope in {"team", self.role.lower()}
            ]
            if role_rules and not set(role_rules).intersection(rule_ids):
                blockers.append("missing_task_prerequisite_rules")
        if not any(state == "inspected" for state in self.source_inspection_state.values()):
            blockers.append("no_inspected_information_source")
        if self._select_build_target(environment, require_readiness=False) is None:
            blockers.append("no_navigable_build_target")
        return blockers

    def _is_build_eligible(self, environment, sim_state=None):
        return self._build_readiness_score() >= self._readiness_threshold() and not self._build_readiness_blockers(environment, sim_state=sim_state)

    def _select_build_target(self, environment, require_readiness=False, include_project=False):
        candidates = []
        for target_name, target in environment.interaction_targets.items():
            if target.get("kind") != "build":
                continue
            point = environment.get_interaction_target_position(target_name, from_position=self.position)
            if point is None:
                continue
            project = environment.construction.projects.get(target_name, {})
            required = project.get("required_resources", {}).get("bricks", 0)
            delivered = project.get("delivered_resources", {}).get("bricks", 0)
            remaining = max(0, required - delivered)
            score = remaining + (0 if project.get("status") == "in_progress" else 100)
            candidates.append((score, target_name, point))

        if not candidates:
            return None
        if require_readiness and not self._is_build_eligible(environment):
            return None

        candidates.sort(key=lambda row: row[0])
        _score, target_name, point = candidates[0]
        if include_project:
            return {"project_id": target_name, "target": point}
        return point

    def _next_goal_key(self):
        return self.goal_manager.next_goal_key()

    def _coerce_goal_status(self, status):
        from modules.goal_state import coerce_goal_status

        return coerce_goal_status(status)

    def _coerce_goal_source(self, source):
        from modules.goal_state import coerce_goal_source

        return coerce_goal_source(source)

    def _goal_priority(self, status, priority):
        from modules.goal_state import goal_priority

        return goal_priority(status, priority)

    def _log_goal_transition(self, sim_state, goal, reason, extra=None):
        self.goal_manager.log_goal_transition(sim_state, goal, reason, extra=extra)

    def _upsert_goal_record(self, **kwargs):
        return self.goal_manager.upsert_goal_record(**kwargs)

    def _refresh_goal_stack_view(self):
        self.goal_manager.refresh_goal_stack_view()

    def current_goal(self):
        return self.goal_manager.current_goal()

    def update_current_goal(self):
        self.goal_manager.refresh_goal_stack_view()

    def push_goal(self, goal, target=None):
        self.goal_manager.push_goal(goal, target=target)

    def pop_goal(self):
        self.goal_manager.pop_goal()

    def _seed_task_defined_goals(self, sim_state=None):
        if not self.task_model:
            return
        mission_goal_key = None
        phase_goal_keys = {}
        for g in self.task_model.goals.values():
            if not g.enabled:
                continue
            status = "candidate"
            source = "task_defined"
            if g.goal_id in set(self.initial_goal_seeds or []):
                status = "active"
            elif g.goal_level == "mission":
                status = "queued"

            parent_goal_key = None
            goal_level = (g.goal_level or "").strip().lower()
            if goal_level == "mission":
                parent_goal_key = None
            elif goal_level == "phase":
                parent_goal_key = mission_goal_key
            else:
                parent_goal_key = mission_goal_key

            rec = self._upsert_goal_record(
                label=g.label,
                goal_id=g.goal_id,
                source=source,
                status=status,
                priority=0.9 if g.goal_level == "mission" else 0.6,
                evidence=[f"goal_level={g.goal_level}", f"phase_scope={g.phase_scope}"],
                completion_conditions=[g.success_conditions] if g.success_conditions else [],
                activation_conditions=["phase_alignment", "prerequisite_rules"],
                parent_goal_key=parent_goal_key,
                goal_level=g.goal_level,
                goal_type="canonical",
                trust_tier="canonical",
                sim_state=sim_state,
                reason="task_goal_seeded",
            )
            if goal_level == "mission":
                mission_goal_key = rec.goal_key
            elif goal_level == "phase":
                phase_goal_keys[g.goal_id] = rec.goal_key

        for key in list(self.goal_order):
            goal = self.goal_registry.get(key)
            if not goal or goal.goal_level != "support" or goal.parent_goal_key:
                continue
            goal.parent_goal_key = mission_goal_key
        self._refresh_goal_stack_view()
    def decide(self, sim_state):
        """Deprecated compatibility wrapper for legacy callers."""
        self.perceive_environment(sim_state)
        self.update_internal_state()
        self._evaluate_goal_state(sim_state.environment)
        self.current_action = self._plan_actions_for_current_goal()
        self._advance_active_actions(dt=1.0)

    def _should_request_explanation(self):
        mode = (self.planner_cadence.explanation_mode or "never").lower()
        next_call = self.planner_call_count + 1
        if mode == "always":
            return True
        if mode == "every_n_calls":
            return next_call % max(1, self.planner_cadence.explanation_every_n_calls) == 0
        if mode == "probability":
            return random.random() < self.planner_cadence.explanation_probability
        return False

    def _phase_matches(self, phase_scope, current_phase_name):
        scope = str(phase_scope or "all").strip().lower()
        if scope in {"", "all"}:
            return True
        phase = str(current_phase_name or "").strip().lower()
        if scope == "planning":
            return "phase 1" in phase or "plan" in phase
        if scope == "phase1":
            return "phase 1" in phase
        if scope == "phase2":
            return "phase 2" in phase
        return scope in phase

    def _goal_definition_for(self, goal_id):
        if not self.task_model:
            return None
        return self.task_model.goals.get(goal_id)

    def _canonical_readiness_goal_id(self):
        if not self.task_model:
            return None

        candidate_ids = []
        if self.current_plan is not None:
            candidate_ids.extend([gid for gid in (self.current_plan.associated_goal_ids or []) if gid])
        candidate_ids.extend([g.get("goal_id") for g in self.goal_stack if g.get("goal_id")])

        for level in ("mission", "phase", "role", "support"):
            for goal_id in candidate_ids:
                definition = self.task_model.goals.get(goal_id)
                if not definition or not definition.enabled:
                    continue
                if str(definition.goal_level).strip().lower() == level:
                    return goal_id
        return None

    def _activate_support_goal(self, label, reason, sim_state=None, priority=0.7, source="derived_from_rule"):
        goal_id = f"SUPPORT_{str(label).strip().upper()}"
        now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
        activation_state = self.support_goal_activation_state.setdefault(goal_id, {"last_reason": None, "last_time": -999.0})
        existing_goal = self.goal_registry.get(goal_id)
        duplicate_reason = (
            existing_goal is not None
            and existing_goal.status in {"active", "queued", "candidate"}
            and str(activation_state.get("last_reason")) == str(reason)
            and (now_ts - float(activation_state.get("last_time", -999.0))) <= 1.0
        )
        if duplicate_reason:
            self._emit_event(
                sim_state,
                "support_goal_activation_deduplicated",
                {"goal_id": goal_id, "label": label, "reason": reason},
            )
            return existing_goal
        mission_goal = next((g for g in self.goal_registry.values() if g.goal_level == "mission"), None)
        goal = self._upsert_goal_record(
            label=label,
            goal_id=goal_id,
            source=source,
            status="active",
            priority=priority,
            evidence=[reason],
            parent_goal_key=mission_goal.goal_key if mission_goal else None,
            activation_conditions=["runtime_trigger"],
            goal_level="support",
            goal_type="runtime_support",
            trust_tier="normal",
            sim_state=sim_state,
            reason="support_goal_activated",
        )
        activation_state["last_reason"] = str(reason)
        activation_state["last_time"] = now_ts
        self.activity_log.append(f"Support goal active: {goal.label} ({reason})")
        return goal

    def _baseline_epistemic_sources_completed(self):
        team_done = self.source_inspection_state.get("Team_Info") == "inspected"
        role_source = f"{self.role}_Info"
        role_done = self.source_inspection_state.get(role_source) == "inspected"
        return bool(team_done and role_done)

    def _has_meaningful_consultable_artifact(self, sim_state):
        if sim_state is None or not hasattr(sim_state, "team_knowledge_manager"):
            return False
        manager = sim_state.team_knowledge_manager
        if not getattr(manager, "artifacts", {}):
            return False
        meaningful_events = {
            "externalized_artifact",
            "construction_externalized",
            "construction_artifact_updated",
            "artifact_uptake",
        }
        return any(str(update.get("event") or "") in meaningful_events for update in list(manager.recent_updates or [])[-8:])

    def _support_goal_executable(self, goal, sim_state, environment):
        if goal is None:
            return False, "missing_goal"
        label = str(goal.label or "").strip().lower()
        now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
        if label == "acquire_missing_dik":
            return bool(self._candidate_information_sources(environment, sim_state=sim_state)), "missing_accessible_information_source"
        if label == "unblock_inspection":
            repeated_stall = any(v >= 3 for v in self.inspect_stall_counts.values())
            return (
                repeated_stall and bool(self._candidate_information_sources(environment, sim_state=sim_state)),
                "no_inspection_stall_or_accessible_source",
            )
        if label == "consult_artifact":
            executable = self._baseline_epistemic_sources_completed() and self._has_meaningful_consultable_artifact(sim_state)
            return executable, "artifact_not_grounded"
        if label == "integrate_new_derivation":
            if not self.derivation_events:
                return False, "no_recent_derivation"
            last_ts = float(self.derivation_events[-1].get("time", -999.0) or -999.0)
            return (now_ts - last_ts) <= 20.0, "derivation_stale"
        if label in {"repair_detected_mismatch", "validate_externalization"}:
            mismatch_detected = any("mismatch with construction" in str(e).lower() for e in self.activity_log[-8:])
            if mismatch_detected:
                return True, "ok"
            construction = getattr(environment, "construction", None)
            for project in (getattr(construction, "projects", {}) or {}).values():
                if project.get("status") in {"needs_repair", "ready_for_validation"}:
                    return True, "ok"
            return False, "no_live_construction_mismatch"
        return True, "ok"

    def _update_goal_states_from_runtime(self, sim_state, environment):
        if not self.task_model:
            self._refresh_goal_stack_view()
            return
        phase_name = (environment.get_current_phase() or {}).get("name", "")
        data_ids, info_ids, knowledge_ids = self._held_dik_ids()
        phase_changed = bool(self.last_phase_name and self.last_phase_name != phase_name)

        for key in list(self.goal_order):
            goal = self.goal_registry.get(key)
            if goal is None:
                continue
            definition = self._goal_definition_for(goal.goal_id)
            if definition is None:
                continue

            in_scope = self._phase_matches(definition.phase_scope, phase_name)
            if not in_scope and goal.status in {"active", "candidate", "queued"}:
                goal.status = "inactive"
                goal.last_transition_reason = "phase_out_of_scope"
                self._log_goal_transition(sim_state, goal, "phase_out_of_scope")
                continue
            if in_scope and goal.status == "inactive":
                goal.status = "candidate"
                goal.last_transition_reason = "phase_in_scope"
                self._log_goal_transition(sim_state, goal, "phase_in_scope")

            prereq_rules = set(definition.prerequisite_rules)
            if prereq_rules and not prereq_rules.intersection(knowledge_ids):
                if goal.status in {"active", "candidate", "queued"}:
                    goal.status = "blocked"
                    goal.blocking_reasons = sorted(set((goal.blocking_reasons or []) + ["missing_prerequisite_rules"]))
                    goal.last_transition_reason = "goal_blocked_prerequisites"
                    self._log_goal_transition(sim_state, goal, "goal_blocked_prerequisites")
                continue

            if goal.status == "blocked":
                reasons = set(goal.blocking_reasons or [])
                reasons.discard("missing_prerequisite_rules")
                goal.blocking_reasons = sorted(reasons)
                goal.status = "candidate"
                goal.last_transition_reason = "goal_unblocked_prerequisites"
                self._log_goal_transition(sim_state, goal, "goal_unblocked_prerequisites")

            if goal.status in {"candidate", "queued"} and in_scope:
                goal.status = "active"
                goal.last_transition_reason = "goal_activated_by_scope"
                self._log_goal_transition(sim_state, goal, "goal_activated_by_scope")

            success = (definition.success_conditions or "").lower()
            if "artifact" in success and any(str(p).startswith("whiteboard:") for p in self.memory_seen_packets):
                if goal.status not in {"satisfied", "invalidated", "abandoned"}:
                    goal.status = "satisfied"
                    goal.last_transition_reason = "goal_satisfied_artifact_evidence"
                    self._log_goal_transition(sim_state, goal, "goal_satisfied_artifact_evidence")
            if "dik" in success and (len(info_ids) + len(knowledge_ids)) >= 3 and goal.status in {"active", "candidate", "queued"}:
                goal.status = "satisfied"
                goal.last_transition_reason = "goal_satisfied_dik_evidence"
                self._log_goal_transition(sim_state, goal, "goal_satisfied_dik_evidence")

            if phase_changed and goal.goal_level == "phase" and in_scope and goal.status == "active":
                goal.priority = self._goal_priority(goal.status, min(1.0, goal.priority + 0.1))
                self._log_goal_transition(sim_state, goal, "goal_reprioritized_phase_transition")

        role_source = f"{self.role}_Info"
        missing_baseline_sources = {
            source_id
            for source_id in ("Team_Info", role_source)
            if source_id in environment.knowledge_packets
            and self._has_packet_access(source_id)
            and self.source_inspection_state.get(source_id) != "inspected"
            and not bool(self.source_exhaustion_state.get(source_id, {}).get("exhausted"))
        }
        critical_needs = self._critical_unmet_source_targets(sim_state, environment)
        info_pressure = float(min(3.0, len(critical_needs) + len(missing_baseline_sources)))
        missing_dik = len(self.known_gaps) > 0 or (len(info_ids) + len(knowledge_ids)) < 2 or bool(missing_baseline_sources) or bool(critical_needs)
        if missing_dik:
            info_goal = self._activate_support_goal(
                "acquire_missing_dik",
                "missing_dik_detected",
                sim_state=sim_state,
                priority=min(0.94, 0.82 + (0.04 * info_pressure)),
                source="derived_from_rule",
            )
            if info_goal is not None and info_goal.status in {"active", "candidate", "queued"}:
                info_goal.priority = self._goal_priority(info_goal.status, min(0.95, 0.78 + (0.05 * info_pressure)))

        if any("mismatch with construction" in e.lower() for e in self.activity_log[-6:]):
            self._activate_support_goal("repair_detected_mismatch", "construction_mismatch_detected", sim_state=sim_state, priority=0.85, source="derived_from_rule")
            self._activate_support_goal("validate_externalization", "validation_needed_after_mismatch", sim_state=sim_state, priority=0.8, source="derived_from_rule")

        if getattr(self, "derivation_events", []):
            last = self.derivation_events[-1]
            self._activate_support_goal("integrate_new_derivation", f"derivation:{last.get('derivation_id')}", sim_state=sim_state, priority=0.65, source="derived_from_rule")

        if (
            sim_state is not None
            and self._baseline_epistemic_sources_completed()
            and self._has_meaningful_consultable_artifact(sim_state)
        ):
            self._activate_support_goal("consult_artifact", "teammate/shared-artifact influenced", sim_state=sim_state, priority=0.6, source="teammate_or_artifact_influenced")
        else:
            for goal in self.goal_registry.values():
                if goal.goal_id == "SUPPORT_CONSULT_ARTIFACT" and goal.status in {"active", "queued", "candidate"}:
                    goal.status = "inactive"
                    goal.last_transition_reason = "consult_artifact_deferred_until_baseline_epistemic_ready"
                    self._log_goal_transition(sim_state, goal, "consult_artifact_deferred_until_baseline_epistemic_ready")

        repeated_stall = any(v >= 3 for v in self.inspect_stall_counts.values())
        if repeated_stall:
            self._activate_support_goal("unblock_inspection", "repeated_inspection_stall", sim_state=sim_state, priority=0.75, source="derived_from_rule")

        for goal in self.goal_registry.values():
            if goal.goal_level != "support" or goal.status not in {"active", "candidate", "queued"}:
                continue
            executable, reason = self._support_goal_executable(goal, sim_state, environment)
            if executable:
                self.support_goal_nonexec_counts[goal.goal_id] = 0
                continue
            nonexec_count = int(self.support_goal_nonexec_counts.get(goal.goal_id, 0)) + 1
            self.support_goal_nonexec_counts[goal.goal_id] = nonexec_count
            next_status = "inactive" if nonexec_count >= 2 else "candidate"
            if goal.status != next_status:
                goal.status = next_status
                goal.last_transition_reason = "support_goal_demoted_non_executable"
                self._log_goal_transition(
                    sim_state,
                    goal,
                    "support_goal_demoted_non_executable",
                    extra={"nonexec_count": nonexec_count, "nonexec_reason": reason},
                )
            self._emit_event(
                sim_state,
                "support_goal_suppressed",
                {"goal_id": goal.goal_id, "label": goal.label, "reason": reason, "nonexec_count": nonexec_count},
            )

        if info_pressure > 0:
            self._emit_event(
                sim_state,
                "grounded_info_goal_preferred",
                {
                    "info_pressure": info_pressure,
                    "critical_source_count": len(critical_needs),
                    "missing_baseline_source_count": len(missing_baseline_sources),
                },
            )

        self._refresh_goal_stack_view()
    def _validate_plan_method_grounding(self, response, context):
        notes = []
        status = "accepted"
        method_id = response.plan.plan_method_id
        if not method_id:
            return status, notes
        method = self.task_model.plan_methods.get(method_id) if self.task_model else None
        if method is None or not method.enabled:
            return "rejected_unknown_method", [f"unknown_plan_method:{method_id}"]

        goal_ids = [g.goal_id for g in response.plan.ordered_goals if getattr(g, "goal_id", None)]
        if goal_ids and method.goal_id not in goal_ids:
            notes.append(f"method_goal_mismatch:{method.goal_id}")
            status = "low_trust"

        phase = context.world_snapshot.get("phase_profile", {}).get("name") or context.world_snapshot.get("phase_state", {}).get("name")
        if not self._phase_matches(method.phase_scope, phase):
            notes.append("phase_scope_mismatch")
            status = "low_trust"

        data_ids, info_ids, knowledge_ids = self._held_dik_ids()
        missing_rules = [r for r in method.required_rules if r and r not in knowledge_ids]
        missing_knowledge = [k for k in method.required_knowledge if k and k not in knowledge_ids]
        missing_information = [i for i in method.required_information if i and i not in info_ids]
        missing_data = [d for d in method.required_data if d and d not in data_ids]
        if missing_rules or missing_knowledge or missing_information or missing_data:
            status = "low_trust"
            if missing_rules:
                notes.append(f"missing_rules:{'|'.join(missing_rules[:4])}")
            if missing_knowledge:
                notes.append(f"missing_knowledge:{'|'.join(missing_knowledge[:4])}")
            if missing_information:
                notes.append(f"missing_information:{'|'.join(missing_information[:4])}")
            if missing_data:
                notes.append(f"missing_data:{'|'.join(missing_data[:4])}")

        legal_actions = {a.get("action_type") for a in context.action_affordances}
        illegal_steps = [s.action_type.value for s in response.plan.ordered_actions if s.action_type.value not in legal_actions]
        if illegal_steps:
            status = "low_trust"
            notes.append(f"illegal_plan_steps:{'|'.join(illegal_steps[:4])}")

        if method.required_data and not (set(method.required_data) & set(data_ids)):
            notes.append("required_data_not_grounded")
            status = "low_trust"

        artifact_types = {
            str((a.metadata or {}).get("type", "")).strip().lower()
            for a in getattr(getattr(context, "task_artifact_state", None), "artifacts", [])
        }
        if "consult_team_artifact" in {s.action_type.value for s in response.plan.ordered_actions} and not artifact_types:
            notes.append("consult_without_available_artifact")
            status = "low_trust"
        return status, notes

    def _validate_and_ground_response_plan(self, response, context, request_packet):
        if not self.task_model:
            return response
        method_status, notes = self._validate_plan_method_grounding(response, context)
        if method_status == "rejected_unknown_method":
            fallback_method = None
            for m in self.task_model.plan_methods.values():
                if not m.enabled:
                    continue
                if response.plan.ordered_goals and m.goal_id == response.plan.ordered_goals[0].goal_id:
                    fallback_method = m.method_id
                    break
            replacement = response.plan.ordered_actions[0] if response.plan.ordered_actions else None
            action = replacement.action_type.value if replacement else "wait"
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"planmethod-fallback-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": response.plan.plan_id,
                        "plan_horizon": max(1, response.plan.plan_horizon),
                        "ordered_goals": [g.__dict__ for g in response.plan.ordered_goals],
                        "ordered_actions": [{"step_index": 0, "action_type": action, "expected_purpose": "fallback after unknown method"}],
                        "next_action": {"step_index": 0, "action_type": action, "expected_purpose": "fallback after unknown method"},
                        "plan_method_id": fallback_method,
                        "confidence": min(0.5, float(response.plan.confidence)),
                        "notes": list(response.plan.notes) + ["unknown_method_fallback"],
                    },
                    "explanation": response.explanation,
                }
            )
            method_status = "mapped_fallback"
            notes = notes + ["plan_method_mapped_to_fallback"]
        setattr(response.plan, "_method_status", method_status)
        setattr(response.plan, "_method_notes", notes)
        return response

    def _make_planner_trace_id(self, request_id):
        return f"trace-{request_id}"

    def _exception_payload(self, exc):
        if exc is None:
            return None
        return {"type": type(exc).__name__, "message": str(exc), "repr": repr(exc)}

    def _append_planner_trace(self, sim_state, payload):
        if sim_state is None or not hasattr(sim_state, "logger"):
            return
        sim_state.logger.append_planner_trace(payload)

    def _build_brain_request(self, sim_state, context, request_explanation, trigger_reason):
        phase = context.world_snapshot.get("phase_profile", {}).get("name", context.world_snapshot.get("phase_state", {}).get("name", "default"))
        observations = [str(e) for e in context.history_bands.get("near_preceding_events", [])[-4:]]
        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {}
        bootstrap_summary = None
        accepted = self.dik_integration_state.get("accepted_updates", {}) if isinstance(self.dik_integration_state, dict) else {}
        accepted_information = [item.get("candidate_id") for item in accepted.get("information", []) if isinstance(item, dict)]
        accepted_knowledge = [item.get("candidate_id") for item in accepted.get("knowledge", []) if isinstance(item, dict)]
        accepted_rules = [item.get("candidate_id") for item in accepted.get("rules", []) if isinstance(item, dict)]
        control_snapshot = self.get_control_state_snapshot()
        compact_control_snapshot = {
            "mode": control_snapshot.get("mode"),
            "previous_mode": control_snapshot.get("previous_mode"),
            "mode_dwell_steps": control_snapshot.get("mode_dwell_steps"),
            "last_transition_reason": control_snapshot.get("last_transition_reason"),
            "recovery_active": bool(control_snapshot.get("recovery_active")),
            "top_features": dict(control_snapshot.get("top_features") or {}),
            "policy_snapshot": dict(control_snapshot.get("policy_snapshot") or {}),
            "method_state": dict(control_snapshot.get("method_state") or {}),
        }
        return AgentBrainRequest(
            request_id=f"{self.agent_id}-{uuid.uuid4().hex[:8]}",
            tick=self.sim_step_count,
            sim_time=float(sim_state.time),
            agent_id=self.agent_id,
            display_name=self.display_name,
            agent_label=self.agent_label,
            task_id=getattr(sim_state.task_model, "task_id", "unknown"),
            phase=str(phase),
            local_context_summary=f"trigger={trigger_reason};build={context.individual_cognitive_state.get('build_readiness',{}).get('status')};accepted_dik={len(accepted_information)+len(accepted_knowledge)+len(accepted_rules)}",
            local_observations=observations,
            working_memory_summary={
                "data": list(context.individual_cognitive_state.get("data_summary", [])[:8]),
                "information": list(context.individual_cognitive_state.get("information_summary", [])[:8]),
                "knowledge": list(context.individual_cognitive_state.get("knowledge_summary", [])[:8]),
                "known_gaps": list(context.individual_cognitive_state.get("known_gaps", [])[:8]),
                "accepted_llm_information_updates": accepted_information[:8],
                "accepted_llm_knowledge_updates": accepted_knowledge[:8],
                "accepted_llm_rule_supports": accepted_rules[:8],
            },
            inbox_summary=list(context.team_state.get("recent_communications", [])[-4:]),
            current_goal_stack=list(context.individual_cognitive_state.get("goal_stack", [])),
            current_plan_summary=dict(context.individual_cognitive_state.get("active_plan", {})),
            allowed_actions=list(context.action_affordances),
            planning_horizon_config={"max_steps": self.brain_config.get("plan_horizon", 3)},
            request_explanation=request_explanation,
            explanation_style=self.brain_config.get("explanation_style"),
            task_context=dict(context.static_task_context),
            rule_context=list(context.individual_cognitive_state.get("knowledge_summary", [])[:8]),
            derivation_context=[str(e) for e in self.derivation_events[-5:]],
            artifact_context=list(context.team_state.get("externalized_artifacts", []))[:4],
            bootstrap_summary=bootstrap_summary,
            control_mode=str(control_snapshot.get("mode") or "BOOTSTRAP"),
            previous_control_mode=control_snapshot.get("previous_mode"),
            mode_dwell_steps=int(control_snapshot.get("mode_dwell_steps", 0) or 0),
            last_transition_reason=str(control_snapshot.get("last_transition_reason") or "none"),
            control_state_snapshot=compact_control_snapshot,
        )

    def _is_productive_action(self, action_type):
        return action_type in {
            ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
            ExecutableActionType.CONSULT_TEAM_ARTIFACT.value,
            ExecutableActionType.REQUEST_ASSISTANCE.value,
            ExecutableActionType.COMMUNICATE.value,
            ExecutableActionType.EXTERNALIZE_PLAN.value,
            ExecutableActionType.TRANSPORT_RESOURCES.value,
            ExecutableActionType.START_CONSTRUCTION.value,
            ExecutableActionType.CONTINUE_CONSTRUCTION.value,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
            ExecutableActionType.VALIDATE_CONSTRUCTION.value,
        }

    def _execute_planner_request_sync(self, sim_state, trigger_reason, request_packet, request_explanation, request_started_at, request_sim_time, request_wallclock_time, trace_id, blocking_sim_barrier=False):
        context = sim_state.brain_context_builder.build(sim_state, self)
        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"provider": sim_state.brain_provider, "config": sim_state.brain_backend_config, "configured_backend": getattr(sim_state, "configured_brain_backend", sim_state.brain_backend_config.backend), "effective_backend": getattr(sim_state, "effective_brain_backend", sim_state.brain_backend_config.backend)}
        provider = runtime["provider"]
        provider_cfg = runtime["config"]
        provider_name = provider.__class__.__name__
        configured_backend = runtime.get("configured_backend", getattr(sim_state, "configured_brain_backend", sim_state.brain_backend_config.backend))
        effective_backend = runtime.get("effective_backend", configured_backend)

        response = None
        provider_trace = None
        schema_validation_errors = []
        legacy_validation_errors = []
        validation_repaired = False
        grounding_status = None
        grounding_notes = []
        runtime_disposition = None
        provider_decide = getattr(provider, "decide", None)
        if callable(provider_decide) and provider.__class__.__name__ == "RuleBrain":
            legacy_decision = provider_decide(context)
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"legacy-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": f"legacy-plan-{request_packet.request_id}",
                        "plan_horizon": 1,
                        "ordered_goals": ([{"goal_id": legacy_decision.goal_update, "description": legacy_decision.goal_update, "priority": 0.8, "status": "active"}] if legacy_decision.goal_update else []),
                        "ordered_actions": [{"step_index": 0, "action_type": legacy_decision.selected_action.value, "target_id": legacy_decision.target_id, "target_zone": legacy_decision.target_zone, "expected_purpose": legacy_decision.reason_summary}],
                        "next_action": {"step_index": 0, "action_type": legacy_decision.selected_action.value, "target_id": legacy_decision.target_id, "target_zone": legacy_decision.target_zone, "expected_purpose": legacy_decision.reason_summary},
                        "plan_method_id": legacy_decision.plan_method_id,
                        "confidence": legacy_decision.confidence,
                    },
                    "explanation": legacy_decision.reason_summary if request_explanation else None,
                }
            )
        elif hasattr(provider, "generate_plan"):
            response = provider.generate_plan(request_packet)
        if response is None:
            legacy_decision = provider.decide(context)
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"legacy-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": f"legacy-plan-{request_packet.request_id}",
                        "plan_horizon": 1,
                        "ordered_goals": [],
                        "ordered_actions": [{"step_index": 0, "action_type": legacy_decision.selected_action.value, "target_id": legacy_decision.target_id, "target_zone": legacy_decision.target_zone, "expected_purpose": legacy_decision.reason_summary}],
                        "next_action": {"step_index": 0, "action_type": legacy_decision.selected_action.value, "target_id": legacy_decision.target_id, "target_zone": legacy_decision.target_zone, "expected_purpose": legacy_decision.reason_summary},
                        "plan_method_id": legacy_decision.plan_method_id,
                        "confidence": legacy_decision.confidence,
                    },
                }
            )

        updated_control_state = context.individual_cognitive_state.get("control_state", {})
        if isinstance(updated_control_state, dict) and updated_control_state:
            self.control_state.update(updated_control_state)
            self._sync_method_state_from_control()

        provider_trace = getattr(provider, "last_trace", None)
        if isinstance(provider_trace, dict):
            runtime_disposition = provider_trace.get("runtime_disposition")
        response = self._validate_and_ground_response_plan(response, context, request_packet)
        plan_obj = getattr(response, "plan", None)
        grounding_status = getattr(plan_obj, "_method_status", None) if plan_obj else None
        grounding_notes = list(getattr(plan_obj, "_method_notes", [])) if plan_obj else []
        legal_action_ids = [a["action_type"] for a in context.action_affordances]
        errors = validate_agent_brain_response(response, legal_action_ids)
        schema_validation_errors = list(errors)
        repaired = False
        if errors:
            repaired = True
            validation_repaired = True
            fallback_step = select_productive_fallback_action(request_packet.allowed_actions)
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"fallback-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": f"fallback-plan-{request_packet.request_id}",
                        "plan_horizon": 1,
                        "ordered_goals": [{"goal_id": "safety", "description": "maintain legal progress", "priority": 1.0, "status": "active"}],
                        "ordered_actions": [fallback_step.__dict__],
                        "next_action": fallback_step.__dict__,
                        "confidence": 1.0,
                    },
                    "explanation": None,
                }
            )

        decision = response.plan.next_action.to_brain_decision(
            confidence=max(0.0, min(1.0, float(response.plan.confidence))),
            plan_method_id=response.plan.plan_method_id,
            next_steps=[step.expected_purpose for step in response.plan.ordered_actions[:3] if step.expected_purpose],
        )
        if decision.selected_action in {ExecutableActionType.INSPECT_INFORMATION_SOURCE, ExecutableActionType.CONSULT_TEAM_ARTIFACT, ExecutableActionType.REQUEST_ASSISTANCE} and not decision.target_id:
            match = next((a for a in context.action_affordances if a.get("action_type") == decision.selected_action.value), None)
            if match:
                decision.target_id = match.get("target_id")
                decision.target_zone = match.get("target_zone")

        decision = self._apply_trait_bias_to_decision(decision, context, sim_state, trigger_reason)
        legacy_errors = validate_brain_decision(decision, [ExecutableActionType(a) for a in legal_action_ids])
        legacy_validation_errors = list(legacy_errors)
        status = "repaired" if repaired else "accepted"
        if repaired and runtime_disposition in {None, "accepted_as_is", "accepted_after_unwrap"}:
            runtime_disposition = "accepted_after_repair"
        if legacy_errors:
            status = "rejected"
            runtime_disposition = "rejected_schema_invalid"
            decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="Fallback due to decision validation failure.", confidence=1.0)

        latency_s = max(0.0, time.perf_counter() - float(request_wallclock_time))
        provider_outcome = getattr(provider, "last_outcome", {}) if isinstance(getattr(provider, "last_outcome", {}), dict) else {}
        provider_trace = provider_trace if isinstance(provider_trace, dict) else {}
        llm_response_received = bool(provider_trace.get("llm_response_received", False))
        llm_response_parsed = bool(provider_trace.get("llm_response_parsed", False))
        llm_response_validated = bool(provider_trace.get("llm_response_validated", False))
        timeout_occurred = bool(provider_trace.get("timeout_occurred", False))
        fallback_used = bool(provider_trace.get("fallback_used", False) or provider_outcome.get("fallback"))
        result_source = provider_trace.get("result_source") or provider_outcome.get("result_source") or configured_backend
        disposition = runtime_disposition or provider_trace.get("runtime_disposition")
        if timeout_occurred and not disposition:
            disposition = "timed_out"
        if fallback_used and not disposition:
            disposition = "fallback_generated"
        if grounding_status in {"rejected_unknown_method"}:
            disposition = "rejected_grounding_failure"
        trace_outcome_category = provider_trace.get("trace_outcome_category") or provider_outcome.get("outcome_category")
        if not trace_outcome_category:
            if timeout_occurred:
                trace_outcome_category = "llm_timeout_with_fallback"
            elif fallback_used and llm_response_received and not llm_response_validated:
                trace_outcome_category = "llm_invalid_with_fallback"
            elif fallback_used and not llm_response_received:
                trace_outcome_category = "llm_error_with_fallback"
            else:
                trace_outcome_category = "llm_success"
        return {
            "request_id": request_packet.request_id,
            "trace_id": trace_id,
            "decision": decision,
            "status": status,
            "response": response,
            "errors": list(errors) + list(legacy_errors),
            "provider_name": provider_name,
            "configured_backend": configured_backend,
            "effective_backend": effective_backend,
            "request_explanation": request_explanation,
            "trigger_reason": trigger_reason,
            "request_started_at": request_started_at,
            "request_sim_time": request_sim_time,
            "request_wallclock_time": request_wallclock_time,
            "latency_s": latency_s,
            "failed": False,
            "timed_out": False,
            "llm_response_received": llm_response_received,
            "llm_response_parsed": llm_response_parsed,
            "llm_response_validated": llm_response_validated,
            "timeout_occurred": timeout_occurred,
            "fallback_used": fallback_used,
            "result_source": result_source,
            "fallback_source": provider_trace.get("fallback_source"),
            "trace_outcome_category": trace_outcome_category,
            "runtime_disposition": disposition,
            "blocking_sim_barrier": bool(blocking_sim_barrier),
            "trace": {
                "trace_id": trace_id,
                "schema_validation_succeeded": not bool(schema_validation_errors),
                "schema_validation_errors": list(schema_validation_errors),
                "legacy_decision_validation_errors": list(legacy_validation_errors),
                "validation_repaired": bool(validation_repaired),
                "grounding_status": grounding_status,
                "grounding_notes": list(grounding_notes),
                "runtime_disposition": disposition,
                "provider_trace": provider_trace or None,
                "normalized_agent_brain_response": {
                    "response_id": response.response_id,
                    "agent_id": response.agent_id,
                    "plan": {
                        "plan_id": response.plan.plan_id,
                        "plan_horizon": response.plan.plan_horizon,
                        "ordered_goals": [g.__dict__ for g in response.plan.ordered_goals],
                        "ordered_actions": [a.__dict__ for a in response.plan.ordered_actions],
                        "next_action": response.plan.next_action.__dict__,
                        "confidence": response.plan.confidence,
                        "plan_method_id": response.plan.plan_method_id,
                        "notes": list(response.plan.notes),
                    },
                    "confidence": response.confidence,
                    "explanation": response.explanation,
                },
            },
        }

    def _submit_planner_request_async(self, sim_state, trigger_reason):
        context = sim_state.brain_context_builder.build(sim_state, self)
        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"provider": sim_state.brain_provider, "config": sim_state.brain_backend_config, "configured_backend": getattr(sim_state, "configured_brain_backend", sim_state.brain_backend_config.backend), "effective_backend": getattr(sim_state, "effective_brain_backend", sim_state.brain_backend_config.backend)}
        provider_cfg = runtime["config"]
        configured_backend = runtime.get("configured_backend", getattr(sim_state, "configured_brain_backend", sim_state.brain_backend_config.backend))
        effective_backend = runtime.get("effective_backend", configured_backend)
        blocking_sim_barrier = self.planner_request_blocks_sim_time(sim_state=sim_state, runtime=runtime)
        request_explanation = self._should_request_explanation()
        request_packet = self._build_brain_request(sim_state, context, request_explanation, trigger_reason)
        trace_id = self._make_planner_trace_id(request_packet.request_id)
        self._planner_request_seq += 1
        request_started_at = self.sim_step_count
        request_sim_time = float(sim_state.time)
        request_wallclock_time = time.perf_counter()
        self.planner_state["status"] = "in_flight"
        self.planner_state["request_id"] = request_packet.request_id
        self.planner_state["trace_id"] = trace_id
        self.planner_state["request_tick"] = self.sim_step_count
        self.planner_state["requested_at"] = request_sim_time
        self.planner_state["requested_wallclock_at"] = request_wallclock_time
        self.planner_state["error"] = None
        self.planner_state["trigger_reason"] = trigger_reason
        self.planner_state["request_payload"] = request_packet.to_dict()
        self.planner_state["configured_backend"] = configured_backend
        self.planner_state["effective_backend"] = effective_backend
        self.planner_state["model"] = provider_cfg.local_model
        self.planner_state["blocking_sim_barrier"] = bool(blocking_sim_barrier)
        self.planner_state["barrier_reason"] = "planner_request_in_flight" if blocking_sim_barrier else None
        self.planner_state["last_result"] = None
        self.planner_state["total_started"] += 1
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_request_submitted"):
            sim_state.logger.record_brain_request_submitted(
                {
                    "request_id": request_packet.request_id,
                    "trace_id": trace_id,
                    "request_kind": "planner",
                    "agent_id": self.agent_id,
                    "display_name": self.display_name,
                    "tick": self.sim_step_count,
                    "sim_time": float(sim_state.time),
                    "configured_backend": configured_backend,
                    "effective_backend": effective_backend,
                    "model": provider_cfg.local_model,
                    "trigger_reason": trigger_reason,
                    "request_payload": request_packet.to_dict(),
                    "status": "in_flight",
                    "blocking_sim_barrier": bool(blocking_sim_barrier),
                }
            )
        self._emit_event(sim_state, "planner_request_started_async", {"request_id": request_packet.request_id, "trace_id": trace_id, "trigger_reason": trigger_reason, "backend": configured_backend, "effective_backend": effective_backend, "timeout": self.planner_cadence.planner_timeout_seconds, "model": provider_cfg.local_model, "queue_depth": 1, "high_latency_local_llm_mode": bool(self.planner_cadence.high_latency_local_llm_mode), "blocking_sim_barrier": bool(blocking_sim_barrier)})
        self._emit_event(sim_state, "planner_request_queue_depth", {"request_id": request_packet.request_id, "trace_id": trace_id, "queue_depth": 1, "backend": configured_backend})

        with self._planner_future_lock:
            self._planner_future = sim_state.planner_executor.submit(
                self._execute_planner_request_sync,
                sim_state,
                trigger_reason,
                request_packet,
                request_explanation,
                request_started_at,
                request_sim_time,
                request_wallclock_time,
                trace_id,
                bool(blocking_sim_barrier),
            )

    def _build_dik_integration_request(self, sim_state, trigger_reason):
        candidate_information_ids = []
        candidate_knowledge_ids = []
        candidate_rule_ids = []
        candidate_information_grounding = {}
        candidate_knowledge_grounding = {}
        candidate_rule_grounding = {}
        if self.task_model is not None:
            derivations_by_output = {}
            for derivation in getattr(self.task_model, "derivations", {}).values():
                if not getattr(derivation, "enabled", True):
                    continue
                derivations_by_output.setdefault(str(derivation.output_element_id), []).append(
                    {
                        "derivation_id": str(derivation.derivation_id),
                        "required_inputs": [str(x) for x in derivation.required_inputs],
                        "optional_inputs": [str(x) for x in derivation.optional_inputs],
                        "min_required_count": int(derivation.min_required_count or 0),
                        "derivation_kind": str(derivation.derivation_kind),
                    }
                )
            for element_id, element in getattr(self.task_model, "dik_elements", {}).items():
                if not getattr(element, "enabled", True):
                    continue
                if element.element_type == "information":
                    candidate_information_ids.append(element_id)
                    candidate_information_grounding[element_id] = list(derivations_by_output.get(element_id, []))
                elif element.element_type == "knowledge":
                    candidate_knowledge_ids.append(element_id)
                    candidate_knowledge_grounding[element_id] = list(derivations_by_output.get(element_id, []))
            for rid, rule in getattr(self.task_model, "rules", {}).items():
                if not getattr(rule, "enabled", True):
                    continue
                candidate_rule_ids.append(rid)
                required_info = [str(x) for x in getattr(rule, "required_information", [])]
                required_knowledge = [str(x) for x in getattr(rule, "required_knowledge", [])]
                candidate_rule_grounding[rid] = {
                    "required_information_ids": required_info,
                    "required_knowledge_ids": required_knowledge,
                    "required_evidence_ids": sorted(set(required_info) | set(required_knowledge)),
                }
        return AgentDIKIntegrationRequest(
            request_id=f"dik-{self.agent_id}-{uuid.uuid4().hex[:8]}",
            tick=self.sim_step_count,
            sim_time=float(sim_state.time),
            agent_id=self.agent_id,
            display_name=self.display_name,
            phase=(sim_state.environment.get_current_phase() or {}).get("name", "default"),
            trigger_reason=str(trigger_reason or "epistemic_change"),
            held_data_ids=[getattr(item, "id", str(item)) for item in list(self.mental_model["data"])],
            held_information_ids=[getattr(item, "id", str(item)) for item in list(self.mental_model["information"])],
            held_knowledge_ids=[str(rule) for rule in list(self.mental_model["knowledge"].rules)],
            recent_new_item_ids=[str(x) for x in self.derivation_events[-10:]],
            recent_communication_ids=[str(x.get("id") or x.get("content") or x) for x in self.communication_log[-8:]],
            recent_artifact_ids=[str(k) for k in list(getattr(sim_state.team_knowledge_manager, "artifacts", {}).keys())[-8:]],
            unresolved_gaps=[str(x) for x in list(self.known_gaps)[:8]],
            contradiction_signals=[e for e in self.activity_log[-10:] if "contradiction" in str(e).lower() or "mismatch" in str(e).lower()],
            candidate_information_ids=candidate_information_ids,
            candidate_knowledge_ids=candidate_knowledge_ids,
            candidate_rule_ids=candidate_rule_ids,
            candidate_information_grounding=candidate_information_grounding,
            candidate_knowledge_grounding=candidate_knowledge_grounding,
            candidate_rule_grounding=candidate_rule_grounding,
            max_candidates_per_type=8,
        )

    def _execute_dik_integration_request_sync(self, sim_state, trigger_reason, request_packet, trace_id):
        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"provider": sim_state.brain_provider}
        provider = runtime["provider"]
        response = provider.generate_dik_integration(request_packet)
        return {
            "request_id": request_packet.request_id,
            "trace_id": trace_id,
            "trigger_reason": trigger_reason,
            "response": response,
        }

    def _accept_dik_integration_candidates(self, sim_state, request_packet, response):
        held_ids = {getattr(item, "id", str(item)) for item in self.mental_model["data"]}
        held_ids |= {getattr(item, "id", str(item)) for item in self.mental_model["information"]}
        held_ids |= {str(rule) for rule in self.mental_model["knowledge"].rules}
        valid_information = set(request_packet.candidate_information_ids)
        valid_knowledge = set(request_packet.candidate_knowledge_ids)
        valid_rules = set(request_packet.candidate_rule_ids)
        accepted = {"information": [], "knowledge": [], "rules": []}
        rejected = {"information": [], "knowledge": [], "rules": []}

        def _process(bucket_name, candidates, valid_ids):
            for item in candidates:
                rec = {
                    "candidate_id": item.candidate_id,
                    "evidence_ids": list(item.evidence_ids),
                    "justification": item.justification,
                    "confidence": float(item.confidence),
                }
                id_ok = item.candidate_id in valid_ids
                evidence_ok = all(eid in held_ids for eid in item.evidence_ids)
                if id_ok and evidence_ok:
                    accepted[bucket_name].append(rec)
                else:
                    rec["rejection_reasons"] = [reason for reason, flag in [("unknown_candidate_id", id_ok), ("evidence_not_currently_held", evidence_ok)] if not flag]
                    rejected[bucket_name].append(rec)

        _process("information", response.candidate_information_updates, valid_information)
        _process("knowledge", response.candidate_knowledge_updates, valid_knowledge)
        _process("rules", response.candidate_rule_supports, valid_rules)
        return accepted, rejected

    def _project_accepted_dik_updates(self, sim_state, accepted_updates):
        projected = {"information": [], "knowledge": [], "rules": []}
        if not accepted_updates:
            return projected
        held_data_ids, held_info_ids, held_knowledge_ids = self._held_dik_ids()
        held_ids = set(held_data_ids) | set(held_info_ids) | set(held_knowledge_ids)
        held_ids |= {normalize_rule_token(r) for r in list(self.mental_model["knowledge"].rules)}

        def _evidence_currently_held(rec):
            evidence_ids = [str(e) for e in rec.get("evidence_ids", []) if str(e).strip()]
            return all(e in held_ids for e in evidence_ids)

        for rec in list(accepted_updates.get("information", [])):
            candidate_id = str(rec.get("candidate_id", "")).strip()
            elements = getattr(self.task_model, "dik_elements", {}) if self.task_model is not None else {}
            element = elements.get(candidate_id) if isinstance(elements, dict) else None
            if not candidate_id or element is None or element.element_type != "information":
                self._emit_event(sim_state, "dik_projection_rejected", {"bucket": "information", "candidate_id": candidate_id, "reason": "noncanonical_or_wrong_type"})
                continue
            if candidate_id in held_info_ids:
                self._emit_event(sim_state, "dik_projection_rejected", {"bucket": "information", "candidate_id": candidate_id, "reason": "already_held"})
                continue
            if not _evidence_currently_held(rec):
                self._emit_event(sim_state, "dik_projection_rejected", {"bucket": "information", "candidate_id": candidate_id, "reason": "evidence_not_currently_held"})
                continue
            info_obj = Information(
                str(getattr(element, "element_id", candidate_id)),
                str(getattr(element, "description", candidate_id)),
                source="deterministic_dik_integration",
                tags=[str(getattr(element, "role_scope", "all")), str(getattr(element, "phase_scope", "all")), "accepted_dik_projection"],
            )
            self.mental_model["information"].add(info_obj)
            held_info_ids.add(candidate_id)
            projected["information"].append({"candidate_id": candidate_id, "evidence_ids": list(rec.get("evidence_ids", [])), "provenance": "accepted_dik_projection"})

        for bucket_name in ("knowledge", "rules"):
            for rec in list(accepted_updates.get(bucket_name, [])):
                candidate_id = str(rec.get("candidate_id", "")).strip()
                normalized = normalize_rule_token(candidate_id)
                if bucket_name == "knowledge":
                    element = self.task_model.dik_elements.get(candidate_id) if self.task_model is not None else None
                    canonical = bool(element is not None and element.element_type == "knowledge")
                else:
                    rules = getattr(self.task_model, "rules", {}) if self.task_model is not None else {}
                    canonical = bool(candidate_id in rules)
                if not canonical:
                    self._emit_event(sim_state, "dik_projection_rejected", {"bucket": bucket_name, "candidate_id": candidate_id, "reason": "noncanonical_or_wrong_type"})
                    continue
                if candidate_id in held_knowledge_ids or normalized in self.mental_model["knowledge"].rules:
                    self._emit_event(sim_state, "dik_projection_rejected", {"bucket": bucket_name, "candidate_id": candidate_id, "reason": "already_held"})
                    continue
                if not _evidence_currently_held(rec):
                    self._emit_event(sim_state, "dik_projection_rejected", {"bucket": bucket_name, "candidate_id": candidate_id, "reason": "evidence_not_currently_held"})
                    continue
                evidence_ids = [str(e) for e in rec.get("evidence_ids", []) if str(e).strip()]
                self.mental_model["knowledge"].add_rule(candidate_id, evidence_ids, inferred_by_agents=[self.name])
                held_knowledge_ids.add(candidate_id)
                held_ids.add(candidate_id)
                held_ids.add(normalized)
                projected[bucket_name].append({"candidate_id": candidate_id, "evidence_ids": evidence_ids, "provenance": "accepted_dik_projection"})

        if any(projected.values()):
            self.last_dik_change_time = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
        return projected

    def _submit_dik_integration_request_async(self, sim_state, trigger_reason):
        request_packet = self._build_dik_integration_request(sim_state, trigger_reason)
        trace_id = self._make_planner_trace_id(request_packet.request_id)
        self.dik_integration_state["status"] = "in_flight"
        self.dik_integration_state["request_id"] = request_packet.request_id
        self.dik_integration_state["trigger_reason"] = trigger_reason
        self.dik_integration_state["request_payload"] = request_packet.to_dict()
        self.dik_integration_state["total_started"] = int(self.dik_integration_state.get("total_started", 0)) + 1
        self._emit_event(
            sim_state,
            "dik_integration_request_submitted",
            {
                "request_id": request_packet.request_id,
                "trigger_reason": trigger_reason,
                "candidate_information_count": len(request_packet.candidate_information_ids),
                "candidate_knowledge_count": len(request_packet.candidate_knowledge_ids),
                "candidate_rule_count": len(request_packet.candidate_rule_ids),
            },
        )
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_request_submitted"):
            sim_state.logger.record_brain_request_submitted(
                {
                    "request_id": request_packet.request_id,
                    "trace_id": trace_id,
                    "request_kind": "dik_integration",
                    "agent_id": self.agent_id,
                    "display_name": self.display_name,
                    "tick": self.sim_step_count,
                    "sim_time": float(sim_state.time),
                    "trigger_reason": trigger_reason,
                    "request_payload": request_packet.to_dict(),
                    "status": "in_flight",
                }
            )
        with self._dik_future_lock:
            self._dik_future = sim_state.planner_executor.submit(
                self._execute_dik_integration_request_sync,
                sim_state,
                trigger_reason,
                request_packet,
                trace_id,
            )

    def _poll_dik_integration_request(self, sim_state):
        with self._dik_future_lock:
            future = self._dik_future
        if future is None or not future.done():
            return
        with self._dik_future_lock:
            self._dik_future = None
        result = future.result()
        response = result.get("response")
        request_payload = self.dik_integration_state.get("request_payload") or {}
        request_packet = AgentDIKIntegrationRequest.from_dict(request_payload)
        accepted, rejected = self._accept_dik_integration_candidates(sim_state, request_packet, response)
        projected = self._project_accepted_dik_updates(sim_state, accepted)
        self.dik_integration_state["status"] = "completed"
        self.dik_integration_state["last_completed_step"] = self.sim_step_count
        self.dik_integration_state["total_completed"] = int(self.dik_integration_state.get("total_completed", 0)) + 1
        self.dik_integration_state["total_rejected"] = int(self.dik_integration_state.get("total_rejected", 0)) + sum(len(v) for v in rejected.values())
        self.dik_integration_state["recent_candidates"] = {
            "information": [{"candidate_id": c.candidate_id, "evidence_ids": list(c.evidence_ids), "confidence": float(c.confidence)} for c in response.candidate_information_updates],
            "knowledge": [{"candidate_id": c.candidate_id, "evidence_ids": list(c.evidence_ids), "confidence": float(c.confidence)} for c in response.candidate_knowledge_updates],
            "rules": [{"candidate_id": c.candidate_id, "evidence_ids": list(c.evidence_ids), "confidence": float(c.confidence)} for c in response.candidate_rule_supports],
        }
        self.dik_integration_state["accepted_updates"] = accepted
        self.dik_integration_state["last_sent_held_count"] = len(request_packet.held_data_ids) + len(request_packet.held_information_ids) + len(request_packet.held_knowledge_ids)
        self.dik_integration_state["last_sent_comm_count"] = len(request_packet.recent_communication_ids)
        self.dik_integration_state["last_sent_artifact_count"] = len(request_packet.recent_artifact_ids)
        self.dik_integration_state["last_result"] = {
            "summary": response.summary,
            "confidence": response.confidence,
            "unresolved_gaps": list(response.unresolved_gaps),
            "contradictions": list(response.contradictions),
            "accepted": accepted,
            "rejected": rejected,
            "projected": projected,
        }
        self._emit_event(sim_state, "dik_integration_candidates_proposed", {"request_id": result.get("request_id"), "counts": {k: len(v) for k, v in self.dik_integration_state["recent_candidates"].items()}})
        self._emit_event(sim_state, "dik_integration_candidates_accepted", {"request_id": result.get("request_id"), "counts": {k: len(v) for k, v in accepted.items()}})
        self._emit_event(sim_state, "dik_integration_candidates_rejected", {"request_id": result.get("request_id"), "counts": {k: len(v) for k, v in rejected.items()}, "rejections": rejected})
        self._emit_event(sim_state, "dik_integration_candidates_projected", {"request_id": result.get("request_id"), "counts": {k: len(v) for k, v in projected.items()}, "projected": projected})
        self._emit_event(
            sim_state,
            "dik_integration_completed",
            {
                "request_id": result.get("request_id"),
                "trace_id": result.get("trace_id"),
                "trigger_reason": result.get("trigger_reason"),
                "candidate_updates_returned": sum(len(v) for v in self.dik_integration_state["recent_candidates"].values()),
                "candidate_updates_accepted": sum(len(v) for v in accepted.values()),
                "candidate_updates_rejected": sum(len(v) for v in rejected.values()),
            },
        )
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_response_phase"):
            sim_state.logger.record_brain_response_phase(
                result.get("request_id"),
                {
                    "trace_id": result.get("trace_id"),
                    "sim_time": float(sim_state.time),
                    "tick": self.sim_step_count,
                    "status": "response_received",
                    "normalized_payload_exists": True,
                    "normalized_payload": self.dik_integration_state.get("last_result"),
                },
            )
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_interpretation_phase"):
                sim_state.logger.record_brain_interpretation_phase(
                result.get("request_id"),
                {
                    "trace_id": result.get("trace_id"),
                    "sim_time": float(sim_state.time),
                    "tick": self.sim_step_count,
                    "status": "candidate_updates_processed",
                    "runtime_disposition": "candidate_updates_processed",
                    "candidate_updates_returned": sum(len(v) for v in self.dik_integration_state["recent_candidates"].values()),
                    "candidate_updates_accepted": sum(len(v) for v in accepted.values()),
                    "candidate_updates_rejected": sum(len(v) for v in rejected.values()),
                    "candidate_updates_projected": sum(len(v) for v in projected.values()),
                    "request_status": "completed",
                },
            )

    def _planner_cooldown_remaining(self, sim_state):
        return max(0.0, float(self.planner_state.get("cooldown_until", 0.0)) - float(sim_state.time))


    def _register_planner_failure(self, sim_state, request_id, reason, timed_out=False):
        self.planner_state["status"] = "timed_out" if timed_out else "failed"
        self.planner_state["blocking_sim_barrier"] = False
        self.planner_state["barrier_reason"] = None
        if timed_out:
            self.planner_state["total_timed_out"] += 1
            self.planner_state["llm_timeout_count"] += 1
            self._timed_out_request_ids.add(str(request_id))
        else:
            self.planner_state["total_failed"] += 1
            self.planner_state["llm_invalid_count"] += 1
        self.planner_state["error"] = str(reason)
        self.planner_state["consecutive_failures"] += 1
        self.planner_state["fallback_generated_count"] += 1
        if timed_out:
            self._emit_event(sim_state, "llm_timeout", {"request_id": request_id, "trace_id": self.planner_state.get("trace_id")})
        else:
            self._emit_event(sim_state, "llm_response_invalid", {"request_id": request_id, "trace_id": self.planner_state.get("trace_id")})
        self._emit_event(sim_state, "fallback_result_generated", {"request_id": request_id, "trace_id": self.planner_state.get("trace_id"), "fallback_source": "fallback_safe_policy", "result_source": "fallback_safe_policy"})
        threshold = self.planner_cadence.degraded_consecutive_failures_threshold
        if self.planner_state["consecutive_failures"] >= threshold:
            if not self.planner_state["degraded_mode"]:
                self.planner_state["degraded_mode"] = True
                self.planner_state["degraded_mode_episodes"] += 1
                runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"config": sim_state.brain_backend_config, "configured_backend": sim_state.configured_brain_backend}
                self._emit_event(sim_state, "backend_degraded_mode_started", {"request_id": request_id, "consecutive_failures": self.planner_state["consecutive_failures"], "threshold": threshold, "backend": runtime.get("configured_backend", sim_state.configured_brain_backend), "model": runtime["config"].local_model})
            cooldown = self.planner_cadence.degraded_cooldown_seconds
            self.planner_state["cooldown_until"] = max(float(self.planner_state.get("cooldown_until", 0.0)), float(sim_state.time) + float(cooldown))

        self._append_planner_trace(
            sim_state,
            {
                "trace_id": self.planner_state.get("trace_id"),
                "request_id": request_id,
                "agent_id": self.agent_id,
                "display_name": self.display_name,
                "tick": self.sim_step_count,
                "sim_time": float(sim_state.time),
                "trigger_reason": self.planner_state.get("trigger_reason"),
                "configured_backend": self.planner_state.get("configured_backend"),
                "effective_backend": self.planner_state.get("effective_backend"),
                "model": self.planner_state.get("model"),
                "timeout_s": float(self.planner_cadence.planner_timeout_seconds),
                "agent_brain_request_payload": self.planner_state.get("request_payload"),
                "planner_result": "timed_out" if timed_out else "failed",
                "fallback": True,
                "fallback_used": True,
                "result_source": "fallback_safe_policy",
                "fallback_source": "fallback_safe_policy",
                "fallback_reason": str(reason),
                "timeout_occurred": bool(timed_out),
                "trace_outcome_category": "llm_timeout_with_fallback" if timed_out else "llm_error_with_fallback",
                "runtime_disposition": "timed_out" if timed_out else "fallback_generated",
                "exception": {"type": "TimeoutError" if timed_out else "PlannerExecutionError", "message": str(reason)},
                "schema_validation_succeeded": False,
                "llm_response_received": False,
                "llm_response_parsed": False,
                "llm_response_validated": False,
                "plan_grounding_succeeded": False,
                "plan_disposition": "failed_before_response",
            },
        )
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_response_phase"):
            sim_state.logger.record_brain_response_phase(
                request_id,
                {
                    "trace_id": self.planner_state.get("trace_id"),
                    "sim_time": float(sim_state.time),
                    "tick": self.sim_step_count,
                    "status": "no_response_timeout" if timed_out else "transport_error",
                    "http_response_received": False,
                    "json_parsed": False,
                    "normalized_payload_exists": False,
                    "repair_retry_attempted": False,
                    "raw_response_available": False,
                    "parsed_payload_available": False,
                    "error": str(reason),
                },
            )
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_interpretation_phase"):
            sim_state.logger.record_brain_interpretation_phase(
                request_id,
                {
                    "trace_id": self.planner_state.get("trace_id"),
                    "sim_time": float(sim_state.time),
                    "tick": self.sim_step_count,
                    "status": "timed_out" if timed_out else "transport_error",
                    "runtime_disposition": "timed_out" if timed_out else "fallback_generated",
                    "planner_result": "timed_out" if timed_out else "failed",
                    "usable_plan": False,
                    "failure_mode": "timeout" if timed_out else "transport_error",
                    "request_status": "timed_out" if timed_out else "failed",
                },
            )

    def _check_inflight_timeout(self, sim_state):
        if self.planner_state.get("status") != "in_flight":
            return
        requested_at = self.planner_state.get("requested_wallclock_at")
        if requested_at is None:
            return
        timeout_s = float(self.planner_cadence.planner_timeout_seconds)
        if self.planner_cadence.unrestricted_local_qwen_mode:
            timeout_s = max(timeout_s, float(self.planner_cadence.planner_timeout_seconds))
        elapsed = max(0.0, time.perf_counter() - float(requested_at))
        if elapsed < timeout_s:
            return
        request_id = self.planner_state.get("request_id")
        self._register_planner_failure(sim_state, request_id, reason="timed out", timed_out=True)
        self._emit_event(sim_state, "planner_request_completed_async", {"request_id": request_id, "trace_id": self.planner_state.get("trace_id"), "result": "timed_out", "latency": elapsed, "timeout": timeout_s, "consecutive_failures": self.planner_state["consecutive_failures"], "cooldown_remaining": self._planner_cooldown_remaining(sim_state)})
    def _poll_planner_request(self, sim_state, environment):
        with self._planner_future_lock:
            fut = self._planner_future
        if fut is None or not fut.done():
            return False
        with self._planner_future_lock:
            self._planner_future = None

        request_id = self.planner_state.get("request_id")
        try:
            result = fut.result()
        except Exception as exc:
            self._register_planner_failure(sim_state, request_id, reason=str(exc), timed_out=False)
            self._emit_event(sim_state, "planner_request_completed_async", {"request_id": request_id, "trace_id": self.planner_state.get("trace_id"), "result": "failed", "error": str(exc), "consecutive_failures": self.planner_state["consecutive_failures"], "cooldown_remaining": self._planner_cooldown_remaining(sim_state)})
            return False
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_response_phase"):
            provider_trace = (result.get("trace", {}) or {}).get("provider_trace") or {}
            attempts = provider_trace.get("attempts") or []
            last_attempt = attempts[-1] if attempts else {}
            sim_state.logger.record_brain_response_phase(
                result.get("request_id"),
                {
                    "trace_id": result.get("trace_id"),
                    "sim_time": float(sim_state.time),
                    "tick": self.sim_step_count,
                    "status": (
                        "no_response_timeout"
                        if result.get("timeout_occurred")
                        else ("response_received" if result.get("llm_response_received") else ("transport_error" if result.get("fallback_used") else "pending"))
                    ),
                    "http_response_received": bool(result.get("llm_response_received")),
                    "json_parsed": bool(result.get("llm_response_parsed")),
                    "normalized_payload_exists": bool(last_attempt.get("normalized_response_payload") or (result.get("trace", {}) or {}).get("normalized_agent_brain_response")),
                    "repair_retry_attempted": bool(provider_trace.get("repair_retry_attempted")),
                    "raw_response_available": bool(last_attempt.get("raw_http_response_text")),
                    "parsed_payload_available": bool(last_attempt.get("extracted_response_payload")),
                    "raw_response": last_attempt.get("raw_http_response_text"),
                    "parsed_payload": last_attempt.get("extracted_response_payload"),
                    "normalized_payload": last_attempt.get("normalized_response_payload") or (result.get("trace", {}) or {}).get("normalized_agent_brain_response"),
                    "provider_trace": provider_trace or None,
                },
            )

        if result.get("request_id") != request_id:
            self.planner_state["total_stale_discarded"] += 1
            self._emit_event(sim_state, "planner_request_result_arrived_stale", {"request_id": result.get("request_id"), "expected_request_id": request_id, "trace_id": result.get("trace_id")})
            self._append_planner_trace(sim_state, {"trace_id": result.get("trace_id"), "request_id": result.get("request_id"), "agent_id": self.agent_id, "sim_time": float(sim_state.time), "planner_result": "stale_discarded", "plan_disposition": "discarded_stale_request_id_mismatch", "runtime_disposition": "stale_discarded_request_id_mismatch", "fallback": True, "fallback_used": True, "fallback_source": "stale_response_guard", "result_source": "fallback_safe_policy", "fallback_reason": "request_id_mismatch", "trace_outcome_category": "stale_response_discarded", "late_result_arrived": bool(result.get("request_id") in self._timed_out_request_ids), "late_result_accepted": False, "stale_discard_reason": "request_id_mismatch"})
            if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_interpretation_phase"):
                sim_state.logger.record_brain_interpretation_phase(result.get("request_id"), {"trace_id": result.get("trace_id"), "sim_time": float(sim_state.time), "tick": self.sim_step_count, "status": "stale_discarded", "runtime_disposition": "stale_discarded_request_id_mismatch", "planner_result": "stale_discarded", "usable_plan": False, "failure_mode": "request_id_mismatch", "request_status": "discarded"})
            return False
        late_result_accepted = False
        late_result_elapsed_s = None
        if result.get("request_id") in self._timed_out_request_ids:
            request_wallclock_started = self.planner_state.get("requested_wallclock_at")
            elapsed_since_request_s = None
            if request_wallclock_started is not None:
                elapsed_since_request_s = max(0.0, time.perf_counter() - float(request_wallclock_started))
                late_result_elapsed_s = elapsed_since_request_s
            stale_grace_s = float(self.planner_cadence.high_latency_stale_result_grace_s)
            if (
                self.planner_cadence.high_latency_local_llm_mode
                and elapsed_since_request_s is not None
                and elapsed_since_request_s <= float(self.planner_cadence.planner_timeout_seconds) + stale_grace_s
            ):
                self._timed_out_request_ids.discard(result.get("request_id"))
                late_result_accepted = True
                self._emit_event(sim_state, "planner_request_result_arrived_after_timeout_accepted", {"request_id": result.get("request_id"), "trace_id": result.get("trace_id"), "elapsed_since_request_s": elapsed_since_request_s, "timeout_s": float(self.planner_cadence.planner_timeout_seconds), "stale_grace_s": stale_grace_s})
            else:
                self._timed_out_request_ids.discard(result.get("request_id"))
                self.planner_state["total_stale_discarded"] += 1
                self._emit_event(sim_state, "planner_request_result_arrived_stale", {"request_id": result.get("request_id"), "reason": "arrived_after_timeout", "trace_id": result.get("trace_id")})
                self._append_planner_trace(sim_state, {"trace_id": result.get("trace_id"), "request_id": result.get("request_id"), "agent_id": self.agent_id, "sim_time": float(sim_state.time), "planner_result": "stale_discarded", "plan_disposition": "discarded_arrived_after_timeout", "runtime_disposition": "stale_discarded_arrived_after_timeout", "fallback": True, "fallback_used": True, "fallback_source": "stale_response_guard", "result_source": "fallback_safe_policy", "fallback_reason": "arrived_after_timeout", "trace_outcome_category": "stale_response_discarded", "trace": result.get("trace"), "late_result_arrived": True, "late_result_accepted": False, "late_result_elapsed_s": late_result_elapsed_s, "stale_discard_reason": "arrived_after_timeout"})
                if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_interpretation_phase"):
                    sim_state.logger.record_brain_interpretation_phase(result.get("request_id"), {"trace_id": result.get("trace_id"), "sim_time": float(sim_state.time), "tick": self.sim_step_count, "status": "stale_discarded", "runtime_disposition": "stale_discarded_arrived_after_timeout", "planner_result": "stale_discarded", "usable_plan": False, "failure_mode": "arrived_after_timeout", "late_result_arrived": True, "late_result_accepted": False, "request_status": "discarded"})
                return False

        self.planner_call_count += 1
        self.planner_state["consecutive_inflight_skips"] = 0
        self.last_planner_step = self.sim_step_count
        self.last_planner_time = sim_state.time
        self.planner_state["completed_at"] = sim_state.time
        self.planner_state["last_latency_s"] = result.get("latency_s")
        self.planner_state["last_result"] = result
        self.planner_state["last_result_request_id"] = request_id
        self.planner_state["total_completed"] += 1
        self.planner_state["blocking_sim_barrier"] = False
        self.planner_state["barrier_reason"] = None

        decision = result["decision"]
        status = result["status"]
        response = result["response"]
        trigger_reason = result["trigger_reason"]

        if result.get("llm_response_received"):
            self._emit_event(sim_state, "brain_provider_response_received", {"configured_backend": result["configured_backend"], "effective_backend": result["effective_backend"], "provider_class": result["provider_name"], "has_plan": bool(getattr(response, "plan", None)), "has_explanation": bool(getattr(response, "explanation", None)), "request_id": request_id, "trace_id": result.get("trace_id"), "result_source": result.get("result_source")})
            self._emit_event(sim_state, "llm_response_received", {"request_id": request_id, "trace_id": result.get("trace_id"), "result_source": result.get("result_source")})
        if result.get("llm_response_validated"):
            self._emit_event(sim_state, "llm_response_valid", {"request_id": request_id, "trace_id": result.get("trace_id"), "result_source": result.get("result_source")})
        if result.get("timeout_occurred"):
            self._emit_event(sim_state, "llm_timeout", {"request_id": request_id, "trace_id": result.get("trace_id")})
        if (not result.get("timeout_occurred")) and result.get("fallback_used") and not result.get("llm_response_validated"):
            self._emit_event(sim_state, "llm_transport_error", {"request_id": request_id, "trace_id": result.get("trace_id")})
        if result.get("llm_response_received") and not result.get("llm_response_validated"):
            self._emit_event(sim_state, "llm_response_invalid", {"request_id": request_id, "trace_id": result.get("trace_id")})
        if result.get("fallback_used"):
            self._emit_event(sim_state, "fallback_result_generated", {"request_id": request_id, "trace_id": result.get("trace_id"), "fallback_source": result.get("fallback_source"), "result_source": result.get("result_source")})
            self.fallback_bootstrap["runtime_fallback_triggers"] = int(self.fallback_bootstrap.get("runtime_fallback_triggers", 0)) + 1
            early_window_steps = max(4, int(self.planner_cadence.planner_interval_steps) * 12)
            if (
                self.sim_step_count <= early_window_steps
                and self.fallback_bootstrap["runtime_fallback_triggers"] >= 2
                and not self._fallback_bootstrap_complete(sim_state=sim_state)
            ):
                self._maybe_hard_demote_backend(sim_state, reason="runtime_fallback_repeated", activate_bootstrap=False)
                self.activate_fallback_bootstrap(sim_state=sim_state, reason="runtime_fallback_repeated")

        self._emit_event(sim_state, "planner_invocation_completed", {"trigger_reason": trigger_reason, "decision_status": status, "selected_action": decision.selected_action.value, "request_explanation": result["request_explanation"], "request_id": request_id, "trace_id": result.get("trace_id"), "result_source": result.get("result_source"), "fallback_used": bool(result.get("fallback_used"))})

        if result["errors"]:
            sim_state.logger.log_event(sim_state.time, "planner_next_action_rejected", {"agent": self.name, "errors": list(result["errors"]), "fallback_action": "wait"})
            self._emit_event(sim_state, "brain_provider_response_invalid", {"errors": list(result["errors"]), "schema_parsing_succeeded": False, "request_id": request_id, "trace_id": result.get("trace_id")})

        result_request_tick = result.get("request_started_at")
        request_tick = self.planner_state.get("request_tick")
        if result_request_tick is None:
            result_request_tick = request_tick
        if (
            request_tick is not None
            and result_request_tick is not None
            and request_tick == result_request_tick
            and self.current_plan is not None
            and trigger_reason not in {"no_active_plan", "plan_invalidated", "plan_completed"}
            and getattr(self.current_plan, "source", None) == "planner"
            and (not self.planner_cadence.high_latency_local_llm_mode)
            and (not bool(result.get("blocking_sim_barrier")))
        ):
            self.planner_state["total_stale_discarded"] += 1
            self.planner_state["status"] = "completed"
            self._emit_event(sim_state, "planner_response_discarded_due_to_state_change", {"request_id": request_id, "trace_id": result.get("trace_id"), "request_tick": self.planner_state["request_tick"], "current_tick": self.sim_step_count, "current_plan_id": getattr(self.current_plan, "plan_id", None)})
            if result.get("fallback_used"):
                self._emit_event(sim_state, "fallback_result_rejected", {"request_id": request_id, "trace_id": result.get("trace_id"), "reason": "state_changed_before_adoption"})
            self._append_planner_trace(sim_state, {"trace_id": result.get("trace_id"), "request_id": request_id, "agent_id": self.agent_id, "sim_time": float(sim_state.time), "planner_result": status, "plan_disposition": "discarded_due_to_state_change", "runtime_disposition": "stale_discarded_state_changed", "fallback": bool(result.get("fallback_used")), "fallback_used": bool(result.get("fallback_used")), "fallback_source": result.get("fallback_source"), "result_source": result.get("result_source"), "fallback_reason": "state_changed_before_adoption", "trace_outcome_category": "stale_response_discarded", "trace": result.get("trace"), "late_result_arrived": late_result_accepted, "late_result_accepted": False, "late_result_elapsed_s": late_result_elapsed_s, "stale_discard_reason": "state_changed_before_adoption"})
            if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_interpretation_phase"):
                sim_state.logger.record_brain_interpretation_phase(request_id, {"trace_id": result.get("trace_id"), "sim_time": float(sim_state.time), "tick": self.sim_step_count, "status": "stale_discarded", "runtime_disposition": "stale_discarded_state_changed", "planner_result": status, "usable_plan": False, "failure_mode": "state_changed_before_adoption", "late_result_arrived": late_result_accepted, "late_result_accepted": False, "request_status": "discarded"})
            return False

        decision = self._apply_policy_pivots(decision, environment, sim_state=sim_state, pivot_origin="planner_response")
        self._adopt_new_plan(decision, trigger_reason, sim_state, response=response, trace_id=result.get("trace_id"), result_source=result.get("result_source"), fallback_used=bool(result.get("fallback_used")) )
        self._refresh_goal_plan_state(decision, sim_state, trigger_reason, response=response)
        self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment, sim_state=sim_state)
        self.planner_state["status"] = "completed"
        self.planner_state["consecutive_failure_sum"] += self.planner_state["consecutive_failures"]
        self.planner_state["consecutive_failure_samples"] += 1
        self.planner_state["consecutive_failures"] = 0
        if self.planner_state["degraded_mode"]:
            self.planner_state["degraded_mode"] = False
            self._emit_event(sim_state, "backend_degraded_mode_ended", {"agent_id": self.agent_id, "request_id": request_id})
        if result.get("fallback_used"):
            self.planner_state["requests_completed_with_fallback"] += 1
            self.planner_state["fallback_adopted_count"] += 1
            self._emit_event(sim_state, "planner_request_completed_with_fallback", {"request_id": request_id, "trace_id": result.get("trace_id"), "result_source": result.get("result_source"), "fallback_source": result.get("fallback_source")})
            self._emit_event(sim_state, "fallback_result_adopted", {"request_id": request_id, "trace_id": result.get("trace_id"), "fallback_source": result.get("fallback_source")})
        else:
            self.planner_state["requests_completed_with_llm"] += 1
            self._emit_event(sim_state, "planner_request_completed_with_llm", {"request_id": request_id, "trace_id": result.get("trace_id"), "result_source": result.get("result_source")})

        if result.get("llm_response_validated"):
            self.planner_state["llm_success_count"] += 1
            self.fallback_bootstrap["runtime_fallback_triggers"] = 0
        if result.get("timeout_occurred"):
            self.planner_state["llm_timeout_count"] += 1
        if result.get("llm_response_received") and not result.get("llm_response_validated"):
            self.planner_state["llm_invalid_count"] += 1
        if result.get("fallback_used") and (not result.get("timeout_occurred")) and (not result.get("llm_response_received") or not result.get("llm_response_validated")):
            if not result.get("llm_response_received"):
                self.planner_state["llm_transport_error_count"] += 1

        self._emit_event(sim_state, "planner_request_completed_async", {"request_id": request_id, "trace_id": result.get("trace_id"), "result": status, "latency": result.get("latency_s"), "consecutive_failures": self.planner_state["consecutive_failures"], "result_source": result.get("result_source"), "fallback_used": bool(result.get("fallback_used"))})
        translated_actions = list(self.current_action or [])
        if result.get("fallback_used"):
            selected_action = decision.selected_action.value
            if self._is_productive_action(selected_action):
                self.planner_state["productive_fallback_action_count"] += 1
            elif selected_action in {ExecutableActionType.OBSERVE_ENVIRONMENT.value, ExecutableActionType.WAIT.value}:
                self.planner_state["idle_fallback_action_count"] += 1

        self._append_planner_trace(sim_state, {
            "trace_id": result.get("trace_id"),
            "request_id": request_id,
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "tick": self.sim_step_count,
            "sim_time": float(sim_state.time),
            "trigger_reason": trigger_reason,
            "configured_backend": result.get("configured_backend"),
            "effective_backend": result.get("effective_backend"),
            "provider_class": result.get("provider_name"),
            "model": self.planner_state.get("model"),
            "timeout_s": float(self.planner_cadence.planner_timeout_seconds),
            "latency_s": result.get("latency_s"),
            "fallback": bool(result.get("fallback_used")),
            "fallback_used": bool(result.get("fallback_used")),
            "fallback_source": result.get("fallback_source"),
            "result_source": result.get("result_source"),
            "fallback_reason": (result.get("trace", {}).get("provider_trace") or {}).get("fallback_reason") or ("validation_repaired" if status == "repaired" else None),
            "llm_response_received": bool(result.get("llm_response_received")),
            "llm_response_parsed": bool(result.get("llm_response_parsed")),
            "llm_response_validated": bool(result.get("llm_response_validated")),
            "timeout_occurred": bool(result.get("timeout_occurred")),
            "trace_outcome_category": result.get("trace_outcome_category") or ("llm_success" if not result.get("fallback_used") else "llm_error_with_fallback"),
            "runtime_disposition": result.get("runtime_disposition") or ("fallback_adopted" if result.get("fallback_used") else "accepted_as_is"),
            "late_result_arrived": late_result_accepted,
            "late_result_accepted": late_result_accepted,
            "late_result_elapsed_s": late_result_elapsed_s,
            "stale_discard_reason": None,
            "agent_brain_request_payload": self.planner_state.get("request_payload"),
            "planner_result": status,
            "schema_validation_succeeded": not bool(result.get("trace", {}).get("schema_validation_errors")),
            "schema_validation_errors": result.get("trace", {}).get("schema_validation_errors"),
            "plan_grounding_succeeded": result.get("trace", {}).get("grounding_status") not in {"rejected_unknown_method", None},
            "plan_grounding_status": result.get("trace", {}).get("grounding_status"),
            "plan_grounding_notes": result.get("trace", {}).get("grounding_notes"),
            "plan_disposition": "adopted",
            "normalized_agent_brain_response": result.get("trace", {}).get("normalized_agent_brain_response"),
            "provider_trace": result.get("trace", {}).get("provider_trace"),
            "provider_attempts": (result.get("trace", {}).get("provider_trace") or {}).get("attempts") if isinstance(result.get("trace", {}).get("provider_trace"), dict) else None,
            "provider_request_payload": (result.get("trace", {}).get("provider_trace") or {}).get("provider_request_payload") if isinstance(result.get("trace", {}).get("provider_trace"), dict) else None,
            "raw_http_response_text": (((result.get("trace", {}).get("provider_trace") or {}).get("attempts") or [{}])[-1].get("raw_http_response_text") if isinstance(result.get("trace", {}).get("provider_trace"), dict) else None),
            "parsed_response_json": (((result.get("trace", {}).get("provider_trace") or {}).get("attempts") or [{}])[-1].get("parsed_response_json") if isinstance(result.get("trace", {}).get("provider_trace"), dict) else None),
            "extracted_response_payload": (((result.get("trace", {}).get("provider_trace") or {}).get("attempts") or [{}])[-1].get("extracted_response_payload") if isinstance(result.get("trace", {}).get("provider_trace"), dict) else None),
            "next_action_summary": {"action_type": decision.selected_action.value, "target_id": decision.target_id, "target_zone": decision.target_zone, "reason": decision.reason_summary},
            "action_translation_outcome": "succeeded" if translated_actions else "none",
            "translated_actions": translated_actions,
        })
        if hasattr(sim_state, "logger") and hasattr(sim_state.logger, "record_brain_interpretation_phase"):
            sim_state.logger.record_brain_interpretation_phase(
                request_id,
                {
                    "trace_id": result.get("trace_id"),
                    "sim_time": float(sim_state.time),
                    "tick": self.sim_step_count,
                    "status": str(result.get("runtime_disposition") or ("fallback_adopted" if result.get("fallback_used") else "accepted_as_is")),
                    "runtime_disposition": result.get("runtime_disposition") or ("fallback_adopted" if result.get("fallback_used") else "accepted_as_is"),
                    "planner_result": status,
                    "usable_plan": True,
                    "adopted_action": decision.selected_action.value,
                    "fallback_source": result.get("fallback_source"),
                    "fallback_used": bool(result.get("fallback_used")),
                    "schema_validation_errors": (result.get("trace", {}) or {}).get("schema_validation_errors"),
                    "plan_grounding_status": (result.get("trace", {}) or {}).get("grounding_status"),
                    "plan_grounding_notes": (result.get("trace", {}) or {}).get("grounding_notes"),
                    "late_result_arrived": late_result_accepted,
                    "late_result_accepted": late_result_accepted,
                    "request_status": "completed",
                },
            )
        selected_action_value = decision.selected_action.value
        if self.selection_loop_guard.get("last_action") == selected_action_value:
            self.selection_loop_guard["consecutive_count"] = int(self.selection_loop_guard.get("consecutive_count", 0) or 0) + 1
        else:
            self.selection_loop_guard["last_action"] = selected_action_value
            self.selection_loop_guard["consecutive_count"] = 1
        sim_state.logger.log_event(
            sim_state.time,
            "brain_decision_outcome",
            {
                "agent": self.name,
                "trigger_reason": trigger_reason,
                "provider": result["provider_name"],
                "configured_brain_backend": result["configured_backend"],
                "effective_brain_backend": result["effective_backend"],
                "decision_status": status,
                "selected_action": selected_action_value,
                "confidence": decision.confidence,
                "errors": result["errors"],
                "request_explanation": result["request_explanation"],
                "explanation_present": bool(response.explanation),
                "planner_call_count": self.planner_call_count,
                "result_source": result.get("result_source"),
                "fallback_used": bool(result.get("fallback_used")),
                "trace_outcome_category": result.get("trace_outcome_category"),
            },
        )
        return True
    def _apply_trait_bias_to_decision(self, decision, context, sim_state, trigger_reason):
        comm = self._trait_value("communication_propensity")
        align = self._trait_value("goal_alignment")
        help_t = self._trait_value("help_tendency")

        selected = decision.selected_action
        reason_bits = []
        ext_utility = self._hook_value("action_utility", "externalize_plan", "utility_weight", default=0.5)
        consult_utility = self._hook_value("action_utility", "consult_team_artifact", "utility_weight", default=0.5)
        assist_utility = self._hook_value("action_utility", "request_assistance", "utility_weight", default=0.5)
        action_repeat_count = int(getattr(self, "loop_counters", {}).get("action_repeats", 0) or 0)
        repeated_selected_action_count = int(getattr(self, "selection_loop_guard", {}).get("consecutive_count", 0) or 0)
        seconds_since_dik_change = (
            None
            if float(getattr(self, "last_dik_change_time", -1.0) or -1.0) < 0.0
            else max(0.0, float(sim_state.time) - float(getattr(self, "last_dik_change_time", 0.0)))
        )
        assistance_stalled = (
            max(action_repeat_count, repeated_selected_action_count) >= 3
            and (seconds_since_dik_change is None or float(seconds_since_dik_change) > 8.0)
        )
        force_externalize = trigger_reason == "new_dik_acquired" and comm >= 0.7 and random.random() < ((comm + ext_utility) / 2.0)

        if force_externalize:
            selected = ExecutableActionType.EXTERNALIZE_PLAN
            reason_bits.append("communication_propensity pushed externalization after DIK change")

        if align >= 0.75 and context.team_state.get("plan_readiness") == "validated_shared_plan" and random.random() < ((align + consult_utility) / 2.0):
            selected = ExecutableActionType.CONSULT_TEAM_ARTIFACT
            reason_bits.append("goal_alignment favored validated team artifact consultation")

        if (
            help_t >= 0.7
            and self._help_context_available(sim_state)
            and not assistance_stalled
            and random.random() < ((help_t + assist_utility) / 2.0)
        ):
            selected = ExecutableActionType.REQUEST_ASSISTANCE
            reason_bits.append("help_tendency redirected toward assistance exchange")

        if selected != decision.selected_action:
            decision.selected_action = selected
            decision.reason_summary = (decision.reason_summary + " | " + "; ".join(reason_bits)).strip(" |")
            sim_state.logger.log_event(
                sim_state.time,
                "trait_influenced_decision",
                {
                    "agent": self.name,
                    "trigger": trigger_reason,
                    "to": selected.value,
                    "rationale": reason_bits,
                    "traits": {
                        "communication_propensity": comm,
                        "goal_alignment": align,
                        "help_tendency": help_t,
                    },
                },
            )
        return decision

    def _next_plan_id(self):
        self.plan_counter += 1
        return f"{self.name}-plan-{self.plan_counter}"

    def _compact_history(self):
        near_window = 8
        near_events = self.activity_log[-near_window:]
        older_events = self.activity_log[:-near_window]
        if older_events:
            snippet = " | ".join(older_events[-3:])
            previous = self.history_compaction["recent_history_summary"]
            merged = " | ".join([part for part in [previous, snippet] if part])
            self.history_compaction["recent_history_summary"] = merged[-400:]
            self.history_compaction["compaction_count"] += 1

        plan_evolution = []
        if self.current_plan is not None:
            plan_evolution.append(
                f"{self.current_plan.plan_id}:{self.current_plan.decision.selected_action.value}:remaining={self.current_plan.remaining_executions}"
            )
            if self.current_plan.invalidation_reason:
                plan_evolution.append(f"invalidated:{self.current_plan.invalidation_reason}")
        self.history_compaction["plan_evolution"] = plan_evolution

    def history_bands(self):
        self._compact_history()
        near_events = self.activity_log[-8:]
        return {
            "current_state_summary": f"goal={self.goal} active_actions={len(self.active_actions)} plan={getattr(self.current_plan, 'plan_id', None)}",
            "near_preceding_events": near_events,
            "recent_history_summary": self.history_compaction["recent_history_summary"],
            "recent_plan_history": list(self.history_compaction["plan_evolution"]),
        }

    def _plan_trigger_reason(self, sim_state, environment):
        now = sim_state.time
        phase_name = (environment.get_current_phase() or {}).get("name")
        phase_profile = sim_state.brain_context_builder._phase_profile(environment)
        phase_stage = phase_profile.get("stage", "execution")
        info_count = len(self.mental_model["information"])
        knowledge_count = len(self.mental_model["knowledge"].rules)
        build_readiness = self._build_readiness_score()
        build_blockers = self._build_readiness_blockers(environment, sim_state=sim_state)
        team_updates = len(sim_state.team_knowledge_manager.recent_updates)
        recent_log = " ".join(self.activity_log[-4:]).lower()

        if self.current_plan is None:
            reason = "no_active_plan"
        elif self.current_plan.remaining_executions <= 0:
            reason = "plan_completed"
        elif self.current_plan.invalidation_reason:
            reason = "plan_invalidated"
        elif (phase_name != self.last_phase_name and self.last_phase_name is not None) or phase_stage != self.last_phase_stage:
            reason = "phase_transition"
        elif info_count > self.last_info_count or knowledge_count > self.last_knowledge_count:
            reason = "new_dik_acquired"
        elif "mismatch with construction" in recent_log:
            reason = "contradiction_detected"
        elif "blocked while moving" in recent_log:
            reason = "path_blocked_or_stalled"
        elif team_updates > self.last_team_update_count:
            reason = "communication_update_received"
        elif build_readiness != self.last_build_readiness or build_blockers != self.last_build_blockers:
            reason = "build_readiness_changed"
        elif self.current_plan and (now - self.current_plan.created_at) >= (self.plan_expiry_s * (1.0 + self._hook_value("plan_control", "continue_current_plan", "persistence_weight", default=0.0))):
            reason = "plan_invalidated"
        elif self.current_plan and (now - self.current_plan.last_reviewed_at) >= self.plan_review_interval_s:
            reason = "periodic_reassessment"
        else:
            reason = None

        self.last_phase_name = phase_name
        self.last_phase_stage = phase_stage
        self.last_info_count = info_count
        self.last_knowledge_count = knowledge_count
        self.last_build_readiness = build_readiness
        self.last_build_blockers = list(build_blockers)
        self.last_team_update_count = team_updates
        return reason

    def _planner_decision_allowed(self, sim_state, trigger_reason):
        cfg = self.planner_cadence
        if not cfg.planner_enabled:
            return False, "planner_disabled"
        if cfg.planner_request_policy == "cadence_with_dik_integration":
            if trigger_reason and trigger_reason in {"no_active_plan", "plan_invalidated"}:
                return True, trigger_reason
            if trigger_reason == "phase_transition":
                return True, trigger_reason
            interval = max(1, int(cfg.split_mode_planning_interval_steps))
            tick = max(0, int(self.sim_step_count))
            if tick == 0 or (tick % interval == 0):
                return True, f"split_mode_cadence:{interval}"
            return False, "split_mode_cadence_not_due"

        if trigger_reason and trigger_reason in {"no_active_plan", "plan_completed", "plan_invalidated"}:
            return True, trigger_reason

        if trigger_reason and cfg.planner_trigger_mask and trigger_reason in cfg.planner_trigger_mask:
            return True, f"trigger:{trigger_reason}"

        cadence_multiplier = cfg.degraded_step_interval_multiplier if self.planner_state.get("degraded_mode") else 1.0
        steps_since = self.sim_step_count - self.last_planner_step if self.last_planner_step >= 0 else self.sim_step_count
        time_since = sim_state.time - self.last_planner_time if self.last_planner_time >= 0 else sim_state.time
        step_due = steps_since >= max(1, int(cfg.planner_interval_steps * cadence_multiplier))
        time_due = time_since >= (cfg.planner_interval_time * cadence_multiplier)
        if step_due or time_due:
            due_bits = []
            if step_due:
                due_bits.append("step_interval")
            if time_due:
                due_bits.append("time_interval")
            return True, "+".join(due_bits)
        return False, "cadence_not_due"

    def _dik_integration_trigger_reason(self, sim_state, trigger_reason):
        if self.planner_cadence.planner_request_policy != "cadence_with_dik_integration":
            return None
        if trigger_reason in {"new_dik_acquired", "communication_update_received", "contradiction_detected"}:
            return trigger_reason
        if trigger_reason == "build_readiness_changed":
            return "readiness_shift_with_epistemic_impact"
        return None

    def _dik_integration_allowed(self, sim_state, trigger_reason):
        if not trigger_reason:
            return False, "not_triggered"
        cooldown = max(1, int(self.planner_cadence.split_mode_dik_integration_cooldown_steps))
        last = int(self.dik_integration_state.get("last_completed_step", -1) or -1)
        if last >= 0 and (self.sim_step_count - last) < cooldown:
            return False, "cooldown_active"
        state = self.dik_integration_state
        held_count = len(self.mental_model["data"]) + len(self.mental_model["information"]) + len(self.mental_model["knowledge"].rules)
        team_comm_count = len(getattr(sim_state.team_knowledge_manager, "recent_updates", []))
        artifact_count = len(getattr(sim_state.team_knowledge_manager, "artifacts", {}))
        delta = max(
            0,
            held_count - int(state.get("last_sent_held_count", 0)),
            team_comm_count - int(state.get("last_sent_comm_count", 0)),
            artifact_count - int(state.get("last_sent_artifact_count", 0)),
        )
        if delta < int(self.planner_cadence.split_mode_dik_batch_threshold):
            return False, "batch_threshold_not_met"
        return True, trigger_reason

    def _refresh_goal_plan_state(self, decision, sim_state, trigger_reason, response=None):
        if response and getattr(response, "plan", None) and response.plan.ordered_goals:
            for idx, g in enumerate(response.plan.ordered_goals[:5]):
                known_task_goal = bool(self.task_model and g.goal_id and g.goal_id in self.task_model.goals)
                canonical_match = None
                if not known_task_goal and self.task_model:
                    for candidate in self.task_model.goals.values():
                        if candidate.enabled and str(candidate.label).strip().lower() == str(g.description or "").strip().lower():
                            canonical_match = candidate
                            break
                if canonical_match:
                    known_task_goal = True
                    goal_id = canonical_match.goal_id
                    source = "task_defined"
                    trust_tier = "canonical"
                else:
                    goal_id = g.goal_id if known_task_goal else None
                    source = "task_defined" if known_task_goal else "planner_proposed"
                    trust_tier = "canonical" if known_task_goal else "low"

                parent = g.parent_goal_id if known_task_goal else None
                goal_level = self.task_model.goals[goal_id].goal_level if (known_task_goal and self.task_model and goal_id in self.task_model.goals) else "support"
                rec = self._upsert_goal_record(
                    label=g.description or g.goal_id,
                    goal_id=goal_id,
                    source=source,
                    status="active" if idx == 0 else (g.status or "queued"),
                    priority=g.priority,
                    parent_goal_key=parent,
                    evidence=[f"planner_trigger={trigger_reason}", f"planner_source={g.source or source}"],
                    goal_level=goal_level,
                    goal_type="canonical" if known_task_goal else "ad_hoc",
                    trust_tier=trust_tier,
                    sim_state=sim_state,
                    reason="planner_goal_refresh",
                )
                if not known_task_goal:
                    self._upsert_goal_record(
                        label=f"adhoc_subgoal:{g.description or g.goal_id}",
                        source="planner_proposed",
                        status="candidate",
                        priority=max(0.3, g.priority * 0.7),
                        parent_goal_key=rec.goal_key,
                        evidence=["wrapped_ad_hoc_goal"],
                        goal_level="support",
                        goal_type="ad_hoc_subgoal",
                        trust_tier="low",
                        sim_state=sim_state,
                        reason="planner_adhoc_goal_wrapped",
                    )
        elif decision.goal_update:
            self._upsert_goal_record(
                label=decision.goal_update,
                source="planner_proposed",
                status="active",
                target=decision.target_id,
                evidence=[f"decision_action={decision.selected_action.value}"],
                goal_level="support",
                goal_type="ad_hoc",
                trust_tier="low",
                sim_state=sim_state,
                reason="planner_decision_goal_update",
            )

        self._update_goal_states_from_runtime(sim_state, sim_state.environment)

        next_steps = decision.next_steps or decision.plan_steps
        if next_steps:
            self.activity_log.append(f"Planner next steps ({trigger_reason}): {' -> '.join(next_steps[:3])}")

        sim_state.logger.log_event(
            sim_state.time,
            "planner_goal_plan_refresh",
            {
                "agent": self.name,
                "trigger_reason": trigger_reason,
                "goal_update": decision.goal_update,
                "plan_method_id": decision.plan_method_id,
                "next_steps": list(next_steps[:5]),
                "active_goal_count": len(self.goal_stack),
            },
        )
    def _adopt_new_plan(self, decision, trigger_reason, sim_state, response=None, trace_id=None, result_source=None, fallback_used=False):
        if self.current_plan is not None and self.current_plan.invalidation_reason is None:
            self.current_plan.invalidation_reason = f"replaced_by_{trigger_reason}"

        plan_obj = getattr(response, "plan", None)
        method_status = getattr(plan_obj, "_method_status", "unspecified") if plan_obj else "unspecified"
        method_notes = list(getattr(plan_obj, "_method_notes", [])) if plan_obj else []
        active_goal_ids = {g.get("goal_id") for g in self.goal_stack if g.get("goal_id")}
        associated_goal_ids = [
            g.goal_id
            for g in getattr(plan_obj, "ordered_goals", [])
            if getattr(g, "goal_id", None) and (not active_goal_ids or g.goal_id in active_goal_ids)
        ] if plan_obj else []

        next_plan_id = getattr(plan_obj, "plan_id", self._next_plan_id())
        if self.loop_counters.get("plan_signature") == next_plan_id:
            self.loop_counters["plan_repeats"] += 1
        else:
            self.loop_counters["plan_signature"] = next_plan_id
            self.loop_counters["plan_repeats"] = 1
        if self.loop_counters["plan_repeats"] >= 3:
            self._emit_event(sim_state, "repeated_plan_loop_detected", {"plan_id": next_plan_id, "repetition_count": self.loop_counters["plan_repeats"], "window_size": 3})

        self.current_plan = PlanRecord(
            plan_id=next_plan_id,
            decision=decision,
            created_at=sim_state.time,
            last_reviewed_at=sim_state.time,
            trigger_reason=trigger_reason,
            remaining_executions=max(12, int(getattr(plan_obj, "plan_horizon", 2)) + 3),
            ordered_goals=[g.__dict__ for g in getattr(plan_obj, "ordered_goals", [])],
            ordered_actions=[a.__dict__ for a in getattr(plan_obj, "ordered_actions", [])],
            explanation=getattr(response, "explanation", None),
            plan_method_id=getattr(plan_obj, "plan_method_id", None),
            plan_method_status=method_status,
            adoption_reason=f"trigger={trigger_reason};status={method_status}",
            validation_notes=method_notes,
            associated_goal_ids=associated_goal_ids,
        )
        self._emit_startup_once(
            sim_state,
            "initial_plan_selected",
            "initial_plan_selected",
            {
                "plan_id": self.current_plan.plan_id,
                "next_action_type": decision.selected_action.value,
                "result_source": result_source,
                "fallback_used": bool(fallback_used),
            },
        )
        self._emit_event(sim_state, "plan_adopted", {"plan_id": self.current_plan.plan_id, "plan_method_id": self.current_plan.plan_method_id, "trust_tier": method_status, "canonical_goal_matches": len([g for g in (self.current_plan.associated_goal_ids or []) if g]), "ad_hoc_goal_count": len([g for g in (self.current_plan.ordered_goals or []) if not g.get("goal_id")]), "next_action_type": decision.selected_action.value, "trace_id": trace_id, "result_source": result_source, "fallback_used": bool(fallback_used)})
        sim_state.logger.log_event(
            sim_state.time,
            "plan_method_grounding_result",
            {
                "agent": self.name,
                "plan_id": self.current_plan.plan_id,
                "plan_method_id": self.current_plan.plan_method_id,
                "plan_method_status": method_status,
                "validation_notes": list(method_notes),
                "adoption_reason": self.current_plan.adoption_reason,
            },
        )

    def _continue_cached_plan(self, sim_state, environment):
        if self.current_plan is None:
            return False
        if self.current_plan.remaining_executions <= 0:
            self.current_plan.invalidation_reason = "plan_completed"
            sim_state.logger.log_event(sim_state.time, "plan_invalidated", {"agent": self.name, "plan_id": self.current_plan.plan_id, "reason": self.current_plan.invalidation_reason, "trace_id": self.planner_state.get("trace_id")})
            return False

        active_goal_ids = {g.get("goal_id") for g in self.goal_stack if g.get("status") in {"active", "queued", "candidate"}}
        plan_goal_ids = set(self.current_plan.associated_goal_ids or [])
        if plan_goal_ids and not (plan_goal_ids & active_goal_ids):
            self.current_plan.invalidation_reason = "plan_goal_mismatch"
            if not self.startup_state.get("first_movement_started"):
                self.planner_state["startup_plan_invalidations"] = int(self.planner_state.get("startup_plan_invalidations", 0)) + 1
                self._emit_event(sim_state, "initial_plan_invalidated", {"plan_id": self.current_plan.plan_id, "reason": self.current_plan.invalidation_reason})
            sim_state.logger.log_event(sim_state.time, "plan_invalidated", {"agent": self.name, "plan_id": self.current_plan.plan_id, "reason": self.current_plan.invalidation_reason, "trace_id": self.planner_state.get("trace_id")})
            return False

        if self.current_plan.decision.selected_action in {
            ExecutableActionType.INSPECT_INFORMATION_SOURCE,
            ExecutableActionType.REQUEST_ASSISTANCE,
        }:
            context = sim_state.brain_context_builder.build(sim_state, self)
            self.current_plan.decision = self._apply_policy_pivots(
                self.current_plan.decision,
                environment,
                sim_state=sim_state,
                context=context,
                pivot_origin="cached_plan",
            )

        self.current_action = self._translate_brain_decision_to_legacy_action(self.current_plan.decision, environment, sim_state=sim_state)
        self.current_plan.remaining_executions -= 1
        self.current_plan.last_reviewed_at = sim_state.time
        sim_state.logger.log_event(
            sim_state.time,
            "brain_plan_continued",
            {
                "agent": self.name,
                "plan_id": self.current_plan.plan_id,
                "remaining_executions": self.current_plan.remaining_executions,
            },
        )
        action_sig = f"{self.current_plan.plan_id}:{self.current_plan.decision.selected_action.value}:{self.current_plan.decision.target_id}"
        if self.loop_counters.get("action_signature") == action_sig:
            self.loop_counters["action_repeats"] += 1
        else:
            self.loop_counters["action_signature"] = action_sig
            self.loop_counters["action_repeats"] = 1
        if self.loop_counters["action_repeats"] >= 3:
            self._emit_event(sim_state, "repeated_action_loop_detected", {"repetition_count": self.loop_counters["action_repeats"], "window_size": 3, "plan_id": self.current_plan.plan_id, "selected_action": self.current_plan.decision.selected_action.value})
        return True

    def _apply_policy_pivots(self, decision, environment, sim_state=None, context=None, pivot_origin="runtime"):
        context = context or (sim_state.brain_context_builder.build(sim_state, self) if sim_state is not None else None)
        rewritten = decision
        handoff = dict(self.post_inspect_handoff or {})
        now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
        if decision.selected_action == ExecutableActionType.INSPECT_INFORMATION_SOURCE and handoff.get("pending") and handoff.get("expires_at", 0.0) >= now_ts:
            # Compatibility-only bootstrap when RuleBrain controller is unavailable.
            followup = self._choose_post_inspect_followup_decision(environment, sim_state=sim_state)
            self.post_inspect_handoff["pending"] = False
            if followup.selected_action != ExecutableActionType.INSPECT_INFORMATION_SOURCE:
                rewritten = followup
                self._emit_event(sim_state, "policy_pivot_applied", {"origin": pivot_origin, "kind": "post_inspect", "previous_action": decision.selected_action.value, "pivoted_to": rewritten.selected_action.value, "reason": handoff.get("outcome")})
            if context is not None and sim_state is not None:
                rerouted = self._reroute_decision_through_rulebrain_controller(
                    context,
                    rewritten,
                    sim_state=sim_state,
                    pivot_origin=pivot_origin,
                    reason=f"post_inspect:{handoff.get('outcome') or 'none'}",
                )
                if rerouted is not None:
                    rewritten = rerouted
        if context is not None and rewritten.selected_action in {ExecutableActionType.INSPECT_INFORMATION_SOURCE, ExecutableActionType.REQUEST_ASSISTANCE}:
            readiness = context.individual_cognitive_state.get("build_readiness", {})
            built_state = context.world_snapshot.get("built_state", [])
            active_incomplete_projects = [item for item in built_state if item.get("state") in {"absent", "in_progress"} and float(item.get("progress", 0.0)) < 1.0]
            mismatch_signals = context.history_bands.get("semantic_plan_evolution", {}).get("unresolved_contradictions", [])
            seconds_since_dik_change = context.individual_cognitive_state.get("seconds_since_dik_change")
            recent_meaningful_epistemic_change = seconds_since_dik_change is not None and float(seconds_since_dik_change) <= 2.0 and bool(mismatch_signals)
            if readiness.get("ready_for_build") and active_incomplete_projects and not recent_meaningful_epistemic_change:
                sorted_affordances = sorted(context.action_affordances, key=lambda a: float(a.get("utility", 0.0)), reverse=True)
                candidate = next((c for c in sorted_affordances if c.get("action_type") in {ExecutableActionType.TRANSPORT_RESOURCES.value, ExecutableActionType.START_CONSTRUCTION.value, ExecutableActionType.CONTINUE_CONSTRUCTION.value}), None)
                if candidate is not None:
                    rewritten = BrainDecision(
                        selected_action=ExecutableActionType(candidate["action_type"]),
                        target_id=candidate.get("target_id"),
                        target_zone=candidate.get("target_zone"),
                        goal_update="satisfy_build_logistics" if candidate.get("action_type") == ExecutableActionType.TRANSPORT_RESOURCES.value else "execute_build",
                        reason_summary=f"Policy pivot ({pivot_origin}): readiness unlocked productive action.",
                        confidence=max(0.8, float(decision.confidence or 0.0)),
                    )
                    self._emit_event(sim_state, "policy_pivot_applied", {"origin": pivot_origin, "kind": "post_readiness", "previous_action": decision.selected_action.value, "pivoted_to": rewritten.selected_action.value, "reason": "readiness_unlocked"})
                    if sim_state is not None:
                        rerouted = self._reroute_decision_through_rulebrain_controller(
                            context,
                            rewritten,
                            sim_state=sim_state,
                            pivot_origin=pivot_origin,
                            reason="readiness_unlocked",
                        )
                        if rerouted is not None:
                            rewritten = rerouted
        return rewritten

    def _reroute_decision_through_rulebrain_controller(self, context, fallback_decision, *, sim_state, pivot_origin, reason):
        """Route local pivots back through RuleBrain so method/step state remains authoritative."""
        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"provider": sim_state.brain_provider}
        provider = runtime.get("provider")
        if provider is None or provider.__class__.__name__ != "RuleBrain":
            return None
        control_state = context.individual_cognitive_state.setdefault("control_state", dict(self.control_state or {}))
        method_state = dict(control_state.get("method_state") or {})
        outcomes = list(method_state.get("recent_step_outcomes", []))
        outcomes.append(
            {
                "tick": int(self.sim_step_count),
                "origin": pivot_origin,
                "reason": reason,
                "fallback_selected_action": fallback_decision.selected_action.value,
            }
        )
        method_state["recent_step_outcomes"] = outcomes[-8:]
        method_state["last_method_switch_reason"] = f"pivot:{reason}"
        control_state["method_state"] = method_state
        context.individual_cognitive_state["control_state"] = control_state
        controller_decision = provider.decide(context)
        updated_control_state = context.individual_cognitive_state.get("control_state", {})
        if isinstance(updated_control_state, dict) and updated_control_state:
            self.control_state.update(updated_control_state)
            self._sync_method_state_from_control()
        self._emit_event(
            sim_state,
            "policy_pivot_rerouted_rulebrain",
            {
                "origin": pivot_origin,
                "reason": reason,
                "fallback_action": fallback_decision.selected_action.value,
                "controller_action": controller_decision.selected_action.value,
                "active_method": self.active_method_id,
                "active_step": self.active_method_step,
            },
        )
        return controller_decision

    def _attempt_local_rule_brain_refresh(self, sim_state, environment, planner_reason):
        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"provider": sim_state.brain_provider, "configured_backend": sim_state.configured_brain_backend}
        provider = runtime.get("provider")
        backend = runtime.get("configured_backend", sim_state.configured_brain_backend)
        if provider is None or (provider.__class__.__name__ != "RuleBrain" and backend != "rule_brain"):
            return False
        context = sim_state.brain_context_builder.build(sim_state, self)
        decision = provider.decide(context)
        decision = self._apply_policy_pivots(decision, environment, sim_state=sim_state, context=context, pivot_origin="local_refresh")
        updated_control_state = context.individual_cognitive_state.get("control_state", {})
        if isinstance(updated_control_state, dict) and updated_control_state:
            self.control_state.update(updated_control_state)
            self._sync_method_state_from_control()
        self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment, sim_state=sim_state)
        self._emit_event(sim_state, "local_policy_refresh_used", {"reason": planner_reason, "backend": backend, "selected_action": decision.selected_action.value, "control_mode": self.control_state.get("mode"), "mode_dwell_steps": self.control_state.get("mode_dwell_steps")})
        return True
    def _translate_brain_decision_to_legacy_action(self, decision, environment, sim_state=None):
        # NOTE: This is still a live legacy adapter in the execution path.
        # Planner outputs are normalized to action-dict records consumed by the
        # existing simulator executor. Retain this bridge until executor cleanup.
        if not self.startup_state.get("first_productive_action_started"):
            self._emit_event(sim_state, "first_action_translation_started", {"planner_action_type": decision.selected_action.value})
        self._emit_event(sim_state, "planner_next_action_selected", {"planner_action_type": decision.selected_action.value, "plan_id": getattr(self.current_plan, "plan_id", None)})
        self._emit_event(sim_state, "executable_action_attempted", {"planner_action_type": decision.selected_action.value, "plan_id": getattr(self.current_plan, "plan_id", None)})
        if self.task_model is not None:
            enabled = set(self.task_model.enabled_actions_for_role(self.role))
            if decision.selected_action.value not in enabled:
                self._emit_event(
                    sim_state,
                    "action_translation_failed",
                    {
                        "planner_action_type": decision.selected_action.value,
                        "failure_category": "illegal_action",
                        "target_id": decision.target_id,
                    },
                )
                return [{"type": "idle", "duration": 1.0, "priority": 1, "decision_action": ExecutableActionType.WAIT.value}]

        mapping = {
            ExecutableActionType.MOVE_TO_TARGET: {"type": "move_to", "duration": 1.0, "priority": 1},
            ExecutableActionType.INSPECT_INFORMATION_SOURCE: {"type": "move_to", "duration": 1.0, "priority": 1},
            ExecutableActionType.COMMUNICATE: {"type": "communicate", "duration": 0.5, "priority": 1},
            ExecutableActionType.REQUEST_ASSISTANCE: {"type": "communicate", "duration": 0.5, "priority": 1},
            ExecutableActionType.MEETING: {"type": "communicate", "duration": 0.8, "priority": 1},
            ExecutableActionType.EXTERNALIZE_PLAN: {"type": "idle", "duration": 1.0, "priority": 1},
            ExecutableActionType.CONSULT_TEAM_ARTIFACT: {"type": "idle", "duration": 1.0, "priority": 1},
            ExecutableActionType.TRANSPORT_RESOURCES: {"type": "transport_resources", "duration": 30.0, "priority": 1},
            ExecutableActionType.START_CONSTRUCTION: {"type": "construct", "duration": 2.0, "priority": 1},
            ExecutableActionType.CONTINUE_CONSTRUCTION: {"type": "construct", "duration": 2.0, "priority": 1},
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION: {"type": "construct", "duration": 2.0, "priority": 1},
            ExecutableActionType.VALIDATE_CONSTRUCTION: {"type": "idle", "duration": 1.0, "priority": 1},
            ExecutableActionType.OBSERVE_ENVIRONMENT: {"type": "idle", "duration": 0.8, "priority": 1},
            ExecutableActionType.REASSESS_PLAN: {"type": "idle", "duration": 0.8, "priority": 1},
            ExecutableActionType.WAIT: {"type": "idle", "duration": 1.0, "priority": 1},
        }
        action = dict(mapping[decision.selected_action])
        action["decision_action"] = decision.selected_action.value
        if self.task_model is not None:
            params = self.task_model.action_parameters.get(decision.selected_action.value)
            if params is not None:
                action["duration"] = params.duration_s
                action["task_parameters"] = dict(params.metadata)

        if decision.selected_action == ExecutableActionType.INSPECT_INFORMATION_SOURCE:
            handoff = dict(self.post_inspect_handoff or {})
            now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
            if handoff.get("pending") and handoff.get("expires_at", 0.0) >= now_ts:
                self.post_inspect_handoff["pending"] = False
            committed_source = self.inspect_pursuit.get("source_id")
            if committed_source and self._inspect_pursuit_active_for(committed_source, now_ts):
                source_id = committed_source
                interaction_target = self.inspect_pursuit.get("target_position")
                self._emit_event(
                    sim_state,
                    "pursuit_reused",
                    {
                        "source_id": source_id,
                        "target": interaction_target,
                        "expires_at": self.inspect_pursuit.get("expires_at"),
                    },
                )
            else:
                source_id, interaction_target = self._resolve_inspect_target(decision, environment, sim_state=sim_state)
            if source_id is None or interaction_target is None:
                if not self.startup_state.get("first_productive_action_started"):
                    self.planner_state["startup_target_resolution_failures"] = int(self.planner_state.get("startup_target_resolution_failures", 0)) + 1
                    self._emit_event(sim_state, "first_target_resolution_failed", {"target_type": "information_source", "requested_target_id": decision.target_id})
                self._emit_event(
                    sim_state,
                    "action_translation_failed",
                    {
                        "planner_action_type": decision.selected_action.value,
                        "failure_category": "unresolved_target",
                        "target_id": decision.target_id,
                    },
                )
                if isinstance(getattr(self, "post_inspect_handoff", None), dict) and self.post_inspect_handoff.get("source_id"):
                    self._emit_event(sim_state, "post_inspect_action_translation_failed", {
                        "source_id": self.post_inspect_handoff.get("source_id"),
                        "dik_changed": bool(self.post_inspect_handoff.get("dik_changed")),
                        "readiness_changed": bool(self.post_inspect_handoff.get("readiness_changed")),
                        "selected_next_action": decision.selected_action.value,
                        "failure_category": "unresolved_target",
                        "post_inspect_blocker_category": self.post_inspect_handoff.get("blocker_category"),
                    })
                self._emit_event(sim_state, "first_action_translation_failed", {"planner_action_type": decision.selected_action.value, "failure_category": "unresolved_target"})
                return [{"type": "idle", "duration": 1.0, "priority": 1}]
            action["target"] = interaction_target
            action["source_target_id"] = source_id
            current_source = self.inspect_session.get("source_id")
            current_state = self.inspect_session.get("state")
            if current_source == source_id and current_state in {"target_selected", "target_reached", "inspection_started", "inspection_completed", "dik_acquired", "post_inspect_derivation_attempted"}:
                self._emit_event(
                    sim_state,
                    "inspect_restarted_duplicate",
                    {
                        "source_id": source_id,
                        "duplicate_reason": "inspect_already_active",
                        "inspect_state": current_state,
                        "goal": self.goal,
                    },
                )
            elif current_source and current_source != source_id and current_state not in {"idle"}:
                self._release_source_slot(environment, source_id=current_source, emit=True, sim_state=sim_state, reason="retargeted_source")
                self._emit_event(
                    sim_state,
                    "inspect_reset",
                    {
                        "source_id": current_source,
                        "new_source_id": source_id,
                        "reason": "retargeted_source",
                        "prior_state": current_state,
                    },
                )
            self.current_inspect_target_id = source_id
            current_zone = environment.get_zone(self.position)
            next_zone = None
            target_meta = environment.interaction_targets.get(source_id)
            if isinstance(target_meta, dict):
                next_zone = target_meta.get("zone")
            if next_zone and current_zone != next_zone:
                self._emit_event(sim_state, "movement_between_knowledge_locations", {"from_zone": current_zone, "to_zone": next_zone, "source_id": source_id})
            if source_id == "Team_Info":
                self._emit_event(sim_state, "moving_to_shared_source", {"source_id": source_id, "from_zone": current_zone, "to_zone": next_zone})
            now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
            if current_source != source_id:
                self.inspect_session = {
                    "source_id": source_id,
                    "target": interaction_target,
                    "state": "target_selected",
                    "started_at": now_ts,
                    "last_updated_at": now_ts,
                    "restarts": 0,
                }
                self._emit_event(
                    sim_state,
                    "inspect_target_selected",
                    {"source_id": source_id, "target": interaction_target, "goal": self.goal},
                )
            if not self._inspect_pursuit_active_for(source_id, now_ts):
                self._commit_inspect_pursuit(
                    source_id,
                    interaction_target,
                    now_ts,
                    slot_id=self.source_access_state.get("slot_id"),
                    sim_state=sim_state,
                )
            if self.source_inspection_state.get(source_id) == "inspected":
                self._set_status(f"Source skipped due to completion: {source_id}")
            self._emit_startup_once(sim_state, "first_productive_action_started", "first_productive_action_started", {"planner_action_type": decision.selected_action.value, "translated_action_type": action.get("type")})
            return [action]

        if decision.target_id:
            interaction_target = environment.get_interaction_target_position(decision.target_id, from_position=self.position)
            if interaction_target is not None:
                action["target"] = interaction_target
            if decision.selected_action in {
                ExecutableActionType.START_CONSTRUCTION,
                ExecutableActionType.CONTINUE_CONSTRUCTION,
                ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
                ExecutableActionType.VALIDATE_CONSTRUCTION,
            }:
                action["project_id"] = decision.target_id
        if decision.selected_action == ExecutableActionType.TRANSPORT_RESOURCES:
            action["duration"] = self._scaled_duration(action.get("duration", 30.0)) * self._duration_scale("transport_resources")
            build_selection = self._select_build_target(environment, require_readiness=False, include_project=True)
            if isinstance(build_selection, dict):
                action["project_id"] = build_selection.get("project_id")
                action["target"] = build_selection.get("target")

        if decision.selected_action in {
            ExecutableActionType.START_CONSTRUCTION,
            ExecutableActionType.CONTINUE_CONSTRUCTION,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
            ExecutableActionType.VALIDATE_CONSTRUCTION,
        }:
            action["duration"] = self._scaled_duration(action["duration"]) * self._duration_scale(decision.selected_action.value)

        if decision.selected_action in {
            ExecutableActionType.START_CONSTRUCTION,
            ExecutableActionType.CONTINUE_CONSTRUCTION,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
            ExecutableActionType.VALIDATE_CONSTRUCTION,
        } and action.get("project_id") not in environment.construction.projects:
            selected = self._select_build_target(environment, require_readiness=False, include_project=True)
            if isinstance(selected, dict):
                action["project_id"] = selected.get("project_id")
                if not action.get("target"):
                    action["target"] = selected.get("target")

        if decision.selected_action in {
            ExecutableActionType.START_CONSTRUCTION,
            ExecutableActionType.CONTINUE_CONSTRUCTION,
            ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
            ExecutableActionType.VALIDATE_CONSTRUCTION,
        }:
            blockers, project_id = self._construction_action_blockers(decision, action, environment, sim_state=sim_state)
            if project_id:
                action["project_id"] = project_id
            if blockers:
                self._emit_event(sim_state, "execution_readiness_failed", {"planner_action_type": decision.selected_action.value, "failure_category": "readiness_not_unlocked", "blockers": blockers, "project_id": project_id})
                return [{"type": "idle", "duration": 1.0, "priority": 1, "decision_action": ExecutableActionType.WAIT.value}]
            else:
                payload = {"planner_action_type": decision.selected_action.value, "project_id": project_id}
                goal_id = self._canonical_readiness_goal_id()
                if goal_id:
                    payload["goal_id"] = goal_id
                self._emit_event(sim_state, "execution_readiness_passed", payload)

        if decision.selected_action in {ExecutableActionType.EXTERNALIZE_PLAN, ExecutableActionType.CONSULT_TEAM_ARTIFACT}:
            action["artifact_action"] = decision.selected_action.value
            self._emit_event(sim_state, "moving_to_externalization_site", {"selected_next_action": decision.selected_action.value, "current_location": self.position})
        if decision.selected_action == ExecutableActionType.REQUEST_ASSISTANCE:
            action["assist_action"] = decision.selected_action.value

        self._emit_event(sim_state, "action_translation_succeeded", {"planner_action_type": decision.selected_action.value, "translated_action_type": action.get("type"), "target_id": decision.target_id, "target_zone": decision.target_zone})
        if decision.selected_action != ExecutableActionType.INSPECT_INFORMATION_SOURCE and isinstance(getattr(self, "post_inspect_handoff", None), dict):
            self.post_inspect_handoff["pending"] = False
        if (
            decision.selected_action != ExecutableActionType.INSPECT_INFORMATION_SOURCE
            and self.inspect_pursuit.get("source_id")
        ):
            self._clear_inspect_pursuit(
                reason="non_inspect_action_selected",
                sim_state=sim_state,
                release_slot=True,
                environment=environment,
            )
        if action.get("type") in {"move_to", "communicate", "construct", "transport_resources"}:
            self._emit_startup_once(sim_state, "first_productive_action_started", "first_productive_action_started", {"planner_action_type": decision.selected_action.value, "translated_action_type": action.get("type")})
        return [action]

    def perceive_environment(self, sim_state):
        self_visible_range = 2.0
        for agent in sim_state.agents:
            if agent.name == self.name:
                continue
            dist = math.hypot(agent.position[0] - self.position[0], agent.position[1] - self.position[1])
            if dist <= self_visible_range:
                inferred_goal = "moving" if agent.target else "idle"
                known_knowledge_ids = {k for k in agent.mental_model["knowledge"].rules}
                self.theory_of_mind[agent.name] = {
                    "goals": [inferred_goal],
                    "knowledge_ids": known_knowledge_ids,
                    "last_seen": sim_state.time
                }

    def update_internal_state(self):
        for name, model in self.theory_of_mind.items():
            if not model["knowledge_ids"]:
                self.activity_log.append(f"ToM: {name} may need assistance")

        # Detect conflicts between rules and construction
        known_rules = self.mental_model["knowledge"].rules
        if known_rules:
            for project in getattr(self, "observed_projects", []):
                if not any(rule in project.get("expected_rules", []) for rule in known_rules):
                    self.activity_log.append("Mismatch with construction: reevaluating knowledge")
                    self.reevaluate_knowledge()

    def evaluate_goals(self):
        """Deprecated: retained for compatibility; delegates to authoritative evaluator."""
        self._evaluate_goal_state(environment=None)

    def _evaluate_goal_state(self, environment):
        """Authoritative goal-state evaluator used by the live simulation update path."""
        if environment is None:
            return

        self._update_goal_states_from_runtime(sim_state=None, environment=environment)
        goal_entry = self.current_goal()
        if not goal_entry:
            self.push_goal("seek_info", self._choose_info_target(environment))
            return

        goal = goal_entry["goal"]
        self.target = goal_entry["target"]

        if goal == "seek_info":
            if self.mental_model["information"]:
                self.pop_goal()
                self.push_goal("share")
            else:
                self.activity_log.append("Still seeking info...")

        elif goal == "share":
            if not self.has_shared:
                self.activity_log.append("Preparing to share information with teammates")
                return

            self.pop_goal()
            if self._is_build_eligible(environment):
                build_selection = self._select_build_target(environment, include_project=True)
                if build_selection is not None:
                    self.push_goal("build", build_selection)
                else:
                    self.activity_log.append("No accessible build interaction target found; gathering more info")
                    self.push_goal("seek_info", self._choose_info_target(environment))
            else:
                self.activity_log.append(
                    f"Build deferred (readiness={self._build_readiness_score()}, blockers={self._build_readiness_blockers(environment)}); continuing information gathering"
                )
                self.push_goal("seek_info", self._choose_info_target(environment))

        elif goal == "build":
            if self.target is None:
                selected = self._select_build_target(environment, include_project=True)
                self.target = selected["target"] if isinstance(selected, dict) else selected

            if self.mental_model["knowledge"].rules:
                self.activity_log.append("Building task engaged")
            elif not self._is_build_eligible(environment):
                self.pop_goal()
                self.push_goal("seek_info", self._choose_info_target(environment))

        elif goal == "idle":
            self.activity_log.append("Idling...")

    def _plan_actions_for_current_goal(self):
        """Authoritative action planner from current goal state."""
        if not self.goal_stack:
            return [{"type": "idle", "duration": 1.0, "priority": 0}]

        top_goal = self.current_goal()
        goal = top_goal["goal"]

        if goal == "seek_info":
            target = top_goal.get("target") or self.target or (7.0, 6.4)
            return [{"type": "move_to", "target": target, "duration": 1.0, "priority": 1}]
        if goal == "share":
            return [{"type": "communicate", "duration": 0.5, "priority": 1}]
        if goal == "build":
            goal_target = top_goal.get("target")
            project_id = goal_target.get("project_id") if isinstance(goal_target, dict) else None
            target_point = goal_target.get("target") if isinstance(goal_target, dict) else goal_target
            return [{"type": "construct", "duration": 2.0, "priority": 1, "project_id": project_id, "target": target_point}]

        return [{"type": "idle", "duration": 1.0, "priority": 0}]

    def _run_goal_management_pipeline(self, dt, environment):
        """Single authoritative goal-management pipeline for agent behavior."""
        self.update_internal_state()
        self._evaluate_goal_state(environment)
        self.current_action = self._plan_actions_for_current_goal()
        self._advance_active_actions(dt, sim_state=None)

    def select_action(self):
        """Deprecated: retained for compatibility; delegates to action planner."""
        actions = self._plan_actions_for_current_goal()
        self.current_action = actions
        return actions

    def perform_action(self, actions):
        if not isinstance(actions, list):
            actions = [actions]

        actions = sorted(actions, key=lambda x: x.get("priority", 1), reverse=True)

        for action in actions:
            if action["type"] == "move_to":
                self.target = action["target"]
                if action.get("source_target_id"):
                    self.current_inspect_target_id = action["source_target_id"]
            elif action["type"] == "communicate":
                self.has_shared = True
                self.activity_log.append("Shared with teammates")
            elif action["type"] == "construct":
                self.activity_log.append("Building...")
            elif action["type"] == "transport_resources":
                if action.get("project_id"):
                    self.activity_log.append(f"Transporting resources to {action.get('project_id')}...")
                else:
                    self.inventory_resources["bricks"] = self.inventory_resources.get("bricks", 0) + 1
                    self.activity_log.append("Transporting resources... (+1 bricks inventory)")
            elif action["type"] == "idle":
                self.activity_log.append("Idling...")

    def absorb_packet(self, packet, accuracy=1.0, sim_state=None, source_id=None):
        source_ref = source_id or packet.get("source_id") or "unknown_source"
        accuracy_modifier = max(-0.2, min(0.2, (float(accuracy) - 1.0) * 0.35))
        for d in packet.get("data", []):
            success, _ = self._attempt_epistemic_transition(
                hook_target="absorb_packet",
                sim_state=sim_state,
                event_payload={
                    "source_id": source_ref,
                    "element_id": d.id,
                    "element_type": "data",
                    "requested_accuracy": float(accuracy),
                },
                attempt_event="packet_absorption_attempted",
                failed_event="packet_absorption_failed",
                context_modifier=accuracy_modifier,
                retry_bonus=0.0,
            )
            if success:
                if d not in self.mental_model["data"]:
                    self.mental_model["data"].add(d)
                    d.acquired_by[self.name] = {"mode": "direct", "from": d.source}
                    DIK_LOG.append({
                        "time": getattr(self, "current_time", 0.0),
                        "agent": self.name,
                        "type": "Data",
                        "id": d.id,
                        "mode": "direct",
                        "from": d.source
                    })
                    self.activity_log.append(f"Absorbed data: {d.id} (from {d.source})")
                    self.last_dik_change_time = getattr(self, "current_time", 0.0)

        for info in packet.get("information", []):
            success, _ = self._attempt_epistemic_transition(
                hook_target="absorb_packet",
                sim_state=sim_state,
                event_payload={
                    "source_id": source_ref,
                    "element_id": info.id,
                    "element_type": "information",
                    "requested_accuracy": float(accuracy),
                },
                attempt_event="packet_absorption_attempted",
                failed_event="packet_absorption_failed",
                context_modifier=accuracy_modifier,
                retry_bonus=0.0,
            )
            if success:
                if info not in self.mental_model["information"]:
                    self.mental_model["information"].add(info)
                    info.acquired_by[self.name] = {"mode": "direct", "from": info.source}
                    DIK_LOG.append({
                        "time": getattr(self, "current_time", 0.0),
                        "agent": self.name,
                        "type": "Information",
                        "id": info.id,
                        "mode": "direct",
                        "from": info.source
                    })
                    self.activity_log.append(f"Absorbed info: {info.id} (from {info.source})")
                    self.last_dik_change_time = getattr(self, "current_time", 0.0)

    def move_toward(self, target, dt, environment, sim_state=None):
        nav = self.navigation
        path_mode = nav.get("path_mode", "grid_astar")
        origin = tuple(self.position)

        def _emit(stage, extra=None):
            payload = {
                "stage": stage,
                "path_mode": path_mode,
                "current_location": self.position,
                "target_location": target,
                "retry_count": int(nav.get("retry_count", 0)),
                "ignore_agent_collision": bool(nav.get("ignore_agent_collision", True)),
            }
            if extra:
                payload.update(extra)
            self._emit_event(sim_state, stage, payload)

        def is_blocking_object(obj):
            obj_type = obj.get("type")
            return obj_type in {"rect", "circle", "blocked"} and not obj.get("passable", False)

        def can_occupy(point):
            blocked_by_env = any(
                environment.is_near_object(point, name, threshold=0.15)
                for name, obj in environment.objects.items()
                if is_blocking_object(obj)
            )
            if blocked_by_env:
                return False, "blocked_zone"
            if not nav.get("ignore_agent_collision", True):
                near_source_slot = False
                if self.current_inspect_target_id and hasattr(environment, "is_source_slot_context"):
                    near_source_slot = environment.is_source_slot_context(point, self.current_inspect_target_id)
                for other in getattr(environment, "agents", []):
                    if other is self:
                        continue
                    threshold = 0.1 if near_source_slot else 0.15
                    if math.hypot(point[0] - other.position[0], point[1] - other.position[1]) < threshold:
                        return False, "agent_collision_block"
            return True, None

        # Determine path freshness.
        needs_new_path = (
            not nav.get("active_path")
            or nav.get("path_target") != tuple(target)
            or nav.get("path_index", 0) >= len(nav.get("active_path", []))
        )

        if math.hypot(target[0] - self.position[0], target[1] - self.position[1]) < 0.01:
            if nav.get("last_target") is not None and tuple(target) != tuple(nav.get("last_target")):
                nav["last_blocker_category"] = "zero_distance_retarget"
                _emit("movement_blocked", {"destination": target, "blocker_category": "zero_distance_retarget"})
                self._emit_event(sim_state, "movement_failed", {"failure_category": "zero_distance_retarget", "path_mode": path_mode})
            else:
                _emit("movement_arrived", {"distance": 0.0})
            nav["last_target"] = tuple(target)
            return

        if needs_new_path:
            if nav.get("active_path") and nav.get("path_target") is not None and tuple(target) != tuple(nav.get("path_target")):
                self._emit_event(sim_state, "movement_abandoned", {"path_mode": path_mode, "previous_target": nav.get("path_target"), "new_target": target, "reason": "retargeted"})
            _emit("path_planning_started", {"origin": self.position, "destination": target})
            plan = environment.plan_path(self.position, target, mode=path_mode)
            if plan.get("status") != "ok" or not plan.get("waypoints"):
                blocker = plan.get("blocker_category", "unknown")
                nav["retry_count"] = int(nav.get("retry_count", 0)) + 1
                if nav["retry_count"] > 1:
                    blocker = "repeated_move_retry"
                nav["last_blocker_category"] = blocker
                _emit("path_planning_failed", {"blocker_category": blocker})
                _emit("movement_blocked", {"blocker_category": blocker})
                self._emit_event(sim_state, "movement_failed", {"failure_category": blocker, "path_mode": path_mode})
                self.activity_log.append(f"Blocked while moving toward {target} ({blocker})")
                return

            nav["active_path"] = list(plan.get("waypoints") or [])
            nav["path_target"] = tuple(target)
            nav["path_index"] = 0
            nav["retry_count"] = 0
            _emit(
                "path_planning_succeeded",
                {
                    "path_length": len(nav["active_path"]),
                    "waypoint_count": len(nav["active_path"]),
                    "from_cache": bool(plan.get("from_cache")),
                },
            )
            if plan.get("from_cache"):
                _emit("path_cached_reused", {"path_length": len(nav["active_path"]), "waypoint_count": len(nav["active_path"])})

        waypoint = nav["active_path"][nav["path_index"]]
        dx, dy = waypoint[0] - self.position[0], waypoint[1] - self.position[1]
        dist = math.hypot(dx, dy)
        while dist < 0.01:
            nav["path_index"] += 1
            if nav["path_index"] >= len(nav["active_path"]):
                nav["last_arrival_position"] = tuple(self.position)
                _emit("movement_arrived", {"destination": target, "path_mode": path_mode, "path_length": len(nav["active_path"])})
                return
            waypoint = nav["active_path"][nav["path_index"]]
            dx, dy = waypoint[0] - self.position[0], waypoint[1] - self.position[1]
            dist = math.hypot(dx, dy)

        _emit("movement_started", {"origin": origin, "destination": target, "path_length": len(nav.get("active_path", [])), "waypoint_index": nav.get("path_index", 0)})
        if not self.startup_state.get("first_movement_started"):
            self._emit_startup_once(sim_state, "first_movement_started", "first_movement_started", {"destination": target, "distance": round(dist, 3)})

        angle = math.atan2(dy, dx)
        self.orientation = angle
        step = min(self.speed * dt, dist)
        new_x = self.position[0] + math.cos(angle) * step
        new_y = self.position[1] + math.sin(angle) * step
        can_move, blocker = can_occupy((new_x, new_y))

        if can_move:
            self.position = (new_x, new_y)
            nav["last_position"] = tuple(self.position)
            nav["last_target"] = tuple(target)
            _emit("movement_progressed", {"remaining_distance": round(max(0.0, dist - step), 3), "waypoint_index": nav.get("path_index", 0)})
            self._emit_event(sim_state, "movement_progress", {"destination": target, "remaining_distance": round(max(0.0, dist - step), 3), "path_mode": path_mode})
        else:
            nav["retry_count"] = int(nav.get("retry_count", 0)) + 1
            block_cat = blocker or "unknown"
            if nav["retry_count"] > 1:
                block_cat = "repeated_move_retry"
                self._emit_event(sim_state, "movement_retried", {"destination": target, "retry_count": nav["retry_count"], "blocker_category": block_cat, "path_mode": path_mode})
            nav["last_blocker_category"] = block_cat
            nav["active_path"] = []
            nav["path_index"] = 0
            _emit("movement_blocked", {"destination": target, "blocker_category": block_cat, "waypoint": waypoint})
            self._emit_event(sim_state, "movement_failed", {"failure_category": block_cat, "path_mode": path_mode})
            if not self.startup_state.get("left_spawn"):
                self.planner_state["startup_movement_blockers"] = int(self.planner_state.get("startup_movement_blockers", 0)) + 1
                self._emit_event(sim_state, "first_movement_blocked", {"destination": target, "blocker_category": block_cat})
            if self.current_inspect_target_id:
                self.inspect_stall_counts[self.current_inspect_target_id] = self.inspect_stall_counts.get(self.current_inspect_target_id, 0) + 1

        self.heart_rate += 1

    def update_physiology(self, exertion=0.0, speaking=False):
        self.heart_rate += exertion * 2
        if speaking:
            self.heart_rate += 1
            self.gsr += 0.01
            self.co2_output += 0.02
        else:
            self.gsr *= 0.95
        self.temperature += 0.005 * exertion
        self.heart_rate = max(60, min(self.heart_rate, 160))
        self.temperature = max(96.0, min(self.temperature, 101.0))
        self.co2_output = 0.04 + 0.01 * abs(self.heart_rate - 70)

    def update_knowledge(self, environment, full_packet_sweep=True, sim_state=None):
        self._ensure_source_state(environment)
        if full_packet_sweep:
            for packet_name, packet_content in environment.knowledge_packets.items():
                if packet_name in self.mental_model["information"]:
                    continue
                if not self._has_packet_access(packet_name):
                    continue
                if self.role not in packet_name and "Team" not in packet_name:
                    continue
                if environment.can_access_info(self.position, packet_name, role=self.role):
                    before = len(self.mental_model["information"])
                    self.absorb_packet(packet_content, accuracy=0.95, sim_state=sim_state, source_id=packet_name)
                    after = len(self.mental_model["information"])
                    if after > before:
                        self.source_inspection_state[packet_name] = "inspected"
                        self._set_status(f"Legacy sweep ingested packet from {packet_name}")
                # Deliberately suppress per-tick access-failure spam from legacy sweep.


        if "mismatch with construction" in " ".join(self.activity_log[-6:]).lower() and self.current_inspect_target_id:
            self.mark_source_revisitable(self.current_inspect_target_id, reason="construction_mismatch")

        self._apply_task_derivations(sim_state=sim_state)

        known_info = list(self.mental_model["information"])
        if known_info:
            from itertools import combinations
            candidate_tags = {}
            for info in known_info:
                for tag in info.tags:
                    candidate_tags.setdefault(tag, []).append(info)
            for tag, group in candidate_tags.items():
                if len(group) >= 2:
                    infer_base = max(self._hook_value("dik_update", "transform_information_to_knowledge", "success_probability", default=0.5), self._trait_value("rule_accuracy"))
                    infer_prob = 0.35 + 0.6 * infer_base
                    if random.random() < infer_prob:
                        self.mental_model["knowledge"].try_infer_rules(group, agent_name=self.name)
                        self.last_dik_change_time = getattr(self, "current_time", 0.0)
                        self.activity_log.append(f"Inferred rule from tag [{tag}] (p={infer_prob:.2f})")

    def decide_next_action(self, environment):
        """Deprecated compatibility wrapper for legacy callers."""
        self._evaluate_goal_state(environment)
        self.current_action = self._plan_actions_for_current_goal()

    def update(self, dt, environment, sim_state=None, planner_lifecycle_already_polled=False):
        self.update_physiology(exertion=0.5)
        self.update_knowledge(environment, full_packet_sweep=(sim_state is None), sim_state=sim_state)
        if sim_state is None:
            # Legacy compatibility path for unit-level agent calls outside full simulation state.
            self._run_goal_management_pipeline(dt, environment)
        else:
            self.perceive_environment(sim_state)
            self.sim_step_count += 1
            if not planner_lifecycle_already_polled:
                self._check_inflight_timeout(sim_state)
                self._poll_planner_request(sim_state, environment)
                self._poll_dik_integration_request(sim_state)
            if self.active_actions:
                self._advance_active_actions(dt, sim_state=sim_state)
            else:
                trigger_reason = self._plan_trigger_reason(sim_state, environment)
                dik_trigger_reason = self._dik_integration_trigger_reason(sim_state, trigger_reason)
                dik_allowed, dik_reason = self._dik_integration_allowed(sim_state, dik_trigger_reason)
                if dik_allowed and self.dik_integration_state.get("status") != "in_flight":
                    self._emit_event(sim_state, "dik_integration_invocation_requested", {"tick": self.sim_step_count, "trigger_reason": dik_reason})
                    self._submit_dik_integration_request_async(sim_state, dik_reason)
                planner_allowed, planner_reason = self._planner_decision_allowed(sim_state, trigger_reason)
                if planner_allowed:
                    cooldown_remaining = self._planner_cooldown_remaining(sim_state)
                    if cooldown_remaining > 0.0:
                        self.planner_state["total_skipped_cooldown"] += 1
                        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"config": sim_state.brain_backend_config, "configured_backend": sim_state.configured_brain_backend}
                        self._emit_event(sim_state, "planner_request_skipped_cooldown", {"reason": planner_reason, "cooldown_remaining": cooldown_remaining, "consecutive_failures": self.planner_state["consecutive_failures"], "backend": runtime.get("configured_backend", sim_state.configured_brain_backend), "model": runtime["config"].local_model})
                    elif self.planner_state.get("status") == "in_flight":
                        self.planner_state["total_skipped_inflight"] += 1
                        self.planner_state["consecutive_inflight_skips"] = int(self.planner_state.get("consecutive_inflight_skips", 0)) + 1
                        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"configured_backend": sim_state.configured_brain_backend, "effective_backend": sim_state.effective_brain_backend, "provider": sim_state.brain_provider}
                        self._emit_event(sim_state, "planner_request_skipped_inflight", {"reason": planner_reason, "request_id": self.planner_state.get("request_id"), "backend": runtime.get("configured_backend", sim_state.configured_brain_backend)})
                        stall_threshold = max(3, int(self.planner_cadence.planner_interval_steps) * 3)
                        early_window_steps = max(4, int(self.planner_cadence.planner_interval_steps) * 12)
                        if (
                            self.sim_step_count <= early_window_steps
                            and int(self.planner_state.get("consecutive_inflight_skips", 0)) >= stall_threshold
                            and int(self.planner_state.get("total_completed", 0)) == 0
                        ):
                            demoted = self._maybe_hard_demote_backend(
                                sim_state,
                                reason="early_inflight_stall",
                                activate_bootstrap=True,
                            )
                            if demoted:
                                self.clear_planner_inflight_state(sim_state=sim_state, reason="early_inflight_stall")
                            else:
                                self.clear_planner_inflight_state(sim_state=sim_state, reason="early_inflight_stall_recover")
                                self.activate_fallback_bootstrap(sim_state=sim_state, reason="early_inflight_stall")
                        if self.current_plan is None and not self.active_actions and not self.current_action:
                            if self.planner_cadence.unrestricted_local_qwen_mode:
                                self._emit_event(sim_state, "planner_waiting_on_inflight_unrestricted", {"reason": "inflight_without_plan", "request_state": self.planner_state.get("status"), "backend": runtime.get("configured_backend", sim_state.configured_brain_backend)})
                            else:
                                startup_decision = BrainDecision(selected_action=ExecutableActionType.INSPECT_INFORMATION_SOURCE, reason_summary="startup-safe action while planner request is in flight", confidence=0.4)
                                self.current_action = self._translate_brain_decision_to_legacy_action(startup_decision, environment, sim_state=sim_state)
                                self._emit_event(sim_state, "ui_safe_fallback_used", {"reason": "inflight_without_plan", "request_state": self.planner_state.get("status"), "backend": runtime.get("configured_backend", sim_state.configured_brain_backend)})
                    else:
                        pending_trace_id = self._make_planner_trace_id(f"{self.agent_id}-{self.sim_step_count}")
                        self._emit_event(sim_state, "planner_invocation_started", {"trigger_reason": planner_reason, "tick": self.sim_step_count, "current_plan_id": getattr(self.current_plan, "plan_id", None), "trace_id": pending_trace_id})
                        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"configured_backend": sim_state.configured_brain_backend, "effective_backend": sim_state.effective_brain_backend, "provider": sim_state.brain_provider}
                        self._emit_event(sim_state, "planner_invocation_requested", {"tick": self.sim_step_count, "trigger_reason": planner_reason, "configured_backend": runtime.get("configured_backend", sim_state.configured_brain_backend), "effective_backend": runtime.get("effective_backend", sim_state.effective_brain_backend), "request_explanation": self._should_request_explanation(), "current_plan_id": getattr(self.current_plan, "plan_id", None), "current_active_goal_ids": [g.get("goal_id") for g in self.goal_stack[:6]], "trace_id": pending_trace_id})
                        self._emit_event(sim_state, "brain_provider_request_started", {"configured_backend": runtime.get("configured_backend", sim_state.configured_brain_backend), "effective_backend": runtime.get("effective_backend", sim_state.effective_brain_backend), "provider_class": runtime["provider"].__class__.__name__, "trace_id": pending_trace_id})
                        self._submit_planner_request_async(sim_state, planner_reason)
                elif self._continue_cached_plan(sim_state, environment):
                    self.planner_state["stale_plan_reuse_count"] += 1
                    sim_state.logger.log_event(sim_state.time, "planner_skipped_due_to_cadence", {"agent": self.name, "reason": planner_reason})
                elif self._attempt_local_rule_brain_refresh(sim_state, environment, planner_reason):
                    sim_state.logger.log_event(sim_state.time, "planner_skipped_local_policy_refresh", {"agent": self.name, "reason": planner_reason})
                else:
                    decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="no active cached plan while planner cadence skips", confidence=1.0)
                    self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment)
                    self.planner_state["ui_safe_fallback_count"] += 1
                    runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"configured_backend": sim_state.configured_brain_backend}
                    self._emit_event(sim_state, "ui_safe_fallback_used", {"reason": planner_reason, "request_state": self.planner_state.get("status"), "backend": runtime.get("configured_backend", sim_state.configured_brain_backend)})
                    sim_state.logger.log_event(sim_state.time, "planner_skipped_without_plan", {"agent": self.name, "reason": planner_reason})
                bootstrap_decision = self._bootstrap_override_decision(environment, sim_state=sim_state)
                if bootstrap_decision is not None:
                    self.current_action = self._translate_brain_decision_to_legacy_action(bootstrap_decision, environment, sim_state=sim_state)
                    self._emit_event(
                        sim_state,
                        "fallback_bootstrap_action_forced",
                        {
                            "action_type": bootstrap_decision.selected_action.value,
                            "target_id": bootstrap_decision.target_id,
                            "activation_reason": self.fallback_bootstrap.get("activation_reason"),
                        },
                    )
                self._advance_active_actions(dt, sim_state=sim_state)

            self._apply_externalization_and_construction_effects(environment, sim_state, dt)
            self._update_goal_states_from_runtime(sim_state, environment)
            if not self.startup_state.get("initial_goal_selected"):
                current = self.current_goal()
                if current:
                    self._emit_startup_once(sim_state, "initial_goal_selected", "initial_goal_selected", {"goal": current.get("goal"), "goal_id": current.get("goal_id")})

        if self.target:
            self.move_toward(self.target, dt, environment, sim_state=sim_state)

        if self.current_inspect_target_id:
            self._inspect_source(environment, self.current_inspect_target_id, sim_state=sim_state)

        if sim_state is not None and not self.startup_state.get("left_spawn"):
            moved = math.hypot(self.position[0] - self.spawn_position[0], self.position[1] - self.spawn_position[1])
            if moved > 0.2:
                self._emit_startup_once(sim_state, "left_spawn", "agent_left_spawn", {"distance": round(moved, 3)})


        # Communication attempt (talk while walking or standing still)
        for agent in environment.agents:
            if agent.name == self.name:
                continue
            dist = math.hypot(agent.position[0] - self.position[0], agent.position[1] - self.position[1])
            if dist <= COMMUNICATION_RADIUS:
                if any(a["type"] == "communicate" for a in self.active_actions) or \
                   any(a["type"] == "communicate" for a in agent.active_actions):
                    self.communicate_with(agent, sim_state=sim_state)

    def _apply_externalization_and_construction_effects(self, environment, sim_state, dt):
        if sim_state is None:
            return

        def _build_access(project_id):
            return environment.get_interaction_access(self.position, project_id, role=self.role)

        def _build_target(project_id):
            return environment.get_interaction_target_position(project_id, from_position=self.position)

        def _pickup_candidates():
            construction = getattr(environment, "construction", None)
            if construction is None:
                return []
            nodes = []
            for pile in construction.resource_nodes.values():
                if int(getattr(pile, "quantity", 0) or 0) <= 0:
                    continue
                nodes.append((pile.pile_id, tuple(pile.position)))
            return nodes

        def _log_mutation_blocked(event_type, payload):
            payload = dict(payload or {})
            payload.setdefault("agent", self.name)
            payload.setdefault("agent_position", (round(self.position[0], 3), round(self.position[1], 3)))
            payload.setdefault("current_action_type", event_type)
            payload.setdefault("transport_state", dict(self.transport_state))
            self._emit_event(sim_state, event_type, payload)

        for action in self.active_actions:
            if action["type"] == "idle" and action.get("artifact_action") == ExecutableActionType.EXTERNALIZE_PLAN.value and action["progress"] == 0:
                artifact_id = f"whiteboard:{self.name}:{int(sim_state.time*10)}"
                rules = list(self.mental_model["knowledge"].rules)[-3:]
                fidelity = max(self._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5), self._trait_value("rule_accuracy"))
                if rules and fidelity < 0.5:
                    rules = rules[:-1]
                sim_state.team_knowledge_manager.externalize_artifact(
                    artifact_id=artifact_id,
                    artifact_type="whiteboard_plan",
                    summary=f"Plan externalized by {self.name}",
                    content={"rules": rules, "goal": self.goal},
                    author=self.name,
                    sim_time=sim_state.time,
                    contributors=[self.name],
                    knowledge_summary=rules,
                    validation_state="validated" if fidelity >= 0.7 else "tentative",
                )
                sim_state.logger.log_event(sim_state.time, "externalization_created", {"agent": self.name, "artifact_id": artifact_id, "type": "whiteboard_plan"})

            if action["type"] == "idle" and action.get("artifact_action") == ExecutableActionType.CONSULT_TEAM_ARTIFACT.value and action["progress"] == 0:
                artifacts = sim_state.team_knowledge_manager.artifacts
                if artifacts:
                    preferred = sorted(
                        artifacts.values(),
                        key=lambda a: ((a.validation_state == "validated") and self._trait_value("goal_alignment"), a.uptake_count),
                        reverse=True,
                    )[0]
                    adopt_prob = self._hook_value("artifact_use", "adopt_externalized_knowledge", "adoption_weight", default=0.5)
                    if random.random() <= adopt_prob:
                        sim_state.team_knowledge_manager.adopt_artifact(preferred.artifact_id, self.name, sim_state.time)
                    self.activity_log.append(f"Consulted shared artifact {preferred.artifact_id}")
                    sim_state.logger.log_event(sim_state.time, "artifact_consulted", {"agent": self.name, "artifact_id": preferred.artifact_id})

            if action["type"] == "communicate" and action.get("assist_action") == ExecutableActionType.REQUEST_ASSISTANCE.value and action["progress"] == 0:
                sim_state.logger.log_event(sim_state.time, "assistance_requested", {"agent": self.name})

            if action["type"] == "idle" and action.get("decision_action") == ExecutableActionType.VALIDATE_CONSTRUCTION.value and action["progress"] == 0:
                project_id = action.get("project_id")
                if not project_id:
                    _log_mutation_blocked(
                        "construction_validation_blocked",
                        {"failure_category": "missing_project_binding", "decision_action": action.get("decision_action")},
                    )
                    continue
                project = environment.construction.projects.get(project_id)
                if project:
                    access = _build_access(project_id)
                    if not access.get("accessible"):
                        _log_mutation_blocked(
                            "construction_validation_blocked",
                            {
                                "project_id": project_id,
                                "failure_category": "not_at_validation_location",
                                "access_reason": access.get("reason"),
                                "expected_interaction_location": _build_target(project_id),
                            },
                        )
                        continue
                    if project.get("status") not in {"ready_for_validation", "needs_repair"}:
                        _log_mutation_blocked(
                            "construction_validation_blocked",
                            {
                                "project_id": project_id,
                                "failure_category": "illegal_project_status",
                                "project_status": project.get("status"),
                            },
                        )
                        continue
                    has_required_rules, missing_rules = self._construction_rule_match(project_id, environment=environment, sim_state=sim_state, include_team=True)
                    is_valid = bool(project.get("correct", True)) and has_required_rules and bool(project.get("resource_complete", False))
                    if has_required_rules:
                        environment.construction.mark_validated(project_id, is_valid=is_valid)
                    else:
                        environment.construction.mark_validated(project_id, is_valid=False)
                    sim_state.team_knowledge_manager.upsert_construction_artifact(project, sim_state.time)
                    self._emit_event(
                        sim_state,
                        "construction_validated_correct" if is_valid else "construction_validated_incorrect",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "structure_type": project.get("type", "unknown"),
                            "decision_action": action.get("decision_action"),
                            "missing_expected_rules": missing_rules,
                        },
                    )
                    if is_valid:
                        self._emit_event(
                            sim_state,
                            "construction_completed",
                            {
                                "agent": self.name,
                                "project_id": project_id,
                                "structure_type": project.get("type", "unknown"),
                                "completion_mode": "validated",
                            },
                        )

            if action["type"] == "construct" and action["progress"] == 0:
                project_id = action.get("project_id")
                if not project_id:
                    _log_mutation_blocked(
                        "construction_progress_blocked",
                        {"failure_category": "missing_project_binding", "decision_action": action.get("decision_action")},
                    )
                    continue
                project = environment.construction.projects.get(project_id)
                if project:
                    decision_action = action.get("decision_action")
                    status_before = project.get("status", "in_progress")
                    access = _build_access(project_id)
                    if not access.get("accessible"):
                        _log_mutation_blocked(
                            "construction_progress_blocked",
                            {
                                "project_id": project_id,
                                "failure_category": "not_at_build_location",
                                "access_reason": access.get("reason"),
                                "decision_action": decision_action,
                                "expected_interaction_location": _build_target(project_id),
                            },
                        )
                        continue
                    if project_id not in self._construction_attempted_projects:
                        self._construction_attempted_projects.add(project_id)
                        self._emit_event(
                            sim_state,
                            "construction_attempt_started",
                            {
                                "agent": self.name,
                                "project_id": project_id,
                                "structure_type": project.get("type", "unknown"),
                                "decision_action": decision_action,
                            },
                        )
                    self._emit_event(
                        sim_state,
                        "construction_build_episode",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "decision_action": decision_action,
                            "status_before": status_before,
                        },
                    )
                    environment.construction.assign_builder(project_id, self.name)

                    fidelity = max(self._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5), self._trait_value("rule_accuracy"))
                    if random.random() > fidelity:
                        project["correct"] = False

                    if decision_action == ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value:
                        if project.get("status") not in {"needs_repair", "ready_for_validation", "in_progress"}:
                            _log_mutation_blocked(
                                "construction_repair_blocked",
                                {
                                    "project_id": project_id,
                                    "failure_category": "illegal_project_status",
                                    "project_status": project.get("status"),
                                },
                            )
                            continue
                        project["correct"] = True
                        if project.get("resource_complete", False):
                            environment.construction.mark_validated(project_id, is_valid=True)
                    sim_state.team_knowledge_manager.upsert_construction_artifact(project, sim_state.time)
                    self._emit_event(
                        sim_state,
                        "construction_externalization_updated",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "correct": project.get("correct", True),
                            "structure_type": project.get("type", "unknown"),
                            "status": project.get("status", "in_progress"),
                            "decision_action": decision_action,
                            "progress": round((int(project.get("delivered_resources", {}).get("bricks", 0) or 0) / max(1, int(project.get("required_resources", {}).get("bricks", 0) or 0))), 4),
                        },
                    )
                    sim_state.logger.log_event(
                        sim_state.time,
                        "construction_externalization_update",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "correct": project.get("correct", True),
                            "structure_type": project.get("type", "unknown"),
                            "status": project.get("status", "in_progress"),
                            "decision_action": decision_action,
                            "inventory_bricks_remaining": self.inventory_resources.get("bricks", 0),
                        },
                    )
                    if project.get("status") == "complete":
                        self._emit_event(
                            sim_state,
                            "construction_validated_correct" if project.get("correct", True) else "construction_validated_incorrect",
                            {
                                "agent": self.name,
                                "project_id": project_id,
                                "structure_type": project.get("type", "unknown"),
                                "decision_action": decision_action,
                            },
                        )

            if action["type"] == "transport_resources":
                project_id = action.get("project_id")
                if not project_id:
                    self._emit_event(
                        sim_state,
                        "construction_transport_blocked",
                        {
                            "agent": self.name,
                            "failure_category": "missing_project_binding",
                            "decision_action": action.get("decision_action"),
                        },
                    )
                    continue
                project = environment.construction.projects.get(project_id)
                if not project:
                    self._emit_event(
                        sim_state,
                        "construction_transport_blocked",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "failure_category": "unknown_project",
                            "decision_action": action.get("decision_action"),
                        },
                    )
                    continue
                if project.get("status") == "complete":
                    self._emit_event(
                        sim_state,
                        "construction_transport_blocked",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "failure_category": "project_already_complete",
                            "decision_action": action.get("decision_action"),
                        },
                    )
                    continue

                if not environment.is_interaction_target_unlocked(project_id):
                    self._emit_event(
                        sim_state,
                        "construction_transport_blocked",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "failure_category": "bridge_access_locked",
                            "decision_action": action.get("decision_action"),
                        },
                    )
                    continue

                required_before = int(project.get("required_resources", {}).get("bricks", 0) or 0)
                delivered_before = int(project.get("delivered_resources", {}).get("bricks", 0) or 0)
                status_before = project.get("status", "in_progress")
                if delivered_before >= required_before and required_before > 0:
                    self._emit_event(
                        sim_state,
                        "construction_transport_blocked",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "failure_category": "resources_already_satisfied",
                            "decision_action": action.get("decision_action"),
                        },
                    )
                    continue

                state = self.transport_state
                if state.get("bound_project_id") != project_id:
                    state.update(
                        {
                            "stage": "pickup",
                            "carrying": {"resource_type": None, "quantity": 0},
                            "pickup_source_id": None,
                            "bound_project_id": project_id,
                        }
                    )
                pickup_id = state.get("pickup_source_id")
                pickup_lookup = dict(_pickup_candidates())
                if state.get("stage") == "pickup":
                    if pickup_id not in pickup_lookup:
                        if not pickup_lookup:
                            _log_mutation_blocked(
                                "construction_transport_blocked",
                                {"project_id": project_id, "failure_category": "no_pickup_resource_source", "decision_action": action.get("decision_action")},
                            )
                            continue
                        pickup_id = sorted(
                            pickup_lookup.keys(),
                            key=lambda pid: math.hypot(self.position[0] - pickup_lookup[pid][0], self.position[1] - pickup_lookup[pid][1]),
                        )[0]
                        state["pickup_source_id"] = pickup_id
                        action["target"] = pickup_lookup[pickup_id]
                        self.target = action["target"]
                    pickup_pos = pickup_lookup.get(pickup_id)
                    if pickup_pos is None:
                        continue
                    pickup_dist = math.hypot(self.position[0] - pickup_pos[0], self.position[1] - pickup_pos[1])
                    if pickup_dist > 0.4:
                        continue
                    state["carrying"] = {"resource_type": "bricks", "quantity": 1}
                    state["stage"] = "in_transit"
                    dropoff = _build_target(project_id)
                    if dropoff is not None:
                        action["target"] = dropoff
                        self.target = dropoff
                    self._emit_event(
                        sim_state,
                        "construction_transport_pickup",
                        {"agent": self.name, "project_id": project_id, "pickup_source_id": pickup_id, "pickup_distance": round(pickup_dist, 4)},
                    )
                    continue

                access = _build_access(project_id)
                expected_location = _build_target(project_id)
                if not access.get("accessible"):
                    continue
                carrying = state.get("carrying") or {}
                if str(carrying.get("resource_type")) != "bricks" or int(carrying.get("quantity", 0) or 0) <= 0:
                    _log_mutation_blocked(
                        "construction_transport_blocked",
                        {
                            "project_id": project_id,
                            "failure_category": "missing_carried_resource_state",
                            "decision_action": action.get("decision_action"),
                            "expected_interaction_location": expected_location,
                        },
                    )
                    state["stage"] = "pickup"
                    continue

                environment.construction.assign_builder(project_id, self.name)
                delivered_ok = environment.construction.deliver_resource(project_id, "bricks", quantity=1)
                if not delivered_ok:
                    _log_mutation_blocked(
                        "construction_transport_blocked",
                        {
                            "project_id": project_id,
                            "failure_category": "construction_delivery_rejected",
                            "decision_action": action.get("decision_action"),
                            "expected_interaction_location": expected_location,
                        },
                    )
                    continue
                state["carrying"] = {"resource_type": None, "quantity": 0}
                state["stage"] = "pickup"
                state["pickup_source_id"] = None

                delivered_after = int(project.get("delivered_resources", {}).get("bricks", 0) or 0)
                required_after = int(project.get("required_resources", {}).get("bricks", 0) or 0)
                status_after = project.get("status", "in_progress")
                progress_before = (delivered_before / required_before) if required_before > 0 else 0.0
                progress_after = (delivered_after / required_after) if required_after > 0 else 0.0

                self._emit_event(
                    sim_state,
                    "construction_resource_delivered",
                    {
                        "agent": self.name,
                        "project_id": project_id,
                        "resource_type": "bricks",
                        "quantity": max(0, delivered_after - delivered_before),
                        "delivered_before": delivered_before,
                        "delivered_after": delivered_after,
                        "required_total": required_after,
                        "progress_before": round(progress_before, 4),
                        "progress_after": round(progress_after, 4),
                        "decision_action": action.get("decision_action"),
                        "legality_checks_passed": True,
                        "expected_interaction_location": expected_location,
                        "agent_position": (round(self.position[0], 3), round(self.position[1], 3)),
                        "distance_to_required_location": round(
                            math.hypot(self.position[0] - expected_location[0], self.position[1] - expected_location[1])
                            if expected_location
                            else -1.0,
                            4,
                        ),
                        "current_action_type": action.get("type"),
                        "transport_state": dict(self.transport_state),
                    },
                )

                if delivered_after != delivered_before:
                    self._emit_event(
                        sim_state,
                        "construction_progress_updated",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "delivered_before": delivered_before,
                            "delivered_after": delivered_after,
                            "required_total": required_after,
                            "progress_before": round(progress_before, 4),
                            "progress_after": round(progress_after, 4),
                            "status_after": status_after,
                            "decision_action": action.get("decision_action"),
                            "agent_position": (round(self.position[0], 3), round(self.position[1], 3)),
                            "expected_interaction_location": expected_location,
                            "distance_to_required_location": round(
                                math.hypot(self.position[0] - expected_location[0], self.position[1] - expected_location[1])
                                if expected_location
                                else -1.0,
                                4,
                            ),
                            "current_action_type": action.get("type"),
                            "transport_state": dict(self.transport_state),
                            "legality_checks_passed": True,
                        },
                    )
                if required_after > 0 and delivered_before < required_after <= delivered_after:
                    self._emit_event(
                        sim_state,
                        "construction_ready_for_validation",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "required_total": required_after,
                            "delivered_total": delivered_after,
                            "decision_action": action.get("decision_action"),
                        },
                    )
                if status_before != "complete" and status_after == "complete":
                    self._emit_event(
                        sim_state,
                        "construction_completed",
                        {
                            "agent": self.name,
                            "project_id": project_id,
                            "structure_type": project.get("type", "unknown"),
                            "decision_action": action.get("decision_action"),
                        },
                    )

    def update_active_actions(self, dt):
        """Deprecated wrapper: use `_advance_active_actions(...)` in live path."""
        self._advance_active_actions(dt, sim_state=None)

    def _advance_active_actions(self, dt, sim_state=None):
        completed = []

        for action in self.active_actions:
            action["progress"] += dt
            if action["progress"] >= action["duration"]:
                completed.append(action)

        for action in completed:
            self.perform_action([action])
            self._emit_event(sim_state, "executable_action_completed", {"action_type": action.get("type"), "decision_action": action.get("decision_action"), "project_id": action.get("project_id")})
            self.active_actions.remove(action)

        if not self.active_actions and self.current_action:
            for action in self.current_action:
                self.active_actions.append({
                    "type": action["type"],
                    "target": action.get("target"),
                    "duration": action.get("duration", 1.0),
                    "progress": 0.0,
                    "priority": action.get("priority", 1),
                    "project_id": action.get("project_id"),
                    "artifact_action": action.get("artifact_action"),
                    "assist_action": action.get("assist_action"),
                    "decision_action": action.get("decision_action"),
                    "source_target_id": action.get("source_target_id"),
                })
                if action.get("type") in {"move_to", "construct", "transport_resources"} and action.get("target") is not None:
                    self.target = action.get("target")
            self.current_action = []

    def generate_message(self):
        messages = []
        if self.mental_model["data"]:
            messages.append({"type": "TDP", "content": list(self.mental_model["data"]), "sender": self.name})
        if self.mental_model["information"]:
            messages.append({"type": "TIP", "content": list(self.mental_model["information"]), "sender": self.name})
        if self.mental_model["knowledge"].rules:
            messages.append({"type": "TKP", "content": list(self.mental_model["knowledge"].rules), "sender": self.name})
        if not self.mental_model["knowledge"].rules:
            messages.append({"type": "TKRQ", "content": ["Requesting help with rules"], "sender": self.name})
        if self.current_goal():
            messages.append({"type": "TGTO", "content": self.current_goal()["goal"], "sender": self.name})
        return messages

    def communicate_with(self, other_agent, sim_state=None):
        messages = self.generate_message()
        message_types = []
        for msg in messages:
            message_types.append(msg.get("type"))
            other_agent.receive_message(msg, from_agent=self.name)

        # Directly transfer unseen Data
        for data in self.mental_model["data"]:
            if data not in other_agent.mental_model["data"]:
                other_agent.mental_model["data"].add(data)
                if other_agent.name not in data.acquired_by:
                    data.acquired_by[other_agent.name] = {
                        "mode": "shared",
                        "from": self.name
                    }
                other_agent.activity_log.append(f"Received data {data.id} from {self.name}")
                DIK_LOG.append({
                    "time": getattr(self, "current_time", 0.0),
                    "agent": other_agent.name,
                    "type": "Data",
                    "id": data.id,
                    "mode": "shared",
                    "from": self.name
                })

        # Directly transfer unseen Information
        for info in self.mental_model["information"]:
            if info not in other_agent.mental_model["information"]:
                other_agent.mental_model["information"].add(info)
                if other_agent.name not in info.acquired_by:
                    info.acquired_by[other_agent.name] = {
                        "mode": "shared",
                        "from": self.name
                    }
                other_agent.activity_log.append(f"Received info {info.id} from {self.name}")
                DIK_LOG.append({
                    "time": getattr(self, "current_time", 0.0),
                    "agent": other_agent.name,
                    "type": "Information",
                    "id": info.id,
                    "mode": "shared",
                    "from": self.name
                })

        # Transfer inferred Rules (Knowledge)
        for rule in self.mental_model["knowledge"].rules:
            if rule not in other_agent.mental_model["knowledge"].rules:
                info_ids = self.mental_model["knowledge"].built_from.get(rule, [])
                other_agent.mental_model["knowledge"].add_rule(rule, info_ids)
                other_agent.activity_log.append(f"Received rule from {self.name}")
                DIK_LOG.append({
                    "time": getattr(self, "current_time", 0.0),
                    "agent": other_agent.name,
                    "type": "Knowledge",
                    "id": rule,
                    "mode": "shared",
                    "from": self.name
                })
                if sim_state is not None:
                    sim_state.logger.log_event(sim_state.time, "rule_adopted", {"agent": other_agent.name, "agent_id": getattr(other_agent, "agent_id", other_agent.name), "rule_id": rule, "adoption_mode": "communication", "from_agent": self.name})

        # Update Theory of Mind estimates
        other_agent.theory_of_mind[self.name] = {
            "goals": [self.goal] if self.goal else [],
            "knowledge_ids": set(self.mental_model["knowledge"].rules),
            "last_seen": None
        }

        # Physiological cost of talking
        self.update_physiology(exertion=0.1, speaking=True)
        other_agent.update_physiology(exertion=0.1, speaking=True)

        self.activity_log.append(f"Communicated with {other_agent.name}")
        if sim_state is not None:
            sim_state.logger.log_event(
                sim_state.time,
                "communication_exchange",
                {"sender": self.name, "receiver": other_agent.name, "message_types": message_types},
            )

    def receive_message(self, message, from_agent=None):
        sender = message.get("sender")
        if not sender:
            sender = from_agent

        if sender not in self.theory_of_mind:
            self.theory_of_mind[sender] = {"goals": [], "knowledge_ids": set(), "last_seen": None}

        mtype = message.get("type")
        mtype = LEGACY_COMMUNICATION_TYPE_MAP.get(mtype, mtype)
        content = message.get("content", [])

        if mtype == "TDP":
            for d in content:
                if d not in self.mental_model["data"]:
                    self.mental_model["data"].add(d)
                    if self.name not in d.acquired_by:
                        d.acquired_by[self.name] = {"mode": "shared", "from": sender}
                    DIK_LOG.append({
                        "time": getattr(self, "current_time", 0.0),
                        "agent": self.name,
                        "type": "Data",
                        "id": d.id,
                        "mode": "shared",
                        "from": sender
                    })

        elif mtype == "TIP":
            for info in content:
                if info not in self.mental_model["information"]:
                    self.mental_model["information"].add(info)
                    if self.name not in info.acquired_by:
                        info.acquired_by[self.name] = {"mode": "shared", "from": sender}
                    DIK_LOG.append({
                        "time": getattr(self, "current_time", 0.0),
                        "agent": self.name,
                        "type": "Information",
                        "id": info.id,
                        "mode": "shared",
                        "from": sender
                    })

        elif mtype == "TKP":
            for rule in content:
                if rule not in self.mental_model["knowledge"].rules:
                    inferred_from = self.theory_of_mind[sender].get("knowledge_ids", [])
                    self.mental_model["knowledge"].add_rule(rule, inferred_from)
                    DIK_LOG.append({
                        "time": getattr(self, "current_time", 0.0),
                        "agent": self.name,
                        "type": "Knowledge",
                        "id": rule,
                        "mode": "shared",
                        "from": sender
                    })

        elif mtype == "TGTO":
            self.theory_of_mind[sender]["goals"] = [content]

        elif mtype == "TKRQ":
            self.known_gaps.update(content)

        elif mtype == "TCR":
            self.reevaluate_knowledge()

        self.activity_log.append(f"Received {mtype} from {sender}")

    def reevaluate_knowledge(self):
        # Recombine info to try and re-infer
        candidate_tags = {}
        for info in self.mental_model["information"]:
            for tag in info.tags:
                candidate_tags.setdefault(tag, []).append(info)

        for tag, group in candidate_tags.items():
            if len(group) >= 2:
                if random.random() < 0.95:  # Retry with high confidence
                    self.mental_model["knowledge"].try_infer_rules(group)
                    self.activity_log.append(f"Reinferred rule from tag [{tag}]")

    def should_request_knowledge(self):
        # Decide whether the agent should explicitly request help
        if not self.mental_model["knowledge"].rules and self.goal not in ["seek_info", "request_info"]:
            return True
        return False


    def compare_and_repair_construction(self, construction, sim_state=None):
        for project in construction.projects.values():
            if not isinstance(project, dict):
                continue
            if not project.get("in_progress", False):
                continue
            project_id = project.get("id", "unknown")
            required = int(project.get("required_resources", {}).get("bricks", 0) or 0)
            delivered = int(project.get("delivered_resources", {}).get("bricks", 0) or 0)
            if required <= 0 or delivered <= 0:
                self._emit_event(sim_state, "mismatch_detection_skipped_not_ready", {"project_id": project_id, "reason": "insufficient_build_state", "required": required, "delivered": delivered})
                continue
            readiness_ratio = delivered / max(1, required)
            if readiness_ratio < 0.5:
                self._emit_event(sim_state, "mismatch_detection_skipped_not_ready", {"project_id": project_id, "reason": "build_state_below_validation_threshold", "readiness_ratio": round(readiness_ratio, 3)})
                continue
            known_rules = {normalize_rule_token(r) for r in self.mental_model["knowledge"].rules}
            if not known_rules:
                self._emit_event(sim_state, "mismatch_detection_skipped_not_ready", {"project_id": project_id, "reason": "no_rules_to_validate_against"})
                continue
            now_ts = float(getattr(sim_state, "time", getattr(self, "current_time", 0.0)))
            last_mismatch = float(self.construction_validation_state["mismatch_last_ts"].get(project_id, -999.0))
            if now_ts - last_mismatch < 5.0:
                self._emit_event(sim_state, "mismatch_detection_skipped_not_ready", {"project_id": project_id, "reason": "cooldown_active", "cooldown_remaining": round(5.0 - (now_ts - last_mismatch), 3)})
                continue
            expected_rules = {normalize_rule_token(r) for r in project.get("expected_rules", [])}
            rule_matches = bool(known_rules.intersection(expected_rules))
            if not rule_matches:
                mismatch_sensitivity = max(self._hook_value("validation_check", "detect_mismatch", "sensitivity", default=0.5), self._trait_value("rule_accuracy"))
                mismatch_detect_prob = min(1.0, 0.25 + 0.7 * mismatch_sensitivity)
                if random.random() <= mismatch_detect_prob:
                    self.construction_validation_state["mismatch_last_ts"][project_id] = now_ts
                    self.activity_log.append(f"Disagrees with approach for {project.get('name', 'Unknown')}")
                    if sim_state is not None:
                        sim_state.logger.log_event(
                            sim_state.time,
                            "construction_mismatch_detected",
                            {"agent": self.name, "project_id": project.get("id", "unknown")},
                        )
                    if readiness_ratio < 0.8:
                        self._emit_event(sim_state, "repair_trigger_suppressed_not_ready", {"project_id": project_id, "reason": "build_not_ready_for_repair", "readiness_ratio": round(readiness_ratio, 3)})
                        continue
                    last_repair = float(self.construction_validation_state["repair_last_ts"].get(project_id, -999.0))
                    if now_ts - last_repair < 5.0:
                        self._emit_event(sim_state, "repair_trigger_suppressed_not_ready", {"project_id": project_id, "reason": "repair_cooldown_active", "cooldown_remaining": round(5.0 - (now_ts - last_repair), 3)})
                        continue
                    if random.random() < self._trait_value("help_tendency"):
                        self.construction_validation_state["repair_last_ts"][project_id] = now_ts
                        project["correct"] = True
                        self.activity_log.append("Triggered correction/repair on construction externalization")
                        if sim_state is not None:
                            sim_state.logger.log_event(
                                sim_state.time,
                                "construction_repair_episode",
                                {"agent": self.name, "project_id": project.get("id", "unknown")},
                            )
                    else:
                        project["correct"] = False


    def draw(self, ax):
        if patches is None:
            return

        x, y = self.position
        angle = self.orientation
        color = ROLE_COLORS.get(self.role, "gray")
        body = patches.Circle((x, y), radius=0.2, facecolor=color, edgecolor='black')
        ax.add_patch(body)
        dx = 0.3 * math.cos(angle)
        dy = 0.3 * math.sin(angle)
        ax.arrow(x, y, dx, dy, head_width=0.1, head_length=0.1, fc=color, ec='black')
        ax.text(x, y + 0.35, self.role, ha='center', fontsize=9)
