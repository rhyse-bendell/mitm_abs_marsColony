# File: modules/agent.py

import math
import random
import threading
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
from modules.brain_contract import AgentBrainRequest, AgentBrainResponse, validate_agent_brain_response
from modules.goal_manager import GoalManager
from modules.goal_state import GOAL_SOURCES, GOAL_STATUSES
from modules.plan_state import PlanRecord

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
        self.inspect_stall_counts = {}
        self.current_inspect_target_id = None
        self.status_last_action = ""

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
        self._planner_request_seq = 0
        self.planner_state = {
            "status": "idle",
            "request_id": None,
            "request_tick": None,
            "requested_at": None,
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
            "total_skipped_cooldown": 0,
            "total_stale_discarded": 0,
            "stale_plan_reuse_count": 0,
            "ui_safe_fallback_count": 0,
            "degraded_mode_episodes": 0,
            "consecutive_failure_sum": 0,
            "consecutive_failure_samples": 0,
        }
        self._planner_future = None
        self._planner_future_lock = threading.Lock()
        self._timed_out_request_ids = set()


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

    def _trait_value(self, name, default=0.5):
        value = getattr(self, name, default)
        return max(0.0, min(1.0, float(value)))

    def _hook_value(self, hook_type, hook_target, parameter, default=None):
        if default is None:
            default = 0.0 if parameter in {"utility_weight", "priority_weight", "externalization_weight", "persistence_weight", "adoption_weight", "sensitivity"} else 1.0
        return float(getattr(self, "hook_effects", {}).get((hook_type, hook_target, parameter), default))

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

    def _candidate_information_sources(self, environment):
        self._ensure_source_state(environment)
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
            if packet_name == f"{self.role}_Info":
                score += 1.5
            if packet_name == "Team_Info":
                score += 1.0
            score -= stalled * 1.5
            score -= math.hypot(point[0] - self.position[0], point[1] - self.position[1]) * 0.2
            candidates.append((score, packet_name, point, status, stalled))

        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates

    def _resolve_inspect_target(self, decision, environment, sim_state=None):
        self._ensure_source_state(environment)
        explicit_target = decision.target_id
        self._emit_event(sim_state, "target_resolution_started", {"target_type": "information_source", "requested_target_id": explicit_target})
        if explicit_target:
            point = environment.get_interaction_target_position(explicit_target, from_position=self.position)
            if point is not None:
                self._set_status(f"Inspect target selected: {explicit_target}")
                self._emit_event(sim_state, "target_resolved", {"target_type": "information_source", "target_id": explicit_target, "candidate_count": 1})
                return explicit_target, point

        candidates = self._candidate_information_sources(environment)
        if not candidates:
            self._set_status("Inspect target resolution failed: no accessible information sources")
            self._emit_event(sim_state, "target_resolution_failed", {"target_type": "information_source", "failure_category": "no_information_source_available"})
            return None, None

        # Conservative retargeting away from repeatedly stalled targets when alternatives exist.
        non_stalled = [c for c in candidates if c[4] < 3]
        chosen = non_stalled[0] if non_stalled else candidates[0]
        if explicit_target is None:
            self._set_status(
                f"Inspect decision missing explicit target; resolved to {chosen[1]} (status={chosen[3]}, stalled={chosen[4]})"
            )
        elif chosen[1] != explicit_target:
            self._set_status(
                f"Inspect target {explicit_target} unreachable; retargeted to {chosen[1]} (status={chosen[3]}, stalled={chosen[4]})"
            )
        self._emit_event(sim_state, "target_resolved", {"target_type": "information_source", "target_id": chosen[1], "candidate_count": len(candidates)})
        return chosen[1], chosen[2]

    def mark_source_revisitable(self, source_id, reason="identified_gap"):
        self.source_inspection_state[source_id] = "revisitable_due_to_gap"
        self._set_status(f"Source marked revisitable due to gap: {source_id} ({reason})")

    def _inspect_source(self, environment, source_id):
        packet = environment.knowledge_packets.get(source_id)
        if packet is None:
            self._set_status(f"Inspect failed: unknown source {source_id}")
            return False

        self.source_inspection_state[source_id] = "in_progress"
        target_pos = environment.get_interaction_target_position(source_id, from_position=self.position)
        if target_pos is None:
            self._set_status(f"Inspect failed: no navigable target for {source_id}")
            return False
        if not environment.can_access_info(self.position, source_id, role=self.role):
            self._set_status(f"Inspect pending: not yet in source zone for {source_id}")
            return False

        self._set_status(f"Arrived at source zone: {source_id}")
        before_ids = {info.id for info in self.mental_model["information"]}
        self.absorb_packet(packet, accuracy=0.95)
        after_ids = {info.id for info in self.mental_model["information"]}
        packet_info_ids = {info.id for info in packet.get("information", [])}
        new_ids = after_ids - before_ids
        self.memory_seen_packets.add(source_id)

        if new_ids:
            self._set_status(f"Source access succeeded: {source_id} (+{len(new_ids)} new items)")
        elif packet_info_ids.issubset(after_ids):
            self._set_status(f"Source already inspected: {source_id}")
        else:
            self._set_status(f"Source access had no uptake: {source_id}")

        if packet_info_ids.issubset(after_ids):
            self.source_inspection_state[source_id] = "inspected"
        else:
            self.source_inspection_state[source_id] = "in_progress"
        return bool(new_ids)



    def _held_dik_ids(self):
        data_ids = {d.id for d in self.mental_model["data"]}
        info_ids = {i.id for i in self.mental_model["information"]}
        knowledge_ids = set(self.mental_model["knowledge"].rules)
        return data_ids, info_ids, knowledge_ids

    def _create_dik_object_from_element(self, element, source_id):
        tags = [element.role_scope, element.phase_scope, element.element_type]
        if element.element_type == "data":
            return Data(element.element_id, element.description, source=source_id, tags=tags)
        if element.element_type == "information":
            return Information(element.element_id, element.description, source=source_id, tags=tags)
        return None

    def _apply_task_derivations(self, sim_state=None):
        if not getattr(self, "task_model", None):
            return

        now = getattr(self, "current_time", 0.0)
        data_ids, info_ids, knowledge_ids = self._held_dik_ids()
        held_ids = data_ids | info_ids | knowledge_ids

        for derivation in self.task_model.derivations.values():
            if not derivation.enabled or derivation.derivation_id in self.executed_derivations:
                continue

            required = set(derivation.required_inputs)
            if required and not required.issubset(held_ids):
                continue
            if derivation.min_required_count and len(required & held_ids) < derivation.min_required_count:
                continue

            output_id = derivation.output_element_id
            element = self.task_model.dik_elements.get(output_id)
            if element is None or not element.enabled:
                continue

            produced = False
            if derivation.output_type == "knowledge" or element.element_type == "knowledge":
                if output_id not in self.mental_model["knowledge"].rules:
                    self.mental_model["knowledge"].add_rule(output_id, sorted(required), inferred_by_agents=[self.name])
                    produced = True
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
                }
                self.executed_derivations.add(derivation.derivation_id)
                self.derivation_events.append(event)
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
                    sim_state.logger.log_event(now, "dik_derivation_executed", event)

    def _build_readiness_score(self):
        info_count = len(self.mental_model["information"])
        knowledge_count = len(self.mental_model["knowledge"].rules)
        inspected_sources = sum(1 for state in self.source_inspection_state.values() if state == "inspected")
        artifact_count = len([p for p in self.memory_seen_packets if isinstance(p, str) and p.startswith("whiteboard:")])
        return info_count + (2 * knowledge_count) + inspected_sources + min(artifact_count, 1)

    def _build_readiness_blockers(self, environment):
        blockers = []
        if len(self.mental_model["information"]) < 2:
            blockers.append("insufficient_information_inspection")
        if len(self.mental_model["knowledge"].rules) < 1:
            blockers.append("insufficient_rule_knowledge")

        if getattr(self, "task_model", None):
            role_rules = [
                r.rule_id for r in self.task_model.rules.values()
                if r.enabled and r.role_scope in {"team", self.role.lower()}
            ]
            if role_rules and not set(role_rules).intersection(set(self.mental_model["knowledge"].rules)):
                blockers.append("missing_task_prerequisite_rules")
        if not any(state == "inspected" for state in self.source_inspection_state.values()):
            blockers.append("no_inspected_information_source")
        if self._select_build_target(environment, require_readiness=False) is None:
            blockers.append("no_navigable_build_target")
        return blockers

    def _is_build_eligible(self, environment):
        return self._build_readiness_score() >= self._readiness_threshold() and not self._build_readiness_blockers(environment)

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

    def _activate_support_goal(self, label, reason, sim_state=None, priority=0.7, source="derived_from_rule"):
        mission_goal = next((g for g in self.goal_registry.values() if g.goal_level == "mission"), None)
        goal = self._upsert_goal_record(
            label=label,
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
        self.activity_log.append(f"Support goal active: {goal.label} ({reason})")

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

        missing_dik = len(self.known_gaps) > 0 or (len(info_ids) + len(knowledge_ids)) < 2
        if missing_dik:
            self._activate_support_goal("acquire_missing_dik", "missing_dik_detected", sim_state=sim_state, priority=0.82, source="derived_from_rule")

        if any("mismatch with construction" in e.lower() for e in self.activity_log[-6:]):
            self._activate_support_goal("repair_detected_mismatch", "construction_mismatch_detected", sim_state=sim_state, priority=0.85, source="derived_from_rule")
            self._activate_support_goal("validate_externalization", "validation_needed_after_mismatch", sim_state=sim_state, priority=0.8, source="derived_from_rule")

        if getattr(self, "derivation_events", []):
            last = self.derivation_events[-1]
            self._activate_support_goal("integrate_new_derivation", f"derivation:{last.get('derivation_id')}", sim_state=sim_state, priority=0.65, source="derived_from_rule")

        if sim_state is not None and sim_state.team_knowledge_manager.recent_updates:
            self._activate_support_goal("consult_artifact", "teammate/shared-artifact influenced", sim_state=sim_state, priority=0.6, source="teammate_or_artifact_influenced")

        repeated_stall = any(v >= 3 for v in self.inspect_stall_counts.values())
        if repeated_stall:
            self._activate_support_goal("unblock_inspection", "repeated_inspection_stall", sim_state=sim_state, priority=0.75, source="derived_from_rule")

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
        return AgentBrainRequest(
            request_id=f"{self.agent_id}-{uuid.uuid4().hex[:8]}",
            tick=self.sim_step_count,
            sim_time=float(sim_state.time),
            agent_id=self.agent_id,
            display_name=self.display_name,
            agent_label=self.agent_label,
            task_id=getattr(sim_state.task_model, "task_id", "unknown"),
            phase=str(phase),
            local_context_summary=f"trigger={trigger_reason};build={context.individual_cognitive_state.get('build_readiness',{}).get('status')}",
            local_observations=observations,
            working_memory_summary={
                "data": list(context.individual_cognitive_state.get("data_summary", [])[:8]),
                "information": list(context.individual_cognitive_state.get("information_summary", [])[:8]),
                "knowledge": list(context.individual_cognitive_state.get("knowledge_summary", [])[:8]),
                "known_gaps": list(context.individual_cognitive_state.get("known_gaps", [])[:8]),
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
        )

    def _execute_planner_request_sync(self, sim_state, trigger_reason, request_packet, request_explanation, request_started_at, request_sim_time, trace_id):
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

        provider_trace = getattr(provider, "last_trace", None)
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
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"fallback-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": f"fallback-plan-{request_packet.request_id}",
                        "plan_horizon": 1,
                        "ordered_goals": [{"goal_id": "safety", "description": "hold legal action", "priority": 1.0, "status": "active"}],
                        "ordered_actions": [{"step_index": 0, "action_type": "wait", "expected_purpose": "fallback legal wait"}],
                        "next_action": {"step_index": 0, "action_type": "wait", "expected_purpose": "fallback legal wait"},
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
        decision = self._apply_trait_bias_to_decision(decision, context, sim_state, trigger_reason)
        legacy_errors = validate_brain_decision(decision, [ExecutableActionType(a) for a in legal_action_ids])
        legacy_validation_errors = list(legacy_errors)
        status = "repaired" if repaired else "accepted"
        if legacy_errors:
            status = "rejected"
            decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="Fallback due to decision validation failure.", confidence=1.0)

        latency_s = max(0.0, sim_state.time - request_sim_time)
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
            "latency_s": latency_s,
            "failed": False,
            "timed_out": False,
            "trace": {
                "trace_id": trace_id,
                "schema_validation_succeeded": not bool(schema_validation_errors),
                "schema_validation_errors": list(schema_validation_errors),
                "legacy_decision_validation_errors": list(legacy_validation_errors),
                "validation_repaired": bool(validation_repaired),
                "grounding_status": grounding_status,
                "grounding_notes": list(grounding_notes),
                "provider_trace": provider_trace if isinstance(provider_trace, dict) else None,
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
        request_explanation = self._should_request_explanation()
        request_packet = self._build_brain_request(sim_state, context, request_explanation, trigger_reason)
        trace_id = self._make_planner_trace_id(request_packet.request_id)
        self._planner_request_seq += 1
        request_started_at = self.sim_step_count
        request_sim_time = float(sim_state.time)
        self.planner_state["status"] = "in_flight"
        self.planner_state["request_id"] = request_packet.request_id
        self.planner_state["trace_id"] = trace_id
        self.planner_state["request_tick"] = self.sim_step_count
        self.planner_state["requested_at"] = request_sim_time
        self.planner_state["error"] = None
        self.planner_state["trigger_reason"] = trigger_reason
        self.planner_state["request_payload"] = request_packet.to_dict()
        self.planner_state["configured_backend"] = configured_backend
        self.planner_state["effective_backend"] = effective_backend
        self.planner_state["model"] = provider_cfg.local_model
        self.planner_state["last_result"] = None
        self.planner_state["total_started"] += 1
        self._emit_event(sim_state, "planner_request_started_async", {"request_id": request_packet.request_id, "trace_id": trace_id, "trigger_reason": trigger_reason, "backend": configured_backend, "effective_backend": effective_backend, "timeout": self.planner_cadence.planner_timeout_seconds, "model": provider_cfg.local_model, "queue_depth": 1})
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
                trace_id,
            )

    def _planner_cooldown_remaining(self, sim_state):
        return max(0.0, float(self.planner_state.get("cooldown_until", 0.0)) - float(sim_state.time))


    def _register_planner_failure(self, sim_state, request_id, reason, timed_out=False):
        self.planner_state["status"] = "timed_out" if timed_out else "failed"
        if timed_out:
            self.planner_state["total_timed_out"] += 1
            self._timed_out_request_ids.add(str(request_id))
        else:
            self.planner_state["total_failed"] += 1
        self.planner_state["error"] = str(reason)
        self.planner_state["consecutive_failures"] += 1
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
                "fallback_reason": str(reason),
                "exception": {"type": "TimeoutError" if timed_out else "PlannerExecutionError", "message": str(reason)},
                "schema_validation_succeeded": False,
                "plan_grounding_succeeded": False,
                "plan_disposition": "failed_before_response",
            },
        )

    def _check_inflight_timeout(self, sim_state):
        if self.planner_state.get("status") != "in_flight":
            return
        requested_at = self.planner_state.get("requested_at")
        if requested_at is None:
            return
        timeout_s = float(self.planner_cadence.planner_timeout_seconds)
        elapsed = float(sim_state.time) - float(requested_at)
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

        if result.get("request_id") != request_id:
            self.planner_state["total_stale_discarded"] += 1
            self._emit_event(sim_state, "planner_request_result_arrived_stale", {"request_id": result.get("request_id"), "expected_request_id": request_id, "trace_id": result.get("trace_id")})
            self._append_planner_trace(sim_state, {"trace_id": result.get("trace_id"), "request_id": result.get("request_id"), "agent_id": self.agent_id, "sim_time": float(sim_state.time), "planner_result": "stale_discarded", "plan_disposition": "discarded_stale_request_id_mismatch", "fallback": True, "fallback_reason": "request_id_mismatch"})
            return False
        if result.get("request_id") in self._timed_out_request_ids:
            self._timed_out_request_ids.discard(result.get("request_id"))
            self.planner_state["total_stale_discarded"] += 1
            self._emit_event(sim_state, "planner_request_result_arrived_stale", {"request_id": result.get("request_id"), "reason": "arrived_after_timeout", "trace_id": result.get("trace_id")})
            self._append_planner_trace(sim_state, {"trace_id": result.get("trace_id"), "request_id": result.get("request_id"), "agent_id": self.agent_id, "sim_time": float(sim_state.time), "planner_result": "stale_discarded", "plan_disposition": "discarded_arrived_after_timeout", "fallback": True, "fallback_reason": "arrived_after_timeout", "trace": result.get("trace")})
            return False

        self.planner_call_count += 1
        self.last_planner_step = self.sim_step_count
        self.last_planner_time = sim_state.time
        self.planner_state["completed_at"] = sim_state.time
        self.planner_state["last_latency_s"] = result.get("latency_s")
        self.planner_state["last_result"] = result
        self.planner_state["last_result_request_id"] = request_id
        self.planner_state["total_completed"] += 1

        decision = result["decision"]
        status = result["status"]
        response = result["response"]
        trigger_reason = result["trigger_reason"]

        self._emit_event(sim_state, "brain_provider_response_received", {"configured_backend": result["configured_backend"], "effective_backend": result["effective_backend"], "provider_class": result["provider_name"], "has_plan": bool(getattr(response, "plan", None)), "has_explanation": bool(getattr(response, "explanation", None)), "request_id": request_id, "trace_id": result.get("trace_id")})
        self._emit_event(sim_state, "planner_invocation_completed", {"trigger_reason": trigger_reason, "decision_status": status, "selected_action": decision.selected_action.value, "request_explanation": result["request_explanation"], "request_id": request_id, "trace_id": result.get("trace_id")})

        if result["errors"]:
            sim_state.logger.log_event(sim_state.time, "planner_next_action_rejected", {"agent": self.name, "errors": list(result["errors"]), "fallback_action": "wait"})
            self._emit_event(sim_state, "brain_provider_response_invalid", {"errors": list(result["errors"]), "schema_parsing_succeeded": False, "request_id": request_id, "trace_id": result.get("trace_id")})

        if self.planner_state["request_tick"] is not None and self.planner_state["request_tick"] < max(0, self.sim_step_count - 2) and self.current_plan is not None and trigger_reason not in {"no_active_plan", "plan_invalidated", "plan_completed"}:
            self.planner_state["total_stale_discarded"] += 1
            self.planner_state["status"] = "completed"
            self._emit_event(sim_state, "planner_response_discarded_due_to_state_change", {"request_id": request_id, "trace_id": result.get("trace_id"), "request_tick": self.planner_state["request_tick"], "current_tick": self.sim_step_count, "current_plan_id": getattr(self.current_plan, "plan_id", None)})
            self._append_planner_trace(sim_state, {"trace_id": result.get("trace_id"), "request_id": request_id, "agent_id": self.agent_id, "sim_time": float(sim_state.time), "planner_result": status, "plan_disposition": "discarded_due_to_state_change", "fallback": bool(result.get("errors")), "fallback_reason": "state_changed_before_adoption", "trace": result.get("trace")})
            return False

        self._adopt_new_plan(decision, trigger_reason, sim_state, response=response, trace_id=result.get("trace_id"))
        self._refresh_goal_plan_state(decision, sim_state, trigger_reason, response=response)
        self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment)
        self.planner_state["status"] = "completed"
        self.planner_state["consecutive_failure_sum"] += self.planner_state["consecutive_failures"]
        self.planner_state["consecutive_failure_samples"] += 1
        self.planner_state["consecutive_failures"] = 0
        if self.planner_state["degraded_mode"]:
            self.planner_state["degraded_mode"] = False
            self._emit_event(sim_state, "backend_degraded_mode_ended", {"agent_id": self.agent_id, "request_id": request_id})
        self._emit_event(sim_state, "planner_request_completed_async", {"request_id": request_id, "trace_id": result.get("trace_id"), "result": status, "latency": result.get("latency_s"), "consecutive_failures": self.planner_state["consecutive_failures"]})
        translated_actions = list(self.current_action or [])
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
            "fallback": bool(result.get("errors")) or bool((result.get("trace", {}).get("provider_trace") or {}).get("fallback")),
            "fallback_reason": (result.get("trace", {}).get("provider_trace") or {}).get("fallback_reason") or ("validation_repaired" if status == "repaired" else None),
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
                "selected_action": decision.selected_action.value,
                "confidence": decision.confidence,
                "errors": result["errors"],
                "request_explanation": result["request_explanation"],
                "explanation_present": bool(response.explanation),
                "planner_call_count": self.planner_call_count,
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
        force_externalize = trigger_reason == "new_dik_acquired" and comm >= 0.7 and random.random() < ((comm + ext_utility) / 2.0)

        if force_externalize:
            selected = ExecutableActionType.EXTERNALIZE_PLAN
            reason_bits.append("communication_propensity pushed externalization after DIK change")

        if align >= 0.75 and context.team_state.get("plan_readiness") == "validated_shared_plan" and random.random() < ((align + consult_utility) / 2.0):
            selected = ExecutableActionType.CONSULT_TEAM_ARTIFACT
            reason_bits.append("goal_alignment favored validated team artifact consultation")

        if help_t >= 0.7 and self._help_context_available(sim_state) and random.random() < ((help_t + assist_utility) / 2.0):
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
        build_blockers = self._build_readiness_blockers(environment)
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
    def _adopt_new_plan(self, decision, trigger_reason, sim_state, response=None, trace_id=None):
        if self.current_plan is not None and self.current_plan.invalidation_reason is None:
            self.current_plan.invalidation_reason = f"replaced_by_{trigger_reason}"

        plan_obj = getattr(response, "plan", None)
        method_status = getattr(plan_obj, "_method_status", "unspecified") if plan_obj else "unspecified"
        method_notes = list(getattr(plan_obj, "_method_notes", [])) if plan_obj else []
        associated_goal_ids = [g.goal_id for g in getattr(plan_obj, "ordered_goals", []) if getattr(g, "goal_id", None)] if plan_obj else []

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
        self._emit_event(sim_state, "plan_adopted", {"plan_id": self.current_plan.plan_id, "plan_method_id": self.current_plan.plan_method_id, "trust_tier": method_status, "canonical_goal_matches": len([g for g in (self.current_plan.associated_goal_ids or []) if g]), "ad_hoc_goal_count": len([g for g in (self.current_plan.ordered_goals or []) if not g.get("goal_id")]), "next_action_type": decision.selected_action.value, "trace_id": trace_id})
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
            sim_state.logger.log_event(sim_state.time, "plan_invalidated", {"agent": self.name, "plan_id": self.current_plan.plan_id, "reason": self.current_plan.invalidation_reason, "trace_id": self.planner_state.get("trace_id")})
            return False

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
    def _translate_brain_decision_to_legacy_action(self, decision, environment, sim_state=None):
        self._emit_event(sim_state, "planner_next_action_selected", {"planner_action_type": decision.selected_action.value, "plan_id": getattr(self.current_plan, "plan_id", None)})
        if self.task_model is not None:
            enabled = set(self.task_model.enabled_actions_for_role(self.role))
            if decision.selected_action.value not in enabled:
                self._emit_event(sim_state, "action_translation_failed", {"planner_action_type": decision.selected_action.value, "failure_category": "illegal_action"})
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
            source_id, interaction_target = self._resolve_inspect_target(decision, environment, sim_state=sim_state)
            if source_id is None or interaction_target is None:
                self._emit_event(sim_state, "action_translation_failed", {"planner_action_type": decision.selected_action.value, "failure_category": "unresolved_target"})
                return [{"type": "idle", "duration": 1.0, "priority": 1}]
            action["target"] = interaction_target
            action["source_target_id"] = source_id
            self.current_inspect_target_id = source_id
            if self.source_inspection_state.get(source_id) == "inspected":
                self._set_status(f"Source skipped due to completion: {source_id}")
            return [action]

        if decision.target_id:
            interaction_target = environment.get_interaction_target_position(decision.target_id, from_position=self.position)
            if interaction_target is not None:
                action["target"] = interaction_target
            if decision.selected_action in {
                ExecutableActionType.START_CONSTRUCTION,
                ExecutableActionType.CONTINUE_CONSTRUCTION,
                ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
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

        if decision.selected_action in {ExecutableActionType.EXTERNALIZE_PLAN, ExecutableActionType.CONSULT_TEAM_ARTIFACT}:
            action["artifact_action"] = decision.selected_action.value
        if decision.selected_action == ExecutableActionType.REQUEST_ASSISTANCE:
            action["assist_action"] = decision.selected_action.value

        self._emit_event(sim_state, "action_translation_succeeded", {"planner_action_type": decision.selected_action.value, "translated_action_type": action.get("type"), "target_id": decision.target_id, "target_zone": decision.target_zone})
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
        self._advance_active_actions(dt)

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
                self.inventory_resources["bricks"] = self.inventory_resources.get("bricks", 0) + 1
                self.activity_log.append("Transporting resources... (+1 bricks inventory)")
            elif action["type"] == "idle":
                self.activity_log.append("Idling...")

    def absorb_packet(self, packet, accuracy=1.0):
        effective_base = max(self._hook_value("dik_update", "absorb_packet", "success_probability", default=0.5), self._trait_value("rule_accuracy"))
        effective_accuracy = max(0.05, min(1.0, accuracy * (0.6 + 0.4 * effective_base)))
        for d in packet.get("data", []):
            if random.random() <= effective_accuracy:
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
            if random.random() <= effective_accuracy:
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
        def is_blocking_object(obj):
            obj_type = obj.get("type")
            # Anything that is not a bridge/line is impassable unless marked passable
            return obj_type in {"rect", "circle", "blocked"} and not obj.get("passable", False)

        def can_occupy(point):
            return not any(
                environment.is_near_object(point, name, threshold=0.15)
                for name, obj in environment.objects.items()
                if is_blocking_object(obj)
            )

        def segment_is_clear(start, end, samples=24):
            for i in range(1, samples + 1):
                t = i / samples
                px = start[0] + (end[0] - start[0]) * t
                py = start[1] + (end[1] - start[1]) * t
                if not can_occupy((px, py)):
                    return False
            return True

        def first_intersected_blocked_zone(start, end):
            for name, obj in environment.objects.items():
                if obj.get("type") != "blocked":
                    continue
                (x1, y1), (x2, y2) = obj["corners"]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                # Quick broad phase: if any sampled segment point enters zone, treat as intersecting.
                for i in range(1, 25):
                    t = i / 24
                    px = start[0] + (end[0] - start[0]) * t
                    py = start[1] + (end[1] - start[1]) * t
                    if x_min <= px <= x_max and y_min <= py <= y_max:
                        return obj
            return None

        def compute_detour_waypoint(start, final_target):
            blocked_zone = first_intersected_blocked_zone(start, final_target)
            if blocked_zone is None:
                # Fallback for circular/rectangular blockers encountered in practice.
                vx, vy = final_target[0] - start[0], final_target[1] - start[1]
                v_len = math.hypot(vx, vy)
                if v_len < 1e-6:
                    return None

                ux, uy = vx / v_len, vy / v_len
                # Perpendicular candidates (left/right) to slip around local obstacles.
                offset = 0.9
                candidates = [
                    (start[0] - uy * offset, start[1] + ux * offset),
                    (start[0] + uy * offset, start[1] - ux * offset),
                ]

                feasible = []
                for wp in candidates:
                    if not can_occupy(wp):
                        continue
                    if not segment_is_clear(start, wp):
                        continue
                    cost = math.hypot(wp[0] - start[0], wp[1] - start[1]) + math.hypot(
                        final_target[0] - wp[0], final_target[1] - wp[1]
                    )
                    feasible.append((cost, wp))

                if not feasible:
                    return None

                feasible.sort(key=lambda item: item[0])
                return feasible[0][1]

            (x1, y1), (x2, y2) = blocked_zone["corners"]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            margin = 0.35

            candidate_waypoints = [
                (x_min - margin, y_min - margin),
                (x_min - margin, y_max + margin),
                (x_max + margin, y_min - margin),
                (x_max + margin, y_max + margin),
            ]

            feasible = []
            for wp in candidate_waypoints:
                if not can_occupy(wp):
                    continue
                if not segment_is_clear(start, wp):
                    continue
                total_cost = math.hypot(wp[0] - start[0], wp[1] - start[1]) + math.hypot(
                    final_target[0] - wp[0], final_target[1] - wp[1]
                )
                feasible.append((total_cost, wp))

            if not feasible:
                return None

            feasible.sort(key=lambda item: item[0])
            return feasible[0][1]

        if self.detour_target is not None:
            if math.hypot(self.detour_target[0] - self.position[0], self.detour_target[1] - self.position[1]) <= 0.2:
                self.detour_target = None

        active_target = self.detour_target if self.detour_target is not None else target

        x, y = self.position
        tx, ty = active_target
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        self._emit_event(sim_state, "movement_started", {"origin": self.position, "destination": active_target, "distance": round(dist, 3)})
        if dist < 0.01:
            self._emit_event(sim_state, "movement_arrived", {"destination": active_target, "distance": round(dist, 3)})
            return

        angle = math.atan2(dy, dx)
        self.orientation = angle
        step = min(self.speed * dt, dist)
        new_x = x + math.cos(angle) * step
        new_y = y + math.sin(angle) * step

        if can_occupy((new_x, new_y)):
            self.position = (new_x, new_y)
            self._emit_event(sim_state, "movement_progress", {"destination": active_target, "remaining_distance": round(max(0.0, dist-step), 3)})
        else:
            if self.detour_target is None:
                detour = compute_detour_waypoint(self.position, target)
                if detour is not None:
                    self.detour_target = detour
                    self.activity_log.append(f"Detouring around obstacle via {detour}")
                else:
                    self.activity_log.append(f"Blocked while moving toward {target}")
                    self._emit_event(sim_state, "movement_blocked", {"destination": target, "blocker_category": "path_blocked"})
                    if self.current_inspect_target_id:
                        self.inspect_stall_counts[self.current_inspect_target_id] = self.inspect_stall_counts.get(self.current_inspect_target_id, 0) + 1
                        self._set_status(
                            f"Inspect stalled for {self.current_inspect_target_id}; stall_count={self.inspect_stall_counts[self.current_inspect_target_id]}"
                        )
            else:
                self.activity_log.append(f"Blocked while moving toward {active_target}")
                self._emit_event(sim_state, "movement_blocked", {"destination": active_target, "blocker_category": "path_blocked"})
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

    def update_knowledge(self, environment, full_packet_sweep=True):
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
                    self.absorb_packet(packet_content, accuracy=0.95)
                    after = len(self.mental_model["information"])
                    if after > before:
                        self.source_inspection_state[packet_name] = "inspected"
                        self._set_status(f"Legacy sweep ingested packet from {packet_name}")
                # Deliberately suppress per-tick access-failure spam from legacy sweep.

        elif self.current_inspect_target_id:
            self._inspect_source(environment, self.current_inspect_target_id)

        if "mismatch with construction" in " ".join(self.activity_log[-6:]).lower() and self.current_inspect_target_id:
            self.mark_source_revisitable(self.current_inspect_target_id, reason="construction_mismatch")

        self._apply_task_derivations(sim_state=None)

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

    def update(self, dt, environment, sim_state=None):
        self.update_physiology(exertion=0.5)
        self.update_knowledge(environment, full_packet_sweep=(sim_state is None))
        self._apply_task_derivations(sim_state=sim_state)
        if sim_state is None:
            # Legacy compatibility path for unit-level agent calls outside full simulation state.
            self._run_goal_management_pipeline(dt, environment)
        else:
            self.perceive_environment(sim_state)
            self.sim_step_count += 1
            self._check_inflight_timeout(sim_state)
            self._poll_planner_request(sim_state, environment)
            if self.active_actions:
                self._advance_active_actions(dt)
            else:
                trigger_reason = self._plan_trigger_reason(sim_state, environment)
                planner_allowed, planner_reason = self._planner_decision_allowed(sim_state, trigger_reason)
                if planner_allowed:
                    cooldown_remaining = self._planner_cooldown_remaining(sim_state)
                    if cooldown_remaining > 0.0:
                        self.planner_state["total_skipped_cooldown"] += 1
                        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"config": sim_state.brain_backend_config, "configured_backend": sim_state.configured_brain_backend}
                        self._emit_event(sim_state, "planner_request_skipped_cooldown", {"reason": planner_reason, "cooldown_remaining": cooldown_remaining, "consecutive_failures": self.planner_state["consecutive_failures"], "backend": runtime.get("configured_backend", sim_state.configured_brain_backend), "model": runtime["config"].local_model})
                    elif self.planner_state.get("status") == "in_flight":
                        self.planner_state["total_skipped_inflight"] += 1
                        runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"configured_backend": sim_state.configured_brain_backend, "effective_backend": sim_state.effective_brain_backend, "provider": sim_state.brain_provider}
                        self._emit_event(sim_state, "planner_request_skipped_inflight", {"reason": planner_reason, "request_id": self.planner_state.get("request_id"), "backend": runtime.get("configured_backend", sim_state.configured_brain_backend)})
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
                else:
                    decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="no active cached plan while planner cadence skips", confidence=1.0)
                    self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment)
                    self.planner_state["ui_safe_fallback_count"] += 1
                    runtime = sim_state.get_agent_brain_runtime(self) if hasattr(sim_state, "get_agent_brain_runtime") else {"configured_backend": sim_state.configured_brain_backend}
                    self._emit_event(sim_state, "ui_safe_fallback_used", {"reason": planner_reason, "request_state": self.planner_state.get("status"), "backend": runtime.get("configured_backend", sim_state.configured_brain_backend)})
                    sim_state.logger.log_event(sim_state.time, "planner_skipped_without_plan", {"agent": self.name, "reason": planner_reason})
                self._advance_active_actions(dt)

            self._apply_externalization_and_construction_effects(environment, sim_state, dt)
            self._update_goal_states_from_runtime(sim_state, environment)

        if self.target:
            self.move_toward(self.target, dt, environment, sim_state=sim_state)


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

            if action["type"] == "construct" and action["progress"] == 0:
                project_id = action.get("project_id") or "Build_Table_B"
                project = environment.construction.projects.get(project_id)
                if project:
                    decision_action = action.get("decision_action")
                    legacy_direct = decision_action is None
                    if not legacy_direct and self.inventory_resources.get("bricks", 0) <= 0:
                        self.activity_log.append("Construction paused: missing transported bricks")
                        sim_state.logger.log_event(sim_state.time, "construction_waiting_for_logistics", {"agent": self.name, "project_id": project_id})
                        continue

                    if not legacy_direct:
                        self.inventory_resources["bricks"] = max(0, self.inventory_resources.get("bricks", 0) - 1)
                    environment.construction.assign_builder(project_id, self.name)
                    environment.construction.deliver_resource(project_id, "bricks", quantity=1)

                    fidelity = max(self._hook_value("construction_fidelity", "start_construction", "fidelity_score", default=0.5), self._trait_value("rule_accuracy"))
                    if random.random() > fidelity:
                        project["correct"] = False

                    if decision_action == ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value:
                        project["correct"] = True
                    sim_state.team_knowledge_manager.upsert_construction_artifact(project, sim_state.time)
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

    def update_active_actions(self, dt):
        """Deprecated wrapper: use `_advance_active_actions(...)` in live path."""
        self._advance_active_actions(dt)

    def _advance_active_actions(self, dt):
        completed = []

        for action in self.active_actions:
            action["progress"] += dt
            if action["progress"] >= action["duration"]:
                completed.append(action)

        for action in completed:
            self.perform_action([action])
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
                })
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
            known_rules = self.mental_model["knowledge"].rules
            if not known_rules:
                continue
            rule_matches = False
            for rule in known_rules:
                if rule in project.get("expected_rules", []):
                    rule_matches = True
                    break
            if not rule_matches:
                mismatch_sensitivity = max(self._hook_value("validation_check", "detect_mismatch", "sensitivity", default=0.5), self._trait_value("rule_accuracy"))
                mismatch_detect_prob = min(1.0, 0.25 + 0.7 * mismatch_sensitivity)
                if random.random() <= mismatch_detect_prob:
                    self.activity_log.append(f"Disagrees with approach for {project.get('name', 'Unknown')}")
                    if sim_state is not None:
                        sim_state.logger.log_event(
                            sim_state.time,
                            "construction_mismatch_detected",
                            {"agent": self.name, "project_id": project.get("id", "unknown")},
                        )
                    if random.random() < self._trait_value("help_tendency"):
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
