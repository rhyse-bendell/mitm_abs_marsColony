# File: modules/agent.py

import math
import random
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
class PlanRecord:
    plan_id: str
    decision: BrainDecision
    created_at: float
    last_reviewed_at: float
    trigger_reason: str
    remaining_executions: int = 2
    invalidation_reason: str | None = None
    ordered_goals: list[dict] | None = None
    ordered_actions: list[dict] | None = None
    explanation: str | None = None


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
        self.goal_stack = []  # A list of goals (most important is first)
        self.goal = None  # Synchronized with top of stack
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

    def _resolve_inspect_target(self, decision, environment):
        self._ensure_source_state(environment)
        explicit_target = decision.target_id
        if explicit_target:
            point = environment.get_interaction_target_position(explicit_target, from_position=self.position)
            if point is not None:
                self._set_status(f"Inspect target selected: {explicit_target}")
                return explicit_target, point

        candidates = self._candidate_information_sources(environment)
        if not candidates:
            self._set_status("Inspect target resolution failed: no accessible information sources")
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

    def current_goal(self):
        return self.goal_stack[-1] if self.goal_stack else None

    def update_current_goal(self):
        self.goal = self.goal_stack[-1]["goal"] if self.goal_stack else None

    def push_goal(self, goal, target=None):
        self.goal_stack.append({"goal": goal, "target": target})
        self.goal = goal  # Keep synchronized
        self.activity_log.append(f"Pushed goal: {goal}")

    def pop_goal(self):
        if self.goal_stack:
            completed = self.goal_stack.pop()
            self.activity_log.append(f"Completed goal: {completed['goal']}")
            self.goal = self.goal_stack[-1]["goal"] if self.goal_stack else None  # Sync again

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

    def _build_rule_based_brain_decision(self, sim_state, trigger_reason):
        context = sim_state.brain_context_builder.build(sim_state, self)
        provider_name = sim_state.brain_provider.__class__.__name__
        request_explanation = self._should_request_explanation()
        request_packet = self._build_brain_request(sim_state, context, request_explanation, trigger_reason)
        sim_state.logger.log_event(sim_state.time, "planner_call_triggered", {"agent": self.name, "reason": trigger_reason, "backend": provider_name})
        sim_state.logger.log_event(
            sim_state.time,
            "brain_decision_query",
            {
                "agent": self.name,
                "trigger_reason": trigger_reason,
                "plan_id": getattr(self.current_plan, "plan_id", None),
                "provider": provider_name,
                "request_explanation": request_explanation,
                "context_meta": {
                    "affordance_count": len(context.action_affordances),
                    "known_gaps": len(context.individual_cognitive_state.get("known_gaps", [])),
                    "build_status": context.individual_cognitive_state.get("build_readiness", {}).get("status"),
                },
            },
        )

        response = None
        provider_decide = getattr(sim_state.brain_provider, "decide", None)
        if callable(provider_decide) and sim_state.brain_provider.__class__.__name__ == "RuleBrain":
            legacy_decision = provider_decide(context)
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"legacy-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": f"legacy-plan-{request_packet.request_id}",
                        "plan_horizon": 1,
                        "ordered_goals": ([{"goal_id": legacy_decision.goal_update, "description": legacy_decision.goal_update, "priority": 0.8, "status": "active"}] if legacy_decision.goal_update else []),
                        "ordered_actions": [
                            {
                                "step_index": 0,
                                "action_type": legacy_decision.selected_action.value,
                                "target_id": legacy_decision.target_id,
                                "target_zone": legacy_decision.target_zone,
                                "expected_purpose": legacy_decision.reason_summary,
                            }
                        ],
                        "next_action": {
                            "step_index": 0,
                            "action_type": legacy_decision.selected_action.value,
                            "target_id": legacy_decision.target_id,
                            "target_zone": legacy_decision.target_zone,
                            "expected_purpose": legacy_decision.reason_summary,
                        },
                        "plan_method_id": legacy_decision.plan_method_id,
                        "confidence": legacy_decision.confidence,
                    },
                    "explanation": legacy_decision.reason_summary if request_explanation else None,
                }
            )
        elif hasattr(sim_state.brain_provider, "generate_plan"):
            response = sim_state.brain_provider.generate_plan(request_packet)
        if response is None:
            legacy_decision = sim_state.brain_provider.decide(context)
            response = AgentBrainResponse.from_dict(
                {
                    "response_id": f"legacy-{request_packet.request_id}",
                    "agent_id": self.agent_id,
                    "plan": {
                        "plan_id": f"legacy-plan-{request_packet.request_id}",
                        "plan_horizon": 1,
                        "ordered_goals": [],
                        "ordered_actions": [
                            {
                                "step_index": 0,
                                "action_type": legacy_decision.selected_action.value,
                                "target_id": legacy_decision.target_id,
                                "target_zone": legacy_decision.target_zone,
                                "expected_purpose": legacy_decision.reason_summary,
                            }
                        ],
                        "next_action": {
                            "step_index": 0,
                            "action_type": legacy_decision.selected_action.value,
                            "target_id": legacy_decision.target_id,
                            "target_zone": legacy_decision.target_zone,
                            "expected_purpose": legacy_decision.reason_summary,
                        },
                        "plan_method_id": legacy_decision.plan_method_id,
                        "confidence": legacy_decision.confidence,
                    },
                }
            )

        self.planner_call_count += 1
        self.last_planner_step = self.sim_step_count
        self.last_planner_time = sim_state.time

        legal_action_ids = [a["action_type"] for a in context.action_affordances]
        errors = validate_agent_brain_response(response, legal_action_ids)
        repaired = False
        if errors:
            repaired = True
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
        status = "repaired" if repaired else "accepted"
        if legacy_errors:
            status = "rejected"
            decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="Fallback due to decision validation failure.", confidence=1.0)

        sim_state.logger.log_event(
            sim_state.time,
            "brain_decision_outcome",
            {
                "agent": self.name,
                "trigger_reason": trigger_reason,
                "provider": provider_name,
                "decision_status": status,
                "selected_action": decision.selected_action.value,
                "confidence": decision.confidence,
                "errors": errors + legacy_errors,
                "request_explanation": request_explanation,
                "explanation_present": bool(response.explanation),
                "planner_call_count": self.planner_call_count,
            },
        )

        return decision, status, response

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

        steps_since = self.sim_step_count - self.last_planner_step if self.last_planner_step >= 0 else self.sim_step_count
        time_since = sim_state.time - self.last_planner_time if self.last_planner_time >= 0 else sim_state.time
        step_due = steps_since >= cfg.planner_interval_steps
        time_due = time_since >= cfg.planner_interval_time
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
            for g in response.plan.ordered_goals[:3]:
                if not self.goal_stack or self.goal_stack[-1].get("goal") != g.description:
                    self.goal_stack.append({"goal": g.description, "target": decision.target_id, "goal_id": g.goal_id})
            if len(self.goal_stack) > 5:
                self.goal_stack = self.goal_stack[-5:]
            self.update_current_goal()
        elif decision.goal_update:
            if not self.goal_stack or self.goal_stack[-1].get("goal") != decision.goal_update:
                self.goal_stack.append({"goal": decision.goal_update, "target": decision.target_id})
                if len(self.goal_stack) > 5:
                    self.goal_stack = self.goal_stack[-5:]
            self.update_current_goal()

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
            },
        )

    def _adopt_new_plan(self, decision, trigger_reason, sim_time, response=None):
        if self.current_plan is not None and self.current_plan.invalidation_reason is None:
            self.current_plan.invalidation_reason = f"replaced_by_{trigger_reason}"
        self.current_plan = PlanRecord(
            plan_id=getattr(getattr(response, "plan", None), "plan_id", self._next_plan_id()),
            decision=decision,
            created_at=sim_time,
            last_reviewed_at=sim_time,
            trigger_reason=trigger_reason,
            remaining_executions=max(12, int(getattr(getattr(response, "plan", None), "plan_horizon", 2)) + 3),
            ordered_goals=[g.__dict__ for g in getattr(getattr(response, "plan", None), "ordered_goals", [])],
            ordered_actions=[a.__dict__ for a in getattr(getattr(response, "plan", None), "ordered_actions", [])],
            explanation=getattr(response, "explanation", None),
        )

    def _continue_cached_plan(self, sim_state, environment):
        if self.current_plan is None:
            return False
        if self.current_plan.remaining_executions <= 0:
            self.current_plan.invalidation_reason = "plan_completed"
            return False

        self.current_action = self._translate_brain_decision_to_legacy_action(self.current_plan.decision, environment)
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
        return True
    def _translate_brain_decision_to_legacy_action(self, decision, environment):
        if self.task_model is not None:
            enabled = set(self.task_model.enabled_actions_for_role(self.role))
            if decision.selected_action.value not in enabled:
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
            source_id, interaction_target = self._resolve_inspect_target(decision, environment)
            if source_id is None or interaction_target is None:
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

        goal = self.goal_stack[-1]["goal"]

        if goal == "seek_info":
            target = self.goal_stack[-1].get("target") or self.target or (7.0, 6.4)
            return [{"type": "move_to", "target": target, "duration": 1.0, "priority": 1}]
        if goal == "share":
            return [{"type": "communicate", "duration": 0.5, "priority": 1}]
        if goal == "build":
            goal_target = self.goal_stack[-1].get("target")
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

    def move_toward(self, target, dt, environment):
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
        if dist < 0.01:
            return

        angle = math.atan2(dy, dx)
        self.orientation = angle
        step = min(self.speed * dt, dist)
        new_x = x + math.cos(angle) * step
        new_y = y + math.sin(angle) * step

        if can_occupy((new_x, new_y)):
            self.position = (new_x, new_y)
        else:
            if self.detour_target is None:
                detour = compute_detour_waypoint(self.position, target)
                if detour is not None:
                    self.detour_target = detour
                    self.activity_log.append(f"Detouring around obstacle via {detour}")
                else:
                    self.activity_log.append(f"Blocked while moving toward {target}")
                    if self.current_inspect_target_id:
                        self.inspect_stall_counts[self.current_inspect_target_id] = self.inspect_stall_counts.get(self.current_inspect_target_id, 0) + 1
                        self._set_status(
                            f"Inspect stalled for {self.current_inspect_target_id}; stall_count={self.inspect_stall_counts[self.current_inspect_target_id]}"
                        )
            else:
                self.activity_log.append(f"Blocked while moving toward {active_target}")
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
            if self.active_actions:
                self._advance_active_actions(dt)
            else:
                trigger_reason = self._plan_trigger_reason(sim_state, environment)
                planner_allowed, planner_reason = self._planner_decision_allowed(sim_state, trigger_reason)
                if planner_allowed:
                    decision, _status, response = self._build_rule_based_brain_decision(sim_state, planner_reason)
                    self._adopt_new_plan(decision, planner_reason, sim_state.time, response=response)
                    self._refresh_goal_plan_state(decision, sim_state, planner_reason, response=response)
                    self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment)
                elif self._continue_cached_plan(sim_state, environment):
                    sim_state.logger.log_event(sim_state.time, "planner_skipped_due_to_cadence", {"agent": self.name, "reason": planner_reason})
                else:
                    decision = BrainDecision(selected_action=ExecutableActionType.WAIT, reason_summary="no active cached plan while planner cadence skips", confidence=1.0)
                    self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment)
                    sim_state.logger.log_event(sim_state.time, "planner_skipped_without_plan", {"agent": self.name, "reason": planner_reason})
                self._advance_active_actions(dt)

            self._apply_externalization_and_construction_effects(environment, sim_state, dt)

        if self.target:
            self.move_toward(self.target, dt, environment)


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
