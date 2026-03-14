# File: modules/agent.py

import math
import random
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


class Agent:
    def __init__(self, name, role, position=(0.0, 0.0), orientation=0.0, speed=1.0):
        self.name = name
        self.role = role
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
        self.history_compaction = {
            "recent_history_summary": "",
            "plan_evolution": [],
            "compaction_count": 0,
        }

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


    def _build_readiness_score(self):
        info_count = len(self.mental_model["information"])
        knowledge_count = len(self.mental_model["knowledge"].rules)
        return info_count + (2 * knowledge_count)

    def _is_build_eligible(self):
        # Lightweight threshold: build should not start from a trivial information fragment.
        return self._build_readiness_score() >= 3

    def _select_build_target(self, environment):
        return environment.get_interaction_target_position("Build_Table_B", from_position=self.position)

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

    def _build_rule_based_brain_decision(self, sim_state, trigger_reason):
        context = sim_state.brain_context_builder.build(sim_state, self)
        provider_name = sim_state.brain_provider.__class__.__name__
        sim_state.logger.log_event(
            sim_state.time,
            "brain_decision_query",
            {
                "agent": self.name,
                "trigger_reason": trigger_reason,
                "plan_id": getattr(self.current_plan, "plan_id", None),
                "provider": provider_name,
                "context_meta": {
                    "affordance_count": len(context.action_affordances),
                    "known_gaps": len(context.individual_cognitive_state.get("known_gaps", [])),
                    "build_status": context.individual_cognitive_state.get("build_readiness", {}).get("status"),
                },
            },
        )
        decision = sim_state.brain_provider.decide(context)
        provider_outcome = getattr(sim_state.brain_provider, "last_outcome", None)
        if provider_outcome and provider_outcome.get("fallback"):
            sim_state.logger.log_event(
                sim_state.time,
                "brain_provider_fallback",
                {
                    "agent": self.name,
                    "provider": provider_name,
                    "fallback_provider": "RuleBrain",
                    "reason": provider_outcome.get("reason"),
                    "latency_ms": provider_outcome.get("latency_ms"),
                },
            )
        legal_actions = [ExecutableActionType(a["action_type"]) for a in context.action_affordances]

        repaired = False
        if not (0.0 <= decision.confidence <= 1.0):
            decision.confidence = max(0.0, min(1.0, decision.confidence))
            repaired = True

        errors = validate_brain_decision(decision, legal_actions)
        if errors and decision.selected_action != ExecutableActionType.WAIT and ExecutableActionType.WAIT in legal_actions:
            repaired = True
            decision = BrainDecision(
                selected_action=ExecutableActionType.WAIT,
                reason_summary="Repaired to WAIT due to validation failure.",
                confidence=1.0,
                assumptions=["simulator legality gate"],
            )
            errors = validate_brain_decision(decision, legal_actions)

        status = "accepted"
        if errors:
            status = "rejected"
            decision = BrainDecision(
                selected_action=ExecutableActionType.WAIT,
                reason_summary="Fallback due to unrecoverable decision validation failure.",
                confidence=1.0,
            )
        elif repaired:
            status = "repaired"

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
                "errors": errors,
                "validation_repaired": repaired,
                "validation_fallback_to_wait": decision.selected_action == ExecutableActionType.WAIT and bool(errors),
            },
        )
        return decision, status

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
        info_count = len(self.mental_model["information"])
        knowledge_count = len(self.mental_model["knowledge"].rules)
        build_readiness = self._build_readiness_score()
        team_updates = len(sim_state.team_knowledge_manager.recent_updates)
        recent_log = " ".join(self.activity_log[-4:]).lower()

        if self.current_plan is None:
            reason = "no_active_plan"
        elif self.current_plan.remaining_executions <= 0:
            reason = "plan_completed"
        elif self.current_plan.invalidation_reason:
            reason = "plan_invalidated"
        elif phase_name != self.last_phase_name and self.last_phase_name is not None:
            reason = "phase_transition"
        elif info_count > self.last_info_count or knowledge_count > self.last_knowledge_count:
            reason = "new_dik_acquired"
        elif "mismatch with construction" in recent_log:
            reason = "contradiction_detected"
        elif "blocked while moving" in recent_log:
            reason = "path_blocked_or_stalled"
        elif team_updates > self.last_team_update_count:
            reason = "communication_update_received"
        elif build_readiness != self.last_build_readiness:
            reason = "build_readiness_changed"
        elif self.current_plan and (now - self.current_plan.created_at) >= self.plan_expiry_s:
            reason = "plan_invalidated"
        elif self.current_plan and (now - self.current_plan.last_reviewed_at) >= self.plan_review_interval_s:
            reason = "periodic_reassessment"
        else:
            reason = None

        self.last_phase_name = phase_name
        self.last_info_count = info_count
        self.last_knowledge_count = knowledge_count
        self.last_build_readiness = build_readiness
        self.last_team_update_count = team_updates
        return reason

    def _adopt_new_plan(self, decision, trigger_reason, sim_time):
        if self.current_plan is not None and self.current_plan.invalidation_reason is None:
            self.current_plan.invalidation_reason = f"replaced_by_{trigger_reason}"
        self.current_plan = PlanRecord(
            plan_id=self._next_plan_id(),
            decision=decision,
            created_at=sim_time,
            last_reviewed_at=sim_time,
            trigger_reason=trigger_reason,
            remaining_executions=2,
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

        if decision.target_id:
            interaction_target = environment.get_interaction_target_position(decision.target_id, from_position=self.position)
            if interaction_target is not None:
                action["target"] = interaction_target
        if decision.selected_action == ExecutableActionType.TRANSPORT_RESOURCES:
            action["duration"] = 30.0
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
            if self._is_build_eligible():
                build_target = self._select_build_target(environment)
                if build_target is not None:
                    self.push_goal("build", build_target)
                else:
                    self.activity_log.append("No accessible build interaction target found; gathering more info")
                    self.push_goal("seek_info", self._choose_info_target(environment))
            else:
                self.activity_log.append(
                    f"Build deferred (readiness={self._build_readiness_score()}); continuing information gathering"
                )
                self.push_goal("seek_info", self._choose_info_target(environment))

        elif goal == "build":
            if self.target is None:
                self.target = self._select_build_target(environment)

            if self.mental_model["knowledge"].rules:
                self.activity_log.append("Building task engaged")
            elif not self._is_build_eligible():
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
            return [{"type": "construct", "duration": 2.0, "priority": 1}]

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
            elif action["type"] == "communicate":
                self.has_shared = True
                self.activity_log.append("Shared with teammates")
            elif action["type"] == "construct":
                self.activity_log.append("Building...")
            elif action["type"] == "idle":
                self.activity_log.append("Idling...")

    def absorb_packet(self, packet, accuracy=1.0):
        for d in packet.get("data", []):
            if random.random() <= accuracy:
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

        for info in packet.get("information", []):
            if random.random() <= accuracy:
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
            else:
                self.activity_log.append(f"Blocked while moving toward {active_target}")

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

    def update_knowledge(self, environment):
        for packet_name, packet_content in environment.knowledge_packets.items():
            if packet_name in self.mental_model["information"]:
                continue
            if not self._has_packet_access(packet_name):
                continue
            if self.role not in packet_name and "Team" not in packet_name:
                continue
            if environment.can_access_info(self.position, packet_name):
                before = len(self.mental_model["information"])
                self.absorb_packet(packet_content, accuracy=0.95)
                after = len(self.mental_model["information"])
                if after > before:
                    self.activity_log.append(f"Ingested packet from {packet_name}")
                else:
                    self.activity_log.append(f"Attempted access to {packet_name} but absorbed no new info")
            else:
                self.activity_log.append(f"Could not access packet {packet_name} (too far or unauthorized)")

        known_info = list(self.mental_model["information"])
        if known_info:
            from itertools import combinations
            candidate_tags = {}
            for info in known_info:
                for tag in info.tags:
                    candidate_tags.setdefault(tag, []).append(info)
            for tag, group in candidate_tags.items():
                if len(group) >= 2:
                    if random.random() < 0.9:
                        self.mental_model["knowledge"].try_infer_rules(group)
                        self.activity_log.append(f"Inferred rule from tag [{tag}]")

    def decide_next_action(self, environment):
        """Deprecated compatibility wrapper for legacy callers."""
        self._evaluate_goal_state(environment)
        self.current_action = self._plan_actions_for_current_goal()

    def update(self, dt, environment, sim_state=None):
        self.update_physiology(exertion=0.5)
        self.update_knowledge(environment)
        if sim_state is None:
            # Legacy compatibility path for unit-level agent calls outside full simulation state.
            self._run_goal_management_pipeline(dt, environment)
        else:
            self.perceive_environment(sim_state)
            if self.active_actions:
                self._advance_active_actions(dt)
            else:
                trigger_reason = self._plan_trigger_reason(sim_state, environment)
                if trigger_reason is None and self._continue_cached_plan(sim_state, environment):
                    pass
                else:
                    if trigger_reason is None:
                        trigger_reason = "no_active_plan"
                    decision, _status = self._build_rule_based_brain_decision(sim_state, trigger_reason)
                    self._adopt_new_plan(decision, trigger_reason, sim_state.time)
                    self.current_action = self._translate_brain_decision_to_legacy_action(decision, environment)
                self._advance_active_actions(dt)

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
                    self.communicate_with(agent)

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
                    "priority": action.get("priority", 1)
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

    def communicate_with(self, other_agent):
        messages = self.generate_message()
        for msg in messages:
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


    def compare_and_repair_construction(self, construction):
        for project in construction.projects:
            if not isinstance(project, dict):
                continue
            if not project.get("in_progress", False):
                continue
            rule_matches = False
            for rule in self.mental_model["knowledge"].rules:
                if rule in project.get("expected_rules", []):
                    rule_matches = True
                    break
            if not rule_matches:
                self.activity_log.append(f"Disagrees with approach for {project.get('name', 'Unknown')}")
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
