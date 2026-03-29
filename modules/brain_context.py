from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from modules.action_schema import ExecutableActionType


@dataclass
class BrainContextPacket:
    static_task_context: Dict[str, Any]
    world_snapshot: Dict[str, Any]
    individual_cognitive_state: Dict[str, Any]
    team_state: Dict[str, Any]
    history_bands: Dict[str, Any]
    action_affordances: List[Dict[str, Any]] = field(default_factory=list)


class BrainContextBuilder:
    def __init__(self, scenario_name: str = "Minerva Mars Colony"):
        self.scenario_name = scenario_name

    def _summarize_structures(self, environment) -> List[Dict[str, Any]]:
        build_targets = {
            name: t for name, t in environment.interaction_targets.items() if t.get("kind") == "build"
        }
        projects_by_id = {p.get("id"): p for p in environment.construction.projects.values() if isinstance(p, dict)}
        summaries: List[Dict[str, Any]] = []

        for target_id, target in build_targets.items():
            object_id = target.get("object")
            project = projects_by_id.get(target_id)
            state = "absent"
            usable = False
            progress = 0.0
            validated_correctness = None
            needs_repair = False
            overloaded = False

            if project:
                raw_status = project.get("status")
                if raw_status == "complete":
                    state = "built"
                    usable = bool(project.get("correct", True))
                else:
                    state = "in_progress"
                    usable = False

                required = project.get("required_resources", {}).get("bricks", 0)
                delivered = project.get("delivered_resources", {}).get("bricks", 0)
                progress = min(1.0, delivered / required) if required else 0.0
                validated_correctness = bool(project.get("correct", True))
                needs_repair = project.get("correct", True) is False
                overloaded = bool(project.get("overloaded", False))

            summaries.append(
                {
                    "structure_id": target_id,
                    "object_id": object_id,
                    "zone": target.get("zone"),
                    "structure_type": (project or {}).get("type", "unknown"),
                    "state": state,
                    "validated_correctness": validated_correctness,
                    "usable": usable,
                    "progress": round(progress, 2),
                    "needs_repair": needs_repair,
                    "overloaded": overloaded,
                    "functional_connections": [f"zone_link:{target.get('zone')}"] if target.get("zone") else [],
                }
            )

        return summaries

    def _phase_profile(self, environment) -> Dict[str, Any]:
        current_phase = environment.get_current_phase() or {"name": "default"}
        if not environment.phases:
            elapsed = float(environment.get_time())
            stage = "early" if elapsed < 40 else ("execution" if elapsed < 90 else "late")
            return {"name": current_phase.get("name", "default"), "stage": stage, "progress": min(1.0, elapsed / 120.0)}

        elapsed = float(environment.get_time())
        cursor = 0.0
        for idx, phase in enumerate(environment.phases):
            duration = float(phase.get("duration_minutes", 0.0) * 60.0)
            phase_start = cursor
            phase_end = phase_start + duration
            if idx == environment.current_phase_index:
                progress = 0.0 if duration <= 0 else max(0.0, min(1.0, (elapsed - phase_start) / duration))
                stage = "early" if progress < 0.35 else ("execution" if progress < 0.8 else "late")
                return {"name": phase.get("name", "default"), "stage": stage, "progress": progress}
            cursor = phase_end

        return {"name": current_phase.get("name", "default"), "stage": "late", "progress": 1.0}

    def _build_readiness(self, agent, structure_summary: List[Dict[str, Any]], environment, team_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
        info_count = len(agent.mental_model["information"])
        knowledge_count = len(agent.mental_model["knowledge"].rules)
        inspected_sources = sum(1 for state in getattr(agent, "source_inspection_state", {}).values() if state == "inspected")
        available_build_targets = [
            target_name
            for target_name, target in environment.interaction_targets.items()
            if target.get("kind") == "build" and environment.get_interaction_target_position(target_name, from_position=agent.position) is not None
        ]
        has_team_artifact = bool((team_state or {}).get("externalized_artifacts"))
        score = info_count + (2 * knowledge_count) + inspected_sources + (1 if has_team_artifact else 0)
        in_progress = sum(1 for s in structure_summary if s["state"] == "in_progress")
        absent = sum(1 for s in structure_summary if s["state"] == "absent")
        blockers = []

        if info_count < 2:
            blockers.append("insufficient_information_inspection")
        if knowledge_count < 1:
            blockers.append("insufficient_rule_knowledge")
        if inspected_sources < 1:
            blockers.append("no_inspected_information_source")
        if not available_build_targets:
            blockers.append("no_navigable_build_target")

        if score < 4 or blockers:
            status = "premature"
        elif in_progress > 0:
            status = "plausible"
        elif absent == 0:
            status = "blocked"
            blockers = blockers + ["no_remaining_absent_structure_targets"]
        else:
            status = "plausible"

        return {
            "status": status,
            "score": score,
            "blockers": blockers,
            "ready_for_build": status == "plausible",
            "inspected_sources": inspected_sources,
            "build_targets_available": len(available_build_targets),
        }

    def _affordances(self, agent, environment, build_readiness: Dict[str, Any] | None = None, phase_profile: Dict[str, Any] | None = None, team_state: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        if phase_profile is None:
            phase_profile = self._phase_profile(environment)
        if build_readiness is None:
            build_readiness = self._build_readiness(agent, self._summarize_structures(environment), environment, team_state=team_state or {})
        stage = phase_profile.get("stage", "execution")
        readiness_ok = bool(build_readiness.get("ready_for_build"))
        mismatch_pressure = any("mismatch" in e.lower() for e in (agent.activity_log[-6:] if agent.activity_log else []))
        has_artifacts = bool((team_state or {}).get("externalized_artifacts"))
        nearby_teammates = sum(
            1
            for other in getattr(environment, "agents", [])
            if other is not agent and ((agent.position[0] - other.position[0]) ** 2 + (agent.position[1] - other.position[1]) ** 2) ** 0.5 <= 20.0
        )
        teammate_help_signals = dict((team_state or {}).get("teammate_help_signals", {}))
        productive_coordination = bool(nearby_teammates > 0 and (agent.known_gaps or any(teammate_help_signals.values()) or has_artifacts))
        communication_no_effect_streak = int((getattr(agent, "communication_state", {}) or {}).get("no_effect_streak", 0) or 0)

        def utility_for(action: ExecutableActionType, target_kind: str | None = None):
            if action == ExecutableActionType.INSPECT_INFORMATION_SOURCE:
                return 0.95 if stage == "early" else (0.55 if not readiness_ok else 0.25)
            if action in {ExecutableActionType.COMMUNICATE, ExecutableActionType.REQUEST_ASSISTANCE}:
                base = 0.7 if productive_coordination else 0.08
                if stage == "execution":
                    base -= 0.15
                if communication_no_effect_streak > 0:
                    base -= min(0.5, 0.15 * communication_no_effect_streak)
                return max(0.01, base)
            if action == ExecutableActionType.EXTERNALIZE_PLAN:
                return 0.8 if stage in {"early", "execution"} else 0.45
            if action == ExecutableActionType.CONSULT_TEAM_ARTIFACT:
                return 0.6 if has_artifacts else 0.2
            if action == ExecutableActionType.TRANSPORT_RESOURCES:
                return 0.75 if readiness_ok and stage != "early" else 0.25
            if action in {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}:
                if not readiness_ok:
                    return 0.05
                return 0.85 if stage in {"execution", "late"} else 0.3
            if action in {ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, ExecutableActionType.VALIDATE_CONSTRUCTION}:
                return 0.9 if mismatch_pressure or stage == "late" or target_kind == "build" else 0.3
            if action in {ExecutableActionType.REASSESS_PLAN, ExecutableActionType.OBSERVE_ENVIRONMENT}:
                return 0.4 if stage == "early" else 0.3
            return 0.2

        legal = [
            {"action_type": ExecutableActionType.OBSERVE_ENVIRONMENT.value, "target_id": None, "target_class": "self", "utility": utility_for(ExecutableActionType.OBSERVE_ENVIRONMENT)},
            {"action_type": ExecutableActionType.REASSESS_PLAN.value, "target_id": None, "target_class": "self", "utility": utility_for(ExecutableActionType.REASSESS_PLAN)},
            {"action_type": ExecutableActionType.WAIT.value, "target_id": None, "target_class": "self", "utility": 0.1},
            {"action_type": ExecutableActionType.COMMUNICATE.value, "target_id": "nearby_agent", "target_class": "team", "utility": utility_for(ExecutableActionType.COMMUNICATE)},
            {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "target_id": "nearby_agent", "target_class": "team", "utility": utility_for(ExecutableActionType.REQUEST_ASSISTANCE)},
            {"action_type": ExecutableActionType.EXTERNALIZE_PLAN.value, "target_id": "whiteboard", "target_class": "artifact", "utility": utility_for(ExecutableActionType.EXTERNALIZE_PLAN)},
            {"action_type": ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, "target_id": "team_artifact", "target_class": "artifact", "utility": utility_for(ExecutableActionType.CONSULT_TEAM_ARTIFACT)},
            {"action_type": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "target_id": "active_construction", "target_class": "build", "utility": utility_for(ExecutableActionType.VALIDATE_CONSTRUCTION, target_kind="build")},
            {"action_type": ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value, "target_id": "active_construction", "target_class": "build", "utility": utility_for(ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, target_kind="build")},
        ]
        for item in legal:
            if item.get("action_type") in {ExecutableActionType.COMMUNICATE.value, ExecutableActionType.REQUEST_ASSISTANCE.value}:
                item["reachable"] = nearby_teammates > 0
                item["productive"] = productive_coordination
                item["no_effect_streak"] = communication_no_effect_streak

        for target_name, target in environment.interaction_targets.items():
            accessible_point = environment.get_interaction_target_position(target_name, from_position=agent.position)
            reachable = accessible_point is not None and environment._segment_is_navigable(agent.position, accessible_point)
            if accessible_point is None:
                continue
            action_type = (
                ExecutableActionType.INSPECT_INFORMATION_SOURCE
                if target.get("kind") == "information"
                else ExecutableActionType.START_CONSTRUCTION
            )
            source_state = None
            if target.get("kind") == "information":
                source_state = getattr(agent, "source_inspection_state", {}).get(target_name, "unseen")
                if source_state == "inspected":
                    continue
            legal.append(
                {
                    "action_type": action_type.value,
                    "target_id": target_name,
                    "target_zone": target.get("zone"),
                    "target_point": accessible_point,
                    "target_class": target.get("kind"),
                    "reachable": reachable,
                    "source_state": source_state,
                    "utility": utility_for(action_type, target_kind=target.get("kind")),
                }
            )

        legal.append(
            {
                "action_type": ExecutableActionType.TRANSPORT_RESOURCES.value,
                "target_id": "resource_zone_to_work_zone",
                "target_class": "logistics",
                "duration_s": 30.0,
                "utility": utility_for(ExecutableActionType.TRANSPORT_RESOURCES),
            }
        )

        task_model = getattr(agent, "task_model", None)
        if not task_model:
            return legal

        enabled_action_ids = set(task_model.enabled_actions_for_role(agent.role))
        filtered = []
        for affordance in legal:
            action_id = affordance.get("action_type")
            if action_id not in enabled_action_ids:
                continue
            target_id = affordance.get("target_id")
            if target_id in environment.interaction_targets:
                target_cfg = environment.interaction_targets[target_id]
                role_scope = [r.lower() for r in target_cfg.get("role_scope", [])]
                if role_scope and "all" not in role_scope and agent.role.lower() not in role_scope:
                    continue
            params = task_model.action_parameters.get(action_id)
            if params:
                affordance = dict(affordance)
                affordance["duration_s"] = params.duration_s
                affordance["task_parameters"] = dict(params.metadata)
            filtered.append(affordance)
        return filtered

    def _world_affordance_summary(self, agent, environment) -> Dict[str, Any]:
        physical_obstacles = [
            {"object_id": name, "kind": "blocked", "zone_corners": obj.get("corners")}
            for name, obj in environment.objects.items()
            if obj.get("type") == "blocked"
        ]

        information_sources = []
        work_targets = []
        unreachable_targets = []
        for target_name, target in environment.interaction_targets.items():
            point = environment.get_interaction_target_position(target_name, from_position=agent.position)
            target_summary = {
                "target_id": target_name,
                "zone": target.get("zone"),
                "interaction_point": point,
            }
            if target.get("kind") == "information":
                information_sources.append(target_summary)
            elif target.get("kind") == "build":
                work_targets.append(target_summary)

            if point is None:
                unreachable_targets.append({"target_id": target_name, "reason": "no_navigable_interaction_point"})

        return {
            "physical_obstacles": physical_obstacles,
            "information_sources": information_sources,
            "work_zones": work_targets,
            "resource_logistics": {
                "sources": [{"zone": "field_resources", "visible_count": len(environment.get_visible_resources(agent.position))}],
                "destinations": [{"zone": t.get("zone"), "target_id": name} for name, t in environment.interaction_targets.items() if t.get("kind") == "build"],
            },
            "unreachable_targets": unreachable_targets,
        }

    def build(self, sim_state, agent) -> BrainContextPacket:
        environment = sim_state.environment
        current_phase = environment.get_current_phase() or {"name": "default"}
        nearby_agents = [
            {
                "name": other.name,
                "role": other.role,
                "position": other.position,
            }
            for other in sim_state.agents
            if other.name != agent.name
        ]

        knowledge_summary = [str(rule) for rule in agent.mental_model["knowledge"].rules]
        information_summary = [getattr(info, "id", str(info)) for info in agent.mental_model["information"]]
        data_summary = [getattr(data, "id", str(data)) for data in agent.mental_model["data"]]

        history_bands = agent.history_bands() if hasattr(agent, "history_bands") else {
            "current_state_summary": f"goal={agent.goal} active_actions={len(agent.active_actions)}",
            "near_preceding_events": agent.activity_log[-8:],
            "recent_history_summary": "",
            "recent_plan_history": [],
        }
        history_events = history_bands.get("near_preceding_events", [])
        repeated_stalls = sum(1 for e in history_events if "blocked while moving" in e.lower())
        recent_plan_evolution = list(history_bands.get("recent_plan_history", []))

        phase_profile = self._phase_profile(environment)
        structure_summary = self._summarize_structures(environment)
        affordance_summary = self._world_affordance_summary(agent, environment)

        active_plan = getattr(agent, "current_plan", None)
        inspected_artifact_ids = {
            aid for aid in agent.memory_seen_packets if isinstance(aid, str) and aid in sim_state.team_knowledge_manager.artifacts
        }
        artifact_summaries = []
        for aid, artifact in sim_state.team_knowledge_manager.artifacts.items():
            artifact_summaries.append(
                {
                    "artifact_id": aid,
                    "type": artifact.artifact_type,
                    "summary": artifact.summary,
                    "author": artifact.author,
                    "contributors": list(getattr(artifact, "contributors", [])),
                    "knowledge_summary": list(getattr(artifact, "knowledge_summary", [])),
                    "validation_state": getattr(artifact, "validation_state", "unvalidated"),
                    "uptake_count": artifact.uptake_count,
                    "consulted_by": list(getattr(artifact, "consulted_by", [])),
                    "inspected_by_agent": aid in inspected_artifact_ids,
                    "adopted_by_agent": aid in inspected_artifact_ids,
                }
            )

        validated_plan_exists = any(a.artifact_type == "plan" for a in sim_state.team_knowledge_manager.artifacts.values())
        teammate_help_signals = {
            other.name: (
                bool(other.known_gaps)
                or "blocked" in (getattr(other, "status_last_action", "").lower())
                or "stalled" in (getattr(other, "status_last_action", "").lower())
            )
            for other in sim_state.agents
            if other.name != agent.name
        }

        task_model = getattr(sim_state, "task_model", None)
        team_state = {
            "team_shared_knowledge": sim_state.team_knowledge_manager.summarize(),
            "teammate_roles": {other.name: other.role for other in sim_state.agents if other.name != agent.name},
            "teammate_inferred_goals": {
                teammate: model.get("goals", []) for teammate, model in agent.theory_of_mind.items()
            },
            "tom_summary": agent.theory_of_mind,
            "recent_shared_updates": sim_state.team_knowledge_manager.recent_updates[-5:],
            "plan_readiness": "validated_shared_plan" if validated_plan_exists else "partial_or_fragmentary_plan",
            "externalized_artifacts": artifact_summaries,
            "teammate_help_signals": teammate_help_signals,
            "derivation_events": list(getattr(agent, "derivation_events", [])[-5:]),
            "task_rules": [r.rule_id for r in task_model.rules.values() if r.enabled] if task_model else [],
            "task_goals": [g.goal_id for g in task_model.goals.values() if g.enabled] if task_model else [],
            "task_plan_methods": [m.method_id for m in task_model.plan_methods.values() if m.enabled] if task_model else [],
        }

        build_readiness = self._build_readiness(agent, structure_summary, environment, team_state=team_state)
        action_affordances = self._affordances(agent, environment, build_readiness=build_readiness, phase_profile=phase_profile, team_state=team_state)

        static_task_context = {
            "mission": self.scenario_name,
            "current_phase": current_phase.get("name", "default"),
            "role": agent.role,
            "role_access_constraints": getattr(agent, "allowed_packet", []),
            "high_level_objectives": [g.label for g in task_model.goals.values() if g.enabled and g.goal_level in {"mission", "phase"}] if task_model else ["build required colony infrastructure", "maintain legal construction"],
            "hard_constraints": [
                "simulator validates legality",
                "simulator validates world truth",
                "simulator enforces phase restrictions",
            ],
        }

        world_snapshot = {
            "sim_time": sim_state.time,
            "phase_state": current_phase,
            "phase_profile": phase_profile,
            "agent_position": agent.position,
            "nearby_agents": nearby_agents,
            "built_state": structure_summary,
            "affordance_map": affordance_summary,
            "resource_status": {"visible_resources": environment.get_visible_resources(agent.position)},
            "legal_actions": action_affordances,
        }

        individual_cognitive_state = {
            "goal_stack": list(agent.goal_stack),
            "active_goal": agent.goal,
            "current_action": agent.current_action,
            "active_actions": list(agent.active_actions),
            "data_summary": data_summary,
            "information_summary": information_summary,
            "knowledge_summary": knowledge_summary,
            "known_gaps": list(agent.known_gaps),
            "certainty_signals": {
                "definitely_known": knowledge_summary[-5:],
                "uncertain_or_missing": sorted(set(list(agent.known_gaps) + (["validated_plan_missing"] if active_plan is None else []))),
                "information_gathering_value": "high" if (len(knowledge_summary) < 2 or build_readiness["status"] != "plausible") else "medium",
            },
            "build_readiness": build_readiness,
            "packets_inspected": list(agent.memory_seen_packets),
            "recent_failed_attempts": [
                e for e in history_events if "blocked" in e.lower() or "could not" in e.lower() or "mismatch" in e.lower()
            ],
            "traits": {
                "communication_propensity": getattr(agent, "communication_propensity", 0.5),
                "goal_alignment": getattr(agent, "goal_alignment", 0.5),
                "help_tendency": getattr(agent, "help_tendency", 0.5),
                "build_speed": getattr(agent, "build_speed", 0.5),
                "rule_accuracy": getattr(agent, "rule_accuracy", 0.5),
            },
            "active_plan": {
                "plan_id": getattr(active_plan, "plan_id", None),
                "created_at": getattr(active_plan, "created_at", None),
                "last_reviewed_at": getattr(active_plan, "last_reviewed_at", None),
                "invalidation_reason": getattr(active_plan, "invalidation_reason", None),
                "remaining_executions": getattr(active_plan, "remaining_executions", None),
                "plan_method_id": getattr(active_plan, "plan_method_id", None),
                "plan_method_status": getattr(active_plan, "plan_method_status", None),
                "validation_notes": list(getattr(active_plan, "validation_notes", []) or []),
                "associated_goal_ids": list(getattr(active_plan, "associated_goal_ids", []) or []),
            },
            "loop_counters": {
                "action_repeats": int(getattr(agent, "loop_counters", {}).get("action_repeats", 0) or 0),
                "plan_repeats": int(getattr(agent, "loop_counters", {}).get("plan_repeats", 0) or 0),
                "selected_action_repeats": int(getattr(agent, "selection_loop_guard", {}).get("consecutive_count", 0) or 0),
                "no_progress_streak": int((getattr(agent, "progress_tracker", {}) or {}).get("no_progress_streak", 0) or 0),
                "observe_no_effect": int((getattr(agent, "progress_tracker", {}) or {}).get("observe_no_effect", 0) or 0),
                "communication_no_effect": int((getattr(agent, "progress_tracker", {}) or {}).get("communication_no_effect", 0) or 0),
            },
            "progress_state": {
                "last_progress_time": (getattr(agent, "progress_tracker", {}) or {}).get("last_progress_time"),
                "last_progress_kind": (getattr(agent, "progress_tracker", {}) or {}).get("last_progress_kind"),
                "no_progress_streak": int((getattr(agent, "progress_tracker", {}) or {}).get("no_progress_streak", 0) or 0),
                "forced_pivot": (getattr(agent, "progress_tracker", {}) or {}).get("forced_pivot"),
                "forced_pivot_until": (getattr(agent, "progress_tracker", {}) or {}).get("forced_pivot_until"),
            },
            "seconds_since_dik_change": (
                None
                if float(getattr(agent, "last_dik_change_time", -1.0) or -1.0) < 0.0
                else max(0.0, float(sim_state.time) - float(getattr(agent, "last_dik_change_time", 0.0)))
            ),
            "control_state": dict(getattr(agent, "control_state", {}) or {}),
            "method_state": dict((getattr(agent, "control_state", {}) or {}).get("method_state") or {}),
            "planner_state": {
                "status": getattr(agent, "planner_state", {}).get("status"),
                "request_id": getattr(agent, "planner_state", {}).get("request_id"),
                "degraded_mode": bool(getattr(agent, "planner_state", {}).get("degraded_mode")),
            },
            "dik_integration_state": {
                "status": getattr(agent, "dik_integration_state", {}).get("status"),
                "last_completed_step": getattr(agent, "dik_integration_state", {}).get("last_completed_step"),
            },
            "inspect_state": {
                "current_target_id": getattr(agent, "current_inspect_target_id", None),
                "session_state": (getattr(agent, "inspect_session", {}) or {}).get("state"),
                "source_access_target": (getattr(agent, "source_access_state", {}) or {}).get("source_id"),
                "source_exhaustion": dict(getattr(agent, "source_exhaustion_state", {}) or {}),
                "source_inspection_state": dict(getattr(agent, "source_inspection_state", {}) or {}),
            },
        }

        history_bands["semantic_plan_evolution"] = {
            "recent_selected_actions": [e for e in recent_plan_evolution if ":" in e][:5],
            "invalidations": [e for e in recent_plan_evolution if "invalidated:" in e],
            "repeated_stalls": repeated_stalls,
            "unresolved_contradictions": [e for e in history_events if "mismatch with construction" in e.lower()],
        }

        return BrainContextPacket(
            static_task_context=static_task_context,
            world_snapshot=world_snapshot,
            individual_cognitive_state=individual_cognitive_state,
            team_state=team_state,
            history_bands=history_bands,
            action_affordances=action_affordances,
        )
