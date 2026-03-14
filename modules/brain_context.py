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

    def _build_readiness(self, agent, structure_summary: List[Dict[str, Any]]) -> Dict[str, Any]:
        info_count = len(agent.mental_model["information"])
        knowledge_count = len(agent.mental_model["knowledge"].rules)
        score = info_count + (2 * knowledge_count)
        in_progress = sum(1 for s in structure_summary if s["state"] == "in_progress")
        absent = sum(1 for s in structure_summary if s["state"] == "absent")

        if score < 3:
            status = "premature"
            blockers = ["insufficient_validated_dik"]
        elif in_progress > 0:
            status = "plausible"
            blockers = []
        elif absent == 0:
            status = "blocked"
            blockers = ["no_remaining_absent_structure_targets"]
        else:
            status = "plausible"
            blockers = []

        return {
            "status": status,
            "score": score,
            "blockers": blockers,
            "ready_for_build": status == "plausible",
        }

    def _affordances(self, agent, environment) -> List[Dict[str, Any]]:
        legal = [
            {"action_type": ExecutableActionType.OBSERVE_ENVIRONMENT.value, "target_id": None, "target_class": "self"},
            {"action_type": ExecutableActionType.REASSESS_PLAN.value, "target_id": None, "target_class": "self"},
            {"action_type": ExecutableActionType.WAIT.value, "target_id": None, "target_class": "self"},
            {"action_type": ExecutableActionType.COMMUNICATE.value, "target_id": "nearby_agent", "target_class": "team"},
            {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "target_id": "nearby_agent", "target_class": "team"},
            {"action_type": ExecutableActionType.EXTERNALIZE_PLAN.value, "target_id": "whiteboard", "target_class": "artifact"},
            {"action_type": ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, "target_id": "team_artifact", "target_class": "artifact"},
            {"action_type": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "target_id": "active_construction", "target_class": "build"},
            {"action_type": ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value, "target_id": "active_construction", "target_class": "build"},
        ]

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
                }
            )

        legal.append(
            {
                "action_type": ExecutableActionType.TRANSPORT_RESOURCES.value,
                "target_id": "resource_zone_to_work_zone",
                "target_class": "logistics",
                "duration_s": 30.0,
            }
        )
        return legal

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

        structure_summary = self._summarize_structures(environment)
        affordance_summary = self._world_affordance_summary(agent, environment)
        action_affordances = self._affordances(agent, environment)
        build_readiness = self._build_readiness(agent, structure_summary)

        static_task_context = {
            "mission": self.scenario_name,
            "current_phase": current_phase.get("name", "default"),
            "role": agent.role,
            "role_access_constraints": getattr(agent, "allowed_packet", []),
            "high_level_objectives": ["build required colony infrastructure", "maintain legal construction"],
            "hard_constraints": [
                "simulator validates legality",
                "simulator validates world truth",
                "simulator enforces phase restrictions",
            ],
        }

        world_snapshot = {
            "sim_time": sim_state.time,
            "phase_state": current_phase,
            "agent_position": agent.position,
            "nearby_agents": nearby_agents,
            "built_state": structure_summary,
            "affordance_map": affordance_summary,
            "resource_status": {"visible_resources": environment.get_visible_resources(agent.position)},
            "legal_actions": action_affordances,
        }

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
            },
        }

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
