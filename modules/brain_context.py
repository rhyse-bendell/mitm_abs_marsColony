"""Planner-visible context construction.

This module limits what a planner can see at decision time. Context is a curated
view (not omniscient truth unless explicitly exposed). Affordances represent
currently possible actions and shape both deterministic RuleBrain and LLM planner
behavior.
"""

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
    """Builds planner-visible snapshots and action affordances per decision step."""
    def __init__(self, scenario_name: str = "Minerva Mars Colony"):
        self.scenario_name = scenario_name

    def _summarize_structures(self, environment) -> List[Dict[str, Any]]:
        build_targets = {name: t for name, t in environment.interaction_targets.items() if t.get("kind") == "build"}
        projects = [p for p in environment.construction.projects.values() if isinstance(p, dict)]
        summaries: List[Dict[str, Any]] = []
        for project in projects:
            target_id = project.get("target_id") or environment.construction.SITE_TO_BUILD_TARGET.get(project.get("site_id"))
            target = build_targets.get(target_id, {})
            raw_status = project.get("status") or "absent"
            state = "built" if raw_status == "complete" else "in_progress"
            usable = raw_status == "complete" and bool(project.get("correct", True))
            required = project.get("required_resources", {}).get("bricks", 0)
            delivered = project.get("delivered_resources", {}).get("bricks", 0)
            material_progress = min(1.0, delivered / required) if required else 0.0
            build_steps = list(project.get("build_steps") or [])
            completed_steps = sum(1 for step in build_steps if bool(step.get("completed")))
            build_progress = min(1.0, completed_steps / max(1, len(build_steps))) if build_steps else 0.0
            resource_complete = bool(project.get("resource_complete", False)) or (required > 0 and delivered >= required)
            readiness = environment.construction.evaluate_project_validation_readiness(project.get("id")) if hasattr(environment.construction, "evaluate_project_validation_readiness") else {}
            summaries.append(
                {
                    "structure_id": project.get("id"),
                    "object_id": target.get("object"),
                    "zone": target.get("zone"),
                    "structure_type": project.get("type", "unknown"),
                    "state": state,
                    "validated_correctness": bool(project.get("correct", True)),
                    "usable": usable,
                    "progress": round(build_progress, 2),
                    "material_progress": round(material_progress, 2),
                    "physical_build_progress": round(build_progress, 2),
                    "project_status": raw_status,
                    "resource_complete": resource_complete,
                    "required_resources": int(required or 0),
                    "delivered_resources": int(delivered or 0),
                    "ready_for_validation": raw_status == "ready_for_validation",
                    "needs_repair": project.get("correct", True) is False,
                    "overloaded": bool(project.get("overloaded", False)),
                    "functional_connections": [f"zone_link:{target.get('zone')}"] if target.get("zone") else [],
                    "validation_readiness": {
                        "physical_ready": bool(readiness.get("physical_ready", False)),
                        "support_ready": bool(readiness.get("support_ready", False)),
                        "rule_ready": bool(readiness.get("rule_ready", False)),
                        "epistemic_ready": bool(readiness.get("epistemic_ready", False)),
                        "validation_ready": bool(readiness.get("validation_ready", False)),
                        "missing_rules": list(readiness.get("missing_rules") or [])[:3],
                        "missing_epistemic_functions": list(readiness.get("missing_epistemic_functions") or []),
                        "repair_actions": [str((opt or {}).get("action") or opt) for opt in list(readiness.get("repair_options") or [])[:3]],
                    },
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

    @staticmethod
    def _new_structure_possible(environment) -> bool:
        """Projects are instantiated dynamically: a buildable site with spare capacity is a build opportunity."""
        construction = getattr(environment, "construction", None)
        finder = getattr(construction, "find_buildable_site_for_structure", None)
        return bool(callable(finder) and finder("house") is not None)

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
        epistemic = agent._epistemic_sufficiency(environment) if hasattr(agent, "_epistemic_sufficiency") else {}

        if info_count < 2:
            blockers.append("insufficient_information_inspection")
        if knowledge_count < 1:
            blockers.append("insufficient_rule_knowledge")
        if inspected_sources < 1:
            blockers.append("no_inspected_information_source")
        if not available_build_targets:
            blockers.append("no_navigable_build_target")
        if epistemic and bool(epistemic.get("role_missing")):
            blockers.append("missing_role_grounding")
        if epistemic and bool(epistemic.get("stale_grounding")):
            blockers.append("stale_epistemic_grounding")

        soft_only_blockers = set(blockers).issubset({"missing_role_grounding", "stale_epistemic_grounding"})
        construction_bootstrap_ready = bool(
            info_count >= 2
            and knowledge_count >= 1
            and inspected_sources >= 1
            and bool(available_build_targets)
            and soft_only_blockers
        )

        if score < 4 or (blockers and not construction_bootstrap_ready):
            status = "premature"
        elif in_progress > 0:
            status = "plausible"
        elif absent == 0 and not self._new_structure_possible(environment):
            status = "blocked"
            blockers = blockers + ["no_remaining_absent_structure_targets"]
        else:
            status = "plausible"

        return {
            "status": status,
            "score": score,
            "blockers": blockers,
            "ready_for_build": status == "plausible",
            "construction_bootstrap_ready": construction_bootstrap_ready,
            "inspected_sources": inspected_sources,
            "build_targets_available": len(available_build_targets),
            "epistemic_sufficiency_score": float(epistemic.get("score", 0.0) or 0.0),
            "epistemic_refresh_pressure": float(epistemic.get("refresh_pressure", 0.0) or 0.0),
        }

    def _affordances(self, agent, environment, build_readiness: Dict[str, Any] | None = None, phase_profile: Dict[str, Any] | None = None, team_state: Dict[str, Any] | None = None, support_subgoals: List[Dict[str, Any]] | None = None, task_methods: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        if phase_profile is None:
            phase_profile = self._phase_profile(environment)
        if build_readiness is None:
            build_readiness = self._build_readiness(agent, self._summarize_structures(environment), environment, team_state=team_state or {})
        stage = phase_profile.get("stage", "execution")
        readiness_ok = bool(build_readiness.get("ready_for_build"))
        bootstrap_bridge_ready = bool(build_readiness.get("construction_bootstrap_ready"))
        epistemic_suff = float(build_readiness.get("epistemic_sufficiency_score", 0.0) or 0.0)
        refresh_pressure = float(build_readiness.get("epistemic_refresh_pressure", 0.0) or 0.0)
        mismatch_pressure = any("mismatch" in e.lower() for e in (agent.activity_log[-6:] if agent.activity_log else []))
        built_state = self._summarize_structures(environment)
        validation_live = any(
            (float(item.get("physical_build_progress", item.get("progress", 0.0)) or 0.0) > 0.0)
            or str(item.get("project_status") or "") in {"needs_repair", "complete"}
            or (
                bool(item.get("ready_for_validation"))
                and float(item.get("physical_build_progress", item.get("progress", 0.0)) or 0.0) > 0.0
            )
            for item in built_state
        )
        materially_ready_unbuilt = any(
            bool(item.get("resource_complete"))
            and str(item.get("project_status") or "") not in {"ready_for_validation", "needs_repair", "complete"}
            and float(item.get("physical_build_progress", item.get("progress", 0.0)) or 0.0) <= 0.0
            for item in built_state
        )
        materially_incomplete_projects = [
            item
            for item in built_state
            if str(item.get("project_status") or "") in {"absent", "in_progress"}
            and int(item.get("required_resources", 0) or 0) > int(item.get("delivered_resources", 0) or 0)
            and float(item.get("physical_build_progress", item.get("progress", 0.0)) or 0.0) <= 0.0
        ]
        capacity_summary = environment.construction.get_site_capacity_summary() if hasattr(environment.construction, "get_site_capacity_summary") else []
        capacity_unlock = environment.construction.get_next_capacity_unlock() if hasattr(environment.construction, "get_next_capacity_unlock") else None
        bridge_unlock_needed = bool(isinstance(capacity_unlock, dict) and capacity_unlock.get("bridge_id"))
        has_artifacts = bool((team_state or {}).get("externalized_artifacts"))
        nearby_teammates = sum(
            1
            for other in getattr(environment, "agents", [])
            if other is not agent and ((agent.position[0] - other.position[0]) ** 2 + (agent.position[1] - other.position[1]) ** 2) ** 0.5 <= 20.0
        )
        teammate_help_signals = dict((team_state or {}).get("teammate_help_signals", {}))
        productive_coordination = bool(nearby_teammates > 0 and (agent.known_gaps or any(teammate_help_signals.values()) or has_artifacts))
        communication_no_effect_streak = int((getattr(agent, "communication_state", {}) or {}).get("no_effect_streak", 0) or 0)

        consult_assessment = dict((team_state or {}).get("consult_assessment") or {})

        def utility_for(action: ExecutableActionType, target_kind: str | None = None):
            if action == ExecutableActionType.INSPECT_INFORMATION_SOURCE:
                base = 0.95 if stage == "early" else (0.55 if not readiness_ok else 0.25)
                base += min(0.45, 0.55 * refresh_pressure)
                return min(1.0, base)
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
                if consult_assessment.get("has_artifacts") and not consult_assessment.get("useful"):
                    return 0.05  # nothing new to learn from it: the same artifact version, no new context
                return 0.6 if has_artifacts else 0.2
            if action == ExecutableActionType.TRANSPORT_RESOURCES:
                if materially_incomplete_projects:
                    return 0.93 if readiness_ok else 0.82
                penalty = 0.35 if epistemic_suff < 0.45 else (0.15 if epistemic_suff < 0.6 else 0.0)
                return max(0.08, (0.75 if readiness_ok and stage != "early" else 0.25) - penalty)
            if action in {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}:
                if not readiness_ok and not bootstrap_bridge_ready:
                    return 0.05
                if not readiness_ok and bootstrap_bridge_ready:
                    return max(0.55, (0.72 if stage in {"execution", "late"} else 0.56) - (0.16 if epistemic_suff < 0.55 else 0.0))
                penalty = 0.3 if epistemic_suff < 0.5 else (0.15 if epistemic_suff < 0.65 else 0.0)
                return max(0.08, (0.85 if stage in {"execution", "late"} else 0.3) - penalty)
            if action in {ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, ExecutableActionType.VALIDATE_CONSTRUCTION}:
                if materially_ready_unbuilt:
                    return 0.01
                if not validation_live:
                    return 0.02
                penalty = 0.55 if epistemic_suff < 0.62 else 0.0
                return max(0.05, (0.9 if mismatch_pressure or stage == "late" or target_kind == "build" else 0.3) - penalty)
            if action in {ExecutableActionType.REASSESS_PLAN, ExecutableActionType.OBSERVE_ENVIRONMENT}:
                contradiction = float((build_readiness or {}).get("epistemic_refresh_pressure", 0.0) or 0.0)
                if readiness_ok:
                    return 0.04 if contradiction < 0.65 else 0.18
                return 0.1 if stage == "early" else 0.06
            return 0.2

        legal = [
            {"action_type": ExecutableActionType.WAIT.value, "target_id": None, "target_class": "self", "utility": 0.1},
            {"action_type": ExecutableActionType.COMMUNICATE.value, "target_id": "nearby_agent", "target_class": "team", "utility": utility_for(ExecutableActionType.COMMUNICATE)},
            {"action_type": ExecutableActionType.REQUEST_ASSISTANCE.value, "target_id": "nearby_agent", "target_class": "team", "utility": utility_for(ExecutableActionType.REQUEST_ASSISTANCE)},
            {"action_type": ExecutableActionType.EXTERNALIZE_PLAN.value, "target_id": "whiteboard", "target_class": "artifact", "utility": utility_for(ExecutableActionType.EXTERNALIZE_PLAN)},
            {"action_type": ExecutableActionType.CONSULT_TEAM_ARTIFACT.value, "target_id": "whiteboard", "target_class": "artifact", "utility": utility_for(ExecutableActionType.CONSULT_TEAM_ARTIFACT),
             "consult_useful": bool(consult_assessment.get("useful")), "consult_redundant": bool(consult_assessment.get("has_artifacts")) and not bool(consult_assessment.get("useful")),
             "consult_reasons": list(consult_assessment.get("reasons") or []), "consult_artifact_id": consult_assessment.get("artifact_id")},
        ]
        for item in legal:
            if item.get("action_type") in {ExecutableActionType.COMMUNICATE.value, ExecutableActionType.REQUEST_ASSISTANCE.value}:
                item["reachable"] = nearby_teammates > 0
                item["productive"] = productive_coordination
                item["no_effect_streak"] = communication_no_effect_streak

        site_rows = {r.get("site_id"): r for r in (capacity_summary or []) if isinstance(r, dict)}
        build_target_site = {v: k for k, v in (getattr(environment.construction, "SITE_TO_BUILD_TARGET", {}) or {}).items()}
        start_options_by_target: Dict[str, List[Dict[str, Any]]] = {}
        list_starts = getattr(environment.construction, "list_start_options", None)
        for option in (list_starts() if callable(list_starts) else []):
            start_options_by_target.setdefault(option["target_id"], []).append(option)
        for target_name, target in environment.interaction_targets.items():
            # Only information sources (inspect) and build sites (start construction) are action targets here;
            # artifacts such as the whiteboard are reached through externalize / consult.
            if target.get("kind") not in {"information", "build"}:
                continue
            if target.get("kind") == "build" and int((site_rows.get(build_target_site.get(target_name), {}) or {}).get("capacity", 1) or 0) <= 0:
                continue  # a site with no capacity can never host a project
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
                source_memory = dict((getattr(agent, "source_memory_state", {}) or {}).get(target_name, {}) or {})
                if source_state == "inspected":
                    refresh_pressure = float((build_readiness or {}).get("epistemic_refresh_pressure", 0.0) or 0.0)
                    confidence = float(source_memory.get("memory_confidence", 1.0) or 1.0)
                    if refresh_pressure < 0.28 and confidence >= 0.62:
                        continue
            base = {
                "action_type": action_type.value,
                "target_id": target_name,
                "target_zone": target.get("zone"),
                "target_point": accessible_point,
                "target_class": target.get("kind"),
                "reachable": reachable,
                "source_state": source_state,
                "utility": utility_for(action_type, target_kind=target.get("kind")),
            }
            if target.get("kind") != "build":
                legal.append(base)
                continue
            # Build site. Two distinct, read-only facts: an unfinished project here can be worked on, and each
            # legal (site, structure type) start may create a new project. Neither says which one to choose.
            for option in start_options_by_target.get(target_name, []):
                legal.append(
                    {
                        **base,
                        "start_mode": "new_project",
                        "action_args": {"structure_type": option["structure_type"]},
                        "required_resources": dict(option["required_resources"]),
                    }
                )

        bricks_available = self._bricks_available(environment, agent)
        for project_id, project in sorted(environment.construction.projects.items()):
            if not isinstance(project, dict) or str(project.get("status") or "") == "complete" or project.get("validated_complete"):
                continue
            point = environment.get_interaction_target_position(project_id, from_position=agent.position)
            if point is None:
                continue
            site_target = environment.construction.SITE_TO_BUILD_TARGET.get(project.get("site_id"))
            base = {
                "target_id": project_id,
                "target_zone": (environment.interaction_targets.get(site_target) or {}).get("zone"),
                "target_point": point,
                "target_class": "build",
                "target_kind": "project",
                "project_type": project.get("type"),
                "reachable": environment._segment_is_navigable(agent.position, point),
            }
            required = int((project.get("required_resources") or {}).get("bricks", 0) or 0)
            delivered = int((project.get("delivered_resources") or {}).get("bricks", 0) or 0)
            steps = list(project.get("build_steps") or [])
            progressed = any(bool(step.get("completed")) for step in steps)
            physically_complete = bool(project.get("structurally_complete"))
            needs_material = not bool(project.get("resource_complete")) and delivered < required and str(project.get("status") or "") != "ready_for_validation"
            # Each fact below is stated per exact project; which one to work on is the Pilot's choice.
            if needs_material and bricks_available:
                legal.append({**base, "action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "work_stage": "material_delivery", "duration_s": 30.0, "utility": utility_for(ExecutableActionType.TRANSPORT_RESOURCES)})
            if not physically_complete:
                if progressed:
                    legal.append({**base, "action_type": ExecutableActionType.CONTINUE_CONSTRUCTION.value, "utility": utility_for(ExecutableActionType.CONTINUE_CONSTRUCTION)})
                else:
                    legal.append({**base, "action_type": ExecutableActionType.START_CONSTRUCTION.value, "start_mode": "existing_project", "utility": utility_for(ExecutableActionType.START_CONSTRUCTION)})
            else:
                legal.append({**base, "action_type": ExecutableActionType.VALIDATE_CONSTRUCTION.value, "utility": utility_for(ExecutableActionType.VALIDATE_CONSTRUCTION, target_kind="build")})
            if progressed:
                legal.append({**base, "action_type": ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value, "utility": utility_for(ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, target_kind="build")})

        has_productive_target = any(
            item.get("action_type")
            in {
                ExecutableActionType.INSPECT_INFORMATION_SOURCE.value,
                ExecutableActionType.START_CONSTRUCTION.value,
                ExecutableActionType.TRANSPORT_RESOURCES.value,
                ExecutableActionType.CONTINUE_CONSTRUCTION.value,
                ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION.value,
                ExecutableActionType.VALIDATE_CONSTRUCTION.value,
            }
            and bool(item.get("reachable", True))
            for item in legal
        )
        if not has_productive_target or refresh_pressure >= 0.72:
            legal.append({"action_type": ExecutableActionType.OBSERVE_ENVIRONMENT.value, "target_id": None, "target_class": "self", "utility": utility_for(ExecutableActionType.OBSERVE_ENVIRONMENT)})
            legal.append({"action_type": ExecutableActionType.REASSESS_PLAN.value, "target_id": None, "target_class": "self", "utility": utility_for(ExecutableActionType.REASSESS_PLAN)})

        if bridge_unlock_needed and bricks_available:
            bridge_id = str(capacity_unlock.get("bridge_id"))
            required = int(capacity_unlock.get("required_resources", 0) or 0)
            delivered = int(capacity_unlock.get("delivered_resources", 0) or 0)
            bridge_status = str(capacity_unlock.get("bridge_status") or "not_started")
            if delivered < required:
                legal.append({"action_type": ExecutableActionType.TRANSPORT_RESOURCES.value, "target_id": bridge_id, "target_class": "bridge", "target_kind": "bridge", "work_stage": "bridge_material_delivery", "reachable": True, "utility": 0.96})
            elif bridge_status != "complete":
                legal.append({"action_type": ExecutableActionType.CONTINUE_CONSTRUCTION.value, "target_id": bridge_id, "target_class": "bridge", "target_kind": "bridge", "work_stage": "bridge_build", "reachable": True, "utility": 0.94})

        list_connections = getattr(environment.construction, "list_connection_options", None)
        for option in (list_connections() if callable(list_connections) else []):
            legal.append(
                {
                    "action_type": ExecutableActionType.CONNECT_STRUCTURES.value,
                    "target_id": None,
                    "action_args": {
                        "connector_project_id": option["connector_project_id"],
                        "endpoint_a_id": option["from_project_id"],
                        "endpoint_b_id": option["to_project_id"],
                    },
                    "target_class": "build",
                    "target_kind": "connection",
                    "connection_type": option["connection_type"],
                    "reachable": True,
                    "utility": 0.1,
                }
            )

        if support_subgoals:
            from modules.support_reasoning import annotate_affordances

            annotate_affordances(legal, support_subgoals)
        if task_methods:
            self._annotate_task_methods(legal, task_methods)

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

    @staticmethod
    def _annotate_task_methods(affordances: List[Dict[str, Any]], task_methods: List[Dict[str, Any]]) -> None:
        """Mark legal affordances that instantiate a task method this agent is competent in. Information only.

        A binding names the structured action_args of the method's intended instance (e.g. structure_type=house), so the
        match is against an affordance that is ALREADY legal; nothing is created and unavailable variants never appear.
        A method that lists connect_structures prefers only connections the support reasoning already identified.
        """
        for aff in affordances:
            serves = []
            for method in task_methods:
                if method["weight"] <= 0.0:
                    continue
                action = aff.get("action_type")
                if action not in method["candidate_actions"]:
                    continue
                binding = method["action_bindings"].get(action)
                if binding:
                    args = aff.get("action_args") or {}
                    if aff.get("start_mode") == "new_project" and all(str(args.get(k)) == str(v) for k, v in binding.items()):
                        serves.append({"method_id": method["method_id"], "goal_id": method["goal_id"], "weight": method["weight"]})
                elif action == "connect_structures" and aff.get("addresses_support_subgoals"):
                    serves.append({"method_id": method["method_id"], "goal_id": method["goal_id"], "weight": method["weight"]})
            if serves:
                aff["serves_task_methods"] = serves

    @staticmethod
    def _support_reasoning(agent, sim_state, environment) -> Dict[str, Any]:
        """Support subgoals this agent can recognise from the DIK it holds (read-only; chooses nothing)."""
        from modules.support_accounting import load_support_model
        from modules.support_reasoning import derive_support_subgoals, load_support_reasoning

        task_id = getattr(getattr(sim_state, "task_model", None), "task_id", None) or "mars_colony"
        config = load_support_reasoning(task_id)
        try:
            model = load_support_model(task_id)
        except FileNotFoundError:
            model = None
        construction = environment.construction
        if not config or not model or not hasattr(construction, "list_connection_options"):
            return {"subgoals": [], "held_basis_only": True}
        data_ids, info_ids, _ = agent._held_dik_ids()
        subgoals = derive_support_subgoals(
            config=config,
            support_model=model,
            held_ids=set(data_ids) | set(info_ids),
            projects=construction.projects,
            links=[dict(l) for l in construction.connectors],
            connection_options=construction.list_connection_options(),
            start_structure_types=construction.startable_structure_types(),
        )
        return {"subgoals": subgoals, "held_basis_only": True}

    @staticmethod
    def _bricks_available(environment, agent=None) -> bool:
        """Read-only legality predicate: a brick source remains, or this agent is already carrying one."""
        carrying = int((((getattr(agent, "transport_state", None) or {}).get("carrying")) or {}).get("quantity", 0) or 0)
        if carrying > 0:
            return True
        nodes = getattr(getattr(environment, "construction", None), "resource_nodes", None)
        if not isinstance(nodes, dict):
            return True  # unknown model: do not hide the action
        return any(int(getattr(pile, "quantity", 0) or 0) > 0 for pile in nodes.values())

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
        team_plan_summaries = []
        for aid, artifact in sim_state.team_knowledge_manager.artifacts.items():
            payload = {
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
            if artifact.artifact_type == "team_plan" and isinstance(getattr(artifact, "content", None), dict):
                content = dict(artifact.content)
                payload["team_plan"] = {
                    "plan_id": content.get("plan_id"),
                    "status": content.get("status"),
                    "goal_ids": list(content.get("goal_ids", [])),
                    "goal_summary": content.get("goal_summary"),
                    "project_targets": list(content.get("project_targets", [])),
                    "assignments_by_role": dict(content.get("assignments_by_role", {})),
                    "trigger_reason": content.get("trigger_reason"),
                    "blocked_reasons": list(content.get("blocked_reasons", [])),
                    "supporters": list(content.get("supporters", [])),
                    "opposers": list(content.get("opposers", [])),
                    "review_at": content.get("review_at"),
                    "expires_at": content.get("expires_at"),
                    "last_plan_event": content.get("last_plan_event"),
                    "plan_event_time": content.get("plan_event_time"),
                }
                team_plan_summaries.append({"artifact_id": aid, **payload["team_plan"]})
            artifact_summaries.append(payload)

        team_knowledge_summary = sim_state.team_knowledge_manager.summarize()
        active_team_plan = dict(team_knowledge_summary.get("active_team_plan") or {})
        team_plan_recent_updates = list(team_knowledge_summary.get("team_plan_recent_updates", []))
        if not team_plan_summaries:
            team_plan_summaries = list(team_knowledge_summary.get("team_plan_summaries", []))
        team_plan_summaries = sorted(
            team_plan_summaries,
            key=lambda item: float(item.get("plan_event_time") or 0.0),
            reverse=True,
        )[:6]
        my_role_assignment = None
        if active_team_plan:
            my_role_assignment = dict(active_team_plan.get("assignments_by_role", {})).get(agent.role)
        validated_plan_exists = any(
            a.artifact_type in {"plan", "team_plan"}
            and (
                a.artifact_type == "plan"
                or (isinstance(getattr(a, "content", None), dict) and str(a.content.get("status", "")).lower() in {"proposed", "committed"})
            )
            for a in sim_state.team_knowledge_manager.artifacts.values()
        )
        committed_team_plan_exists = any(
            a.artifact_type == "team_plan"
            and isinstance(getattr(a, "content", None), dict)
            and str(a.content.get("status", "")).lower() == "committed"
            for a in sim_state.team_knowledge_manager.artifacts.values()
        )
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
        recent_updates = list(sim_state.team_knowledge_manager.recent_updates[-24:])
        recent_construction_updates = [
            update
            for update in recent_updates
            if str(update.get("event", "")).startswith("construction_")
            or str(update.get("event", "")).startswith("project_marked_")
            or str(update.get("event", "")) == "project_state_published_to_team_knowledge"
        ][-8:]
        construction_artifacts = [
            {
                "artifact_id": aid,
                "project_id": artifact.content.get("project_id"),
                "structure_type": artifact.content.get("structure_type"),
                "status": artifact.content.get("status"),
                "resource_complete": bool(artifact.content.get("resource_complete", False)),
                "required_resources": dict(artifact.content.get("required_resources", {})),
                "delivered_resources": dict(artifact.content.get("delivered_resources", {})),
                "progress_ratio": float(artifact.content.get("progress_ratio", 0.0) or 0.0),
                "correct": bool(artifact.content.get("correct", True)),
                "validated": bool(artifact.content.get("validated", False)),
                "last_status_event": artifact.content.get("last_status_event"),
                "status_changed_at": artifact.content.get("status_changed_at"),
            }
            for aid, artifact in sim_state.team_knowledge_manager.artifacts.items()
            if isinstance(getattr(artifact, "content", None), dict) and str(artifact.artifact_id).startswith("construction:")
        ]
        team_state = {
            "team_shared_knowledge": team_knowledge_summary,
            "teammate_roles": {other.name: other.role for other in sim_state.agents if other.name != agent.name},
            "teammate_inferred_goals": {
                teammate: model.get("goals", []) for teammate, model in agent.theory_of_mind.items()
            },
            "tom_summary": agent.theory_of_mind,
            "recent_shared_updates": recent_updates[-5:],
            "recent_construction_updates": recent_construction_updates,
            "plan_readiness": "validated_shared_plan" if committed_team_plan_exists else ("proposed_shared_plan" if validated_plan_exists else "partial_or_fragmentary_plan"),
            "externalized_artifacts": artifact_summaries,
            "active_team_plan": active_team_plan,
            "team_plan_summaries": team_plan_summaries,
            "my_role_assignment": my_role_assignment,
            "team_plan_recent_updates": team_plan_recent_updates[-8:],
            "construction_artifacts": construction_artifacts,
            "teammate_help_signals": teammate_help_signals,
            "derivation_events": list(getattr(agent, "derivation_events", [])[-5:]),
            "task_rules": [r.rule_id for r in task_model.rules.values() if r.enabled] if task_model else [],
            "task_goals": [g.goal_id for g in task_model.goals.values() if g.enabled] if task_model else [],
            "task_plan_methods": [m.method_id for m in task_model.plan_methods.values() if m.enabled] if task_model else [],
        }

        build_readiness = self._build_readiness(agent, structure_summary, environment, team_state=team_state)
        team_state["consult_assessment"] = agent._consult_assessment(environment, sim_state=sim_state)
        support_reasoning = self._support_reasoning(agent, sim_state, environment)
        task_methods = agent._task_method_status(current_phase.get("name", "default"))
        action_affordances = self._affordances(agent, environment, build_readiness=build_readiness, phase_profile=phase_profile, team_state=team_state, support_subgoals=support_reasoning["subgoals"], task_methods=task_methods)

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
            "support_deficits": environment.construction.get_support_deficit_report() if hasattr(environment.construction, "get_support_deficit_report") else [],
            "site_capacity_summary": environment.construction.get_site_capacity_summary() if hasattr(environment.construction, "get_site_capacity_summary") else [],
            "capacity_unlock": environment.construction.get_next_capacity_unlock() if hasattr(environment.construction, "get_next_capacity_unlock") else None,
            # Observable resource fact (a pile still holds bricks, or this agent carries one); transport itself is offered per project.
            "brick_source_available": self._bricks_available(environment, agent),
        }
        # Recognised support deficits are derived from what THIS agent holds plus observable construction state.
        # (`support_deficits` above is the unfiltered observation of attached-link requirements, kept as before.)

        mechanism_profile = {
            "communication_propensity": getattr(agent, "communication_propensity", 0.5),
            "goal_alignment": getattr(agent, "goal_alignment", 0.5),
            "help_tendency": getattr(agent, "help_tendency", 0.5),
            "build_speed": getattr(agent, "build_speed", 0.5),
            "rule_accuracy": getattr(agent, "rule_accuracy", 0.5),
            "artifact_consultation_tendency": agent._trait_value("artifact_consultation_tendency"),
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
            "support_reasoning": support_reasoning,
            "task_methods": task_methods,
            "certainty_signals": {
                "definitely_known": knowledge_summary[-5:],
                "uncertain_or_missing": sorted(set(list(agent.known_gaps) + (["validated_plan_missing"] if active_plan is None else []))),
                "information_gathering_value": "high" if (len(knowledge_summary) < 2 or build_readiness["status"] != "plausible") else "medium",
            },
            "build_readiness": build_readiness,
            "packets_inspected": list(agent.memory_seen_packets),
            "recent_action_outcomes": [dict(o) for o in list(getattr(agent, "recent_action_outcomes", []) or [])[-6:]],
            "recent_failed_attempts": [
                e for e in history_events if "blocked" in e.lower() or "could not" in e.lower() or "mismatch" in e.lower()
            ],
            "mechanism_profile": mechanism_profile,
            # Legacy alias retained for backward-compatible readers.
            "traits": mechanism_profile,
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
                "source_memory": dict(getattr(agent, "source_memory_state", {}) or {}),
            },
            "epistemic_sufficiency": agent._epistemic_sufficiency(environment) if hasattr(agent, "_epistemic_sufficiency") else {},
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
