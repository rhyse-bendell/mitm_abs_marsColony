"""Universal mecha-side action legality gate.

ActionGate evaluates pilot-selected cockpit actions before legacy execution. It
reports blockers and possible reroutes without executing actions or choosing
strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from modules.action_schema import BrainDecision, ExecutableActionType, decision_action_args, validate_action_args


@dataclass
class ActionGateResult:
    legal: bool
    normalized_decision: BrainDecision
    blockers: list[str] = field(default_factory=list)
    available_reroutes: list[BrainDecision] = field(default_factory=list)
    source: str = "mecha_action_gate"
    visibility: str = "agent_observable"
    target_id: Optional[str] = None
    target_kind: Optional[str] = None
    project_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentActionGate:
    def evaluate(self, *, agent, decision: BrainDecision, sim_state) -> ActionGateResult:
        action = decision.selected_action
        construction = getattr(getattr(sim_state, "environment", None), "construction", None)
        startable = construction.startable_structure_types() if construction is not None and hasattr(construction, "startable_structure_types") else None
        arg_errors = validate_action_args(action, decision_action_args(decision), allowed_structure_types=startable)
        if arg_errors:
            # Structural rejection only: the Pilot's request is reported back, never repaired or replaced.
            return ActionGateResult(
                False,
                decision,
                blockers=["malformed_action_args", *arg_errors],
                target_id=decision.target_id,
                metadata={"action_args": decision_action_args(decision)},
            )
        target_blocked = self._check_project_target_authority(decision=decision, construction=construction)
        if target_blocked is not None:
            return target_blocked
        if action == ExecutableActionType.INSPECT_INFORMATION_SOURCE:
            return self._evaluate_inspect(agent=agent, decision=decision, sim_state=sim_state)
        if action == ExecutableActionType.MOVE_TO_TARGET:
            return self._evaluate_move(agent=agent, decision=decision, sim_state=sim_state)
        if action in {ExecutableActionType.COMMUNICATE, ExecutableActionType.REQUEST_ASSISTANCE}:
            return self._evaluate_communication(agent=agent, decision=decision, sim_state=sim_state)
        if action in {ExecutableActionType.EXTERNALIZE_PLAN, ExecutableActionType.CONSULT_TEAM_ARTIFACT}:
            return self._evaluate_artifact(agent=agent, decision=decision, sim_state=sim_state)
        if action == ExecutableActionType.TRANSPORT_RESOURCES:
            return self._evaluate_transport(agent=agent, decision=decision, sim_state=sim_state)
        if action in {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}:
            return self._evaluate_construction(agent=agent, decision=decision, sim_state=sim_state)
        if action == ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION:
            return self._evaluate_repair(agent=agent, decision=decision, sim_state=sim_state)
        if action == ExecutableActionType.VALIDATE_CONSTRUCTION:
            return self._evaluate_validation(agent=agent, decision=decision, sim_state=sim_state)
        if action == ExecutableActionType.CONNECT_STRUCTURES:
            return self._evaluate_connection(agent=agent, decision=decision, sim_state=sim_state)
        return self._evaluate_meta(agent=agent, decision=decision, sim_state=sim_state)

    _PROJECT_TASKWORK = {
        ExecutableActionType.TRANSPORT_RESOURCES,
        ExecutableActionType.START_CONSTRUCTION,
        ExecutableActionType.CONTINUE_CONSTRUCTION,
        ExecutableActionType.VALIDATE_CONSTRUCTION,
        ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
    }

    def _check_project_target_authority(self, *, decision: BrainDecision, construction) -> Optional[ActionGateResult]:
        """Project-specific taskwork must name its project; the gate never picks one of several for the Pilot.

        An exact project id (or the bridge) always passes. A site alias, ``active_construction`` or an absent target is
        a legacy convenience that is accepted only when it names exactly one unfinished project; when several are
        possible the request is blocked and the legal candidates are reported as information for a new Pilot decision.
        """
        if decision.selected_action not in self._PROJECT_TASKWORK or construction is None:
            return None
        if decision.selected_action == ExecutableActionType.START_CONSTRUCTION and decision_action_args(decision).get("structure_type"):
            return None  # typed START creates a new project; it does not address an existing one
        projects = getattr(construction, "projects", None) or {}
        target = str(decision.target_id or "").strip()
        if target in projects or target == "bridge_bc":
            return None
        unfinished = sorted(str(p.get("id")) for p in projects.values() if isinstance(p, dict) and str(p.get("status") or "") != "complete" and p.get("id"))
        if target.lower() in self._ACTIVE_CONSTRUCTION_PLACEHOLDERS or not target:
            candidates = unfinished
        else:
            is_alias = target in getattr(construction, "PROJECT_TO_SITE", {}) or target in getattr(construction, "LEGACY_PROJECT_TO_SITE", {})
            if is_alias:
                site_id = getattr(construction, "PROJECT_TO_SITE", {}).get(target) or getattr(construction, "LEGACY_PROJECT_TO_SITE", {}).get(target)
                candidates = [pid for pid in unfinished if projects[pid].get("site_id") == site_id]
            else:
                candidates = unfinished
        if len(candidates) <= 1:
            return None
        return ActionGateResult(
            False,
            decision,
            blockers=["ambiguous_project_target" if target else "project_target_required"],
            target_id=decision.target_id,
            metadata={"legal_candidate_project_ids": candidates, "requested_target": target or None},
        )

    def _pass(self, decision: BrainDecision) -> ActionGateResult:
        # TODO(architecture): expand family-specific legality checks here.
        return ActionGateResult(legal=True, normalized_decision=decision, target_id=decision.target_id)

    def _evaluate_inspect(self, *, agent, decision, sim_state):
        # A generative Pilot's INSPECT of a source it may not access is refused explicitly; the Mecha never inspects a different
        # source in its place and never turns it into COMMUNICATE / REQUEST_ASSISTANCE (those stay separate choices for the Pilot).
        # The Procedural Baseline keeps its documented scaffold retargeting (its selections must not change).
        if hasattr(agent, "is_generative_pilot") and agent.is_generative_pilot() and decision.target_id:
            blocker = agent.source_access_blocker(decision.target_id, getattr(sim_state, "environment", None))
            if blocker:
                return ActionGateResult(False, decision, blockers=[blocker], target_id=decision.target_id, metadata={"requested_source": str(decision.target_id)})
        return self._pass(decision)

    def _evaluate_move(self, *, agent, decision, sim_state):
        """Read-only legality of a movement request: a named interaction target / project, or an explicit (x, y). The Environment owns
        navigability and reachability; nothing is moved here and no other destination is ever substituted."""
        environment = getattr(sim_state, "environment", None)
        args = decision_action_args(decision)
        has_xy = "x" in args and "y" in args
        if (("x" in args) != ("y" in args)) or (has_xy and decision.target_id):
            return ActionGateResult(False, decision, blockers=["move_destination_ambiguous"], target_id=decision.target_id)
        if environment is None:
            return self._pass(decision)
        if has_xy:
            point = (float(args["x"]), float(args["y"]))
            if not environment.is_point_navigable(point):
                return ActionGateResult(False, decision, blockers=["destination_not_navigable"], target_id=None, metadata={"destination": point})
            if not environment.path_reachable(tuple(agent.position), point):
                return ActionGateResult(False, decision, blockers=["destination_unreachable"], target_id=None, metadata={"destination": point})
            return ActionGateResult(True, decision, target_id=None, metadata={"destination": point})
        target = str(decision.target_id or "").strip()
        if not target:
            return ActionGateResult(False, decision, blockers=["move_destination_required"], target_id=decision.target_id)
        point = environment.get_interaction_target_position(target, from_position=tuple(agent.position))
        if point is None:
            known = target in environment.interaction_targets or target in getattr(getattr(environment, "construction", None), "projects", {})
            blocker = "target_locked_until_bridge_access" if known and not environment.is_interaction_target_unlocked(target) else "unknown_move_target"
            return ActionGateResult(False, decision, blockers=[blocker], target_id=decision.target_id)
        if not environment.path_reachable(tuple(agent.position), tuple(point)):
            return ActionGateResult(False, decision, blockers=["destination_unreachable"], target_id=decision.target_id, metadata={"destination": tuple(point)})
        return ActionGateResult(True, decision, target_id=decision.target_id, metadata={"destination": tuple(point)})

    @staticmethod
    def _unencountered_attachments(agent, args):
        """Attached DIK ids the sender has NOT encountered (read-only). Only information the person was actually shown can be shared."""
        ids = list(args.get("attachments") or [])
        if not ids or not hasattr(agent, "encountered_dik"):
            return []
        encountered = agent.encountered_dik()
        return [i for i in ids if i not in encountered]

    def _evaluate_communication(self, *, agent, decision, sim_state):
        args = decision_action_args(decision)
        if "attachments" in args and "text" not in args:
            return ActionGateResult(False, decision, blockers=["attachments_require_a_message"], target_id=decision.target_id)
        missing = self._unencountered_attachments(agent, args)
        if missing:
            return ActionGateResult(False, decision, blockers=["attachment_not_encountered"], target_id=decision.target_id, metadata={"attachments": list(args.get("attachments") or [])})
        if decision.selected_action == ExecutableActionType.COMMUNICATE and "text" in args:
            # A typed message: its recipient scope must be the team or an existing teammate (never the sender).
            scope = str(args.get("scope") or "team")
            if scope.lower() != "team":
                names = {str(getattr(a, "name", "")).lower() for a in getattr(sim_state, "agents", []) if a is not agent} | {str(getattr(a, "role", "")).lower() for a in getattr(sim_state, "agents", []) if a is not agent}
                if scope.lower() not in names:
                    return ActionGateResult(False, decision, blockers=["unknown_recipient"], target_id=decision.target_id, metadata={"scope": scope})
        return self._pass(decision)

    def _evaluate_artifact(self, *, agent, decision, sim_state):
        args = decision_action_args(decision)
        if decision.selected_action == ExecutableActionType.EXTERNALIZE_PLAN and getattr(agent, "is_human_participant", lambda: False)() and "text" not in args:
            return ActionGateResult(False, decision, blockers=["empty_message"], target_id=decision.target_id)  # a person writes the note; nothing is authored for them
        if decision.selected_action == ExecutableActionType.EXTERNALIZE_PLAN and args:
            if "text" not in args:
                return ActionGateResult(False, decision, blockers=["attachments_require_a_note"], target_id=decision.target_id)
            if self._unencountered_attachments(agent, args):
                return ActionGateResult(False, decision, blockers=["attachment_not_encountered"], target_id=decision.target_id)
        return self._pass(decision)

    def _evaluate_transport(self, *, agent, decision, sim_state):
        # Read-only legality: a transport needs a brick source to pick up from.
        nodes = getattr(getattr(getattr(sim_state, "environment", None), "construction", None), "resource_nodes", None)
        carrying = int((((getattr(agent, "transport_state", None) or {}).get("carrying")) or {}).get("quantity", 0) or 0)
        if carrying <= 0 and isinstance(nodes, dict) and not any(int(getattr(pile, "quantity", 0) or 0) > 0 for pile in nodes.values()):
            return ActionGateResult(
                False,
                decision,
                blockers=["no_pickup_resource_source"],
                target_id=decision.target_id,
                metadata={"resource_nodes": {pid: int(getattr(p, "quantity", 0) or 0) for pid, p in nodes.items()}},
            )
        # An exact project target is checked as asked; the gate never substitutes a different project.
        construction = getattr(getattr(sim_state, "environment", None), "construction", None)
        project = (getattr(construction, "projects", None) or {}).get(str(decision.target_id or ""))
        if isinstance(project, dict):
            status = str(project.get("status") or "")
            required = int((project.get("required_resources") or {}).get("bricks", 0) or 0)
            delivered = int((project.get("delivered_resources") or {}).get("bricks", 0) or 0)
            if status in {"complete", "ready_for_validation"} or bool(project.get("resource_complete")) or (required > 0 and delivered >= required):
                return ActionGateResult(False, decision, blockers=["project_does_not_need_material"], target_id=decision.target_id, project_id=project.get("id"))
        return self._pass(decision)

    def _evaluate_construction(self, *, agent, decision, sim_state):
        construction = getattr(getattr(sim_state, "environment", None), "construction", None)
        if decision.selected_action != ExecutableActionType.START_CONSTRUCTION or construction is None:
            return self._pass(decision)
        structure_type = str(decision_action_args(decision).get("structure_type") or "").strip().lower()
        if structure_type:
            # Read-only legality of creating a new project of this type here; the project is created only when the
            # committed START executes.
            legal, blockers, site_id = construction.check_start(decision.target_id, structure_type)
            design_errors = self._design_errors(agent, construction, structure_type, decision_action_args(decision))
            if design_errors:
                legal, blockers = False, list(blockers) + design_errors
            return ActionGateResult(
                legal,
                decision,
                blockers=list(blockers),
                target_id=decision.target_id,
                metadata={"structure_type": structure_type, "site_id": site_id},
            )
        # Untyped START only continues an existing unfinished project; it never creates one.
        target = str(decision.target_id or "").strip()
        exact = (getattr(construction, "projects", None) or {}).get(target)
        if isinstance(exact, dict) and str(exact.get("status") or "") == "complete":
            return ActionGateResult(False, decision, blockers=["project_already_complete"], target_id=decision.target_id, project_id=exact.get("id"))
        is_site_target = target in getattr(construction, "PROJECT_TO_SITE", {}) or target in getattr(construction, "LEGACY_PROJECT_TO_SITE", {})
        if is_site_target and not construction.resolve_project_id(target, create_if_missing=False):
            return ActionGateResult(False, decision, blockers=["structure_type_required_for_new_project"], target_id=decision.target_id)
        return self._pass(decision)

    @staticmethod
    def _design_errors(agent, construction, structure_type, args):
        """Structural check of a participant design (task-package schema). A person must choose every property when starting a designable structure;
        no default is ever filled in on their behalf. Artificial Pilots may omit the design (the legacy fidelity model applies)."""
        schema = getattr(construction, "design_schema", None)
        if schema is None or not schema.has(structure_type):
            return ["design_not_supported_for_structure"] if args.get("design") else []
        design = args.get("design")
        if design is None:
            return ["design_required"] if getattr(agent, "is_human_participant", lambda: False)() else []
        return schema.validate_choice(structure_type, design, complete=True)

    def _evaluate_connection(self, *, agent, decision, sim_state):
        # Read-only legality of connect(connector, endpoint, endpoint). Any role may perform a legal connection;
        # whether a Pilot chooses it is a decision-knowledge question the gate deliberately does not answer.
        construction = getattr(getattr(sim_state, "environment", None), "construction", None)
        args = decision_action_args(decision)
        connector_id, from_id, to_id = (str(args.get(k) or "") for k in ("connector_project_id", "endpoint_a_id", "endpoint_b_id"))
        if construction is None or not (connector_id and from_id and to_id):
            return ActionGateResult(False, decision, blockers=["malformed_action_args"], target_id=decision.target_id)
        legal, blockers = construction.check_connection(connector_id, from_id, to_id)
        return ActionGateResult(
            legal,
            decision,
            blockers=list(blockers),
            target_id=decision.target_id,
            project_id=connector_id,
            metadata={"connector_project_id": connector_id, "from_project_id": from_id, "to_project_id": to_id},
        )

    def _evaluate_repair(self, *, agent, decision, sim_state):
        design = decision_action_args(decision).get("design")
        if getattr(agent, "is_human_participant", lambda: False)():
            construction = getattr(getattr(sim_state, "environment", None), "construction", None)
            project = (getattr(construction, "projects", None) or {}).get(str(decision.target_id or ""))
            if construction is not None and isinstance(project, dict) and not construction.is_design_authored(project):
                return ActionGateResult(False, decision, blockers=["design_not_applicable"], target_id=decision.target_id)
            if design is None:
                return ActionGateResult(False, decision, blockers=["design_required"], target_id=decision.target_id)
        if design is not None:
            construction = getattr(getattr(sim_state, "environment", None), "construction", None)
            project = (getattr(construction, "projects", None) or {}).get(str(decision.target_id or ""))
            if not isinstance(project, dict) or not construction.is_design_authored(project):
                return ActionGateResult(False, decision, blockers=["design_not_applicable"], target_id=decision.target_id)
            errors = construction.design_schema.validate_choice(project.get("type"), design, complete=False)
            if errors:
                return ActionGateResult(False, decision, blockers=errors, target_id=decision.target_id, project_id=project.get("id"))
        return self._pass(decision)

    def _evaluate_meta(self, *, agent, decision, sim_state):
        return self._pass(decision)

    _ACTIVE_CONSTRUCTION_PLACEHOLDERS = {"active_construction"}

    def _resolve_existing_project_id(self, *, construction, requested: Any) -> Optional[str]:
        requested_id = str(requested or "").strip()
        if not requested_id or construction is None:
            return None
        projects = getattr(construction, "projects", {}) or {}
        if requested_id in projects:
            return requested_id
        resolver = getattr(construction, "resolve_project_id", None)
        if callable(resolver):
            return resolver(requested_id, create_if_missing=False, unambiguous_only=True)
        return None

    def _resolve_active_construction_project_id(self, *, agent, construction) -> Optional[str]:
        projects = getattr(construction, "projects", {}) or {}
        if not projects:
            return None
        # ``active_construction`` is only an alias for the sole unfinished project. With several it is ambiguous, and the
        # gate does not break the tie from agent state or ordering: the Pilot must name the project.
        unfinished = [p for p in projects.values() if isinstance(p, dict) and str(p.get("status") or "") != "complete"]
        if len(unfinished) != 1:
            return None
        return str(unfinished[0].get("id")) if unfinished[0].get("id") else None

    def _legacy_active_construction_heuristic(self, *, agent, construction) -> Optional[str]:
        """Retired tie-breaker (agent focus, closure ownership, ordering); no longer reachable from the gate."""
        projects = getattr(construction, "projects", {}) or {}

        def existing(candidate: Any) -> Optional[str]:
            return self._resolve_existing_project_id(construction=construction, requested=candidate)

        for state_name in ("project_closure_state", "construction_validation_state", "transport_state"):
            state = getattr(agent, state_name, None)
            if isinstance(state, dict):
                project_id = existing(state.get("project_id"))
                if project_id:
                    return project_id

        project_id = existing(getattr(agent, "target", None))
        if project_id:
            return project_id

        current_plan = getattr(agent, "current_plan", None)
        plan_decision = getattr(current_plan, "decision", None)
        project_id = existing(getattr(plan_decision, "target_id", None))
        if project_id:
            return project_id

        active_intent = getattr(agent, "active_intent", None)
        if isinstance(active_intent, dict):
            project_id = existing(active_intent.get("target"))
            if project_id:
                return project_id

        agent_name = str(getattr(agent, "name", "") or "")
        ordered_projects = sorted(projects.values(), key=lambda p: str(p.get("id") or ""))
        for project in ordered_projects:
            if str(project.get("closure_owner") or "") == agent_name and str(project.get("status") or "") == "ready_for_validation":
                return str(project.get("id"))
        for status in ("ready_for_validation", "needs_repair"):
            for project in ordered_projects:
                if str(project.get("status") or "") == status:
                    return str(project.get("id"))

        get_active = getattr(construction, "get_active_projects", None)
        active_projects = get_active() if callable(get_active) else [p for p in ordered_projects if p.get("started") and str(p.get("status") or "") != "complete"]
        candidates = [p for p in active_projects if isinstance(p, dict) and str(p.get("status") or "") != "complete"]
        if not candidates:
            return None

        def score(project: dict[str, Any]) -> tuple[int, int, int, str]:
            actor_match = str(project.get("last_actor") or project.get("current_actor") or "") == agent_name
            return (
                0 if actor_match else 1,
                0 if project.get("structurally_complete") else 1,
                0 if project.get("resource_complete") else 1,
                str(project.get("id") or ""),
            )

        selected = sorted(candidates, key=score)[0]
        return str(selected.get("id")) if selected.get("id") else None

    def _resolve_project_id_from_decision(self, *, agent, decision: BrainDecision, sim_state) -> Optional[str]:
        construction = getattr(getattr(sim_state, "environment", None), "construction", None)
        requested = str(decision.target_id or "").strip()
        if construction is None:
            return requested or None
        project_id = self._resolve_existing_project_id(construction=construction, requested=requested)
        if project_id:
            return project_id
        if requested.lower() in self._ACTIVE_CONSTRUCTION_PLACEHOLDERS:
            return self._resolve_active_construction_project_id(agent=agent, construction=construction)
        return None

    def _evaluate_validation(self, *, agent, decision, sim_state):
        construction = getattr(getattr(sim_state, "environment", None), "construction", None)
        project_id = self._resolve_project_id_from_decision(agent=agent, decision=decision, sim_state=sim_state)
        project = (getattr(construction, "projects", {}) or {}).get(str(project_id or "")) if construction is not None else None
        if not isinstance(project, dict):
            return ActionGateResult(False, decision, blockers=["project_not_found"], target_id=decision.target_id, project_id=project_id)

        required = int((project.get("required_resources") or {}).get("bricks", 0) or 0)
        delivered = int((project.get("delivered_resources") or {}).get("bricks", 0) or 0)
        resource_complete = bool(project.get("resource_complete")) or (required > 0 and delivered >= required)
        structurally_complete = bool(project.get("structurally_complete"))
        progress = project.get("progress")
        if not structurally_complete:
            reroutes: list[BrainDecision] = []
            if resource_complete:
                reroutes.append(BrainDecision(ExecutableActionType.START_CONSTRUCTION, target_id=project_id, reason_summary="Physical construction is required before validation.", confidence=1.0))
                reroutes.append(BrainDecision(ExecutableActionType.CONTINUE_CONSTRUCTION, target_id=project_id, reason_summary="Continue physical construction before validation.", confidence=1.0))
            else:
                reroutes.append(BrainDecision(ExecutableActionType.TRANSPORT_RESOURCES, target_id=project_id, reason_summary="Materials must be delivered before construction and validation.", confidence=1.0))
            blockers = ["physical_incomplete"]
            if not bool(project.get("started")) or not progress:
                blockers.insert(0, "physical_build_not_started")
            return ActionGateResult(
                False,
                decision,
                blockers=blockers,
                available_reroutes=reroutes,
                target_id=decision.target_id,
                project_id=project_id,
                metadata={"required_resources": required, "delivered_resources": delivered, "resource_complete": resource_complete, "structurally_complete": structurally_complete, "progress": progress},
            )

        readiness = construction.evaluate_project_validation_readiness(project_id) if hasattr(construction, "evaluate_project_validation_readiness") else {"validation_ready": True, "blockers": []}
        if getattr(agent, "is_human_participant", lambda: False)():
            # A person may ATTEMPT validation whenever it is physically/functionally possible. Rule knowledge and workspace evidence are simulated
            # cognition; they are not a legality condition for a human.
            readiness = {**readiness, "validation_ready": bool(readiness.get("physical_ready")) and bool(readiness.get("support_ready")),
                         "blockers": [b for b in (readiness.get("blockers") or []) if b in {"physical_incomplete", "missing_functional_support"}]}
        if not bool(readiness.get("validation_ready", False)):
            return ActionGateResult(False, decision, blockers=list(readiness.get("blockers") or ["validation_not_ready"]), target_id=decision.target_id, project_id=project_id, metadata={"validation_readiness": readiness})
        return ActionGateResult(True, decision, target_id=decision.target_id, project_id=project_id, metadata={"validation_readiness": readiness})
