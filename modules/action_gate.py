"""Universal mecha-side action legality gate.

ActionGate evaluates pilot-selected cockpit actions before legacy execution. It
reports blockers and possible reroutes without executing actions or choosing
strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from modules.action_schema import BrainDecision, ExecutableActionType


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
        if action == ExecutableActionType.INSPECT_INFORMATION_SOURCE:
            return self._evaluate_inspect(agent=agent, decision=decision, sim_state=sim_state)
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
        return self._evaluate_meta(agent=agent, decision=decision, sim_state=sim_state)

    def _pass(self, decision: BrainDecision) -> ActionGateResult:
        # TODO(architecture): expand family-specific legality checks here.
        return ActionGateResult(legal=True, normalized_decision=decision, target_id=decision.target_id)

    def _evaluate_inspect(self, *, agent, decision, sim_state):
        return self._pass(decision)

    def _evaluate_communication(self, *, agent, decision, sim_state):
        return self._pass(decision)

    def _evaluate_artifact(self, *, agent, decision, sim_state):
        return self._pass(decision)

    def _evaluate_transport(self, *, agent, decision, sim_state):
        return self._pass(decision)

    def _evaluate_construction(self, *, agent, decision, sim_state):
        return self._pass(decision)

    def _evaluate_repair(self, *, agent, decision, sim_state):
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
            return resolver(requested_id, create_if_missing=False)
        return None

    def _resolve_active_construction_project_id(self, *, agent, construction) -> Optional[str]:
        projects = getattr(construction, "projects", {}) or {}
        if not projects:
            return None

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
        if not bool(readiness.get("validation_ready", False)):
            return ActionGateResult(False, decision, blockers=list(readiness.get("blockers") or ["validation_not_ready"]), target_id=decision.target_id, project_id=project_id, metadata={"validation_readiness": readiness})
        return ActionGateResult(True, decision, target_id=decision.target_id, project_id=project_id, metadata={"validation_readiness": readiness})
