from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TeamArtifact:
    artifact_id: str
    artifact_type: str
    summary: str
    content: Dict[str, Any]
    author: str
    created_at: float
    source: str = "agent_externalization"
    uptake_count: int = 0
    contributors: List[str] = field(default_factory=list)
    knowledge_summary: List[str] = field(default_factory=list)
    validation_state: str = "unvalidated"
    consulted_by: List[str] = field(default_factory=list)


@dataclass
class TeamKnowledgeManager:
    validated_knowledge: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, TeamArtifact] = field(default_factory=dict)
    recent_updates: List[Dict[str, Any]] = field(default_factory=list)

    def add_validated_knowledge(self, key: str, summary: str, author: str, sim_time: float) -> None:
        self.validated_knowledge[key] = summary
        self.recent_updates.append(
            {"event": "validated_knowledge", "key": key, "author": author, "time": sim_time}
        )

    def externalize_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        summary: str,
        content: Dict[str, Any],
        author: str,
        sim_time: float,
        source: str = "agent_externalization",
        contributors: Optional[List[str]] = None,
        knowledge_summary: Optional[List[str]] = None,
        validation_state: str = "unvalidated",
    ) -> TeamArtifact:
        artifact = TeamArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            summary=summary,
            content=content,
            author=author,
            created_at=sim_time,
            source=source,
            contributors=list(contributors or []),
            knowledge_summary=list(knowledge_summary or []),
            validation_state=validation_state,
        )
        self.artifacts[artifact_id] = artifact
        self.recent_updates.append(
            {"event": "externalized_artifact", "artifact_id": artifact_id, "author": author, "time": sim_time}
        )
        return artifact

    def get_artifact(self, artifact_id: str) -> Optional[TeamArtifact]:
        return self.artifacts.get(artifact_id)

    def adopt_artifact(self, artifact_id: str, agent_name: str, sim_time: float) -> Optional[TeamArtifact]:
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return None
        artifact.uptake_count += 1
        if agent_name not in artifact.consulted_by:
            artifact.consulted_by.append(agent_name)
        self.recent_updates.append(
            {"event": "artifact_uptake", "artifact_id": artifact_id, "agent": agent_name, "time": sim_time}
        )
        return artifact

    def upsert_construction_artifact(self, project: Dict[str, Any], sim_time: float) -> Optional[TeamArtifact]:
        project_id = project.get("id")
        if not project_id:
            return None
        artifact_id = f"construction:{project_id}"
        structure_type = project.get("type", "unknown")
        delivered = int(project.get("delivered_resources", {}).get("bricks", 0) or 0)
        required = int(project.get("required_resources", {}).get("bricks", 0) or 0)
        status = project.get("status", "in_progress")
        resource_complete = bool(project.get("resource_complete", False)) or (required > 0 and delivered >= required)
        progress_ratio = min(1.0, (delivered / required)) if required > 0 else 0.0
        validated = bool(project.get("validated_complete", False))
        validation_state = "validated" if project.get("correct", True) and status == "complete" else (
            "mismatch" if project.get("correct") is False else "in_progress"
        )
        summary = f"{structure_type} progress={delivered}/{required} status={status}"
        knowledge_summary = list(project.get("expected_rules", []))
        contributors = sorted(project.get("builders", set())) if isinstance(project.get("builders"), set) else []
        provenance = dict(project.get("provenance") or {})
        held_rule_ids = list(provenance.get("held_rule_ids_at_build", []))
        held_information_ids = list(provenance.get("held_information_ids_at_build", []))
        held_data_ids = list(provenance.get("held_data_ids_at_build", []))
        content = {
            "project_id": project_id,
            "structure_type": structure_type,
            "status": status,
            "resource_complete": resource_complete,
            "progress_ratio": round(progress_ratio, 4),
            "validated": validated,
            "correct": project.get("correct", True),
            "expected_rules": knowledge_summary,
            "held_rule_ids_at_build": held_rule_ids,
            "held_information_ids_at_build": held_information_ids,
            "held_data_ids_at_build": held_data_ids,
            "held_expected_rules_locally": bool(provenance.get("held_expected_rules_locally", False)),
            "missing_expected_rules": list(provenance.get("missing_expected_rules", [])),
            "team_rule_snapshot_ids": list(provenance.get("team_rule_snapshot_ids", [])),
            "acting_agent": provenance.get("last_actor") or project.get("last_actor"),
            "contributors": list(provenance.get("contributors", contributors)),
            "last_update_time": provenance.get("last_update_time", sim_time),
            "provenance_timeline": list(provenance.get("timeline", [])),
            "delivered_resources": dict(project.get("delivered_resources", {})),
            "required_resources": dict(project.get("required_resources", {})),
            "last_status_event": "construction_externalized",
            "status_changed_at": sim_time,
        }

        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            artifact_type = project.get("artifact_type", f"construction_{structure_type}")
            if isinstance(artifact_type, str) and not artifact_type.startswith("construction_"):
                artifact_type = f"construction_{structure_type}"
            artifact = self.externalize_artifact(
                artifact_id=artifact_id,
                artifact_type=artifact_type,
                summary=summary,
                content=content,
                author=project.get("author", "system"),
                sim_time=sim_time,
                source="construction_state",
                contributors=contributors,
                knowledge_summary=knowledge_summary,
                validation_state=validation_state,
            )
            initial_event = (
                "construction_completed"
                if status == "complete"
                else "construction_ready_for_validation"
                if status == "ready_for_validation"
                else "construction_materials_satisfied"
                if resource_complete
                else "construction_status_changed"
            )
            artifact.content["last_status_event"] = initial_event
            self.recent_updates.append(
                {"event": "construction_externalized", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time}
            )
            self.recent_updates.append(
                {"event": initial_event, "artifact_id": artifact_id, "project_id": project_id, "status": status, "resource_complete": resource_complete, "time": sim_time}
            )
            self.recent_updates.append(
                {"event": "project_state_published_to_team_knowledge", "artifact_id": artifact_id, "project_id": project_id, "status": status, "resource_complete": resource_complete, "time": sim_time}
            )
            return artifact

        previous = dict(artifact.content or {})
        previous_status = str(previous.get("status", "") or "")
        previous_resource_complete = bool(previous.get("resource_complete", False))
        previous_correct = bool(previous.get("correct", True))
        status_event = "construction_artifact_updated"

        if previous_status != status:
            status_event = "construction_status_changed"
            self.recent_updates.append(
                {
                    "event": "construction_status_changed",
                    "artifact_id": artifact_id,
                    "project_id": project_id,
                    "status_before": previous_status,
                    "status_after": status,
                    "time": sim_time,
                }
            )
        if (not previous_resource_complete) and resource_complete:
            status_event = "construction_materials_satisfied"
            self.recent_updates.append(
                {"event": "construction_materials_satisfied", "artifact_id": artifact_id, "project_id": project_id, "status": status, "time": sim_time}
            )
            self.recent_updates.append(
                {"event": "project_marked_materially_satisfied", "artifact_id": artifact_id, "project_id": project_id, "status": status, "time": sim_time}
            )
        if previous_status != "ready_for_validation" and status == "ready_for_validation":
            status_event = "construction_ready_for_validation"
            self.recent_updates.append(
                {"event": "construction_ready_for_validation", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time}
            )
            self.recent_updates.append(
                {"event": "project_marked_ready_for_validation", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time}
            )
        if previous_status != "complete" and status == "complete":
            status_event = "construction_completed"
            self.recent_updates.append(
                {"event": "construction_completed", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time}
            )
        if previous_correct is False and bool(project.get("correct", True)):
            self.recent_updates.append(
                {"event": "construction_corrected", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time}
            )

        content["last_status_event"] = status_event
        content["status_changed_at"] = sim_time if (
            previous_status != status
            or previous_resource_complete != resource_complete
            or previous_correct != bool(project.get("correct", True))
        ) else previous.get("status_changed_at", sim_time)

        changed = (
            artifact.summary != summary
            or artifact.validation_state != validation_state
            or artifact.content != content
            or artifact.knowledge_summary != knowledge_summary
            or artifact.contributors != contributors
        )
        artifact.summary = summary
        artifact.content = content
        artifact.validation_state = validation_state
        artifact.knowledge_summary = knowledge_summary
        artifact.contributors = contributors
        if changed:
            self.recent_updates.append({"event": "construction_artifact_updated", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time})
            self.recent_updates.append({"event": "construction_artifact_provenance_updated", "artifact_id": artifact_id, "project_id": project_id, "time": sim_time})
            self.recent_updates.append(
                {
                    "event": "project_state_published_to_team_knowledge",
                    "artifact_id": artifact_id,
                    "project_id": project_id,
                    "status": status,
                    "resource_complete": resource_complete,
                    "last_status_event": status_event,
                    "time": sim_time,
                }
            )
        return artifact

    @staticmethod
    def _team_plan_artifact_id(plan_id: str) -> str:
        return f"team_plan:{plan_id}"

    @staticmethod
    def _team_plan_summary(content: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "plan_id": str(content.get("plan_id", "")),
            "status": str(content.get("status", "")),
            "goal_ids": list(content.get("goal_ids", [])),
            "goal_summary": content.get("goal_summary", ""),
            "project_targets": list(content.get("project_targets", [])),
            "assignments_by_role": dict(content.get("assignments_by_role", {})),
            "trigger_reason": content.get("trigger_reason", ""),
            "blocked_reasons": list(content.get("blocked_reasons", [])),
            "supporters": list(content.get("supporters", [])),
            "opposers": list(content.get("opposers", [])),
            "review_at": content.get("review_at"),
            "expires_at": content.get("expires_at"),
            "last_plan_event": content.get("last_plan_event"),
            "plan_event_time": content.get("plan_event_time"),
        }

    def list_team_plans(self) -> List[Dict[str, Any]]:
        plans: List[Dict[str, Any]] = []
        for artifact in self.artifacts.values():
            if artifact.artifact_type != "team_plan" or not isinstance(artifact.content, dict):
                continue
            content = dict(artifact.content)
            summary = self._team_plan_summary(content)
            summary["artifact_id"] = artifact.artifact_id
            summary["author"] = artifact.author
            summary["created_at"] = artifact.created_at
            plans.append(summary)
        plans.sort(key=lambda item: float(item.get("plan_event_time") or item.get("created_at") or 0.0), reverse=True)
        return plans

    def get_active_team_plan(self) -> Optional[Dict[str, Any]]:
        """
        Deterministic active-plan selection:
        1) newest committed team plan by plan_event_time/created_at
        2) else newest proposed team plan by plan_event_time/created_at
        """
        plans = self.list_team_plans()
        committed = [p for p in plans if p.get("status") == "committed"]
        if committed:
            return committed[0]
        proposed = [p for p in plans if p.get("status") == "proposed"]
        if proposed:
            return proposed[0]
        return None

    def propose_team_plan(
        self,
        *,
        plan_id: str,
        proposed_by: str,
        sim_time: float,
        goal_ids: Optional[List[str]] = None,
        goal_summary: str = "",
        trigger_reason: str = "",
        project_targets: Optional[List[str]] = None,
        assignments_by_role: Optional[Dict[str, Any]] = None,
        evidence_refs: Optional[List[str]] = None,
        blocked_reasons: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        review_at: Optional[float] = None,
        expires_at: Optional[float] = None,
        supporters: Optional[List[str]] = None,
        opposers: Optional[List[str]] = None,
        consulted_by: Optional[List[str]] = None,
        supersedes_plan_id: Optional[str] = None,
    ) -> TeamArtifact:
        artifact_id = self._team_plan_artifact_id(plan_id)
        existing = self.artifacts.get(artifact_id)
        previous = dict(existing.content) if existing and isinstance(existing.content, dict) else {}
        event_name = "team_plan_updated" if existing is not None else "team_plan_proposed"
        content = {
            "plan_id": plan_id,
            "status": "proposed",
            "proposed_by": proposed_by or previous.get("proposed_by"),
            "proposed_at": previous.get("proposed_at", sim_time),
            "committed_by": previous.get("committed_by"),
            "committed_at": previous.get("committed_at"),
            "goal_ids": list(goal_ids if goal_ids is not None else previous.get("goal_ids", [])),
            "goal_summary": goal_summary or str(previous.get("goal_summary", "")),
            "trigger_reason": trigger_reason or str(previous.get("trigger_reason", "")),
            "project_targets": list(project_targets if project_targets is not None else previous.get("project_targets", [])),
            "assignments_by_role": dict(assignments_by_role if assignments_by_role is not None else previous.get("assignments_by_role", {})),
            "evidence_refs": list(evidence_refs if evidence_refs is not None else previous.get("evidence_refs", [])),
            "blocked_reasons": list(blocked_reasons if blocked_reasons is not None else previous.get("blocked_reasons", [])),
            "success_criteria": list(success_criteria if success_criteria is not None else previous.get("success_criteria", [])),
            "review_at": review_at if review_at is not None else previous.get("review_at"),
            "expires_at": expires_at if expires_at is not None else previous.get("expires_at"),
            "last_plan_event": event_name,
            "plan_event_time": sim_time,
            "supporters": list(supporters if supporters is not None else previous.get("supporters", [])),
            "opposers": list(opposers if opposers is not None else previous.get("opposers", [])),
            "consulted_by": list(consulted_by if consulted_by is not None else previous.get("consulted_by", [])),
            "supersedes_plan_id": supersedes_plan_id if supersedes_plan_id is not None else previous.get("supersedes_plan_id"),
        }
        summary = f"team_plan {plan_id} proposed: {content['goal_summary']}".strip()
        if existing is None:
            artifact = self.externalize_artifact(
                artifact_id=artifact_id,
                artifact_type="team_plan",
                summary=summary,
                content=content,
                author=proposed_by or "system",
                sim_time=sim_time,
                source="team_plan_lifecycle",
                contributors=[proposed_by] if proposed_by else [],
                validation_state="tentative",
            )
        else:
            existing.summary = summary
            existing.content = content
            existing.source = "team_plan_lifecycle"
            if proposed_by and proposed_by not in existing.contributors:
                existing.contributors.append(proposed_by)
            existing.validation_state = "tentative"
            artifact = existing
        self.recent_updates.append({"event": event_name, "artifact_id": artifact_id, "plan_id": plan_id, "author": proposed_by, "time": sim_time})
        if event_name != "team_plan_proposed":
            self.recent_updates.append({"event": "team_plan_proposed", "artifact_id": artifact_id, "plan_id": plan_id, "author": proposed_by, "time": sim_time})
        return artifact

    def commit_team_plan(self, *, plan_id: str, committed_by: str, sim_time: float) -> Optional[TeamArtifact]:
        artifact_id = self._team_plan_artifact_id(plan_id)
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return None
        content = dict(artifact.content or {})
        content["status"] = "committed"
        content["committed_by"] = committed_by
        content["committed_at"] = sim_time
        content["last_plan_event"] = "team_plan_committed"
        content["plan_event_time"] = sim_time
        artifact.content = content
        artifact.summary = f"team_plan {plan_id} committed: {content.get('goal_summary', '')}".strip()
        artifact.validation_state = "validated"
        if committed_by and committed_by not in artifact.contributors:
            artifact.contributors.append(committed_by)
        self.recent_updates.append({"event": "team_plan_committed", "artifact_id": artifact_id, "plan_id": plan_id, "author": committed_by, "time": sim_time})
        return artifact

    def record_team_plan_response(
        self,
        *,
        plan_id: str,
        responder: str,
        response_type: str,
        sim_time: float,
        role: Optional[str] = None,
        reason: str = "",
    ) -> Optional[TeamArtifact]:
        artifact_id = self._team_plan_artifact_id(plan_id)
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return None
        content = dict(artifact.content or {})
        normalized = str(response_type or "").strip().lower()
        supporters = set(content.get("supporters", []) or [])
        opposers = set(content.get("opposers", []) or [])
        clarification_requests = list(content.get("clarification_requests", []) or [])
        assignment_ack = dict(content.get("assignment_acknowledged_by", {}) or {})
        assignment_declined = dict(content.get("assignment_declined_by", {}) or {})

        event_name = "team_plan_updated"
        if normalized == "agree":
            supporters.add(responder)
            opposers.discard(responder)
            event_name = "team_plan_agreed"
        elif normalized == "disagree":
            opposers.add(responder)
            supporters.discard(responder)
            event_name = "team_plan_disagreed"
        elif normalized == "request_clarification":
            clarification_requests.append({"agent": responder, "role": role, "reason": reason, "time": sim_time})
            event_name = "team_plan_clarification_requested"
        elif normalized == "assignment_accept":
            assignment_ack[str(role or responder)] = {"agent": responder, "reason": reason, "time": sim_time}
            event_name = "team_plan_assignment_accepted"
        elif normalized == "assignment_decline":
            assignment_declined[str(role or responder)] = {"agent": responder, "reason": reason, "time": sim_time}
            event_name = "team_plan_assignment_declined"

        content["supporters"] = sorted(supporters)
        content["opposers"] = sorted(opposers)
        content["clarification_requests"] = clarification_requests[-10:]
        content["assignment_acknowledged_by"] = assignment_ack
        content["assignment_declined_by"] = assignment_declined
        content["last_plan_event"] = event_name
        artifact.content = content
        artifact.summary = f"team_plan {plan_id} {content.get('status', 'proposed')}: {content.get('goal_summary', '')}".strip()
        if responder and responder not in artifact.contributors:
            artifact.contributors.append(responder)

        self.recent_updates.append(
            {"event": event_name, "artifact_id": artifact_id, "plan_id": plan_id, "agent": responder, "role": role, "reason": reason, "time": sim_time}
        )
        self.recent_updates.append(
            {"event": "team_plan_updated", "artifact_id": artifact_id, "plan_id": plan_id, "agent": responder, "response_type": normalized, "time": sim_time}
        )
        return artifact

    def update_team_plan_assignments(
        self,
        *,
        plan_id: str,
        assignments_by_role: Dict[str, Any],
        updated_by: str,
        sim_time: float,
        reason: str = "",
    ) -> Optional[TeamArtifact]:
        artifact_id = self._team_plan_artifact_id(plan_id)
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return None
        content = dict(artifact.content or {})
        content["assignments_by_role"] = dict(assignments_by_role or {})
        content["last_assignment_revision"] = {"updated_by": updated_by, "reason": reason, "time": sim_time}
        content["last_plan_event"] = "team_plan_assignments_updated"
        content["plan_event_time"] = sim_time
        artifact.content = content
        if updated_by and updated_by not in artifact.contributors:
            artifact.contributors.append(updated_by)
        self.recent_updates.append(
            {"event": "team_plan_assignments_updated", "artifact_id": artifact_id, "plan_id": plan_id, "agent": updated_by, "reason": reason, "time": sim_time}
        )
        self.recent_updates.append(
            {"event": "team_plan_updated", "artifact_id": artifact_id, "plan_id": plan_id, "agent": updated_by, "reason": reason, "time": sim_time}
        )
        return artifact

    def summarize(self) -> Dict[str, Any]:
        team_plan_summaries = self.list_team_plans()
        return {
            "validated_knowledge": dict(self.validated_knowledge),
            "artifact_summaries": {
                aid: {
                    "type": artifact.artifact_type,
                    "summary": artifact.summary,
                    "author": artifact.author,
                    "created_at": artifact.created_at,
                    "uptake_count": artifact.uptake_count,
                    "validation_state": artifact.validation_state,
                    "consulted_by": list(artifact.consulted_by),
                }
                for aid, artifact in self.artifacts.items()
            },
            "active_team_plan": self.get_active_team_plan(),
            "team_plan_summaries": team_plan_summaries,
            "team_plan_recent_updates": [u for u in self.recent_updates if str(u.get("event", "")).startswith("team_plan_")][-12:],
            "recent_updates": self.recent_updates[-10:],
        }
