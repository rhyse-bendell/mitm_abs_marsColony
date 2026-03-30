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

    def summarize(self) -> Dict[str, Any]:
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
            "recent_updates": self.recent_updates[-10:],
        }
