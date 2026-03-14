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
        delivered = project.get("delivered_resources", {}).get("bricks", 0)
        required = project.get("required_resources", {}).get("bricks", 0)
        status = project.get("status", "in_progress")
        validation_state = "validated" if project.get("correct", True) and status == "complete" else (
            "mismatch" if project.get("correct") is False else "in_progress"
        )
        summary = f"{structure_type} progress={delivered}/{required} status={status}"
        knowledge_summary = list(project.get("expected_rules", []))
        contributors = sorted(project.get("builders", set())) if isinstance(project.get("builders"), set) else []
        content = {
            "project_id": project_id,
            "structure_type": structure_type,
            "status": status,
            "correct": project.get("correct", True),
            "expected_rules": knowledge_summary,
            "delivered_resources": dict(project.get("delivered_resources", {})),
            "required_resources": dict(project.get("required_resources", {})),
        }

        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            artifact = self.externalize_artifact(
                artifact_id=artifact_id,
                artifact_type=f"construction_{structure_type}",
                summary=summary,
                content=content,
                author=project.get("author", "system"),
                sim_time=sim_time,
                source="construction_state",
                contributors=contributors,
                knowledge_summary=knowledge_summary,
                validation_state=validation_state,
            )
            self.recent_updates.append(
                {"event": "construction_externalized", "artifact_id": artifact_id, "time": sim_time}
            )
            return artifact

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
            self.recent_updates.append({"event": "construction_artifact_updated", "artifact_id": artifact_id, "time": sim_time})
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
