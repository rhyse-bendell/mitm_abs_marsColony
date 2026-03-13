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
    ) -> TeamArtifact:
        artifact = TeamArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            summary=summary,
            content=content,
            author=author,
            created_at=sim_time,
            source=source,
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
        self.recent_updates.append(
            {"event": "artifact_uptake", "artifact_id": artifact_id, "agent": agent_name, "time": sim_time}
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
                }
                for aid, artifact in self.artifacts.items()
            },
            "recent_updates": self.recent_updates[-10:],
        }
