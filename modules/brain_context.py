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

    def _affordances(self, agent, environment) -> List[Dict[str, Any]]:
        legal = [
            {"action_type": ExecutableActionType.OBSERVE_ENVIRONMENT.value, "target_id": None},
            {"action_type": ExecutableActionType.REASSESS_PLAN.value, "target_id": None},
            {"action_type": ExecutableActionType.WAIT.value, "target_id": None},
            {"action_type": ExecutableActionType.COMMUNICATE.value, "target_id": "nearby_agent"},
        ]

        for target_name, target in environment.interaction_targets.items():
            action_type = (
                ExecutableActionType.INSPECT_INFORMATION_SOURCE
                if target.get("kind") == "information"
                else ExecutableActionType.START_CONSTRUCTION
            )
            legal.append(
                {
                    "action_type": action_type.value,
                    "target_id": target_name,
                    "target_zone": target.get("zone"),
                }
            )

        legal.append(
            {
                "action_type": ExecutableActionType.TRANSPORT_RESOURCES.value,
                "target_id": "resource_zone_to_work_zone",
                "duration_s": 30.0,
            }
        )
        return legal

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
            "accessible_info_sources": list(environment.knowledge_packets.keys()),
            "work_zones": list(environment.interaction_targets.keys()),
            "built_structures": list(environment.construction.projects.values()),
            "in_progress_projects": environment.construction.get_active_projects(),
            "resource_status": {"visible_resources": environment.get_visible_resources(agent.position)},
            "blocked_or_unreachable": [],
            "legal_actions": self._affordances(agent, environment),
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
            "packets_inspected": list(agent.memory_seen_packets),
            "recent_failed_attempts": [
                e for e in history_bands["near_preceding_events"] if "blocked" in e.lower() or "could not" in e.lower()
            ],
            "active_plan": {
                "plan_id": getattr(getattr(agent, "current_plan", None), "plan_id", None),
                "created_at": getattr(getattr(agent, "current_plan", None), "created_at", None),
                "last_reviewed_at": getattr(getattr(agent, "current_plan", None), "last_reviewed_at", None),
                "invalidation_reason": getattr(getattr(agent, "current_plan", None), "invalidation_reason", None),
            },
        }

        team_state = {
            "team_shared_knowledge": sim_state.team_knowledge_manager.summarize(),
            "teammate_roles": {other.name: other.role for other in sim_state.agents if other.name != agent.name},
            "teammate_inferred_goals": {
                teammate: model.get("goals", []) for teammate, model in agent.theory_of_mind.items()
            },
            "tom_summary": agent.theory_of_mind,
            "recent_shared_updates": sim_state.team_knowledge_manager.recent_updates[-5:],
        }

        return BrainContextPacket(
            static_task_context=static_task_context,
            world_snapshot=world_snapshot,
            individual_cognitive_state=individual_cognitive_state,
            team_state=team_state,
            history_bands=history_bands,
            action_affordances=self._affordances(agent, environment),
        )
