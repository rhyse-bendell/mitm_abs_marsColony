"""Canonical, controller-neutral context: the only agent view any controller (Procedural Baseline, LLM-backed, future) receives.

`build_canonical_context(packet, ...)` projects the middleware's BrainContextPacket onto an explicit whitelist of LEGITIMATE agent state:
identity, goals, observations, held DIK, known gaps, received communications/artifacts, the agent's known/adopted team-plan context,
recent outcomes/blockers and the currently exposed legal action candidates.

Excluded by construction (never copied): hidden world truth (e.g. construction correctness), other agents' private DIK, the Environment
or SimulationState objects, logs and the filesystem, and the live team-knowledge store. `assert_canonical_context_clean` rejects a
context that carries any forbidden key at any depth (used by the middleware before a request is handed to a controller and by tests).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

SCHEMA = "controller_context/1"

FORBIDDEN_KEYS = frozenset({
    "correct", "hidden_correct", "build_error", "environment", "sim_state", "simulation_state", "env_ref", "_env_ref",
    "mental_model", "team_knowledge_manager", "logger", "log_path", "session_folder", "filesystem",
})

_WORLD_KEYS = ("sim_time", "phase_profile", "agent_position", "nearby_agents", "built_state", "site_capacity_summary", "resource_status",
               "brick_source_available", "capacity_unlock", "support_deficits")


def _pick(src: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: src[k] for k in keys if k in src}


def build_canonical_context(packet, *, agent_id: str, role: str, controller_type: str) -> Dict[str, Any]:
    cog = packet.individual_cognitive_state or {}
    team = packet.team_state or {}
    world = packet.world_snapshot or {}
    static = packet.static_task_context or {}
    ctx = {
        "schema": SCHEMA,
        "identity": {"agent_id": agent_id, "role": role, "controller_type": controller_type},
        "task": _pick(static, ("mission", "current_phase", "high_level_objectives", "hard_constraints", "role_access_constraints")),
        "goals": {"active_goal": cog.get("active_goal"), "goal_stack": list(cog.get("goal_stack") or [])},
        "observations": _pick(world, _WORLD_KEYS),
        "held_dik": {
            "data": list(cog.get("data_summary") or []),
            "information": list(cog.get("information_summary") or []),
            "knowledge": list(cog.get("knowledge_summary") or []),
        },
        "known_gaps": list(cog.get("known_gaps") or []),
        "communications_and_artifacts": {
            "externalized_artifacts": list(team.get("externalized_artifacts") or []),
            "recent_shared_updates": list(team.get("recent_shared_updates") or []),
            "recent_construction_updates": list(team.get("recent_construction_updates") or []),
        },
        "team_plan_context": team.get("team_plan_context"),
        "recent_outcomes": {
            "recent_action_outcomes": list(cog.get("recent_action_outcomes") or []),
            "recent_failed_attempts": list(cog.get("recent_failed_attempts") or []),
            "near_preceding_events": list((packet.history_bands or {}).get("near_preceding_events") or []),
        },
        "legal_actions": list(packet.action_affordances or []),
    }
    return ctx


def find_forbidden_keys(value: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            here = f"{path}.{k}" if path else str(k)
            if str(k) in FORBIDDEN_KEYS:
                found.append(here)
            found.extend(find_forbidden_keys(v, here))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            found.extend(find_forbidden_keys(v, f"{path}[{i}]"))
    return found


def assert_canonical_context_clean(ctx: Dict[str, Any]) -> Dict[str, Any]:
    bad = find_forbidden_keys(ctx)
    if bad:
        raise ValueError(f"canonical controller context carries forbidden keys: {bad[:5]}")
    return ctx
