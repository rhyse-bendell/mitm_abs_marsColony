"""Canonical, controller-neutral context: the only agent view any controller (Procedural Baseline, LLM-backed, future) receives.

`build_canonical_context(packet, ...)` projects the middleware's BrainContextPacket onto an explicit whitelist of LEGITIMATE agent state:
identity, goals, observations, held DIK, known gaps, received communications/artifacts, the agent's known/adopted team-plan context,
recent outcomes/blockers (including the most recent rejected action) and the action candidates, split into `executable_actions` (currently
selectable) and `blocked_actions` (visible for reasoning, with blocker reasons, never selectable).

Excluded by construction (never copied): hidden world truth (e.g. construction correctness), other agents' private DIK, the Environment
or SimulationState objects, logs and the filesystem, and the live team-knowledge store. `assert_canonical_context_clean` rejects a
context that carries any forbidden key at any depth (used by the middleware before a request is handed to a controller and by tests).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

SCHEMA = "controller_context/2"

FORBIDDEN_KEYS = frozenset({
    "correct", "hidden_correct", "build_error", "environment", "sim_state", "simulation_state", "env_ref", "_env_ref",
    "mental_model", "team_knowledge_manager", "logger", "log_path", "session_folder", "filesystem",
})

_WORLD_KEYS = ("sim_time", "phase_profile", "agent_position", "nearby_agents", "built_state", "site_capacity_summary", "resource_status",
               "brick_source_available", "capacity_unlock", "support_deficits")


def affordance_blockers(aff: Dict[str, Any]) -> list[str]:
    """Blocker tokens the affordance builder itself states for an offered action (existing affordance truth; no legality is computed here).

    An action is advertised-but-not-selectable when the builder marks it not ready: REPAIR carries `repair_ready` + `repair_blockers`
    (the same check that gates its execution), VALIDATE carries `validation_ready` + `validation_blockers` (the gate's readiness read), and
    INSPECT of another role's private source carries `source_accessible: False`. Returns [] for an action that is currently selectable."""
    if aff.get("action_type") == "repair_or_correct_construction" and aff.get("repair_ready") is False:
        return [str(b) for b in (aff.get("repair_blockers") or [])] or ["repair_not_ready"]
    if aff.get("action_type") == "validate_construction" and aff.get("validation_ready") is False:
        # same readiness the ActionGate applies at commit (carried on the affordance by the builder)
        return [str(b) for b in (aff.get("validation_blockers") or [])] or ["validation_not_ready"]
    if aff.get("action_type") == "inspect_information_source" and aff.get("source_accessible") is False:
        return [str(aff.get("source_blocker") or "private_source_inaccessible")]
    return []


def partition_affordances(affordances: Iterable[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    """-> (executable, blocked). Executable actions are currently selectable and are the ONLY ones a generative controller may be given ids for;
    blocked actions stay visible for reasoning with their blocker reasons and are never selectable."""
    executable: list[Dict[str, Any]] = []
    blocked: list[Dict[str, Any]] = []
    for aff in affordances or []:
        reasons = affordance_blockers(aff)
        if reasons:
            blocked.append({**aff, "blockers": reasons})
        else:
            executable.append(aff)
    return executable, blocked


def last_rejection(outcomes: Iterable[Dict[str, Any]]) -> Dict[str, Any] | None:
    """The most recent rejected (blocked) action outcome the agent recorded, with how many outcomes have been recorded since."""
    rows = [o for o in (outcomes or []) if isinstance(o, dict)]
    for i in range(len(rows) - 1, -1, -1):
        if rows[i].get("status") == "blocked":
            row = {k: rows[i].get(k) for k in ("time", "action", "target", "blocker", "consecutive_identical_failures") if rows[i].get(k) is not None}
            row["outcomes_since"] = len(rows) - 1 - i
            return row
    return None


def _pick(src: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: src[k] for k in keys if k in src}


def build_canonical_context(packet, *, agent_id: str, role: str, controller_type: str) -> Dict[str, Any]:
    cog = packet.individual_cognitive_state or {}
    team = packet.team_state or {}
    world = packet.world_snapshot or {}
    static = packet.static_task_context or {}
    executable, blocked = partition_affordances(packet.action_affordances or [])
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
        "last_rejection": last_rejection(cog.get("recent_action_outcomes") or []),
        "executable_actions": executable,
        "blocked_actions": blocked,
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
