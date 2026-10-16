"""Prompt construction and deterministic budgeting for the Local Ollama Pilot.

The prompt is built ONLY from the controller-neutral canonical context (modules/controller_context.py). Nothing else about the simulator
reaches the model. Budgeting is by priority tiers, never by truncating the serialised text:

  tier 1 (always sent, never dropped): identity, task, goals, observations, known gaps/blockers, team-plan assignment, legal affordances
  tier 2 (as budget allows, item by item): held DIK, communications/artifacts, recent action outcomes
  tier 3 (as budget allows, item by item): older history and lower-relevance context

If tier 1 alone exceeds the budget, only its optional detail is shrunk, in a fixed order (observation items, then optional affordance fields,
then gaps); the legal affordance list itself is never reduced. Omissions are counted and recorded; omitted content is never summarised
from anything outside the canonical context.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

PROMPT_VERSION = "canonical_choice_v1"

SYSTEM_PROMPTS = {
    "canonical_choice_v1": (
        "You control one crew member in a Mars colony construction task. You know only what appears in the CONTEXT below; "
        "nothing else about the world is available to you. Each turn you must choose exactly ONE of the listed legal_actions "
        "by its affordance_id. You cannot create new actions, targets or arguments. "
        "Reply with a single JSON object and nothing else: "
        '{"affordance_id": "<one id from legal_actions>", "arguments": {}, "decision_note": "<optional, at most one short sentence>"}. '
        "Put arguments only if the chosen action lists action_args; copy them unchanged."
    ),
}

AFFORDANCE_ESSENTIAL = ("action_type", "target_id", "action_args", "reachable", "plan_assignment")
AFFORDANCE_OPTIONAL = (
    "target_class", "start_mode", "work_stage", "duration_s", "required_resources", "materials_staged", "materials_deficit", "repair_ready", "repair_blockers",
    "validation_failed", "validation_candidate", "observed_repair_need", "observed_failure_age_s", "consult_useful", "consult_redundant", "consult_reasons",
    "consult_artifact_type", "provision_unsolicited", "provision_solicited", "dik_gap_count", "externalize_gap", "team_plan_opportunity",
    "addresses_support_subgoals", "serves_task_methods", "reinspect_informative", "source_state",
)
BUILT_STATE_KEYS = ("structure_id", "structure_type", "project_status", "state", "material_progress", "physical_build_progress", "resource_complete",
                    "required_resources", "delivered_resources", "ready_for_validation", "validation_failed", "observed_repair_need")


def estimate_tokens(text: str) -> int:
    """Deterministic, tokenizer-free, deliberately conservative estimate (2.8 characters per token; measured on dense JSON prompts with
    qwen3.5 the true ratio was about 2.9). Only the ordering of budget decisions depends on it; the real count is recorded per call."""
    return int(len(text) / 2.8) + 1


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=True)


def _cost(value: Any) -> int:
    return estimate_tokens(_dumps(value))


def affordance_id(index: int) -> str:
    return f"a{index}"


def curate_affordance(index: int, aff: Dict[str, Any], optional: bool = True) -> Dict[str, Any]:
    keys = AFFORDANCE_ESSENTIAL + (AFFORDANCE_OPTIONAL if optional else ())
    row = {"affordance_id": affordance_id(index)}
    row.update({k: aff[k] for k in keys if k in aff and aff[k] not in (None, [], {}, "")})
    return row


def _short(text: Any, n: int) -> str:
    s = str(text or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def _project_priority(item: Dict[str, Any], targeted: set) -> Tuple:
    unfinished = str(item.get("project_status")) != "complete"
    return (0 if item.get("structure_id") in targeted else 1, 0 if (item.get("validation_failed") or item.get("observed_repair_need")) else 1, 0 if unfinished else 1, str(item.get("structure_id")))


def _plan_section(team_plan_context: Dict[str, Any]) -> Dict[str, Any]:
    tpc = team_plan_context or {}
    mine = tpc.get("my_assignment")
    plan = tpc.get("known_plan") or {}
    out: Dict[str, Any] = {"unread_plan_artifact_on_workspace": bool(tpc.get("unread_live_plan_artifact")), "adopted": bool(tpc.get("adopted")), "review": tpc.get("review") or {}}
    if plan:
        out["known_plan"] = {k: plan.get(k) for k in ("plan_id", "status_when_read", "proposed_by", "goal_summary", "expires_at", "expired", "read_via") if plan.get(k) not in (None, "")}
    if mine:
        out["my_assignment"] = {k: mine.get(k) for k in ("status", "grounded", "grounding_reason", "action_types", "target", "success_criterion", "progress", "recognized_gap_tokens") if mine.get(k) not in (None, [], {})}
    return out


def build_prompt(canonical: Dict[str, Any], *, budget_tokens: int, prompt_version: str = PROMPT_VERSION) -> Dict[str, Any]:
    """-> {system, user, affordance_map, schema, metadata}. Pure function of (canonical, budget, version)."""
    if prompt_version not in SYSTEM_PROMPTS:
        raise ValueError(f"unknown prompt version {prompt_version!r}")
    system = SYSTEM_PROMPTS[prompt_version]
    legal = list(canonical.get("legal_actions") or [])
    amap = {affordance_id(i): a for i, a in enumerate(legal)}
    targeted = {str(a.get("target_id")) for a in legal if a.get("target_id")}
    obs = dict(canonical.get("observations") or {})
    built_all = sorted([{k: p.get(k) for k in BUILT_STATE_KEYS if k in p} for p in (obs.get("built_state") or []) if isinstance(p, dict)], key=lambda p: _project_priority(p, targeted))
    outcomes = list((canonical.get("recent_outcomes") or {}).get("recent_action_outcomes") or [])
    failed = list((canonical.get("recent_outcomes") or {}).get("recent_failed_attempts") or [])
    comms = canonical.get("communications_and_artifacts") or {}
    held = canonical.get("held_dik") or {}
    artifacts = [{"artifact_id": a.get("artifact_id"), "type": a.get("type"), "author": a.get("author"), "summary": _short(a.get("summary"), 160), "read_by_me": bool(a.get("inspected_by_agent"))}
                 for a in (comms.get("externalized_artifacts") or []) if isinstance(a, dict)]

    # tier-1 scalar/compact sections
    t1 = {
        "identity": canonical.get("identity") or {},
        "task": {k: v for k, v in (canonical.get("task") or {}).items() if v not in (None, [], {})},
        "goals": {"active_goal": (canonical.get("goals") or {}).get("active_goal"), "goal_stack": list((canonical.get("goals") or {}).get("goal_stack") or [])[:6]},
        "known_gaps": sorted(str(g) for g in (canonical.get("known_gaps") or [])),
        "team_plan": _plan_section(canonical.get("team_plan_context") or {}),
    }
    observation_scalars = {k: obs[k] for k in ("sim_time", "phase_profile", "agent_position", "brick_source_available", "capacity_unlock", "resource_status", "site_capacity_summary", "support_deficits") if k in obs}
    nearby = list(obs.get("nearby_agents") or [])

    shrink_log: List[str] = []
    keep_projects, keep_nearby, optional_fields, keep_gaps = len(built_all), len(nearby), True, len(t1["known_gaps"])

    def render_tier1() -> Dict[str, Any]:
        sec = dict(t1)
        sec["known_gaps"] = t1["known_gaps"][:keep_gaps]
        sec["observations"] = {**observation_scalars, "nearby_agents": nearby[:keep_nearby], "projects": built_all[:keep_projects]}
        sec["legal_actions"] = [curate_affordance(i, a, optional=optional_fields) for i, a in enumerate(legal)]
        return sec

    def tier1_cost() -> int:
        return _cost(render_tier1())

    fixed = estimate_tokens(system) + 64  # instruction line + framing
    avail = max(0, int(budget_tokens) - fixed)
    # fixed, ordered shrink steps for tier 1 (legal affordance ids/essentials are never removed)
    while tier1_cost() > avail and keep_projects > 3:
        keep_projects = max(3, keep_projects // 2)
        shrink_log.append(f"projects->{keep_projects}")
    while tier1_cost() > avail and keep_nearby > 0:
        keep_nearby = 0
        shrink_log.append("nearby_agents->0")
    if tier1_cost() > avail and optional_fields:
        optional_fields = False
        shrink_log.append("affordance_optional_fields->dropped")
    while tier1_cost() > avail and keep_gaps > 8:
        keep_gaps = 8
        shrink_log.append("known_gaps->8")
    tier1_over = tier1_cost() > avail
    sections = render_tier1()
    used = tier1_cost()

    meta_sections: Dict[str, Any] = {}
    meta_sections["legal_actions"] = {"tier": 1, "total_items": len(legal), "included_items": len(sections["legal_actions"]), "omitted_items": 0, "optional_fields_dropped": not optional_fields}
    meta_sections["observations.projects"] = {"tier": 1, "total_items": len(built_all), "included_items": keep_projects, "omitted_items": len(built_all) - keep_projects}
    meta_sections["observations.nearby_agents"] = {"tier": 1, "total_items": len(nearby), "included_items": keep_nearby, "omitted_items": len(nearby) - keep_nearby}
    meta_sections["known_gaps"] = {"tier": 1, "total_items": len(t1["known_gaps"]), "included_items": keep_gaps, "omitted_items": len(t1["known_gaps"]) - keep_gaps}

    # tiers 2 and 3: (name, tier, ordered items) -- items most relevant/most recent first
    def newest_first(seq):
        return list(reversed(list(seq)))

    lists = [
        ("held_dik.knowledge", 2, "held_dik", "knowledge", sorted(str(x) for x in (held.get("knowledge") or []))),
        ("held_dik.information", 2, "held_dik", "information", sorted(str(x) for x in (held.get("information") or []))),
        ("held_dik.data", 2, "held_dik", "data", sorted(str(x) for x in (held.get("data") or []))),
        ("communications.artifacts", 2, "communications_and_artifacts", "artifacts", artifacts),
        ("recent_outcomes.blockers", 2, "recent_outcomes", "recent_failed_attempts", newest_first(failed)),
        ("recent_outcomes.actions", 2, "recent_outcomes", "recent_action_outcomes", newest_first(outcomes)),
        ("communications.recent_shared_updates", 3, "communications_and_artifacts", "recent_shared_updates", newest_first(comms.get("recent_shared_updates") or [])),
        ("communications.recent_construction_updates", 3, "communications_and_artifacts", "recent_construction_updates", newest_first(comms.get("recent_construction_updates") or [])),
        ("recent_outcomes.older_events", 3, "recent_outcomes", "near_preceding_events", newest_first((canonical.get("recent_outcomes") or {}).get("near_preceding_events") or [])),
    ]
    for name, tier, group, key, items in lists:
        included: List[Any] = []
        for item in items:
            c = _cost(item) + 2
            if used + c > avail:
                break  # items are in priority order: once one does not fit, all later ones of this section are omitted
            included.append(item)
            used += c
        if included or items:
            sections.setdefault(group, {})[key] = included
        meta_sections[name] = {"tier": tier, "total_items": len(items), "included_items": len(included), "omitted_items": len(items) - len(included)}

    ordered_keys = ["identity", "task", "goals", "observations", "known_gaps", "team_plan", "legal_actions", "held_dik", "communications_and_artifacts", "recent_outcomes"]
    user_body = {k: sections[k] for k in ordered_keys if k in sections}
    user = "CONTEXT:\n" + _dumps(user_body) + "\n\nChoose exactly one affordance_id from legal_actions and reply with the JSON object only."
    schema = {
        "type": "object",
        "properties": {"affordance_id": {"type": "string", "enum": list(amap)}, "arguments": {"type": "object"}, "decision_note": {"type": "string", "maxLength": 200}},
        "required": ["affordance_id"],
    }
    omitted = {n: m["omitted_items"] for n, m in meta_sections.items() if m["omitted_items"]}
    metadata = {
        "prompt_version": prompt_version, "budget_tokens": int(budget_tokens), "available_for_context_tokens": avail,
        "estimated_prompt_tokens": estimate_tokens(system) + estimate_tokens(user), "tier1_over_budget": bool(tier1_over), "tier1_shrink_steps": shrink_log,
        "omitted_item_counts": omitted, "sections": meta_sections, "affordance_count": len(legal),
        "estimator": "chars/2.8 (deterministic, tokenizer-free, conservative)",
    }
    return {"system": system, "user": user, "affordance_map": amap, "schema": schema, "metadata": metadata,
            "prompt_hash": hashlib.sha256((system + "\n" + user).encode("utf-8")).hexdigest()[:16]}
