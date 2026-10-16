"""Team-plan lifecycle policy and process measures (generic engine layer; task content comes from the task manifest).

Lifecycle: proposed -> committed -> (superseded | expired). A plan commits only on EXPLICIT support under a configurable rule
(task_manifest.json -> "team_plan_policy"); silence, clarification requests and declines never count as support, and support given
to one version of a plan's content does not carry over to a materially different version.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List

# Engine fallback only when a task package defines no team_plan_policy. The Mars task states its own in task_manifest.json.
DEFAULT_TEAM_PLAN_POLICY: Dict[str, Any] = {
    "commit_rule": {"proposer_must_support": True, "min_total_supporters": 2, "min_non_proposer_supporters": 1},
    "support_response_types": ["agree", "assignment_accept"],
    "review_after_s": 20.0,
    "response_engagement_min": 0.25,
}

PLAN_TASKWORK_ACTIONS = frozenset({"transport_resources", "start_construction", "continue_construction", "repair_or_correct_construction", "validate_construction", "connect_structures"})


def load_team_plan_policy(manifest: Dict[str, Any] | None) -> Dict[str, Any]:
    policy = json.loads(json.dumps(DEFAULT_TEAM_PLAN_POLICY))
    override = dict((manifest or {}).get("team_plan_policy") or {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(policy.get(key), dict):
            policy[key].update(value)
        else:
            policy[key] = value
    return policy


def plan_content_signature(content: Dict[str, Any]) -> str:
    """What a plan commits the team TO (goals, targets, assignments, success criteria); not response bookkeeping or timestamps."""
    material = {
        "goal_ids": sorted(str(g) for g in (content.get("goal_ids") or [])),
        "project_targets": sorted(str(t) for t in (content.get("project_targets") or [])),
        "assignments_by_role": content.get("assignments_by_role") or {},
        "success_criteria": sorted(str(c) for c in (content.get("success_criteria") or [])),
    }
    return hashlib.sha1(json.dumps(material, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12]


def effective_supporters(content: Dict[str, Any], policy: Dict[str, Any]) -> List[str]:
    """Supporters whose recorded support refers to the plan's CURRENT content."""
    sig = content.get("content_signature")
    by_sig = dict(content.get("supporter_signatures") or {})
    return sorted(s for s in (content.get("supporters") or []) if by_sig.get(s) == sig)


def commitment_met(content: Dict[str, Any], policy: Dict[str, Any]) -> tuple[bool, str]:
    rule = dict(policy.get("commit_rule") or {})
    proposer = str(content.get("proposed_by") or "")
    supporters = effective_supporters(content, policy)
    if rule.get("proposer_must_support", True) and proposer not in supporters:
        return False, "proposer_has_not_supported_own_proposal"
    others = [s for s in supporters if s != proposer]
    if len(others) < int(rule.get("min_non_proposer_supporters", 1)):
        return False, "no_non_proposer_supporter"
    if len(supporters) < int(rule.get("min_total_supporters", 2)):
        return False, "quorum_not_reached"
    return True, "quorum_reached"


class PlanProcessTracker:
    """Aggregates the lifecycle event stream into process measures (descriptive; never feeds back into behaviour)."""

    COUNTED = {
        "team_plan_proposed": "proposals",
        "team_plan_consulted": "consultations",
        "team_plan_agreed": "support_responses",
        "team_plan_assignment_accepted": "support_responses",
        "team_plan_clarification_requested": "clarification_responses",
        "team_plan_disagreed": "declines",
        "team_plan_assignment_declined": "declines",
        "team_plan_committed": "committed_plans",
        "team_plan_adopted_locally": "local_adoptions",
        "team_plan_assignment_applied": "assignments_applied",
        "team_plan_assignment_started": "assignments_started",
        "team_plan_assignment_completed": "assignments_completed",
        "team_plan_assignment_blocked": "assignments_blocked",
        "team_plan_assignment_deviated": "assignments_deviated",
        "team_plan_reassessed": "reassessments",
        "team_plan_superseded": "plans_superseded",
        "team_plan_expired": "plans_expired",
    }

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {v: 0 for v in set(self.COUNTED.values())}
        self.proposed_at: Dict[str, float] = {}
        self.commit_latencies: List[float] = []
        self.plans_with_response: set = set()
        self.taskwork_completed = 0
        self.aligned_taskwork_completed = 0
        self.active: Dict[str, set] = {}  # plan_id -> agents currently holding an unretired, grounded assignment
        self.peak_aligned_agents = 0
        self.peak_plan = None
        self._last_t = 0.0
        self._aligned_agent_seconds = 0.0
        self._occupied_seconds = 0.0
        self.adopting_agents: set = set()
        self.assigned_agents: set = set()
        self.pairs: Dict[str, set] = {k: set() for k in ("adopted", "applied", "started", "completed", "deviated", "blocked")}  # distinct (plan, agent)

    def _integrate(self, t: float) -> None:
        dt = max(0.0, t - self._last_t)
        n = max((len(v) for v in self.active.values()), default=0)
        if n:
            self._aligned_agent_seconds += n * dt
            self._occupied_seconds += dt
        self._last_t = t

    def on_event(self, event: Dict[str, Any]) -> None:
        name = event.get("event_type")
        payload = event.get("payload_data") or {}
        t = float(event.get("time") or 0.0)
        if name == "executable_action_completed" and payload.get("action_type") in PLAN_TASKWORK_ACTIONS:
            self.taskwork_completed += 1
            return
        if name not in self.COUNTED:
            return
        self._integrate(t)
        self.counts[self.COUNTED[name]] += 1
        plan_id = str(payload.get("plan_id") or "")
        agent = str(payload.get("agent") or payload.get("agent_role") or "")
        if name == "team_plan_proposed" and plan_id:
            self.proposed_at.setdefault(plan_id, t)
        elif name == "team_plan_committed" and plan_id in self.proposed_at:
            self.commit_latencies.append(t - self.proposed_at[plan_id])
        elif name == "team_plan_adopted_locally":
            self.adopting_agents.add(agent)
            self.pairs["adopted"].add((plan_id, agent))
        elif name == "team_plan_assignment_applied" and payload.get("grounded", True):
            self.pairs["applied"].add((plan_id, agent))
            self.active.setdefault(plan_id, set()).add(agent)
            self.assigned_agents.add(agent)
            n = len(self.active[plan_id])
            if n > self.peak_aligned_agents:
                self.peak_aligned_agents, self.peak_plan = n, plan_id
        elif name == "team_plan_assignment_started":
            self.pairs["started"].add((plan_id, agent))
        elif name == "team_plan_assignment_deviated":
            self.pairs["deviated"].add((plan_id, agent))
        elif name == "team_plan_assignment_completed":
            self.pairs["completed"].add((plan_id, agent))
            self.aligned_taskwork_completed += 1
            self.active.get(plan_id, set()).discard(agent)
        elif name == "team_plan_assignment_blocked":
            self.pairs["blocked"].add((plan_id, agent))
            self.active.get(plan_id, set()).discard(agent)
        elif name in {"team_plan_superseded", "team_plan_expired"}:
            self.active.pop(plan_id, None)

    def summary(self, now: float, n_agents: int) -> Dict[str, Any]:
        self._integrate(now)
        c = self.counts
        rate = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
        return {
            **c,
            "mean_time_proposal_to_commitment_s": round(sum(self.commit_latencies) / len(self.commit_latencies), 2) if self.commit_latencies else None,
            "commitment_rate": rate(c["committed_plans"], c["proposals"]),
            "distinct_plan_agent_adoptions": len(self.pairs["adopted"]),
            "distinct_grounded_assignments": len(self.pairs["applied"]),
            "local_adoption_rate": rate(len(self.pairs["adopted"]), c["committed_plans"] * n_agents),
            "assignment_start_rate": rate(len(self.pairs["started"]), len(self.pairs["applied"])),
            "assignment_completion_rate": rate(len(self.pairs["completed"]), len(self.pairs["applied"])),
            "deviation_or_block_rate": rate(len((self.pairs["deviated"] | self.pairs["blocked"]) & self.pairs["applied"]), len(self.pairs["applied"])),
            "supersession_rate": rate(c["plans_superseded"], c["proposals"]),
            "expiry_rate": rate(c["plans_expired"], c["proposals"]),
            "taskwork_completed": self.taskwork_completed,
            "taskwork_aligned_with_committed_plan": self.aligned_taskwork_completed,
            "taskwork_aligned_proportion": rate(self.aligned_taskwork_completed, self.taskwork_completed),
            "peak_agents_aligned_to_one_plan": self.peak_aligned_agents,
            "mean_agents_aligned_while_any_plan_active": round(self._aligned_agent_seconds / self._occupied_seconds, 3) if self._occupied_seconds else 0.0,
            "agents_that_adopted": len(self.adopting_agents - {""}),
        }
