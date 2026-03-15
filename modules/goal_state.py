from dataclasses import dataclass

GOAL_STATUSES = {"inactive", "candidate", "active", "queued", "blocked", "satisfied", "invalidated", "abandoned"}
GOAL_SOURCES = {
    "task_defined",
    "planner_proposed",
    "derived_from_rule",
    "teammate_or_artifact_influenced",
    "teammate_artifact_influenced",  # backwards-compatible alias
    "legacy_seed",
}


@dataclass
class GoalRecord:
    goal_key: str
    goal_id: str | None
    label: str
    source: str
    status: str
    priority: float
    target: object = None
    parent_goal_key: str | None = None
    evidence: list[str] | None = None
    activation_conditions: list[str] | None = None
    completion_conditions: list[str] | None = None
    invalidation_reasons: list[str] | None = None
    blocking_reasons: list[str] | None = None
    goal_level: str | None = None
    goal_type: str | None = None
    trust_tier: str = "normal"
    last_transition_reason: str | None = None


def coerce_goal_status(status):
    normalized = str(status or "candidate").strip().lower()
    return normalized if normalized in GOAL_STATUSES else "candidate"


def coerce_goal_source(source):
    normalized = str(source or "planner_proposed").strip().lower()
    return normalized if normalized in GOAL_SOURCES else "planner_proposed"


def goal_priority(status, priority):
    base = float(priority if priority is not None else 0.5)
    status_bias = {
        "active": 0.35,
        "queued": 0.2,
        "candidate": 0.1,
        "blocked": -0.2,
        "inactive": -0.3,
        "satisfied": -0.6,
        "invalidated": -0.7,
        "abandoned": -0.8,
    }.get(status, 0.0)
    return max(0.0, min(1.0, base + status_bias))
