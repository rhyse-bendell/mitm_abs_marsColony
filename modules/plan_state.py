from dataclasses import dataclass

from modules.action_schema import BrainDecision


@dataclass
class PlanRecord:
    plan_id: str
    decision: BrainDecision
    created_at: float
    last_reviewed_at: float
    trigger_reason: str
    remaining_executions: int = 2
    invalidation_reason: str | None = None
    ordered_goals: list[dict] | None = None
    ordered_actions: list[dict] | None = None
    explanation: str | None = None
    plan_method_id: str | None = None
    plan_method_status: str = "unspecified"
    adoption_reason: str | None = None
    validation_notes: list[str] | None = None
    associated_goal_ids: list[str] | None = None
