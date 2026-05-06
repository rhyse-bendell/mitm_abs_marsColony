"""State container for the Procedural Baseline Pilot controller.

This state belongs to the Procedural Baseline Pilot, not to the Agent Mecha.
It is currently introduced as an extraction target. Existing Agent fields may
temporarily remain for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProceduralBaselineControllerState:
    mode: str = "BOOTSTRAP"
    previous_mode: Optional[str] = None
    mode_entered_step: int = 0
    mode_dwell_steps: int = 0
    last_transition_reason: str = "agent_initialized"
    last_transition_features: dict[str, Any] = field(default_factory=dict)
    mode_history: list[dict[str, Any]] = field(default_factory=lambda: [{"step": 0, "mode": "BOOTSTRAP", "reason": "agent_initialized"}])
    transition_history: list[dict[str, Any]] = field(default_factory=list)
    recovery_active: bool = False
    last_policy_snapshot: dict[str, Any] = field(default_factory=dict)
    active_method_id: Optional[str] = None
    active_method_instance: Optional[dict[str, Any]] = None
    active_method_step: Optional[str] = None
    method_started_tick: Optional[int] = None
    step_started_tick: Optional[int] = None
    step_retry_count: int = 0
    recent_step_outcomes: list[dict[str, Any]] = field(default_factory=list)
    method_history: list[dict[str, Any]] = field(default_factory=list)
    method_transition_history: list[dict[str, Any]] = field(default_factory=list)
    abandoned_methods: list[str] = field(default_factory=list)
    method_cooldowns: dict[str, Any] = field(default_factory=dict)
    source_cooldowns: dict[str, Any] = field(default_factory=dict)
    source_exhaustion: dict[str, Any] = field(default_factory=dict)
    last_method_switch_reason: Optional[str] = None
