from __future__ import annotations

from enum import Enum


class ParameterLayer(str, Enum):
    PILOT_CAPABILITY = "pilot_capability"
    MECHA_CAPABILITY = "mecha_capability"
    ENVIRONMENT_TASK = "environment_task"
    ANALYSIS_ONLY = "analysis_only"


PILOT_CAPABILITY_PARAMETERS = {
    "communication_propensity",
    "goal_alignment",
    "help_tendency",
    "artifact_externalization_tendency",
    "artifact_consultation_tendency",
    "teammate_model_accuracy",
    "validation_thoroughness",
    "build_readiness_sensitivity",
    "mismatch_detection_sensitivity",
    "planning_horizon",
    "reassessment_threshold",
}

MECHA_CAPABILITY_PARAMETERS = {
    "communication_range",
    "communication_bandwidth",
    "message_noise",
    "memory_capacity",
    "memory_decay_rate",
    "dik_uptake_fidelity",
    "artifact_uptake_fidelity",
    "movement_speed",
    "carry_capacity",
    "inspection_duration",
    "construction_action_duration",
    "validation_action_duration",
}

ENVIRONMENT_TASK_PARAMETERS = {
    "site_b_capacity",
    "site_c_capacity",
    "bridge_bc_cost",
    "pile_a_quantity",
    "pile_c_quantity",
    "housing_cost",
    "greenhouse_cost",
    "water_generator_cost",
    "phase_duration",
    "source_distribution",
}


def classify_parameter(name: str) -> ParameterLayer | None:
    if name in PILOT_CAPABILITY_PARAMETERS:
        return ParameterLayer.PILOT_CAPABILITY
    if name in MECHA_CAPABILITY_PARAMETERS:
        return ParameterLayer.MECHA_CAPABILITY
    if name in ENVIRONMENT_TASK_PARAMETERS:
        return ParameterLayer.ENVIRONMENT_TASK
    return None
