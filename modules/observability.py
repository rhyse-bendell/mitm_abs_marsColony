TRANSLATION_FAILURE_CATEGORIES = {
    "illegal_action",
    "unknown_action",
    "missing_target",
    "unresolved_target",
    "invalid_zone",
    "unsupported_plan_step",
    "no_enabled_action_mapping",
}

TARGET_FAILURE_CATEGORIES = {
    "no_information_source_available",
    "target_missing",
    "unresolved_target",
    "invalid_zone",
    "target_unreachable",
}

MOVEMENT_BLOCKER_CATEGORIES = {
    "no_path_found",
    "target_unreachable",
    "blocked_zone",
    "spawn_conflict",
    "agent_collision_block",
    "zero_distance_retarget",
    "path_invalidated",
    "target_drifted",
    "repeated_move_retry",
    "arrival_without_progress",
    "target_missing",
    "action_precondition_failed",
    "agent_waiting_on_teammate",
    "planner_loop",
    "unknown",
}
