"""The Procedural Baseline starts structures itself and treats concrete build work as non-deadlock."""

from modules.action_schema import ExecutableActionType
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain

CAPACITY = [{"site_id": "site_b", "capacity": 8, "used": 0, "remaining": 8, "buildable": True}]
FULL = [{"site_id": "site_b", "capacity": 8, "used": 8, "remaining": 0, "buildable": True}]


def _project(*, status="in_progress", progress=0.0, resource_complete=False, delivered=0):
    return {
        "structure_id": "p1",
        "state": "in_progress",
        "progress": progress,
        "physical_build_progress": progress,
        "project_status": status,
        "ready_for_validation": False,
        "validation_readiness": {"validation_ready": False},
        "needs_repair": False,
        "resource_complete": resource_complete,
        "required_resources": 10,
        "delivered_resources": delivered,
    }


def _context(built_state, site_capacity=CAPACITY, mode="COORDINATE", dwell=60, no_progress=4):
    affordances = [
        {"action_type": "start_construction", "utility": 0.5, "target_id": "Build_Site_B", "reachable": True},
        {"action_type": "continue_construction", "utility": 0.5, "target_id": "Build_Site_B", "reachable": True},
        {"action_type": "consult_team_artifact", "utility": 0.6, "target_id": "whiteboard"},
        {"action_type": "externalize_plan", "utility": 0.8, "target_id": "whiteboard"},
        {"action_type": "transport_resources", "utility": 0.4, "target_id": "resource_zone_to_work_zone", "reachable": True},
        {"action_type": "wait", "utility": 0.1, "target_id": None},
    ]
    return BrainContextPacket(
        static_task_context={"role": "Engineer"},
        world_snapshot={"sim_time": 200.0, "phase_profile": {"stage": "execution", "name": "Phase 1"}, "built_state": built_state, "site_capacity_summary": site_capacity},
        individual_cognitive_state={
            "traits": {},
            "known_gaps": [],
            "build_readiness": {"ready_for_build": True},
            "goal_stack": [{"goal_id": f"g{i}"} for i in range(4)],
            "loop_counters": {"no_progress_streak": no_progress},
            "progress_state": {"no_progress_streak": no_progress},
            "seconds_since_dik_change": 50.0,
            "control_state": {"mode": mode, "mode_dwell_steps": dwell, "mode_history": [], "transition_history": []},
            "inspect_state": {"source_exhaustion": {}},
            "epistemic_sufficiency": {"score": 0.3, "refresh_pressure": 0.0},
        },
        team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
        history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
        action_affordances=affordances,
    )


def test_a_new_structure_is_needed_only_when_nothing_is_unfinished_and_a_site_has_room():
    brain = RuleBrain()
    assert brain._extract_features(_context([]))["new_structure_needed"] == 1.0
    assert brain._extract_features(_context([_project(delivered=3)]))["new_structure_needed"] == 0.0  # unfinished work exists
    assert brain._extract_features(_context([], site_capacity=FULL))["new_structure_needed"] == 0.0  # nowhere to build


def test_baseline_starts_a_structure_itself_when_none_exists():
    for mode in ("COORDINATE", "LOGISTICS", "CONSTRUCT"):
        brain = RuleBrain()
        context = _context([], mode=mode)
        chosen = [brain.decide(context).selected_action for _ in range(6)]
        assert chosen[-1] == ExecutableActionType.START_CONSTRUCTION, (mode, chosen)


def test_concretely_actionable_build_work_is_not_a_coordination_deadlock():
    brain = RuleBrain()
    ready_to_build = _context([_project(progress=0.0, resource_complete=True, delivered=10)])
    assert brain._extract_features(ready_to_build)["coordination_deadlock_pressure"] == 0.0
    # materials missing and nothing concrete to do for it -> the existing deadlock reading is unchanged
    needs_materials = _context([_project(delivered=3)], site_capacity=FULL)
    assert brain._extract_features(needs_materials)["coordination_deadlock_pressure"] > 0.0


def test_baseline_builds_a_resource_complete_project_rather_than_reconsulting_the_plan():
    brain = RuleBrain()
    context = _context([_project(progress=0.0, resource_complete=True, delivered=10)], mode="COORDINATE")
    chosen = [brain.decide(context).selected_action for _ in range(8)]
    assert chosen[-1] in {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}, chosen
