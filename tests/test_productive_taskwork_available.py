"""productive_taskwork_available: scarce resources are not the same as no work left.

The feature is derived only from observed project state and the legal affordances. When it is false the
Procedural Baseline idles/monitors itself; when work reappears it resumes.
"""

from types import SimpleNamespace

from modules.action_schema import ExecutableActionType
from modules.brain_context import BrainContextBuilder
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain

CAPACITY = [{"site_id": "site_b", "capacity": 8, "used": 0, "remaining": 8, "buildable": True}]
FULL = [{"site_id": "site_b", "capacity": 8, "used": 8, "remaining": 0, "buildable": True}]

# what an exhausted colony still offers: no brick source, so no transport affordance
BASE = [
    {"action_type": "wait", "utility": 0.1, "target_id": None},
    {"action_type": "communicate", "utility": 0.3, "target_id": "nearby_agent", "reachable": True, "productive": False},
    {"action_type": "externalize_plan", "utility": 0.8, "target_id": "whiteboard"},
    {"action_type": "consult_team_artifact", "utility": 0.6, "target_id": "whiteboard"},
    {"action_type": "start_construction", "utility": 0.5, "target_id": "Build_Site_B", "reachable": True},
    {"action_type": "validate_construction", "utility": 0.05, "target_id": "active_construction"},
    {"action_type": "repair_or_correct_construction", "utility": 0.05, "target_id": "active_construction"},
    {"action_type": "reassess_plan", "utility": 0.06, "target_id": None},
    {"action_type": "observe_environment", "utility": 0.06, "target_id": None},
]
TRANSPORT = {"action_type": "transport_resources", "utility": 0.8, "target_id": "resource_zone_to_work_zone"}


def _project(*, status="in_progress", progress=0.0, resource_complete=False, delivered=0, required=10, validation_ready=False, needs_repair=False):
    return {
        "structure_id": "p1",
        "state": "in_progress",
        "progress": progress,
        "physical_build_progress": progress,
        "project_status": status,
        "ready_for_validation": validation_ready,
        "validation_readiness": {"validation_ready": validation_ready},
        "needs_repair": needs_repair,
        "resource_complete": resource_complete,
        "required_resources": required,
        "delivered_resources": delivered,
    }


def _context(built_state, affordances, site_capacity=FULL, mode="COORDINATE"):
    return BrainContextPacket(
        static_task_context={"role": "Engineer"},
        world_snapshot={"sim_time": 1300.0, "phase_profile": {"stage": "late", "name": "Phase 2"}, "built_state": built_state, "site_capacity_summary": site_capacity},
        individual_cognitive_state={
            "traits": {},
            "known_gaps": [],
            "build_readiness": {"ready_for_build": True},
            "goal_stack": [{"goal_id": f"g{i}"} for i in range(4)],
            "loop_counters": {"no_progress_streak": 4},
            "progress_state": {"no_progress_streak": 4},
            "seconds_since_dik_change": 500.0,
            "control_state": {"mode": mode, "mode_dwell_steps": 60, "mode_history": [], "transition_history": []},
            "inspect_state": {"source_exhaustion": {}},
            "epistemic_sufficiency": {"score": 0.8, "refresh_pressure": 0.0},
        },
        team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
        history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
        action_affordances=list(affordances),
    )


def _feature(built_state, affordances, site_capacity=FULL):
    return RuleBrain()._extract_features(_context(built_state, affordances, site_capacity))["productive_taskwork_available"]


def test_nothing_can_advance_means_no_productive_taskwork():
    assert _feature([], BASE, FULL) == 0.0
    # finished/complete structures and a satisfied project do not count as remaining work
    assert _feature([_project(status="complete", delivered=10, resource_complete=True, progress=1.0)], BASE, FULL) == 0.0
    # an unfinished project that needs bricks but has no brick source anywhere
    assert _feature([_project(delivered=3)], BASE, FULL) == 0.0
    # room to build but no brick source: a start cannot be supplied
    assert _feature([], BASE, CAPACITY) == 0.0


def test_a_carried_or_available_brick_for_an_unfinished_project_is_productive():
    assert _feature([_project(delivered=3)], BASE + [TRANSPORT], FULL) == 1.0


def test_transport_affordance_is_legal_when_only_a_carried_brick_remains():
    empty = SimpleNamespace(construction=SimpleNamespace(resource_nodes={"pile_a": SimpleNamespace(quantity=0), "pile_c": SimpleNamespace(quantity=0)}))
    holding = SimpleNamespace(transport_state={"carrying": {"resource_type": "bricks", "quantity": 1}})
    empty_handed = SimpleNamespace(transport_state={"carrying": {"resource_type": None, "quantity": 0}})
    assert BrainContextBuilder._bricks_available(empty, holding) is True
    assert BrainContextBuilder._bricks_available(empty, empty_handed) is False


def test_a_staged_project_that_can_be_built_is_productive_without_bricks():
    assert _feature([_project(delivered=10, resource_complete=True, progress=0.3)], BASE, FULL) == 1.0


def test_a_validation_ready_project_is_productive():
    ready = _project(status="ready_for_validation", delivered=10, resource_complete=True, progress=1.0, validation_ready=True)
    assert _feature([ready], BASE, FULL) == 1.0


def test_a_project_needing_repair_is_productive():
    broken = _project(status="needs_repair", delivered=10, resource_complete=True, progress=1.0, needs_repair=True)
    assert _feature([broken], BASE, FULL) == 1.0


def test_a_startable_structure_with_bricks_and_room_is_productive():
    assert _feature([], BASE + [TRANSPORT], CAPACITY) == 1.0
    assert _feature([], BASE + [TRANSPORT], FULL) == 0.0  # bricks but nowhere to build


def test_baseline_idles_instead_of_repeating_impossible_or_purposeless_actions_when_nothing_can_advance():
    forbidden = {
        ExecutableActionType.TRANSPORT_RESOURCES,
        ExecutableActionType.START_CONSTRUCTION,
        ExecutableActionType.CONTINUE_CONSTRUCTION,
        ExecutableActionType.CONSULT_TEAM_ARTIFACT,
        ExecutableActionType.EXTERNALIZE_PLAN,
        ExecutableActionType.COMMUNICATE,
    }
    for mode in ("COORDINATE", "LOGISTICS", "CONSTRUCT", "MONITOR"):
        brain = RuleBrain()
        context = _context([], BASE, FULL, mode=mode)
        chosen = [brain.decide(context).selected_action for _ in range(30)]
        assert not (set(chosen[10:]) & forbidden), (mode, chosen)
        assert chosen[-1] in {ExecutableActionType.WAIT, ExecutableActionType.OBSERVE_ENVIRONMENT, ExecutableActionType.REASSESS_PLAN}, (mode, chosen)


def test_baseline_resumes_productive_work_when_it_reappears_after_idling():
    plan_already_shared = [a for a in BASE if a["action_type"] != "externalize_plan"]
    work_context = _context([_project(delivered=3)], plan_already_shared + [TRANSPORT], FULL)
    fresh = [RuleBrain().decide(work_context).selected_action for _ in range(1)]
    assert fresh == [ExecutableActionType.TRANSPORT_RESOURCES]

    brain = RuleBrain()
    idle_context = _context([], plan_already_shared, FULL)
    for _ in range(20):
        brain.decide(idle_context)
    chosen = [brain.decide(work_context).selected_action for _ in range(12)]
    assert chosen[-1] == ExecutableActionType.TRANSPORT_RESOURCES, chosen
    assert chosen.count(ExecutableActionType.TRANSPORT_RESOURCES) >= 8, chosen


def test_an_unsupplyable_unfinished_project_is_not_a_coordination_deadlock():
    # unfinished project, no brick source anywhere: no work can advance, and coordination cannot create bricks
    stuck = [_project(delivered=3)]
    features = RuleBrain()._extract_features(_context(stuck, BASE, FULL))
    assert features["productive_taskwork_available"] == 0.0
    assert features["coordination_deadlock_pressure"] == 0.0
    assert features["no_active_plan_pressure"] == 0.0
    for mode in ("COORDINATE", "MONITOR"):
        brain = RuleBrain()
        chosen = [brain.decide(_context(stuck, BASE, FULL, mode=mode)).selected_action for _ in range(30)]
        assert set(chosen[10:]) <= {ExecutableActionType.WAIT, ExecutableActionType.OBSERVE_ENVIRONMENT, ExecutableActionType.REASSESS_PLAN}, (mode, chosen)


def test_a_fully_built_structure_awaiting_validation_knowledge_is_still_open_work():
    built_not_ready = _project(status="in_progress", delivered=10, resource_complete=True, progress=1.0, validation_ready=False)
    assert _feature([built_not_ready], BASE, FULL) == 1.0


def test_policy_random_draw_advances_with_time_in_an_unchanged_state():
    # the same agent/mode/affordance count must not replay one draw on every tick
    import random
    from modules import brain_provider

    seeds = []
    real = random.Random

    class Spy(real):
        def __init__(self, seed=None):
            seeds.append(seed)
            super().__init__(seed)

    brain_provider.random.Random = Spy
    try:
        brain = RuleBrain()
        for sim_time in (1300.0, 1301.0):
            context = _context([], BASE, FULL)
            context.world_snapshot["sim_time"] = sim_time
            brain.decide(context)
    finally:
        brain_provider.random.Random = real
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds), seeds
