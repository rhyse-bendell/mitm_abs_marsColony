"""Neutral compact_panel seed-0 absorbing CONSULT/WAIT state: its four causes, each pinned.

1. closure/support pivots must not replace a legal Pilot action with WAIT;
2. `reachable` means a legal route exists (planner), `straight_line_clear` is the separate direct-segment test;
3. concrete-work / deadlock features follow path reachability;
4. a tentative closure owner that never commits does not hold the project indefinitely.
"""

import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_provider import RuleBrain
from modules.simulation import SimulationState
from tests.test_observed_validation_repair import _ready_house as _ready_house_base

# standing point from the failing run: the direct segment to the Table_B build point crosses an obstacle, A* routes around it
BLOCKED_STRAIGHT_POS = (7.25, 3.7)
TABLE_B_POINT = (5.6, 3.8)


def _ready_house(cm, last_actor="Engineer"):
    """A validation-ready project whose builder is `last_actor`: construction names it the tentative closure owner."""
    pid = _ready_house_base(cm)
    cm.projects[pid]["last_actor"] = last_actor
    cm.update()
    return pid


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="neutral_stall", seed=0)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _agent(sim, name):
    return next(a for a in sim.agents if a.name == name)


# ---------------------------------------------------------------- 2. reachability semantics


def test_path_reachable_is_independent_of_straight_line(sim):
    env = sim.environment
    assert not env._segment_is_navigable(BLOCKED_STRAIGHT_POS, TABLE_B_POINT)
    assert env.path_reachable(BLOCKED_STRAIGHT_POS, TABLE_B_POINT)
    assert not env.path_reachable(BLOCKED_STRAIGHT_POS, (6.5, 3.4))  # inside blocked Table_A: legality is not weakened


def test_project_affordances_are_reachable_by_path_not_straight_line(sim):
    pid = _ready_house(sim.environment.construction)
    agent = _agent(sim, "Engineer")
    agent.position = BLOCKED_STRAIGHT_POS
    packet = sim.brain_context_builder.build(sim, agent)
    validate = [a for a in packet.action_affordances if a["action_type"] == "validate_construction" and a.get("target_id") == pid]
    assert validate and validate[0]["straight_line_clear"] is False and validate[0]["reachable"] is True


# ---------------------------------------------------------------- 3. deadlock / concrete-work features


def test_routable_work_is_concrete_work_not_a_coordination_deadlock(sim):
    _ready_house(sim.environment.construction)
    agent = _agent(sim, "Engineer")
    agent.position = BLOCKED_STRAIGHT_POS
    packet = sim.brain_context_builder.build(sim, agent)
    assert all(a.get("straight_line_clear") is False for a in packet.action_affordances if a.get("target_class") == "build")
    features = RuleBrain()._extract_features(packet)
    assert features["validation_work_pending"] == 1.0
    assert features["coordination_deadlock_pressure"] == 0.0 and features["no_active_plan_pressure"] == 0.0


# ---------------------------------------------------------------- 1. Pilot authority


@pytest.mark.parametrize(
    "kind",
    [
        ExecutableActionType.CONSULT_TEAM_ARTIFACT,
        ExecutableActionType.INSPECT_INFORMATION_SOURCE,
        ExecutableActionType.EXTERNALIZE_PLAN,
    ],
)
@pytest.mark.parametrize("owner", ["Engineer", "Architect"])  # a teammate owns the closure / the actor is the named (uncommitted) owner
def test_closure_support_pivots_never_replace_a_pilot_action_with_wait(sim, kind, owner):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    project = cm.projects[pid]
    assert project["closure_owner"] and project["closure_status"] == "assigned"
    project["closure_owner"] = owner
    actor = _agent(sim, "Architect")
    actor.control_state["mode"] = "COORDINATE"  # out of the startup BOOTSTRAP guard, which would return the decision untouched
    target = {"consult_team_artifact": "whiteboard", "inspect_information_source": "Team_Info", "externalize_plan": "whiteboard"}[kind.value]
    decision = BrainDecision(selected_action=kind, target_id=target, confidence=0.9)
    out = actor._apply_policy_pivots(decision, sim.environment, sim_state=sim, context=sim.brain_context_builder.build(sim, actor), pivot_origin="test")
    assert out.selected_action != ExecutableActionType.WAIT
    assert not [e for e in sim.captured_events if e[0] in {"closure_support_focus_applied", "blocked_project_support_pressure_applied"} and e[1].get("substituted")]


# ---------------------------------------------------------------- 4. closure ownership


def _tick_reconcile(sim, n):
    for _ in range(n):
        sim.environment.construction.update()
        sim._reconcile_uncommitted_closure_owners()


def test_uncommitted_tentative_owner_is_cleared_and_not_recreated(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    project = cm.projects[pid]
    owner = project["closure_owner"]
    assert owner and project["closure_status"] == "assigned"  # site_b_food_connector_001-style: named owner, no commitment
    assert not _agent(sim, owner).project_closure_state.get("active")
    _tick_reconcile(sim, sim.CLOSURE_UNCOMMITTED_GRACE_TICKS + 2)
    assert project["closure_owner"] is None and project["closure_status"] == "idle"
    assert pid not in cm._closure_reservations
    assert any(e[0] == "closure_uncommitted_reservation_cleared" for e in sim.captured_events)
    _tick_reconcile(sim, 10)  # construction.update() must not hand the project straight back to an uncommitted agent
    assert project["closure_owner"] is None and project["closure_status"] == "idle"


def test_committed_owner_is_preserved_and_ownership_stays_exclusive(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    owner_name = cm.projects[pid]["closure_owner"]
    owner = _agent(sim, owner_name)
    assert owner._start_project_closure_commitment(pid, environment=sim.environment, sim_state=sim, ttl_s=14.0)
    _tick_reconcile(sim, 10)
    assert cm.projects[pid]["closure_owner"] == owner_name and cm.projects[pid]["closure_status"] == "in_progress"
    other = next(a for a in sim.agents if a.name != owner_name)
    assert not cm.claim_project_closure(pid, other.name, now_ts=sim.time)  # no second owner while the reservation holds


def test_cleared_project_can_still_be_claimed_by_whoever_commits(sim):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    named = cm.projects[pid]["closure_owner"]
    _tick_reconcile(sim, sim.CLOSURE_UNCOMMITTED_GRACE_TICKS + 1)
    assert cm.projects[pid]["closure_owner"] is None
    other = next(a for a in sim.agents if a.name != named)
    assert other._start_project_closure_commitment(pid, environment=sim.environment, sim_state=sim, ttl_s=14.0)
    assert cm.projects[pid]["closure_owner"] == other.name and cm.projects[pid]["closure_status"] == "in_progress"
