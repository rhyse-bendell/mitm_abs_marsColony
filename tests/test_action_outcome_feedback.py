"""Gate G: rejected actions produce structured, Pilot-visible outcomes and cannot loop silently."""

import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.brain_context import BrainContextPacket
from modules.brain_provider import RuleBrain
from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="outcome_feedback")
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _blocked(agent, sim, project="site_b_house_001", category="no_pickup_resource_source"):
    agent._emit_event(sim, "construction_transport_blocked", {"project_id": project, "failure_category": category, "decision_action": "transport_resources"})


def test_blocked_attempts_become_structured_outcomes_with_a_failure_streak(sim):
    agent = sim.agents[0]
    for _ in range(3):
        _blocked(agent, sim)
    last = agent.recent_action_outcomes[-1]
    assert last["status"] == "blocked" and last["action"] == "transport_resources"
    assert last["target"] == "site_b_house_001" and last["blocker"] == "no_pickup_resource_source"
    assert last["consecutive_identical_failures"] == 3
    repeated = [p for name, p in sim.captured_events if name == "repeated_action_failure"]
    assert len(repeated) == 1 and repeated[0]["consecutive_identical_failures"] == 3


def test_a_different_blocker_starts_its_own_streak_and_a_success_clears_the_action(sim):
    agent = sim.agents[0]
    for _ in range(2):
        _blocked(agent, sim)
    _blocked(agent, sim, category="construction_delivery_rejected")
    assert agent.recent_action_outcomes[-1]["consecutive_identical_failures"] == 1
    _blocked(agent, sim)  # the original blocker's streak continues where it left off
    assert agent.recent_action_outcomes[-1]["consecutive_identical_failures"] == 3
    agent._emit_event(sim, "action_execution_succeeded", {"decision_action": "transport_resources", "project_id": "site_b_house_001"})
    assert agent.recent_action_outcomes[-1]["status"] == "succeeded"
    _blocked(agent, sim)
    assert agent.recent_action_outcomes[-1]["consecutive_identical_failures"] == 1


def test_alternating_failures_of_different_actions_each_keep_their_own_streak(sim):
    agent = sim.agents[0]
    for _ in range(4):
        agent._emit_event(sim, "mecha_action_blocked", {"selected_action": "validate_construction", "project_id": "p", "blockers": ["physical_incomplete"]})
        _blocked(agent, sim, project="p")
    by_action = {o["action"]: o["consecutive_identical_failures"] for o in agent.recent_action_outcomes[-2:]}
    assert by_action == {"validate_construction": 4, "transport_resources": 4}


def test_outcomes_reach_the_pilot_context_and_request(sim):
    agent = sim.agents[0]
    _blocked(agent, sim)
    context = sim.brain_context_builder.build(sim, agent)
    assert context.individual_cognitive_state["recent_action_outcomes"][-1]["blocker"] == "no_pickup_resource_source"
    request = agent._build_brain_request(sim, context, False, "unit_test")
    assert request.recent_action_outcomes[-1]["status"] == "blocked"
    assert request.to_dict()["recent_action_outcomes"][-1]["target"] == "site_b_house_001"


def test_transport_is_only_offered_while_a_brick_source_exists(sim):
    agent = sim.agents[0]
    sim.environment.construction.create_project("site_b", structure_type="house")  # transport is offered per project that needs bricks
    offered = lambda: {a["action_type"] for a in sim.brain_context_builder.build(sim, agent).action_affordances}
    assert "transport_resources" in offered()
    for pile in sim.environment.construction.resource_nodes.values():
        pile.quantity = 0
    assert "transport_resources" not in offered()


def test_exhausted_resources_end_the_transport_attempt_instead_of_repeating_it(sim):
    agent = sim.agents[0]
    cm = sim.environment.construction
    project_id, _ = cm.create_project("site_b", structure_type="house")
    for pile in cm.resource_nodes.values():
        pile.quantity = 0
    decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id=project_id, confidence=0.9)
    decision._action_gate_checked = True  # exercise the execution layer directly (the gate would stop this earlier)
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    for _ in range(5):
        agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)

    blocked = [p for name, p in sim.captured_events if name == "construction_transport_blocked" and p.get("failure_category") == "no_pickup_resource_source"]
    assert len(blocked) == 1  # one rejection, not one per tick
    assert agent.active_actions == []
    assert agent.recent_action_outcomes[-1]["blocker"] == "no_pickup_resource_source"


def _context(outcomes):
    return BrainContextPacket(
        static_task_context={"role": "Botanist"},
        world_snapshot={"sim_time": 100.0, "phase_profile": {"stage": "execution", "name": "Phase 1"}, "built_state": [
            {"structure_id": "p1", "state": "in_progress", "progress": 0.0, "physical_build_progress": 0.0, "project_status": "in_progress",
             "ready_for_validation": False, "validation_readiness": {"validation_ready": False}, "needs_repair": False,
             "resource_complete": False, "required_resources": 10, "delivered_resources": 0}]},
        individual_cognitive_state={
            "traits": {}, "known_gaps": [], "build_readiness": {"ready_for_build": True},
            "goal_stack": [{"goal_id": f"g{i}"} for i in range(4)],
            "loop_counters": {}, "progress_state": {}, "seconds_since_dik_change": 50.0,
            "control_state": {"mode": "LOGISTICS", "mode_dwell_steps": 40, "mode_history": [], "transition_history": []},
            "inspect_state": {"source_exhaustion": {}}, "epistemic_sufficiency": {"score": 0.9, "refresh_pressure": 0.0},
            "recent_action_outcomes": outcomes,
        },
        team_state={"externalized_artifacts": [], "teammate_help_signals": {}, "tom_summary": {}},
        history_bands={"semantic_plan_evolution": {"unresolved_contradictions": []}},
        action_affordances=[
            {"action_type": "transport_resources", "utility": 0.9, "target_id": "resource_zone_to_work_zone", "reachable": True},
            {"action_type": "communicate", "utility": 0.2, "target_id": "nearby_agent", "reachable": True},
            {"action_type": "wait", "utility": 0.1, "target_id": None},
        ],
    )


def test_baseline_deprioritises_an_action_that_keeps_failing_the_same_way():
    streak = {"time": 99.0, "tick": 99, "status": "blocked", "action": "transport_resources", "target": "p1",
              "blocker": "construction_delivery_rejected", "consecutive_identical_failures": 8}
    control = RuleBrain().decide(_context([]))
    assert control.selected_action == ExecutableActionType.TRANSPORT_RESOURCES
    feedback = RuleBrain().decide(_context([streak]))
    assert feedback.selected_action != ExecutableActionType.TRANSPORT_RESOURCES
    # old evidence ages out: the action is retried once the failure is no longer recent
    assert RuleBrain().decide(_context([dict(streak, time=10.0)])).selected_action == ExecutableActionType.TRANSPORT_RESOURCES
    # a short streak (below the repeat threshold) is not treated as a loop
    short = dict(streak, consecutive_identical_failures=2)
    assert RuleBrain().decide(_context([short])).selected_action == ExecutableActionType.TRANSPORT_RESOURCES


def test_action_gate_blocks_transport_when_no_brick_source_exists(sim):
    agent = sim.agents[0]
    decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="resource_zone_to_work_zone", confidence=0.9)
    assert sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim).legal is True
    for pile in sim.environment.construction.resource_nodes.values():
        pile.quantity = 0
    result = sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)
    assert result.legal is False and result.blockers == ["no_pickup_resource_source"]
    # the blocked attempt is translated to an idle with the blocker reported, and recorded as an outcome
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    assert actions[0]["blockers"] == ["no_pickup_resource_source"]
    assert agent.recent_action_outcomes[-1]["blocker"] == "no_pickup_resource_source"


def test_reassess_suppression_does_not_invent_a_transport_that_cannot_happen(sim):
    agent = sim.agents[0]
    cm = sim.environment.construction
    project_id, _ = cm.create_project("site_b", structure_type="house")
    cm.projects[project_id]["started"] = True
    reassess = BrainDecision(selected_action=ExecutableActionType.REASSESS_PLAN, target_id=project_id, confidence=0.5)
    with_bricks = agent._suppress_reassess_when_project_work_available(reassess, sim.environment, sim_state=sim)
    assert with_bricks.selected_action == ExecutableActionType.TRANSPORT_RESOURCES
    for pile in cm.resource_nodes.values():
        pile.quantity = 0
    without = agent._suppress_reassess_when_project_work_available(reassess, sim.environment, sim_state=sim)
    assert without.selected_action != ExecutableActionType.TRANSPORT_RESOURCES


def test_an_agent_carrying_a_brick_can_still_deliver_when_the_piles_are_empty(sim):
    agent = sim.agents[0]
    cm = sim.environment.construction
    project_id, _ = cm.create_project("site_b", structure_type="house")
    for pile in cm.resource_nodes.values():
        pile.quantity = 0
    decision = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id=project_id, confidence=0.9)
    assert sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim).legal is False
    assert "transport_resources" not in {a["action_type"] for a in sim.brain_context_builder.build(sim, agent).action_affordances}

    agent.transport_state["carrying"] = {"resource_type": "bricks", "quantity": 1}  # mid-delivery
    assert sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim).legal is True
    assert "transport_resources" in {a["action_type"] for a in sim.brain_context_builder.build(sim, agent).action_affordances}
    assert agent._brick_source_available(sim.environment) is True
