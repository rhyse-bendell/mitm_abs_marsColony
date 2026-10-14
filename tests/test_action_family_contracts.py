"""Gate I: canonical-path contracts for the Pilot-facing action families.

For each family: legal invocation works through the owning subsystem, invalid invocation does not
corrupt state, and the authoritative mutation happens where it should (agent memory / team
knowledge / Construction), never in the Pilot.
"""

import copy
import tempfile
from unittest.mock import patch

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState

A = ExecutableActionType

READY = {
    "score": 1.0,
    "role_missing": False,
    "stale_grounding": False,
    "refresh_pressure": 0.0,
    "construction_state_needs_recheck": False,
    "sufficient_for_validation": True,
}


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="family_contracts", seed=4)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _decision(action, target=None):
    return BrainDecision(selected_action=action, target_id=target, confidence=0.9)


def _construction_snapshot(sim):
    cm = sim.environment.construction
    return copy.deepcopy(
        (cm.projects, {k: p.quantity for k, p in cm.resource_nodes.items()}, {k: (b.status, b.delivered_resources) for k, b in cm.bridges.items()})
    )


def _agent(sim, role):
    return next(a for a in sim.agents if a.role == role)


def _execute(sim, agent, decision):
    actions = agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    agent._apply_externalization_and_construction_effects(sim.environment, sim, 1.0)
    return actions


# ------------------------------------------------------------------ INSPECT_INFORMATION_SOURCE


def test_inspect_changes_agent_memory_only_and_a_private_source_of_another_role_is_rejected(sim):
    agent = _agent(sim, "Architect")
    held = lambda: len(agent.mental_model["information"]) + len(agent.mental_model["data"])
    n_before = held()
    bricks_before = {k: p.quantity for k, p in sim.environment.construction.resource_nodes.items()}
    for _ in range(25):  # the baseline's first move is to inspect: movement -> slot -> inspect -> DIK
        sim.update(1.0)
    assert held() > n_before  # DIK acquired by the agent through the inspect path
    inspected = {p.get("source_id") for n, p in sim.captured_events if n == "inspect_completed" and p.get("agent") == "Architect"}
    assert inspected and inspected <= {"Team_Info", "Architect_Info"}  # never another role's private packet
    inspect_events = [n for n, _ in sim.captured_events if n.startswith("inspect_")]
    assert inspect_events  # inspection is logged by the agent; it produced no construction-side effect of its own
    assert all(n != "construction_resource_delivered" for n, _ in sim.captured_events)
    assert {k: p.quantity for k, p in sim.environment.construction.resource_nodes.items()} == bricks_before

    allowed, _reason = agent._agent_can_target_information_source("Botanist_Info", sim.environment)
    assert allowed is False
    snapshot = held()
    actions = agent._translate_brain_decision_to_legacy_action(_decision(A.INSPECT_INFORMATION_SOURCE, "Botanist_Info"), sim.environment, sim_state=sim)
    assert actions[0].get("source_target_id") != "Botanist_Info"
    assert held() == snapshot


# ------------------------------------------------------------------ COMMUNICATE / REQUEST_ASSISTANCE


def _run_communicate(sim, agent, action_enum):
    actions = agent._translate_brain_decision_to_legacy_action(_decision(action_enum, "nearby_agent"), sim.environment, sim_state=sim)
    assert actions[0]["type"] == "communicate"
    agent.active_actions = [dict(a, progress=0.0) for a in actions]
    agent._advance_active_actions(1.0, sim_state=sim)


@pytest.mark.parametrize("action_enum", [A.COMMUNICATE, A.REQUEST_ASSISTANCE])
def test_communication_reaches_a_nearby_teammate_and_is_ineffective_without_one(sim, action_enum):
    arch, eng = _agent(sim, "Architect"), _agent(sim, "Engineer")
    eng.position = (arch.position[0] + 1.0, arch.position[1])
    before_world = _construction_snapshot(sim)
    _run_communicate(sim, arch, action_enum)
    assert any(n == "communication_exchange" and p.get("sender") == "Architect" for n, p in sim.captured_events)
    assert _construction_snapshot(sim) == before_world

    sim.captured_events.clear()
    for other in (eng, _agent(sim, "Botanist")):
        other.position = (arch.position[0] + 500.0, arch.position[1])  # nobody in range
    streak = arch.communication_state.get("no_effect_streak", 0)
    _run_communicate(sim, arch, action_enum)
    assert any(n == "communication_ineffective" for n, _ in sim.captured_events)
    assert arch.communication_state.get("no_effect_streak", 0) == streak + 1
    assert not any(n == "communication_exchange" for n, _ in sim.captured_events)


# ------------------------------------------------------------------ EXTERNALIZE_PLAN / CONSULT_TEAM_ARTIFACT


def test_externalize_writes_team_knowledge_and_consult_reads_it(sim):
    author, reader = _agent(sim, "Architect"), _agent(sim, "Botanist")
    whiteboard = sim.environment.get_interaction_target_position("whiteboard", from_position=author.position)
    team_knowledge = sim.team_knowledge_manager
    before_world = _construction_snapshot(sim)
    n_artifacts = len(team_knowledge.artifacts)

    author.position = whiteboard
    _execute(sim, author, _decision(A.EXTERNALIZE_PLAN, "whiteboard"))
    assert len(team_knowledge.artifacts) > n_artifacts  # team knowledge, the owning subsystem, holds the artifact
    assert _construction_snapshot(sim) == before_world

    reader.position = whiteboard
    n_after_write = len(team_knowledge.artifacts)
    _execute(sim, reader, _decision(A.CONSULT_TEAM_ARTIFACT, "whiteboard"))
    assert any(n == "artifact_consulted" and p.get("agent") == "Botanist" for n, p in sim.captured_events)
    assert len(team_knowledge.artifacts) == n_after_write  # consulting neither creates nor destroys artifacts


# ------------------------------------------------------------------ TRANSPORT / START / CONTINUE


def test_transport_delivery_mutates_through_construction_and_is_gated_without_a_source(sim):
    agent = _agent(sim, "Engineer")
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", structure_type="house")
    assert sim.action_gate.evaluate(agent=agent, decision=_decision(A.TRANSPORT_RESOURCES, pid), sim_state=sim).legal is True

    agent.position = sim.environment.get_interaction_target_position(pid, from_position=agent.position)
    agent.transport_state.update(
        {"stage": "in_transit", "carrying": {"resource_type": "bricks", "quantity": 1}, "pickup_source_id": "pile_a", "bound_project_id": pid}
    )
    with patch.object(cm, "deliver_resource", wraps=cm.deliver_resource) as spy:
        _execute(sim, agent, _decision(A.TRANSPORT_RESOURCES, pid))
    assert spy.called and cm.projects[pid]["delivered_resources"]["bricks"] == 1

    for pile in cm.resource_nodes.values():
        pile.quantity = 0
    snapshot = _construction_snapshot(sim)
    result = sim.action_gate.evaluate(agent=agent, decision=_decision(A.TRANSPORT_RESOURCES, pid), sim_state=sim)
    assert result.legal is False and result.blockers == ["no_pickup_resource_source"]
    assert _construction_snapshot(sim) == snapshot


def test_construction_builds_through_construction_and_refuses_unstaged_materials(sim):
    agent = _agent(sim, "Architect")
    cm = sim.environment.construction
    pid, _ = cm.create_project("site_b", structure_type="house")
    agent.position = sim.environment.get_interaction_target_position(pid, from_position=agent.position)
    steps = lambda: sum(1 for s in cm.projects[pid]["build_steps"] if s.get("completed"))

    def run(action):
        with patch.object(agent, "_build_readiness_blockers", return_value=[]), patch.object(agent, "_epistemic_sufficiency", return_value=READY):
            _execute(sim, agent, _decision(action, pid))

    run(A.START_CONSTRUCTION)  # no materials staged: refused, nothing built
    assert steps() == 0 and cm.projects[pid]["structurally_complete"] is False

    required = cm.projects[pid]["required_resources"]["bricks"]
    cm.projects[pid]["started"] = True
    cm.projects[pid]["staged_resources"] = {"bricks": required}
    cm.projects[pid]["delivered_resources"] = {"bricks": required}
    cm.update()
    with patch.object(cm, "execute_build_step", wraps=cm.execute_build_step) as spy:
        run(A.START_CONSTRUCTION)
        first = steps()
        run(A.CONTINUE_CONSTRUCTION)
    assert spy.called and first >= 1 and steps() >= first  # steps are completed by Construction.execute_build_step


# ------------------------------------------------------------------ WAIT / REASSESS / OBSERVE


@pytest.mark.parametrize("action_enum", [A.WAIT, A.REASSESS_PLAN, A.OBSERVE_ENVIRONMENT])
def test_passive_actions_translate_to_idle_and_do_not_touch_the_world(sim, action_enum):
    agent = _agent(sim, "Botanist")
    before = _construction_snapshot(sim)
    actions = _execute(sim, agent, _decision(action_enum))
    assert actions[0]["type"] == "idle" and actions[0]["decision_action"] == action_enum.value
    assert _construction_snapshot(sim) == before


# ------------------------------------------------------------------ affordance grounding / no implicit creation


def test_start_construction_is_offered_only_for_buildable_sites_with_capacity(sim):
    agent = _agent(sim, "Architect")
    offered = [(a["action_type"], a.get("target_id")) for a in sim.brain_context_builder.build(sim, agent).action_affordances]
    starts = {t for act, t in offered if act == "start_construction"}
    assert "Build_Site_B" in starts
    assert "Build_Site_A" not in starts  # site_a capacity is 0
    assert "whiteboard" not in starts  # an artifact is not a construction site


@pytest.mark.parametrize(
    "decision",
    [_decision(A.EXTERNALIZE_PLAN, "whiteboard"), _decision(A.CONSULT_TEAM_ARTIFACT, "whiteboard"), _decision(A.COMMUNICATE, "nearby_agent"), _decision(A.WAIT)],
)
def test_translating_non_construction_actions_never_creates_projects(sim, decision):
    agent = _agent(sim, "Architect")
    agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)
    assert sim.environment.construction.projects == {}


def test_a_committed_start_instantiates_its_project_through_construction(sim):
    agent = _agent(sim, "Architect")
    cm = sim.environment.construction
    assert cm.projects == {}
    with patch.object(cm, "create_project", wraps=cm.create_project) as spy:
        with patch.object(agent, "_build_readiness_blockers", return_value=[]), patch.object(agent, "_epistemic_sufficiency", return_value=READY):
            agent._translate_brain_decision_to_legacy_action(_decision(A.START_CONSTRUCTION, "Build_Site_B"), sim.environment, sim_state=sim)
    assert spy.called and len(cm.projects) == 1 and next(iter(cm.projects.values()))["site_id"] == "site_b"
