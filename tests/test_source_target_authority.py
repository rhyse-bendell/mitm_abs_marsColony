"""Pilot authority over INSPECT targets: a legal source the Pilot selected is the source that executes.

Revisit cooldown, exhaustion, "dependency unchanged", an unusable access point and a live pursuit of ANOTHER source are facts the Middleware may report
(events) but none of them may swap the Pilot's substantive target for a different one. A missing/unknown target is not a legal Pilot target; it is still normalised.
"""
import contextlib
import io
import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.simulation import SimulationState

INSPECT = ExecutableActionType.INSPECT_INFORMATION_SOURCE


@pytest.fixture()
def sim():
    with contextlib.redirect_stdout(io.StringIO()):
        s = SimulationState(speed=1.0, flash_mode=True, seed=3, outputs_root=tempfile.mkdtemp())
    yield s
    with contextlib.redirect_stdout(io.StringIO()):
        s.stop()


def _agent(sim):
    return next(a for a in sim.agents if a.role == "Architect")


def _events(sim, name):
    return [e for e in sim.logger.get_recent_events(400) if e["event_type"] == name]


def _decision(target):
    return BrainDecision(selected_action=INSPECT, target_id=target, reason_summary="pilot choice", confidence=0.9)


def test_revisit_cooldown_does_not_replace_the_selected_source(sim):
    agent = _agent(sim)
    agent.source_memory_state.setdefault("Architect_Info", {})["revisit_cooldown_until"] = sim.time + 120.0
    source, position = agent._resolve_inspect_target(_decision("Architect_Info"), sim.environment, sim_state=sim)
    assert source == "Architect_Info" and position is not None
    noted = _events(sim, "source_revisit_noted")
    assert noted and noted[-1]["payload_data"]["reason"] == "revisit_cooldown_active" and noted[-1]["payload_data"]["substituted"] is False
    assert not _events(sim, "next_source_target_selected")  # the candidate re-ranking never ran


def test_exhausted_source_with_unchanged_dependency_does_not_replace_the_selected_source(sim):
    agent = _agent(sim)
    agent.source_inspection_state["Architect_Info"] = "inspected"
    agent.source_exhaustion_state["Architect_Info"] = {"inspect_count": 2, "last_dik_changed": False, "exhausted": True, "exhausted_for_acquisition": True,
                                                       "exhausted_dependency_signature": agent._readiness_dependency_signature(sim.environment, sim_state=sim)}
    assert agent._source_exhausted_without_dependency_change("Architect_Info", sim.environment, sim_state=sim)
    source, _ = agent._resolve_inspect_target(_decision("Architect_Info"), sim.environment, sim_state=sim)
    assert source == "Architect_Info"
    assert not _events(sim, "next_source_target_selected") and not _events(sim, "source_revisit_suppressed")


def test_exhausted_and_not_revisitable_source_is_still_the_selected_source(sim):
    agent = _agent(sim)
    agent.source_exhaustion_state["Team_Info"] = {"inspect_count": 3, "last_dik_changed": False, "exhausted": True, "exhausted_for_acquisition": True}
    agent.source_memory_state.setdefault("Team_Info", {})["revisitable_for_verification"] = False
    source, _ = agent._resolve_inspect_target(_decision("Team_Info"), sim.environment, sim_state=sim)
    assert source == "Team_Info"


def test_unusable_access_point_is_reported_unresolved_not_redirected(sim):
    agent = _agent(sim)
    agent._select_source_access_target = lambda *a, **k: None
    source, position = agent._resolve_inspect_target(_decision("Team_Info"), sim.environment, sim_state=sim)
    assert (source, position) == (None, None)
    failed = _events(sim, "target_resolution_failed")
    assert failed and failed[-1]["payload_data"]["failure_category"] == "source_access_point_unavailable"
    assert not _events(sim, "next_source_target_selected")


def test_a_live_pursuit_of_another_source_does_not_override_a_new_explicit_choice(sim):
    agent = _agent(sim)
    now = float(sim.time)
    agent._commit_inspect_pursuit("Architect_Info", (6.45, 1.1), now, sim_state=sim)
    assert agent._inspect_pursuit_active_for("Architect_Info", now)
    action = agent._translate_brain_decision_to_legacy_action(_decision("Team_Info"), sim.environment, sim_state=sim)[0]
    assert action.get("source_target_id") == "Team_Info"
    assert not _events(sim, "pursuit_reused")
    # continuation of the SAME source (or a decision with no target at all) may still reuse the live pursuit
    agent._commit_inspect_pursuit("Architect_Info", (6.45, 1.1), now, sim_state=sim)
    same = agent._translate_brain_decision_to_legacy_action(_decision("Architect_Info"), sim.environment, sim_state=sim)[0]
    assert same.get("source_target_id") == "Architect_Info" and _events(sim, "pursuit_reused")


def test_missing_or_unknown_target_is_normalised_to_an_accessible_source_never_a_private_one(sim):
    agent = _agent(sim)
    for target in (None, "", "NoSuchSource"):
        source, _ = agent._resolve_inspect_target(_decision(target), sim.environment, sim_state=sim)
        assert source in {"Team_Info", "Architect_Info"}  # shared or own; never Botanist_Info / Engineer_Info


def test_a_private_source_is_refused_not_rerouted(sim):
    agent = _agent(sim)
    source, position = agent._resolve_inspect_target(_decision("Botanist_Info"), sim.environment, sim_state=sim)
    assert (source, position) == (None, None)
    assert not _events(sim, "next_source_target_selected")


def test_end_to_end_every_selected_source_is_the_resolved_source():
    """Procedural run: wherever a Pilot-selected source reaches the resolver, the resolved source is the same one."""
    with contextlib.redirect_stdout(io.StringIO()):
        s = SimulationState(speed=1.0, flash_mode=True, seed=0, outputs_root=tempfile.mkdtemp())
        seen = []
        s.logger.register_event_listener(lambda e: seen.append(e))
        for _ in range(240):
            s.update(1.0)
        s.stop()
    pairs = [(e["payload_data"].get("current_source_target"), e["payload_data"].get("next_source_target")) for e in seen if e["event_type"] == "next_source_target_selected"]
    assert all(cur is None or cur == nxt for cur, nxt in pairs), [p for p in pairs if p[0] is not None and p[0] != p[1]]
    assert not [e for e in seen if e["event_type"] in {"source_revisit_suppressed", "source_revisit_deferred"}]
    assert not [e for e in seen if e["event_type"] == "source_retargeted_due_to_role_private_mismatch"]


def test_revisit_cooldown_is_a_pre_selection_fact_visible_to_every_controller(sim):
    """The cooldown is represented BEFORE the choice: the INSPECT affordance is blocked (Ollama sees it under blocked_actions with a blocker, no id;
    the Procedural Baseline drops it from its candidates). Nothing is swapped after a choice."""
    from modules.controller_context import build_canonical_context, partition_affordances

    agent = _agent(sim)
    sim.time = 20.0
    agent.source_memory_state.setdefault("Team_Info", {})["revisit_cooldown_until"] = 50.0
    packet = sim.brain_context_builder.build(sim, agent)
    aff = next(a for a in packet.action_affordances if a.get("action_type") == "inspect_information_source" and a.get("target_id") == "Team_Info")
    assert aff["source_accessible"] is False and aff["source_blocker"] == "source_revisit_cooldown" and aff["source_cooldown_until"] == 50.0
    ctx = build_canonical_context(packet, agent_id=agent.agent_id, role=agent.role, controller_type="local_ollama")
    assert [b for b in ctx["blocked_actions"] if b.get("target_id") == "Team_Info" and b["blockers"] == ["source_revisit_cooldown"]]
    assert not [x for x in ctx["executable_actions"] if x.get("target_id") == "Team_Info" and x.get("action_type") == "inspect_information_source"]
    sim.time = 60.0  # the cooldown has expired: selectable again
    packet2 = sim.brain_context_builder.build(sim, agent)
    aff2 = next(a for a in packet2.action_affordances if a.get("action_type") == "inspect_information_source" and a.get("target_id") == "Team_Info")
    assert aff2.get("source_accessible") is not False
