"""Pilot authority over substantive actions, and the self-inspection absorbing loop.

Authority: the middleware may translate, gate (block/reject with a reported reason) and execute a substantive Pilot action;
it may not silently replace it with a different action. Rewrite classes pinned here: forced recovery pivot, post-inspect
follow-up, redundant-artifact / whiteboard escape hatch, material-satisfaction successor, plus the legality report.

Inspection: repeated inspection stays useful while it can recover missing/relevant DIK, but a source whose last inspection
gained nothing and whose relevant content (held DIK, construction state, phase) is unchanged cannot keep being pinned
as "stale" by team-level lack of progress.
"""

import random
import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.simulation import SimulationState
from tests.test_experimental_panel import IDS, _age, _held
from tests.test_observed_validation_repair import _ready_house

SUBSTANTIVE = [
    A.INSPECT_INFORMATION_SOURCE, A.CONSULT_TEAM_ARTIFACT, A.COMMUNICATE, A.REQUEST_ASSISTANCE, A.EXTERNALIZE_PLAN, A.MEETING,
    A.TRANSPORT_RESOURCES, A.START_CONSTRUCTION, A.CONTINUE_CONSTRUCTION, A.VALIDATE_CONSTRUCTION, A.REPAIR_OR_CORRECT_CONSTRUCTION, A.CONNECT_STRUCTURES,
]


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="authority", seed=0)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _actor(sim):
    a = next(x for x in sim.agents if x.name == "Architect")
    a.control_state["mode"] = "COORDINATE"  # out of the startup BOOTSTRAP guard, which returns decisions untouched ...
    a.target = (3.0, 3.0)  # ... as does a pending runtime target while start-up grounding is incomplete
    return a


def _pivot(sim, actor, decision):
    return actor._apply_policy_pivots(decision, sim.environment, sim_state=sim, context=sim.brain_context_builder.build(sim, actor), pivot_origin="test")


def _rejections(sim):
    return [p for n, p in sim.captured_events if n == "mecha_strategic_substitution_rejected"]


# ------------------------------------------------------------------ authority: forced recovery pivot


@pytest.mark.parametrize("pilot_action", SUBSTANTIVE)
def test_forced_recovery_pivot_never_replaces_a_substantive_pilot_action(sim, pilot_action):
    actor = _actor(sim)
    forced = A.REASSESS_PLAN if pilot_action == A.INSPECT_INFORMATION_SOURCE else A.INSPECT_INFORMATION_SOURCE
    actor.progress_tracker["forced_pivot"] = forced.value
    actor.progress_tracker["forced_pivot_until"] = sim.time + 60.0
    out = _pivot(sim, actor, BrainDecision(selected_action=pilot_action, target_id=None, confidence=0.9))
    assert out.selected_action == pilot_action
    rejected = _rejections(sim)
    if rejected:  # (an INSPECT Pilot choice may simply not be touched by a forced reassess; whenever a rewrite is proposed it is reported)
        assert rejected[-1]["pilot_action"] == pilot_action.value and rejected[-1]["proposed_action"] == forced.value and rejected[-1]["applied"] is False
    else:
        assert pilot_action == A.INSPECT_INFORMATION_SOURCE


@pytest.mark.parametrize("pilot_action", [A.WAIT, A.OBSERVE_ENVIRONMENT, A.REASSESS_PLAN])
def test_non_substantive_pilot_choices_remain_open_to_regulation(sim, pilot_action):
    actor = _actor(sim)
    actor.progress_tracker["forced_pivot"] = A.INSPECT_INFORMATION_SOURCE.value
    actor.progress_tracker["forced_pivot_until"] = sim.time + 60.0
    out = _pivot(sim, actor, BrainDecision(selected_action=pilot_action, confidence=0.9))
    assert out.selected_action == A.INSPECT_INFORMATION_SOURCE and not _rejections(sim)


# ------------------------------------------------------------------ authority: post-inspect follow-up, escape hatch, redundant artifact


@pytest.mark.parametrize("pilot_action,target", [(A.INSPECT_INFORMATION_SOURCE, "Team_Info"), (A.CONSULT_TEAM_ARTIFACT, "whiteboard"), (A.EXTERNALIZE_PLAN, "whiteboard")])
def test_post_inspect_followup_does_not_substitute_the_pilot_action(sim, pilot_action, target):
    actor = _actor(sim)
    actor.post_inspect_handoff = {"pending": True, "expires_at": sim.time + 30.0, "outcome": "inspect_success_no_new_dik", "dik_changed": False, "readiness_changed": False, "source_id": "Architect_Info"}
    out = _pivot(sim, actor, BrainDecision(selected_action=pilot_action, target_id=target, confidence=0.9))
    assert out.selected_action == pilot_action


@pytest.mark.parametrize("pilot_action", [A.CONSULT_TEAM_ARTIFACT, A.EXTERNALIZE_PLAN])
def test_whiteboard_escape_hatch_and_redundant_artifact_do_not_substitute_the_pilot_action(sim, pilot_action):
    actor = _actor(sim)
    actor.artifact_coordination_state["escape_hatch_pending"] = True
    out = _pivot(sim, actor, BrainDecision(selected_action=pilot_action, target_id="whiteboard", confidence=0.9))
    assert out.selected_action == pilot_action
    actor._artifact_action_is_redundant = lambda *a, **k: True  # a known no-op artifact action is blocked by the gate and reported, not turned into another action
    out = _pivot(sim, actor, BrainDecision(selected_action=pilot_action, target_id="whiteboard", confidence=0.9))
    assert out.selected_action == pilot_action


# ------------------------------------------------------------------ authority: illegal actions are reported, not substituted


def test_illegal_pilot_action_is_reported_as_illegal_and_not_replaced(sim):
    actor = _actor(sim)
    actor.progress_tracker["forced_pivot"] = A.INSPECT_INFORMATION_SOURCE.value
    actor.progress_tracker["forced_pivot_until"] = sim.time + 60.0
    out = _pivot(sim, actor, BrainDecision(selected_action=A.VALIDATE_CONSTRUCTION, target_id="no_such_project", confidence=0.9))
    assert out.selected_action == A.VALIDATE_CONSTRUCTION  # the gate, not a pivot, decides what happens to it
    assert _rejections(sim)[-1]["pilot_action_legal"] is False


def test_material_satisfaction_successor_cannot_replace_a_pilot_transport(sim):
    actor = _actor(sim)
    pid = _ready_house(sim.environment.construction)  # materially satisfied: transporting to it is not needed
    pilot = BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id=pid, confidence=0.9)
    actor._pilot_decision_anchor = pilot
    successor = actor._no_active_plan_successor_for_materially_satisfied_project(pilot, sim.environment, sim_state=sim)
    assert successor.selected_action != A.TRANSPORT_RESOURCES  # the scaffold does propose an alternative ...
    out = actor._enforce_pilot_substantive_authority(successor, sim_state=sim, pivot_origin="local_refresh_successor")
    assert out is pilot  # ... but it is information, not a replacement
    assert _rejections(sim)[-1]["proposed_action"] == successor.selected_action.value


def test_same_action_retargeting_is_translation_not_substitution(sim):
    actor = _actor(sim)
    pilot = BrainDecision(selected_action=A.INSPECT_INFORMATION_SOURCE, target_id="Team_Info", confidence=0.9)
    actor._pilot_decision_anchor = pilot
    retargeted = BrainDecision(selected_action=A.INSPECT_INFORMATION_SOURCE, target_id="Architect_Info", confidence=0.9)
    assert actor._enforce_pilot_substantive_authority(retargeted, sim_state=sim) is retargeted and not _rejections(sim)


def test_fresh_pilot_decision_reanchors_so_a_requery_is_not_a_rewrite(sim):
    actor = _actor(sim)
    first = BrainDecision(selected_action=A.CONSULT_TEAM_ARTIFACT, target_id="whiteboard", confidence=0.9)
    second = BrainDecision(selected_action=A.START_CONSTRUCTION, confidence=0.9)
    actor._pilot_decision_anchor = first
    actor._pilot_decision_anchor = second  # what _select_decision_through_pilot_adapter does for every Pilot (re-)query
    assert actor._enforce_pilot_substantive_authority(second, sim_state=sim) is second and not _rejections(sim)


# ------------------------------------------------------------------ inspection: source-memory confidence


def _verify_static(sim, actor, confidence=0.6):
    """Both grounding sources inspected, nothing gained, content unchanged since (what an exhausted, verified source looks like)."""
    env = sim.environment
    actor._ensure_source_state(env)
    sig = actor._source_content_signature(env, sim)
    for src in ("Team_Info", f"{actor.role}_Info"):
        actor.source_inspection_state[src] = "inspected"
        actor.source_exhaustion_state[src] = {"inspect_count": 2, "exhausted": True, "exhausted_for_acquisition": True, "revisitable_for_verification": True, "exhausted_content_signature": sig}
        actor.source_memory_state[src].update({"ever_inspected": True, "last_inspected_time": sim.time, "last_verified_time": sim.time, "memory_confidence": confidence})
    return sig


def _stall_pressure(actor):
    actor.progress_tracker["no_progress_streak"] = 12  # team has made no progress for a long time
    actor.activity_log.append("mismatch with construction")  # and a recent contradiction


def _confidence(actor):
    return min(actor.source_memory_state[s]["memory_confidence"] for s in ("Team_Info", f"{actor.role}_Info"))


def test_first_meaningful_inspection_is_allowed(sim):
    actor = _actor(sim)
    epi = actor._epistemic_sufficiency(sim.environment, sim_state=sim)
    assert epi["role_missing"] and epi["refresh_pressure"] >= 0.48  # never inspected: strong, legitimate pull to inspect
    packet = sim.brain_context_builder.build(sim, actor)
    role = [a for a in packet.action_affordances if a["action_type"] == "inspect_information_source" and a.get("target_id") == "Architect_Info"]
    assert role and role[0]["reachable"] and role[0]["source_state"] == "unseen"
    assert not actor._source_exhausted_without_dependency_change("Architect_Info", sim.environment, sim_state=sim)


def test_unchanged_verified_source_is_not_pinned_stale_by_team_inactivity(sim):
    actor = _actor(sim)
    _verify_static(sim, actor, confidence=0.6)
    _stall_pressure(actor)
    for _ in range(40):  # many ticks of stalled progress (the old code decayed confidence on every call)
        actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert _confidence(actor) == pytest.approx(0.6)
    assert actor._epistemic_sufficiency(sim.environment, sim_state=sim)["stale_grounding"] is False


def test_unverified_source_still_decays_under_stalled_progress(sim):
    actor = _actor(sim)
    _verify_static(sim, actor, confidence=0.6)
    for src in ("Team_Info", "Architect_Info"):
        actor.source_exhaustion_state[src]["exhausted_for_acquisition"] = False  # the last inspection DID gain something: doubt is still open
        actor.source_exhaustion_state[src]["exhausted_content_signature"] = None
    _stall_pressure(actor)
    for _ in range(10):
        actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert _confidence(actor) <= 0.25


def test_repeated_no_gain_inspection_cannot_pin_the_source_at_the_floor(sim):
    """Each no-gain re-inspection verifies the source (+0.12 in production); under stalled progress the old decay erased that each tick."""
    actor = _actor(sim)
    _verify_static(sim, actor, confidence=0.08)
    _stall_pressure(actor)
    for _ in range(6):
        for src in ("Team_Info", "Architect_Info"):
            m = actor.source_memory_state[src]
            m["memory_confidence"] = max(0.25, min(0.96, m["memory_confidence"] + 0.12))
            m["last_verified_time"] = sim.time
        for _tick in range(5):
            actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert _confidence(actor) > 0.45
    assert actor._epistemic_sufficiency(sim.environment, sim_state=sim)["stale_grounding"] is False  # inspection pressure ends


def test_forgotten_relevant_dik_reopens_the_source_and_can_be_reacquired(sim):
    actor = _actor(sim)
    _held(sim, actor, *IDS)
    actor._sync_dik_ledger(sim)
    sig = _verify_static(sim, actor, confidence=0.6)
    actor.memory_reliability = 0.0
    _age(sim, actor, 900)
    forgotten = [k for k, e in actor.dik_ledger.items() if e["forgotten_at"] is not None]
    assert forgotten
    assert actor._source_content_signature(sim.environment, sim) != sig  # relevant content changed: the verification no longer applies
    assert actor.source_exhaustion_state["Architect_Info"]["exhausted_for_acquisition"] is False
    assert actor.source_inspection_state["Architect_Info"] == "unseen"
    _stall_pressure(actor)
    before = _confidence(actor)
    actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert _confidence(actor) < before  # doubt resumes, so re-inspection is attractive again
    actor.memory_reliability = 1.0
    _held(sim, actor, forgotten[0])  # an ordinary re-inspection restores it: no knowledge is granted any other way
    actor._sync_dik_ledger(sim)
    assert actor.dik_ledger[forgotten[0]]["forgotten_at"] is None and actor.experimental_metrics["dik_reacquired"] >= 1


def test_a_new_knowledge_gap_or_construction_change_reopens_inspection(sim):
    actor = _actor(sim)
    sig = _verify_static(sim, actor, confidence=0.6)
    _stall_pressure(actor)
    actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert _confidence(actor) == pytest.approx(0.6)
    sim.environment.construction.create_project("site_b", "house")  # the world the agent reasons about changed
    assert actor._source_content_signature(sim.environment, sim) != sig
    actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert _confidence(actor) < 0.6


# ------------------------------------------------------------------ end to end (real runs)


def _run(seed, constituent_overrides, seconds, probe_times=()):
    random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="auth_e2e", seed=seed, brain_backend="rule_brain",
                             experimental_settings={"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "constituent_overrides": constituent_overrides, "profile_mode": "compact_panel"})
        last_sel, first_attempt, mismatches, selected = {}, {}, [], 0
        taskwork_at = {}

        def lst(e):
            nonlocal selected
            p = e["payload_data"]
            n = e["event_type"]
            if n == "pilot_selection_via_adapter":
                last_sel[p["agent"]] = p["selected_action"]
                first_attempt[p["agent"]] = True
                selected += 1
            elif n == "startup_bootstrap_grounding_forced":
                first_attempt[p["agent"]] = False  # the documented start-up grounding scaffold (BOOTSTRAP mode) is a separate, explicit fallback
            elif n == "executable_action_attempted" and first_attempt.get(p.get("agent")):
                first_attempt[p["agent"]] = False
                if last_sel.get(p["agent"]) and p.get("planner_action_type") != last_sel[p["agent"]]:
                    mismatches.append((last_sel[p["agent"]], p.get("planner_action_type")))

        st.logger.register_event_listener(lst)
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            for i in range(seconds):
                st.update(1.0)
                if (i + 1) in probe_times:
                    taskwork_at[i + 1] = sum(a.experimental_metrics.get("taskwork_actions", 0) for a in st.agents)
            st.stop()
    return mismatches, selected, taskwork_at


def test_substantive_pilot_actions_are_not_rewritten_end_to_end():
    mismatches, selected, _ = _run(seed=0, constituent_overrides={}, seconds=240)
    assert selected > 100
    substantive = {a.value for a in SUBSTANTIVE}
    assert [m for m in mismatches if m[0] in substantive] == []  # what the Pilot selected is what was attempted


def test_extern_low_seed4_does_not_enter_the_self_inspection_absorbing_loop():
    """Characterization batch case: from t~100 all agents re-inspected their own role source and taskwork froze at 7 for the rest of the run."""
    _, _, taskwork = _run(seed=4, constituent_overrides={"externalization_propensity": 0.1}, seconds=420, probe_times=(100, 420))
    assert taskwork[420] > taskwork[100] + 10
