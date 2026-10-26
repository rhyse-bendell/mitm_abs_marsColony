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


def test_a_new_knowledge_gap_reopens_inspection_but_a_construction_change_alone_does_not(sim):
    actor = _actor(sim)
    sig = _verify_static(sim, actor, confidence=0.6)
    _stall_pressure(actor)
    sim.environment.construction.create_project("site_b", "house")  # the world changed: a static brief is no more informative for it
    actor._refresh_source_memory_state(sim.environment, sim_state=sim)
    assert actor._source_content_signature(sim.environment, sim) == sig and _confidence(actor) == pytest.approx(0.6)
    actor.known_gaps.add("information:I_ENG_CONNECTOR_RULES")  # a newly recognised gap the source might fill
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
    assert selected > 80  # a sufficiency floor, not a rate: neutral durations are the authored ones since the single-application duration fix (was x0.875)
    substantive = {a.value for a in SUBSTANTIVE}
    assert [m for m in mismatches if m[0] in substantive] == []  # what the Pilot selected is what was attempted


def test_extern_low_seed4_does_not_enter_the_self_inspection_absorbing_loop():
    """Characterization batch case: from t~100 all agents re-inspected their own role source and taskwork froze at 7 for the rest of the run."""
    _, _, taskwork = _run(seed=4, constituent_overrides={"externalization_propensity": 0.1}, seconds=420, probe_times=(100, 420))
    assert taskwork[420] > taskwork[100] + 10


# ------------------------------------------------------------------ start-up grounding is the Pilot's own behaviour


def test_pilot_itself_selects_the_required_bootstrap_inspections(sim):
    actor = next(x for x in sim.agents if x.name == "Architect")
    assert actor.control_state["mode"] == "BOOTSTRAP"
    decide = lambda: actor._select_decision_through_pilot_adapter(sim, sim.brain_context_builder.build(sim, actor), "test")  # noqa: E731
    first = decide()
    assert first.selected_action == A.INSPECT_INFORMATION_SOURCE and first.target_id == "Team_Info"  # shared brief first
    assert not [e for n, e in sim.captured_events if n == "startup_bootstrap_grounding_forced"]  # nothing was substituted to get here


def test_bootstrap_inspection_bonus_is_scoped_to_pending_required_sources(sim):
    from modules.brain_provider import RuleBrain

    brain = RuleBrain()
    actor = next(x for x in sim.agents if x.name == "Architect")
    features = brain._extract_features(sim.brain_context_builder.build(sim, actor))
    affs = [{"action_type": "inspect_information_source", "target_id": t, "utility": 0.95} for t in ("Team_Info", "Architect_Info", "Engineer_Info")]
    kw = dict(sorted_affordances=affs, features=features, mechanism_profile={})
    pending = brain._score_actions_for_mode(selected_mode="BOOTSTRAP", startup_required_pending=frozenset({"Team_Info"}), **kw)
    none = brain._score_actions_for_mode(selected_mode="BOOTSTRAP", startup_required_pending=frozenset(), **kw)
    other_mode = brain._score_actions_for_mode(selected_mode="ACQUIRE_DIK", startup_required_pending=frozenset({"Team_Info"}), **kw)
    assert pending["0:inspect_information_source"] == pytest.approx(none["0:inspect_information_source"] + brain.STARTUP_GROUNDING_BONUS)
    assert pending["1:inspect_information_source"] == pytest.approx(none["1:inspect_information_source"])  # not required: no bonus
    assert other_mode["0:inspect_information_source"] < pending["0:inspect_information_source"] - 10  # only the Pilot's BOOTSTRAP mode


def test_startup_grounding_completes_without_any_rewrite_or_extra_dik():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="boot", seed=0, brain_backend="rule_brain",
                             experimental_settings={"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "constituent_overrides": {}, "profile_mode": "compact_panel"})
        names = []
        st.logger.register_event_listener(lambda e: names.append(e["event_type"]))
        import contextlib
        import io

        prefix = {"Architect": ("D_ARCH_", "I_ARCH_"), "Engineer": ("D_ENG_", "I_ENG_"), "Botanist": ("D_BOT_", "I_BOT_")}
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(12):  # before any teammate could have communicated: only the agent's own inspections can have supplied DIK
                st.update(1.0)
            for agent in st.agents:
                held = {d.id for d in agent.mental_model["data"]} | {i.id for i in agent.mental_model["information"]}
                assert not [h for role, pre in prefix.items() if role != agent.role for h in held if h.startswith(pre)]  # no role-private DIK of another role
            for _ in range(48):
                st.update(1.0)
        assert "startup_bootstrap_grounding_forced" not in names and "fallback_bootstrap_action_forced" not in names
        for agent in st.agents:  # grounding still completes, by the Pilot's own inspections
            assert agent.source_memory_state["Team_Info"]["ever_inspected"] and agent.source_memory_state[f"{agent.role}_Info"]["ever_inspected"]
        st.stop()


# ------------------------------------------------------------------ repair obligations: class pressure + aging, not candidate counts


def _failed_house(sim, age_s):
    cm = sim.environment.construction
    pid = _ready_house(cm)
    p = cm.projects[pid]
    p.update({"correct": False, "validated_complete": False, "validation_attempted": True, "closure_status": "failed", "status": "needs_repair", "last_validation_attempt_time": sim.time - age_s})
    return pid


def test_repair_affordance_states_how_long_the_observed_failure_has_stood(sim):
    actor = next(x for x in sim.agents if x.name == "Engineer")
    sim.time = 100.0
    pid = _failed_house(sim, age_s=30.0)
    repair = [a for a in sim.brain_context_builder.build(sim, actor).action_affordances if a["action_type"] == "repair_or_correct_construction" and a["target_id"] == pid]
    assert repair and repair[0]["observed_failure_age_s"] == pytest.approx(30.0)


def _real_features(sim, **override):
    from modules.brain_provider import RuleBrain

    brain = RuleBrain()
    actor = next(x for x in sim.agents if x.name == "Engineer")
    f = brain._extract_features(sim.brain_context_builder.build(sim, actor))
    f.update(override)
    return brain, f


def test_validate_mode_does_not_grow_with_the_number_of_validation_candidates(sim):
    brain, f = _real_features(sim, validation_work_pending=1.0, validation_pending_projects=0.33)
    few = brain._compute_mode_scores(f, {"validate_construction"}, {"mode": "LOGISTICS"}, {})[0]["VALIDATE"]
    f["validation_pending_projects"] = 1.0
    many = brain._compute_mode_scores(f, {"validate_construction"}, {"mode": "LOGISTICS"}, {})[0]["VALIDATE"]
    assert few == pytest.approx(many)


@pytest.mark.parametrize("many_candidates", [0.33, 1.0])
def test_an_aged_executable_repair_obligation_outranks_an_incumbent_validate_mode(sim, many_candidates):
    brain, f = _real_features(sim, validation_work_pending=1.0, validation_pending_projects=many_candidates, artifact_validation_available=1.0,
                              observed_repair_need=1.0, repair_executable=1.0, repair_pressure=1.0, repair_obligation_age=1.0)
    scores = brain._compute_mode_scores(f, {"validate_construction", "repair_or_correct_construction"}, {"mode": "VALIDATE", "mode_dwell_steps": 6}, {})[0]
    assert scores["REPAIR"] > scores["VALIDATE"]


def test_repair_mode_pressure_increases_with_the_age_of_the_obligation(sim):
    brain, f = _real_features(sim, observed_repair_need=1.0, repair_executable=1.0, repair_pressure=1.0)
    score = lambda age: brain._compute_mode_scores({**f, "repair_obligation_age": age}, {"repair_or_correct_construction"}, {"mode": "LOGISTICS"}, {})[0]["REPAIR"]  # noqa: E731
    assert score(0.0) < score(0.5) < score(1.0)


def test_no_obligation_pressure_without_the_knowledge_to_repair(sim):
    brain, f = _real_features(sim)
    base = {"action_type": "repair_or_correct_construction", "target_id": "p", "utility": 0.9, "observed_failure_age_s": 90.0}
    kw = dict(features=f, mechanism_profile={}, selected_mode="REPAIR")
    ready = brain._score_actions_for_mode(sorted_affordances=[{**base, "repair_ready": True}], **kw)["0:repair_or_correct_construction"]
    young = brain._score_actions_for_mode(sorted_affordances=[{**base, "repair_ready": True, "observed_failure_age_s": 0.0}], **kw)["0:repair_or_correct_construction"]
    unready = brain._score_actions_for_mode(sorted_affordances=[{**base, "repair_ready": False}], **kw)["0:repair_or_correct_construction"]
    assert ready > young > unready + 3.0  # aged > fresh; a repair the agent cannot ground gets no pressure (and keeps its existing penalty)
    # the feature only counts affordances the agent can actually ground
    actor = next(x for x in sim.agents if x.name == "Engineer")
    sim.time = 100.0
    _failed_house(sim, age_s=60.0)
    actor.mental_model["knowledge"].rules.clear()  # holds nothing that could ground the repair
    feats = brain._extract_features(sim.brain_context_builder.build(sim, actor))
    assert feats["repair_executable"] == 0.0 and feats["repair_obligation_age"] == 0.0 and feats["repair_knowledge_gap"] == 1.0


# ------------------------------------------------------------------ the Pilot is told when re-reading a source cannot inform it


def _inspect_aff(sim, actor, source):
    packet = sim.brain_context_builder.build(sim, actor)
    return next((a for a in packet.action_affordances if a["action_type"] == "inspect_information_source" and a["target_id"] == source), None)


def test_reinspection_is_flagged_uninformative_only_for_an_exhausted_unchanged_source(sim):
    actor = _actor(sim)
    assert _inspect_aff(sim, actor, "Architect_Info")["reinspect_informative"] is True  # never inspected: first inspection is informative
    _verify_static(sim, actor, confidence=0.3)
    aff = _inspect_aff(sim, actor, "Architect_Info")
    assert aff is not None and aff["reinspect_informative"] is False  # exhausted, nothing changed: still legal, but not informative
    actor.known_gaps.add("information:I_ENG_CONNECTOR_RULES")
    assert _inspect_aff(sim, actor, "Architect_Info")["reinspect_informative"] is True  # a new gap reopens it
    actor.known_gaps.discard("information:I_ENG_CONNECTOR_RULES")
    assert _inspect_aff(sim, actor, "Architect_Info")["reinspect_informative"] is False
    actor.source_exhaustion_state["Architect_Info"]["exhausted_for_acquisition"] = False  # the last inspection DID gain something
    assert _inspect_aff(sim, actor, "Architect_Info")["reinspect_informative"] is True


def test_forgotten_dik_makes_reinspection_informative_again(sim):
    actor = _actor(sim)
    _held(sim, actor, *IDS)
    actor._sync_dik_ledger(sim)
    _verify_static(sim, actor, confidence=0.3)
    assert _inspect_aff(sim, actor, "Architect_Info")["reinspect_informative"] is False
    actor.memory_reliability = 0.0
    _age(sim, actor, 900)
    assert _inspect_aff(sim, actor, "Architect_Info")["reinspect_informative"] is True  # what could be recovered is real: inspecting stays useful


def test_pilot_penalises_uninformative_reinspection_but_not_the_first_inspection(sim):
    from modules.brain_provider import RuleBrain

    brain = RuleBrain()
    actor = _actor(sim)
    features = brain._extract_features(sim.brain_context_builder.build(sim, actor))
    kw = dict(features=features, mechanism_profile={}, selected_mode="ACQUIRE_DIK")
    base = {"action_type": "inspect_information_source", "target_id": "Architect_Info", "utility": 0.95}
    new = brain._score_actions_for_mode(sorted_affordances=[{**base, "reinspect_informative": True}], **kw)["0:inspect_information_source"]
    old = brain._score_actions_for_mode(sorted_affordances=[{**base, "reinspect_informative": False}], **kw)["0:inspect_information_source"]
    absent = brain._score_actions_for_mode(sorted_affordances=[dict(base)], **kw)["0:inspect_information_source"]
    assert new - old == pytest.approx(brain.NONPRODUCTIVE_REINSPECT_PENALTY) and absent == pytest.approx(new)  # a generic affordance without the fact is unaffected


def test_an_uninformative_reinspection_earns_no_method_step_credit():
    from modules.brain_provider import RuleBrain

    brain = RuleBrain()
    affs = [
        {"action_type": "inspect_information_source", "target_id": "Architect_Info", "utility": 0.95, "reinspect_informative": False},
        {"action_type": "inspect_information_source", "target_id": "Team_Info", "utility": 0.95, "reinspect_informative": True},
        {"action_type": "inspect_information_source", "target_id": "Engineer_Info", "utility": 0.95},  # an affordance without the fact (generic Pilot input)
    ]
    scores = brain._score_actions_for_method_step(sorted_affordances=affs, step_id="inspect_role_source", method_state={"role_source_id": "Architect_Info"})
    assert "0:inspect_information_source" not in scores and "1:inspect_information_source" in scores and "2:inspect_information_source" in scores
