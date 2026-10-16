"""Minimal team-plan lifecycle: PROPOSE -> RESPOND -> COMMIT (explicit support only) -> LOCAL ADOPTION -> Pilot PREFERENCE -> REASSESS.

What is pinned: the commitment rule (proposer + one explicit non-proposer supporter; silence/clarification/decline never count;
material revision needs new support), assignments built only from the proposer's own legal affordances, grounded adoption as a
Pilot-visible preference (never a forced action, never a hidden-truth assignment), reassessment on completion/blockage/review/expiry,
supersession/expiry, and that low coordination_propensity gates engagement without disabling taskwork.
"""

import contextlib
import io
import random
import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.simulation import SimulationState
from modules.team_knowledge import TeamKnowledgeManager
from modules.team_plan_lifecycle import PlanProcessTracker, load_team_plan_policy


def _plan(m, plan_id="p1", proposer="Architect", t=0.0, assignments=None, **kw):
    return m.propose_team_plan(
        plan_id=plan_id, proposed_by=proposer, sim_time=t, goal_ids=["g"], goal_summary="g", project_targets=["proj_a"],
        assignments_by_role=assignments if assignments is not None else {"Architect": {"task": "build", "action_type": "continue_construction", "project_target": "proj_a"}},
        supporters=kw.pop("supporters", [proposer]), expires_at=kw.pop("expires_at", 100.0), review_at=t + 20.0, **kw,
    )


def _status(m, plan_id="p1"):
    return m.artifacts[f"team_plan:{plan_id}"].content["status"]


def _respond(m, who, kind, t=1.0, plan_id="p1"):
    return m.record_team_plan_response(plan_id=plan_id, responder=who, response_type=kind, sim_time=t, role=who, reason="test")


# ------------------------------------------------------------------ commitment rule (A-E)


def test_A_proposer_alone_cannot_commit():
    m = TeamKnowledgeManager()
    _plan(m)
    assert _status(m) == "proposed"
    assert m.commit_team_plan(plan_id="p1", committed_by="Architect", sim_time=1.0) is None  # the explicit API enforces the same rule
    assert _status(m) == "proposed"


def test_B_proposer_plus_one_non_proposer_supporter_commits():
    m = TeamKnowledgeManager()
    _plan(m)
    _respond(m, "Botanist", "agree")
    assert _status(m) == "committed"
    committed = [e for e in m.drain_lifecycle_events() if e["event_type"] == "team_plan_committed"]
    assert committed and committed[0]["payload"]["supporters"] == ["Architect", "Botanist"]


def test_B2_assignment_accept_is_explicit_support_but_the_proposer_must_also_support():
    m = TeamKnowledgeManager()
    _plan(m)
    _respond(m, "Engineer", "assignment_accept")
    assert _status(m) == "committed"
    m2 = TeamKnowledgeManager()
    _plan(m2, supporters=[])  # the proposer has not supported its own proposal
    _respond(m2, "Engineer", "agree")
    assert _status(m2) == "proposed"


def test_C_silence_is_not_support():
    m = TeamKnowledgeManager()
    _plan(m)
    assert m.expire_due_plans(50.0) == [] and _status(m) == "proposed"
    assert m.artifacts["team_plan:p1"].content["supporters"] == ["Architect"]
    m.expire_due_plans(100.0)
    assert _status(m) == "expired"  # never committed by waiting


def test_D_clarification_and_decline_do_not_count_as_support():
    m = TeamKnowledgeManager()
    _plan(m)
    _respond(m, "Botanist", "request_clarification")
    _respond(m, "Engineer", "disagree")
    _respond(m, "Botanist", "assignment_decline")
    assert _status(m) == "proposed"
    assert m.artifacts["team_plan:p1"].content["supporters"] == ["Architect"]
    _respond(m, "Botanist", "agree", t=2.0)
    assert _status(m) == "committed"


def test_E_materially_revised_plan_needs_new_support_but_an_unchanged_one_keeps_it():
    m = TeamKnowledgeManager()
    m.commit_policy["commit_rule"]["min_total_supporters"] = 3
    _plan(m)
    _respond(m, "Botanist", "agree")
    assert _status(m) == "proposed"  # 2 of 3 under this stricter quorum
    _plan(m, assignments={"Architect": {"task": "validate", "action_type": "validate_construction", "project_target": "proj_a"}}, t=3.0)  # material change
    content = m.artifacts["team_plan:p1"].content
    assert content["supporters"] == ["Architect"]
    _respond(m, "Engineer", "agree", t=4.0)
    assert _status(m) == "proposed"  # Botanist's earlier support did not silently carry over
    _respond(m, "Botanist", "agree", t=5.0)
    assert _status(m) == "committed"
    m2 = TeamKnowledgeManager()
    m2.commit_policy["commit_rule"]["min_total_supporters"] = 3
    _plan(m2)
    _respond(m2, "Botanist", "agree")
    _plan(m2, t=3.0)  # identical content: the same proposal, support intact
    assert set(m2.artifacts["team_plan:p1"].content["supporters"]) == {"Architect", "Botanist"}


def test_policy_is_read_from_the_task_package_not_hard_coded():
    import json
    from pathlib import Path

    manifest = json.loads(Path("config/tasks/mars_colony/task_manifest.json").read_text(encoding="utf-8"))
    policy = load_team_plan_policy(manifest)
    assert policy["commit_rule"] == {"proposer_must_support": True, "min_total_supporters": 2, "min_non_proposer_supporters": 1}
    stricter = load_team_plan_policy({"team_plan_policy": {"commit_rule": {"min_total_supporters": 3}}})
    assert stricter["commit_rule"]["min_total_supporters"] == 3 and stricter["commit_rule"]["proposer_must_support"] is True


# ------------------------------------------------------------------ supersession / expiry (J)


def test_J_supersession_and_expiry():
    m = TeamKnowledgeManager()
    _plan(m, "p1")
    _respond(m, "Botanist", "agree", plan_id="p1")
    _plan(m, "p2", proposer="Botanist", t=5.0, supersedes_plan_id="p1", assignments={"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "proj_a"}})
    _respond(m, "Architect", "agree", t=6.0, plan_id="p2")
    assert _status(m, "p2") == "committed" and _status(m, "p1") == "superseded"
    assert m.get_active_team_plan()["plan_id"] == "p2"
    names = [e["event_type"] for e in m.drain_lifecycle_events()]
    assert names.count("team_plan_committed") == 2 and "team_plan_superseded" in names
    m.expire_due_plans(1000.0)
    assert _status(m, "p2") == "expired" and m.get_active_team_plan() is None
    assert "team_plan_expired" in [e["event_type"] for e in m.drain_lifecycle_events()]


# ------------------------------------------------------------------ agent side


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="planlife", seed=0)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _agent(sim, name):
    return next(x for x in sim.agents if x.name == name)


def _afford(action, target, **kw):
    return {"action_type": action, "target_id": target, "reachable": True, "utility": 0.9, **kw}


def _names(sim, event):
    return [p for n, p in sim.captured_events if n == event]


def _commit_for(sim, assignments, proposer="Architect", supporter="Botanist"):
    m = sim.team_knowledge_manager
    _plan(m, "px", proposer=proposer, t=sim.time, assignments=assignments, expires_at=sim.time + 200.0)
    _respond(m, supporter, "agree", t=sim.time, plan_id="px")
    return m._team_plan_summary(dict(m.artifacts["team_plan:px"].content))


def test_F_committed_plan_is_visible_and_adoptable_by_a_teammate(sim):
    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("transport_resources", "proj_a")]
    summary = _commit_for(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "proj_a"}})
    assert sim.team_knowledge_manager.get_active_team_plan()["status"] == "committed"
    ctx = sim.brain_context_builder.build(sim, bot)
    assert ctx.team_state["team_shared_knowledge"]["active_team_plan"]["plan_id"] == "px"
    bot._adopt_committed_team_plan(sim, summary, reason="test")
    assert bot.team_plan_state["assignment_state"]["status"] == "applied"
    applied = _names(sim, "team_plan_assignment_applied")
    assert applied and applied[-1]["grounded"] is True and applied[-1]["plan_id"] == "px"
    assert _names(sim, "team_plan_adopted_locally") and _names(sim, "team_plan_assignment_acknowledged")


def test_F2_recognition_comes_from_reading_the_plan_not_from_the_store(sim):
    bot = _agent(sim, "Botanist")
    _commit_for(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "proj_a"}})
    assert bot._team_plan_requires_uptake(sim) is False  # never read: no per-tick omniscient adoption
    assert not bot.team_plan_state["last_adopted_plan_id"]


def test_G_assignment_is_a_preference_not_a_command(sim):
    from modules.brain_provider import RuleBrain

    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("transport_resources", "proj_a")]
    bot._adopt_committed_team_plan(sim, _commit_for(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "proj_a"}}), reason="test")
    ann = bot.plan_assignment_annotation("transport_resources", "proj_a")
    assert ann == {"plan_id": "px"} and bot.plan_assignment_annotation("transport_resources", "other") is None and bot.plan_assignment_annotation("validate_construction", "proj_a") is None
    brain = RuleBrain()
    feats = brain._extract_features(sim.brain_context_builder.build(sim, bot))
    base = {"action_type": "transport_resources", "target_id": "proj_a", "utility": 0.9}
    score = lambda aff, coord: brain._score_actions_for_mode(sorted_affordances=[aff], features=feats, mechanism_profile={"coordination_propensity": coord}, selected_mode="LOGISTICS")["0:transport_resources"]  # noqa: E731
    with_pref, without = score({**base, "plan_assignment_plan_id": "px"}, 0.8), score(base, 0.8)
    assert with_pref - without == pytest.approx(brain.EXPERIMENTAL_TERM_WEIGHTS["plan_assignment"] * 0.8)
    assert score({**base, "plan_assignment_plan_id": "px"}, 0.0) == pytest.approx(score(base, 0.0))  # no coordination, no pull
    # a Pilot that picks something else is not overridden: the deviation is reported and the choice stands
    ctx = sim.brain_context_builder.build(sim, bot)
    ctx.action_affordances = [{**base, "plan_assignment_plan_id": "px"}]
    other = BrainDecision(selected_action=A.WAIT, target_id=None, confidence=0.5)
    bot._note_plan_assignment_selection(sim, other, ctx)
    assert other.selected_action == A.WAIT
    dev = _names(sim, "team_plan_assignment_deviated")
    assert dev and dev[-1]["reason"] == "pilot_chose_other_legal_action" and bot.team_plan_state["assignment_state"]["status"] == "applied"
    chosen = BrainDecision(selected_action=A.TRANSPORT_RESOURCES, target_id="proj_a", confidence=0.5)
    bot._note_plan_assignment_selection(sim, chosen, ctx)
    assert bot.team_plan_state["assignment_state"]["status"] == "started" and _names(sim, "team_plan_assignment_started")


def test_H_ungrounded_assignment_is_recorded_and_never_executed_magically(sim):
    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("inspect_information_source", "Team_Info")]  # no legal repair for this agent
    summary = _commit_for(sim, {"Botanist": {"task": "repair", "action_type": "repair_or_correct_construction", "project_target": "proj_a"}})
    bot._adopt_committed_team_plan(sim, summary, reason="test")
    assert bot.team_plan_state["assignment_state"]["status"] == "ungrounded"
    assert _names(sim, "team_plan_assignment_applied")[-1]["grounded"] is False
    assert _names(sim, "team_plan_assignment_blocked")[-1]["reason"] == "no_matching_legal_action_for_me"
    assert bot.plan_assignment_annotation("repair_or_correct_construction", "proj_a") is None  # no preference without grounding
    assert sim.team_knowledge_manager.artifacts["team_plan:px"].content["assignment_declined_by"]  # reported, not silently dropped
    bot._last_pilot_affordances = [_afford("repair_or_correct_construction", "proj_a", repair_ready=False)]
    assert bot._assignment_grounding({"task": "repair", "action_type": "repair_or_correct_construction"}, "proj_a") == (False, "repair_not_groundable_by_my_knowledge")


def test_I_completed_or_blocked_assignment_triggers_reassessment(sim):
    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("validate_construction", "proj_a")]
    bot._adopt_committed_team_plan(sim, _commit_for(sim, {"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "proj_a"}}), reason="test")
    bot._emit_event(sim, "executable_action_completed", {"action_type": "validate", "decision_action": "validate_construction", "project_id": "proj_a"})
    assert bot.team_plan_state["assignment_state"]["status"] == "completed" and _names(sim, "team_plan_assignment_completed")
    assert _names(sim, "team_plan_reassessed")[-1]["trigger"] == "assignment_completed"
    assert bot.plan_assignment_annotation("validate_construction", "proj_a") is None  # a finished assignment stops pulling
    # blocked: the assigned action stops being legal for this agent
    bot.team_plan_state["assignment_state"] = {"plan_id": "px", "action_types": ["validate_construction"], "target": "proj_a", "status": "applied", "deviations": 0, "absent_streak": 0, "last_deviation_key": None}
    ctx = sim.brain_context_builder.build(sim, bot)
    ctx.action_affordances = [_afford("wait", None)]
    for _ in range(2):
        bot._note_plan_assignment_selection(sim, BrainDecision(selected_action=A.WAIT, target_id=None, confidence=0.5), ctx)
    assert bot.team_plan_state["assignment_state"]["status"] == "blocked" and bot.team_plan_state["revision_due"] == "assignment_blocked"
    assert _names(sim, "team_plan_assignment_blocked")[-1]["reason"] == "assigned_action_no_longer_legal_for_me"
    assert _names(sim, "team_plan_reassessed")[-1]["trigger"] == "assignment_blocked"


def test_I2_review_time_and_expiry_reassess_the_adopted_plan(sim):
    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("validate_construction", "proj_a")]
    sim.time = 10.0
    bot._adopt_committed_team_plan(sim, _commit_for(sim, {"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "proj_a"}}), reason="test")
    bot.team_plan_state["assignment_state"]["status"] = "completed"
    sim.time = bot.team_plan_state["next_review_at"] + 0.1
    bot._team_plan_lifecycle_tick(sim)
    assert _names(sim, "team_plan_reassessed")[-1]["trigger"] == "review_time_reached" and bot.team_plan_state["revision_due"]
    sim.time = bot.team_plan_state["adopted_plan_summary"]["expires_at"] + 1
    bot._team_plan_lifecycle_tick(sim)
    assert _names(sim, "team_plan_reassessed")[-1]["trigger"] == "expiry_reached" and bot.team_plan_state["revision_due"] == "plan_expired"


def test_revision_opportunity_is_open_only_when_a_reassessment_asked_for_it(sim):
    bot = _agent(sim, "Botanist")
    _commit_for(sim, {"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "proj_a"}})
    active = sim.team_knowledge_manager.get_active_team_plan()
    assert bot._team_plan_opportunity(sim, active) is False  # a live committed plan: no new proposal opportunity
    bot.team_plan_state["revision_due"] = "assignment_blocked"
    assert bot._team_plan_opportunity(sim, active) is True


def test_K_low_coordination_gates_engagement_but_not_taskwork():
    random.seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        st = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="lowcoord", seed=0, brain_backend="rule_brain",
                             experimental_settings={"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "constituent_overrides": {"coordination_propensity": 0.1}, "profile_mode": "compact_panel"})
        done = []
        st.logger.register_event_listener(lambda e: done.append(e["payload_data"].get("decision_action")) if e["event_type"] == "executable_action_completed" else None)
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(240):
                st.update(1.0)
        taskwork = {"transport_resources", "start_construction", "continue_construction", "validate_construction", "repair_or_correct_construction"}
        assert any(str(getattr(d, "value", d)) in taskwork for d in done)  # low coordination does not disable taskwork
        assert st.plan_process.counts["committed_plans"] == 0  # and no plan was committed without explicit support
        assert st.team_plan_policy["response_engagement_min"] > 0.1
        st.stop()


def test_K2_engagement_scales_with_coordination_and_support_needs_grounding(sim):
    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("transport_resources", "proj_a")]
    summary = {"plan_id": "z", "status": "proposed", "proposed_by": "Architect", "assignments_by_role": {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "proj_a"}}}
    bot.coordination_propensity = bot.goal_alignment = 0.1
    assert bot._evaluate_team_plan_fit(sim, summary)[0] == ""  # silence, not agreement
    assert bot._should_emit_team_plan_response(sim, summary, "") is False
    bot.coordination_propensity = bot.goal_alignment = 0.9
    assert bot._evaluate_team_plan_fit(sim, summary) == ("assignment_accept", "grounded_in_my_legal_actions")
    bot._last_pilot_affordances = []
    assert bot._evaluate_team_plan_fit(sim, summary)[0] == "assignment_decline"  # engaged but cannot ground it: says so
    own = {**summary, "proposed_by": "Botanist"}
    assert bot._evaluate_team_plan_fit(sim, own)[0] == ""  # the proposer already supports its own proposal


def test_L_assignments_come_only_from_the_proposers_own_legal_view(sim):
    arch = _agent(sim, "Architect")
    arch._last_pilot_affordances = [
        _afford("continue_construction", "visible_a"),
        _afford("validate_construction", "visible_b"),
        _afford("repair_or_correct_construction", "visible_c", repair_ready=False),  # the proposer cannot ground it: not proposed
    ]
    assignments = arch._build_plan_assignments(sim)
    targets = {a["project_target"] for a in assignments.values() if isinstance(a, dict)}
    assert targets == {"visible_a", "visible_b"}
    assert assignments["Architect"]["action_type"] == "validate_construction"  # priority order: validation before construction
    assert all(a["basis"] == "proposer_legal_affordance" for a in assignments.values())
    assert len({(a["action_type"], a["project_target"]) for a in assignments.values()}) == len(assignments)  # distinct work, not one task three times
    arch._last_pilot_affordances = []
    assert arch._build_plan_assignments(sim) == {"Architect": "lead_next_action"}  # nothing legitimately known: no invented assignment


def test_M_adoption_forces_neither_target_nor_intent_and_no_substitution_events(sim):
    bot = _agent(sim, "Botanist")
    bot._last_pilot_affordances = [_afford("transport_resources", "proj_a")]
    before_target, before_category = bot.target, bot.active_intent.get("category")
    bot._adopt_committed_team_plan(sim, _commit_for(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "proj_a"}}), reason="test")
    assert bot.target == before_target and bot.active_intent.get("category") == before_category
    assert not [p for n, p in sim.captured_events if n in {"startup_bootstrap_grounding_forced", "fallback_bootstrap_action_forced"}]
    assert not [p for n, p in sim.captured_events if n == "mecha_strategic_substitution_rejected" and p.get("applied")]


# ------------------------------------------------------------------ measures


def test_process_tracker_counts_distinct_agent_plan_pairs_and_alignment():
    t = PlanProcessTracker()
    ev = lambda time, name, **p: t.on_event({"time": time, "event_type": name, "payload_data": p})  # noqa: E731
    ev(0, "team_plan_proposed", plan_id="p", agent="A")
    ev(4, "team_plan_committed", plan_id="p", agent="B")
    for who in ("A", "B", "A"):
        ev(5, "team_plan_adopted_locally", plan_id="p", agent=who)
    ev(5, "team_plan_assignment_applied", plan_id="p", agent="A", grounded=True)
    ev(5, "team_plan_assignment_applied", plan_id="p", agent="B", grounded=True)
    ev(6, "team_plan_assignment_started", plan_id="p", agent="A")
    ev(8, "team_plan_assignment_completed", plan_id="p", agent="A")
    ev(9, "executable_action_completed", action_type="validate_construction")
    s = t.summary(10.0, 3)
    assert s["mean_time_proposal_to_commitment_s"] == 4.0 and s["commitment_rate"] == 1.0
    assert s["distinct_plan_agent_adoptions"] == 2 and s["local_adoption_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["assignment_start_rate"] == 0.5 and s["assignment_completion_rate"] == 0.5
    assert s["peak_agents_aligned_to_one_plan"] == 2 and s["taskwork_aligned_proportion"] == 1.0
