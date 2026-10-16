"""Coordination hardening: controller-neutral plan context, criterion-based assignment lifecycle, single active plan, epistemic boundary.

Pinned here:
- the store admits at most one proposed-or-committed plan; a new plan enters only beside a terminal incumbent or by explicitly
  naming the incumbent as the plan it supersedes (a competing proposal is refused and logged, never silently replacing it);
- a plan leaves the live states as `completed`/`blocked` once every concretely assigned role has reported;
- an assignment is retired by its task-level success criterion (material condition met / repair obligation resolved / validation
  result observed / physically complete), not by one low-level completion; partial progress is counted as plan-aligned taskwork;
- an assignment the agent cannot ground stays visible as ungrounded with a recognized gap (no knowledge granted) and may ground later;
- the Pilot-facing `team_plan_context/1` contains only what the agent acquired by reading/receiving/authoring a plan.
"""

import tempfile

import pytest

from modules.simulation import SimulationState
from modules.team_knowledge import TeamKnowledgeManager
from modules.team_plan_lifecycle import PlanProcessTracker


def _plan(m, plan_id="p1", proposer="Architect", t=0.0, assignments=None, **kw):
    return m.propose_team_plan(
        plan_id=plan_id, proposed_by=proposer, sim_time=t, goal_ids=["g"], goal_summary="g", project_targets=["proj_a"],
        assignments_by_role=assignments if assignments is not None else {"Architect": {"task": "build", "action_type": "continue_construction", "project_target": "proj_a"}},
        supporters=kw.pop("supporters", [proposer]), expires_at=kw.pop("expires_at", 100.0), review_at=t + 20.0, **kw,
    )


def _status(m, plan_id="p1"):
    return m.artifacts[f"team_plan:{plan_id}"].content["status"]


def _commit(m, plan_id="p1", t=1.0, supporter="Botanist"):
    m.record_team_plan_response(plan_id=plan_id, responder=supporter, response_type="agree", sim_time=t, role=supporter, reason="test")
    assert _status(m, plan_id) == "committed"


# ------------------------------------------------------------------ single active plan (store)


def test_a_competing_proposal_is_refused_not_silently_replacing_the_incumbent():
    m = TeamKnowledgeManager()
    _plan(m, "p1")
    assert _plan(m, "p2", proposer="Botanist", t=2.0) is None
    assert "team_plan:p2" not in m.artifacts and _status(m, "p1") == "proposed"
    refused = [e for e in m.drain_lifecycle_events() if e["event_type"] == "team_plan_proposal_refused"]
    assert refused and refused[0]["payload"]["active_plan_id"] == "p1"
    _commit(m, "p1")
    assert _plan(m, "p3", proposer="Engineer", t=3.0) is None  # also while the incumbent is committed
    assert _status(m, "p1") == "committed"


def test_explicit_supersession_is_the_only_way_to_enter_beside_a_live_plan():
    m = TeamKnowledgeManager()
    _plan(m, "p1")
    _commit(m, "p1")
    assert _plan(m, "p2", proposer="Botanist", t=2.0, supersedes_plan_id="other") is None  # naming some other plan does not count
    assert _plan(m, "p2", proposer="Botanist", t=2.0, supersedes_plan_id="p1") is not None
    assert _status(m, "p1") == "superseded" and _status(m, "p2") == "proposed"
    assert sum(p["status"] in {"proposed", "committed"} for p in m.list_team_plans()) == 1
    assert "explicit_supersession_by_proposal" in [e["payload"].get("reason") for e in m.drain_lifecycle_events() if e["event_type"] == "team_plan_superseded"]


def test_a_terminal_incumbent_does_not_block_a_new_plan():
    m = TeamKnowledgeManager()
    _plan(m, "p1", expires_at=10.0)
    m.expire_due_plans(50.0)
    assert _status(m, "p1") == "expired"
    assert _plan(m, "p2", proposer="Botanist", t=51.0) is not None


def test_policy_can_disable_the_single_active_plan_rule():
    m = TeamKnowledgeManager()
    m.commit_policy["single_active_plan"] = False
    _plan(m, "p1")
    assert _plan(m, "p2", proposer="Botanist", t=2.0) is not None


def test_plan_completes_or_blocks_only_when_every_assigned_role_has_reported():
    two = {"Architect": {"task": "validate", "action_type": "validate_construction", "project_target": "a"}, "Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "b"}}
    m = TeamKnowledgeManager()
    _plan(m, "p1", assignments=two)
    _commit(m, "p1")
    m.record_assignment_outcome(plan_id="p1", role="Architect", outcome="completed", sim_time=5.0)
    assert _status(m, "p1") == "committed"  # the other role has not reported
    m.record_assignment_outcome(plan_id="p1", role="Botanist", outcome="blocked", sim_time=6.0)
    assert _status(m, "p1") == "completed"
    assert _plan(m, "p2", proposer="Engineer", t=7.0) is not None  # a completed plan frees the slot without any supersession
    m2 = TeamKnowledgeManager()
    _plan(m2, "q1", assignments=two)
    _commit(m2, "q1")
    for r in ("Architect", "Botanist"):
        m2.record_assignment_outcome(plan_id="q1", role=r, outcome="blocked", sim_time=5.0)
    assert _status(m2, "q1") == "blocked"
    assert {"team_plan_blocked"} <= {e["event_type"] for e in m2.drain_lifecycle_events()}


# ------------------------------------------------------------------ agent side


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=[], project_root=tmp, flash_mode=True, experiment_name="planhard", seed=0)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        state.captured_events = events
        try:
            yield state
        finally:
            state.stop()


def _agent(sim, name):
    return next(x for x in sim.agents if x.name == name)


def _names(sim, event):
    return [p for n, p in sim.captured_events if n == event]


def _afford(action, target, **kw):
    return {"action_type": action, "target_id": target, "reachable": True, "utility": 0.9, **kw}


def _project(sim, structure="house"):
    pid, _ = sim.environment.construction.create_project("site_b", structure)
    assert pid
    return pid, sim.environment.construction.projects[pid]


def _adopt(sim, bot, assignment, plan_id="px"):
    m = sim.team_knowledge_manager
    _plan(m, plan_id, proposer="Architect", t=sim.time, assignments={bot.role: assignment}, expires_at=sim.time + 200.0)
    _commit(m, plan_id, t=sim.time)
    summary = m._team_plan_summary(dict(m.artifacts[f"team_plan:{plan_id}"].content))
    bot._note_known_plan(sim, summary, via="consulted")
    bot._adopt_committed_team_plan(sim, summary, reason="test")
    return summary


def _done(sim, bot, action, pid):
    bot._emit_event(sim, "executable_action_completed", {"action_type": "construct" if action != "transport_resources" else "transport_resources", "decision_action": action, "project_id": pid})


def test_transport_assignment_stays_active_until_the_material_condition_is_met(sim):
    bot = _agent(sim, "Botanist")
    pid, proj = _project(sim)
    required = int(proj["required_resources"]["bricks"])
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    _adopt(sim, bot, {"task": "transport", "action_type": "transport_resources", "project_target": pid})
    state = bot.team_plan_state["assignment_state"]
    assert state["status"] == "applied"
    proj["delivered_resources"]["bricks"] = max(1, required // 2)
    _done(sim, bot, "transport_resources", pid)
    assert state["status"] in {"applied", "started"} and state["aligned_actions"] == 1  # one delivery is partial progress, not completion
    assert bot.plan_assignment_annotation("transport_resources", pid) is not None
    proj["delivered_resources"]["bricks"] = required
    _done(sim, bot, "transport_resources", pid)
    assert state["status"] == "completed" and state["aligned_actions"] == 2
    completed = _names(sim, "team_plan_assignment_completed")[-1]
    assert completed["contributed"] is True and completed["criterion"]["status"] == "met"
    assert len(_names(sim, "team_plan_aligned_action_completed")) == 2
    assert sim.team_knowledge_manager.artifacts["team_plan:px"].content["assignment_outcomes"]["Botanist"]["outcome"] == "completed"


def test_movement_completion_never_counts_as_assignment_progress(sim):
    bot = _agent(sim, "Botanist")
    pid, _proj = _project(sim)
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    _adopt(sim, bot, {"task": "transport", "action_type": "transport_resources", "project_target": pid})
    bot._emit_event(sim, "executable_action_completed", {"action_type": "move_to", "decision_action": "transport_resources", "project_id": pid})
    assert bot.team_plan_state["assignment_state"]["aligned_actions"] == 0 and bot.team_plan_state["assignment_state"]["status"] == "applied"


def test_repair_assignment_persists_until_the_repair_obligation_resolves(sim):
    bot = _agent(sim, "Engineer")
    pid, proj = _project(sim)
    proj.update(validation_attempted=True, closure_status="failed", validated_complete=False)
    bot._last_pilot_affordances = [_afford("repair_or_correct_construction", pid, repair_ready=True)]
    _adopt(sim, bot, {"task": "repair", "action_type": "repair_or_correct_construction", "project_target": pid})
    state = bot.team_plan_state["assignment_state"]
    _done(sim, bot, "repair_or_correct_construction", pid)
    assert state["status"] in {"applied", "started"}  # a repair action happened but the observed failure still stands
    proj["closure_status"] = "pending"  # resolved by the world, as an applied repair would
    bot._tick_assignment(sim)
    assert state["status"] == "completed"


def test_validate_assignment_completes_when_the_validation_obligation_resolves(sim):
    bot = _agent(sim, "Architect")
    pid, proj = _project(sim)
    proj["structurally_complete"] = True
    bot._last_pilot_affordances = [_afford("validate_construction", pid)]
    _adopt(sim, bot, {"task": "validate", "action_type": "validate_construction", "project_target": pid})
    state = bot.team_plan_state["assignment_state"]
    bot._tick_assignment(sim)
    assert state["status"] in {"applied", "started"}
    proj.update(validated_complete=True, status="complete")  # someone else's validation also resolves it
    bot._tick_assignment(sim)
    assert state["status"] == "completed" and _names(sim, "team_plan_assignment_completed")[-1]["contributed"] is False


def test_a_criterion_already_met_at_adoption_is_invalidated_not_applied(sim):
    bot = _agent(sim, "Botanist")
    pid, proj = _project(sim)
    proj["delivered_resources"]["bricks"] = proj["required_resources"]["bricks"]
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    _adopt(sim, bot, {"task": "transport", "action_type": "transport_resources", "project_target": pid})
    assert bot.team_plan_state["assignment_state"]["status"] == "invalidated"
    assert bot.plan_assignment_annotation("transport_resources", pid) is None
    assert _names(sim, "team_plan_assignment_invalidated")


def test_ungrounded_assignment_is_visible_as_a_gap_and_can_ground_later_without_granting_knowledge(sim):
    bot = _agent(sim, "Engineer")
    pid, proj = _project(sim)
    proj.update(validation_attempted=True, closure_status="failed", validated_complete=False)
    blockers = ["missing_expected_rule:R_TEST"]
    bot._last_pilot_affordances = [_afford("repair_or_correct_construction", pid, repair_ready=False, repair_blockers=blockers)]
    held_before = {getattr(r, "id", r) for r in bot.mental_model["knowledge"].rules}
    _adopt(sim, bot, {"task": "repair", "action_type": "repair_or_correct_construction", "project_target": pid})
    state = bot.team_plan_state["assignment_state"]
    assert state["status"] == "ungrounded" and state["grounding_reason"] == "repair_not_groundable_by_my_knowledge"
    assert any("missing_expected_rule" in str(g) for g in bot.known_gaps)  # recognized gap, not granted knowledge
    ctx = sim.brain_context_builder.build(sim, bot).team_state["team_plan_context"]
    assert ctx["my_assignment"]["grounded"] is False and ctx["my_assignment"]["status"] == "ungrounded"
    assert ctx["my_assignment"]["recognized_gap_tokens"] and ctx["my_assignment"]["success_criterion"]["status"] == "open"
    assert bot.plan_assignment_annotation("repair_or_correct_construction", pid) is None  # no pull toward an action it cannot ground
    assert {getattr(r, "id", r) for r in bot.mental_model["knowledge"].rules} == held_before
    bot._last_pilot_affordances = [_afford("repair_or_correct_construction", pid, repair_ready=True, repair_blockers=[])]
    bot._tick_assignment(sim)
    assert state["status"] == "applied" and _names(sim, "team_plan_assignment_grounded_later")


# ------------------------------------------------------------------ controller-neutral context and epistemic boundary


def test_unread_plan_is_only_an_existence_flag_everywhere_in_the_pilot_context(sim):
    bot = _agent(sim, "Botanist")
    m = sim.team_knowledge_manager
    _plan(m, "px", assignments={"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "SECRET_TARGET"}}, expires_at=200.0)
    _commit(m, "px")
    ctx = sim.brain_context_builder.build(sim, bot)
    tpc = ctx.team_state["team_plan_context"]
    assert tpc["schema"] == "team_plan_context/1" and tpc["known_plan"] is None and tpc["my_assignment"] is None and tpc["unread_live_plan_artifact"] is True
    blob = repr(ctx.team_state) + repr(ctx.individual_cognitive_state) + repr(ctx.world_snapshot) + repr(ctx.action_affordances)
    assert "SECRET_TARGET" not in blob
    assert not [u for u in ctx.team_state["recent_shared_updates"] if str(u.get("event", "")).startswith("team_plan_")]
    assert not ctx.team_state["team_shared_knowledge"]["team_plan_summaries"] and not ctx.team_state["team_shared_knowledge"]["active_team_plan"]
    assert bot.known_plan_summary_for_sharing(sim) is None  # an agent cannot pass on a plan it never read


def test_context_after_reading_exposes_the_plan_as_read_and_the_assignment_markers(sim):
    bot = _agent(sim, "Botanist")
    pid, proj = _project(sim)
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    _adopt(sim, bot, {"task": "transport", "action_type": "transport_resources", "project_target": pid})
    ctx = sim.brain_context_builder.build(sim, bot)
    tpc = ctx.team_state["team_plan_context"]
    assert tpc["adopted"] is True and tpc["adopted_plan_id"] == "px"
    assert tpc["known_plan"]["status_when_read"] == "committed" and tpc["known_plan"]["read_via"] == "consulted"
    mine = tpc["my_assignment"]
    assert mine["grounded"] is True and mine["target"] == pid and mine["action_types"] == ["transport_resources"]
    assert mine["success_criterion"]["kind"] == "transport" and mine["progress"]["aligned_actions"] == 0
    assert tpc["review"]["next_review_at"] is not None and tpc["review"]["expired"] is False
    marked = [a for a in ctx.action_affordances if a.get("plan_assignment_plan_id")]
    assert marked and all(a["plan_assignment"]["plan_id"] == "px" for a in marked)
    assert {(a["action_type"], a["target_id"]) for a in marked} == {(x["action_type"], x["target_id"]) for x in mine["matching_affordances"]}


def test_a_non_baseline_pilot_receives_the_same_plan_facts_from_the_canonical_request(sim):
    """The facts reach the provider request contract (task_context), so a non-RuleBrain Pilot needs nothing from the store."""
    from modules.brain_provider import _request_from_context_packet

    bot = _agent(sim, "Botanist")
    pid, _proj = _project(sim)
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    _adopt(sim, bot, {"task": "transport", "action_type": "transport_resources", "project_target": pid})
    request = _request_from_context_packet(sim.brain_context_builder.build(sim, bot))
    tpc = request.task_context["team_plan_context"]
    assert tpc["my_assignment"]["target"] == pid and tpc["known_plan"]["plan_id"] == "px"
    assert request.task_context["active_team_plan"]["plan_id"] == "px"  # derived from the agent's acquired copy only


def test_message_sharing_carries_only_the_senders_acquired_copy(sim):
    a, b = _agent(sim, "Architect"), _agent(sim, "Botanist")
    m = sim.team_knowledge_manager
    _plan(m, "px", expires_at=200.0)
    _commit(m, "px")
    assert a.known_plan_summary_for_sharing(sim) is None
    a._note_known_plan(sim, m._team_plan_summary(dict(m.artifacts["team_plan:px"].content)), via="authored")
    shared = a.known_plan_summary_for_sharing(sim)
    assert shared and shared["plan_id"] == "px" and shared["status"] == "committed"
    assert b.known_plan_summary_for_sharing(sim) is None  # nothing travelled yet


# ------------------------------------------------------------------ measures


def test_tracker_counts_partial_progress_plan_exits_and_committed_seconds():
    t = PlanProcessTracker()
    ev = lambda time, name, **p: t.on_event({"time": time, "event_type": name, "payload_data": p})  # noqa: E731
    ev(0, "team_plan_proposed", plan_id="p", agent="A")
    ev(2, "team_plan_committed", plan_id="p", agent="B")
    ev(3, "team_plan_assignment_applied", plan_id="p", agent="A", grounded=True)
    for tm in (4, 5, 6):
        ev(tm, "team_plan_aligned_action_completed", plan_id="p", agent="A")
        ev(tm, "executable_action_completed", action_type="transport_resources", decision_action="transport_resources")
    ev(7, "executable_action_completed", action_type="transport_resources", decision_action="transport_resources")  # taskwork that no assignment asked for
    ev(8, "team_plan_assignment_completed", plan_id="p", agent="A", aligned_actions=3)
    ev(12, "team_plan_completed", plan_id="p")
    ev(13, "team_plan_proposal_refused", plan_id="q")
    s = t.summary(20.0, 3)
    assert s["taskwork_aligned_proportion"] == 0.75 and s["assignments_completed_with_aligned_work"] == 1
    assert s["committed_plan_seconds"] == 10.0 and s["plan_exits"]["completed"] == 1 and s["proposals_refused_active_plan"] == 1


# ------------------------------------------------------------------ review / churn / event semantics


def test_review_time_never_replaces_a_plan_by_itself(sim):
    bot = _agent(sim, "Engineer")  # holds no assignment in the plan
    m = sim.team_knowledge_manager
    _plan(m, "px", proposer="Architect", t=sim.time, assignments={"Botanist": {"task": "validate", "action_type": "validate_construction", "project_target": "a"}}, expires_at=sim.time + 200.0)
    _commit(m, "px", t=sim.time)
    summary = m._team_plan_summary(dict(m.artifacts["team_plan:px"].content))
    bot._note_known_plan(sim, summary, via="consulted")
    bot._adopt_committed_team_plan(sim, summary, reason="test")
    sim.time = bot.team_plan_state["next_review_at"] + 0.1
    bot._team_plan_lifecycle_tick(sim)
    review = _names(sim, "team_plan_reassessed")[-1]
    assert review["trigger"] == "review_time_reached" and review["outcome"] == "continue" and not bot.team_plan_state["revision_due"]


def test_supersession_and_expiry_of_an_assignment_have_their_own_events_not_blocked(sim):
    bot = _agent(sim, "Botanist")
    pid, _proj = _project(sim)
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    _adopt(sim, bot, {"task": "transport", "action_type": "transport_resources", "project_target": pid})
    sim.time = bot.team_plan_state["adopted_plan_summary"]["expires_at"] + 1
    bot._team_plan_lifecycle_tick(sim)
    assert _names(sim, "team_plan_assignment_expired") and not _names(sim, "team_plan_assignment_blocked")
    assert bot.team_plan_state["revision_due"] == "plan_expired"


def test_an_agent_that_authored_a_plan_does_not_see_its_own_version_as_unread(sim):
    a = _agent(sim, "Architect")
    m = sim.team_knowledge_manager
    _plan(m, "px", expires_at=200.0)
    assert a._unread_live_plan_artifact(sim) is True
    a._note_known_plan(sim, m._team_plan_summary(dict(m.artifacts["team_plan:px"].content)), via="authored")
    assert a._unread_live_plan_artifact(sim) is False
    _commit(m, "px")  # a newer version exists that the author has not read
    assert a._unread_live_plan_artifact(sim) is True


# ------------------------------------------------------------------ criterion precision and knowledge externalisation under the single-plan rule


def test_an_earlier_failed_validation_does_not_resolve_a_new_validate_assignment(sim):
    bot = _agent(sim, "Architect")
    pid, proj = _project(sim)
    proj.update(structurally_complete=True, validation_attempted=True, closure_status="failed", validated_complete=False, last_validation_attempt_time=0.0)
    bot._last_pilot_affordances = [_afford("validate_construction", pid)]
    sim.time = 5.0
    _adopt(sim, bot, {"task": "validate", "action_type": "validate_construction", "project_target": pid})
    state = bot.team_plan_state["assignment_state"]
    assert state["status"] == "applied"  # the old failed attempt predates the assignment: the obligation is still open
    bot._tick_assignment(sim)
    assert state["status"] == "applied"
    proj["last_validation_attempt_time"] = 9.0  # a validation attempted after the assignment was taken up resolves it, pass or fail
    bot._tick_assignment(sim)
    assert state["status"] == "completed"


def test_a_refused_competing_plan_still_externalises_the_authors_knowledge_without_touching_the_plan(sim):
    author, other = _agent(sim, "Botanist"), _agent(sim, "Architect")
    for kind in ("data", "information"):
        for item in list(author.mental_model[kind]):
            author.mental_model[kind].discard(item)
    element = sim.task_model.dik_elements["I_BOT_GREENHOUSE_DEPENDENCIES"]
    author.mental_model["information"].add(author._create_dik_object_from_element(element, source_id="test"))
    m = sim.team_knowledge_manager
    m.artifacts.clear()
    _plan(m, "px", proposer="Architect", t=sim.time, expires_at=sim.time + 200.0)
    before = dict(m.artifacts["team_plan:px"].content)
    added = author._contribute_to_live_plan_artifact(sim)
    art = m.artifacts["team_plan:px"]
    assert added == ["I_BOT_GREENHOUSE_DEPENDENCIES"] and [i.id for i in art.carried_information] == added
    assert dict(art.content) == before and set(m.artifacts) == {"team_plan:px"}  # the plan itself is unchanged and no second plan exists
    assert author._contribute_to_live_plan_artifact(sim) == []  # nothing new the second time
    assert "carries_unheld_dik" in other._artifact_consult_reasons("team_plan:px", art, set(), None, None)[0]  # a teammate can now acquire it by reading


# ------------------------------------------------------------------ adoption through either legitimate acquisition channel


def _committed_summary(sim, assignments):
    m = sim.team_knowledge_manager
    m.artifacts.clear()
    _plan(m, "px", proposer="Architect", t=sim.time, assignments=assignments, expires_at=sim.time + 200.0)
    _commit(m, "px", t=sim.time)
    return m._team_plan_summary(dict(m.artifacts["team_plan:px"].content))


def _tick_uptake(sim, bot):
    summary, source = bot._plan_uptake_candidate(sim)
    if summary is not None:
        bot._adopt_committed_team_plan(sim, summary, reason="active_team_plan_requires_uptake", source=source)
    return source


def test_artifact_acquisition_permits_adoption_and_is_recorded_as_artifact(sim):
    bot = _agent(sim, "Botanist")
    pid, _p = _project(sim)
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    summary = _committed_summary(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": pid}})
    assert _tick_uptake(sim, bot) is None  # nothing read, nothing received
    bot._note_known_plan(sim, summary, via="consulted")
    art = sim.team_knowledge_manager.artifacts["team_plan:px"]
    bot.artifact_consult_sig_memory[bot._artifact_version_signature(art)] = {}
    assert _tick_uptake(sim, bot) == "artifact"
    assert bot.team_plan_state["assignment_state"]["status"] == "applied"
    assert _names(sim, "team_plan_adopted_locally")[-1]["acquisition_source"] == "artifact"


def test_full_message_acquisition_permits_adoption_without_an_artifact_read(sim):
    bot = _agent(sim, "Botanist")
    pid, _p = _project(sim)
    bot._last_pilot_affordances = [_afford("transport_resources", pid)]
    summary = _committed_summary(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": pid}})
    art = sim.team_knowledge_manager.artifacts["team_plan:px"]
    assert bot._artifact_version_signature(art) not in bot.artifact_consult_sig_memory  # the artifact was never consulted
    bot._note_known_plan(sim, summary, via="message")
    assert _tick_uptake(sim, bot) == "message"
    assert bot.team_plan_state["assignment_state"]["status"] == "applied" and bot.team_plan_state["known_plans"]["px"]["source"] == "message"
    assert _names(sim, "team_plan_adopted_locally")[-1]["acquisition_source"] == "message"
    ctx = sim.brain_context_builder.build(sim, bot).team_state["team_plan_context"]
    assert ctx["adopted"] and ctx["known_plan"]["read_via"] == "message" and ctx["my_assignment"]["grounded"]
    assert _tick_uptake(sim, bot) is None  # taken up once; a re-read does not re-adopt the same version


def test_partial_or_summary_only_message_does_not_permit_adoption(sim):
    bot = _agent(sim, "Botanist")
    _committed_summary(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "x"}})
    bot._note_known_plan(sim, {"plan_id": "px", "status": "committed", "goal_summary": "do things"}, via="message")  # no assignments
    assert _tick_uptake(sim, bot) is None
    bot._note_known_plan(sim, {"plan_id": "px", "status": "proposed", "goal_summary": "g", "assignments_by_role": {"Botanist": {"task": "t", "action_type": "transport_resources"}}}, via="message")
    assert _tick_uptake(sim, bot) is None  # not committed as far as the sender knew
    assert not bot.team_plan_state.get("assignment_state")


def test_no_plan_knowledge_appears_without_legitimate_acquisition(sim):
    bot = _agent(sim, "Botanist")
    _commit_summary = _committed_summary(sim, {"Botanist": {"task": "transport", "action_type": "transport_resources", "project_target": "SECRET"}})
    assert _commit_summary["status"] == "committed"
    assert _tick_uptake(sim, bot) is None
    ctx = sim.brain_context_builder.build(sim, bot)
    assert "SECRET" not in repr(ctx.team_state) + repr(ctx.action_affordances)
    assert ctx.team_state["team_plan_context"]["known_plan"] is None and ctx.team_state["team_plan_context"]["adopted"] is False


def test_an_older_relayed_copy_never_replaces_a_newer_one(sim):
    bot = _agent(sim, "Botanist")
    newer = {"plan_id": "px", "status": "committed", "plan_event_time": 50.0, "assignments_by_role": {"Botanist": {"task": "t", "action_type": "transport_resources"}}}
    bot._note_known_plan(sim, newer, via="consulted")
    bot._note_known_plan(sim, {**newer, "status": "proposed", "plan_event_time": 10.0}, via="message")
    assert bot.team_plan_state["known_plans"]["px"]["summary"]["status"] == "committed"
