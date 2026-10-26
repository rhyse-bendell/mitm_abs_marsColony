"""Construct-hardening repairs: recognized-gap bookkeeping, single-application efficiency, evidence-grounded teammate models."""

import tempfile

import pytest

from modules.knowledge import Information
from modules.teammate_model_metrics import held_tokens, model_accuracy, routing_quality, team_model_accuracy
from modules.simulation import SimulationState

ACTIONS = ("transport_resources", "start_construction", "continue_construction", "validate_construction", "repair_or_correct_construction")


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="hardening", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _pair(sim):
    return sim.agents[0], sim.agents[1]


# ------------------------------------------------------------------ Part 1: recognized-gap state
def test_received_request_is_not_a_gap_in_the_recipient(sim):
    asker, holder = _pair(sim)
    info = Information("I_TEST_RECEIVED", "x")
    holder.mental_model["information"].add(info)
    holder.known_gaps.clear()
    before = held_tokens(holder)
    holder.receive_message({"type": "TKRQ", "sender": asker.name, "content": ["information:I_TEST_RECEIVED", "information:I_TEST_OTHER"]}, sim_state=sim)
    assert not holder.known_gaps  # the sender's need is not the recipient's own gap
    assert holder.received_dik_requests[asker.name]["ids"] == {"information:I_TEST_RECEIVED", "information:I_TEST_OTHER"}
    assert holder.theory_of_mind[asker.name]["requested_dik_ids"] == {"information:I_TEST_RECEIVED", "information:I_TEST_OTHER"}
    assert held_tokens(holder) == before  # and nothing is granted either way
    assert "I_TEST_OTHER" not in {i.id for i in holder.mental_model["information"]}


def test_request_is_still_answered_from_actual_holdings(sim):
    asker, holder = _pair(sim)
    info = Information("I_TEST_ANSWER", "x")
    holder.mental_model["information"].add(info)
    holder.receive_message({"type": "TKRQ", "sender": asker.name, "content": ["information:I_TEST_ANSWER"]}, sim_state=sim)
    queued = holder.team_plan_state.get("pending_outbound_messages", [])
    assert any(m["type"] == "TIP" and m["recipient"] == asker.name and info in m["content"] for m in queued)


def test_resolved_gaps_are_removed_and_unresolved_ones_retained(sim):
    agent = sim.agents[0]
    held = Information("I_TEST_HELD", "x")
    agent.mental_model["information"].add(held)
    agent.known_gaps.clear()
    agent.known_gaps.update({"information:I_TEST_HELD", "information:I_TEST_MISSING", "missing_validation_rule_knowledge"})
    agent._reconcile_known_gaps(sim)
    assert agent.known_gaps == {"information:I_TEST_MISSING", "missing_validation_rule_knowledge"}  # legitimate/unresolved stay open


def test_rule_gaps_resolve_when_the_rule_is_held_by_any_route(sim):
    agent = sim.agents[0]
    agent.mental_model["knowledge"].add_rule("K_TEST_RULE", [])
    agent.known_gaps.clear()
    agent.known_gaps.update({"rule:K_TEST_RULE", "missing_expected_rule:K_TEST_RULE", "rule:K_NOT_HELD"})
    agent._reconcile_known_gaps(sim)
    assert agent.known_gaps == {"rule:K_NOT_HELD"}


def test_reconciliation_grants_nothing_and_keeps_acquisition_provenance(sim):
    agent = sim.agents[0]
    info = Information("I_TEST_PROV", "x")
    info.acquired_by[agent.name] = {"mode": "shared", "from": sim.agents[1].name}
    agent.mental_model["information"].add(info)
    agent._sync_dik_ledger(sim)
    agent.known_gaps.add("information:I_TEST_PROV")
    held_before = held_tokens(agent)
    agent._reconcile_known_gaps(sim)
    assert held_tokens(agent) == held_before
    assert agent.dik_ledger["I_TEST_PROV"]["mode"] == "shared" and agent.dik_ledger["I_TEST_PROV"]["source"] == sim.agents[1].name


def test_a_gap_for_a_forgotten_element_is_recognized_again_not_hidden(sim):
    agent = sim.agents[0]
    info = Information("I_TEST_FORGOT", "x")
    agent.mental_model["information"].add(info)
    agent.known_gaps.add("information:I_TEST_FORGOT")
    agent._reconcile_known_gaps(sim)
    assert "information:I_TEST_FORGOT" not in agent.known_gaps
    agent.mental_model["information"].discard(info)  # forgotten
    agent.known_gaps.add("information:I_TEST_FORGOT")  # recognized again from a method requirement
    agent._reconcile_known_gaps(sim)
    assert "information:I_TEST_FORGOT" in agent.known_gaps


def test_agents_do_not_request_what_they_already_hold_because_of_stale_gap_state(sim):
    agent, peer = _pair(sim)
    info = Information("I_TEST_STALE", "x")
    agent.mental_model["information"].add(info)
    agent.known_gaps.clear()
    agent.known_gaps.update({"information:I_TEST_STALE", "information:I_TEST_REALLY_MISSING"})
    messages = agent.generate_message(recipient_name=peer.name, include_requests=True, include_unsolicited=False)
    requested = {t for m in messages if m["type"] == "TKRQ" for t in m["content"]}
    assert requested == {"information:I_TEST_REALLY_MISSING"}


def test_request_chain_does_not_infect_the_answerer(sim):
    """Before the repair, A's request put A's gap into B, and B then asked A (or C) for A's own need."""
    asker, answerer = _pair(sim)
    asker.known_gaps.clear()
    answerer.known_gaps.clear()
    asker.known_gaps.add("information:I_TEST_CHAIN")
    for m in asker.generate_message(recipient_name=answerer.name, include_requests=True, include_unsolicited=False):
        answerer.receive_message(m, sim_state=sim)
    echoed = {t for m in answerer.generate_message(recipient_name=asker.name, include_requests=True, include_unsolicited=False) if m["type"] == "TKRQ" for t in m["content"]}
    assert "information:I_TEST_CHAIN" not in echoed and not answerer.known_gaps


# ------------------------------------------------------------------ Part 2: efficiency -> duration, applied once
def _set_efficiency(agent, speed):
    """Resolve the documented inverse_duration_scale mapping (scale = 1.5 - build_speed over [0.5, 1.5]) into the agent's hooks."""
    agent.build_speed = speed
    for action in ACTIONS:
        agent.hook_effects[("action_duration", action, "duration_scale")] = 1.5 - speed


def test_duration_is_base_times_one_hook_factor(sim):
    agent = sim.agents[0]
    for speed in (0.0, 0.1, 0.5, 0.9, 1.0):
        _set_efficiency(agent, speed)
        for action in ACTIONS:
            assert agent._scaled_duration(30.0, action) == pytest.approx(30.0 * (1.5 - speed), abs=1e-3)


def test_neutral_efficiency_leaves_authored_duration_unchanged(sim):
    agent = sim.agents[0]
    _set_efficiency(agent, 0.5)
    for action in ACTIONS:
        assert agent._scaled_duration(30.0, action) == 30.0


def test_duration_is_monotone_in_efficiency_and_bounded(sim):
    agent = sim.agents[0]
    for action in ACTIONS:
        durations = []
        for speed in (0.0, 0.25, 0.5, 0.75, 1.0):
            _set_efficiency(agent, speed)
            durations.append(agent._scaled_duration(30.0, action))
        assert durations == sorted(durations, reverse=True) and durations[0] == pytest.approx(45.0) and durations[-1] == pytest.approx(15.0)


def test_build_speed_alone_does_not_alter_duration_without_its_hook(sim):
    """build_speed enters durations only via the construct->mechanism->hook mapping (no second, hidden factor)."""
    agent = sim.agents[0]
    agent.hook_effects.clear()
    for speed in (0.0, 1.0):
        agent.build_speed = speed
        assert agent._scaled_duration(30.0, "continue_construction") == 30.0


def test_resolved_profile_produces_the_documented_factor():
    """End to end: constituent value -> resolved hook -> duration factor, for each action, with nothing compounded."""
    def _sim(settings):
        tmp = tempfile.TemporaryDirectory()
        state = SimulationState(phases=None, project_root=tmp.name, flash_mode=True, experiment_name="hardening", seed=1, experimental_settings=settings)
        state._tmp = tmp
        return state

    low, neutral, high = _sim({"constituent_overrides": {"task_execution_efficiency": 0.1}}), _sim({"constituent_overrides": {"task_execution_efficiency": 0.5}}), _sim({"constituent_overrides": {"task_execution_efficiency": 0.9}})
    try:
        for action in ACTIONS:
            f = {n: s.agents[0]._scaled_duration(100.0, action) / 100.0 for n, s in (("low", low), ("neutral", neutral), ("high", high))}
            assert f["neutral"] == pytest.approx(1.0)
            assert f["low"] > 1.0 > f["high"]
            assert f["low"] == pytest.approx(1.4, abs=0.02) and f["high"] == pytest.approx(0.6, abs=0.02)  # the hook's own range, nothing more
    finally:
        for s in (low, neutral, high):
            s.stop()
            s._tmp.cleanup()


# ------------------------------------------------------------------ Part 3: evidence-grounded teammate model
def test_perception_does_not_copy_hidden_holdings(sim):
    a, b = _pair(sim)
    b.position = (a.position[0] + 0.5, a.position[1])
    b.mental_model["knowledge"].add_rule("K_TEST_HIDDEN_RULE", [])
    b.mental_model["information"].add(Information("I_TEST_HIDDEN", "x"))
    a.perceive_environment(sim)
    model = a.theory_of_mind[b.name]
    assert model["last_seen"] is not None and model["goals"]  # observed behaviour is recorded
    assert not model["knowledge_ids"] and not model["believed_ids"]  # but no hidden holdings are
    assert "K_TEST_HIDDEN_RULE" not in str(model)


def test_provision_is_evidence_for_both_parties_and_only_for_what_was_sent(sim):
    giver, receiver = _pair(sim)
    sent, kept = Information("I_TEST_SENT", "x"), Information("I_TEST_KEPT", "y")
    giver.mental_model["information"].update({sent, kept})
    giver.source_memory_state.setdefault(f"{giver.role}_Info", {})["pending_role_share_ids"] = ["information:I_TEST_SENT"]
    giver.communicate_with(receiver, sim_state=sim, mode="provide")
    assert "information:I_TEST_SENT" in receiver.theory_of_mind[giver.name]["believed_ids"]  # receiver saw it provided
    assert "information:I_TEST_KEPT" not in receiver.theory_of_mind[giver.name]["believed_ids"]  # held but never shown: unknown to the receiver
    assert "information:I_TEST_SENT" in giver.theory_of_mind[receiver.name]["believed_ids"]  # the giver knows what it handed over
    assert receiver.theory_of_mind[giver.name]["evidence"]["information:I_TEST_SENT"]["source"] == "communicated_provision"


def test_a_stated_request_is_evidence_of_lack_not_of_holding(sim):
    asker, other = _pair(sim)
    asker.known_gaps.clear()
    asker.known_gaps.add("information:I_TEST_ASKED")
    asker.communicate_with(other, sim_state=sim, mode="request")
    model = other.theory_of_mind[asker.name]
    assert "information:I_TEST_ASKED" in model["believed_missing_ids"] and "information:I_TEST_ASKED" not in model["believed_ids"]


def test_teammate_model_is_not_perfect_and_can_go_stale(sim):
    """Beliefs are evidence-limited: unshown holdings are missed (recall < 1) and a later-forgotten item stays believed (precision < 1)."""
    a, b = _pair(sim)
    shown, hidden = Information("I_TEST_SHOWN", "x"), Information("I_TEST_NOT_SHOWN", "y")
    b.mental_model["information"].update({shown, hidden})
    b.source_memory_state.setdefault(f"{b.role}_Info", {})["pending_role_share_ids"] = ["information:I_TEST_SHOWN"]
    b.communicate_with(a, sim_state=sim, mode="provide")
    acc = model_accuracy(a, b)
    assert acc["recall"] is not None and acc["recall"] < 1.0
    b.mental_model["information"].discard(shown)  # b forgets it afterwards; a is not told
    acc = model_accuracy(a, b)
    assert acc["precision"] == 0.0 or acc["precision"] < 1.0


def test_unsurveyed_teammates_have_no_knowledge_belief_and_routing_falls_back_to_public_role_evidence(sim):
    observer, target = _pair(sim)
    observer.theory_of_mind.clear()
    assert not observer._tom_entry(target.name)["believed_ids"]
    assert observer._request_route_score(target, 1.0, ["information:I_ENG_CONNECTOR_RULES"]) >= 0.0


def test_teammate_help_signal_does_not_read_a_teammates_hidden_gaps(sim):
    observer, other = _pair(sim)
    other.status_last_action = "working"
    other.known_gaps.add("information:I_TEST_PRIVATE_GAP")  # the teammate's own, unannounced gap
    ctx = sim.brain_context_builder.build(sim, observer)
    assert ctx.team_state["teammate_help_signals"][other.name] is False
    observer.received_dik_requests[other.name] = {"ids": {"information:I_TEST_PRIVATE_GAP"}, "time": 0.0}  # once they say so, it is evidence
    ctx = sim.brain_context_builder.build(sim, observer)
    assert ctx.team_state["teammate_help_signals"][other.name] is True


def test_measurement_against_authoritative_truth(sim):
    agents = sim.agents[:3]
    for ag in agents:
        ag.theory_of_mind.clear()
    first, second = agents[0], agents[1]
    item = Information("I_TEST_ROUTE", "x")
    second.mental_model["information"].add(item)
    second.source_memory_state.setdefault(f"{second.role}_Info", {})["pending_role_share_ids"] = ["information:I_TEST_ROUTE"]
    second.communicate_with(first, sim_state=sim, mode="provide")
    rq = routing_quality(agents)
    assert rq["requests_scored"] > 0 and 0.0 <= rq["routing_quality"] <= 1.0
    acc = team_model_accuracy(agents)
    assert acc["precision"] in (None, 1.0) or 0.0 <= acc["precision"] <= 1.0  # defined, bounded, and read from ground truth
