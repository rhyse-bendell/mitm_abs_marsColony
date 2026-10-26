"""Request -> response lifecycle: received requests persist as response opportunities; the recipient's Pilot decides whether and when to answer."""

import tempfile

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.knowledge import Information
from modules.request_ledger import REQUEST_TTL_S, ReceivedRequestLedger
from modules.simulation import SimulationState


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="reqresp", seed=1)
        state.events = []
        state.logger.register_event_listener(lambda e: state.events.append((e["event_type"], e["payload_data"])))
        try:
            yield state
        finally:
            state.stop()


def _names(sim, kind):
    return [p for n, p in sim.events if n == kind]


def _setup(sim, held=("I_TEST_X",), asked=("I_TEST_X",)):
    asker, holder = sim.agents[0], sim.agents[1]
    holder.position = (asker.position[0] + 1.0, asker.position[1])
    for eid in held:
        holder.mental_model["information"].add(Information(eid, "content of " + eid))
    asker.known_gaps.clear()
    asker.known_gaps.update({f"information:{e}" for e in asked})
    return asker, holder


def _request(sim, asker, holder):
    asker.communicate_with(holder, sim_state=sim, mode="request")


def _respond(sim, holder, requester, *, run=True):
    """The holder's Pilot selects COMMUNICATE addressed to the requester; the Mecha translates and the action executes."""
    decision = BrainDecision(selected_action=A.COMMUNICATE, target_id=requester.name, confidence=0.6)
    action = holder._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]
    if run:
        holder.active_actions = [dict(action, progress=0.0)]
        holder._advance_active_actions(0.1, sim)
    return action


def test_full_lifecycle_request_to_gap_resolution(sim):
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    assert "information:I_TEST_X" not in {f"information:{i.id}" for i in asker.mental_model["information"]}
    assert not holder.known_gaps  # the request is not the recipient's gap
    rec = holder.request_ledger.open_record(asker.name)
    assert rec and rec["requested_ids"] == {"information:I_TEST_X"} and holder.request_ledger.status(rec) == "open"
    # the Mecha did not answer on the Pilot's behalf
    assert not [m for m in holder.team_plan_state.get("pending_outbound_messages", []) if m["type"] in {"TDP", "TIP", "TKP"}]
    # recognized as a response opportunity, exposed to the Pilot as an affordance addressed to the requester
    ctx = sim.brain_context_builder.build(sim, holder)
    offers = [a for a in ctx.action_affordances if a.get("response_to_request")]
    assert offers and offers[0]["target_id"] == asker.name and offers[0]["can_provide"] == ["information:I_TEST_X"] and offers[0]["reachable"] is True
    # Pilot selects the response: selection is recorded, nothing has been provided yet
    action = _respond(sim, holder, asker, run=False)
    assert action["recipient"] == asker.name and action["response_to_request"] == rec["request_id"]
    assert _names(sim, "dik_response_selected") and not _names(sim, "dik_response_provided")
    # delivery attempt -> actual provision -> uptake
    holder.active_actions = [dict(action, progress=0.0)]
    holder._advance_active_actions(0.1, sim)
    assert _names(sim, "dik_response_delivery_attempted")
    provided = _names(sim, "dik_response_provided")[0]
    assert provided["provided_ids"] == ["information:I_TEST_X"] and provided["request_status"] == "provided_all"
    uptake = _names(sim, "dik_request_uptake")[0]
    assert uptake["acquired_ids"] == ["information:I_TEST_X"]
    assert "information:I_TEST_X" in {f"information:{i.id}" for i in asker.mental_model["information"]}
    # requester-side gap resolution is the requester's own, separate bookkeeping (ledger sync + reconciliation)
    assert "information:I_TEST_X" not in asker.known_gaps
    assert not _names(sim, "dik_request_gap_resolved")  # not yet synced into the requester's ledger
    sim.time += 1.0
    asker._sync_dik_ledger(sim)
    asker._reconcile_known_gaps(sim)
    assert _names(sim, "dik_request_gap_resolved")[0]["element_id"] == "I_TEST_X"
    assert asker.dik_ledger["I_TEST_X"]["mode"] == "shared" and asker.dik_ledger["I_TEST_X"]["source"] == holder.name


def test_selection_is_not_provision(sim):
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    _respond(sim, holder, asker, run=False)
    rec = holder.request_ledger.open_record(asker.name)
    assert rec["selections"] == 1 and not rec["delivered"] and holder.request_ledger.status(rec) == "open"


def test_unanswered_request_stays_open_and_nothing_is_sent(sim):
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    sim.time += 10.0
    holder._expire_received_requests(sim)
    assert holder.request_ledger.open_record(asker.name) is not None
    assert "I_TEST_X" not in {i.id for i in asker.mental_model["information"]}


def test_request_expires_after_ttl(sim):
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    sim.time += REQUEST_TTL_S + 1.0
    holder._expire_received_requests(sim)
    assert holder.request_ledger.open_record(asker.name) is None
    closed = _names(sim, "dik_request_closed")
    assert closed and closed[0]["reason"] == "expired" and closed[0]["provided_ids"] == []
    assert not holder.response_opportunities()


def test_partial_answer_provides_only_held_items_and_keeps_the_rest_open(sim):
    asker, holder = _setup(sim, held=("I_TEST_X",), asked=("I_TEST_X", "I_TEST_Y"))
    _request(sim, asker, holder)
    offer = holder.response_opportunities()[0]
    assert offer["can_provide"] == ["information:I_TEST_X"] and offer["cannot_provide"] == ["information:I_TEST_Y"]
    _respond(sim, holder, asker)
    assert _names(sim, "dik_response_provided")[0]["provided_ids"] == ["information:I_TEST_X"]
    assert "I_TEST_Y" not in {i.id for i in asker.mental_model["information"]}  # an element the recipient does not possess is never transmitted
    rec = holder.request_ledger.open_record(asker.name)
    assert rec is not None and holder.request_ledger.status(rec) == "partially_provided"
    assert holder.request_ledger.outstanding_ids(rec) == {"information:I_TEST_Y"}
    # the recipient later acquires the rest: a new actionable response, not a duplicate of the first
    holder.mental_model["information"].add(Information("I_TEST_Y", "late"))
    assert holder.response_opportunities()[0]["can_provide"] == ["information:I_TEST_Y"]


def test_no_duplicate_reply_and_no_loop(sim):
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    _respond(sim, holder, asker)
    assert holder.request_ledger.open_record(asker.name) is None  # closed once everything asked for was provided
    sim.events.clear()
    holder.generate_message(recipient_name=asker.name, include_requests=False, include_unsolicited=False)
    assert holder._provision_solicited is False and holder._provided_element_ids == []
    _respond(sim, holder, asker)  # a second attempt sends no DIK
    assert not _names(sim, "dik_response_provided")
    assert not [m for m in holder.generate_message(recipient_name=asker.name, include_requests=False) if m["type"] == "TKRQ"]  # a reply never triggers a request


def test_element_not_possessed_is_never_transmitted(sim):
    asker, holder = _setup(sim, held=(), asked=("I_TEST_X",))
    _request(sim, asker, holder)
    assert holder.response_opportunities()[0]["can_provide"] == []
    msgs = holder.generate_message(recipient_name=asker.name, include_requests=False, include_unsolicited=False)
    assert not [m for m in msgs if m["type"] in {"TDP", "TIP", "TKP"}]


def test_already_satisfied_requester_does_not_ask(sim):
    asker, holder = _setup(sim)
    asker.mental_model["information"].add(Information("I_TEST_X", "already known"))
    msgs = asker.generate_message(recipient_name=holder.name, include_requests=True, include_unsolicited=False)
    assert not [m for m in msgs if m["type"] == "TKRQ"]
    _request(sim, asker, holder)
    assert holder.request_ledger.open_record(asker.name) is None


def test_stale_duplicate_request_after_delivery_is_not_a_new_need():
    ledger = ReceivedRequestLedger()
    ledger.receive("A", {"information:X", "information:Y"}, 10.0)
    ledger.mark_provided("A", ["information:X"], 12.0)
    ledger.receive("A", {"information:X", "information:Y"}, 12.0)  # duplicate that does not postdate the delivery
    assert ledger.open_record("A")["delivered"] == {"information:X": 12.0}
    ledger.receive("A", {"information:X", "information:Y"}, 30.0)  # a later re-ask: the requester still lacks it
    assert "information:X" not in ledger.open_record("A")["delivered"]


def test_withdrawn_request_closes(sim):
    ledger = ReceivedRequestLedger()
    ledger.receive("A", {"information:X"}, 1.0)
    ledger.receive("A", set(), 2.0)
    assert ledger.open_record("A") is None and ledger.history[-1]["closed_reason"] == "withdrawn"


def test_response_requires_spatial_range(sim):
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    holder.position = (asker.position[0] + 500.0, asker.position[1])
    ctx = sim.brain_context_builder.build(sim, holder)
    offer = [a for a in ctx.action_affordances if a.get("response_to_request")][0]
    assert offer["reachable"] is False
    _respond(sim, holder, asker)
    assert _names(sim, "dik_response_delivery_failed") and not _names(sim, "dik_response_provided")
    assert holder.request_ledger.open_record(asker.name) is not None  # still open: the Pilot can move and retry


def test_pilot_chooses_and_procedural_policy_can_choose_to_respond(sim):
    """The Procedural Baseline scores the per-requester affordance; the Mecha never picks it."""
    asker, holder = _setup(sim)
    _request(sim, asker, holder)
    ctx = sim.brain_context_builder.build(sim, holder)
    offer = [a for a in ctx.action_affordances if a.get("response_to_request")]
    assert len(offer) == 1 and offer[0]["provision_solicited"] == 1
    generic = [a for a in ctx.action_affordances if a.get("action_type") == "communicate" and not a.get("response_to_request")]
    assert generic and all(a["provision_solicited"] == 0 for a in generic)


def test_request_propensity_creates_real_requests_not_false_gaps(sim):
    asker, holder = _setup(sim)
    holder.known_gaps.clear()
    before = set(holder.known_gaps)
    for _ in range(3):
        _request(sim, asker, holder)
    assert holder.known_gaps == before  # repeated requests never create gaps in the recipient
    assert len(holder.request_ledger.open_records()) == 1  # and repeated asks merge into one record
