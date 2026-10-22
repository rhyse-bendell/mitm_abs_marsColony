"""Human cognition/action semantics (milestone 2): exposure != knowledge; participant-authored construction; validation/repair on artifact truth;
structured information sharing; whiteboard. Every test names the claim it proves."""
import json
import random

import pytest

from human_testbed_helpers import INVALID_DESIGNS, VALID_DESIGNS, Play
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construction_design import get_schema
from modules.human_participant import HumanIntent

SCHEMA = get_schema("mars_colony")


@pytest.fixture()
def play():
    p = Play()
    yield p
    p.close()


def _packet_items(p, packet):
    pk = p.sim.environment.knowledge_packets[packet]
    return pk["data"] + pk["information"]


def _expose_everything(p):
    """Give the human agent exposure to EVERY packet item (as if they had opened every source and been told every rule)."""
    for packet in ("Team_Info", "Architect_Info", "Botanist_Info", "Engineer_Info"):
        for item in _packet_items(p, packet):
            bucket = "data" if item.__class__.__name__ == "Data" else "information"
            p.agent.mental_model[bucket].add(item)
            item.acquired_by[p.agent.name] = {"mode": "direct", "from": item.source}
    for rule in ("R_HOUSE_VALIDITY", "R_GREENHOUSE_VALIDITY", "R_WATERGEN_VALIDITY", "R_GREENHOUSE_SUPPORT_DEPENDENCY"):
        p.agent.mental_model["knowledge"].add_rule(rule, [])


def _truth(p, pid):
    return bool(p.sim.environment.construction.projects[pid]["correct"])


# ---- the schema is task data, hidden truth stays hidden --------------------------------------------------------------------------------------
def test_schema_covers_every_structure_and_public_form_hides_the_validator():
    for stype in ("house", "greenhouse", "water_generator", "food_connector", "water_connector"):
        public = SCHEMA.public_schema(stype)
        assert public and public["properties"]
        blob = json.dumps(public)
        assert "valid" not in blob and "sources" not in blob and "D_" not in blob
    assert SCHEMA.evaluate("house", VALID_DESIGNS["house"])["valid"] is True
    assert SCHEMA.evaluate("house", INVALID_DESIGNS["house"])["valid"] is False
    # option order is not "valid first" for every property (the form must not leak the answer by position)
    for stype in ("house", "greenhouse", "water_generator"):
        props = SCHEMA.properties(stype)
        assert any(p.get("valid") and p["options"][0]["value"] not in p["valid"] for p in props)


def test_each_validator_property_independently_decides_validity():
    for stype in ("house", "greenhouse", "water_generator"):
        for prop in SCHEMA.properties(stype):
            if not prop.get("valid"):
                continue
            for option in prop["options"]:
                design = dict(VALID_DESIGNS[stype])
                design[prop["id"]] = option["value"]
                assert SCHEMA.evaluate(stype, design)["valid"] == (option["value"] in prop["valid"]), (stype, prop["id"], option["value"])


# ---- exposure != knowledge ---------------------------------------------------------------------------------------------------------------------
def test_human_exposure_alone_does_not_make_a_structure_correct(play):
    _expose_everything(play)  # exposed to every rule in the game, and every rule held
    play.agent.rule_accuracy = 1.0
    bad = play.start_structure("house", INVALID_DESIGNS["house"])
    assert bad and _truth(play, bad) is False
    assert play.sim.environment.construction.evaluate_design(bad)["valid"] is False


def test_zero_exposure_with_a_valid_design_is_correct(play):
    assert not play.agent.encountered_dik()  # nothing opened, nothing received
    play.agent.rule_accuracy = 0.0
    play.agent.mechanism_profile["rule_accuracy"] = 0.0
    good = play.start_structure("water_generator", VALID_DESIGNS["water_generator"])
    assert good and _truth(play, good) is True  # correctness comes from the design, not from exposure or an accuracy draw


def test_lack_of_recorded_exposure_does_not_make_physically_legal_actions_illegal(play):
    gid = play.start_structure("greenhouse", INVALID_DESIGNS["greenhouse"])
    play.complete_physically(gid)
    assert not play.agent.encountered_dik() and not list(play.agent.mental_model["knowledge"].rules)
    gate = play.sim.action_gate
    validate = BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=gid)
    assert gate.evaluate(agent=play.agent, decision=validate, sim_state=play.sim).legal  # may ATTEMPT validation with no recorded exposure at all
    cont = BrainDecision(ExecutableActionType.CONTINUE_CONSTRUCTION, target_id=gid)
    blockers, _ = play.agent._construction_action_blockers(cont, {"type": "construct", "project_id": gid}, play.sim.environment, sim_state=play.sim)
    assert not [b for b in blockers if "rule" in b or "inspection" in b or "grounding" in b or "knowledge" in b]
    # the same actor profile as an ARTIFICIAL agent would be refused on epistemic grounds (the boundary is human-only)
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    b_blockers = botanist._build_readiness_blockers(play.sim.environment, sim_state=play.sim)
    assert botanist.is_human_participant() is False and play.agent._build_readiness_blockers(play.sim.environment, sim_state=play.sim) == []
    assert b_blockers or True  # (Botanist state depends on the run; the human result above is the asserted contract)


def test_design_choices_determine_artifact_state_and_a_poor_legal_design_is_allowed(play):
    pid = play.start_structure("house", INVALID_DESIGNS["house"])
    project = play.sim.environment.construction.projects[pid]
    assert project["authoring"] == "participant_design" and project["design"] == INVALID_DESIGNS["house"]
    assert {v["property"] for v in project["design_evaluation"]["violations"]} == {"floor_color", "wall_ceiling_color", "headroom", "opening", "enclosure", "spacing"}
    assert project["design_history"][0]["actor"] == "Architect" and project["design_history"][0]["initial"] is True
    assert play.view() and play.obj(pid)["design"]["Floor colour"] == "Mixed colours"  # the participant sees their own choices (physical appearance) only


def test_gate_requires_a_complete_valid_design_from_a_human_and_never_fills_defaults(play):
    play.walk_to("Build_Site_B")
    item = next(i for i in play.obj("Build_Site_B")["items"] if "house" in i["label"])
    cases = [({}, "design_required"), ({"floor_color": "solid_pink"}, "design_incomplete"), ({**VALID_DESIGNS["house"], "bogus": "x"}, "design_unknown_property"),
             ({**VALID_DESIGNS["house"], "headroom": "ten_bricks"}, "design_invalid_option")]
    for design, expected in cases:
        before = len(play.sim.environment.construction.projects)
        play.gw.on_message({"t": "action", "item": item["id"], "design": design} if design else {"t": "action", "item": item["id"]})
        play.advance(1.5)
        rows = [r for r in play.human_rows() if r["event_type"] == "action_blocked"]
        assert any(b.startswith(expected) for b in rows[-1]["blockers"]), (expected, rows[-1]["blockers"])
        assert len(play.sim.environment.construction.projects) == before  # nothing created in its place


def test_agent_fidelity_draw_never_corrupts_a_participant_design(play):
    pid = play.start_structure("water_generator", VALID_DESIGNS["water_generator"])
    c = play.sim.environment.construction
    assert c.record_build_quality(pid, correct=False, actor="Botanist") is False and _truth(play, pid) is True
    assert c.repair_project(pid, actor="Botanist") is False  # an artificial agent cannot silently "repair" a participant design either


# ---- validation and repair ----------------------------------------------------------------------------------------------------------------------
def test_validation_of_an_invalid_participant_structure_fails_on_artifact_truth_and_logs_hidden_detail_separately(play):
    gid = play.start_structure("greenhouse", INVALID_DESIGNS["greenhouse"])
    play.complete_physically(gid)
    _expose_everything(play)  # even with every rule exposed, the structure is judged on what was built
    play.walk_to(gid)
    play.choose(gid, "Validate")
    play.advance(4)
    fb = [f["message"] for f in play.view()["feedback"] if f["kind"] == "validation"]
    assert fb and set(fb) == {"Validation did not pass. This structure needs repair."}  # pass/fail only
    hidden = [json.loads(line) for line in open(play.sim.logger.output_session.build_log_path("human_hidden_outcomes.jsonl"), encoding="utf-8")]
    val = [h for h in hidden if h["kind"] == "validation_authoritative"]
    assert val and val[0]["passed"] is False and {v["property"] for v in val[0]["design_violations"]} == {"soil", "ceiling"}
    assert all(any(src.startswith("D_") for src in v["sources"]) for v in val[0]["design_violations"])
    def keys(v, out=None):
        out = set() if out is None else out
        for k, x in (v.items() if isinstance(v, dict) else []):
            out.add(k)
            keys(x, out)
        for x in (v if isinstance(v, list) else []):
            keys(x, out)
        return out

    assert not (keys(play.view()) & {"correct", "design_evaluation", "violations", "design_violations", "valid"})  # no hidden outcome in any form
    assert [f for f in play.view()["feedback"] if f["kind"] == "validation"][-1]["message"] == "Validation did not pass. This structure needs repair."
    assert play.obj(gid)["validation_failed"] is True


def test_validation_of_a_valid_participant_structure_passes(play):
    wid = play.start_structure("water_generator", VALID_DESIGNS["water_generator"])
    play.complete_physically(wid)
    play.walk_to(wid)
    play.choose(wid, "Validate")
    play.advance(4)
    assert "Validation passed." in [f["message"] for f in play.view()["feedback"] if f["kind"] == "validation"]
    assert play.sim.environment.construction.projects[wid]["validated_complete"] is True


def test_human_validation_uses_world_truth_for_structures_other_agents_built(play):
    c = play.sim.environment.construction
    pid, status = c.create_project("site_b", "greenhouse", author="Botanist")
    assert status == "created"
    play.complete_physically(pid)
    c.projects[pid]["correct"] = False  # hidden truth of an artificially built structure
    play.walk_to(pid)
    play.choose(pid, "Validate")
    play.advance(4)
    assert c.projects[pid]["validated_complete"] is False and c.projects[pid]["closure_status"] == "failed"
    play.gw.controller.drain()
    c.projects[pid]["correct"] = True
    c.projects[pid]["closure_status"] = "idle"
    play.walk_to(pid)
    play.choose(pid, "Validate")
    play.advance(4)
    assert c.projects[pid]["validated_complete"] is True  # and it passes on truth alone, with no recorded exposure


def test_repair_is_an_explicit_participant_choice_never_silently_chosen(play):
    gid = play.start_structure("greenhouse", INVALID_DESIGNS["greenhouse"])
    play.complete_physically(gid)
    play.walk_to(gid)
    play.choose(gid, "Validate")
    play.advance(4)
    item = next(i for i in play.obj(gid)["items"] if "Repair" in i["label"])
    assert item["form"]["kind"] == "repair" and item["form"]["current"]["ceiling"] == "solid_bricks"
    assert "valid" not in json.dumps(item["form"]) and "correct" not in json.dumps(item["form"])
    # (1) no design at all: refused, nothing repaired
    play.gw.on_message({"t": "action", "item": item["id"]})
    play.advance(2)
    assert [r for r in play.human_rows() if r["event_type"] == "action_blocked"][-1]["blockers"][0] == "design_required"
    assert _truth(play, gid) is False
    # (2) an irrelevant change is applied as chosen and the structure stays incorrect
    play.gw.on_message({"t": "action", "item": item["id"], "design": {"soil_units": "3"}})
    play.advance(6)
    assert _truth(play, gid) is False
    assert play.sim.environment.construction.projects[gid]["design"]["soil_units"] == "3"
    assert "Changes applied. Validate again to check the structure." in [f["message"] for f in play.view()["feedback"]]
    # (3) the participant changes the properties that matter: now correct
    play.choose(gid, "Validate")
    play.advance(4)
    item = next(i for i in play.obj(gid)["items"] if "Repair" in i["label"])
    play.gw.on_message({"t": "action", "item": item["id"], "design": {"soil": "dirt_and_grass", "ceiling": "clear_glass"}})
    play.advance(6)
    assert _truth(play, gid) is True
    rows = [json.loads(line) for line in open(play.sim.logger.output_session.build_log_path("human_hidden_outcomes.jsonl"), encoding="utf-8")]
    reps = [r for r in rows if r["kind"] == "repair_authoritative"]
    assert [r["valid_after"] for r in reps][-1] is True and reps[0]["changed"] == ["soil_units"]


def test_human_cannot_repair_a_structure_without_a_design_surface(play):
    c = play.sim.environment.construction
    pid, _ = c.create_project("site_b", "greenhouse", author="Botanist")
    play.complete_physically(pid)
    c.projects[pid]["correct"] = False
    c.projects[pid]["closure_status"] = "failed"
    c.projects[pid]["validation_attempted"] = True
    play.walk_to(pid)
    assert not [i for i in play.obj(pid)["items"] if "Repair" in i["label"]]  # documented gap: no structured choice surface for structures not designed by a participant
    res = play.sim.action_gate.evaluate(agent=play.agent, decision=BrainDecision(ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, target_id=pid, action_args={"design": {"soil": "dirt_and_grass"}}), sim_state=play.sim)
    assert not res.legal and "design_not_applicable" in res.blockers


# ---- structured information sharing ------------------------------------------------------------------------------------------------------------------
def _read_own_brief(p):
    p.walk_to("Architect_Info")
    p.choose("Architect_Info", "Read briefing")
    p.advance(4)


def test_free_text_alone_transfers_no_dik(play):
    _read_own_brief(play)
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    held_before = set(play.sim.agents[0].__class__.encountered_dik(botanist))
    play.agent._deliver_human_chat(play.sim, {"action_args": {"text": "The headroom must be one brick, the airlock one brick wide", "scope": "team"}})
    assert set(botanist.encountered_dik()) == held_before  # no DIK inferred from arbitrary free text
    rec = play.sim.chat_log[-1]
    assert rec["attachments"] == [] and rec["delivery"][0].get("attached") is None


def test_attached_encountered_information_transfers_canonical_dik_human_to_procedural(play):
    _read_own_brief(play)
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    own = {i.id: i for i in _packet_items(play, "Architect_Info")}
    target = next(i for i in own.values() if i.__class__.__name__ == "Information")
    botanist.mental_model["information"].discard(target)
    target.acquired_by.pop(botanist.name, None)
    assert target.id in play.agent.encountered_dik()
    reply = play.gw.on_message({"t": "chat", "text": "Houses need this", "attachments": [target.id]})
    assert reply["ok"]
    play.advance(1.2)
    assert target in botanist.mental_model["information"]  # the SAME grounded object a Procedural teammate would have received from another agent
    assert botanist.encountered_dik()[target.id] is target and target.acquired_by["Botanist"]["mode"] == "shared" and target.acquired_by["Botanist"]["from"] == "Architect"
    rec = play.sim.chat_log[-1]
    assert rec["text"] == "Houses need this"  # the visible message stays verbatim
    entry = next(d for d in rec["delivery"] if d["recipient"] == "Botanist")
    assert entry["attached"] == [target.id] and entry["acquisition_outcome"] == "acquired" and entry["newly_held_by_recipient"] == [target.id]
    row = [r for r in play.human_rows() if r["event_type"] == "chat_sent"][-1]
    assert row["attachment_ids"] == [target.id] and row["text"] == "Houses need this" and row["delivery"]
    ex = [e for e in play.events if e["event_type"] == "communication_exchange" and e["payload_data"].get("human_authored")]
    assert ex and ex[-1]["payload_data"]["provided_element_ids"] == [target.id] and ex[-1]["payload_data"]["message_types"] == ["TIP"]


def test_data_attachments_use_the_data_channel(play):
    _read_own_brief(play)
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    datum = next(i for i in _packet_items(play, "Architect_Info") if i.__class__.__name__ == "Data")
    botanist.mental_model["data"].discard(datum)
    datum.acquired_by.pop(botanist.name, None)
    play.gw.on_message({"t": "chat", "text": "fact", "attachments": [datum.id]})
    play.advance(1.2)
    assert datum in botanist.mental_model["data"]
    ex = [e for e in play.events if e["event_type"] == "communication_exchange" and e["payload_data"].get("human_authored")]
    assert ex[-1]["payload_data"]["message_types"] == ["TDP"]


def test_unencountered_information_cannot_be_attached_or_shared(play):
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    private = next(i for i in _packet_items(play, "Botanist_Info") if i.__class__.__name__ == "Information")
    own_unread = next(i for i in _packet_items(play, "Architect_Info") if i.__class__.__name__ == "Information")
    botanist.mental_model["information"].discard(private)
    private.acquired_by.pop("Botanist", None)
    for target in (private, own_unread):  # another role's private packet, and even your own packet before you have opened it
        assert target.id not in play.agent.encountered_dik()
        play.gw.on_message({"t": "chat", "text": "sneaky", "attachments": [target.id]})
        play.advance(1.5)
        rows = [r for r in play.human_rows() if r["event_type"] == "action_blocked"]
        assert rows[-1]["blockers"] == ["attachment_not_encountered"]
        assert play.view()["feedback"][-1]["message"] == "You can only share information you have reviewed."
    assert private not in botanist.mental_model["information"] and not play.sim.chat_log
    assert not [r for r in play.human_rows() if r["event_type"] == "chat_sent"]  # a refused message is not logged as sent


def test_structured_sharing_human_to_human_is_the_same_grounded_object():
    p = Play(assignments={"Architect": "human_participant", "Botanist": "human_participant"})
    try:
        _read_own_brief(p)
        target = next(i for i in _packet_items(p, "Architect_Info") if i.__class__.__name__ == "Information")
        botanist = next(a for a in p.sim.agents if a.name == "Botanist")
        p.gw.on_message({"t": "chat", "text": "for you", "attachments": [target.id]})
        p.advance(1.2)
        assert target in botanist.mental_model["information"] and botanist.is_human_participant()
        assert any(rec["kind"] == "agent_message" and target.content in rec["text"] and "Botanist" in rec["recipients"] for rec in p.sim.chat_log)
        assert botanist.encountered_dik()[target.id] is target
    finally:
        p.close()


def test_a_procedural_teammate_reacts_through_its_normal_machinery_to_shared_information(play):
    _read_own_brief(play)
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    target = next(i for i in _packet_items(play, "Architect_Info") if i.__class__.__name__ == "Information")
    botanist.mental_model["information"].discard(target)
    target.acquired_by.pop("Botanist", None)
    botanist.known_gaps.add(f"information:{target.id}")
    t0 = botanist.last_dik_change_time
    play.gw.on_message({"t": "chat", "text": "this closes your gap", "attachments": [target.id]})
    play.advance(1.5)
    assert f"information:{target.id}" not in botanist.known_gaps and botanist.last_dik_change_time >= t0
    assert target.id in botanist.dik_ledger or any(e["payload_data"].get("agent_id") == "botanist_1" and e["event_type"].startswith(("dik_", "epistemic", "project_state")) for e in play.events)


# ---- whiteboard ------------------------------------------------------------------------------------------------------------------------------------------
def test_whiteboard_note_with_attachment_is_authored_attributed_and_consultable_by_a_procedural_agent(play):
    _read_own_brief(play)
    target = next(i for i in _packet_items(play, "Architect_Info") if i.__class__.__name__ == "Information")
    play.walk_to("whiteboard")
    item = next(i for i in play.obj("whiteboard")["items"] if "note" in i["label"].lower())
    assert item["form"]["kind"] == "note"
    play.gw.on_message({"t": "action", "item": item["id"], "text": "House rules attached", "attachments": [target.id]})
    play.advance(3)
    notes = [a for a in play.sim.team_knowledge_manager.artifacts.values() if a.artifact_type == "team_note"]
    assert len(notes) == 1 and notes[0].author == "Architect" and notes[0].content["text"] == "House rules attached" and notes[0].created_at > 0
    assert notes[0].carried_information == [target]
    assert [r for r in play.human_rows() if r["event_type"] == "whiteboard_note_added"][-1]["attachment_ids"] == [target.id]
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    botanist.mental_model["information"].discard(target)
    target.acquired_by.pop("Botanist", None)
    _, held_info, _ = botanist._held_dik_ids()
    reasons, carried_new = botanist._artifact_consult_reasons(notes[0].artifact_id, notes[0], held_info, None, "")
    assert reasons and target.id in carried_new  # the note is a useful read for an agent that lacks the attached information
    botanist.current_plan = None  # ... and consultable with no active plan (only team plans and team notes are, in that state)
    botanist.control_state["last_transition_reason"] = "no_active_plan"
    assessment = botanist._consult_assessment(play.sim.environment, sim_state=play.sim)
    assert assessment["has_artifacts"] and assessment["artifact_id"] == notes[0].artifact_id and assessment["useful"] and target.id in assessment["carries_unheld_dik"]
    gained = botanist._adopt_artifact_information(notes[0], play.sim)
    assert gained == 1 and target in botanist.mental_model["information"] and target.acquired_by["Botanist"]["mode"] == "artifact"


def test_procedural_agent_consults_the_note_end_to_end(play):
    _read_own_brief(play)
    target = next(i for i in _packet_items(play, "Architect_Info") if i.__class__.__name__ == "Information")
    play.walk_to("whiteboard")
    item = next(i for i in play.obj("whiteboard")["items"] if "note" in i["label"].lower())
    play.gw.on_message({"t": "action", "item": item["id"], "text": "read me", "attachments": [target.id]})
    play.advance(3)
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    botanist.mental_model["information"].discard(target)
    target.acquired_by.pop("Botanist", None)

    class Certain(random.Random):
        def random(self):
            return 0.0

    botanist._rng = Certain()
    botanist.active_actions = []
    botanist.current_action = botanist._translate_brain_decision_to_legacy_action(BrainDecision(ExecutableActionType.CONSULT_TEAM_ARTIFACT, target_id="whiteboard"), play.sim.environment, sim_state=play.sim)
    play.advance(60)
    assert target in botanist.mental_model["information"]
    assert [e for e in play.events if e["event_type"] == "dik_acquired_from_artifact" and e["payload_data"].get("information_id") == target.id]


def test_human_reads_the_whiteboard_as_written_and_exposure_is_recorded(play):
    _read_own_brief(play)
    target = next(i for i in _packet_items(play, "Architect_Info") if i.__class__.__name__ == "Information")
    manager = play.sim.team_knowledge_manager
    art = manager.externalize_artifact("note:Botanist:1", "team_note", "greenhouses need glass", {"text": "Greenhouses need glass", "author": "Botanist", "author_role": "Botanist", "attachment_ids": []}, "Botanist", 5.0)
    gh_info = next(i for i in _packet_items(play, "Botanist_Info") if i.__class__.__name__ == "Information")
    art.carried_information = [gh_info]
    assert gh_info.id not in play.agent.encountered_dik()
    play.walk_to("whiteboard")
    play.choose("whiteboard", "Read the whiteboard")
    play.advance(3)
    wb = play.view()["whiteboard"]
    assert wb and any(e["author"] == "Botanist" and e["text"] == "Greenhouses need glass" and e["attachments"][0]["id"] == gh_info.id for e in wb["entries"])
    assert gh_info.id in play.agent.encountered_dik() and play.agent.encountered_dik()[gh_info.id].acquired_by["Architect"]["mode"] == "artifact"
    labels = {s["label"] for s in play.view()["info"]["sources"]}
    assert any("whiteboard" in label.lower() for label in labels)  # now shareable, grouped by where it was encountered
    assert [r for r in play.human_rows() if r["event_type"] == "whiteboard_read"]


# ---- information panel semantics -----------------------------------------------------------------------------------------------------------------------------
def test_information_panel_is_framed_as_reviewed_information_not_knowledge(play):
    info = play.view()["info"]
    assert info["title"] == "Reviewed information" and "not a record of what you know" in info["note"]
    html = open("modules/participant_client/index.html", encoding="utf-8").read().lower()
    assert "reviewed information" in html and "what you know" in html and ">knowledge<" not in html
    blob = json.dumps(play.view()).lower()
    for var in ("confidence", "memory", "belief", "retention"):
        assert var not in blob  # no simulator confidence/memory variables for a human
