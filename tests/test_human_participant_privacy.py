"""Privacy boundary of the participant view (H4): the projection must never expose anything the participant could not legitimately know."""
import json

import pytest

from human_testbed_helpers import VALID_DESIGNS, Play

FORBIDDEN_KEYS = {
    "correct", "hidden_correct", "build_error", "utility", "utility_score", "mental_model", "team_knowledge_manager", "knowledge_summary", "known_gaps",
    "controller_prompt", "prompt", "final_measures", "measures", "derivation_events", "mechanism_profile", "experimental_profile", "theory_of_mind",
    "dik_ledger", "support_deficits", "provenance", "expected_rules", "held_rule_ids_at_build", "validation_readiness", "mechanism_overrides",
    "controller_type_of_others", "planner_state", "control_state",
}


def _keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for k, v in value.items():
            found.add(str(k))
            _keys(v, found)
    elif isinstance(value, list):
        for v in value:
            _keys(v, found)
    return found


def _packet_texts(play, packet):
    pk = play.sim.environment.knowledge_packets[packet]
    return [str(i.content) for i in pk["data"] + pk["information"]]


@pytest.fixture()
def play():
    p = Play()
    yield p
    p.close()


def test_view_has_no_forbidden_keys_at_any_depth(play):
    play.advance(5)
    assert not (_keys(play.view()) & FORBIDDEN_KEYS)
    play.walk_to("Architect_Info")
    play.choose("Architect_Info", "Read briefing")
    play.advance(4)
    assert not (_keys(play.view()) & FORBIDDEN_KEYS)


def test_other_roles_private_packets_never_appear_in_the_view(play):
    others = _packet_texts(play, "Botanist_Info") + _packet_texts(play, "Engineer_Info")
    shared = set(_packet_texts(play, "Team_Info")) | set(_packet_texts(play, "Architect_Info"))
    others = [t for t in others if t not in shared]
    assert others
    play.walk_to("Architect_Info")
    play.choose("Architect_Info", "Read briefing")
    play.advance(4)
    play.walk_to("Botanist_Info")
    play.choose("Botanist_Info", "Read briefing")  # denied
    play.advance(3)
    blob = json.dumps(play.view())
    for text in others:
        assert text not in blob


def test_information_panel_holds_only_what_was_opened_and_nothing_derived(play):
    assert play.view()["info"]["sources"] == []  # nothing before any interaction
    play.walk_to("Architect_Info")
    play.choose("Architect_Info", "Read briefing")
    play.advance(4)
    sources = play.view()["info"]["sources"]
    assert [s["id"] for s in sources] == ["Architect_Info"]
    texts = {i["text"] for s in sources for i in s["items"]}
    assert texts <= set(_packet_texts(play, "Architect_Info"))
    # simulator-derived rules/knowledge (a model of the person's beliefs) are not shown even though the agent object holds them
    rules = [str(r) for r in play.agent.mental_model["knowledge"].rules]
    blob = json.dumps(play.view())
    assert all(r not in blob for r in rules) or not rules


def test_view_is_invariant_to_hidden_construction_correctness(play):
    pid = play.start_structure("house", VALID_DESIGNS["house"])
    construction = play.sim.environment.construction
    strip = lambda v: json.dumps({k: v[k] for k in ("objects", "info", "agents")}, sort_keys=True)  # noqa: E731
    construction.projects[pid]["correct"] = True
    a = strip(play.view())
    construction.projects[pid]["correct"] = False
    play.gw.publish(force=True)
    b = strip(play.view())
    assert a == b
    assert "correct" not in json.dumps(play.view())


def test_other_agents_private_dik_is_not_exposed_only_role_activity(play):
    play.advance(60)  # the artificial agents read their briefs meanwhile
    botanist = next(a for a in play.sim.agents if a.name == "Botanist")
    private = [str(getattr(i, "content", i)) for i in list(botanist.mental_model["data"]) + list(botanist.mental_model["information"])]
    shared = set(_packet_texts(play, "Team_Info"))
    private = [t for t in private if t not in shared]
    blob = json.dumps(play.view())
    assert all(t not in blob for t in private)
    for other in play.view()["agents"]:
        assert set(other) == {"name", "role", "color", "x", "y", "activity"}  # position + coarse observable activity only


def test_no_observer_or_controller_identity_data_in_the_participant_view(play):
    play.advance(10)
    blob = json.dumps(play.view()).lower()
    for token in ("procedural", "ollama", "controller_type", "utility", "teamwork_potential", "mechanism", "prompt", "planner", "seed"):
        assert token not in blob, token


def test_denied_attempt_gives_only_the_participant_message_not_the_internal_blocker(play):
    play.walk_to("Botanist_Info")
    play.choose("Botanist_Info", "Read briefing")
    play.advance(3)
    blob = json.dumps(play.view()["feedback"])
    assert "private_source_inaccessible" not in blob and "You are not permitted to view another role's private briefing." in blob


def test_pile_contents_are_visible_only_when_close(play):
    far = play.obj("pile_c")
    assert far["near"] is False and "status" not in far
    play.walk_to("pile_a")
    assert play.obj("pile_a")["status"] in {"stocked", "empty"} and "quantity" not in json.dumps(play.obj("pile_a"))
