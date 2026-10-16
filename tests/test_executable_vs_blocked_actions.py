"""Ollama affordance contract: only currently executable actions get affordance ids; blocked actions stay visible with blocker reasons."""
import json

from modules.controller_context import affordance_blockers, last_rejection, partition_affordances
from modules.ollama_pilot import OllamaPilotAdapter
from modules.ollama_prompt import build_prompt

REPAIR_BLOCKED = {"action_type": "repair_or_correct_construction", "target_id": "house_1", "observed_repair_need": True, "repair_ready": False,
                  "repair_blockers": ["missing_expected_rule:R_X", "repair_project_rules_not_held"], "observed_failure_age_s": 12.0, "reachable": True}
REPAIR_READY = {**REPAIR_BLOCKED, "repair_ready": True, "repair_blockers": []}
WAIT = {"action_type": "wait", "target_id": None}
INSPECT = {"action_type": "inspect_information_source", "target_id": "src_1", "reachable": True}


def _canon(affs, outcomes=()):
    ex, bl = partition_affordances(affs)
    return {"identity": {"agent_id": "a", "role": "Engineer"}, "executable_actions": ex, "blocked_actions": bl, "last_rejection": last_rejection(list(outcomes)),
            "observations": {}, "recent_outcomes": {"recent_action_outcomes": list(outcomes)}}


def test_repair_not_ready_is_blocked_not_executable():
    ex, bl = partition_affordances([INSPECT, REPAIR_BLOCKED, WAIT])
    assert [a["action_type"] for a in ex] == ["inspect_information_source", "wait"]
    assert len(bl) == 1 and bl[0]["blockers"] == REPAIR_BLOCKED["repair_blockers"] and bl[0]["target_id"] == "house_1"


def test_repair_ready_stays_executable():
    assert affordance_blockers(REPAIR_READY) == []
    ex, bl = partition_affordances([REPAIR_READY])
    assert len(ex) == 1 and not bl


def test_blocked_actions_get_no_affordance_id_and_are_not_in_the_schema_enum():
    built = build_prompt(_canon([INSPECT, REPAIR_BLOCKED, WAIT]), budget_tokens=8000)
    assert set(built["affordance_map"]) == {"a0", "a1"}
    assert all(a["action_type"] != "repair_or_correct_construction" for a in built["affordance_map"].values())
    assert built["schema"]["properties"]["affordance_id"]["enum"] == ["a0", "a1"]
    body = json.loads(built["user"].split("CONTEXT:\n", 1)[1].split("\n\nChoose", 1)[0])
    assert [b["action_type"] for b in body["blocked_actions"]] == ["repair_or_correct_construction"]
    assert "affordance_id" not in body["blocked_actions"][0] and body["blocked_actions"][0]["blockers"]
    assert all("affordance_id" in a for a in body["executable_actions"])


def test_model_cannot_select_a_blocked_action_by_any_id():
    built = build_prompt(_canon([INSPECT, REPAIR_BLOCKED, WAIT]), budget_tokens=8000)
    for forged in ("a2", "a3", "repair", "blocked_0"):
        parsed, error = OllamaPilotAdapter._validate(json.dumps({"affordance_id": forged}), built["affordance_map"])
        assert parsed is None and error.startswith("unknown_affordance")


def test_blocked_section_is_omitted_when_nothing_is_blocked():
    built = build_prompt(_canon([INSPECT, WAIT]), budget_tokens=8000)
    assert "blocked_actions" not in built["user"] and "last_rejection" not in built["user"]


def test_last_rejection_is_prominent_and_before_the_action_lists():
    outcomes = [{"time": 5.0, "status": "blocked", "action": "repair_or_correct_construction", "target": "house_1", "blocker": "repair_project_rules_not_held", "consecutive_identical_failures": 2},
                {"time": 9.0, "status": "succeeded", "action": "inspect_information_source", "target": "src_1"}]
    rej = last_rejection(outcomes)
    assert rej["action"] == "repair_or_correct_construction" and rej["blocker"] == "repair_project_rules_not_held" and rej["outcomes_since"] == 1
    user = build_prompt(_canon([INSPECT, WAIT], outcomes), budget_tokens=8000)["user"]
    assert user.index('"last_rejection"') < user.index('"executable_actions"')
    assert last_rejection([{"status": "succeeded"}]) is None


def test_prompt_adds_no_coaching():
    built = build_prompt(_canon([INSPECT, REPAIR_BLOCKED, WAIT]), budget_tokens=8000)
    text = (built["system"] + built["user"]).lower()
    assert "don't repair" not in text and "do not repair" not in text and "avoid" not in text
