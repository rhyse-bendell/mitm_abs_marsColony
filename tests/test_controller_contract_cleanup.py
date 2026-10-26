"""Controller-contract cleanup: the Procedural Baseline chooses only from the canonical executable affordance set, and a validation's
execution is reported distinctly from its pass/fail result."""
import json

import pytest

from tests.construction_helpers import inject_defect

from modules.action_schema import ExecutableActionType
from modules.brain_provider import RuleBrain
from modules.controller_context import partition_affordances, last_rejection
from modules.ollama_prompt import build_prompt
from modules.procedural_baseline_pilot import ProceduralBaselinePilotAdapter
from tests.test_baseline_validation_pending_work import READY, _context, sim, _make_project, _run_validation  # noqa: F401

VALIDATE_BLOCKED = {"action_type": "validate_construction", "utility": 0.9, "target_id": "p1", "reachable": True,
                    "validation_ready": False, "validation_blockers": ["connector_has_no_valid_endpoint_link"]}
VALIDATE_READY = {**VALIDATE_BLOCKED, "validation_ready": True, "validation_blockers": []}
REPAIR_BLOCKED = {"action_type": "repair_or_correct_construction", "utility": 0.9, "target_id": "p1", "reachable": True,
                  "repair_ready": False, "repair_blockers": ["repair_project_rules_not_held"], "observed_failure_age_s": 30.0}
INSPECT_PRIVATE = {"action_type": "inspect_information_source", "utility": 0.9, "target_id": "Botanist_Info", "reachable": True,
                   "source_accessible": False, "source_blocker": "private_source_inaccessible"}
BASE = [
    {"action_type": "consult_team_artifact", "utility": 0.6, "target_id": "whiteboard"},
    {"action_type": "transport_resources", "utility": 0.3, "target_id": "resource_zone_to_work_zone", "reachable": True},
    {"action_type": "communicate", "utility": 0.1, "target_id": "nearby_agent", "reachable": True},
    {"action_type": "wait", "utility": 0.1, "target_id": None},
]


def _selections(affordances, n=40):
    out = []
    for mode in ("LOGISTICS", "CONSTRUCT", "COORDINATE", "MONITOR", "VALIDATE", "REPAIR"):
        brain = RuleBrain()
        context = _context(READY, mode=mode, affordances=affordances)
        for _ in range(n // 6 + 1):
            d = brain.decide(context)
            out.append((d.selected_action.value, d.target_id))
    return out


@pytest.mark.parametrize("blocked", [VALIDATE_BLOCKED, REPAIR_BLOCKED, INSPECT_PRIVATE])
def test_rulebrain_never_selects_a_blocked_affordance(blocked):
    ex, bl = partition_affordances(BASE + [blocked])
    assert [b["action_type"] for b in bl] == [blocked["action_type"]]  # it is blocked under the canonical boundary
    chosen = _selections(BASE + [blocked])
    assert chosen and all(a != blocked["action_type"] for a, _ in chosen), sorted(set(chosen))


def test_validation_is_selectable_once_it_becomes_executable():
    assert partition_affordances([VALIDATE_READY])[1] == []
    chosen = _selections(BASE + [VALIDATE_READY])
    assert any(a == "validate_construction" for a, _ in chosen)  # same project/context as the blocked case above, nothing else changed


def test_procedural_fallback_is_also_restricted_to_executable_actions():
    adapter = ProceduralBaselinePilotAdapter()
    context = _context(READY, affordances=[VALIDATE_BLOCKED, REPAIR_BLOCKED, INSPECT_PRIVATE, {"action_type": "wait", "target_id": None}])
    decision = adapter.choose_fallback_action(agent=None, context_packet=context, reason="test")
    assert decision.selected_action == ExecutableActionType.WAIT
    context = _context(READY, affordances=[VALIDATE_READY, {"action_type": "wait", "target_id": None}])
    assert adapter.choose_fallback_action(agent=None, context_packet=context, reason="test").selected_action == ExecutableActionType.VALIDATE_CONSTRUCTION


def test_llm_executable_set_is_the_same_boundary_and_unchanged():
    affs = BASE + [VALIDATE_BLOCKED, REPAIR_BLOCKED, INSPECT_PRIVATE]
    ex, bl = partition_affordances(affs)
    canon = {"identity": {"agent_id": "a", "role": "Engineer"}, "executable_actions": ex, "blocked_actions": bl, "last_rejection": None,
             "observations": {}, "recent_outcomes": {"recent_action_outcomes": []}}
    built = build_prompt(canon, budget_tokens=8000)
    offered = {a["action_type"] for a in built["affordance_map"].values()}
    assert offered == {a["action_type"] for a in BASE}
    assert {b["action_type"] for b in bl} == {"validate_construction", "repair_or_correct_construction", "inspect_information_source"}


# ------------------------------------------------------------------------ execution vs result


def _last(agent):
    return agent.recent_action_outcomes[-1]


def test_a_failed_validation_is_reported_as_completed_execution_with_failed_result(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=True)
    inject_defect(sim.environment.construction, project_id, update=False)  # a realized defect (an execution slip), not a flag set from outside
    _run_validation(sim, agent, project_id)
    rec = _last(agent)
    assert rec["action"] == "validate_construction" and rec["target"] == project_id
    assert rec["execution_status"] == "completed" and rec["validation_result"] == "failed"
    assert rec["status"] != "succeeded"  # the ambiguous word is not used for a validation
    assert "missing_expected_rules" not in json.dumps(rec) and "is_valid" not in rec  # no hidden reasons / authoritative fields
    assert last_rejection(agent.recent_action_outcomes) is None  # still not a gate rejection
    # the authoritative research log keeps the detail
    assert any(n == "construction_validated_incorrect" for n, _ in sim.captured_events)


def test_a_passed_validation_is_reported_as_passed(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=True)
    _run_validation(sim, agent, project_id)
    rec = _last(agent)
    assert rec["execution_status"] == "completed" and rec["validation_result"] == "passed"


def test_non_validation_outcomes_are_unchanged(sim):
    agent = sim.agents[0]
    agent._emit_event(sim, "action_execution_succeeded", {"decision_action": "transport_resources", "project_id": "p"})
    rec = _last(agent)
    assert rec["status"] == "succeeded" and "validation_result" not in rec and "execution_status" not in rec


def test_validation_result_reaches_the_controller_prompt(sim):
    agent = sim.agents[0]
    project_id = _make_project(sim, ready=True)
    inject_defect(sim.environment.construction, project_id, update=False)  # a realized defect (an execution slip), not a flag set from outside
    _run_validation(sim, agent, project_id)
    canon = {"identity": {"agent_id": "a", "role": "Engineer"}, "executable_actions": [{"action_type": "wait", "target_id": None}], "blocked_actions": [],
             "last_rejection": None, "observations": {}, "recent_outcomes": {"recent_action_outcomes": list(agent.recent_action_outcomes)}}
    assert '"validation_result":"failed"' in build_prompt(canon, budget_tokens=8000)["user"].replace(" ", "")
