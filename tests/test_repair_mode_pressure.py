"""Observed, epistemically executable repair needs generate REPAIR-mode pressure that unrelated validation candidates
cannot permanently starve. Generic policy only: no condition/teamwork special-casing, no granted knowledge."""

import pytest

from modules.action_schema import ExecutableActionType
from modules.brain_provider import RuleBrain
from tests.test_baseline_validation_pending_work import READY, _context, _project

REPAIR = ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION
VALIDATE = ExecutableActionType.VALIDATE_CONSTRUCTION


def _failed(pid="water_gen_1"):
    p = _project(pid, status="needs_repair", validation_ready=False)
    p.update({"needs_repair": True, "validation_failed": True, "observed_repair_need": True})
    return p


def _affordances(*, repair_ready, with_validate=True):
    aff = [
        {"action_type": "repair_or_correct_construction", "utility": 0.5, "target_id": "water_gen_1", "reachable": True,
         "repair_ready": repair_ready, "observed_repair_need": True},
        {"action_type": "consult_team_artifact", "utility": 0.6, "target_id": "whiteboard"},
        {"action_type": "communicate", "utility": 0.1, "target_id": "nearby_agent", "reachable": True},
        {"action_type": "wait", "utility": 0.1, "target_id": None},
    ]
    if with_validate:
        aff.append({"action_type": "validate_construction", "utility": 0.35, "target_id": "p1", "reachable": True})
    return aff


def _scores(built, aff, mechanism=None, mode="MONITOR"):
    brain = RuleBrain()
    ctx = _context(built, mode=mode, dwell=79, affordances=aff)
    if mechanism is not None:
        ctx.individual_cognitive_state["mechanism_profile"] = mechanism
    feats = brain._extract_features(ctx)
    legal = {a["action_type"] for a in aff}
    scores, guards = brain._compute_mode_scores(feats, legal, ctx.individual_cognitive_state["control_state"],
                                                ctx.individual_cognitive_state.get("mechanism_profile", {}))
    scores, _ = brain._apply_transition_guards(scores, guards, feats, legal)
    return scores, guards, feats


LOW_TEAMWORK = {"communication_propensity": 0.0, "request_propensity": 0.0, "teamwork": 0.0}
NEUTRAL = {"communication_propensity": 0.5, "request_propensity": 0.5, "teamwork": 0.5}


@pytest.mark.parametrize("mech", [LOW_TEAMWORK, NEUTRAL, {"communication_propensity": 1.0, "request_propensity": 1.0, "teamwork": 1.0}])
def test_repair_ready_agent_is_not_starved_by_unrelated_validation_candidates(mech):
    scores, guards, feats = _scores(READY + [_failed()], _affordances(repair_ready=True), mech)
    assert feats["repair_executable"] == 1.0 and feats["validation_work_pending"] == 1.0
    assert guards["REPAIR"] and guards["VALIDATE"]
    assert scores["REPAIR"] >= scores["VALIDATE"]  # repair is a live competitor, not drowned out


@pytest.mark.parametrize("mode", ["VALIDATE", "MONITOR", "COORDINATE", "LOGISTICS"])
def test_low_teamwork_repair_ready_agent_can_select_repair_mode_and_action(mode):
    brain = RuleBrain()
    ctx = _context(READY + [_failed()], mode=mode, affordances=_affordances(repair_ready=True))
    ctx.individual_cognitive_state["mechanism_profile"] = LOW_TEAMWORK
    chosen = [brain.decide(ctx).selected_action for _ in range(8)]
    assert REPAIR in chosen, chosen


def test_agent_lacking_rules_cannot_magically_repair():
    scores, guards, feats = _scores(READY + [_failed()], _affordances(repair_ready=False), LOW_TEAMWORK)
    assert feats["repair_executable"] == 0.0 and feats["repair_knowledge_gap"] == 1.0
    assert guards["REPAIR"] is False  # still gated by the affordance's own epistemic check
    brain = RuleBrain()
    ctx = _context(READY + [_failed()], mode="VALIDATE", affordances=_affordances(repair_ready=False))
    ctx.individual_cognitive_state["mechanism_profile"] = LOW_TEAMWORK
    assert all(brain.decide(ctx).selected_action != REPAIR for _ in range(8))


def test_repair_is_not_forced_when_no_repair_need_exists():
    aff = [a for a in _affordances(repair_ready=True) if a["action_type"] != "repair_or_correct_construction"]
    scores, guards, feats = _scores(READY, aff, LOW_TEAMWORK)
    assert feats["observed_repair_need"] == 0.0
    assert guards["REPAIR"] is False and "REPAIR" not in scores


def test_repair_pressure_has_no_mechanism_profile_dependence():
    low = _scores(READY + [_failed()], _affordances(repair_ready=True), LOW_TEAMWORK)[0]["REPAIR"]
    high = _scores(READY + [_failed()], _affordances(repair_ready=True), NEUTRAL)[0]["REPAIR"]
    assert low == high
