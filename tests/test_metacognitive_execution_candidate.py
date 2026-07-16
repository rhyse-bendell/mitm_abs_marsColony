from types import SimpleNamespace

from modules.action_schema import ExecutableActionType
from modules.agent import Agent


PRIORITY = [
    ExecutableActionType.VALIDATE_CONSTRUCTION,
    ExecutableActionType.TRANSPORT_RESOURCES,
    ExecutableActionType.START_CONSTRUCTION,
    ExecutableActionType.CONTINUE_CONSTRUCTION,
    ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION,
]


class CapturingLogger:
    def __init__(self):
        self.events = []

    def log_event(self, time, event_type, payload):
        self.events.append({"time": time, "event_type": event_type, "payload": payload})


def _agent():
    return Agent("Ava", "Engineer")


def _context(*actions):
    return SimpleNamespace(action_affordances=list(actions))


def _affordance(action_type, target_id):
    return {"action_type": action_type.value, "target_id": target_id, "utility": 1.0}


def _sim_state():
    return SimpleNamespace(time=12.0, logger=CapturingLogger())


def test_first_candidate_blocked_later_candidate_legal(monkeypatch):
    agent = _agent()
    sim_state = _sim_state()
    calls = []

    def blockers(probe, translated_probe, environment, sim_state=None):
        calls.append(probe.selected_action)
        if probe.selected_action == PRIORITY[0]:
            return ["blocked"], probe.target_id
        return [], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=sim_state,
        context=_context(_affordance(PRIORITY[0], "project_a"), _affordance(PRIORITY[1], "project_b")),
    )

    assert result is not None
    assert result.selected_action == PRIORITY[1]
    assert result.selected_action != PRIORITY[0]
    assert calls == [PRIORITY[0], PRIORITY[1]]


def test_all_advertised_candidates_blocked_emits_no_success_event(monkeypatch):
    agent = _agent()
    sim_state = _sim_state()

    def blockers(probe, translated_probe, environment, sim_state=None):
        return [f"blocked_{probe.selected_action.value}"], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=sim_state,
        context=_context(_affordance(PRIORITY[0], "project_a"), _affordance(PRIORITY[1], "project_b")),
    )

    assert result is None
    assert [e for e in sim_state.logger.events if e["event_type"] == "project_focus_bound_to_action"] == []


def test_first_advertised_candidate_legal_emits_locally_grounded_event(monkeypatch):
    agent = _agent()
    sim_state = _sim_state()

    def blockers(probe, translated_probe, environment, sim_state=None):
        return [], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=sim_state,
        context=_context(_affordance(PRIORITY[0], "legal_project")),
    )

    assert result.selected_action == PRIORITY[0]
    events = [e for e in sim_state.logger.events if e["event_type"] == "project_focus_bound_to_action"]
    assert len(events) == 1
    payload = events[0]["payload"]
    assert payload["selected_action"] == PRIORITY[0].value
    assert payload["project_id"] == "legal_project"
    assert payload["reason"] == "metacognitive_execution_candidate"


def test_absent_action_affordance_is_skipped(monkeypatch):
    agent = _agent()
    calls = []

    def blockers(probe, translated_probe, environment, sim_state=None):
        calls.append(probe.selected_action)
        return [], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=_sim_state(),
        context=_context(_affordance(PRIORITY[2], "later_project")),
    )

    assert result.selected_action == PRIORITY[2]
    assert PRIORITY[0] not in calls
    assert PRIORITY[1] not in calls
    assert calls == [PRIORITY[2]]


def test_no_relevant_affordances_returns_none_without_blocker_check(monkeypatch):
    agent = _agent()
    calls = []

    def blockers(probe, translated_probe, environment, sim_state=None):
        calls.append(probe.selected_action)
        return [], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=_sim_state(),
        context=_context({"action_type": "wait", "target_id": "none"}),
    )

    assert result is None
    assert calls == []


def test_selected_affordance_target_is_retained_in_probe_translation_and_result(monkeypatch):
    agent = _agent()
    observed = {}

    def blockers(probe, translated_probe, environment, sim_state=None):
        observed["probe_target_id"] = probe.target_id
        observed["translated_project_id"] = translated_probe.get("project_id")
        return [], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    distinctive_target = "distinctive_affordance_target"
    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=_sim_state(),
        context=_context(_affordance(PRIORITY[0], distinctive_target)),
        preferred_project_id="different_preferred_project",
    )

    assert result is None
    assert observed == {}

    result = agent._metacognitive_execution_candidate(
        SimpleNamespace(),
        sim_state=_sim_state(),
        context=_context(_affordance(PRIORITY[0], distinctive_target)),
    )

    assert observed["probe_target_id"] == distinctive_target
    assert observed["translated_project_id"] == distinctive_target
    assert result.target_id == distinctive_target
