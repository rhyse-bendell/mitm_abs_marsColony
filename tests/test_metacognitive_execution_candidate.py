from types import SimpleNamespace

from modules.action_schema import BrainDecision, ExecutableActionType
from modules.agent import Agent
from modules.environment import Environment


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

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
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

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
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

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
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

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
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

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
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

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
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


def _snapshot_construction(construction):
    return {
        "projects": set(construction.projects.keys()),
        "counters": dict(getattr(construction, "_project_counters", {}) or {}),
        "site_counts": {
            site_id: len([p for p in construction.projects.values() if p.get("site_id") == site_id])
            for site_id in construction.sites
        },
    }


def test_generic_transport_probe_does_not_create_project():
    agent = _agent()
    env = Environment()
    sim_state = _sim_state()
    before = _snapshot_construction(env.construction)

    result = agent._metacognitive_execution_candidate(
        env,
        sim_state=sim_state,
        context=_context({"action_type": "transport_resources", "target_id": "resource_zone_to_work_zone"}),
    )

    assert result is None
    assert _snapshot_construction(env.construction) == before


def test_existing_build_target_alias_resolves_read_only():
    agent = _agent()
    env = Environment()
    project_id = env.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
    before = _snapshot_construction(env.construction)

    result = agent._metacognitive_execution_candidate(
        env,
        sim_state=_sim_state(),
        context=_context(_affordance(ExecutableActionType.TRANSPORT_RESOURCES, "Build_Site_B")),
    )

    assert result is not None
    assert result.target_id == project_id
    assert _snapshot_construction(env.construction) == before


def test_later_same_family_concrete_affordance_can_be_selected_read_only():
    agent = _agent()
    env = Environment()
    project_id = env.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
    before = _snapshot_construction(env.construction)

    result = agent._metacognitive_execution_candidate(
        env,
        sim_state=_sim_state(),
        context=_context(
            {"action_type": "transport_resources", "target_id": "resource_zone_to_work_zone"},
            _affordance(ExecutableActionType.TRANSPORT_RESOURCES, project_id),
        ),
    )

    assert result is not None
    assert result.target_id == project_id
    assert _snapshot_construction(env.construction) == before


def test_first_same_family_candidate_blocked_second_legal(monkeypatch):
    agent = _agent()
    env = SimpleNamespace(construction=SimpleNamespace(projects={"project_a": {}, "project_b": {}}))
    calls = []

    def blockers(probe, translated_probe, environment, sim_state=None, **kwargs):
        calls.append(probe.target_id)
        if probe.target_id == "project_a":
            return ["blocked"], probe.target_id
        return [], probe.target_id

    monkeypatch.setattr(agent, "_construction_action_blockers", blockers)

    result = agent._metacognitive_execution_candidate(
        env,
        sim_state=_sim_state(),
        context=_context(
            _affordance(ExecutableActionType.TRANSPORT_RESOURCES, "project_a"),
            _affordance(ExecutableActionType.TRANSPORT_RESOURCES, "project_b"),
        ),
    )

    assert calls == ["project_a", "project_b"]
    assert result is not None
    assert result.target_id == "project_b"


def test_preferred_project_prevents_cross_project_fallback(monkeypatch):
    agent = _agent()
    env = Environment()
    project_a = env.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
    project_b, _ = env.construction.create_project(site_id="site_b", structure_type="house")
    before = _snapshot_construction(env.construction)

    def readiness_ok(environment, sim_state=None):
        return []

    monkeypatch.setattr(agent, "_build_readiness_blockers", readiness_ok)

    result = agent._metacognitive_execution_candidate(
        env,
        sim_state=_sim_state(),
        context=_context(
            _affordance(ExecutableActionType.START_CONSTRUCTION, "unknown_target_for_a"),
            _affordance(ExecutableActionType.START_CONSTRUCTION, project_b),
        ),
        preferred_project_id=project_a,
    )

    assert result is None
    assert _snapshot_construction(env.construction) == before


def test_readiness_unlock_evaluates_initial_candidate_once(monkeypatch):
    agent = _agent()
    env = Environment()
    decision = BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=None)
    candidate = BrainDecision(selected_action=ExecutableActionType.TRANSPORT_RESOURCES, target_id="project_a")
    agent.last_build_blockers = ["previously_blocked"]
    calls = []

    def candidate_stub(environment, sim_state=None, context=None, preferred_project_id=None):
        calls.append(preferred_project_id)
        return candidate if preferred_project_id is None else None

    monkeypatch.setattr(agent, "_metacognitive_execution_candidate", candidate_stub)
    monkeypatch.setattr(agent, "_build_readiness_blockers", lambda environment, sim_state=None: [])

    agent._apply_metacognitive_regulation(decision, env, sim_state=_sim_state(), context=_context())

    assert calls == [None]


def test_readiness_unlock_does_not_duplicate_project_focus_success_event(monkeypatch):
    agent = _agent()
    env = Environment()
    project_id = env.construction.resolve_project_id("Build_Site_B", create_if_missing=True)
    agent.last_build_blockers = ["previously_blocked"]
    sim_state = _sim_state()

    monkeypatch.setattr(agent, "_build_readiness_blockers", lambda environment, sim_state=None: [])

    agent._apply_metacognitive_regulation(
        BrainDecision(selected_action=ExecutableActionType.WAIT, target_id=None),
        env,
        sim_state=sim_state,
        context=_context(_affordance(ExecutableActionType.TRANSPORT_RESOURCES, project_id)),
    )

    events = [e for e in sim_state.logger.events if e["event_type"] == "project_focus_bound_to_action"]
    assert len(events) == 1
    assert events[0]["payload"]["project_id"] == project_id
