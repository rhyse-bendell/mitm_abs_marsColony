"""First-pass compact experimental panel: seven constituents, gestalt generation, explicit overrides, wiring and safety."""

import copy
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from modules.action_schema import BrainDecision, ExecutableActionType as A
from modules.experimental_profile import CONSTITUENTS, resolve_experimental_profile
from modules.experiment_runner import run_condition
from modules.simulation import SimulationState


def _sim(settings=None, seed=1):
    tmp = tempfile.TemporaryDirectory()
    state = SimulationState(phases=None, project_root=tmp.name, flash_mode=True, experiment_name="panel", seed=seed, experimental_settings=settings or {})
    state._tmp = tmp
    return state


@pytest.fixture
def make_sim():
    made = []

    def factory(settings=None, seed=1):
        s = _sim(settings, seed)
        made.append(s)
        return s

    yield factory
    for s in made:
        s.stop()
        s._tmp.cleanup()


# ------------------------------------------------------------- profile resolution
TEAMWORK = {"dik_push_propensity", "dik_request_propensity", "externalization_propensity", "coordination_propensity", "backup_propensity"}


def test_teamwork_potential_changes_exactly_its_five_generated_constituents():
    low = resolve_experimental_profile({"teamwork_potential": 0.25, "taskwork_potential": 0.5})["resolved_constituents"]
    high = resolve_experimental_profile({"teamwork_potential": 0.75, "taskwork_potential": 0.5})["resolved_constituents"]
    assert {k for k in CONSTITUENTS if low[k] != high[k]} == TEAMWORK
    assert all(high[k] > low[k] for k in TEAMWORK)


def test_taskwork_potential_changes_task_execution_efficiency_only():
    low = resolve_experimental_profile({"teamwork_potential": 0.5, "taskwork_potential": 0.25})["resolved_constituents"]
    high = resolve_experimental_profile({"teamwork_potential": 0.5, "taskwork_potential": 0.75})["resolved_constituents"]
    assert {k for k in CONSTITUENTS if low[k] != high[k]} == {"task_execution_efficiency"}


def test_memory_reliability_is_independent_of_both_gestalts_by_default():
    values = {resolve_experimental_profile({"teamwork_potential": t, "taskwork_potential": k})["resolved_constituents"]["memory_reliability"] for t in (0.0, 0.5, 1.0) for k in (0.0, 0.5, 1.0)}
    assert values == {1.0}


def test_explicit_constituent_overrides_replace_never_add_to_generated_values():
    profile = resolve_experimental_profile({"teamwork_potential": 0.9}, {"dik_push_propensity": 0.1})
    assert profile["generated_constituents"]["dik_push_propensity"] > 0.5  # the gestalt wanted this high
    assert profile["resolved_constituents"]["dik_push_propensity"] == pytest.approx(0.1)  # and the explicit setting won outright
    assert profile["constituent_source"]["dik_push_propensity"] == "explicit_override"
    assert profile["constituent_source"]["dik_request_propensity"].startswith("generated:")  # unspecified stays generated
    assert profile["resolved_constituents"]["dik_request_propensity"] == resolve_experimental_profile({"teamwork_potential": 0.9})["resolved_constituents"]["dik_request_propensity"]


def test_unknown_constituents_are_rejected():
    with pytest.raises(ValueError):
        resolve_experimental_profile({}, {"trust": 0.4})


def test_the_resolved_profile_is_recorded_per_agent_and_reaches_the_runtime_mechanisms(make_sim):
    sim = make_sim({"constructs": {"teamwork_potential": 0.25, "taskwork_potential": 0.75}, "constituent_overrides": {"task_execution_efficiency": 0.3}})
    for agent in sim.agents:
        resolved = agent.experimental_profile["resolved_constituents"]
        assert resolved["task_execution_efficiency"] == pytest.approx(0.3)  # explicit beats the taskwork-generated value
        assert agent.build_speed == pytest.approx(0.3) and agent.dik_push_propensity == pytest.approx(resolved["dik_push_propensity"])
        assert agent.goal_alignment == pytest.approx(resolved["coordination_propensity"]) and agent.help_tendency == pytest.approx(resolved["backup_propensity"])
        assert agent.artifact_externalization_tendency == pytest.approx(resolved["externalization_propensity"])
        assert agent.memory_reliability == 1.0
    manifest = sim._experimental_manifest()
    assert set(manifest["per_agent"]) == {a.name for a in sim.agents}
    assert manifest["per_agent"]["Architect"]["explicit_constituent_overrides"] == {"task_execution_efficiency": 0.3}


def test_same_seed_and_config_resolve_identically(make_sim):
    settings = {"constructs": {"teamwork_potential": 0.6}, "constituent_overrides": {"memory_reliability": 0.4}}
    a, b = make_sim(settings, 5), make_sim(settings, 5)
    assert [x.experimental_profile for x in a.agents] == [x.experimental_profile for x in b.agents]
    assert [x.mechanism_profile for x in a.agents] == [x.mechanism_profile for x in b.agents]


def test_per_agent_run_settings_and_legacy_overrides_follow_the_documented_precedence(make_sim):
    sim = make_sim({"constructs": {"teamwork_potential": 0.5, "taskwork_potential": 0.5}, "per_agent": {"Engineer": {"constituent_overrides": {"backup_propensity": 0.95}}}})
    by_name = {a.name: a for a in sim.agents}
    assert by_name["Engineer"].backup_propensity == pytest.approx(0.95) and by_name["Architect"].backup_propensity == pytest.approx(0.5)
    configs = [{"name": "Architect", "role": "Architect", "constructs": {"teamwork_potential": 0.9}, "constituent_overrides": {"dik_push_propensity": 0.2}, "mechanism_overrides": {"build_speed": 0.9}}]
    tmp = tempfile.TemporaryDirectory()
    legacy = SimulationState(agent_configs=configs, phases=None, project_root=tmp.name, flash_mode=True, experiment_name="legacy", seed=1)
    try:
        agent = legacy.agents[0]
        assert agent.dik_push_propensity == pytest.approx(0.2)  # explicit constituent beat the gestalt
        assert agent.build_speed == pytest.approx(0.9)  # a legacy low-level override is a compat/debug override applied last
        assert agent.experimental_profile["legacy_overrides_of_constituent_mechanisms"]["build_speed"]["legacy_override"] == pytest.approx(0.9)
    finally:
        legacy.stop()
        tmp.cleanup()


# ------------------------------------------------------- behavioural separation
def _scores(sim, agent, mutate=None, **mech):
    provider = sim.brain_provider
    context = sim.brain_context_builder.build(sim, agent)
    affs = [dict(a) for a in context.action_affordances]
    if mutate:
        mutate(affs)
    features = provider._extract_features(context)
    profile = {"goal_alignment": 0.5, "help_tendency": 0.5, "artifact_consultation_tendency": 0.5, "artifact_externalization_tendency": 0.5, "dik_push_propensity": 0.5, "dik_request_propensity": 0.5, "coordination_propensity": 0.5, "backup_propensity": 0.5, **mech}
    scores = provider._score_actions_for_mode(sorted_affordances=affs, selected_mode="COORDINATE", features=features, mechanism_profile=profile)
    return {f"{a['action_type']}:{a.get('target_id')}:{i}": scores[f"{i}:{a['action_type']}"] for i, a in enumerate(affs)}


def _pick(scores, action):
    return [v for k, v in scores.items() if k.startswith(action.value + ":")][0]


def _set(action, **fields):
    def mutate(affs):
        for a in affs:
            if a["action_type"] == action.value:
                a.update(fields)
    return mutate


def test_high_push_raises_provision_without_raising_requests(make_sim):
    sim = make_sim()
    agent = sim.agents[0]
    mutate = lambda affs: (_set(A.COMMUNICATE, provision_unsolicited=2)(affs), _set(A.REQUEST_ASSISTANCE, dik_gap_count=0)(affs))
    low, high = _scores(sim, agent, mutate, dik_push_propensity=0.1), _scores(sim, agent, mutate, dik_push_propensity=0.9)
    assert _pick(high, A.COMMUNICATE) > _pick(low, A.COMMUNICATE) + 1.0
    assert _pick(high, A.REQUEST_ASSISTANCE) == pytest.approx(_pick(low, A.REQUEST_ASSISTANCE))


def test_high_request_raises_requests_for_recognized_gaps_without_raising_unsolicited_provision(make_sim):
    sim = make_sim()
    agent = sim.agents[0]
    mutate = lambda affs: (_set(A.COMMUNICATE, provision_unsolicited=2)(affs), _set(A.REQUEST_ASSISTANCE, dik_gap_count=3)(affs))
    low, high = _scores(sim, agent, mutate, dik_request_propensity=0.1), _scores(sim, agent, mutate, dik_request_propensity=0.9)
    assert _pick(high, A.REQUEST_ASSISTANCE) > _pick(low, A.REQUEST_ASSISTANCE) + 1.0
    assert _pick(high, A.COMMUNICATE) == pytest.approx(_pick(low, A.COMMUNICATE))
    # no recognized gap -> the parameter is inert; no recognized provision opportunity -> push is inert
    none_gap = lambda affs: _set(A.REQUEST_ASSISTANCE, dik_gap_count=0)(affs)
    assert _pick(_scores(sim, agent, none_gap, dik_request_propensity=0.9), A.REQUEST_ASSISTANCE) == pytest.approx(_pick(_scores(sim, agent, none_gap, dik_request_propensity=0.1), A.REQUEST_ASSISTANCE))
    none_prov = lambda affs: _set(A.COMMUNICATE, provision_unsolicited=0)(affs)
    assert _pick(_scores(sim, agent, none_prov, dik_push_propensity=0.9), A.COMMUNICATE) == pytest.approx(_pick(_scores(sim, agent, none_prov, dik_push_propensity=0.1), A.COMMUNICATE))


def test_opportunities_come_from_the_agents_own_state_not_from_the_world(make_sim):
    sim = make_sim()
    agent = sim.agents[0]
    assert agent._experimental_opportunities(sim, sim.environment)["dik_gap_count"] == 0
    agent.known_gaps.add("information:I_ENG_CONNECTOR_RULES")
    assert agent._experimental_opportunities(sim, sim.environment)["dik_gap_count"] == 1
    ctx = sim.brain_context_builder.build(sim, agent)
    assert [a for a in ctx.action_affordances if a["action_type"] == A.REQUEST_ASSISTANCE.value][0]["dik_gap_count"] == 1


def test_communicate_provides_and_request_requests(make_sim):
    sim = make_sim()
    giver, taker = sim.agents[1], sim.agents[0]
    giver.known_gaps.add("information:I_ARCH_HOUSING_DEPENDENCIES")
    provide = giver.generate_message("Architect", include_requests=False)
    request = giver.generate_message("Architect", include_unsolicited=False)
    assert "TKRQ" not in {m["type"] for m in provide}
    assert "TKRQ" in {m["type"] for m in request}


def test_request_routing_uses_the_agents_own_teammate_model_and_role_knowledge(make_sim):
    sim = make_sim()
    asker = sim.agents[0]  # Architect, missing an Engineer-brief element
    asker.known_gaps.add("information:I_ENG_CONNECTOR_RULES")
    near_botanist = (1.0, sim.agents[2])
    far_engineer = (5.0, sim.agents[1])
    assert asker._route_request_receiver([near_botanist, far_engineer]).role == "Engineer"  # element ownership is role knowledge, not teammate peeking
    asker.known_gaps.clear()
    assert asker._route_request_receiver([near_botanist, far_engineer]).role == "Botanist"  # nothing to route on: nearest


def test_externalization_propensity_changes_externalization_choice_through_the_pilot(make_sim):
    sim = make_sim()
    agent = sim.agents[0]
    agent.control_state["mode"] = "COORDINATE"
    cm = sim.environment.construction
    cm.create_project("site_b", "house")  # an unfinished project with an empty epistemic workspace: a real externalization gap
    def rate(value):
        agent.artifact_externalization_tendency = value
        agent.known_gaps.clear()
        n = 0
        for i in range(60):
            agent._rng = random.Random(i)
            n += agent._select_decision_through_pilot_adapter(sim, sim.brain_context_builder.build(sim, agent), "test").selected_action == A.EXTERNALIZE_PLAN
        return n
    assert rate(1.0) > rate(0.0)


def _held(sim, agent, *ids):
    for eid in ids:
        element = sim.task_model.dik_elements[eid]
        obj = agent._create_dik_object_from_element(element, source_id="test")
        agent.mental_model["data" if element.element_type == "data" else "information"].add(obj)


def _age(sim, agent, seconds, step=5.0):
    start = sim.time
    t = start
    while t < start + seconds:
        t += step
        sim.time = t
        agent._sync_dik_ledger(sim)
        agent._update_memory(sim)


IDS = ["I_ARCH_HOUSING_GEOMETRY", "I_ARCH_HOUSING_CAPACITY", "I_ARCH_HOUSING_AESTHETICS", "I_ARCH_HOUSING_DEPENDENCIES", "D_ARCH_HOUSE_NEEDS_FOOD_AND_WATER", "D_ARCH_CAPACITY_PER_2X2"]


def test_memory_reliability_one_retains_everything_over_the_horizon(make_sim):
    sim = make_sim({"constituent_overrides": {"memory_reliability": 1.0}})
    agent = sim.agents[0]
    _held(sim, agent, *IDS)
    _age(sim, agent, 2000)
    assert agent.experimental_metrics["dik_forgotten"] == 0 and agent.experimental_metrics_snapshot()["dik_retained"] >= len(IDS)


def test_low_memory_reliability_forgets_deterministically_under_a_fixed_seed(make_sim):
    def lost(seed):
        sim = make_sim({"constituent_overrides": {"memory_reliability": 0.0}}, seed)
        agent = sim.agents[0]
        agent.memory_reliability = 0.0
        _held(sim, agent, *IDS)
        _age(sim, agent, 600)
        return sorted(k for k, e in agent.dik_ledger.items() if e["forgotten_at"] is not None)
    first = lost(4)
    assert first and first == lost(4)  # something is lost, identically on a rerun with the same seed


def test_forgotten_dik_can_be_legitimately_reacquired_and_world_truth_is_untouched(make_sim):
    sim = make_sim()
    agent = sim.agents[0]
    agent.memory_reliability = 0.0
    _held(sim, agent, *IDS)
    before_world = copy.deepcopy(sim.environment.construction.projects)
    _age(sim, agent, 900)
    forgotten = [k for k, e in agent.dik_ledger.items() if e["forgotten_at"] is not None]
    assert forgotten
    agent.memory_reliability = 1.0  # stop further loss, then reacquire through an ordinary acquisition
    _held(sim, agent, forgotten[0])
    agent._sync_dik_ledger(sim)
    assert agent.dik_ledger[forgotten[0]]["forgotten_at"] is None and agent.experimental_metrics["dik_reacquired"] == 1
    assert sim.environment.construction.projects == before_world


def test_forgetting_reopens_the_source_and_externalized_copies_survive(make_sim):
    sim = make_sim()
    author, reader = sim.agents[0], sim.agents[1]
    _held(sim, author, "I_ARCH_HOUSING_DEPENDENCIES")
    sim.team_knowledge_manager.artifacts.clear()
    art = sim.team_knowledge_manager.externalize_artifact("whiteboard:x:1", "whiteboard_plan", "s", {"rules": []}, author.name, 1.0)
    from modules.support_reasoning import plan_relevant_dik_ids, load_support_reasoning
    assert "I_ARCH_HOUSING_DEPENDENCIES" in plan_relevant_dik_ids(load_support_reasoning("mars_colony"))
    author._attach_carried_information(art, sim)
    author.source_inspection_state["Architect_Info"] = "inspected"
    author.memory_reliability = 0.0
    author._sync_dik_ledger(sim)
    _age(sim, author, 1500)
    assert author.dik_ledger["I_ARCH_HOUSING_DEPENDENCIES"]["forgotten_at"] is not None
    assert author.source_inspection_state["Architect_Info"] == "unseen"  # the brief can be read again
    assert [i.id for i in art.carried_information] == ["I_ARCH_HOUSING_DEPENDENCIES"]  # the shared copy is unaffected
    assert reader._adopt_artifact_information(art, sim) == 1  # and it is still available to teammates
    assert author._adopt_artifact_information(art, sim) == 1  # including the forgetter, which reacquires from the artifact


def test_coordination_propensity_changes_plan_behaviour_not_task_speed(make_sim):
    low, high = make_sim({"constituent_overrides": {"coordination_propensity": 0.0}}), make_sim({"constituent_overrides": {"coordination_propensity": 1.0}})
    a_low, a_high = low.agents[0], high.agents[0]
    assert a_low.build_speed == a_high.build_speed and a_low._duration_scale("continue_construction") == a_high._duration_scale("continue_construction")
    plan = _set(A.EXTERNALIZE_PLAN, team_plan_opportunity=True, externalize_gap=False)
    s_low, s_high = _scores(low, a_low, plan, coordination_propensity=0.0), _scores(high, a_high, plan, coordination_propensity=1.0)
    assert _pick(s_high, A.EXTERNALIZE_PLAN) > _pick(s_low, A.EXTERNALIZE_PLAN) + 1.5
    none = _set(A.EXTERNALIZE_PLAN, team_plan_opportunity=False, externalize_gap=False)
    assert _pick(_scores(high, a_high, none, coordination_propensity=1.0), A.EXTERNALIZE_PLAN) == pytest.approx(_pick(_scores(low, a_low, none, coordination_propensity=0.0), A.EXTERNALIZE_PLAN))


def test_backup_propensity_changes_helping_not_build_duration(make_sim):
    low, high = make_sim({"constituent_overrides": {"backup_propensity": 0.0}}), make_sim({"constituent_overrides": {"backup_propensity": 1.0}})
    a_low, a_high = low.agents[1], high.agents[1]  # Engineer
    assert a_low._duration_scale("continue_construction") == a_high._duration_scale("continue_construction")
    cm = high.environment.construction
    low.environment.construction.create_project("site_b", "greenhouse")
    pid, _ = cm.create_project("site_b", "greenhouse")  # another role's structure
    ctx = high.brain_context_builder.build(high, a_high)
    needs = [a for a in ctx.action_affordances if a.get("target_id") == pid and a.get("backup_need")]
    assert needs and all(a["backup_need"] == "other_role_structure" for a in needs)
    own = cm.create_project("site_b", "water_generator")[0]
    assert not [a for a in high.brain_context_builder.build(high, a_high).action_affordances if a.get("target_id") == own and a.get("backup_need")]
    transport = _set(A.TRANSPORT_RESOURCES, backup_need="other_role_structure")
    assert _pick(_scores(high, a_high, transport, backup_propensity=1.0), A.TRANSPORT_RESOURCES) > _pick(_scores(low, a_low, transport, backup_propensity=0.0), A.TRANSPORT_RESOURCES) + 2.0


def test_task_execution_efficiency_changes_durations_without_touching_knowledge_or_teamwork(make_sim):
    slow, fast = make_sim({"constituent_overrides": {"task_execution_efficiency": 0.0}}), make_sim({"constituent_overrides": {"task_execution_efficiency": 1.0}})
    s, f = slow.agents[0], fast.agents[0]
    for action in ("transport_resources", "start_construction", "continue_construction", "validate_construction", "repair_or_correct_construction"):
        assert s._duration_scale(action) > f._duration_scale(action)
    for name in ("dik_push_propensity", "dik_request_propensity", "coordination_propensity", "backup_propensity", "artifact_externalization_tendency", "memory_reliability", "rule_accuracy"):
        assert getattr(s, name) == getattr(f, name)


# -------------------------------------------------------- authority / safety
def test_constituents_never_mutate_world_state_or_pick_actions_in_the_middleware(make_sim):
    sim = make_sim({"constituent_overrides": {k: 1.0 for k in CONSTITUENTS}})
    agent = sim.agents[0]
    cm = sim.environment.construction
    before = copy.deepcopy((cm.projects, cm.connectors, {k: v.quantity for k, v in cm.resource_nodes.items()}))
    for _ in range(3):
        sim.brain_context_builder.build(sim, agent)
    wait = BrainDecision(selected_action=A.WAIT, confidence=0.8)
    out = agent._apply_policy_pivots(wait, sim.environment, sim_state=sim, pivot_origin="test")
    assert out.selected_action in {A.WAIT, A.REASSESS_PLAN, A.OBSERVE_ENVIRONMENT} and out.selected_action not in agent._PILOT_STATE_MUTATING_ACTIONS
    assert before == copy.deepcopy((cm.projects, cm.connectors, {k: v.quantity for k, v in cm.resource_nodes.items()}))


def test_extreme_profiles_produce_no_attempted_state_creating_substitutions():
    for settings in ({"constituent_overrides": {k: 1.0 for k in CONSTITUENTS}}, {"constructs": {"teamwork_potential": 0.0, "taskwork_potential": 1.0}}):
        attempts = []
        random.seed(2)
        tmp = tempfile.TemporaryDirectory()
        sim = SimulationState(phases=None, project_root=tmp.name, flash_mode=True, experiment_name="authority", seed=2, experimental_settings=settings)
        sim.logger.register_event_listener(lambda e: attempts.append(e["event_type"]) if e["event_type"] in {"mecha_project_substitution_rejected", "recovery_pivot_declined_project_taskwork"} else None)
        try:
            for _ in range(150):
                sim.update(1.0)
        finally:
            sim.stop()
            tmp.cleanup()
        assert attempts == []


# ------------------------------------------------------- batch / GUI / manifest
def test_headless_runner_accepts_constituent_overrides_and_records_them():
    summary = run_condition(seed=3, duration_s=20, experimental_settings={"constructs": {"teamwork_potential": 0.25}, "constituent_overrides": {"backup_propensity": 0.9, "memory_reliability": 0.5}})
    per_agent = summary["manifest_condition"]["per_agent"]
    assert all(row["resolved_constituents"]["backup_propensity"] == pytest.approx(0.9) for row in per_agent.values())
    assert all(row["explicit_constituent_overrides"] == {"backup_propensity": 0.9, "memory_reliability": 0.5} for row in per_agent.values())
    assert "experimental" in summary["measures"] and summary["measures"]["experimental"]["team"]["dik_acquired"] >= 0


def test_the_session_manifest_contains_gestalt_overrides_resolved_profiles_mechanisms_and_provenance():
    tmp = tempfile.TemporaryDirectory()
    sim = SimulationState(phases=None, project_root=tmp.name, flash_mode=True, experiment_name="manifest", seed=7, experimental_settings={"constructs": {"teamwork_potential": 0.8}, "constituent_overrides": {"memory_reliability": 0.6}})
    try:
        for _ in range(5):
            sim.update(1.0)
    finally:
        sim.stop()
    manifest_path = next(Path(tmp.name).glob("**/session_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    panel = manifest["experimental_panel"]
    architect = panel["per_agent"]["Architect"]
    assert manifest["seed"] == 7 and manifest["pilot_id"]
    assert architect["gestalt_constructs"]["teamwork_potential"] == pytest.approx(0.8)
    assert architect["explicit_constituent_overrides"] == {"memory_reliability": 0.6}
    assert set(architect["resolved_constituents"]) == set(CONSTITUENTS) and architect["resolved_mechanisms"] and architect["hook_effects"]
    assert panel["provenance_hashes"] and any(k.startswith("experimental/experimental_profile") for k in panel["provenance_hashes"])
    assert "first-pass" in panel["status"].lower() or "first_pass" in panel["panel"]
    final = json.loads(next(Path(tmp.name).glob("**/final_measures.json")).read_text(encoding="utf-8"))
    assert "per_agent" in final["experimental"] and "team" in final["experimental"]
    tmp.cleanup()


class _V:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def test_gui_settings_preserve_constituent_overrides_and_do_not_inject_legacy_sliders():
    try:
        from interface import MarsColonyInterface
    except Exception:  # headless CI without Tk
        pytest.skip("interface unavailable")
    role = "Architect"
    fake = SimpleNamespace(
        MAX_AGENT_PANELS=3,
        num_agents_var=_V(1),
        agent_card_order=[role],
        active_roles={role: _V(True)},
        agent_profiles={role: {"team": _V("Custom"), "task": _V("High"), "team_value": _V(0.9), "task_value": _V(0.5)}},
        agent_traits={role: {"build_speed": _V(0.85), "goal_alignment": _V(0.85)}},
        agent_constituents={role: {"dik_push_propensity": _V("0.2"), "dik_request_propensity": _V("")}},
        legacy_override_enabled={role: _V(False)},
        packet_access={role: {"Team_Packet": _V(True)}},
        agent_identity={role: {"display_name": _V(role), "alias": _V(""), "template_id": None}},
        agent_brain_settings={role: {k: _V(v) for k, v in {"backend": "rule_brain", "local_model": "", "fallback_backend": "", "timeout_s": 5, "max_retries": 0}.items()}},
        agent_planner_settings={role: {k: _V(v) for k, v in {"planner_interval_steps": 3, "planner_timeout_seconds": 5, "planner_max_retries": 0, "planner_request_policy": "legacy", "planning_interval_steps": 3, "dik_integration_cooldown_steps": 3, "dik_integration_batch_threshold": 3, "degraded_consecutive_failures_threshold": 3, "degraded_cooldown_seconds": 1, "degraded_step_interval_multiplier": 1}.items()}},
        PLANNER_DEFAULTS={"planner_request_policy": "legacy"},
    )
    config = MarsColonyInterface.build_agent_configs(fake)[0]
    assert config["constituent_overrides"] == {"dik_push_propensity": 0.2}  # blank entries stay profile-generated
    assert config["mechanism_overrides"] == {}  # presets/sliders do not silently defeat the constituent profile
    assert config["constructs"]["teamwork_potential"] == pytest.approx(0.9)
    fake.legacy_override_enabled[role].set(True)
    assert MarsColonyInterface.build_agent_configs(fake)[0]["mechanism_overrides"] == {"build_speed": 0.85, "goal_alignment": 0.85}


def test_identical_seeded_configurations_give_identical_profiles_and_baseline_traces():
    def run():
        random.seed(6)
        tmp = tempfile.TemporaryDirectory()
        sim = SimulationState(phases=None, project_root=tmp.name, flash_mode=True, experiment_name="det", seed=6, experimental_settings={"constructs": {"teamwork_potential": 0.7}, "constituent_overrides": {"memory_reliability": 0.2}})
        trace = []
        sim.logger.register_event_listener(lambda e: trace.append((round(e["time"], 1), e["event_type"], e["payload_data"].get("agent"))) if e["event_type"] in {"pilot_selection_via_adapter", "dik_forgotten", "dik_reacquired", "artifact_consulted"} else None)
        try:
            for _ in range(120):
                sim.update(1.0)
            profiles = [a.experimental_profile["resolved_constituents"] for a in sim.agents]
            metrics = [a.experimental_metrics_snapshot() for a in sim.agents]
        finally:
            sim.stop()
            tmp.cleanup()
        return profiles, trace, metrics
    assert run() == run()