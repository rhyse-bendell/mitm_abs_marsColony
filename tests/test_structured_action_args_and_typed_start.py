"""Structured ``action_args`` on the Pilot contract, and typed START_CONSTRUCTION creating the project type the Pilot chose.

Capability, not policy: nothing here asserts that a Pilot chooses any structure; only that every task-defined structure
can be expressed, gated, translated and executed, and that no project exists unless an explicit START created it.
"""

import ast
import copy
import dataclasses
import tempfile
from pathlib import Path

import pytest

from modules.action_schema import (
    ACTION_ARG_SPECS,
    BrainDecision,
    ExecutableActionType,
    validate_action_args,
)
from modules.brain_contract import AgentBrainResponse, AgentPlan, PlannedActionStep
from modules.simulation import SimulationState

START = ExecutableActionType.START_CONSTRUCTION
CONNECT = ExecutableActionType.CONNECT_STRUCTURES
MODULES = Path(__file__).resolve().parents[1] / "modules"
TASK_STRUCTURES = ["house", "greenhouse", "water_generator", "food_connector", "water_connector"]


@pytest.fixture
def sim():
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="typed_start_test", seed=1)
        try:
            yield state
        finally:
            state.stop()


def _world_state(cm):
    return copy.deepcopy((cm.projects, cm.connectors, cm._project_counters, {k: v.started_structures for k, v in cm.sites.items()}))


def _start(structure_type, target="Build_Site_B"):
    return BrainDecision(selected_action=START, target_id=target, action_args={"structure_type": structure_type}, confidence=0.8)


def _translate(agent, sim, decision):
    return agent._translate_brain_decision_to_legacy_action(decision, sim.environment, sim_state=sim)[0]


def _built(cm, structure_type, site="site_b"):
    pid, _ = cm.create_project(site, structure_type=structure_type, author="system")
    cm.projects[pid].update({"structurally_complete": True, "validated_complete": True, "status": "complete", "correct": True})
    return pid


# ---- schema -----------------------------------------------------------------------------------------------------------


def test_brain_decision_action_args_round_trip():
    decision = _start("greenhouse")
    assert decision.action_args == {"structure_type": "greenhouse"}
    assert dataclasses.replace(decision).action_args == {"structure_type": "greenhouse"}
    assert dataclasses.asdict(decision)["action_args"] == {"structure_type": "greenhouse"}
    assert BrainDecision(START, action_args=decision.action_args).action_args is not decision.action_args  # copied, not aliased


def test_planned_action_step_carries_action_args_into_the_brain_decision():
    step = PlannedActionStep.from_dict({"step_index": 0, "action_type": "start_construction", "target_id": "Build_Site_B", "action_args": {"structure_type": "water_generator"}})
    assert step.action_args == {"structure_type": "water_generator"}
    decision = step.to_brain_decision(confidence=0.7)
    assert decision.selected_action is START and decision.target_id == "Build_Site_B" and decision.action_args == {"structure_type": "water_generator"}
    plan = AgentPlan.from_dict({"plan_id": "p", "plan_horizon": 1, "ordered_goals": [], "ordered_actions": [step.__dict__], "confidence": 0.5})
    assert plan.next_action.action_args == {"structure_type": "water_generator"}  # survives the __dict__ round trip used by the baseline


def test_decisions_without_action_args_are_unchanged():
    assert BrainDecision(selected_action=ExecutableActionType.WAIT).action_args == {}
    step = PlannedActionStep.from_dict({"step_index": 0, "action_type": "wait"})
    assert step.action_args == {} and step.to_brain_decision(confidence=0.5).action_args == {}
    legacy_start = BrainDecision(selected_action=START, target_id="site_b_house_001")
    assert legacy_start.action_args == {} and legacy_start.target_id == "site_b_house_001"


def test_non_mapping_action_args_fail_closed():
    step = PlannedActionStep.from_dict({"step_index": 0, "action_type": "start_construction", "action_args": "greenhouse"})
    errors = validate_action_args(START, step.action_args)
    assert errors and any("unsupported_action_arg" in e for e in errors)
    assert validate_action_args(START, "greenhouse") == ["action_args_not_a_mapping"]


def test_arg_validation_is_per_action():
    # actions with structured arguments: typed start / connect, and (construction hardening) the specification choice on continue / repair and knowledge externalization
    assert set(ACTION_ARG_SPECS) == {START, CONNECT, ExecutableActionType.CONTINUE_CONSTRUCTION, ExecutableActionType.REPAIR_OR_CORRECT_CONSTRUCTION, ExecutableActionType.EXTERNALIZE_KNOWLEDGE}
    assert validate_action_args(START, {}) == []
    assert validate_action_args(START, {"structure_type": "greenhouse"}) == []
    assert "unsupported_action_arg:site_id" in validate_action_args(START, {"structure_type": "greenhouse", "site_id": "site_b"})
    assert "action_arg_not_a_string:structure_type" in validate_action_args(START, {"structure_type": 3})
    assert "malformed_structure_type" in validate_action_args(START, {"structure_type": "Green House|x"})
    assert "unknown_structure_type:castle" in validate_action_args(START, {"structure_type": "castle"}, allowed_structure_types=TASK_STRUCTURES)
    assert "unsupported_action_arg:structure_type" in validate_action_args(ExecutableActionType.TRANSPORT_RESOURCES, {"structure_type": "house"})
    assert validate_action_args(ExecutableActionType.WAIT, {}) == []


# ---- task-package-derived start options -----------------------------------------------------------------------------------


def test_startable_types_come_from_the_task_package_templates(sim):
    cm = sim.environment.construction
    assert cm.startable_structure_types() == TASK_STRUCTURES
    assert "bridge_bc" not in cm.startable_structure_types()  # bridges stay on the separate capacity-unlock path
    expected = {t.structure_type for t in sim.task_model.construction_templates.values() if t.enabled}
    assert set(cm.startable_structure_types()) == expected


def test_a_disabled_template_is_not_startable(sim):
    cm = sim.environment.construction
    template = next(t for t in cm.project_templates.values() if t.get("type") == "greenhouse")
    template["enabled"] = False
    assert "greenhouse" not in cm.startable_structure_types()
    legal, blockers, _ = cm.check_start("Build_Site_B", "greenhouse")
    assert not legal and "unknown_structure_type" in blockers


# ---- affordances / probes are read-only -----------------------------------------------------------------------------------


def test_start_affordances_list_each_legal_type_and_are_read_only(sim):
    cm = sim.environment.construction
    before = _world_state(cm)
    for agent in sim.agents:
        affordances = sim.brain_context_builder.build(sim, agent).action_affordances
        starts = [a for a in affordances if a["action_type"] == START.value]
        typed = [a for a in starts if a.get("start_mode") == "new_project"]
        assert {(a["target_id"], a["action_args"]["structure_type"]) for a in typed} == {("Build_Site_B", t) for t in TASK_STRUCTURES}, agent.role
        assert all(set(a["action_args"]) == {"structure_type"} for a in typed)
        assert not [a for a in starts if a.get("start_mode") == "existing_project"]  # nothing unfinished yet
        assert all(a["target_id"] != "Build_Site_A" for a in starts)  # zero-capacity site
        assert all(a["target_id"] != "Build_Site_C" for a in starts)  # locked behind the bridge
    assert _world_state(cm) == before
    cm.list_start_options(); cm.startable_structure_types()
    for t in TASK_STRUCTURES + ["castle"]:
        cm.check_start("Build_Site_B", t)
    assert _world_state(cm) == before


def test_probes_and_target_resolution_never_create_projects(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    before = _world_state(cm)
    for decision in (_start("greenhouse"), BrainDecision(selected_action=START, target_id="Build_Site_B")):
        agent._construction_action_blockers(decision, {"type": "idle", "decision_action": START.value}, sim.environment, sim_state=sim, probe_only=True)
        agent._construction_action_blockers(decision, {"type": "idle", "decision_action": START.value}, sim.environment, sim_state=sim)  # non-probe lookup path
        agent._construction_project_for_action(decision, {"project_id": None}, sim.environment)
        agent._select_build_target(sim.environment, include_project=True, requested_project_id="Build_Site_B")
        agent._bind_project_to_decision_if_needed(decision, sim.environment, sim_state=sim, reason="test")
        sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)
    assert _world_state(cm) == before


# ---- gate -------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("structure_type", TASK_STRUCTURES)
def test_every_role_may_start_every_task_structure(sim, structure_type):
    assert {a.role for a in sim.agents} >= {"Architect", "Engineer", "Botanist"}
    for agent in sim.agents:
        result = sim.action_gate.evaluate(agent=agent, decision=_start(structure_type), sim_state=sim)
        assert result.legal, (agent.role, structure_type, result.blockers)


def test_invalid_or_malformed_structure_types_are_rejected_without_creating_anything(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    before = _world_state(cm)
    for bad, blocker in (("castle", "unknown_structure_type:castle"), ("bridge_bc", "unknown_structure_type:bridge_bc"), ("House", "malformed_structure_type"), ("a|b", "malformed_structure_type")):
        result = sim.action_gate.evaluate(agent=agent, decision=_start(bad), sim_state=sim)
        assert not result.legal and "malformed_action_args" in result.blockers and blocker in result.blockers, (bad, result.blockers)
        action = _translate(agent, sim, _start(bad))
        assert action["type"] == "idle" and action.get("blocked_action")  # reported as blocked, not repaired into a house
    extra = BrainDecision(selected_action=START, target_id="Build_Site_B", action_args={"structure_type": "house", "cost": 1})
    assert "unsupported_action_arg:cost" in sim.action_gate.evaluate(agent=agent, decision=extra, sim_state=sim).blockers
    assert _world_state(cm) == before


def test_typed_start_respects_site_capacity_and_the_bridge_lock(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    locked = sim.action_gate.evaluate(agent=agent, decision=_start("house", "Build_Site_C"), sim_state=sim)
    assert not locked.legal and "site_locked_by_bridge" in locked.blockers
    none = sim.action_gate.evaluate(agent=agent, decision=_start("house", "Build_Site_A"), sim_state=sim)
    assert not none.legal and "site_capacity_reached" in none.blockers  # site A has zero capacity
    cm.sites["site_b"].capacity = 1
    pid, _ = cm.create_project("site_b", "house")
    full = sim.action_gate.evaluate(agent=agent, decision=_start("greenhouse"), sim_state=sim)
    assert not full.legal and "site_capacity_reached" in full.blockers
    assert not [a for a in sim.brain_context_builder.build(sim, agent).action_affordances if a.get("start_mode") == "new_project"]
    cm.bridges["bridge_bc"].status = "complete"
    unlocked = sim.action_gate.evaluate(agent=agent, decision=_start("greenhouse", "Build_Site_C"), sim_state=sim)
    assert unlocked.legal, unlocked.blockers


def test_untyped_start_never_creates_a_project(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    decision = BrainDecision(selected_action=START, target_id="Build_Site_B", confidence=0.8)
    result = sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)
    assert not result.legal and "structure_type_required_for_new_project" in result.blockers
    before = _world_state(cm)
    action = _translate(agent, sim, decision)
    assert action["type"] == "idle"
    assert _world_state(cm) == before and cm.projects == {}
    # an untyped START still continues an existing unfinished project
    pid, _ = cm.create_project("site_b", "greenhouse")
    assert sim.action_gate.evaluate(agent=agent, decision=BrainDecision(selected_action=START, target_id="Build_Site_B"), sim_state=sim).legal
    assert sim.action_gate.evaluate(agent=agent, decision=BrainDecision(selected_action=START, target_id=pid), sim_state=sim).legal


# ---- execution --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("structure_type", TASK_STRUCTURES)
def test_a_committed_typed_start_creates_exactly_that_project_type(sim, structure_type):
    cm = sim.environment.construction
    template = next(t for t in cm.project_templates.values() if t.get("type") == structure_type)
    for agent in sim.agents:  # each role, on a fresh world each time
        cm.projects.clear(); cm._project_counters.clear()
        decision = _start(structure_type)
        action = _translate(agent, sim, decision)
        assert action["type"] == "construct", (agent.role, action)
        assert len(cm.projects) == 1, (agent.role, list(cm.projects))
        project = next(iter(cm.projects.values()))
        assert project["type"] == structure_type and project["site_id"] == "site_b" and project["author"] == agent.name
        assert project["required_resources"]["bricks"] == int(template["required"])  # cost from the task template, not Pilot code
        assert action["project_id"] == project["id"] and action["action_args"] == {"structure_type": structure_type}
        assert project["id"].startswith(f"site_b_{structure_type}_")


def test_the_project_is_created_once_even_if_the_same_decision_is_translated_again(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    decision = _start("water_generator")
    first = _translate(agent, sim, decision)
    second = _translate(agent, sim, decision)
    assert len(cm.projects) == 1 and first["project_id"] == second["project_id"]
    other = _start("water_generator")  # a new Pilot START is a new project
    _translate(agent, sim, other)
    assert len(cm.projects) == 2


def test_a_typed_start_binds_to_its_own_new_project_not_another_unfinished_one(sim):
    cm = sim.environment.construction
    existing, _ = cm.create_project("site_b", "house")
    agent = sim.agents[0]
    decision = _start("greenhouse")
    kept = agent._bind_project_to_decision_if_needed(decision, sim.environment, sim_state=sim, reason="test")
    assert kept is decision and kept.action_args == {"structure_type": "greenhouse"}
    action = _translate(agent, sim, decision)
    created = cm.projects[action["project_id"]]
    assert created["type"] == "greenhouse" and action["project_id"] != existing


def test_a_connector_start_builds_a_connector_project_and_does_not_attach_a_link(sim):
    cm = sim.environment.construction
    agent = sim.agents[0]
    action = _translate(agent, sim, _start("water_connector"))
    assert cm.projects[action["project_id"]]["type"] == "water_connector"
    assert cm.connectors == [] and cm.projects[action["project_id"]]["connections"] == []


def test_connect_structures_never_instantiates_a_connector_project(sim):
    cm = sim.environment.construction
    cm.sites["site_b"].capacity = 30
    agent = sim.agents[0]
    gen, house = _built(cm, "water_generator"), _built(cm, "house")  # endpoints exist, but no connector project was started
    before = _world_state(cm)
    decision = BrainDecision(selected_action=CONNECT, action_args={"connector_project_id": "site_b_water_connector_001", "endpoint_a_id": gen, "endpoint_b_id": house})
    assert "connector_project_not_found" in sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim).blockers
    _translate(agent, sim, decision)
    assert _world_state(cm) == before
    assert not any(p["type"] in {"water_connector", "food_connector"} for p in cm.projects.values())
    # an unfinished (started but unbuilt) connector cannot be used to connect yet
    connector, _ = cm.create_project("site_b", "water_connector")
    unbuilt = BrainDecision(selected_action=CONNECT, action_args={"connector_project_id": connector, "endpoint_a_id": gen, "endpoint_b_id": house})
    assert "connector_not_structurally_complete" in sim.action_gate.evaluate(agent=agent, decision=unbuilt, sim_state=sim).blockers


# ---- full canonical contract ----------------------------------------------------------------------------------------------


def _through_the_contract(sim, agent, step_payload):
    """Pilot response -> grounding/validation -> BrainDecision -> bind -> pivots (the runtime order, before translation)."""
    context = sim.brain_context_builder.build(sim, agent)
    request = agent._build_sync_pilot_request(sim, context, "test")
    response = AgentBrainResponse.from_dict(
        {
            "response_id": "r1",
            "agent_id": request.agent_id,
            "plan": {"plan_id": "p1", "plan_horizon": 1, "ordered_goals": [], "ordered_actions": [step_payload], "next_action": step_payload, "confidence": 0.8},
        }
    )
    runtime = sim.get_agent_brain_runtime(agent) if hasattr(sim, "get_agent_brain_runtime") else {"provider": sim.brain_provider}
    grounded = agent._ground_validate_and_decide(response, context, request, sim, runtime)
    decision = grounded["decision"]
    decision = agent._bind_project_to_decision_if_needed(decision, sim.environment, sim_state=sim, reason="local_refresh_no_active_plan")
    decision = agent._apply_policy_pivots(decision, sim.environment, sim_state=sim, context=context, pivot_origin="local_refresh")
    return grounded, decision


@pytest.mark.parametrize("structure_type", ["greenhouse", "water_generator", "food_connector"])
def test_structure_type_survives_the_whole_canonical_contract(sim, structure_type):
    cm = sim.environment.construction
    agent = sim.agents[1]
    step = {"step_index": 0, "action_type": "start_construction", "target_id": "Build_Site_B", "expected_purpose": "test", "action_args": {"structure_type": structure_type}}
    grounded, decision = _through_the_contract(sim, agent, step)
    assert not grounded["schema_validation_errors"] and not grounded["repaired"]
    assert decision.selected_action is START and decision.action_args == {"structure_type": structure_type} and decision.target_id == "Build_Site_B"
    assert cm.projects == {}  # nothing created before the commit
    gate = sim.action_gate.evaluate(agent=agent, decision=decision, sim_state=sim)
    assert gate.legal and gate.normalized_decision.action_args == {"structure_type": structure_type}
    action = _translate(agent, sim, decision)
    assert [p["type"] for p in cm.projects.values()] == [structure_type]
    assert action["project_id"] in cm.projects
    agent.current_action = [action]
    agent.active_actions = []
    agent._advance_active_actions(0.0, sim)
    assert agent.active_actions[0]["project_id"] == action["project_id"] and agent.active_actions[0]["action_args"] == {"structure_type": structure_type}
    assert len(cm.projects) == 1


def test_structured_connect_survives_the_whole_canonical_contract_and_executes(sim):
    cm = sim.environment.construction
    cm.sites["site_b"].capacity = 30
    gen, green, conn = _built(cm, "water_generator"), _built(cm, "greenhouse"), _built(cm, "water_connector")
    agent = sim.agents[2]
    args = {"connector_project_id": conn, "endpoint_a_id": gen, "endpoint_b_id": green}
    step = {"step_index": 0, "action_type": "connect_structures", "expected_purpose": "test", "action_args": args}
    grounded, decision = _through_the_contract(sim, agent, step)
    assert not grounded["schema_validation_errors"]
    assert decision.selected_action is CONNECT and decision.action_args == args and decision.target_id is None
    assert cm.connectors == []
    action = _translate(agent, sim, decision)
    agent.current_action = [action]
    agent.active_actions = []
    for _ in range(6):
        agent._advance_active_actions(1.0, sim)
    assert len(cm.connectors) == 1 and {cm.connectors[0]["from_project_id"], cm.connectors[0]["to_project_id"]} == {gen, green}


def test_a_legacy_packed_connect_step_is_normalized_at_the_contract_boundary(sim):
    cm = sim.environment.construction
    cm.sites["site_b"].capacity = 30
    gen, green, conn = _built(cm, "water_generator"), _built(cm, "greenhouse"), _built(cm, "water_connector")
    step = {"step_index": 0, "action_type": "connect_structures", "target_id": f"{conn}|{gen}|{green}", "expected_purpose": "legacy"}
    grounded, decision = _through_the_contract(sim, sim.agents[0], step)
    assert decision.action_args == {"connector_project_id": conn, "endpoint_a_id": gen, "endpoint_b_id": green} and decision.target_id is None


def test_the_procedural_baseline_emits_structured_args_for_the_affordance_it_chose(sim):
    agent = sim.agents[0]
    seen = set()
    for _ in range(40):
        context = sim.brain_context_builder.build(sim, agent)
        decision = sim.brain_provider.decide(context)
        sim.time += 1.0
        if decision.selected_action is START:
            assert set(decision.action_args) == {"structure_type"} and decision.target_id == "Build_Site_B"
            seen.add(decision.action_args["structure_type"])
        else:
            assert decision.action_args == {}
    assert seen <= set(TASK_STRUCTURES)


# ---- middleware must not create projects -------------------------------------------------------------------------------------


def test_only_the_typed_start_commit_creates_projects_at_runtime():
    """Static guard: no runtime module asks the ConstructionManager to create a project except _commit_typed_start."""
    offenders = []
    for path in sorted(MODULES.glob("*.py")):
        if path.name == "construction.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(fn):
                if not isinstance(node, ast.Call):
                    continue
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                if name in {"create_project", "find_or_create_connector_project"} or (name == "start_new_project" and (path.name, fn.name) != ("agent.py", "_commit_typed_start")):
                    offenders.append((path.name, fn.name, name))
                for kw in node.keywords:
                    if kw.arg in {"create_if_missing", "allow_project_creation"} and not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
                        # the only tolerated non-literal-False use is _select_build_target forwarding its own (default False) parameter
                        if (path.name, fn.name) != ("agent.py", "_select_build_target"):
                            offenders.append((path.name, fn.name, kw.arg))
    assert offenders == []


def test_middleware_pivots_create_no_unrequested_structures(sim):
    cm = sim.environment.construction
    for agent in sim.agents:
        for action in (ExecutableActionType.WAIT, ExecutableActionType.REASSESS_PLAN, ExecutableActionType.CONTINUE_CONSTRUCTION, ExecutableActionType.TRANSPORT_RESOURCES, START):
            agent._apply_policy_pivots(BrainDecision(selected_action=action, target_id="Build_Site_B", confidence=0.7), sim.environment, sim_state=sim, pivot_origin="test")
            _translate(agent, sim, BrainDecision(selected_action=action, target_id="Build_Site_B", confidence=0.7))
    assert cm.projects == {} and cm.connectors == []

def test_a_seeded_baseline_run_creates_projects_and_links_only_through_explicit_pilot_actions():
    """Scoped replacement for what the bridge e2e test could not show without middleware: provenance of every structure.

    Not a performance test: it asserts nothing about how many or which structures the baseline chooses, only that each
    project traces to a committed typed START and each connector link to a committed connect_structures.
    """
    with tempfile.TemporaryDirectory() as tmp:
        state = SimulationState(phases=None, project_root=tmp, flash_mode=True, experiment_name="provenance_test", seed=2)
        events = []
        state.logger.register_event_listener(lambda e: events.append((e["event_type"], e["payload_data"])))
        try:
            for _ in range(260):
                state.update(1.0)
            cm = state.environment.construction
            projects = {pid: p["type"] for pid, p in cm.projects.items()}
            links = list(cm.connectors)
        finally:
            state.stop()
    created = {p["project_id"]: p["structure_type"] for name, p in events if name == "project_created_by_start"}
    assert projects == created and projects, "every project must come from an explicit typed START"
    assert set(created.values()) <= set(TASK_STRUCTURES)
    started = [p for name, p in events if name == "pilot_selection_via_adapter" and p["selected_action"] == START.value and p.get("action_args")]
    assert len(started) >= len(created)  # created projects never exceed explicit typed START selections
    succeeded = [p for name, p in events if name == "connection_attempt_succeeded"]
    assert len(links) == len(succeeded)  # every graph link is a committed connect_structures


@pytest.mark.parametrize("missing", ["connector_project_id", "endpoint_a_id", "endpoint_b_id"])
def test_connect_rejects_each_missing_canonical_arg_and_malformed_values(sim, missing):
    agent = sim.agents[0]
    full = {"connector_project_id": "site_b_water_connector_001", "endpoint_a_id": "site_b_house_001", "endpoint_b_id": "site_b_water_generator_001"}
    partial = {k: v for k, v in full.items() if k != missing}
    result = sim.action_gate.evaluate(agent=agent, decision=BrainDecision(selected_action=CONNECT, action_args=partial), sim_state=sim)
    assert not result.legal and f"missing_action_arg:{missing}" in result.blockers
    for bad in (7, "has space", "a|b", ""):
        result = sim.action_gate.evaluate(agent=agent, decision=BrainDecision(selected_action=CONNECT, action_args={**full, missing: bad}), sim_state=sim)
        assert not result.legal and "malformed_action_args" in result.blockers, bad


def test_start_args_cannot_smuggle_extra_semantics_or_non_strings(sim):
    agent = sim.agents[0]
    for args, token in (
        ({"structure_type": ["house"]}, "action_arg_not_a_string:structure_type"),
        ({"structure_type": "house", "site_id": "site_a"}, "unsupported_action_arg:site_id"),
        ({"structure_type": "house", "required_resources": {"bricks": 0}}, "unsupported_action_arg:required_resources"),
    ):
        result = sim.action_gate.evaluate(agent=agent, decision=BrainDecision(selected_action=START, target_id="Build_Site_B", action_args=args), sim_state=sim)
        assert not result.legal and token in result.blockers
