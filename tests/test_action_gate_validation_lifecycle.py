from types import SimpleNamespace

from modules.action_gate import AgentActionGate
from modules.action_schema import BrainDecision, ExecutableActionType
from modules.construction import ConstructionManager


def _sim_with_project():
    cm = ConstructionManager()
    project_id, status = cm.create_project("site_b", structure_type="house", project_id_override="site_b_house_001")
    assert status in {"created", "exists"}
    return SimpleNamespace(environment=SimpleNamespace(construction=cm)), cm, project_id


def test_resource_complete_but_physically_unbuilt_validation_is_blocked_with_build_reroute():
    sim, cm, project_id = _sim_with_project()
    project = cm.projects[project_id]
    project["delivered_resources"]["bricks"] = project["required_resources"]["bricks"]
    project["resource_complete"] = True

    result = AgentActionGate().evaluate(agent=SimpleNamespace(name="A"), decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id), sim_state=sim)

    assert result.legal is False
    assert {"physical_build_not_started", "physical_incomplete"} & set(result.blockers)
    assert {r.selected_action for r in result.available_reroutes} & {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}
    assert cm.projects[project_id].get("validated_complete") is False


def test_material_incomplete_validation_is_blocked_with_transport_reroute():
    sim, cm, project_id = _sim_with_project()

    result = AgentActionGate().evaluate(agent=SimpleNamespace(name="A"), decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id), sim_state=sim)

    assert result.legal is False
    assert "physical_incomplete" in result.blockers
    assert ExecutableActionType.TRANSPORT_RESOURCES in {r.selected_action for r in result.available_reroutes}


def test_physically_complete_and_validation_ready_project_is_legal():
    sim, cm, project_id = _sim_with_project()
    project = cm.projects[project_id]
    required = project["required_resources"]["bricks"]
    project["delivered_resources"]["bricks"] = required
    project["resource_complete"] = True
    project["structurally_complete"] = True
    project["functional_support_complete"] = True
    project["support_requirements"] = {}
    project["epistemic_workspace"]["entries"] = [{"entry_type": "claim"}, {"entry_type": "evidence"}, {"entry_type": "design_note"}]

    result = AgentActionGate().evaluate(agent=SimpleNamespace(name="A"), decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id=project_id), sim_state=sim)

    assert result.legal is True
    assert cm.projects[project_id].get("validated_complete") is False


def test_active_construction_placeholder_resolves_validation_ready_project():
    sim, cm, project_id = _sim_with_project()
    project = cm.projects[project_id]
    required = project["required_resources"]["bricks"]
    project["delivered_resources"]["bricks"] = required
    project["resource_complete"] = True
    project["structurally_complete"] = True
    project["functional_support_complete"] = True
    project["support_requirements"] = {}
    project["epistemic_workspace"]["entries"] = [{"entry_type": "claim"}, {"entry_type": "evidence"}, {"entry_type": "design_note"}]
    agent = SimpleNamespace(name="A", project_closure_state={"project_id": project_id})

    result = AgentActionGate().evaluate(agent=agent, decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="active_construction"), sim_state=sim)

    assert result.legal is True
    assert result.project_id == project_id
    assert cm.projects[project_id].get("status") != "complete"
    assert cm.projects[project_id].get("validated_complete") is False


def test_active_construction_placeholder_resolves_unbuilt_project_and_blocks_with_build_reroute():
    sim, cm, project_id = _sim_with_project()
    project = cm.projects[project_id]
    project["delivered_resources"]["bricks"] = project["required_resources"]["bricks"]
    project["resource_complete"] = True
    agent = SimpleNamespace(name="A", project_closure_state={"project_id": project_id})

    result = AgentActionGate().evaluate(agent=agent, decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="active_construction"), sim_state=sim)

    assert result.legal is False
    assert result.project_id == project_id
    assert "physical_incomplete" in result.blockers
    assert {r.selected_action for r in result.available_reroutes} & {ExecutableActionType.START_CONSTRUCTION, ExecutableActionType.CONTINUE_CONSTRUCTION}
    assert {r.target_id for r in result.available_reroutes} == {project_id}
    assert cm.projects[project_id].get("status") != "complete"
    assert cm.projects[project_id].get("validated_complete") is False


def test_active_construction_placeholder_without_active_project_returns_not_found():
    cm = ConstructionManager()
    sim = SimpleNamespace(environment=SimpleNamespace(construction=cm))
    agent = SimpleNamespace(name="A")

    result = AgentActionGate().evaluate(agent=agent, decision=BrainDecision(ExecutableActionType.VALIDATE_CONSTRUCTION, target_id="active_construction"), sim_state=sim)

    assert result.legal is False
    assert "project_not_found" in result.blockers
    assert result.project_id is None
